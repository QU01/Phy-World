"""Motor del estimador estructural (F0.5): ensembles del L1 con origen y población honestos.

Dos diferencias deliberadas con ``validate.forecast`` del Atlas (que el port conserva para W2):

1. **Población y capital humano después del origen** no se leen del dato observado (fuga D1):
   se extrapolan con la tasa de los 5 años previos al origen. En el origen 2019 del modo vivo se
   usa la proyección WPP 2024 que trae ``build_world(horizon=2030)``, publicada en 2024-07.
2. **Se devuelven los eventos** que una pregunta de Prime Radiant necesita: conflicto realizado
   miembro a miembro, golpes exitosos, crisis financieras y régimen democrático.

El origen lo fija la fuente con más retraso del reanclaje: el PIB de la PWT (``lag_years`` 3).
Una pregunta hecha a mediados del año Y se pronostica desde el origen Y-3, nunca desde Y.
"""
from __future__ import annotations

import dataclasses
import warnings
from datetime import date

import numpy as np

from world4.data import vintages

K_MAX = 11                      # años de pronóstico guardados por origen (validado: <= 10)
ORIGIN_SOURCE = "pwt"           # el reanclaje del Atlas usa el PIB de la PWT
L1_LAST_DATA_YEAR = 2019        # el panel del Atlas acaba en 2019
LIVE_HORIZON_YEAR = 2030

EVENTS = {
    "conflict": {"source": "ucdp", "what": "armed conflict with >= 25 battle deaths in the year (UCDP/PRIO incidence)"},
    "coup": {"source": "geopol", "what": "successful coup in the year (Powell and Thyne)"},
    "crisis": {"source": "macro_fin", "what": "banking, currency or sovereign-debt crisis in the year (Laeven-Valencia + JST)"},
    "democracy": {"source": "vdem", "what": "democratic regime (V-Dem Regimes of the World >= 2) at the end of the target year",
                  "state": True},
}
NUMERIC = {
    "gdp_pc_growth": {"source": "pwt", "unit": "% per year", "what": "real GDP per capita growth in the target year (PWT rgdpna/pop)"},
    "debt_gdp": {"source": "macro_fin", "unit": "% of GDP", "what": "gross public debt / GDP at the end of the target year"},
}


def _quiet():
    warnings.simplefilter("ignore")


def atlas_modules():
    _quiet()
    from world4.data import atlas
    atlas.require()
    from world4.core.l1_atlas import behavior, calibrate, model, validate
    return model, calibrate, validate, behavior


def origin_for(as_of: date) -> int:
    """Último año que el L1 puede usar como origen sin datos que aún no se habían publicado."""
    return min(vintages.latest_year(ORIGIN_SOURCE, as_of), L1_LAST_DATA_YEAR)


def honest_world(W, t0: int):
    """Copia de ``W`` con población y capital humano extrapolados desde el origen ``t0``."""
    pop, hc = W.pop.copy(), W.hc.copy()
    lo = max(0, t0 - 5)
    for a in (pop, hc):
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where((a[:, lo] > 0) & np.isfinite(a[:, lo]), (a[:, t0] / a[:, lo]) ** (1.0 / max(t0 - lo, 1)), 1.0)
        g = np.where(np.isfinite(g), np.clip(g, 0.9, 1.1), 1.0)
        for t in range(t0 + 1, a.shape[1]):
            a[:, t] = a[:, t0] * g ** (t - t0)
    return dataclasses.replace(W, pop=pop, hc=hc, _locf=W._locf)


_W30 = None


def live_world():
    global _W30
    if _W30 is None:
        model, *_ = atlas_modules()
        _W30 = model.build_world(horizon=LIVE_HORIZON_YEAR)
    return _W30


def ensemble(origin_year: int, R: int = 32, seed: int | None = None, live: bool = False) -> dict:
    """Pronóstico del L1 desde ``origin_year`` con la calibración de corte <= origen y la regla de
    conducta estimada hasta el origen. Devuelve arrays por año relativo k = 0..K_MAX:

    conflict_real (R,N,K+1) uint8   conflicto REALIZADO en el año k (255 = desconocido)
    hazard        (R,N,K+1) float32 probabilidad de conflicto en el año k dado el año anterior
    coup, crisis  (R,N,K+1) uint8   evento realizado en el año k
    demo          (R,N,K+1) uint8   régimen democrático al final del año k
    ly            (R,N,K+1) float32 log PIB per cápita
    debt          (R,N,K+1) float32 deuda / PIB
    """
    model, calibrate, validate, behavior = atlas_modules()
    t0 = origin_year - 1950
    if live:
        if origin_year != L1_LAST_DATA_YEAR:
            raise ValueError("el modo vivo sólo existe desde el último año de datos (2019)")
        Wx = live_world()
    else:
        if not (1970 <= origin_year <= L1_LAST_DATA_YEAR - 1):
            raise ValueError(f"origen {origin_year} fuera de [1970, {L1_LAST_DATA_YEAR - 1}]")
        Wx = honest_world(calibrate.W, t0)
    params, cut = validate.params_for(origin_year)
    seed = t0 if seed is None else seed
    bm = behavior.BehaviorModel(cutoff=t0).fit(calibrate.W, calibrate.LY)
    bc = behavior.BehaviorPolicy(bm, params, noise=1.0, freeze=t0, seed=seed + 7)
    pol = lambda s, t: s.hist_actions(t) if t <= t0 else bc(s, t)  # noqa: E731
    t1 = min(t0 + K_MAX + 2, Wx.T)                      # +1 para leer el conflicto realizado de t0+K
    sim = model.Simulator(Wx, params, R=R, seed=seed)
    o = sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0)
    K1 = K_MAX + 1

    def window(a, shift=0):
        out = np.full(a.shape[:2] + (K1,), np.nan, np.float32)
        lo, hi = t0 + shift, min(t0 + shift + K1, a.shape[2])
        if hi > lo:
            out[:, :, : hi - lo] = a[:, :, lo:hi]
        return out

    def u8(a):
        return np.where(np.isfinite(a), (a > 0.5).astype(np.uint8), np.uint8(255)).astype(np.uint8)

    return {
        "origin": origin_year, "params_cutoff": cut, "seed": seed, "live": live, "R": R,
        "conflict_real": u8(window(o["conflict"], shift=1)),   # el conflicto de t se registra en t+1
        "hazard": window(o["hazard"]),
        "coup": u8(window(o["coup"])), "crisis": u8(window(o["crisis"])), "demo": u8(window(o["demo"])),
        "ly": window(o["ly"]), "debt": window(o["debt"]),
    }


# ------------------------------------------------------------------ probabilidades de pregunta
def event_members(ens: dict, event: str, i: int, k1: int, k2: int) -> np.ndarray:
    """Probabilidad por miembro de que ``event`` ocurra en los años relativos k1..k2 (estado al
    final de k2 para ``democracy``). NaN para miembros sin el país activo."""
    if event == "democracy":
        d = ens["demo"][:, i, k2].astype(float)
        return np.where(d == 255, np.nan, d)
    if event == "conflict":
        real = ens["conflict_real"][:, i, k1:k2 + 1].astype(float)
        haz = ens["hazard"][:, i, k1:k2 + 1].astype(float)
        known = real != 255
        hit = np.any((real == 1) & known, 1).astype(float)
        p_unknown = 1 - np.prod(np.where(known, 1.0, 1 - np.nan_to_num(haz, nan=0.0)), 1)
        p = 1 - (1 - hit) * (1 - p_unknown)
        return np.where(np.isfinite(haz[:, 0]), p, np.nan)
    a = ens[event][:, i, k1:k2 + 1].astype(float)
    valid = np.all(a != 255, 1)
    return np.where(valid, np.any(a == 1, 1).astype(float), np.nan)


def numeric_members(ens: dict, variable: str, i: int, k: int) -> np.ndarray:
    if variable == "gdp_pc_growth":
        return 100.0 * (ens["ly"][:, i, k].astype(float) - ens["ly"][:, i, k - 1].astype(float))
    if variable == "debt_gdp":
        return 100.0 * ens["debt"][:, i, k].astype(float)
    raise KeyError(variable)
