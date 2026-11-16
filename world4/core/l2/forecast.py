"""Ensembles del L2 desde un origen, y el puente con el L1 en las dos direcciones.

**L1 -> L2 (lento a rápido).** El L1 da, por país y año, la probabilidad de conflicto de su ensemble con el
origen honesto de F0.5 (tres años antes: el retraso de la PWT). Entra como un rasgo más de la activación de
las díadas (``l1``), en logit. Sólo se usa si se pide (``l1=True``): W13 mide si ayuda, y el L2 corre sin el
Atlas instalado.

**L2 -> L1 (rápido a lento).** ``conflict_members`` devuelve, miembro a miembro, si un país supera 25 muertes
en un año: el mismo evento ``conflict`` del estimador estructural (UCDP, incidencia anual), pero salido de un
modelo mensual que parte del último mes con datos y no de un panel que acaba en 2019.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world4.core.l2 import agents
from world4.core.l2 import panel as pn

L1_LAG_YEARS = 3                         # el origen honesto del L1 va tres años por detrás (structural.engine)
CONFLICT_DEATHS = 25                     # umbral anual de conflicto armado de UCDP


@dataclass
class Forecast:
    region: str
    t0: int
    horizon: int
    members: int
    model: agents.AgentModel
    run: agents.AgentRun
    panel: pn.Panel = field(repr=False)
    displaced: dict | None = None
    displacement_params: dict | None = None
    notes: list = field(default_factory=list)

    def deaths(self, level: str = "country") -> np.ndarray:
        """(R, U, H) muertes por mes."""
        s = self.run.site_deaths
        return self.panel.to_cells(s) if level == "cell" else self.panel.to_countries(s)

    def months(self) -> list[str]:
        return [self.panel.label(self.t0 + h) for h in range(self.horizon)]


def l1_covariate(P: pn.Panel, n_months: int) -> np.ndarray:
    """(K, n_months) logit de la probabilidad de conflicto del L1 para el año de cada mes. El año ``y`` usa el
    ensemble de origen ``min(y - 3, 2019)``: nunca uno posterior a lo que se sabía tres años antes."""
    from world4.structural import engine, observed, store
    iso = list(observed.panel()["iso3"])
    idx = [iso.index(c) if c in iso else -1 for c in P.iso3]
    out = np.zeros((P.K, n_months))
    cache = {}
    for t in range(n_months):
        y = P.ym(t)[0]
        if y not in cache:
            o = min(y - L1_LAG_YEARS, engine.L1_LAST_DATA_YEAR)
            ens = store.get(o, live=o == engine.L1_LAST_DATA_YEAR)
            k = min(y - o, engine.K_MAX)
            p = np.full(P.K, np.nan)
            for j, i in enumerate(idx):
                if i >= 0:
                    m = engine.event_members(ens, "conflict", i, k, k)
                    p[j] = np.nanmean(m) if np.isfinite(m).any() else np.nan
            p = np.clip(np.where(np.isfinite(p), p, np.nanmean(p) if np.isfinite(p).any() else 0.1), 0.01, 0.99)
            cache[y] = np.log(p / (1 - p))
        out[:, t] = cache[y]
    return out


def run(P: pn.Panel, t0: int | None = None, horizon: int = 12, members: int = 64, seed: int = 0, l1: bool = False,
        foot: dict | None = None, births: bool = True, displacement: bool = True, disp_params: dict | None = None,
        **model_options) -> Forecast:
    """Pronóstico desde el mes ``t0`` (por defecto, el siguiente al último dato). Sólo usa lo anterior a ``t0``."""
    t0 = P.T if t0 is None else int(t0)
    if not 0 < t0 <= P.T:
        raise ValueError(f"origin {t0} outside the panel (1..{P.T})")
    notes = []
    full_T, last_final = P.T, P.last_final_month
    if t0 < P.T:
        P = P.truncated(t0)                      # por construcción: nada de t0 en adelante llega al modelo
    cov = l1_covariate(P, t0 + horizon) if l1 else None
    M = agents.fit(P, t0, l1=None if cov is None else cov[:, :t0 + 1], seed=seed, foot=foot, **model_options)
    r = agents.simulate(P, M, horizon, members=members, seed=seed, l1=None if cov is None else cov[:, t0:], births=births)
    if t0 == full_T and t0 > last_final + 1:
        notes.append(f"months after {P.label(last_final)} are UCDP candidate events (only those coded "
                     "'Clear'): the last months of history are provisional and undercounted")
    disp, th = None, None
    if displacement:
        from world4.core.l2 import displacement as dm
        th = disp_params or dm.fit(P, t0)
        disp = dm.simulate(P, th, t0, r.site_deaths)
        disp["asylum_share"], disp["asylum_top"] = dm.asylum_shares(P, t0)
    return Forecast(region=P.region, t0=t0, horizon=horizon, members=members, model=M, run=r, panel=P,
                    displaced=disp, displacement_params=th, notes=notes)


def conflict_members(f: Forecast, year: int) -> np.ndarray:
    """(R, K) 1.0 si el país supera ``CONFLICT_DEATHS`` muertes de violencia estatal en el año ``year`` del
    pronóstico; NaN si el año no cae entero en el horizonte. Los meses del año anteriores al origen se toman
    del dato observado."""
    P = f.panel
    lo, hi = P.month(year, 1), P.month(year, 12)
    if hi >= f.t0 + f.horizon or hi < f.t0:
        return np.full((f.members, P.K), np.nan)
    sb = f.run.country_type[:, :, :, 0]                                       # (R, K, H) violencia estatal
    total = sb[:, :, max(lo - f.t0, 0):hi - f.t0 + 1].sum(2)
    if lo < f.t0:
        total = total + P.deaths("country", "state_based")[:, lo:f.t0].sum(1)[None]
    return (total >= CONFLICT_DEATHS).astype(float)
