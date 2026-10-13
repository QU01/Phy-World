"""Lo observado: desenlaces de eventos y variables numéricas, y la tasa base con datos (A).

Todo se lee del panel del Atlas instalado. ``known_until`` aplica el retraso de publicación de
``sources.json``: una tasa base o un recalibrador a fecha ``as_of`` sólo ven años publicados.
"""
from __future__ import annotations

from datetime import date
from functools import lru_cache

import numpy as np

from world4.data import vintages
from world4.structural.engine import EVENTS, L1_LAST_DATA_YEAR, NUMERIC, atlas_modules

BASE_WINDOW_YEARS = 20       # clase de referencia: los últimos 20 años publicados
SHRINK_M = 10.0              # fuerza del prior de zona, en años-país equivalentes


@lru_cache(maxsize=1)
def panel() -> dict:
    """Arrays (N, T) con NaN donde no hay dato, 1950-2019."""
    _, calibrate, *_ = atlas_modules()
    W = calibrate.W
    obs = np.asarray(W.observed)
    regime = np.asarray(W.extra["regime_row"], float)
    return {
        "iso3": list(W.iso3),
        "conflict": np.where(obs, np.asarray(calibrate.CONF, float), np.nan),
        "coup": np.asarray(W.extra["coup_success_any"], float),
        "crisis": np.asarray(W.extra["crisis_any"], float),
        "democracy": np.where(np.isfinite(regime), (regime >= 2).astype(float), np.nan),
        "ly": np.asarray(calibrate.LY, float),
        "debt": np.asarray(calibrate.DEBT, float),
    }


def known_until(source: str, as_of: date) -> int:
    return min(vintages.latest_year(source, as_of), L1_LAST_DATA_YEAR)


def last_data_year(var: str) -> int:
    a = panel()[var]
    cols = np.where(np.isfinite(a).any(0))[0]
    return 1950 + int(cols.max())


def event_outcome(event: str, i: int, y1: int, y2: int) -> float:
    """1/0 si el evento ocurrió en [y1, y2] (estado al final de y2 para democracy); NaN si falta dato."""
    a = panel()[event][i]
    if y2 > L1_LAST_DATA_YEAR or y1 < 1950:
        return np.nan
    if event == "democracy":
        return float(a[y2 - 1950])
    seg = a[y1 - 1950: y2 - 1950 + 1]
    return np.nan if not np.all(np.isfinite(seg)) else float(np.any(seg > 0))


def numeric_outcome(variable: str, i: int, y: int) -> float:
    P = panel()
    if not (1951 <= y <= L1_LAST_DATA_YEAR):
        return np.nan
    if variable == "gdp_pc_growth":
        return float(100 * (P["ly"][i, y - 1950] - P["ly"][i, y - 1951]))
    return float(100 * P["debt"][i, y - 1950])


def zones(year: int) -> np.ndarray:
    """Centro (2) / semiperiferia (1) / periferia (0) por cuantiles de renta, como el Atlas."""
    ly = panel()["ly"][:, year - 1950]
    q80, q50 = np.nanquantile(ly, 0.8), np.nanquantile(ly, 0.5)
    z = np.where(ly >= q80, 2, np.where(ly >= q50, 1, 0)).astype(float)
    return np.where(np.isfinite(ly), z, np.nan)


def _annual(event: str, until: int) -> np.ndarray:
    """Indicador anual (N, años) de la clase de referencia; democracy = transición a autocracia."""
    a = panel()[event][:, : until - 1950 + 1]
    if event == "democracy":
        prev, cur = a[:, :-1], a[:, 1:]
        out = np.where(prev == 1, np.where(np.isfinite(cur), (cur == 0).astype(float), np.nan), np.nan)
        return np.concatenate([np.full((a.shape[0], 1), np.nan), out], 1)
    return a


def base_rate(event: str, i: int, as_of: date, y1: int, y2: int) -> dict:
    """Tasa base con datos (A): frecuencia anual del país en los últimos 20 años publicados,
    encogida hacia la de su zona de renta, y llevada a la ventana como 1-(1-p)^h. Para
    ``democracy`` (pregunta de estado) es la probabilidad de seguir/ser democracia en y2 a partir
    del último estado conocido y la tasa anual de ruptura o transición de la zona."""
    src = EVENTS[event]["source"]
    until = known_until(src, as_of)
    lo = max(1950, until - BASE_WINDOW_YEARS + 1)
    zpw = known_until("pwt", as_of)
    z = zones(zpw)
    A = _annual(event, until)[:, lo - 1950:]
    if event == "democracy":
        state = panel()["democracy"][i, until - 1950]
        D = panel()["democracy"][:, : until - 1950 + 1]
        prev, cur = D[:, lo - 1950 - 1 + 1: -1], D[:, lo - 1950 + 1:]
        same_zone = z == z[i] if np.isfinite(z[i]) else np.ones_like(z, bool)
        if not np.isfinite(state):
            return {"status": "no_data", "until": until}
        if state == 1:
            m = (prev == 1) & np.isfinite(cur) & same_zone[:, None]
            rate = float(((cur == 0) & m).sum() / max(m.sum(), 1))       # ruptura anual
            p_year_keep = 1 - rate
        else:
            m = (prev == 0) & np.isfinite(cur) & same_zone[:, None]
            rate = float(((cur == 1) & m).sum() / max(m.sum(), 1))       # transición anual
            p_year_keep = 1 - rate
        h = y2 - until
        p_same = p_year_keep ** h
        p = p_same if state == 1 else 1 - p_same
        return {"status": "ok", "p_window": float(p), "p_year": rate, "until": until, "state": float(state),
                "reference_class": (f"{'democracies' if state == 1 else 'autocracies'} in the same income zone, "
                                    f"{lo}-{until}: annual {'breakdown' if state == 1 else 'transition'} rate"),
                "events": int(round(rate * m.sum())), "trials": int(m.sum()), "sources": [src]}
    own = A[i]
    n_c, e_c = float(np.isfinite(own).sum()), float(np.nansum(own))
    peers = (z == z[i]) if np.isfinite(z[i]) else np.ones_like(z, bool)
    zone_rate = float(np.nansum(A[peers]) / max(np.isfinite(A[peers]).sum(), 1))
    p_year = (e_c + SHRINK_M * zone_rate) / (n_c + SHRINK_M)
    h = y2 - y1 + 1
    return {"status": "ok", "p_window": float(1 - (1 - p_year) ** h), "p_year": float(p_year), "until": until,
            "reference_class": (f"{panel()['iso3'][i]} {lo}-{until} ({int(e_c)} event-years of {int(n_c)}), "
                                f"shrunk to its income zone ({zone_rate:.3f}/year, m={SHRINK_M:g})"),
            "events": int(e_c), "trials": int(max(n_c, 1)), "sources": [src]}


def markov_rate(event: str, i: int, as_of: date, y1: int, y2: int) -> float:
    """Segunda referencia (sólo informe): frecuencia del desenlace en la zona CONDICIONADA al
    estado del último año publicado (evento en los últimos 3 años), con la misma distancia
    entre último dato y ventana que la pregunta."""
    src = EVENTS[event]["source"]
    until = known_until(src, as_of)
    gap, h = y1 - until, y2 - y1
    P = panel()[event]
    z = zones(known_until("pwt", as_of))
    peers = (z == z[i]) if np.isfinite(z[i]) else np.ones_like(z, bool)

    def state(j_arr, s):
        if event == "democracy":
            return P[j_arr, s - 1950]
        seg = P[j_arr, max(0, s - 1950 - 2): s - 1950 + 1]
        return np.where(np.all(np.isfinite(seg), 1), np.any(seg > 0, 1).astype(float), np.nan)

    my = state(np.array([i]), until)[0]
    if not np.isfinite(my):
        return np.nan
    hits = tot = 0.0
    for s in range(max(1955, until - BASE_WINDOW_YEARS), until - gap - h + 1):
        js = np.where(peers)[0]
        st = state(js, s)
        a1, a2 = s + gap, s + gap + h
        if event == "democracy":
            out = P[js, a2 - 1950]
        else:
            seg = P[js, a1 - 1950: a2 - 1950 + 1]
            out = np.where(np.all(np.isfinite(seg), 1), np.any(seg > 0, 1).astype(float), np.nan)
        m = (st == my) & np.isfinite(out)
        hits += float(out[m].sum())
        tot += float(m.sum())
    return (hits + 1.0) / (tot + 2.0)
