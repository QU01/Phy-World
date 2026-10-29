"""Lo que usan la API, la CLI y (más adelante) la interfaz en React: pronóstico desde un origen con el modelo
guardado si existe, y la carga útil JSON por columnas (``SCHEMA``) con todas las celdas o las que superan un
umbral."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from world4.conflict import features as F
from world4.conflict import forecast as FC
from world4.conflict.grid import TYPES, Events, Grid
from world4.data import conflict_grid as cdata

SCHEMA = "world4.conflict_grid/1"
CAVEATS = (
    "unit: PRIO-GRID 2.0 cell (0.5 x 0.5 degrees, ~55 x 55 km at the equator) x calendar month; an event is a UCDP GED "
    "event of organized violence (sb state-based, ns non-state, os one-sided) located with where_prec <= 4",
    "the last months are UCDP Candidate events (preliminary, monthly releases, about a third flagged 'Check'); "
    "older months are the final GED 26.1",
    "two models averaged in logit space with a fixed weight 0.5 (not tuned on the validation): a direct logistic "
    "hurdle per horizon and a discrete-time spatio-temporal Hawkes process simulated forward",
    "state-based deaths: P(event) x log-normal size; the size model is the weakest part (residual sd ~1.3 in log1p)",
    "conflict-history and geography only: no news, politics, economy or climate drivers yet (ViEWS adds them; "
    "World4's L1 could supply country-level ones)",
    "backtests use the revised final GED as predictors, not what was published at each origin (revision leak)",
    "PRIO-GRID covariates are fixed at their last published year (2010-2014) and cells keep their 2014 country",
)


def models_dir() -> Path:
    return cdata.root() / "models"


def model_path(origin: int) -> Path:
    return models_dir() / f"model_{cdata.month_label(origin)}.json"


def load_world() -> tuple[Grid, Events]:
    import warnings
    warnings.simplefilter("ignore")
    g = Grid()
    return g, Events(g)


def _resolve_origin(ev: Events, origin) -> int:
    last = ev.n_months - 1
    if origin is None:
        return last
    T = cdata.month_index(origin) if isinstance(origin, str) else int(origin)
    if not (F.BURN_IN + FC.HORIZON + 24 <= T <= last):
        raise ValueError(f"origin {cdata.month_label(T)} outside [{cdata.month_label(F.BURN_IN + FC.HORIZON + 24)}, "
                         f"{cdata.month_label(last)}] (data end there)")
    return T


def get_model(grid: Grid, ev: Events, T: int, *, refit: bool = False, log=None) -> tuple[FC.Model, dict]:
    """Modelo y estados del origen T: lee ``models/model_<T>.json`` o entrena (unos minutos) y lo guarda."""
    p = model_path(T)
    if p.is_file() and not refit:
        try:
            model = FC.Model.from_dict(json.loads(p.read_text(encoding="utf-8")))
            sw = F.Sweep(ev, grid)
            sw.run_to(T)
            snap = {"X": sw.features(), "Z": sw.hawkes(), "E": sw.E, "near": sw.near_history()}
            return model, snap
        except ValueError:
            pass
    if log:
        log(f"fitting the conflict-grid model for origin {cdata.month_label(T)} (a few minutes)")
    rows = F.build(ev, grid, until=T, snapshots=[T], log=log)
    model = FC.fit(ev, grid, rows, T, log=log)
    models_dir().mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    return model, rows["snapshots"][T]


def forecast(origin=None, *, samples: int = 200, seed: int = 0, refit: bool = False, log=None) -> dict:
    grid, ev = load_world()
    T = _resolve_origin(ev, origin)
    model, snap = get_model(grid, ev, T, refit=refit, log=log)
    P = FC.predict(model, grid, snap, samples=samples, seed=seed)
    P["decomposition"] = FC.decompose(model, grid, snap)
    P["_grid"], P["_ev"], P["_model"], P["_snap"] = grid, ev, model, snap
    return P


def _r(a, nd=5):
    return np.round(np.asarray(a, float), nd).tolist()


def payload(P: dict, *, min_p: float = 0.0, monthly: bool = True, bbox=None, countries=None,
            validation: dict | None = None) -> dict:
    """Carga útil JSON por columnas. ``min_p``: sólo celdas con P(algún evento en la ventana) >= min_p;
    ``bbox`` = (lon_min, lat_min, lon_max, lat_max); ``countries``: códigos Gleditsch-Ward."""
    grid, ev, model = P["_grid"], P["_ev"], P["_model"]
    T, H = P["origin"], P["horizon"]
    gw = grid.country_codes[grid.country]
    keep = P["p_window"]["any"] >= min_p
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        keep &= (grid.lon >= x0) & (grid.lon <= x1) & (grid.lat >= y0) & (grid.lat <= y1)
    if countries is not None:
        keep &= np.isin(gw, np.asarray(list(countries)))
    idx = np.nonzero(keep)[0]
    idx = idx[np.argsort(-P["p_window"]["any"][idx], kind="stable")]
    recent_e = ev.window("count", "any", T - 11, T + 1).sum(0)
    recent_d = sum(ev.window("deaths", v, T - 11, T + 1).sum(0) for v in TYPES)
    dec = P["decomposition"]
    lam = {k: sum(dec[v][k] for v in TYPES) for k in ("background", "own", "neighbours", "country")}
    tot = sum(lam.values()) + 1e-12
    cells = {"gid": grid.gid[idx].tolist(), "lon": _r(grid.lon[idx], 2), "lat": _r(grid.lat[idx], 2),
             "country_gw": gw[idx].tolist(),
             **{f"p_window_{o}": _r(P["p_window"][o][idx]) for o in FC.OUTCOMES},
             **{f"events_expected_{v}": _r(P["events"][v][:, idx].sum(0), 3) for v in TYPES},
             "deaths_sb_expected": _r(P["deaths"]["mean"][:, idx].sum(0), 2),
             **{f"driver_{k}": _r(lam[k][idx] / tot[idx], 3) for k in lam},
             "events_last12": recent_e[idx].astype(int).tolist(), "deaths_last12": _r(recent_d[idx], 0)}
    if monthly:
        for o in FC.OUTCOMES:
            cells[f"p_month_{o}"] = [_r(P["p"][o][:, i]) for i in idx]
        cells["deaths_sb_p50"] = [_r(P["deaths"]["p50"][:, i], 1) for i in idx]
        cells["deaths_sb_p90"] = [_r(P["deaths"]["p90"][:, i], 1) for i in idx]
    # países: eventos esperados y celdas en riesgo (todas las celdas, no sólo las devueltas)
    cty = []
    exp_any = sum(P["events"][v].sum(0) for v in TYPES)
    for c in np.unique(gw):
        s = gw == c
        if c < 0:
            continue
        e = float(exp_any[s].sum())
        if e < 0.05:
            continue
        cty.append({"gw": int(c), "events_expected": round(e, 2),
                    "deaths_sb_expected": round(float(P["deaths"]["mean"][:, s].sum()), 1),
                    "cells_p_any_over_50": int((P["p_window"]["any"][s] >= 0.5).sum()),
                    "max_p_any": round(float(P["p_window"]["any"][s].max()), 4)})
    cty.sort(key=lambda r: -r["events_expected"])
    months = [cdata.month_label(T + h) for h in range(1, H + 1)]
    return {
        "schema": SCHEMA, "generated": date.today().isoformat(), "origin": cdata.month_label(T), "months": months,
        "data": {"ged_version": ev.versions[0], "candidate_version": ev.versions[1],
                 "last_final_month": cdata.month_label(ev.last_final_month), "last_month": cdata.month_label(ev.n_months - 1),
                 "rules": cdata.RULES},
        "grid": {"system": "PRIO-GRID 2.0", "resolution_deg": 0.5, "n_cells": int(grid.n),
                 "cell_polygon": "[lon-0.25, lat-0.25, lon+0.25, lat+0.25] around the given centre"},
        "outcomes": {"sb": "state-based", "ns": "non-state", "os": "one-sided", "any": "any of the three"},
        "columns": {
            "p_window_<o>": f"P(at least one event of type <o> in the cell in {months[0]}..{months[-1]})",
            "p_month_<o>": "P(at least one event) per month, list of horizon length",
            "events_expected_<v>": "expected number of events in the window (Hawkes simulation mean)",
            "deaths_sb_expected": "expected state-based deaths in the window",
            "deaths_sb_p50/p90": "per-month quantiles of state-based deaths",
            "driver_<k>": "share of next month's Hawkes intensity from the structural background, the cell's own "
                          "past events, its neighbours (rings 1-2) and its country",
            "events_last12": "observed events of any type in the 12 months to the origin"},
        "units": {"lon": "degrees east (cell centre)", "lat": "degrees north (cell centre)",
                  "country_gw": "Gleditsch-Ward code", "gid": "PRIO-GRID cell id",
                  **{f"p_window_{o}": "probability" for o in FC.OUTCOMES},
                  **{f"p_month_{o}": "probability" for o in FC.OUTCOMES},
                  **{f"events_expected_{v}": "events in the window" for v in TYPES},
                  "deaths_sb_expected": "battle-related deaths in the window (UCDP best)",
                  "deaths_sb_p50": "deaths in the month", "deaths_sb_p90": "deaths in the month",
                  **{f"driver_{k}": "share of next month's intensity" for k in ("background", "own", "neighbours",
                                                                                 "country")},
                  "events_last12": "events", "deaths_last12": "deaths", "gw": "Gleditsch-Ward code",
                  "events_expected": "events in the window", "max_p_any": "probability",
                  "cells_p_any_over_50": "cells", "cells_returned": "cells", "min_p": "probability",
                  "expected_events_window": "events", "expected_deaths_sb_window": "deaths",
                  "branching_ratio": "expected offspring events per event", "validation": "see metric names",
                  "n_cells": "cells", "resolution_deg": "degrees"},
        "band": "per-cell monthly lists are the forecast path over the window, not a band; the uncertainty is the "
                "probability itself, and for deaths the p50/p90 lists",
        "exploratory": False,
        "cells": cells, "countries": cty,
        "summary": {"cells_returned": int(len(idx)), "min_p": min_p,
                    "expected_events_window": {v: round(float(P["events"][v].sum()), 1) for v in TYPES},
                    "expected_deaths_sb_window": round(float(P["deaths"]["mean"].sum()), 0),
                    "branching_ratio": model.branching()},
        "validation": validation, "caveats": list(CAVEATS),
        "sources": ["Sundberg & Melander (2013) UCDP GED, JPR 50(4); Högbladh (2026) GED codebook 26.1 (CC BY 4.0)",
                    "Hegre, Croicu, Eck & Högbladh (2020) UCDP Candidate Events Dataset (CC BY 4.0)",
                    "Tollefsen, Strand & Buhaug (2012) PRIO-GRID, JPR 49(2)"]}


def validation_headline(path=None) -> dict | None:
    """Lo esencial de la última validación guardada (``reports/conflicto_celdas/validacion.json``)."""
    from world4 import home
    p = Path(path) if path else home.PKG.parent / "reports" / "conflicto_celdas" / "validacion.json"
    if not p.is_file():
        return None
    v = json.loads(p.read_text(encoding="utf-8"))
    s = v["summary"]
    pick = lambda o, sub, m, k: s.get(o, {}).get(sub, {}).get(m, {}).get("all_h", {}).get(k)  # noqa: E731
    return {"origins": v["meta"]["origins"], "run_date": v["meta"]["run_date"],
            "any_event_aupr": {m: pick("any", "all", m, "aupr") for m in ("ensemble", "bm_neighbors12", "bm_persistence")},
            "any_event_brier": {m: pick("any", "all", m, "brier") for m in ("ensemble", "bm_neighbors12", "bm_persistence")},
            "sb_deaths_crps_africa_me": {m: pick("sb_deaths", "ame", m, "crps")
                                         for m in ("ensemble", "bm_zero", "bm_neighbors12")}}
