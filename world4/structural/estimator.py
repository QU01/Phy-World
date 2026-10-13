"""Estimador estructural para Prime Radiant (F0.5): una pregunta enlazada -> un bloque JSON.

Petición (dict, normalmente leída de un JSON que escribe Studio)::

    {"variable": "conflict" | "coup" | "crisis" | "democracy" | "gdp_pc_growth" | "debt_gdp",
     "region": "COL",                      # ISO3
     "as_of": "2026-10-11",                # fecha del pronóstico (lo que se sabía)
     "closes_at": "2028-12-31",            # fin de la ventana / año objetivo
     "kind": "binary" | "numeric",
     "edges": [0, 2, 4]}                   # sólo numéricas, en la unidad de la variable

Respuesta: ``status`` es "ok" o uno de los diagnósticos que Studio convierte en
``structural_*`` sin abortar la corrida (unbound, out_of_domain, asof_violation, unavailable).
Nunca se entrega una cifra fuera del dominio medido sin ``exploratory: true``.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np

from world4 import MODEL_REVISION
from world4.structural import engine, observed, questions, recal, store

VARIABLES = {**{k: {**v, "kind": "binary"} for k, v in engine.EVENTS.items()},
             **{k: {**v, "kind": "numeric"} for k, v in engine.NUMERIC.items()}}
K_EVALUATED = (questions.ORIGIN_LAG + min(questions.H_RANGE), questions.ORIGIN_LAG + max(questions.H_RANGE))
K_VALIDATED = 10
# Fecha de publicación de la proyección WPP 2024 que usa el origen vivo
LIVE_EARLIEST_AS_OF = date(2024, 7, 11)

CAVEATS = (
    "annual model: a window is whole calendar years; the first year counts from 1 July",
    "the L1 is the Atlas del Sistema-Mundo with honest population (D1 closed here) but psi_t pinned (D3)",
    "the ensemble has 32 members; probabilities are recalibrated on past questions known at as_of",
)


def _date(x) -> date:
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return datetime.fromisoformat(str(x).replace("Z", "+00:00")).date()


def _fail(status: str, message: str, **extra) -> dict:
    return {"status": status, "message": message, "model_revision": MODEL_REVISION, **extra}


def estimate(req: dict) -> dict:
    var = req.get("variable")
    if var not in VARIABLES:
        return _fail("unbound", f"variable {var!r} is not a World4 observable", known=sorted(VARIABLES))
    spec = VARIABLES[var]
    kind = req.get("kind", spec["kind"])
    if kind != spec["kind"]:
        return _fail("unbound", f"{var} answers {spec['kind']} questions, not {kind}")
    try:
        as_of, closes = _date(req["as_of"]), _date(req["closes_at"])
    except (KeyError, ValueError) as exc:
        return _fail("unbound", f"bad dates: {exc}")
    try:
        P = observed.panel()
        S = store.load()
    except store.StoreMissing as exc:
        return _fail("unavailable", str(exc))
    except Exception as exc:                                          # noqa: BLE001
        return _fail("unavailable", f"{type(exc).__name__}: {exc}")
    region = req.get("region")
    if region not in P["iso3"]:
        return _fail("unbound", f"region {region!r} is not one of the 180 ISO3 codes of the L1")
    i = P["iso3"].index(region)

    origin = engine.origin_for(as_of)
    live = origin == engine.L1_LAST_DATA_YEAR
    if live and as_of < LIVE_EARLIEST_AS_OF:
        return _fail("asof_violation", f"the live origin uses WPP 2024 (published {LIVE_EARLIEST_AS_OF}); "
                     f"as_of {as_of} is earlier and no honest 2019 origin exists in the store")
    y1 = as_of.year + (1 if as_of >= date(as_of.year, 7, 1) else 0)
    y2 = closes.year
    y1 = min(y1, y2)
    k1, k2 = y1 - origin, y2 - origin
    if not (1 <= k1 <= k2 <= K_VALIDATED):
        return _fail("out_of_domain", f"window {y1}-{y2} is {k1}-{k2} years from the origin {origin}; "
                     f"the L1 has out-of-sample tests only 1-{K_VALIDATED} years ahead", origin=origin)
    ens = store.get(origin, live=live)
    k_cal = int(np.clip(k2, *K_EVALUATED))
    exploratory = not (K_EVALUATED[0] <= k2 <= K_EVALUATED[1])
    caveats = list(CAVEATS)
    if live:
        caveats.append(f"origin {origin}: the L1 has no data after {engine.L1_LAST_DATA_YEAR}; "
                       f"{as_of.year - origin} years pass without assimilation (COVID-19 included)")
    if exploratory:
        caveats.append(f"k2={k2} is outside the evaluated range {K_EVALUATED}: recalibrated with k2={k_cal}")
    base = {"status": "ok", "variable": var, "what": spec["what"], "region": region, "kind": kind,
            "as_of": as_of.isoformat(), "window": [y1, y2], "origin": origin, "k": [k1, k2],
            "live": live, "exploratory": exploratory, "caveats": caveats,
            "provenance": {"model_revision": MODEL_REVISION, "atlas_commit": S["meta"]["atlas_commit"],
                           "params_cutoff": ens["params_cutoff"], "members": S["meta"]["R"],
                           "store_version": store.STORE_VERSION}}

    if kind == "binary":
        m = engine.event_members(ens, var, i, k1, k2)
        if np.isfinite(m).sum() < questions.MIN_MEMBERS:
            return _fail("unavailable", f"{region} is not active in the L1 at {origin}")
        p_raw = float(np.nanmean(m))
        cal = questions.event_calibrator(var, k_cal, as_of)
        br = observed.base_rate(var, i, as_of, y1, y2)
        return {**base, "p_raw": p_raw, "p": float(cal(p_raw)), "calibrator": cal.describe(),
                "member_spread": float(np.nanstd(m)),
                "base_rate_data": br if br["status"] == "ok" else None}

    edges = req.get("edges")
    if not edges or len(edges) < 2:
        return _fail("unbound", "a numeric question needs edges")
    m = engine.numeric_members(ens, var, i, k2)
    if np.isfinite(m).sum() < questions.MIN_MEMBERS:
        return _fail("unavailable", f"{region} is not active in the L1 at {origin}")
    cal = questions.numeric_calibrator(var, k_cal, as_of)
    probs = recal.bin_probabilities(cal, m, edges)
    q10, q50, q90 = (float(v) for v in cal.quantile(m, [0.1, 0.5, 0.9]))
    return {**base, "unit": spec["unit"], "edges": list(map(float, edges)),
            "distribution": {f"b{j}": float(p) for j, p in enumerate(probs)},
            "quantiles": {"p10": q10, "p50": q50, "p90": q90}, "calibrator": cal.describe(),
            "raw_quantiles": dict(zip(("p10", "p50", "p90"), map(float, np.nanpercentile(m, [10, 50, 90]))))}
