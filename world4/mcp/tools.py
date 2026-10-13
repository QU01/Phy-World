"""F9: las herramientas de World4 como funciones PURAS (patrón de TurboStudio): el servidor MCP (``server.py``) y
cualquier capa HTTP llaman a estas mismas funciones, así que no pueden divergir.

Cada resultado es JSON, lleva unidad y banda (o dice por qué no la tiene) y ``exploratory`` (W11:
``world4.honesty.audit`` lo comprueba sobre todas ellas). Ninguna escribe en disco ni lanza procesos.
"""
from __future__ import annotations

from dataclasses import asdict


def _int(v, name):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v:
        raise ValueError(f"{name} must be an integer")
    return int(v)


def w4_catalog() -> dict:
    from world4 import api
    return {"observables": api.catalog(), "exploratory": False,
            "note": "L1 variables: validated_horizon_years is the measured out-of-sample horizon; L0: behaviour modes"}


def w4_query(variable: str, region: str, as_of: int, target: int, members: int = 32) -> dict:
    from world4 import api
    d = api.query(variable, region, _int(as_of, "as_of"), _int(target, "target"), members=_int(members, "members"))
    return asdict(d)


def w4_levers() -> dict:
    from world4.scenarios import levers, narratives
    return {"levers": levers.describe(), "narratives": {k: v["what"] for k, v in narratives.NARRATIVES.items()},
            "exploratory": True, "note": "lever effects are [M] assumptions; their uncertainty is in the ensemble"}


def w4_evaluate_policy(theta: dict | None = None, narrative: str = "ssp2", members: int = 100,
                       vulnerability: bool = True) -> dict:
    import numpy as np
    from world4.scenarios import levers, robust
    th = np.zeros(len(levers.NAMES))
    for k, v in (theta or {}).items():
        if k not in levers.NAMES:
            raise ValueError(f"unknown lever {k!r}; levers: {', '.join(levers.NAMES)}")
        th[levers.NAMES.index(k)] = float(v)
    levers.to_policy(th)
    return robust.evaluate_policy(th, members=_int(members, "members"), narrative=narrative,
                                  vulnerability=bool(vulnerability))


def w4_l0r_bands(start: int = 2019, years: int = 81, members: int = 200, explore: bool = False) -> dict:
    from world4 import api
    return api.run_l0r(_int(start, "start"), _int(years, "years"), posterior=True, explore=bool(explore),
                       members=_int(members, "members"))


def w4_loop_dominance(top: int = 3, step: int = 10) -> dict:
    from world4 import api
    rows = api.loop_dominance(top=_int(top, "top"), step=_int(step, "step"))
    return {"table": [{"year": r["year"], "loops": [{"loop": n, "score": s} for n, s in r["loops"]]} for r in rows],
            "unit": "relative Loops-that-Matter score (dimensionless, signed: + reinforcing, - balancing)",
            "band": "none: one deterministic L0 run (the score explains a trajectory, it is not an estimate)",
            "exploratory": True, "fidelity": "L0"}


def _schema(props: dict, required=()) -> dict:
    return {"type": "object", "properties": props, "required": list(required), "additionalProperties": False}


TOOLS = {
    "w4_catalog": (w4_catalog, "Variables World4 can answer, with unit, fidelity and measured horizon.", _schema({})),
    "w4_query": (w4_query, "L1 distribution (p10/p50/p90 with unit and caveats) of a variable for a country, from "
                           "an origin year to a target 1-10 years ahead (1970-2018 origins).",
                 _schema({"variable": {"type": "string"}, "region": {"type": "string", "description": "ISO3 or WLD"},
                          "as_of": {"type": "integer"}, "target": {"type": "integer"},
                          "members": {"type": "integer", "default": 32}}, ("variable", "region", "as_of", "target"))),
    "w4_levers": (w4_levers, "The F8 policy levers (box, unit, declared sign) and the SSP-like narratives.", _schema({})),
    "w4_evaluate_policy": (w4_evaluate_policy, "One policy over the L0 ensemble to 2100: robust objectives, limits, "
                                               "comparison with no policy and its vulnerabilities (PRIM). Exploratory.",
                           _schema({"theta": {"type": "object", "additionalProperties": {"type": "number"}},
                                    "narrative": {"type": "string", "enum": ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]},
                                    "members": {"type": "integer", "default": 100},
                                    "vulnerability": {"type": "boolean", "default": True}})),
    "w4_l0r_bands": (w4_l0r_bands, "World trajectories of the regional L0 with the F7 posterior: p10-p90 bands of "
                                   "population, GDP, energy, cereals, cropland and GMST. A behaviour mode.",
                     _schema({"start": {"type": "integer"}, "years": {"type": "integer"},
                              "members": {"type": "integer"}, "explore": {"type": "boolean"}})),
    "w4_loop_dominance": (w4_loop_dominance, "Dominant feedback loops of the L0 (Loops that Matter score) by decade.",
                          _schema({"top": {"type": "integer"}, "step": {"type": "integer"}})),
}


def call(name: str, arguments: dict | None = None) -> dict:
    if name not in TOOLS:
        raise KeyError(name)
    fn, _, schema = TOOLS[name]
    args = dict(arguments or {})
    extra = set(args) - set(schema["properties"])
    if extra:
        raise ValueError(f"unknown arguments {sorted(extra)}")
    missing = [k for k in schema["required"] if k not in args]
    if missing:
        raise ValueError(f"missing arguments {missing}")
    return fn(**args)
