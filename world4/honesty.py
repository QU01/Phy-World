"""W11: honestidad verificable (hoja de ruta, principio 6).

Tres reglas, cada una con su detector y su mutante:

1. **Ninguna cifra sin unidad ni banda.** En un resultado JSON, toda cantidad numérica (salvo los metadatos de
   ``META``: años, miembros, contadores, semillas...) necesita una unidad en su objeto o en uno que lo contenga
   (``unit``, ``units`` o ``*_unit``). Todo ``p50`` necesita ``p10`` y ``p90``. Una serie (lista de números) sin
   banda en su objeto necesita una banda en un antepasado o una nota ``band`` que diga por qué no la tiene.
2. **La palabra «validated» está prohibida** salvo como nombre del horizonte MEDIDO fuera de muestra o como la
   propia regla que la prohíbe.
3. **``exploratory`` correcto en todo camino:** cada resultado público lo declara; el L0 y todo lo de F8 lo
   llevan en verdadero; el L1 dentro de su horizonte medido (1-10 años), en falso.

``measure()`` las aplica a la API pública, a todas las herramientas MCP y al informe HTML, y comprueba que cada
detector caza su mutante (quitar la etiqueta, la unidad o la banda; plantar la palabra).
"""
from __future__ import annotations

import re
from pathlib import Path

META = {
    "members", "R", "year", "years", "origin", "target", "as_of", "cutoff", "seed", "n", "index", "model_revision",
    "horizon", "start", "end", "population", "feasible", "evaluations", "seconds", "bytes", "countries", "box",
    "lo", "hi", "range", "coverage", "density", "support", "cut", "base_rate", "failures", "boxes", "trajectory",
    "validated_horizon_years", "declared_sign", "params_cutoff", "posterior_cutoff", "violated_share", "frac",
    "top", "step", "n_monotone", "rationality_index", "theta", "constraints_minimised", "atlas_commit",
    "better_than_baseline_share", "share", "g_best", "g_closest",
}
UNIT_KEYS = ("unit", "units")
ALLOWED_VALIDATED = (re.compile(r"validated_horizon_years"), re.compile(r"VALIDATED_HORIZON_YEARS"),
                     re.compile(r"'validated' forbidden"), re.compile(r"forbids 'validated'"),
                     re.compile(r"validated <= "), re.compile(r"ALLOWED_VALIDATED"),
                     re.compile(r"«validated»"), re.compile(r"\"validated\""))
WORD = re.compile(r"\bvalidated\b", re.I)


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _numlist(x) -> bool:
    return isinstance(x, list) and len(x) > 1 and all(_is_num(v) or v is None for v in x) and any(_is_num(v) for v in x)


def _units_here(d: dict):
    """(una unidad que cubre todo lo de abajo, nombres que cubre un dict ``units``)."""
    whole = isinstance(d.get("unit"), str) or any(k.endswith("_unit") for k in d) or isinstance(d.get("units"), str)
    named = set(d["units"]) if isinstance(d.get("units"), dict) else set()
    return whole, named


def _has_band(d: dict) -> bool:
    return ("p10" in d and "p90" in d) or "band" in d or "percentiles" in d


def audit(obj, path: str = "$", unit_above: bool = False, band_above: bool = False, named=frozenset(),
          lineage=()) -> list[str]:
    """Violaciones de la regla 1 en un resultado JSON. Una cadena ``unit`` cubre todo lo que cuelga de su objeto;
    un dict ``units`` sólo cubre las claves que nombra (en cualquier nivel por debajo)."""
    out = []
    if isinstance(obj, dict):
        whole, nm = _units_here(obj)
        unit_here = unit_above or whole
        named = frozenset(named | nm)
        band_here = band_above or _has_band(obj)
        if "p50" in obj and not ("p10" in obj and "p90" in obj):
            out.append(f"{path}: p50 without p10/p90")
        for k, v in obj.items():
            if k in META or k in UNIT_KEYS or k.endswith("_unit"):
                continue
            p = f"{path}.{k}"
            covered = unit_here or k in named or any(x in named for x in lineage)
            if _is_num(v) or _numlist(v):
                if not covered:
                    out.append(f"{p}: number without unit")
                if _numlist(v) and k not in ("p10", "p50", "p90") and not band_here:
                    out.append(f"{p}: series without band (or a 'band' note saying why)")
            elif isinstance(v, (dict, list)):
                out += audit(v, p, unit_here, band_here, named, lineage + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                out += audit(v, f"{path}[{i}]", unit_above, band_above, named, lineage)
    return out


def offending_words(text: str) -> list[str]:
    """Regla 2: líneas con «validated» fuera de los usos permitidos."""
    bad = []
    for ln in text.splitlines():
        s = ln
        for a in ALLOWED_VALIDATED:
            s = a.sub("", s)
        if WORD.search(s):
            bad.append(ln.strip()[:160])
    return bad


def exploratory_ok(result: dict, expected: bool) -> list[str]:
    """Regla 3."""
    if "exploratory" not in result:
        return ["missing 'exploratory'"]
    if bool(result["exploratory"]) != expected:
        return [f"exploratory={result['exploratory']} but expected {expected}"]
    return []


def code_scan(root: Path | None = None) -> dict:
    from world4 import home
    root = root or home.PKG
    # el port del Atlas no se toca (W2); honesty.py contiene la palabra porque ES la regla
    files = [p for p in root.rglob("*.py") if "l1_atlas" not in p.parts and p.name != "honesty.py"]
    bad = {str(p.relative_to(root.parent)): offending_words(p.read_text(encoding="utf-8")) for p in files}
    return {k: v for k, v in bad.items() if v}


def measure(log=print, with_query: bool = True, with_report: bool = True) -> dict:
    """Aplica las tres reglas a todos los caminos públicos y comprueba los mutantes."""
    import copy
    import json
    import warnings
    warnings.simplefilter("ignore")
    from dataclasses import asdict

    from world4 import api
    from world4.mcp import tools
    paths, viol = {}, {}

    def check(name, result, expected_exploratory):
        js = json.loads(json.dumps(result, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        v = audit(js) + exploratory_ok(js, expected_exploratory) + offending_words(json.dumps(js))
        paths[name] = js
        if v:
            viol[name] = v[:12]

    log("W11: catalog, L0 bands, policy, loops (API and MCP)")
    check("api.catalog", {"observables": api.catalog(), "exploratory": False}, False)
    check("api.run_l0r", api.run_l0r(years=30), True)
    check("api.run_l0r(posterior)", api.run_l0r(years=30, posterior=True, members=40), True)
    check("api.evaluate_policy", api.evaluate_policy({"carbon_price": 100}, members=40), True)
    for name in tools.TOOLS:
        if name == "w4_query":
            continue
        kw = {"members": 40} if name in ("w4_evaluate_policy", "w4_l0r_bands") else {}
        expected = name != "w4_catalog"
        check(f"mcp.{name}", tools.call(name, kw), expected)
    if with_query:
        log("W11: L1 query inside its measured horizon")
        check("api.query", asdict(api.query("gdp_pc", "MEX", 2005, 2010, members=8)), False)
        check("mcp.w4_query", tools.call("w4_query", {"variable": "gdp_pc", "region": "MEX", "as_of": 2005,
                                                      "target": 2010, "members": 8}), False)
    from world4.conflict import service as cg_service
    from world4.data import conflict_grid as cg_data
    if cg_data.available() and any(cg_service.models_dir().glob("model_*.json")):
        log("W11: conflict-grid forecast (F11), with its saved model")
        last = sorted(cg_service.models_dir().glob("model_*.json"))[-1].stem.split("_")[1]
        check("api.conflict_grid_forecast", api.conflict_grid_forecast(last, min_p=0.2, samples=40), False)
    html_bad = []
    if with_report:
        log("W11: HTML report")
        from world4 import report
        d = report.collect(None, members=40, campaign=False, log=lambda *a: None)
        page = report.build(d)
        n_fig, n_band = page.count("<svg"), page.count("<polygon")
        n_unit = len(re.findall(r'<tspan class="u">\[[^\]]+\]</tspan>', page))
        if n_fig == 0 or n_unit < n_fig or n_band < n_fig:
            html_bad.append(f"{n_fig} figures, {n_unit} with a unit, {n_band} bands")
        if "Exploratory." not in page:
            html_bad.append("the report does not say it is exploratory")
        html_bad += offending_words(page)
        check("report.json", {k: v for k, v in d.items() if k != "provenance"}, True)
    code = code_scan()
    # mutantes: cada detector debe cazar el suyo
    pol = paths["api.evaluate_policy"]
    m1 = copy.deepcopy(pol); m1.pop("exploratory")
    m2 = copy.deepcopy(paths["api.run_l0r(posterior)"]); m2["world"]["gmst"].pop("unit")
    m3 = copy.deepcopy(paths["api.run_l0r(posterior)"]); m3["world"]["pop"].pop("p90")
    m4 = copy.deepcopy(paths["api.run_l0r"]); m4.pop("band")
    mutants = {"label removed (exploratory)": bool(exploratory_ok(m1, True)),
               "unit removed": bool(audit(m2)), "band removed (p90)": bool(audit(m3)),
               "series without band note": bool(audit(m4)),
               "planted 'validated'": bool(offending_words('print("this model is validated")'))}
    ok = not viol and not html_bad and not code and all(mutants.values())
    res = {"paths": sorted(paths), "violations": viol, "report": html_bad, "code_words": code, "mutants": mutants,
           "ok": ok}
    log(f"W11: {len(paths)} public paths, {sum(len(v) for v in viol.values())} violations; report "
        f"{'clean' if not html_bad else html_bad}; code {'clean' if not code else list(code)}; mutants caught "
        f"{sum(mutants.values())}/{len(mutants)}")
    return res
