"""Servidor HTTP de la interfaz (``world4_cli.py --serve``): una API JSON sobre ``world4.api`` y los módulos
del sistema, y la aplicación React compilada (``ui/dist``) en ``/``.

Sólo la biblioteca estándar (``http.server``), sin dependencias nuevas: es un servidor local de un usuario,
no un servicio público. Lo caro se guarda en memoria por argumentos (y el pronóstico por celda también en
``runs/conflict_grid/``); dos peticiones iguales a la vez esperan a la misma cuenta. Si faltan datos, la
respuesta es 503 con el remedio (``{"error", "remedy"}``) y la interfaz lo muestra tal cual.

Rutas (todas GET salvo ``/api/policy``):

    /api/status                         datos instalados y qué vistas pueden correr
    /api/sources                        registro de fuentes (W12)
    /api/l0?scenario=1                  World3-03, escenarios 1-5 del libro de 2004
    /api/social?scenario=1&n=40&feedback=1&seed=1     World3 <-> sociedad de 180 países (bandas + mapa)
    /api/catalog                        variables del L1
    /api/l1?variable=&region=&as_of=&horizon=&members=   trayectoria p10/p50/p90 del L1
    /api/levers                         palancas de F8 y narrativas
    POST /api/policy {theta, narrative, members}      F8 (necesita los datos de F5)
    /api/l0r?posterior=1&members=100    L0 regional de F6 con bandas (necesita F5)
    /api/conflict/meta                  orígenes posibles, modelos guardados, nombres de países (UCDP)
    /api/conflict/forecast?origin=&min_p=0.01         carga útil world4.conflict_grid/1
    /api/conflict/cell?gid=&origin=     historia mensual observada de una celda + su pronóstico
    /api/conflict/validation            resumen de reports/conflicto_celdas/validacion.json
    /api/density?origin=2025&h=1&cells=1    densidad de eventos por país (y por celda) con las regresiones del Atlas
    /api/density/scenario?origin=2019&members=32  densidad año a año sobre las trayectorias del L1
    /api/density/study                  resumen del estudio de factores (reports/factores_eventos)
"""
from __future__ import annotations

import gzip
import json
import threading
import traceback
import warnings
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from world4 import home

UI_DIST = home.PKG.parent / "ui" / "dist"
_locks: dict = {}
_cache: dict = {}
_glock = threading.Lock()


class DataMissing(RuntimeError):
    def __init__(self, msg: str, remedy: str):
        super().__init__(msg)
        self.remedy = remedy


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if hasattr(o, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(o)
    return str(o)


def cached(key, fn):
    """Calcula una vez por clave; las peticiones concurrentes con la misma clave esperan a la primera."""
    with _glock:
        if key in _cache:
            return _cache[key]
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if key not in _cache:
            _cache[key] = fn()
        return _cache[key]


# ---------------------------------------------------------------- estado y fuentes
def status() -> dict:
    from world4 import MODEL_REVISION
    from world4.data import atlas, climate, conflict_grid, energy, food, wpp
    have = {"atlas": atlas.verify()["ok"], "wpp": wpp.available(), "energy": energy.available(),
            "climate": climate.available(), "food": food.available(), "conflict_grid": conflict_grid.available()}
    l1 = have["atlas"] and have["wpp"] and have["energy"]
    remedy = {k: f"python world4_cli.py --install-{k.replace('_', '-')}" for k in have}
    caps = {
        "conflict": {"ok": have["conflict_grid"], "needs": ["conflict_grid"]},
        "social": {"ok": True, "needs": []},
        "l0": {"ok": True, "needs": []},
        "l1": {"ok": l1, "needs": ["atlas", "wpp", "energy"]},
        "l0r": {"ok": l1 and have["food"], "needs": ["atlas", "wpp", "energy", "food"]},
        "policy": {"ok": l1 and have["food"], "needs": ["atlas", "wpp", "energy", "food"]},
    }
    for c in caps.values():
        c["missing"] = [k for k in c["needs"] if not have[k]]
        c["remedy"] = [remedy[k] for k in c["missing"]]
    return {"model_revision": MODEL_REVISION, "data": have, "remedy": remedy, "capabilities": caps,
            "ui_built": (UI_DIST / "index.html").is_file()}


def sources() -> dict:
    p = home.PKG / "data" / "sources.json"
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- World3 (L0)
L0_SERIES = {
    "pop": ("Población", "personas"), "le": ("Esperanza de vida", "años"),
    "fpc": ("Alimento per cápita", "kg equivalente vegetal / persona / año"),
    "iopc": ("Producción industrial per cápita", "USD de 1968 / persona / año"),
    "sopc": ("Servicios per cápita", "USD de 1968 / persona / año"),
    "nrfr": ("Recursos no renovables restantes", "fracción de la dotación inicial"),
    "ppolx": ("Contaminación persistente", "índice, 1970 = 1"),
    "al": ("Tierra cultivable", "hectáreas"), "hwi": ("Índice de bienestar humano", "índice 0-1"),
    "hef": ("Huella ecológica", "índice, 1 = capacidad de la Tierra"),
    "cbr": ("Tasa bruta de natalidad", "nacimientos / 1000 personas / año"),
    "cdr": ("Tasa bruta de mortalidad", "muertes / 1000 personas / año"),
}
L0_SCENARIOS = {
    1: "Negocios como de costumbre (Limits to Growth, 30-Year Update)", 2: "Más recursos no renovables",
    3: "+ control de la contaminación", 4: "+ rendimiento de la tierra", 5: "+ protección contra la erosión",
}   # los que define world4.core.l0.SCENARIOS (1-5 del libro de 2004)


def l0(scenario: int = 1) -> dict:
    from world4 import api
    from world4.core import l0 as l0mod
    if scenario not in l0mod.SCENARIOS:
        raise ValueError(f"scenario must be one of {sorted(l0mod.SCENARIOS)}")
    r = api.run_l0(scenario=scenario)
    t = np.asarray(r.time)
    keep = np.isclose(t % 1.0, 0.0)
    series = {k: np.asarray(r.series[k])[keep, 0].tolist() for k in L0_SERIES if k in r.series}
    return {"scenario": scenario, "scenarios": L0_SCENARIOS, "years": t[keep].astype(int).tolist(),
            "series": series, "labels": {k: v[0] for k, v in L0_SERIES.items()},
            "units": {k: v[1] for k, v in L0_SERIES.items()}, "fidelity": "L0", "exploratory": True,
            "band": "one deterministic run of World3-03: a behaviour mode, not a forecast"}


# ---------------------------------------------------------------- World3 <-> sociedad
SOCIAL_WORLD = {
    "in_conflict": ("Países en conflicto civil", "países"), "conflict_onset": ("Inicios de conflicto", "países / año"),
    "coup_success": ("Golpes de Estado exitosos", "golpes / año"), "in_crisis": ("Países en crisis de deuda", "países"),
    "demo_pop": ("Población en democracia", "fracción de la población mundial"),
    "pop_conflict": ("Población en países en conflicto", "fracción de la población mundial"),
    "loss": ("Pérdida de PIB por eventos", "fracción del PIB mundial"),
    "capacity": ("Capacidad de implementar política", "fracción"),
    "gdp_index": ("PIB mundial", "índice, 2020 = 1"),
}
SOCIAL_W3 = {"pop": ("Población (World3)", "personas"), "iopc": ("Producción industrial per cápita", "USD 1968"),
             "fpc": ("Alimento per cápita", "kg / persona / año"), "le": ("Esperanza de vida", "años"),
             "ppolx": ("Contaminación persistente", "índice, 1970 = 1"), "nrfr": ("Recursos restantes", "fracción")}


def _bands(a: np.ndarray) -> dict:
    q = np.nanpercentile(np.asarray(a, float), [10, 50, 90], axis=1)
    return {"p10": q[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist()}


def social(scenario: int = 1, n: int = 40, feedback: bool = True, seed: int = 1) -> dict:
    from world4.core import l0 as l0mod
    from world4.core import social as smod
    cons, tabs = l0mod.SCENARIOS[scenario]
    w3, soc = smod.run_coupled(n=n, seed=seed, feedback=feedback, constants=cons or None, tables=tabs or None)
    t = np.asarray(w3.time)
    sel = (t >= 2000) & np.isclose(t % 1.0, 0.0)
    world = {k: {**_bands(soc.series[k]), "label": v[0], "unit": v[1]} for k, v in SOCIAL_WORLD.items()
             if k in soc.series}
    w3s = {k: {**_bands(np.asarray(w3.series[k])[sel]), "label": v[0], "unit": v[1]} for k, v in SOCIAL_W3.items()
           if k in w3.series}
    cty = {k: np.round(np.asarray(soc.country[k], float), 4).tolist() for k in ("p_conflict", "conflict", "p_coup",
                                                                               "dem", "crisis")}
    return {"scenario": scenario, "scenarios": L0_SCENARIOS, "members": n, "feedback": feedback,
            "years": [int(y) for y in soc.years], "world": world, "w3_years": t[sel].astype(int).tolist(),
            "world3": w3s, "iso3": list(soc.iso3), "country": cty,
            "country_units": {"p_conflict": "probability of civil-conflict onset in the year (member mean)",
                              "conflict": "share of members in civil conflict", "p_coup": "probability of a coup attempt",
                              "dem": "share of members democratic", "crisis": "share of members in sovereign-debt crisis"},
            "groups": {g: {k: _bands(v) for k, v in d.items()} for g, d in soc.groups.items()},
            "fidelity": "L0 + social", "exploratory": True,
            "band": "p10-p90 over ensemble members (country noise, event draws); World3 is deterministic unless the "
                    "feedback is on"}


# ---------------------------------------------------------------- L1
def l1(variable: str, region: str, as_of: int, horizon: int = 10, members: int = 32) -> dict:
    from world4 import api
    from world4.data import atlas
    if not atlas.verify()["ok"]:
        raise DataMissing("Atlas data not installed", "python world4_cli.py --install-atlas")
    try:
        return api.query_path(variable, region, as_of, horizon, members=members)
    except api.QueryError as exc:
        msg = str(exc)
        if "install" in msg:
            raise DataMissing(msg, msg.split(": ")[-1]) from exc
        raise ValueError(msg) from exc


def catalog() -> dict:
    from world4 import api
    from world4.core.l1_atlas import calibrate
    try:
        iso = list(calibrate.W.iso3)
    except Exception:  # noqa: BLE001 (sin Atlas: el catálogo sigue sirviendo)
        iso = []
    return {"observables": [o for o in api.catalog() if o["fidelity"] == "L1"], "regions": iso,
            "first_origin": api.L1_FIRST_ORIGIN, "last_year": api.L1_LAST_YEAR,
            "horizon": api.VALIDATED_HORIZON_YEARS, "exploratory": False}


# ---------------------------------------------------------------- F8 / F6
def levers() -> dict:
    from world4.scenarios import levers as lv
    from world4.scenarios import narratives
    return {"levers": [{"name": x.name, "lo": x.lo, "hi": x.hi, "unit": x.unit, "what": x.what, "target": x.target,
                        "sign": x.sign} for x in lv.LEVERS],
            "narratives": list(narratives.NARRATIVES), "default_narrative": narratives.DEFAULT}


def _food_guard():
    from world4.data import atlas, food
    if not atlas.verify()["ok"]:
        raise DataMissing("Atlas data not installed", "python world4_cli.py --install-atlas")
    if not food.available():
        raise DataMissing("the regional L0 (F6) and the policy search (F8) need the FAOSTAT food data (F5)",
                          "python world4_cli.py --install-food")


def policy(theta: dict, narrative: str = "ssp2", members: int = 100) -> dict:
    from world4 import api
    _food_guard()
    return api.evaluate_policy(theta, members=members, narrative=narrative)


def l0r(posterior: bool = True, members: int = 100) -> dict:
    from world4 import api
    _food_guard()
    return api.run_l0r(posterior=posterior, members=members)


# ---------------------------------------------------------------- conflicto por celda (F11)
@lru_cache(maxsize=1)
def _ged_names() -> dict:
    """{Gleditsch-Ward: nombre del país} según UCDP GED (el nombre que usa UCDP para ese código)."""
    import pandas as pd

    from world4.data import conflict_grid as cdata
    p = cdata.root() / "raw" / "ged261-csv.zip"
    if not p.is_file():
        return {}
    d = pd.read_csv(p, usecols=["country", "country_id"]).drop_duplicates("country_id", keep="last")
    return {int(g): str(n) for g, n in zip(d.country_id, d.country)}


def _conflict_world():
    from world4.conflict import service
    from world4.data import conflict_grid as cdata
    if not cdata.available():
        raise DataMissing("conflict-grid data not installed", "python world4_cli.py --install-conflict-grid")
    return cached(("conflict_world",), service.load_world)


def conflict_meta() -> dict:
    from world4.conflict import features as F
    from world4.conflict import forecast as FC
    from world4.conflict import service
    from world4.data import conflict_grid as cdata
    grid, ev = _conflict_world()
    first, last = F.BURN_IN + FC.HORIZON + 24, ev.n_months - 1
    saved = sorted(p.stem.split("_")[1] for p in service.models_dir().glob("model_*.json"))
    gw = grid.country_codes[grid.country]
    boxes, points = {}, {}
    for c in np.unique(gw):
        if c < 0:
            continue
        s = np.nonzero(gw == c)[0]
        boxes[int(c)] = [float(grid.lon[s].min() - 0.25), float(grid.lat[s].min() - 0.25),
                         float(grid.lon[s].max() + 0.25), float(grid.lat[s].max() + 0.25)]
        # la celda del país más cercana a la media de sus celdas: cae dentro del país (la media puede no caer)
        k = s[np.argmin((grid.lon[s] - grid.lon[s].mean()) ** 2 + (grid.lat[s] - grid.lat[s].mean()) ** 2)]
        points[int(c)] = [float(grid.lon[k]), float(grid.lat[k])]
    return {"first_origin": cdata.month_label(first), "last_month": cdata.month_label(last),
            "last_final_month": cdata.month_label(ev.last_final_month), "saved_origins": saved,
            "versions": list(ev.versions), "country_names": {str(k): v for k, v in _ged_names().items()},
            "country_bbox": {str(k): v for k, v in boxes.items()},
            "country_point": {str(k): v for k, v in points.items()}, "rules": cdata.RULES}


def conflict_forecast(origin: str | None = None, min_p: float = 0.01) -> bytes:
    """La carga útil gzip (se guarda también en ``runs/conflict_grid/forecast_<origin>_<min_p>.json``)."""
    from world4.conflict import service
    from world4.data import conflict_grid as cdata
    grid, ev = _conflict_world()
    T = service._resolve_origin(ev, origin)
    label = cdata.month_label(T)

    def build():
        path = home.runs() / "conflict_grid" / f"forecast_{label}_{min_p:g}.json"
        if path.is_file():
            raw = path.read_bytes()
        else:
            P = service.forecast(label)
            out = service.payload(P, min_p=min_p, validation=service.validation_headline())
            raw = json.dumps(out, ensure_ascii=False, default=_jsonable).encode("utf-8")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        return gzip.compress(raw, 6)

    return cached(("conflict_forecast", label, min_p), build)


def conflict_cell(gid: int, origin: str | None = None, start: str = "2010-01") -> dict:
    from world4.conflict.grid import TYPES
    from world4.conflict import service
    from world4.data import conflict_grid as cdata
    grid, ev = _conflict_world()
    T = service._resolve_origin(ev, origin)
    k = np.searchsorted(grid.gid, gid)
    if k >= grid.n or grid.gid[k] != gid:
        raise ValueError(f"gid {gid} is not a PRIO-GRID land cell")
    m0 = max(cdata.month_index(start), 0)
    hist = {v: ev.count[v][m0:T + 1, k].toarray().ravel().astype(int).tolist() for v in TYPES}
    deaths = {v: ev.deaths[v][m0:T + 1, k].toarray().ravel().tolist() for v in TYPES}
    gw = int(grid.country_codes[grid.country[k]])
    nb = grid.ring1[k].indices
    return {"gid": int(gid), "lon": float(grid.lon[k]), "lat": float(grid.lat[k]), "country_gw": gw,
            "country": _ged_names().get(gw, f"GW {gw}"), "origin": cdata.month_label(T),
            "months": [cdata.month_label(m) for m in range(m0, T + 1)], "events": hist, "deaths": deaths,
            "events_total_since_1989": {v: int(ev.count[v][: T + 1, k].sum()) for v in TYPES},
            "neighbours_events_last12": int(sum(ev.count[v][T - 11:T + 1][:, nb].sum() for v in TYPES)),
            "static": {name: float(grid.static[k, i]) for i, name in enumerate(
                ("log1p_pop", "log1p_capdist_km", "log1p_bdist_km", "log1p_ttime_min", "mountains", "urban",
                 "log1p_imr", "excluded", "resources"))},
            "units": {"events": "events per month", "deaths": "UCDP best deaths per month"},
            "band": "observed history, no band"}


def conflict_validation() -> dict:
    p = home.PKG.parent / "reports" / "conflicto_celdas" / "validacion.json"
    if not p.is_file():
        raise DataMissing("no validation run saved", "python world4_cli.py --validate-conflict-grid")

    def build():
        v = json.loads(p.read_text(encoding="utf-8"))
        per = {}
        for r in v["per_origin_month"]:
            if r["subset"] == "all" and r["outcome"] in ("any", "sb") and "aupr" in r:
                per.setdefault(r["outcome"], {}).setdefault(r["model"], {}).setdefault(r["origin"], []).append(r["aupr"])
        by_origin = {o: {m: {k: float(np.mean(x)) for k, x in d.items()} for m, d in ms.items()} for o, ms in per.items()}
        return {"meta": v["meta"], "summary": v["summary"], "window12": v["window12"],
                "calibration": v["calibration"], "hawkes": v["hawkes"], "aupr_by_origin": by_origin}

    return cached(("conflict_validation", p.stat().st_mtime), build)


# ---------------------------------------------------------------- densidad de eventos (F13)
def _density_guard():
    from world4.data import atlas, conflict_grid
    if not atlas.verify()["ok"]:
        raise DataMissing("the density regressions need the Atlas", "python world4_cli.py --install-atlas")
    if not conflict_grid.available():
        raise DataMissing("the density regressions need UCDP GED", "python world4_cli.py --install-conflict-grid")


def density(origin, h, cells):
    from world4 import api
    _density_guard()
    try:
        return api.event_density(origin, h, cells=cells)
    except api.QueryError as exc:
        raise ValueError(str(exc)) from exc


def density_scenario(origin, members):
    from world4 import api
    _density_guard()
    return api.event_density_scenario(origin, members=members)


def density_study() -> dict:
    from world4.conflict import density as D
    if not D.STUDY.is_file():
        raise DataMissing("no factor study saved", "python world4_cli.py --study-factors")

    def build():
        r = json.loads(D.STUDY.read_text(encoding="utf-8"))
        out = {"meta": {k: v for k, v in r["meta"].items() if k != "factors"}, "labels": {}, "horizons": {}}
        for g, fs in r["meta"]["factors"].items():
            for f, lab in fs.items():
                out["labels"][f] = {"label": lab, "group": g}
        for k, v in r.items():
            if not k.startswith("h"):
                continue
            out["horizons"][k] = {
                "historia": v["historia"], "null_skill": v["null"]["skill"], "todo_atlas": v["todo_atlas"]["skill"],
                "todo_atlas_mas_world3": v["todo_atlas_mas_world3"]["skill"],
                "groups": {g: x["skill"] for g, x in v["groups"].items()},
                "unique": v["unique_group_contribution"],
                "single": {f: {"skill": x["skill"], "improved": x["ls_improved_share"], "group": x["group"]}
                           for f, x in v["single"].items()},
                "forward": [{"add": p["add"], "skill": p["skill"], "aupr": p["aupr_any"]} for p in v["forward"]["path"]],
                "coefficients": v["coefficients"]}
        bt = D.STUDY.with_name("backtest.json")
        out["backtest"] = json.loads(bt.read_text(encoding="utf-8"))["summary"] if bt.is_file() else None
        return out

    return cached(("density_study", D.STUDY.stat().st_mtime), build)


# ---------------------------------------------------------------- HTTP
def _q(qs, name, cast=str, default=None):
    v = qs.get(name)
    if not v or v[0] == "":
        return default
    if cast is bool:
        return v[0] not in ("0", "false", "no")
    return cast(v[0])


def route(method: str, path: str, qs: dict, body: bytes | None):
    """(status, payload) donde payload es un objeto JSON o bytes ya en gzip."""
    if path == "/api/status":
        return 200, status()
    if path == "/api/sources":
        return 200, cached(("sources",), sources)
    if path == "/api/l0":
        s = _q(qs, "scenario", int, 1)
        return 200, cached(("l0", s), lambda: l0(s))
    if path == "/api/social":
        args = (_q(qs, "scenario", int, 1), min(_q(qs, "n", int, 40), 200), _q(qs, "feedback", bool, True),
                _q(qs, "seed", int, 1))
        return 200, cached(("social",) + args, lambda: social(*args))
    if path == "/api/catalog":
        return 200, cached(("catalog",), catalog)
    if path == "/api/l1":
        args = (_q(qs, "variable"), _q(qs, "region"), _q(qs, "as_of", int, 2009), _q(qs, "horizon", int, 10),
                min(_q(qs, "members", int, 32), 128))
        if not args[0] or not args[1]:
            raise ValueError("variable and region are required")
        return 200, cached(("l1",) + args, lambda: l1(*args))
    if path == "/api/levers":
        return 200, cached(("levers",), levers)
    if path == "/api/policy" and method == "POST":
        req = json.loads(body or b"{}")
        theta = {k: float(v) for k, v in (req.get("theta") or {}).items()}
        key = ("policy", tuple(sorted(theta.items())), req.get("narrative", "ssp2"), int(req.get("members", 100)))
        return 200, cached(key, lambda: policy(theta, key[2], key[3]))
    if path == "/api/l0r":
        args = (_q(qs, "posterior", bool, True), min(_q(qs, "members", int, 100), 400))
        return 200, cached(("l0r",) + args, lambda: l0r(*args))
    if path == "/api/conflict/meta":
        return 200, cached(("conflict_meta",), conflict_meta)
    if path == "/api/conflict/forecast":
        return 200, conflict_forecast(_q(qs, "origin"), _q(qs, "min_p", float, 0.01))
    if path == "/api/conflict/cell":
        gid = _q(qs, "gid", int)
        if gid is None:
            raise ValueError("gid is required")
        return 200, conflict_cell(gid, _q(qs, "origin"))
    if path == "/api/conflict/validation":
        return 200, conflict_validation()
    if path == "/api/density":
        args = (_q(qs, "origin", int, None), _q(qs, "h", int, 1), _q(qs, "cells", bool, False))
        return 200, cached(("density",) + args, lambda: density(*args))
    if path == "/api/density/scenario":
        args = (_q(qs, "origin", int, 2019), min(_q(qs, "members", int, 32), 128))
        return 200, cached(("density_scenario",) + args, lambda: density_scenario(*args))
    if path == "/api/density/study":
        return 200, density_study()
    return 404, {"error": f"no route {path}"}


class Handler(SimpleHTTPRequestHandler):
    server_version = "World4UI/1"

    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(UI_DIST), **kw)

    def log_message(self, fmt, *args):  # una línea corta por petición de la API
        if self.path.startswith("/api/"):
            print(f"  {self.command} {self.path[:120]} -> {args[1] if len(args) > 1 else ''}", flush=True)

    def _send(self, code: int, payload) -> None:
        if isinstance(payload, bytes):
            data, enc = payload, "gzip"
        else:
            raw = json.dumps(_clean(payload), ensure_ascii=False, default=_jsonable).encode("utf-8")
            data, enc = (gzip.compress(raw, 5), "gzip") if len(raw) > 2048 else (raw, None)
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        if enc:
            self.send_header("Content-Encoding", enc)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _api(self, method: str) -> None:
        u = urlparse(self.path)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) if method == "POST" else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                code, payload = route(method, u.path, parse_qs(u.query), body)
        except DataMissing as exc:
            code, payload = 503, {"error": str(exc), "remedy": exc.remedy}
        except (ValueError, KeyError) as exc:
            code, payload = 400, {"error": str(exc)}
        except Exception as exc:  # noqa: BLE001 (un fallo del modelo no tumba el servidor)
            from world4.data.atlas import AtlasDataError
            names = ("DataError", "DataMissing")
            if isinstance(exc, AtlasDataError) or type(exc).__name__.endswith(names):
                code, payload = 503, {"error": str(exc), "remedy": _remedy_from(str(exc))}
            else:
                traceback.print_exc()
                code, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        self._send(code, payload)

    def do_GET(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._api("GET")
        if not (UI_DIST / "index.html").is_file():
            return self._send(503, {"error": "the interface is not built",
                                    "remedy": "cd ui && npm install && npm run build"})
        p = urlparse(self.path).path
        if p != "/" and not (UI_DIST / p.lstrip("/")).is_file():
            self.path = "/"                                   # rutas de la SPA -> index.html
        return super().do_GET()

    def do_POST(self):  # noqa: N802
        if self.path.startswith("/api/"):
            return self._api("POST")
        self._send(404, {"error": "not found"})


def _remedy_from(msg: str) -> str:
    import re
    m = re.search(r"python world4_cli\.py [^\s:)]+(?: \[[^\]]*\])?", msg)
    return m.group(0) if m else ""


def _clean(o):
    """NaN/inf -> null (JSON no los admite)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, np.integer)):
        return _clean(o.item())
    if hasattr(o, "__dataclass_fields__"):
        return _clean(_jsonable(o))
    return o


def serve(port: int = 8765, host: str = "127.0.0.1", log=print) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    log(f"World4 interface on http://{host}:{port}/  (API at /api/status; Ctrl+C stops)")
    if not (UI_DIST / "index.html").is_file():
        log("the React interface is not built yet: cd ui && npm install && npm run build")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
