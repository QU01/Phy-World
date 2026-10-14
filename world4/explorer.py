"""F10: los datos del explorador de World4 en Quasar Studio (globo 3D, grafo de nodos y panel de preguntas).

Tres cargas JSON, todas con unidad y banda (o una nota que dice por qué no la llevan), ``exploratory`` y procedencia:

* ``world_payload``: **los países como agentes.** El L1 F1-F5 desde el origen vivo (2019) a 10 años. Por país y año:
  el estado (PIB per cápita, población, posición en el sistema-mundo, asabiya, deuda, democracia, riesgo de
  conflicto, crisis y golpe, drenaje por intercambio desigual, rentas netas, CO2, energía, calorías) y las
  **decisiones** del agente: el reparto de su renta entre los seis canales de la regla de conducta del Atlas
  (capital, tecnología propia, importar del centro, extraer recursos propios, invertir fuera, redistribuir).
  Encima, los **flujos** entre países: el comercio (cuotas COW de exportación x apertura x PIB) y el drenaje del
  intercambio desigual, repartido entre socios con la MISMA regla del L1 (``theta_d X S_exp gap``: el L1 registra el
  total por país, no la matriz). Y el L0 regional a 2100 con la política y la narrativa de la corrida.
* ``geo_payload``: la geometría Natural Earth 1:110m del Atlas (verificada por SHA-256), con centroides.
* ``graph_payload``: **el grafo de la corrida**: de los datos (con su SHA-256 y licencia) a los sectores, los agentes,
  los mercados, el L0, las palancas, la búsqueda y los objetivos, y de ahí a Prime Radiant y QUO; cada nodo con los
  números de ESTA corrida. Más el grafo de lazos del L0 con su dominancia por década (*Loops that Matter*).

Lo que NO es: un pronóstico más allá de 10 años (el L0 a 2100 es un modo de comportamiento) ni las bandas
recalibradas de W8 (éstas son las del ensemble crudo del L1; las recalibradas viven en ``api.query``).
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

SCHEMA = "world4-explorer-1"
HORIZON = 10
TOP_TRADE, TOP_DRAIN, TOP_PARTNERS = 160, 120, 3

CHANNEL_LABELS = {"k": "domestic capital", "r": "own technology and industry", "m": "imports from the core",
                  "x": "own resource extraction", "f": "resource investment abroad", "w": "redistribution"}

# variable pública -> (clave del L1, unidad, transformación, etiqueta, tipo)
#   "q": cuantiles p10/p50/p90 del ensemble; "freq": frecuencia de un indicador 0/1 entre miembros
L1_SERIES = {
    "gdp_pc": ("ly", "2017 USD per person (PWT rgdpna/pop)", np.exp, "GDP per person", "q"),
    "population": ("pop", "million persons", None, "Population", "q"),
    "coreness": ("core", "index 0-1 (world-system position: sigmoid of relative productivity)", None,
                 "Coreness", "q"),
    "asabiya": ("asab", "collective solidarity S (index 0-1)", None, "Asabiya", "q"),
    "debt_gdp": ("debt", "public debt / GDP (fraction)", None, "Public debt", "q"),
    "democracy": ("demo", "probability of a democratic regime (V-Dem RoW >= 2)", None, "Democracy", "q"),
    "conflict_hazard": ("hazard", "annual probability of civil conflict (>= 25 battle deaths)", None,
                        "Conflict hazard", "q"),
    "drain": ("drain", "unequal-exchange loss / GDP (fraction)", None, "Value drain", "q"),
    "net_rents": ("rent_in", "net rents received / GDP (fraction; negative = paid abroad)", None, "Net rents", "q"),
    "co2": ("co2", "MtCO2 per year, fossil and industrial", None, "CO2 emissions", "q"),
    "resource_rents": ("res_share", "resource rents / GDP (fraction)", None, "Resource rents", "q"),
    "primary_energy": ("energy_PE", "TWh of primary energy per year", None, "Primary energy", "q"),
    "food_supply": ("food_kcal", "kcal per person per day", None, "Food supply", "q"),
    "crisis": ("crisis", "share of ensemble members with a financial crisis in the year", None, "Financial crisis",
               "freq"),
    "coup": ("coup", "share of ensemble members with a successful coup in the year", None, "Coup", "freq"),
    "conflict": ("conflict", "share of ensemble members with civil conflict in the year", None, "Civil conflict",
                 "freq"),
}

L1_CAVEATS = (
    "raw L1 ensemble p10/p50/p90: not the W8-recalibrated bands (those live in api.query and the structural estimator)",
    "origin 2019: no data after 2019 is assimilated (the Atlas panel ends in 2019)",
    "agents' decisions after 2019 come from the behaviour rule cloned from 1950-2019 (BehaviorModel); the levers of "
    "the run act on the L0, not on the agents (declared: the L1 has no policy levers in v0.1)",
    "trade flows: COW export shares held at their last year (2014) x simulated GDP x openness [S]",
    "value-drain flows: the L1 records each country's total loss; the split among partners repeats the L1 rule "
    "(theta_d X S_exp gap) on ensemble medians, with GDP per person standing in for output per person [S]",
)


def _r(x, sig: int = 4):
    """Redondeo a ``sig`` cifras significativas; NaN/inf -> None (JSON)."""
    a = np.asarray(x, float)
    with np.errstate(all="ignore"):
        mag = np.where((a != 0) & np.isfinite(a), np.floor(np.log10(np.abs(a))), 0)
        f = 10.0 ** (sig - 1 - mag)
        out = np.round(a * f) / f
    if out.ndim == 0:
        v = float(out)
        return v if math.isfinite(v) else None
    return _nest(out)


def _nest(a):
    if a.ndim == 1:
        return [None if not math.isfinite(v) else float(v) for v in a]
    return [_nest(x) for x in a]


# ----------------------------------------------------------------------------------------------- geometría
def geo_payload() -> dict:
    """GeoJSON de los países (Natural Earth 1:110m del Atlas, verificada por SHA-256) con centroides."""
    from world4 import dashboard
    from world4.core.l1_atlas.geo import load_countries
    fc = load_countries(dashboard.install_geo())
    man = json.loads(dashboard.MANIFEST.read_text(encoding="utf-8"))
    for f in fc["features"]:
        f["properties"]["centroid"] = _centroid(f["geometry"])
    return {"type": "FeatureCollection", "features": fc["features"], "source": man.get("source"),
            "sha256": man.get("sha256"), "note": "Natural Earth 1:110m via world-atlas, from the Atlas repo"}


def _centroid(geom) -> list:
    """Centroide aproximado del polígono más grande (media de vértices en la esfera: cruza bien el antimeridiano)."""
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    best, area = None, -1.0
    for poly in polys:
        ring = np.asarray(poly[0], float)
        if len(ring) < 3:
            continue
        x, y = ring[:, 0], ring[:, 1]
        a = abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))) / 2
        if a > area:
            best, area = ring, a
    if best is None:
        return [0.0, 0.0]
    lon, lat = np.radians(best[:, 0]), np.radians(best[:, 1])
    v = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], 1).mean(0)
    return [round(float(np.degrees(np.arctan2(v[1], v[0]))), 2),
            round(float(np.degrees(np.arctan2(v[2], np.hypot(v[0], v[1])))), 2)]


# ------------------------------------------------------------------------------------------------- el L1
def _l1(R: int, seed: int, log) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.structural import engine
    W = engine.live_world()
    t0 = engine.L1_LAST_DATA_YEAR - 1950
    from world4.data import climate as cdata
    from world4.data import food as fdata
    f5 = coupling.F5Config() if (cdata.available() and fdata.available()) else None
    f4 = coupling.F4Config() if cdata.available() else None
    log(f"explorer: L1 F1-F{5 if f5 else (4 if f4 else 3)} from {1950 + t0}, {R} members, {HORIZON} years")
    t = time.time()
    f = coupling.forecast(t0, R=R, seed=seed, world=W, horizon=HORIZON, f4=f4, f5=f5, raw=True)
    log(f"explorer: L1 done in {time.time() - t:.1f} s")
    return {"f": f, "W": W, "t0": t0, "sectors": "F1-F5" if f5 else ("F1-F4" if f4 else "F1-F3")}


def _get(f: dict, key: str):
    if key in f and isinstance(f[key], np.ndarray) and f[key].ndim == 3:
        return f[key]
    return f["raw"].get(key)


def _flows(f: dict, W, t0: int, keep: np.ndarray) -> dict:
    """Comercio y drenaje bilaterales por año (top-K), en millones de USD de 2017 por año."""
    from world4.core.l1_atlas import validate
    params, _ = validate.params_for(1950 + t0)
    te = min(t0, W.T - 1)                                     # la red de comercio del L1 en el pronóstico
    S = np.asarray(W.S_exp[te], float) if getattr(W, "S_exp", None) is not None else None
    ly = np.nanmedian(f["ly"], 0)                             # (N, T)
    pop = np.nanmedian(f["pop"], 0)
    core = np.nanmedian(f["raw"]["core"], 0)
    drain = np.nanmedian(f["raw"]["drain"], 0)
    gdp = np.exp(ly) * pop                                    # millones de USD de 2017 (pop en millones)
    o = np.nan_to_num(np.asarray(W.open_[:, min(te, W.open_.shape[1] - 1)], float))
    T = ly.shape[1]
    trade, drain_out = [], []
    nk = int(keep.sum())
    partners = {k: [[[] for _ in range(T)] for _ in range(nk)] for k in ("trade_out", "trade_in", "drain_out", "drain_in")}
    if S is None:
        return {"trade": [], "drain": [], "partners": partners, "unit": "million 2017 USD per year",
                "note": "the Atlas world has no bilateral trade network (S_exp absent)"}
    pos = np.cumsum(keep) - 1                                  # índice completo -> índice en ``countries``
    for k in range(T):
        X = np.nan_to_num(o * gdp[:, k])
        M = X[:, None] * S
        M[~keep, :] = 0
        M[:, ~keep] = 0
        flat = np.argsort(M, axis=None)[::-1][:TOP_TRADE]
        trade.append([[int(pos[i]), int(pos[j]), _r(M[i, j])] for i, j in zip(*np.unravel_index(flat, M.shape))
                      if M[i, j] > 0])
        ypc = np.exp(ly[:, k])
        with np.errstate(all="ignore"):
            ratio = np.nan_to_num((ypc[None, :] / ypc[:, None]) ** params.beta_px - 1)
        wgt = S * np.maximum(ratio, 0) * np.nan_to_num(core[None, :, k]) * np.nan_to_num(1 - core[:, None, k])
        wgt[~keep, :] = 0
        wgt[:, ~keep] = 0
        tot = wgt.sum(1, keepdims=True)
        D = np.nan_to_num(drain[:, k] * gdp[:, k])[:, None] * np.divide(wgt, tot, out=np.zeros_like(wgt),
                                                                        where=tot > 0)
        flat = np.argsort(D, axis=None)[::-1][:TOP_DRAIN]
        drain_out.append([[int(pos[i]), int(pos[j]), _r(D[i, j])] for i, j in zip(*np.unravel_index(flat, D.shape))
                          if D[i, j] > 0])
        for name, A in (("trade", M), ("drain", D)):
            for direction, B in (("out", A), ("in", A.T)):
                top = np.argsort(B, axis=1)[:, ::-1][:, :TOP_PARTNERS]
                for i in np.where(keep)[0]:
                    partners[f"{name}_{direction}"][pos[i]][k] = [[int(pos[j]), _r(B[i, j])] for j in top[i] if B[i, j] > 0]
    return {"trade": trade, "drain": drain_out, "partners": partners,
            "unit": "million 2017 USD per year",
            "row_format": "trade/drain: per year [origin index, destination index, value]; partners: per country and "
                          f"year the {TOP_PARTNERS} largest [partner index, value] in each direction",
            "band": "none: flows on ensemble medians (the L1 does not record the bilateral matrix per member)",
            "note": "trade = openness x GDP x COW export share (2014 network held); drain = unequal-exchange loss "
                    "split among partners with the L1 rule theta_d X S_exp gap"}


def _series(f: dict, keep: np.ndarray) -> dict:
    out = {}
    for name, (key, unit, tr, label, kind) in L1_SERIES.items():
        a = _get(f, key)
        if a is None:
            continue
        a = np.asarray(a, float)[:, keep, :]
        if tr is not None:
            with np.errstate(all="ignore"):
                a = tr(a)
        if kind == "freq":
            out[name] = {"label": label, "unit": unit, "mean": _r(np.nanmean(a, 0), 3),
                         "band": "none: a frequency over the R members; its sampling error is about "
                                 "+-1/sqrt(R)"}
        else:
            q = np.nanpercentile(a, [10, 50, 90], axis=0)
            out[name] = {"label": label, "unit": unit, "p10": _r(q[0]), "p50": _r(q[1]), "p90": _r(q[2])}
    return out


def _channels(f: dict, keep: np.ndarray) -> dict:
    from world4.core.l1_atlas.model import CHANNELS
    a = np.stack([np.nanmedian(np.asarray(f["raw"][c], float)[:, keep, :], 0) for c in CHANNELS], -1)
    return {"names": list(CHANNELS), "labels": CHANNEL_LABELS, "unit": "share of national income (fraction)",
            "share": _r(a, 3),
            "band": "none: ensemble median of each channel (shares of a member need not add up across medians)",
            "note": "years <= origin are the observed spending; after it, the behaviour rule cloned 1950-origin"}


WORLD_SERIES = {"Tg": ("gmst", "degC above 1850-1900"),
                "psi": ("resource_rent_share", "world resource rents / world GDP (fraction)"),
                "price_oil": ("oil_price", "real USD 2019 per barrel")}


def _world(f: dict) -> dict:
    out = {}
    for key, (name, unit) in WORLD_SERIES.items():
        if key in f:
            q = np.nanpercentile(np.asarray(f[key], float), [10, 50, 90], axis=0)
            out[name] = {"unit": unit, "p10": _r(q[0]), "p50": _r(q[1]), "p90": _r(q[2])}
    return out


def _l0_regions(theta, narrative: str, members: int, seed: int) -> dict:
    """L0 regional 2019-2100 con la política y la narrativa: bandas por región y del mundo, y la base sin política."""
    from world4.core.l0r import regions
    from world4.scenarios import narratives, robust
    mult = narratives.sow(members, narrative, seed)
    th = np.asarray(theta, float)
    rec = ("P", "Y", "Tg", "cpc", "debt", "nfs", "E")
    r = robust.run(th, mult, record=rec)
    r0 = robust.run(np.zeros_like(th), mult, record=rec)
    years = list(range(robust.START, robust.START + r["P"].shape[0]))

    def band(x, sig=4):
        q = np.nanpercentile(x, [10, 50, 90], axis=1)        # (3, years, ...)
        return {"p10": _r(q[0], sig), "p50": _r(q[1], sig), "p90": _r(q[2], sig)}

    with np.errstate(all="ignore"):
        gpc = r["Y"] / r["P"]
        cpc = r["cpc"] / np.maximum(r["cpc"][0:1], 1e-12)
        world = {
            "gmst": {"unit": "degC above 1850-1900", **band(r["Tg"][:, :, 0], 3),
                     "baseline_p50": _r(np.nanmedian(r0["Tg"][:, :, 0], 1), 3)},
            "population": {"unit": "million persons", **band(r["P"].sum(2)),
                           "baseline_p50": _r(np.nanmedian(r0["P"].sum(2), 1))},
            "gdp_pc": {"unit": "2017 USD PPP per person", **band(r["Y"].sum(2) / r["P"].sum(2)),
                       "baseline_p50": _r(np.nanmedian(r0["Y"].sum(2) / r0["P"].sum(2), 1))},
            "primary_energy": {"unit": "TWh of primary energy per year", **band(r["E"].sum(2)),
                               "baseline_p50": _r(np.nanmedian(r0["E"].sum(2), 1))},
            "nonfossil_share": {"unit": "non-fossil share of primary energy (fraction)",
                                **band((r["nfs"] * r["E"]).sum(2) / r["E"].sum(2), 3),
                                "baseline_p50": _r(np.nanmedian((r0["nfs"] * r0["E"]).sum(2) / r0["E"].sum(2), 1), 3)},
        }
        reg = {}
        for j, c in enumerate(regions.CODES):
            reg[c] = {"name": regions.NAMES[c],
                      "gdp_pc": {"unit": "2017 USD PPP per person", **band(gpc[:, :, j])},
                      "population": {"unit": "million persons", **band(r["P"][:, :, j])},
                      "debt_gdp": {"unit": "public debt / GDP (fraction)", **band(r["debt"][:, :, j], 3)},
                      "cereal_pc": {"unit": "cereal per person, index 2019 = 1", **band(cpc[:, :, j], 3)},
                      "nonfossil_share": {"unit": "non-fossil share of primary energy (fraction)",
                                          **band(r["nfs"][:, :, j], 3)}}
    return {"years": years, "world": world, "regions": reg, "members": members, "narrative": narrative,
            "band": "p10-p90 over the states of the world (F7 posterior x deep uncertainty x lever efficacy); "
                    "the declared discrepancy of the L0 is NOT added here",
            "exploratory": True, "fidelity": "L0"}


def world_payload(*, theta=None, narrative: str = "ssp2", members: int = 200, R: int = 32, seed: int = 0,
                  with_l1: bool = True, log=print) -> dict:
    """Los agentes, los flujos y el L0 regional de una corrida (ver el docstring del módulo)."""
    from world4 import MODEL_REVISION
    from world4.core.l0r import regions
    from world4.scenarios import levers
    th = np.zeros(len(levers.NAMES)) if theta is None else np.asarray(theta, float)
    levers.to_policy(th)
    out = {"schema": SCHEMA, "model_revision": MODEL_REVISION, "narrative": narrative,
           "theta": dict(zip(levers.NAMES, th.tolist())), "exploratory": True}
    log(f"explorer: L0 regional 2019-2100, narrative {narrative}, {members} states of the world")
    out["l0"] = _l0_regions(th, narrative, members, seed)
    from world4.core.l1_atlas import calibrate
    W0 = calibrate.W
    r0 = regions.region_index(W0.iso3, W0.isonum)
    out["l0"]["countries"] = [{"iso3": str(a), "isonum": int(b), "region": regions.CODES[k]}
                              for a, b, k in zip(W0.iso3, W0.isonum, r0) if k >= 0]
    if not with_l1:
        out["l1"] = None
        out["l1_note"] = "fidelity L0: the country agents (L1) were not run; the globe colours each country by its L0 region"
        return out
    from world4.core.l0r import regions as rg
    l1 = _l1(R, seed, log)
    f, W, t0 = l1["f"], l1["W"], l1["t0"]
    keep = np.isfinite(np.asarray(f["ly"], float)[:, :, 0]).any(0)
    iso3 = [str(s) for s in np.asarray(W.iso3)[keep]]
    isonum = [int(v) for v in np.asarray(W.isonum)[keep]]
    ridx = rg.region_index(W.iso3, W.isonum)[keep]
    T = f["ly"].shape[2]
    out["l1"] = {
        "origin": 1950 + t0, "years": list(range(1950 + t0, 1950 + t0 + T)), "members": R, "seed": seed,
        "sectors": l1["sectors"],
        "countries": [{"iso3": a, "isonum": b, "region": regions.CODES[k] if k >= 0 else None}
                      for a, b, k in zip(iso3, isonum, ridx)],
        "series": _series(f, keep), "channels": _channels(f, keep),
        "flows": _flows(f, W, t0, keep), "world": _world(f), "caveats": list(L1_CAVEATS),
        "exploratory": False,
        "horizon_note": "1-10 years from the origin: inside the horizon measured out of sample (Atlas rolling "
                        "origins); the bands are raw, not recalibrated",
    }
    return out


# ---------------------------------------------------------------------------------------------- el grafo
def _data_nodes() -> list:
    from world4.data import atlas
    from world4 import dashboard
    from pathlib import Path
    base = Path(dashboard.MANIFEST).parent
    src = json.loads((base / "sources.json").read_text(encoding="utf-8"))["sources"]

    def man(name):
        try:
            return json.loads((base / f"{name}_manifest.json").read_text(encoding="utf-8"))
        except OSError:
            return {}

    def files(m):
        fs = m.get("files")
        return len(fs) if isinstance(fs, (list, dict)) else (1 if m.get("sha256") else 0)

    def avail(mod):
        try:
            return bool(mod.available())
        except Exception:                                         # noqa: BLE001 (un dato ausente no tumba el grafo)
            return False

    from world4.data import climate, energy, food, wpp
    rows = [
        ("data_atlas", "Atlas panel 1950-2019", f"QU01/Reproducible-Research-Project-2 @ {atlas.ATLAS_COMMIT}",
         files(man("atlas")), True, ["pwt", "wdi", "ucdp", "vdem"]),
        ("data_wpp", "UN WPP 2024 (ages and sexes)", "population by single age and sex", files(man("wpp")),
         avail(wpp), ["wpp"]),
        ("data_energy", "Energy Institute / OWID", "primary energy mix by source", files(man("energy")),
         avail(energy), ["ei"]),
        ("data_climate", "FaIR 1.6.0 + observed CO2", "calibrated, constrained FaIR ensemble",
         files(man("climate")), avail(climate), ["fair"]),
        ("data_food", "FAOSTAT", "cereals, land, N, irrigation, food supply", files(man("food")), avail(food),
         ["faostat"]),
    ]
    out = []
    for nid, label, what, nfiles, ok, keys in rows:
        lic = sorted({src[k]["license"] for k in keys if k in src}) or ["see sources.json"]
        out.append({"id": nid, "kind": "data", "label": label, "status": "ok" if ok else "missing",
                    "description": what,
                    "metrics": [{"label": "files with SHA-256", "value": nfiles, "unit": "files"}],
                    "provenance": {"licences": lic, "manifest": f"world4/data/{nid[5:]}_manifest.json"}})
    return out


def _med_world(series: dict, key: str, agg: str = "sum"):
    s = series.get(key)
    if not s:
        return None
    a = np.asarray([[np.nan if v is None else v for v in row] for row in s.get("p50") or s.get("mean")], float)
    return np.nansum(a, 0) if agg == "sum" else np.nanmean(a, 0)


def graph_payload(world: dict, policy: dict | None = None, search: dict | None = None,
                  loops: dict | None = None) -> dict:
    """El grafo de información de la corrida (nodos con los números de ESTA corrida) y el de lazos del L0."""
    nodes, edges = _data_nodes(), []
    l1, l0 = world.get("l1"), world["l0"]
    years = (l1 or {}).get("years") or []

    def node(nid, kind, label, desc, metrics=(), status="ok", series=None, caveats=(), provenance=None):
        nodes.append({"id": nid, "kind": kind, "label": label, "description": desc, "status": status,
                      "metrics": list(metrics), "series": series, "caveats": list(caveats),
                      "provenance": provenance or {}})

    def m(label, value, unit):
        return {"label": label, "value": None if value is None or not np.isfinite(value) else round(float(value), 4),
                "unit": unit}

    has_l1 = l1 is not None
    sec = (l1 or {}).get("sectors", "")
    ser = (l1 or {}).get("series", {})
    pop = _med_world(ser, "population") if has_l1 else None
    co2 = _med_world(ser, "co2") if has_l1 else None
    en = _med_world(ser, "primary_energy") if has_l1 else None
    conf = _med_world(ser, "conflict") if has_l1 else None
    l1_status = "ok" if has_l1 else "skipped"

    def spark(arr, unit, label):
        if arr is None:
            return None
        return {"years": years, "values": _r(arr), "unit": unit, "label": label,
                "band": "none: world sum of country medians (a summary line, not an estimate)"}

    node("sec_population", "sector", "F1 population and human capital",
         "cohorts by single age and sex (WPP 2024) and schooling by cohort, coupled to the L1",
         [m("world population, end", pop[-1] if pop is not None else None, "million persons")], l1_status,
         spark(pop, "million persons", "world population"))
    node("sec_resources", "sector", "F2 world resource markets",
         "oil, gas, coal and metals markets with depletion: the world resource-rent share psi_t",
         [], l1_status)
    node("sec_energy", "sector", "F3 energy with EROI",
         "primary-energy mix by country (fossil, nuclear, hydro, wind + solar, other), EROI and net energy",
         [m("world primary energy, end", en[-1] if en is not None else None, "TWh per year")],
         l1_status if "F3" in sec or "F4" in sec or "F5" in sec else "skipped",
         spark(en, "TWh per year", "world primary energy"))
    node("sec_climate", "sector", "F4 climate (FaIR v2.2.4)",
         "FaIR with the calibrated, constrained ensemble 1.6.0; emissions from F1-F3 and F5",
         [m("world fossil CO2, end", co2[-1] if co2 is not None else None, "MtCO2 per year")],
         l1_status if ("F4" in sec or "F5" in sec) else "skipped", spark(co2, "MtCO2 per year", "world fossil CO2"))
    node("sec_food", "sector", "F5 food, land, N and water",
         "cereals, cropland, pasture, forest, mineral N, irrigation and calories per country",
         [], l1_status if "F5" in sec else "skipped")
    nch = len((l1 or {}).get("countries") or [])
    node("agents", "agents", f"L1 country agents ({nch or 180})",
         "the Atlas ABM: each country splits its income among six channels with the behaviour rule cloned "
         "from 1950-2019; world-system, structural-demographic and metaethnic-frontier layers",
         [m("countries simulated", nch, "countries"),
          m("expected countries in civil conflict, end", conf[-1] if conf is not None else None, "countries"),
          m("ensemble members", (l1 or {}).get("members"), "members")], l1_status,
         spark(conf, "countries (expected)", "countries in civil conflict"),
         list((l1 or {}).get("caveats") or []))
    node("flows", "flows", "World-system flows",
         "bilateral trade (COW shares) and the unequal-exchange drain from periphery to core",
         [m("trade links shown per year", TOP_TRADE, "links"), m("drain links shown per year", TOP_DRAIN, "links")],
         l1_status)
    wg = l0["world"]
    node("l0", "model", "L0 regional (10 regions, 112 stocks)",
         "the same structure as the L1 aggregated to 10 IAM-style regions; F7 posterior (cut-off 2019)",
         [m("GMST 2100, p50", (wg["gmst"]["p50"] or [None])[-1], "degC above 1850-1900"),
          m("GMST 2100, p90", (wg["gmst"]["p90"] or [None])[-1], "degC above 1850-1900"),
          m("world population 2100, p50", (wg["population"]["p50"] or [None])[-1], "million persons"),
          m("states of the world", l0.get("members"), "members")], "ok",
         {"years": l0["years"], "values": wg["gmst"]["p50"], "unit": "degC above 1850-1900", "label": "GMST p50",
          "band": "p10-p90 in the L0 payload"},
         ["a behaviour mode under deep uncertainty to 2100, not a forecast"])
    th = world.get("theta") or {}
    node("levers", "policy", "Policy levers (theta)",
         f"narrative {world.get('narrative')}; the levers act on the L0 from 2019 with a ramp; effects are [M]",
         [m(k, v, "lever units (see w4_levers)") for k, v in th.items()], "ok",
         provenance={"narrative": world.get("narrative")})
    if search:
        node("search", "search", "Robust policy search (phycore.Designer)",
             "NSGA-II over the L0 ensemble; limits as constraints (MORDM pattern)",
             [m("L0 ensemble evaluations", search.get("evaluations"), "evaluations"),
              m("front size", search.get("front_size"), "policies"),
              m("feasible on the front", search.get("feasible_on_front"), "policies"),
              m("wall clock", search.get("seconds"), "s")], "ok")
    if policy:
        units = policy.get("units") or {}
        node("objectives", "objectives", "Robust objectives",
             "minimised over the ensemble: minus p10 welfare, p90 peak GMST, minus p10 of the poorest region",
             [m(k, v, units.get(k, "")) for k, v in (policy.get("objectives") or {}).items()], "ok")
        walls = [k for k, v in (policy.get("constraints") or {}).items() if v is not None and v > 0]
        node("limits", "constraints", "Declared limits",
             "food floor, land cap, debt cap (g <= 0 is inside)",
             [m(k, v, units.get(k, "")) for k, v in (policy.get("constraints") or {}).items()],
             "violated" if walls else "ok", caveats=[f"{w} violated (g > 0)" for w in walls])
    node("prime_radiant", "consumer", "Prime Radiant (structural stage)",
         "World4's L1 answers structural questions (conflict, coup, crisis, democracy, growth, debt) in the d1 "
         "forecast, with the as-of cut respected", [], "ok")
    node("quo", "consumer", "QUO", "reads runs with phy_run_status and asks World4 with world4_read", [], "ok")

    def e(a, b, label, kind="data"):
        if any(n["id"] == a for n in nodes) and any(n["id"] == b for n in nodes):
            edges.append({"id": f"{a}->{b}", "source": a, "target": b, "label": label, "kind": kind})

    e("data_atlas", "agents", "panel, behaviour rule, calibration cuts")
    e("data_wpp", "sec_population", "age rates at the origin")
    e("data_energy", "sec_energy", "mix and intensities")
    e("data_climate", "sec_climate", "FaIR configs, observed CO2")
    e("data_food", "sec_food", "FAOSTAT equations <= origin")
    e("sec_population", "agents", "population, human capital", "coupling")
    e("sec_resources", "agents", "psi_t (resource-rent share)", "coupling")
    e("agents", "sec_resources", "extraction and demand", "coupling")
    e("sec_energy", "sec_resources", "fossil demand", "coupling")
    e("sec_energy", "sec_climate", "fossil CO2", "coupling")
    e("sec_food", "sec_climate", "N2O, CH4, land-use CO2", "coupling")
    e("sec_climate", "agents", "GMST -> damages, conflict", "coupling")
    e("sec_climate", "sec_food", "temperature -> yields", "coupling")
    e("sec_food", "sec_population", "calorie shortfall -> e0", "coupling")
    e("agents", "flows", "trade, drain, FDI", "coupling")
    e("flows", "agents", "rents in / out", "coupling")
    e("data_atlas", "l0", "regional aggregates 1992-2019")
    e("agents", "l0", "same structure, aggregated", "structure")
    e("levers", "l0", "policy from 2019", "policy")
    e("l0", "objectives", "ensemble metrics", "evaluation")
    e("l0", "limits", "ensemble metrics", "evaluation")
    e("objectives", "search", "f", "evaluation")
    e("limits", "search", "g", "evaluation")
    e("search", "levers", "next candidates", "policy")
    e("agents", "prime_radiant", "structural estimate (d1 block)", "consumer")
    e("l0", "quo", "run summary", "consumer")
    e("agents", "quo", "country trajectories", "consumer")
    out = {"schema": SCHEMA, "nodes": nodes, "edges": edges, "exploratory": True,
           "units": {"metrics": "each metric carries its own unit"},
           "band": "node metrics are medians or declared summaries; bands live in the world payload"}
    if loops:
        out["loops"] = loops
    return out


def loops_payload(top: int = 4, step: int = 10) -> dict:
    """El grafo de stocks y flujos del L0 (vista mundial) con la dominancia de lazos por década, sin política."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l0r import loops, model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    m0 = model.L0R(model.calibrate(W, 2019), model.initial(W, 2019))
    run = m0.run(81, keep_states=True)
    res = loops.analyse(*loops.l0_world_view(m0, run))
    stocks = list(model.REG_STOCKS) + list(model.WORLD_STOCKS)
    links = set()
    for name in res["loops"]:
        parts = [p.strip() for p in name.split("->")]
        # "s1 -[f1]" ... -> s1
        seq = []
        for p in parts:
            if "-[" in p:
                s, f = p.split("-[")
                seq.append((s.strip(), f.rstrip("]").strip()))
        for k, (s, f) in enumerate(seq):
            tgt = seq[(k + 1) % len(seq)][0]
            links.add((s, tgt, f))
    table = loops.dominance_table(res, top=top, step=step)
    return {"stocks": stocks, "links": [{"source": a, "target": b, "flow": f} for a, b, f in sorted(links)],
            "loops": res["loops"], "dominant_by_decade": [
                {"year": r["year"], "loops": [{"loop": n, "score": s} for n, s in r["loops"]]} for r in table],
            "unit": "relative Loops-that-Matter score (dimensionless, signed: + reinforcing, - balancing)",
            "band": "none: one deterministic L0 run without policy (the score explains a trajectory)",
            "exploratory": True, "fidelity": "L0"}
