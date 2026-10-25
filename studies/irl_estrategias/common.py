"""Piezas compartidas del estudio «política revelada (IRL) y estrategias L1 x L0».

* L1 = el Atlas del Sistema-Mundo portado (180 países), con población y capital humano endógenos (F1),
  mercados de recursos (F2) y energía con EROI (F3). Clima: la TCRE del Atlas (los datos de FaIR y
  FAOSTAT no se pudieron descargar en este entorno, así que F4/F5 quedan apagados y se declara).
* L0 = World3-03 (``world4.core.l0``), corrido en ensemble sobre incertidumbre profunda.

El acoplamiento L1 -> L0 es «blando» y se declara [M]: cada estrategia se corre en el L1 de 2019 a 2030
con números aleatorios comunes, se miden cinco señales mundiales relativas a la línea base y esas señales
fijan, desde 2020, las constantes de política de World3 (ver ``world3_constants``).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")

OUT = Path(__file__).with_name("out")
OUT.mkdir(exist_ok=True)
FOCUS = ["USA", "CHN", "IND", "BRA", "MEX", "DEU", "RUS", "JPN", "NGA", "ZAF", "IDN", "SAU", "ARG", "COL",
         "EGY", "KOR", "TUR", "FRA", "GBR", "VEN"]
CHANNELS = ["k", "r", "m", "x", "f", "w"]
CH_ES = {"k": "Capital doméstico", "r": "Tecnología/I+D", "m": "Importar bienes del centro",
         "x": "Extracción de recursos", "f": "Inversión exterior", "w": "Redistribución"}
FEATS = ["bienestar", "poder", "elite", "paz", "autonomia", "inercia"]
ORIGIN, END = 2019, 2030
HORIZON = END - ORIGIN
GAMMA = 0.95
POLICY_YEAR = 2020.0


def dump(name, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        raise TypeError(type(o))
    (OUT / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=conv), encoding="utf-8")


def load(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- estrategias (L1)
# offsets: desplazamiento del logit de la regla clonada (el mismo instrumento que el IRL, ±0,5 = su región
# de confianza). calib: perillas F7 del L1 (por año), sin coste explícito en el L1; las estrategias
# «verdes» pagan un coste declarado [M] en el canal k (la inversión sale del capital doméstico).
STRATEGIES = {
    "base": dict(label="Línea base (regla revelada)", offsets={}, calib={},
                 what="lo que los países harían: la regla de conducta clonada de los datos"),
    "desarrollista": dict(label="Estado desarrollista", offsets={"k": 0.5, "r": 0.5}, calib={},
                          what="más inversión doméstica y más I+D/educación"),
    "conocimiento": dict(label="Economía del conocimiento", offsets={"r": 0.5, "m": 0.25}, calib={},
                         what="I+D y educación + importación de bienes con tecnología incorporada"),
    "bienestar_social": dict(label="Estado de bienestar", offsets={"w": 0.5}, calib={},
                             what="más gasto social y transferencias"),
    "extractivista": dict(label="Extractivismo", offsets={"x": 0.5}, calib={},
                          what="más extracción propia de recursos"),
    "austeridad": dict(label="Austeridad", offsets={"k": -0.25, "w": -0.5}, calib={},
                       what="recorte de gasto social e inversión pública"),
    "apertura": dict(label="Apertura comercial y financiera", offsets={"m": 0.5, "f": 0.5}, calib={},
                     what="más importaciones del centro y más inversión fuera"),
    "verde": dict(label="Transición verde", offsets={"r": 0.25, "k": -0.2},
                  calib={"intensity": -0.012, "nonfossil": 0.05},
                  what="eficiencia energética (-1,2 %/año) y difusión no fósil (+0,05 logit/año), pagada con capital"),
    "demografica": dict(label="Transición demográfica", offsets={"r": 0.25, "w": 0.25}, calib={"tfr": -0.01},
                        what="educación de niñas y planificación familiar (TFR -1 %/año) con gasto social"),
    "verde_desarrollista": dict(label="Desarrollo verde", offsets={"k": 0.3, "r": 0.5, "x": -0.25},
                                calib={"intensity": -0.012, "nonfossil": 0.05},
                                what="desarrollismo con eficiencia, no fósiles y menos extracción"),
}


def green_cost(calib: dict) -> float:
    """Coste declarado [M] de las perillas verdes y demográfica en logit de k."""
    return (-0.15 * abs(calib.get("intensity", 0.0)) / 0.012 - 0.15 * calib.get("nonfossil", 0.0) / 0.05
            - 0.05 * abs(calib.get("tfr", 0.0)) / 0.01)


_LIVE = {}


def live():
    if "W" not in _LIVE:
        from world4.structural import engine
        _LIVE["W"] = engine.live_world()
        _LIVE["t0"] = engine.L1_LAST_DATA_YEAR - 1950
    return _LIVE["W"], _LIVE["t0"]


def l1_run(offsets=None, calib=None, R=32, seed=0, mask=None):
    """Corre el L1 2019 -> 2030. ``offsets`` {canal: escalar o (N,)}; ``mask`` (N,) bool limita la estrategia
    a esos países (el resto sigue la regla). Devuelve el pronóstico con ``raw``."""
    from world4.core import coupling
    W, t0 = live()
    offs = None
    if offsets:
        offs = {}
        for ch, v in offsets.items():
            a = np.broadcast_to(np.asarray(v, float), (W.N,)).copy()
            if mask is not None:
                a = np.where(mask, a, 0.0)
            offs[ch] = np.broadcast_to(a[None], (R, W.N)).copy()
    if calib and mask is not None:
        raise ValueError("the L1 calibration knobs are world-wide: no mask")
    return coupling.forecast(t0, R=R, seed=seed, world=W, horizon=HORIZON, f3=coupling.F3Config(),
                             offsets=offs, calib=calib or None, raw=True)


def irl_theta():
    """θ revelado (en unidades de las features) estimado en ``s1_irl``; si no está, el del Atlas."""
    p = OUT / "irl.json"
    if p.exists():
        d = load("irl.json")["global"]
        th = np.array([d["theta_std"][k] for k in FEATS]) / np.array(d["sd"])
        return th / np.abs(th).max()
    from world4.scenarios import conduct
    return np.array([conduct.irl_results()["theta_unidades"][k] for k in FEATS])


def country_metrics(f):
    """Por miembro y país: (R, N) en 2030 y valores descontados 2019-2030."""
    from world4.core.l1_atlas import irl as airl
    W, t0 = live()
    raw = f["raw"]
    pop = f["pop"]                                     # (R, N, H+1) millones
    gdp = np.exp(f["ly"])
    phi = airl.reward_features(raw, W, t0)            # (R, N, H, F)
    val = airl.discounted(phi, GAMMA)                 # (R, N, F)
    fossil = f.get("energy_gen_fossil")
    return {"gdp_pc": gdp[:, :, -1], "cons_pc": raw["cons_pc"][:, :, -1], "hc": f["hc"][:, :, -1],
            "hazard": np.nanmean(f["hazard"][:, :, 1:], 2), "debt": f["debt"][:, :, -1], "pop": pop[:, :, -1],
            "fossil_twh": np.nansum(fossil[:, :, 1:], 2) if fossil is not None else None,
            "pe_twh": np.nansum(f["energy_PE"][:, :, 1:], 2), "val": val,
            "gdp_pc0": gdp[:, :, 0], "hc0": f["hc"][:, :, 0]}


def world_signals(f):
    """Señales mundiales (R,) que pasan al L0."""
    raw = f["raw"]
    pop = f["pop"]
    Y = np.exp(f["ly"]) * pop                         # PIB (millones de USD 2017)
    Yw = np.nansum(Y, 1)                              # (R, H+1)
    Cw = np.nansum(raw["cons_pc"] * pop, 1)
    Pw = np.nansum(pop, 1)
    fos = np.nansum(f["energy_gen_fossil"], 1)
    pe = np.nansum(f["energy_PE"], 1)
    k = np.nansum(raw["k"] * Y, 1) / Yw
    wsh = np.nansum(raw["w"] * Y, 1) / Yw
    x = np.nansum(raw["x"] * Y, 1) / Yw
    return {"Yw": Yw[:, -1], "CY": (Cw / Yw)[:, -1], "Pw": Pw[:, -1], "gY": np.log(Yw[:, -1] / Yw[:, 0]) / HORIZON,
            "fos_int": (fos / Yw)[:, -1], "fos_int0": (fos / Yw)[:, 0], "pe_int": (pe / Yw)[:, -1],
            "fossil_cum": np.nansum(fos[:, 1:], 1), "nonfossil_share": (1 - fos / pe)[:, -1],
            "k_share": np.nanmean(k[:, 1:], 1), "w_share": np.nanmean(wsh[:, 1:], 1), "x_share": np.nanmean(x[:, 1:], 1),
            "Tg": f["Tg"][:, -1], "hazard_w": np.nanmean(np.nansum(f["hazard"][:, :, 1:] * pop[:, :, 1:], 1)
                                                         / np.nansum(pop[:, :, 1:], 1), 1)}


# ---------------------------------------------------------------- L0: World3-03
STRATA = 8


def world3_ensemble(n=400, seed=0):
    """Estados del mundo de World3 (incertidumbre profunda [M]) filtrados por la historia: población
    2020 en [7,3; 8,3] mil millones (dato 7,8). Además de las constantes por miembro, cada miembro cae en
    uno de ``STRATA`` estratos (hipercubo latino) de progreso tecnológico AUTÓNOMO desde 2020 (recursos,
    contaminación, rendimientos agrícolas, %/año): las tablas de World3 no son vectoriales."""
    from world4.core import l0
    rng = np.random.default_rng(seed)
    m = int(n * 4)
    ens = {"nri": rng.uniform(1e12, 3e12, m),                 # recursos no renovables iniciales (LtG: 1-2e12)
           "ahl70": 1.5 * rng.uniform(0.6, 1.6, m),          # vida media de asimilación de contaminación
           "imef": 0.1 * rng.uniform(0.6, 1.4, m),           # contaminación por unidad de recurso
           "tdd": rng.uniform(10, 30, m),                    # retardo de adopción tecnológica
           "lfh": rng.uniform(0.6, 0.8, m),                  # fracción cosechada
           "dcfsn": rng.uniform(3.4, 4.2, m)}                # tamaño de familia deseado
    r = l0.run(constants=ens, n=m, t1=2021.0, record=("pop",))
    p2020 = r.series["pop"][np.searchsorted(r.time, 2020.0)]
    ok = np.where((p2020 > 7.3e9) & (p2020 < 8.3e9))[0][:n]
    ens = {k: v[ok] for k, v in ens.items()}
    u = (np.stack([rng.permutation(STRATA) for _ in range(3)], 1) + rng.uniform(size=(STRATA, 3))) / STRATA
    tech = {"R0": 0.02 * u[:, 0], "P0": 0.03 * u[:, 1], "Y0": 0.02 * u[:, 2]}
    ens["_stratum"] = np.arange(len(ok)) % STRATA
    return {"members": ens, "tech": tech, "n": len(ok), "drawn": m}


def world3_constants(sig, base):
    """Señales del L1 (estrategia y línea base) -> cambios de política de World3 desde ``POLICY_YEAR``,
    persistentes hasta 2100. [M] = supuesto del estudio, declarado:

    * fioac2 (fracción del producto industrial consumida) escala con C/Y de la estrategia / C/Y base;
    * tecnología de recursos (%/año, se suma a la autónoma) = mejora anual de la intensidad fósil del PIB
      frente a la base;
    * tecnología de contaminación (%/año, idem) = esa mejora más la ganancia anual de cuota no fósil;
    * isopc2 (servicios per cápita indicados) escala con el gasto social (w) de la estrategia / base;
    * icormrrct (capital por producto) escala con exp(-10 x diferencia de crecimiento no debida a k);
    * dcfsn (familia deseada) escala con (población 2030 estrategia / base)^4.
    """
    mean = lambda d, k: float(np.nanmean(d[k]))  # noqa: E731
    cy = mean(sig, "CY") / mean(base, "CY")
    d_fos = np.log(mean(base, "fos_int") / mean(sig, "fos_int")) / HORIZON
    d_nf = (mean(sig, "nonfossil_share") - mean(base, "nonfossil_share")) / HORIZON
    w_rel = mean(sig, "w_share") / max(mean(base, "w_share"), 1e-9)
    gk = 0.35 * (mean(sig, "k_share") - mean(base, "k_share")) / max(mean(base, "k_share"), 1e-9) / HORIZON
    d_g = (mean(sig, "gY") - mean(base, "gY")) - gk
    p_rel = mean(sig, "Pw") / mean(base, "Pw")
    return {"fioac2_factor": float(np.clip(cy, 0.7, 1.3)), "resource_tech_rate": float(np.clip(d_fos, -0.05, 0.05)),
            "pollution_tech_rate": float(np.clip(d_fos + d_nf, -0.05, 0.06)),
            "services_factor": float(np.clip(w_rel, 0.7, 1.5)), "icor_factor": float(np.clip(np.exp(-d_g * 10), 0.8, 1.2)),
            "family_factor": float(np.clip(p_rel ** 4, 0.7, 1.3))}


def world3_metrics(mp, ens):
    from world4.core import l0
    mem, tech = ens["members"], ens["tech"]
    keys = ("pop", "iopc", "fpc", "ppolx", "hef", "hwi", "le", "nrfr")
    n = ens["n"]
    S = None
    for s in range(STRATA):
        idx = np.where(mem["_stratum"] == s)[0]
        C = {k: v[idx] for k, v in mem.items() if not k.startswith("_")}
        C.update(pyear=POLICY_YEAR, drur=1.0, dppolx=1e-3, fioac2=0.43 * mp["fioac2_factor"],
                 dcfsn=C["dcfsn"] * mp["family_factor"])
        T = {"rtcm": (-1, 0, 1, [-(tech["R0"][s] + mp["resource_tech_rate"]), 0.0]),
             "pptcm": (-1, 0, 1, [-(tech["P0"][s] + mp["pollution_tech_rate"]), 0.0]),
             "lytcrm": (0, 1, 1, [0.0, tech["Y0"][s]])}
        lo, hi, st, vals = l0.TABLES["isopc1"]
        T["isopc2"] = (lo, hi, st, list(np.array(vals) * mp["services_factor"]))
        lo, hi, st, vals = l0.TABLES["icormrrct"]
        T["icormrrct"] = (lo, hi, st, list(np.array(vals) * mp["icor_factor"]))
        r = l0.run(constants=C, tables=T, n=len(idx), t1=2100.0, record=keys)
        if S is None:
            t = r.time
            S = {k: np.empty((len(t), n)) for k in keys}
        for k in keys:
            S[k][:, idx] = r.series[k]
    after = t >= 2020
    pop, iopc = S["pop"], S["iopc"]

    def drop(x):
        peak = np.maximum.accumulate(x, 0)
        return np.max(np.where(after[:, None], 1 - x / peak, 0), 0)
    dpop, dio = drop(pop), drop(iopc)
    collapse = (dpop >= 0.2) | (dio >= 0.5)
    cyear = np.full(n, np.nan)
    for j in np.where(collapse)[0]:
        cand = []
        for x, thr in ((iopc[:, j], 0.5), (pop[:, j], 0.2)):
            i = np.where(after & ((1 - x / np.maximum.accumulate(x)) >= thr))[0]
            if len(i):
                cand.append(t[i[0]])
        cyear[j] = min(cand)
    over = np.sum(np.maximum(S["hef"][after] - 1, 0), 0) * (t[1] - t[0])
    at = lambda x, yr: x[np.searchsorted(t, yr)]  # noqa: E731
    return {"p_collapse": float(collapse.mean()), "p_collapse_pop": float((dpop >= 0.2).mean()),
            "p_collapse_io": float((dio >= 0.5).mean()),
            "collapse_year_median": float(np.nanmedian(cyear)) if collapse.any() else None,
            "p_collapse_2050": float(np.mean(cyear <= 2050)), "p_collapse_2070": float(np.mean(cyear <= 2070)),
            "p_severe": float(np.mean((dpop >= 0.4) | (dio >= 0.75))),
            "severity_pop_median": float(np.median(dpop)), "severity_io_median": float(np.median(dio)),
            "ppolx_peak": float(np.median(S["ppolx"][after].max(0))), "overshoot_hef_years": float(np.median(over)),
            "nrfr_2100": float(np.median(S["nrfr"][-1])), "hwi_2050": float(np.median(at(S["hwi"], 2050))),
            "hwi_2100": float(np.median(S["hwi"][-1])), "le_2100": float(np.median(S["le"][-1])),
            "pop_2100": float(np.median(pop[-1])), "pop_peak": float(np.median(pop.max(0))),
            "iopc_2050": float(np.median(at(iopc, 2050))), "fpc_2050": float(np.median(at(S["fpc"], 2050))),
            "bands": {k: np.percentile(S[k][::10], [10, 50, 90], 1).T.tolist() for k in ("pop", "iopc", "ppolx", "hwi", "fpc", "hef")},
            "years": t[::10].tolist(), "_collapse": collapse}
