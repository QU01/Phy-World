"""World3 ↔ sociedad: el ensemble de 500 mundos del estudio de escalas, con y sin la vuelta de la sociedad.

Cada experimento corre el mismo ensemble (misma semilla y filtro histórico que ``reports/escenarios``) dos veces
con números aleatorios comunes:

* ``ida``     la sociedad sigue a World3 pero no lo toca (World3 queda idéntico al de los informes previos);
* ``acople``  los eventos destruyen capital de World3 y la capacidad de implementar política cae con ellos.

Experimentos: inercia (geología base y abundante); con recursos abundantes (donde la política cambia el destino),
medio paquete y el paquete Limits to Growth desde 2027, y tres pruebas de estrés de la vuelta sobre este último
(huellas ×2, riesgos ×1,5, ambas). Salida: ``acoplado.json``.

    PYTHONPATH=. python reports/acople/simular.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "reports" / "escenarios"))
sys.path.insert(0, str(ROOT / "reports" / "arbol_decision"))
import estudio as E  # noqa: E402
import proyeccion as base  # noqa: E402

from world4.core import social  # noqa: E402

warnings.simplefilter("ignore")
OUT = HERE / "acoplado.json"
SEED = 20261022
REC = E.REC + ("so",)
EXPERIMENTS = {   # con la geología base todos los mundos colapsan con o sin política (estudio de escalas)
    "inercia": {"name": "Inercia, recursos base", "pol": E.OFF, "geo": 1.0},
    "inercia_abund": {"name": "Inercia, recursos abundantes", "pol": E.OFF, "geo": 2.0},
    "moderado_abund": {"name": "Medio paquete desde 2027, recursos abundantes", "pol": E.package(0.5), "geo": 2.0},
    "ltg_abund": {"name": "Paquete Limits to Growth desde 2027, recursos abundantes", "pol": E.REF, "geo": 2.0},
    "estres_huella": {"name": "Paquete LtG, abundantes, huellas en el PIB ×2", "pol": E.REF, "geo": 2.0,
                      "kw": {"impulse_scale": 2.0}},
    "estres_riesgo": {"name": "Paquete LtG, abundantes, riesgos ×1,5", "pol": E.REF, "geo": 2.0,
                      "kw": {"hazard_scale": 1.5}},
    "estres_doble": {"name": "Paquete LtG, abundantes, huellas ×2 y riesgos ×1,5", "pol": E.REF, "geo": 2.0,
                     "kw": {"impulse_scale": 2.0, "hazard_scale": 1.5}},
}
SOC_KEYS = ("conflict_onset", "coup", "coup_success", "irregular_exit", "dem_breakdown", "democratization",
            "sovdebt_onset", "in_conflict", "pop_conflict", "demo_pop", "demo_countries", "valley10",
            "countries_valley10", "loss", "gdp_index", "capacity", "growth_world")
W3_KEYS = ("pop", "le", "iopc", "fpc", "ppolx", "hwi", "hef")
SNAP = (2030, 2045, 2060, 2080, 2100)


def ensemble(n=500):
    rng = np.random.default_rng(base.SEED)
    obs = base.observed_population()
    cand = base.sample_uncertain(6000, rng)
    c = base.constants_for(cand, res_abundant=np.zeros(6000), year=E.NEVER, clean=False, agro=False, demo=False,
                           suff=False)
    yrs, s = base.simulate(cand, base.tables_for(False, False), c)
    p2000 = s["pop"][np.searchsorted(yrs, 2000)] * 1e-9
    p2020 = s["pop"][np.searchsorted(yrs, 2020)] * 1e-9
    ok = (np.abs(p2000 / obs[2000] - 1) < 0.06) & (np.abs(p2020 / obs[2020] - 1) < 0.06)
    idx = np.flatnonzero(ok)[:n]
    return {k: v[idx] for k, v in cand.items()}


def q3(a, axis=1):
    return [[float(f"{v:.4g}") for v in np.nanpercentile(a, p, axis=axis)] for p in (10, 50, 90)]


def one(U, spec, feedback):
    n = len(U["nri_mult"])
    hook = social.Coupler(n, seed=SEED, feedback=feedback, capacity=feedback, **spec.get("kw", {}))
    yrs, w3 = E.run(U, spec["pol"], geo=spec["geo"], rec=REC, hook=hook)
    return yrs, w3, hook.result()


def summarise(yrs, w3, soc):
    o = E.outcomes(yrs, w3)
    summ = E.summary(o)
    keep = (yrs >= 1990) & ((yrs - 1990) % 2 == 0)
    sy = soc.years
    out = {"destiny": summ["p_destiny"], "le_min": summ["le_min"], "le_2100": summ["le_2100"],
           "pop_peak": summ.get("pop_peak"), "pop_2100": summ.get("pop_2100"),
           "w3": {k: q3(w3[k][keep] * base.VARS.get(k, (None, None, 1))[2]) for k in W3_KEYS},
           "soc": {k: q3(soc[k]) for k in SOC_KEYS},
           "soc_mean": {k: [float(f"{v:.4g}") for v in soc[k].mean(1)] for k in SOC_KEYS}}
    dec = {}
    for d0 in range(2020, 2100, 10):
        m = (sy >= d0) & (sy < d0 + 10)
        dec[f"{d0}s"] = {k: float(soc[k][m].sum(0).mean()) for k in ("conflict_onset", "coup", "coup_success",
                                                                    "sovdebt_onset", "dem_breakdown",
                                                                    "democratization", "irregular_exit")}
        dec[f"{d0}s"].update({k: float(soc[k][m].mean()) for k in ("pop_conflict", "demo_pop", "valley10", "capacity")})
    out["decades"] = dec
    out["groups"] = {g: {k: [float(f"{v:.4g}") for v in (np.median(x, 1) if k == "gdp_pc" else x.mean(1))]
                         for k, x in soc.groups[g].items()}
                     for g in social.GROUPS}
    # eventos según el crecimiento mundial del año (el puente del ingreso, a escala del mundo)
    gw = soc["growth_world"]
    bins = [-1, -0.02, -0.005, 0.005, 0.02, 1]
    out["by_world_growth"] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (gw >= lo) & (gw < hi)
        out["by_world_growth"].append({"bin": [lo, hi], "years": int(m.sum()),
                                       **{k: float(soc[k][m].mean()) if m.any() else None
                                          for k in ("conflict_onset", "coup", "sovdebt_onset", "dem_breakdown",
                                                    "in_conflict")}})
    si = [int(np.searchsorted(sy, y)) for y in SNAP]
    out["country"] = {k: np.round(soc.country[k][si], 4).tolist() for k in ("p_conflict", "conflict", "dem", "lny", "p_coup")}
    out["io_rel"] = None
    return out, o


def no_valley(n, P, g=0.015):
    """La misma sociedad (semilla, demografía de WPP) con un mundo que crece ``g`` al año sin valle: separa lo que
    pone el valle de World3 de lo que pone la demografía."""
    soc = social.Society(P, n, seed=SEED)
    soc.ypc_prev = np.ones(n)
    rows, years = [], list(range(2020, 2101))
    for y in years:
        rows.append(soc.step(y, np.full(n, (1 + g) ** (y - 2019)))[2])
    S = {k: np.stack([r[k] for r in rows]) for k in SOC_KEYS}
    sy = np.asarray(years)
    dec = {}
    for d0 in range(2020, 2100, 10):
        m = (sy >= d0) & (sy < d0 + 10)
        dec[f"{d0}s"] = {k: float(S[k][m].sum(0).mean()) for k in ("conflict_onset", "coup", "coup_success",
                                                                  "sovdebt_onset", "dem_breakdown", "democratization",
                                                                  "irregular_exit")}
        dec[f"{d0}s"].update({k: float(S[k][m].mean()) for k in ("pop_conflict", "demo_pop", "valley10")})
    return {"growth": g, "decades": dec, "soc": {k: q3(S[k]) for k in SOC_KEYS}}


def main():
    t0 = time.time()
    U = ensemble()
    n = len(U["nri_mult"])
    print(f"ensemble: {n} mundos")
    P = social.load_params()
    res = {"meta": {"generated": time.strftime("%Y-%m-%d"), "members": n, "seed": SEED, "snap": SNAP,
                    "start": 2020, "end": 2100},
           "destinies": E.DESTINIES, "experiments": {}, "iso3": P["countries"]["iso3"],
           "names": P["countries"]["name"], "region": P["countries"]["region"],
           "region_codes": P["countries"]["region_codes"], "years_w3": None, "years_soc": None}
    for key, spec in EXPERIMENTS.items():
        t1 = time.time()
        r = {"name": spec["name"]}
        outs = {}
        for mode, fb in (("ida", False), ("acople", True)):
            yrs, w3, soc = one(U, spec, fb)
            r[mode], outs[mode] = summarise(yrs, w3, soc)
            if res["years_w3"] is None:
                keep = (yrs >= 1990) & ((yrs - 1990) % 2 == 0)
                res["years_w3"], res["years_soc"] = yrs[keep].tolist(), soc.years.tolist()
            outs[mode + "_w3"] = w3
        # diferencias pareadas por mundo (mismos números aleatorios)
        a, b = outs["ida"], outs["acople"]
        wa, wb = outs["ida_w3"], outs["acople_w3"]
        i0, i1 = int(np.searchsorted(yrs, 2020)), int(np.searchsorted(yrs, 2100))
        io_rel = wb["io"][i0:i1 + 1].sum(0) / wa["io"][i0:i1 + 1].sum(0) - 1
        r["paired"] = {"le_min": base.q(b["le_min"] - a["le_min"]), "le_2100": base.q(b["le_2100"] - a["le_2100"]),
                       "pop_2100": base.q((b["pop_2100"] - a["pop_2100"])), "io_2020_2100": base.q(io_rel),
                       "destiny_changed": float(np.mean(a["destiny"] != b["destiny"])),
                       "transitions": [[int(np.sum((a["destiny"] == i) & (b["destiny"] == j))) for j in range(5)]
                                       for i in range(5)]}
        res["experiments"][key] = r
        print(f"{key:15s} {time.time() - t1:5.1f} s  destinos ida {np.round(r['ida']['destiny'], 3)} "
              f"acople {np.round(r['acople']['destiny'], 3)}  PIB perdido 2100 (mediana) "
              f"{1 - r['acople']['soc']['gdp_index'][1][-1]:.3f}")
    res["sin_valle"] = no_valley(n, P)
    hist = json.loads((HERE / "calibracion.json").read_text())["history"]
    res["historia"] = {"years": hist["years"], "obs": hist["obs"], "countries": hist["countries"]}
    res["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null"), encoding="utf-8")
    print(f"listo en {res['meta']['runtime_s']} s -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
