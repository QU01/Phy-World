"""Geopolítica y sociedad 2019-2100 con la capa L1 de Phy-World (Atlas del Sistema-Mundo, 180 países).

Corre el L1 acoplado (F1 población y capital humano endógenos, F2 mercados de recursos, F3 energía) desde el
origen vivo 2019 hasta 2100 en cuatro escenarios, con números aleatorios comunes, y resume lo que el L1 tiene
de sociedad y geopolítica: riesgo de conflicto civil, golpes, democracia, crisis de deuda, índice de presión
política (PSI = movilización de masas × élites × estrés fiscal, Goldstone/Turchin), élites, asabiya,
centro/periferia e intercambio desigual.

Escenarios [M]:
* inercia       la regla de conducta clonada de los datos (lo que «harían» los países);
* transicion    perillas de F7 del L1: intensidad energética −1 %/año, logit no fósil +0,05/año,
                ln TFR −0,5 %/año (la traducción al L1 de las palancas tecnológicas y demográficas);
* autonomia     la conducta que «les convendría» según el problema inverso del Atlas (``scenarios.conduct``):
                más tecnología propia y extracción propia, menos importación del centro y menos inversión
                fuera, más redistribución;
* combinado     transicion + autonomia;
* estancamiento desde 2020 la frontera de productividad retrocede 0,5 %/año (g0 = −0,005 en lugar de ≈ +0,003): una prueba
                de estrés que traslada al L1 la descapitalización del valle de World3, que el L1 no genera solo.

Límites declarados: el L1 está validado a 1-10 años; a 2100 es un modo de comportamiento. Sin datos de clima
(FaIR) ni de alimentos (FAOSTAT) en esta sesión, la temperatura es la TCRE del Atlas y no hay sector de
alimentos: la crisis alimentaria del núcleo World3 NO entra aquí. Salida: ``geopolitica.json``.

    PYTHONPATH=. python reports/escenarios/geopolitica.py
"""
from __future__ import annotations

import json
import os
import time
import warnings
from multiprocessing import get_context
from pathlib import Path

import numpy as np

OUT = Path(__file__).with_name("geopolitica.json")
T0 = 69                      # 2019 - 1950: el origen vivo del L1
HORIZON = 81                 # 2019 -> 2100
CHUNKS, PER_CHUNK = int(os.environ.get("W4_CHUNKS", 4)), int(os.environ.get("W4_PER_CHUNK", 16))  # 64 miembros, 4 procesos
SNAP = (2019, 2030, 2050, 2075, 2100)
SCENARIOS = {
    "inercia": {"name": "Inercia", "calib": None, "conduct": False},
    "transicion": {"name": "Transición", "calib": {"intensity": -0.01, "nonfossil": 0.05, "tfr": -0.005},
                   "conduct": False},
    "autonomia": {"name": "Autonomía y redistribución", "calib": None, "conduct": True},
    "combinado": {"name": "Transición + autonomía", "calib": {"intensity": -0.01, "nonfossil": 0.05, "tfr": -0.005},
                  "conduct": True},
    # el puente con el valle del núcleo World3: la frontera tecnológica retrocede en vez de avanzar
    "estancamiento": {"name": "Estancamiento", "calib": None, "conduct": False, "g0_after": -0.005},
}


def _patch(g0_after=None):
    """Dos ganchos sobre el simulador del Atlas, sin tocar el port:

    * el acople sustituye el PSI por país por la psi_t mundial de los mercados: se guarda aparte como psi_c;
    * ``g0_after``: la deriva de la frontera tecnológica cambia sólo DESPUÉS del origen (la historia
      1950-2019 queda igual en todos los escenarios)."""
    import dataclasses

    from world4.core.l1_atlas import model as amodel
    amodel.Simulator._G0_AFTER = g0_after
    if "psi_c" in amodel.Simulator.REC:
        return
    amodel.Simulator.REC = list(amodel.Simulator.REC) + ["psi_c"]
    orig = amodel.Simulator.step

    def step(self, actions):
        g = amodel.Simulator._G0_AFTER
        if g is not None and self.t > T0 and self.p.g0 != g:
            self.p = dataclasses.replace(self.p, g0=g)
        rec = orig(self, actions)
        rec["psi_c"] = rec["psi"]
        return rec
    amodel.Simulator.step = step


def _wavg(x, w):
    w = np.where(np.isfinite(x), w, 0.0)
    return np.nansum(np.nan_to_num(x) * w, 1) / np.maximum(np.nansum(w, 1), 1e-12)


def _chunk(args):
    key, seed = args
    warnings.simplefilter("ignore")
    sc = SCENARIOS[key]
    _patch(sc.get("g0_after"))
    from world4.core import coupling
    from world4.core.l0r import regions
    from world4.core.l1_atlas import model as amodel
    from world4.scenarios import conduct
    W = amodel.build_world(horizon=2100)
    f = coupling.forecast(T0, R=PER_CHUNK, seed=seed, world=W, horizon=HORIZON, f3=coupling.F3Config(),
                          calib=sc["calib"], offsets=conduct.advised_offsets() if sc["conduct"] else None, raw=True)
    raw = f["raw"]
    pop = np.nan_to_num(f["pop"])                         # (R, N, T) millones
    Y = np.exp(f["ly"]) * pop
    core = raw["core"]
    periph = np.nan_to_num(core) < 0.5
    lpsi = np.log(np.where(raw["psi_c"] > 0, raw["psi_c"], np.nan))
    gen = {k[11:]: np.nan_to_num(f[k]) for k in f if k.startswith("energy_gen_")}
    tot = sum(gen.values())
    world = {
        "pop": np.nansum(pop, 1) / 1e3,                                        # miles de millones
        "conflicts": np.nansum(f["hazard"], 1),                                # países en conflicto esperados
        "pop_conflict": _wavg(f["hazard"], pop),                               # riesgo medio por persona
        "demo_pop": _wavg(f["demo"], pop),
        "demo_countries": np.nanmean(f["demo"], 1),
        "coups": np.nansum(raw["coup"], 1),
        "crises": np.nansum(raw["crisis"], 1),
        "debt": _wavg(f["debt"], Y),
        "gdp_pc": np.nansum(Y, 1) / np.maximum(np.nansum(pop, 1), 1e-9),
        "lpsi": _wavg(lpsi, pop),
        "elite_share": _wavg(raw["E"], pop),
        "elite_n": _wavg(raw["ne"], pop),
        "asab": _wavg(raw["asab"], pop),
        "core_pop": np.nansum(np.where(~periph, pop, 0), 1) / np.maximum(np.nansum(pop, 1), 1e-9),
        "gap": (np.nansum(np.where(~periph, Y, 0), 1) / np.maximum(np.nansum(np.where(~periph, pop, 0), 1), 1e-9))
               / (np.nansum(np.where(periph, Y, 0), 1) / np.maximum(np.nansum(np.where(periph, pop, 0), 1), 1e-9)),
        "drain": _wavg(np.where(periph, raw["drain"], np.nan), np.where(periph, Y, 0)),
        "Tg": f["Tg"],
        "coups_decade": np.stack([np.nansum(raw["coup"][:, :, 1 + 10 * d:11 + 10 * d], (1, 2)) for d in range(8)], 1),
        "nonfossil": 1 - np.nansum(gen["fossil"], 1) / np.maximum(np.nansum(tot, 1), 1e-9),
    }
    ridx = regions.region_index(W.iso3, W.isonum)
    years = 2019 + np.arange(HORIZON + 1)
    si = [int(np.searchsorted(years, y)) for y in SNAP]
    reg = {}
    for j, code in enumerate(regions.CODES):
        m = ridx == j
        pr = pop[:, m]
        reg[code] = {
            "hazard": _wavg(f["hazard"][:, m], pr)[:, si],
            "demo": _wavg(f["demo"][:, m], pr)[:, si],
            "gdp_pc": (np.nansum(Y[:, m], 1) / np.maximum(np.nansum(pr, 1), 1e-9))[:, si],
            "psi_rel": 100 * np.exp(_wavg(lpsi[:, m], pr)[:, si] - _wavg(lpsi[:, m], pr)[:, si[:1]]),
            "coups_decade": np.stack([np.nansum(raw["coup"][:, m, max(0, k - 10):k + 1], (1, 2)) for k in si], 1),
            "pop": (np.nansum(pr, 1) / 1e3)[:, si],
        }
    country = {"hazard": f["hazard"][:, :, si], "demo": f["demo"][:, :, si], "lpsi": lpsi[:, :, si],
               "gdp_pc": np.exp(f["ly"])[:, :, si]}
    return key, world, reg, country, list(W.iso3), ridx.tolist()


def q3(a, axis=0):
    return [[float(f"{v:.4g}") for v in np.nanpercentile(a, p, axis=axis)] for p in (10, 50, 90)]


def main():
    t0 = time.time()
    jobs = [(k, 1000 + 97 * c) for k in SCENARIOS for c in range(CHUNKS)]
    with get_context("spawn").Pool(CHUNKS) as pool:
        res = pool.map(_chunk, jobs)
    print(f"L1: {len(jobs)} trozos en {time.time() - t0:.0f} s")
    from world4.core.l0r import regions
    years = list(range(2019, 2019 + HORIZON + 1))
    out = {"meta": {"generated": time.strftime("%Y-%m-%d"), "members": CHUNKS * PER_CHUNK, "origin": 2019, "end": 2100,
                    "runtime_s": None, "snap": SNAP},
           "scenarios": {k: {"name": v["name"], "calib": v["calib"], "conduct": v["conduct"],
                             "overrides": v.get("overrides")} for k, v in SCENARIOS.items()},
           "decades": [f"{2020 + 10 * d}s" for d in range(8)], "years": years, "regions": {c: regions.NAMES[c] for c in regions.CODES}, "world": {}, "region": {},
           "country": {}}
    from world4.scenarios import conduct
    out["advised_offsets"] = conduct.advised_offsets()
    iso, ridx = res[0][4], res[0][5]
    out["iso3"], out["ridx"] = iso, ridx
    for key in SCENARIOS:
        parts = [r for r in res if r[0] == key]
        wd = {k: np.concatenate([p[1][k] for p in parts], 0) for k in parts[0][1]}
        wd["psi_index"] = 100 * np.exp(wd["lpsi"] - wd["lpsi"][:, :1])
        out["world"][key] = {k: q3(v) for k, v in wd.items() if k != "lpsi"}
        out["world"][key]["lpsi_2019"] = float(np.median(wd["lpsi"][:, 0]))
        rg = {}
        for code in regions.CODES:
            d = {k: np.concatenate([p[2][code][k] for p in parts], 0) for k in parts[0][2][code]}
            rg[code] = {k: [float(f"{x:.4g}") for x in (np.nanmean(v, 0) if k in ("hazard", "coups_decade")
                                                       else np.nanmedian(v, 0))] for k, v in d.items()}
        out["region"][key] = rg
        cd = {k: np.concatenate([p[3][k] for p in parts], 0) for k in parts[0][3]}
        # el riesgo de conflicto por país es bimodal (en guerra ≈ 0,95, en paz ≈ 0,02): media = probabilidad
        out["country"][key] = {k: np.round(np.nanmean(v, 0) if k == "hazard" else np.nanmedian(v, 0), 4).tolist()
                               for k, v in cd.items()}   # (N, len(SNAP))
    out["meta"]["runtime_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")).replace("NaN", "null"), encoding="utf-8")
    print(f"listo en {out['meta']['runtime_s']} s -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
