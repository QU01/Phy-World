"""Candado P2: el agregado regional del L1 coincide con el L0 regional en la historia, dentro de una banda
declarada.

El L0 corre LIBRE desde ``BASE`` (1992: el primer año con las 10 regiones completas en PWT, FAOSTAT y
energía) hasta 2019, con los parámetros estimados con datos <= 2019 (P2 es un candado de COHERENCIA
estructural, en muestra; la variante fuera de muestra, con corte en el año inicial, se informa aparte).
El agregado del L1 sobre los mismos países (los observados en 1992):

* el L1 que pronostica está ASIMILADO al dato en el origen (``assimilate``) y sus sectores endógenos sólo
  actúan después del origen: su agregado histórico operativo es el dato (PWT, FAOSTAT, energía, HadCRUT5).
* la corrida LIBRE del Atlas desde 1950 (``l1_history_gdp``) se aleja mucho del dato (China al 37 % y
  África subsahariana al 165 % del PIB de 2019): se informa como hallazgo, no se usa de referencia.

Bandas (declaradas antes de medir, relativas salvo la GMST):
"""
from __future__ import annotations

import numpy as np

BASE, END = 1992, 2019
BANDS = {"pop": 0.05, "gdp": 0.25, "energy": 0.25, "cereal": 0.25, "cropland": 0.10, "gmst": 0.25}
MIN_CELLS = 0.80            # celdas región-año dentro de banda
__doc__ += "\n".join(f"    {k}: ±{v:g}{' °C' if k == 'gmst' else ''}" for k, v in BANDS.items()) + (
    f"\n\nCriterio: el agregado MUNDIAL dentro de banda todos los años y >= {MIN_CELLS:.0%} de las celdas región-año.")


def l1_history_gdp(W, R: int = 8) -> np.ndarray:
    """PIB (N, T) de la simulación histórica del L1 (mediana de R miembros)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l1_atlas import model, validate
    params, _ = validate.params_for(2019)
    sim = model.Simulator(W, params, R=R, seed=0)
    o = sim.run(0, W.T, policy="hist")
    return np.nanmedian(np.exp(o["ly"]), 0) * W.pop


def run(cutoff: int = END, start: int = BASE, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l0r import data as rdata
    from world4.core.l0r import model, regions
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    p = model.calibrate(W, cutoff)
    ini = model.initial(W, start)
    H = END - start
    r = model.L0R(p, ini).run(H)
    R = rdata.regional(W)
    L, I, tb = R["levels"], R["logindex"], R["base"]
    sl = slice(start - 1950, END - 1950 + 1)
    members, ridx = R["members"], R["ridx"]
    lev = lambda name: np.exp(np.log(np.maximum(L[name][:, tb], 1))[:, None] + I[name])[:, sl]  # noqa: E731
    l1 = {"pop": L["pop"][:, sl], "gdp": lev("Y"), "energy": L["pe"][:, sl], "cereal": lev("cereal"),
          "cropland": lev("cropland")}
    l0 = {"pop": r["P"][:, 0].T, "gdp": (np.exp(r["ly"][:, 0]) * r["P"][:, 0]).T, "energy": r["E"][:, 0].T,
          "cereal": (r["cpc"][:, 0] * r["P"][:, 0] * 1e3).T,
          "cropland": None}
    # tierras: el stock del L0
    st = model.L0R(p, ini).run(H, record=("lnC",))
    l0["cropland"] = np.exp(st["lnC"][:, 0]).T
    w = rdata.world(W)
    years = np.arange(start, END + 1)
    out = {"base": start, "end": END, "cutoff": cutoff, "bands": BANDS, "variables": {}}
    ok_all = True
    for k in ("pop", "gdp", "energy", "cereal", "cropland"):
        a, b = l0[k], l1[k]
        rel = a / b - 1
        wrel = a.sum(0) / b.sum(0) - 1
        good = np.isfinite(rel)
        cells = float(np.mean(np.abs(rel[good]) <= BANDS[k]))
        world_in = bool(np.all(np.abs(wrel) <= BANDS[k]))
        worst = int(np.nanargmax(np.abs(rel[:, -1])))
        out["variables"][k] = {"world_rel": wrel.tolist(), "world_inside_all_years": world_in,
                               "cells_inside": cells, "world_rel_end": float(wrel[-1]),
                               "region_rel_end": {regions.CODES[i]: float(rel[i, -1]) for i in range(rdata.NR)},
                               "worst_region_end": regions.CODES[worst]}
        ok_all &= world_in and cells >= MIN_CELLS
    tg = r["Tg"][:, 0, 0]
    gobs = w["gmst"][start - 1950:END - 1950 + 1]
    gin = float(np.mean(np.abs(tg - gobs) <= BANDS["gmst"]))
    out["variables"]["gmst"] = {"abs_err": (tg - gobs).tolist(), "inside_share": gin, "err_end": float(tg[-1] - gobs[-1])}
    ok_all &= gin >= MIN_CELLS
    out["years"] = years.tolist()
    out["ok"] = bool(ok_all)
    free = l1_history_gdp(W)
    out["l1_free_history_gdp_vs_data_end"] = {
        regions.CODES[k]: float(np.nansum(np.where(members & (ridx == k), free[:, END - 1950], 0))
                                / np.nansum(np.where(members & (ridx == k), W.Y[:, END - 1950], 0)))
        for k in range(rdata.NR)}
    log("P2 L1 vs L0 " + f"{start}-{END}: " + "; ".join(
        f"{k} world {v['world_rel_end']:+.1%} in {END}, cells {v['cells_inside']:.0%}" for k, v in out["variables"].items()
        if k != "gmst") + f"; GMST inside {gin:.0%} ({out['variables']['gmst']['err_end']:+.2f} degC in {END})")
    return out
