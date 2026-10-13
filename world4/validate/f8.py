"""F8: relaciones metamórficas de las palancas de política (se suman a W6).

Cada palanca, en el mundo mediano (posterior de F7 en su mediana, eficacias = 1), con valores 0, ½ y el
máximo de su caja, debe mover su objetivo con el signo declarado en ``scenarios.levers`` y de forma
monótona. El mutante (el efecto de las renovables con el signo cambiado) debe romper la relación.
"""
from __future__ import annotations

import numpy as np

from world4.scenarios import levers


def _median_world() -> dict:
    from world4.calib import posterior
    post = posterior.load()["l0r"]
    return {n: np.array([float(np.median(post["theta"][:, j]))]) for j, n in enumerate(post["names"])}


def target(r: dict, name: str) -> float:
    from world4.core.l0r import regions
    i50 = 31
    ssa = regions.CODES.index("SSA")
    if name == "nonfossil_share_2050":
        return float(np.sum(r["nfs"][i50, 0] * r["E"][i50, 0]) / np.sum(r["E"][i50, 0]))
    if name == "energy_2050":
        return float(np.sum(r["E"][i50, 0]))
    if name == "cereal_yield_2100":
        return float(np.mean(r["lnYld"][-1, 0]))
    if name == "gdp_pc_SSA_2100":
        return float(r["Y"][-1, 0, ssa] / r["P"][-1, 0, ssa])
    if name == "population_SSA_2100":
        return float(r["P"][-1, 0, ssa])
    if name == "debt_world_2100":
        Y = r["Y"][-1, 0]
        return float(np.sum(r["debt"][-1, 0] * Y) / np.sum(Y))
    raise KeyError(name)


def lever_monotonicity(mutant: bool = False) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l0r import model
    from world4.scenarios import robust
    p, ini = robust._base()
    mult = _median_world()
    old = model.EPS_R
    if mutant:
        model.EPS_R = -abs(old)
    try:
        out = {}
        for j, lv in enumerate(levers.LEVERS):
            vals = []
            for x in (0.0, 0.5 * lv.hi, lv.hi):
                th = np.zeros(len(levers.NAMES))
                th[j] = x
                pol = {k: v for k, v in levers.to_policy(th).items()}
                r = model.L0R(p, ini, M=1, mult=mult, policy=pol).run(
                    robust.END - robust.START, record=("nfs", "E", "lnYld", "Y", "P", "debt"))
                vals.append(target(r, lv.target))
            d = np.diff(vals) * lv.sign
            out[lv.name] = {"target": lv.target, "values": vals, "declared_sign": lv.sign,
                            "monotone": bool(np.all(d > 0))}
    finally:
        model.EPS_R = old
    return out


def run(log=print) -> dict:
    ok = lever_monotonicity()
    mut = lever_monotonicity(mutant=True)
    res = {"levers": ok, "n_monotone": sum(v["monotone"] for v in ok.values()), "n": len(ok),
           "mutant": "renewables effect with the sign inverted",
           "mutant_caught": not mut["renewables"]["monotone"]}
    log(f"F8 levers: {res['n_monotone']}/{res['n']} monotone with the declared sign; mutant caught: {res['mutant_caught']}")
    return res
