"""F8: narrativas tipo SSP como CONJUNTOS DE PARÁMETROS del L0, no como trayectorias impuestas.

Cada narrativa desplaza las perillas del posterior de F7 (multiplica los multiplicadores, suma a los
desplazamientos) y fija el centro de las no identificadas. La lectura cualitativa es la de O'Neill et al.
(2017, *Global Environmental Change* 42); la traducción a números es [M]: no se ajustó a la base de
escenarios SSP del IIASA (eso es L3). El modelo decide la trayectoria; la narrativa sólo dice en qué mundo.

``sow(n, narrative)`` da el ensemble de estados del mundo de la búsqueda robusta:

* las perillas identificadas, del posterior de F7 (``models/posterior.npz``);
* las no identificadas (recurso, daño, EROI, sustitución, déficit, calor), de su caja de previa
  (incertidumbre profunda: ninguna distribución es mejor que otra);
* la eficacia de cada grupo de palancas, U(0,5; 1,5) [M];
* y encima, la narrativa.
"""
from __future__ import annotations

import numpy as np

# narrativa -> {perilla: ("x", factor) | ("+", desplazamiento)}
NARRATIVES = {
    "ssp1": {"what": "sustainability: fast demographic transition, convergence, efficiency, clean diffusion",
             "cbr": ("x", 0.92), "lam_A": ("x", 1.5), "d_int": ("+", -0.003), "d_z": ("+", 0.02),
             "d_yield": ("+", 0.002)},
    "ssp2": {"what": "middle of the road: the F7 posterior as it is"},
    "ssp3": {"what": "regional rivalry: slow transition and convergence, weak productivity and yields",
             "cbr": ("x", 1.08), "lam_A": ("x", 0.4), "d_gA": ("+", -0.003), "d_yield": ("+", -0.002),
             "d_z": ("+", -0.01)},
    "ssp4": {"what": "inequality: divergence between regions, some clean energy in the rich ones",
             "lam_A": ("x", 0.3), "d_int": ("+", -0.002), "cbr": ("x", 1.03)},
    "ssp5": {"what": "fossil-fuelled development: fast growth on abundant fossil energy",
             "d_gA": ("+", 0.003), "d_int": ("+", 0.002), "d_z": ("+", -0.01), "resource": ("x", 1.5),
             "cbr": ("x", 0.95), "lam_A": ("x", 1.5)},
}
DEFAULT = "ssp2"


def apply(mult: dict, narrative: str) -> dict:
    if narrative not in NARRATIVES:
        raise ValueError(f"unknown narrative {narrative!r}; narratives: {', '.join(NARRATIVES)}")
    out = {k: np.array(v, float) for k, v in mult.items()}
    for k, rule in NARRATIVES[narrative].items():
        if k == "what":
            continue
        op, val = rule
        base = out.get(k, np.ones_like(next(iter(out.values()))) if op == "x" else
                       np.zeros_like(next(iter(out.values()))))
        out[k] = base * val if op == "x" else base + val
    return out


def sow(n: int, narrative: str = DEFAULT, seed: int = 0) -> dict:
    """``n`` estados del mundo: {perilla o eficacia: (n,)}."""
    from world4.calib import posterior
    from world4.scenarios import levers
    rng = np.random.default_rng(seed)
    post = posterior.load()["l0r"]
    m = posterior.draw_l0r(post, n, rng, explore=True)
    for k, (lo, hi) in levers.EFFECT_BOX.items():
        m[k] = rng.uniform(lo, hi, n)
    return apply(m, narrative)
