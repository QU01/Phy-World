"""F8: el vector θ de palancas de política, con caja, unidades y el efecto que se declara.

Las palancas actúan sobre el L0 regional (``core.l0r.model``, constantes ``EPS_*``, todas [M]) desde el
primer año de la corrida, con una rampa de ``RAMP`` años hasta el efecto pleno. Todas se pagan: el gasto
sale de la inversión (menos capital, menos PIB después), el precio del carbono quita ``DW`` × recaudación
del PIB, la consolidación fiscal recorta inversión. Su eficacia es incierta (``EFFECT``): un factor por
estado del mundo que entra en el ensemble de la búsqueda robusta y en PRIM.

Signo declarado sobre su objetivo (relaciones metamórficas de F8, como las de W6):
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world4.core.l0r import model


@dataclass(frozen=True)
class Lever:
    name: str
    lo: float
    hi: float
    unit: str
    what: str
    target: str            # salida del L0 que debe moverse
    sign: int              # +1 sube, -1 baja con la palanca


LEVERS = (
    Lever("carbon_price", 0.0, 200.0, "USD/tCO2", "world carbon price, revenue recycled", "nonfossil_share_2050", +1),
    Lever("renewables", 0.0, 0.02, "share of GDP", "public investment in non-fossil deployment (from investment)",
          "nonfossil_share_2050", +1),
    Lever("efficiency", 0.0, 0.02, "share of GDP", "energy-efficiency investment (from investment)",
          "energy_2050", -1),
    Lever("agri_rd", 0.0, 0.005, "share of GDP", "agricultural R&D and extension (from investment)",
          "cereal_yield_2100", +1),
    Lever("aid", 0.0, 0.01, "share of donor GDP", "NAM, EUR, OPA -> SSA, SAS as capital (half arrives)",
          "gdp_pc_SSA_2100", +1),
    Lever("fertility", 0.0, 0.01, "share of GDP in SSA, SAS, MEA", "girls' education and family planning",
          "population_SSA_2100", -1),
    Lever("consolidation", 0.0, 1.0, "fraction of the deficit", "fiscal consolidation (costs public investment)",
          "debt_world_2100", -1),
)
NAMES = tuple(lv.name for lv in LEVERS)
BOX = np.array([(lv.lo, lv.hi) for lv in LEVERS], float)
EFFECT_BOX = {k: (0.5, 1.5) for k in model.EFFECT}       # eficacia de cada grupo de palancas [M]
assert NAMES == model.POLICY

__doc__ += "\n".join(f"    {lv.name:14s} [{lv.lo:g}, {lv.hi:g}] {lv.unit}: {lv.what}; {lv.target} "
                     f"{'+' if lv.sign > 0 else '-'}" for lv in LEVERS)


def to_policy(theta) -> dict:
    """θ (d,) o (n, d) -> dict para ``model.L0R(policy=...)``."""
    th = np.atleast_2d(np.asarray(theta, float))
    if th.shape[1] != len(NAMES):
        raise ValueError(f"theta has {th.shape[1]} values; the levers are {NAMES}")
    lo, hi = BOX[:, 0], BOX[:, 1]
    if np.any(th < lo - 1e-12) or np.any(th > hi + 1e-12):
        raise ValueError(f"theta outside the lever box {dict(zip(NAMES, BOX.tolist()))}")
    return {n: th[:, j] for j, n in enumerate(NAMES)}


def parse(text: str) -> np.ndarray:
    """``"carbon_price=100,renewables=0.01"`` -> θ (las demás a 0)."""
    th = np.zeros(len(NAMES))
    if not text:
        return th
    for part in text.split(","):
        k, _, v = part.partition("=")
        k = k.strip()
        if k not in NAMES:
            raise ValueError(f"unknown lever {k!r}; levers: {', '.join(NAMES)}")
        th[NAMES.index(k)] = float(v)
    to_policy(th)
    return th


def describe() -> list[dict]:
    return [{"name": lv.name, "box": [lv.lo, lv.hi], "unit": lv.unit, "what": lv.what, "target": lv.target,
             "declared_sign": lv.sign, "mark": "[M]"} for lv in LEVERS]
