"""Recursos no renovables acotados con datos (V6): una curva de coste en lugar de un depósito que se vacía.

World3 trata los recursos no renovables como un stock fijo (``nri``) y una tabla (``fcaor``) que dice qué parte del
capital industrial hace falta para extraerlos según lo que queda: 5 % mientras sobra, 50 % cuando queda el 30 %.
Ni el stock ni la tabla se pudieron estimar con la historia (V3, V5). Dos fuentes los acotan:

* **Reservas probadas** (BP 2016: relación reservas/producción de 2015 por la producción de ese año). Con lo ya
  emitido dan casi exactamente el stock de World3: es lo que YA está probado, un suelo y no un valor central.
* **Las curvas de oferta de GCAM** (derivadas de Rogner 1997): por combustible y grado, cuánta energía hay y a qué
  coste de extracción. En carbono suman once veces el stock de World3, pero a coste creciente.

Aquí el stock pasa a ser el total de la curva y la tabla de World3 se sustituye por

    fcaor = 5 % · max(1, coste medio de extracción / coste medio de 1970)        (con tope en 95 %)

El 5 % es el valor de World3 con recursos abundantes, que es lo que había en 1970. El coste medio depende de
cuánto se ha extraído de cada combustible:

* hasta el último dato (o hasta ``cutoff``, para que una prueba fuera de muestra no vea la mezcla de combustibles
  posterior a su corte), lo observado (emisiones acumuladas de carbón, petróleo y gas);
* después, entre dos extremos que marca ``substitution``:
  ``0``: cada combustible mantiene su cuota de las emisiones de los últimos diez años (no se sustituyen);
  ``1``: se extrae siempre el grado más barato que quede de cualquier combustible, y el coste no baja del actual.

La cantidad se mide en CO2 porque las emisiones del modelo se anclan a las reales (``carbon``).

Lo que NO cubre: metales y minerales, que también están en el stock de World3; y la curva es una evaluación de
1997 en dólares de 1975, con grados no convencionales especulativos (el último del gas es el 85 % de todo el gas) [declarado].
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

FUELS = {"coal": "cum_coal_co2", "crude oil": "cum_oil_co2", "natural gas": "cum_gas_co2"}
ANNUAL = {"coal": "coal_co2", "crude oil": "oil_co2", "natural gas": "gas_co2"}
BASE_SHARE = 0.05                  # ``fcaor`` de World3 con recursos abundantes
CAP = 0.95
ANCHOR = 1970
C_TO_CO2 = 3.664
TABLE_POINTS = 1000
SHARE_YEARS = 10


@lru_cache(maxsize=1)
def grades() -> dict:
    """La curva de oferta del mundo: por combustible, grados ordenados por coste con su CO2 (Gt) y su energía (EJ)."""
    import pandas as pd
    from world4.data import world
    raw = world.root() / "raw"
    curves = pd.read_csv(raw / "gcam-fos-curves.csv", comment="#")
    coef = pd.read_csv(raw / "gcam-primary-fuel-ccoef.csv", comment="#")
    kgc = dict(zip(coef.iloc[:, 0], coef.iloc[:, 1]))                       # kg C por GJ
    out = {}
    for fuel in FUELS:
        g = curves[curves.resource == fuel].groupby("extractioncost", as_index=False)["available"].sum()
        g = g[g.available > 0].sort_values("extractioncost")
        ej = g.available.to_numpy(float)
        out[fuel] = {"cost": g.extractioncost.to_numpy(float), "ej": ej, "kgc_per_gj": float(kgc[fuel]),
                     "gt": ej * kgc[fuel] * C_TO_CO2 / 1000.0}               # EJ · kgC/GJ = Mt C
    return out


def fuel_cost(fuel: str, extracted):
    """Coste de extracción (dólares de 1975 por GJ) del grado en el que está ``fuel`` con ``extracted`` GtCO2 sacadas."""
    g = grades()[fuel]
    i = np.searchsorted(np.cumsum(g["gt"]), np.asarray(extracted, float), side="right")
    return g["cost"][np.minimum(i, len(g["cost"]) - 1)]


@lru_cache(maxsize=8)
def history(cutoff: int | None = None) -> dict:
    """Lo observado hasta ``cutoff`` (None: todo): emisiones acumuladas por combustible y su suma, la cuota de cada
    uno en los últimos ``SHARE_YEARS`` años y el peso de cada uno en la energía (cuota / contenido de carbono)."""
    from world4.data import world
    w = world.load()
    if cutoff is not None and cutoff < ANCHOR + SHARE_YEARS:
        raise ValueError(f"resources need a cutoff >= {ANCHOR + SHARE_YEARS}, got {cutoff}")
    ok = np.all([np.isfinite(w[k]) for k in FUELS.values()], axis=0) & (w["years"] <= (cutoff or 9999))
    years = w["years"][ok]
    cum = {f: w[k][ok] / 1e3 for f, k in FUELS.items()}
    recent = {f: float(np.nansum(w[k][ok][-SHARE_YEARS:])) for f, k in ANNUAL.items()}
    tot = sum(recent.values())
    share = {f: v / tot for f, v in recent.items()}
    e = {f: share[f] / grades()[f]["kgc_per_gj"] for f in FUELS}
    es = sum(e.values())
    return {"years": years, "cum": cum, "total": sum(cum.values()), "share": share,
            "energy_weight": {f: v / es for f, v in e.items()}}


def _fixed(q, cutoff=None):
    """Coste medio con las cuotas fijas: la historia observada y, después, cada combustible con su cuota."""
    h = history(cutoff)
    q = np.asarray(q, float)
    last = h["total"][-1]
    cost = 0.0
    for f in FUELS:
        qf = np.where(q <= last, np.interp(q, h["total"], h["cum"][f]), h["cum"][f][-1] + h["share"][f] * (q - last))
        cost = cost + h["energy_weight"][f] * fuel_cost(f, qf)
    return cost


def _cheapest(q, cutoff=None):
    """Coste medio si desde el último dato se extrae siempre el grado más barato que quede: no baja del actual y
    sube cuando se acaban los grados más baratos que él."""
    h = history(cutoff)
    q = np.asarray(q, float)
    last = h["total"][-1]
    now = float(_fixed(last, cutoff))
    left = []
    for f in FUELS:
        g = grades()[f]
        used = h["cum"][f][-1]
        rest = np.clip(np.cumsum(g["gt"]) - used, 0.0, g["gt"])              # lo que queda de cada grado
        left += [(c, r) for c, r in zip(g["cost"], rest) if r > 0]
    left.sort()
    edges = np.cumsum([r for _, r in left])
    costs = np.array([c for c, _ in left])
    marginal = costs[np.minimum(np.searchsorted(edges, np.maximum(q - last, 0.0), side="right"), len(costs) - 1)]
    return np.where(q <= last, _fixed(q, cutoff), np.maximum(now, marginal))


def _last(cutoff=None) -> float:
    return float(history(cutoff)["total"][-1])


def total_gt(scale: float = 1.0, cutoff: int | None = None) -> float:
    """Todo el carbono fósil de la curva (GtCO2); ``scale`` multiplica lo que queda por extraer tras el último dato."""
    full = float(sum(g["gt"].sum() for g in grades().values()))
    return _last(cutoff) + (full - _last(cutoff)) * scale


def cost(q, substitution: float = 0.5, scale: float = 1.0, cutoff: int | None = None):
    """Coste medio de extracción (dólares de 1975 por GJ) con ``q`` GtCO2 fósiles emitidas en total. ``scale``
    estira o encoge lo que queda de cada grado tras el último dato: la curva es una evaluación, no una medida."""
    q = np.asarray(q, float)
    last = _last(cutoff)
    q = np.where(q > last, last + (q - last) / scale, q)
    return (1 - substitution) * _fixed(q, cutoff) + substitution * _cheapest(q, cutoff)


def capital_share(q, substitution: float = 0.5, scale: float = 1.0, cutoff: int | None = None):
    """La ``fcaor`` de World3 que corresponde a ``q`` GtCO2 emitidas: 5 % por el coste relativo al de 1970."""
    c70 = float(_fixed(_anchor()["emitted_gt"], cutoff))               # mismos pesos arriba y abajo: 1970 da el 5 %
    return np.clip(BASE_SHARE * np.maximum(1.0, cost(q, substitution, scale, cutoff) / c70), BASE_SHARE, CAP)


@lru_cache(maxsize=1)
def _anchor() -> dict:
    """El enganche con World3 en 1970: GtCO2 por unidad de recurso (las emisiones reales de 1970 por el uso de
    recursos del modelo en 1970), las unidades que el modelo lleva gastadas ese año y lo emitido de verdad. Se
    mide en la corrida con la población observada, que es la de todas las variantes vigentes; antes de 1975 no
    hay tecnología estimada y el coste es plano, así que no depende de nada de lo que aquí se cambia."""
    from world4.core import l0
    from world4.core.l0v import carbon, model
    r, _ = model.run(fertility="observed", mortality="observed", t1=float(ANCHOR) + 1.0, record=("nr", "pop"))
    years, use = carbon.resource_use(r)
    o = carbon.observed()
    h = history()
    t = np.asarray(r.time, float)
    used = float(l0.CONSTANTS["nri"] - np.asarray(r["nr"], float)[np.isclose(t, ANCHOR), 0][0])
    return {"gt_per_unit": float(o["fi"][o["years"] == ANCHOR][0] / use[years == ANCHOR][0]), "used_units": used,
            "emitted_gt": float(h["total"][h["years"] == ANCHOR][0])}


def gt_per_unit() -> float:
    return _anchor()["gt_per_unit"]


@dataclass(frozen=True)
class Resources:
    """Los recursos de la curva de GCAM, para ``model.run(resources=...)``.

    ``substitution`` (0 a 1): cuánto se sustituyen los combustibles entre sí (ver el módulo).
    ``scale``: factor sobre lo que queda por extraer de cada grado (1: la curva tal cual).
    ``cutoff``: último año de la mezcla de combustibles observada que se usa (None: todos)."""
    substitution: float = 0.5
    scale: float = 1.0
    cutoff: int | None = None

    def nri(self) -> float:
        """El stock inicial de World3 con el que, en 1970, al modelo le queda lo que le queda a la curva."""
        a = _anchor()
        return (total_gt(self.scale, self.cutoff) - a["emitted_gt"]) / a["gt_per_unit"] + a["used_units"]

    def emitted(self, nrfr):
        """GtCO2 fósiles emitidas en total cuando al modelo le queda la fracción ``nrfr`` de su stock."""
        return total_gt(self.scale, self.cutoff) - np.asarray(nrfr, float) * self.nri() * gt_per_unit()

    def constants(self) -> dict:
        return {"nri": self.nri()}

    def tables(self) -> dict:
        frac = np.linspace(0.0, 1.0, TABLE_POINTS + 1)
        share = capital_share(self.emitted(frac), self.substitution, self.scale, self.cutoff)
        return {"fcaor1": (0.0, 1.0, 1.0 / TABLE_POINTS, [float(v) for v in share])}


def params(resources, cutoff: int | None = None) -> Resources:
    """``"estimated"`` es la curva central con la mezcla de combustibles observada hasta ``cutoff``; un
    ``Resources`` pasa tal cual."""
    return Resources(cutoff=cutoff) if isinstance(resources, str) else resources


def reserves_check() -> dict:
    """Las reservas probadas de 2015 (BP) en GtCO2, lo emitido hasta entonces y cómo quedan frente al stock de
    World3 y frente al total de la curva de GCAM."""
    import pandas as pd
    from world4.core import l0
    from world4.data import world
    raw = world.root() / "raw"
    rp = pd.read_csv(raw / "bp-reserves-production-ratio-2016.csv")
    rp = rp[rp.Entity == "World"].iloc[0]
    year = int(rp.Year)
    prod = pd.read_csv(raw / "fossil-fuel-production-2022.csv")
    prod = prod[(prod.Entity == "World") & (prod.Year == year)].iloc[0]
    cols = {"coal": (2, "Coal production (TWh)"), "crude oil": (3, "Oil production (TWh)"),
            "natural gas": (4, "Gas production (TWh)")}
    g = grades()
    reserves = {f: float(rp.iloc[i]) * float(prod[c]) * 0.0036 * g[f]["kgc_per_gj"] * C_TO_CO2 / 1000.0
                for f, (i, c) in cols.items()}                               # años · TWh/año · EJ/TWh · MtC/EJ
    h = history()
    emitted = {f: float(h["cum"][f][h["years"] == year][0]) for f in FUELS}
    return {"year": year, "proved_reserves_gt": reserves, "emitted_gt": emitted,
            "reserves_plus_emitted_gt": float(sum(reserves.values()) + sum(emitted.values())),
            "world3_stock_gt": float(l0.CONSTANTS["nri"] * gt_per_unit()), "curve_total_gt": total_gt(),
            "cost_now": {f: float(fuel_cost(f, emitted[f])) for f in FUELS},
            "cost_at_reserves": {f: float(fuel_cost(f, emitted[f] + reserves[f])) for f in FUELS},
            "curve_by_fuel_gt": {f: float(g[f]["gt"].sum()) for f in FUELS}}
