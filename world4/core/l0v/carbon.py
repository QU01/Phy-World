"""Carbono y clima para el L0 vigente (fase 4): el contaminante que sí se acumuló.

La «contaminación persistente» de World3 dura unos dos años en el ambiente y no se parece a nada que se pueda
medir a escala mundial. Lo que sí se mide es el CO2: aquí las emisiones salen del uso de recursos no renovables
del modelo y, con dos relaciones estimadas con datos, dan el CO2 de la atmósfera y la temperatura global.

    emisiones fósiles e industria (GtCO2/año) = las reales de 1970 · uso de recursos(t) / uso de recursos(1970)
    emisiones por uso del suelo               = las observadas; después del último dato o del origen, la media
                                                de los diez años anteriores [S]
    CO2 atmosférico (ppm) = c0 + airborne · emisiones acumuladas / 7,82
    temperatura global    = t0 + tcre · emisiones acumuladas / 1000

* ``airborne`` (la fracción de lo emitido que sigue en el aire) y ``c0`` se estiman con Mauna Loa, 1959-``cutoff``
  (con ``c0`` fijo en 280 ppm la recta se queda 11 ppm corta en 2023: lo emitido antes de 1959 se absorbió más);
* ``tcre`` y ``t0`` con GISTEMP, 1900-``cutoff``. Es una TCRE EFECTIVA: carga al CO2 todo el calentamiento, también
  el del metano y lo que enfrían los aerosoles, así que sale mayor que la del CO2 solo [declarado].
* Las dos relaciones son lineales en las emisiones acumuladas: valen mientras se siga emitiendo. Cuando las
  emisiones caen, el CO2 real baja despacio y esta fórmula lo deja fijo; la temperatura sí se queda donde está.

**Daño (regla, no estimación).** El rendimiento agrícola pierde ``yield_damage`` (5 %) por cada grado de
calentamiento global desde el origen. Viene del rango que F5 midió con FAOSTAT (de -2,7 % por grado local en un
país templado a -5 % en uno cálido) y de que la tierra se calienta más que la media del planeta [S]. Lo aplica el
gancho de ``technology`` sobre la curva de rendimiento. Antes del origen no hay daño aparte: la curva de
rendimiento se estimó con los rendimientos reales, que ya lo llevan dentro.

La contaminación persistente de World3 sigue en el modelo con sus efectos (suelo y esperanza de vida), pero el
marcador ya no la compara con el CO2.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

PPM_GT = 7.82                 # GtCO2 por ppm de CO2 atmosférico
PREINDUSTRIAL = 280.0
ANCHOR = 1970
LUC_YEARS = 10


@dataclass(frozen=True)
class Carbon:
    airborne: float           # fracción de las emisiones que sigue en la atmósfera (pendiente desde 1959)
    c0: float                 # ppm: ordenada de esa recta
    tcre: float               # °C por 1000 GtCO2 acumuladas (efectiva)
    t0: float                 # °C sobre 1951-1980 con cero emisiones acumuladas
    cutoff: int
    yield_damage: float = 0.05
    damage_floor: float = 0.2

    def ppm(self, cumulative):
        return self.c0 + self.airborne * np.asarray(cumulative, float) / PPM_GT

    def gmst(self, cumulative):
        return self.t0 + self.tcre * np.asarray(cumulative, float) / 1000.0

    def damage(self, warming):
        """Multiplicador del rendimiento con ``warming`` grados de calentamiento desde el origen."""
        return np.maximum(self.damage_floor, 1.0 - self.yield_damage * np.maximum(np.asarray(warming, float), 0.0))


def observed(obs: dict | None = None) -> dict:
    """Las series observadas en GtCO2: ``years``, ``fi``, ``luc`` (por año), ``cum`` (acumuladas, las dos), ``ppm``,
    ``gmst``."""
    if obs is None:
        from world4.data import world
        obs = world.load()
    return {"years": np.asarray(obs["years"], int), "fi": obs["co2_fi"] / 1e3, "luc": obs["luc_co2"] / 1e3,
            "cum": (obs["cum_co2_fi"] + obs["cum_luc_co2"]) / 1e3, "ppm": obs["co2_ppm"], "gmst": obs["gmst"]}


def estimate(cutoff: int | None = None, obs: dict | None = None) -> Carbon:
    """Las tres perillas con datos <= ``cutoff`` (None: todos)."""
    o = observed(obs)
    y = o["years"]
    cut = int(y[np.isfinite(o["gmst"]) & np.isfinite(o["cum"])][-1]) if cutoff is None else int(cutoff)
    a = np.isfinite(o["ppm"]) & np.isfinite(o["cum"]) & (y <= cut)
    airborne, c0 = np.polyfit(o["cum"][a] / PPM_GT, o["ppm"][a], 1)
    g = np.isfinite(o["gmst"]) & np.isfinite(o["cum"]) & (y <= cut)
    slope, t0 = np.polyfit(o["cum"][g] / 1000.0, o["gmst"][g], 1)
    return Carbon(airborne=float(airborne), c0=float(c0), tcre=float(slope), t0=float(t0), cutoff=cut)


@lru_cache(maxsize=8)
def params(cutoff: int | None = None) -> Carbon:
    return estimate(cutoff)


def luc_after(o: dict, year: int) -> float:
    """La regla de las emisiones por uso del suelo después de ``year``: la media de los diez años hasta ``year``."""
    sel = (o["years"] <= year) & (o["years"] > year - LUC_YEARS) & np.isfinite(o["luc"])
    return float(o["luc"][sel].mean())


def resource_use(run, member: int = 0):
    """Uso anual de recursos de una corrida en años enteros (diferencia centrada del stock ``nr``)."""
    t = np.asarray(run.time, float)
    ok = np.isclose(t, np.round(t))
    years = np.round(t[ok]).astype(int)
    nr = np.asarray(run["nr"], float)[ok, member]
    use = np.full(len(years), np.nan)
    use[1:-1] = (nr[:-2] - nr[2:]) / (years[2:] - years[:-2])
    use[0] = (nr[0] - nr[1]) / (years[1] - years[0])
    use[-1] = (nr[-2] - nr[-1]) / (years[-1] - years[-2])
    return years, use


def diagnose(run, par: Carbon | None = None, origin: int | None = None, obs: dict | None = None, member: int = 0) -> dict:
    """Carbono y clima que IMPLICA una corrida, sin retroalimentarla: ``years``, ``emissions`` (GtCO2/año, fósil +
    suelo), ``cumulative``, ``co2_ppm``, ``gmst``. Con ``origin`` la historia hasta ese año es la observada y el
    modelo pone las emisiones desde entonces; sin él, el modelo las pone desde 1900."""
    o = observed(obs)
    par = par or estimate(None, obs)
    years, use = resource_use(run, member)
    oy = o["years"]
    fi70 = float(o["fi"][oy == ANCHOR][0])
    fi = fi70 * use / use[years == ANCHOR][0]
    last_luc = int(oy[np.isfinite(o["luc"])][-1])
    rule_from = last_luc if origin is None else min(int(origin), last_luc)
    luc = np.where(years <= rule_from, np.interp(years, oy, np.nan_to_num(o["luc"])), luc_after(o, rule_from))
    emissions = fi + luc
    start = int(years[0]) if origin is None else int(origin)
    base = float(o["cum"][oy == start][0]) - (float(emissions[years == start][0]) if origin is None else 0.0)
    after = years > start if origin is not None else years >= start
    cum = np.where(years <= start if origin is not None else False, np.interp(years, oy, np.nan_to_num(o["cum"])),
                   base + np.cumsum(np.where(after, emissions, 0.0)))
    if origin is not None:                                            # la historia: emisiones observadas
        hist = years <= start
        emissions = np.where(hist, np.interp(years, oy, np.nan_to_num(o["fi"] + o["luc"])), emissions)
    return {"years": years, "emissions": emissions, "cumulative": cum, "co2_ppm": par.ppm(cum), "gmst": par.gmst(cum)}


def check(run, par: Carbon | None = None, origin: int | None = None, window=(2000, 2023), obs: dict | None = None) -> dict:
    """Error absoluto medio en la ventana: CO2 atmosférico (ppm) y temperatura global (°C)."""
    o = observed(obs)
    d = diagnose(run, par, origin, obs)
    out = {}
    for key, name in (("ppm", "co2_ppm"), ("gmst", "gmst")):
        sel = (o["years"] >= window[0]) & (o["years"] <= window[1]) & np.isfinite(o[key]) & np.isin(o["years"], d["years"])
        m = np.interp(o["years"][sel], d["years"], d[name])
        out[name + "_mae"] = float(np.abs(m - o[key][sel]).mean())
        out[name + "_last"] = [int(o["years"][sel][-1]), float(m[-1]), float(o[key][sel][-1])]
    return out


def resource_base(run, obs: dict | None = None, member: int = 0) -> dict:
    """Los recursos no renovables de una corrida en la moneda del carbono: GtCO2 que emitiría quemarlos, con la
    misma equivalencia que las emisiones (las reales de 1970 por unidad de uso de 1970). ``total`` es el stock
    inicial; ``left`` lo que queda cada año."""
    o = observed(obs)
    years, use = resource_use(run, member)
    per_unit = float(o["fi"][o["years"] == ANCHOR][0]) / use[years == ANCHOR][0]
    t = np.asarray(run.time, float)
    ok = np.isclose(t, np.round(t))
    nr = np.asarray(run["nr"], float)[ok, member]
    return {"years": years, "gt_per_unit": per_unit, "total": float(nr[0] * per_unit), "left": nr * per_unit}
