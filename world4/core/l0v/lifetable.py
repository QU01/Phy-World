"""Familia de tablas de vida por sexo, indexada por la esperanza de vida, estimada con UN WPP 2024.

World3 da UNA esperanza de vida mundial; un modelo de cohortes necesita la mortalidad por edad y sexo que le
corresponde. Aquí se estima con todos los países y quinquenios de las estimaciones de WPP (1950-2023; unos 3300
pares país-periodo, con e0 de menos de 30 a más de 85 años): para cada nivel de e0 de una rejilla, el log de la
tasa de mortalidad de cada edad y sexo es una regresión local lineal (núcleo gaussiano) sobre la e0 de ambos
sexos. La regresión local lineal extrapola con la pendiente del borde cuando el modelo pide una e0 fuera de lo
observado (los colapsos de World3 bajan de 30 años) [S: fuera del rango de los datos es una extrapolación].

``e0 de ambos sexos`` de una tabla = media de la de hombres y mujeres con la razón de sexos al nacer (1,05).
Después de estimar la familia se calcula la e0 que cada nivel da DE VERDAD y se invierte por interpolación, de
modo que ``mx(e0)`` devuelve tasas cuya e0 es la pedida (prueba).

**El mundo no es un país medio.** A igual e0, el agregado mundial tiene MÁS mortalidad infantil y MENOS
mortalidad de mayores que un país típico (los niños que mueren están en los países pobres y los mayores viven en
los ricos), y la diferencia crece con el tiempo: en 2016-2020 la tasa de 1-4 años del mundo es 2,8 veces la de la
familia de países a su misma e0, y la de 75+ un 30 % menor (medido). Por eso ``world_family`` usa, dentro del
rango observado (e0 de 49 a 72), las tablas del propio agregado mundial de WPP, interpoladas por la e0 OFICIAL
de WPP de cada quinquenio; fuera de él las prolonga con la forma de la familia de países, manteniendo el
residuo del borde [S]. Así ``mx(e0)`` con la e0 oficial de un año devuelve la mortalidad mundial de ese año.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from world4.sectors.population import life_expectancy, single_age

SRB = 1.05
A = 101
LEVELS = np.arange(15.0, 92.51, 1.25)
BANDWIDTH = 2.0


def e0_both(mxM, mxF, srb: float = SRB):
    """e0 de ambos sexos de tasas por edad simple (..., 101) de hombres y mujeres."""
    return (srb * life_expectancy(mxM) + life_expectancy(mxF)) / (1 + srb)


@dataclass(frozen=True)
class Family:
    levels: np.ndarray        # (L,) nivel nominal
    logmx: np.ndarray         # (L, 2, 101) log de tasas; sexo 0 = hombres, 1 = mujeres
    e0: np.ndarray            # (L,) e0 de ambos sexos que da cada nivel, creciente

    def mx(self, e0):
        """Tasas (..., 2, 101) cuya e0 de ambos sexos es ``e0`` (recortada al rango de la familia)."""
        e0 = np.clip(np.asarray(e0, float), self.e0[0], self.e0[-1])
        pos = np.interp(e0, self.e0, np.arange(len(self.e0)))
        i = np.minimum(pos.astype(int), len(self.e0) - 2)
        w = (pos - i)[..., None, None]
        return np.exp((1 - w) * self.logmx[i] + w * self.logmx[i + 1])


def fit(mxM5, mxF5, levels=LEVELS, bandwidth: float = BANDWIDTH) -> Family:
    """Familia a partir de tasas quinquenales de WPP ``(N, 22)`` por sexo (filas con NaN se descartan)."""
    ok = np.isfinite(mxM5).all(-1) & np.isfinite(mxF5).all(-1)
    lm = np.log(np.maximum(np.stack([single_age(mxM5[ok]), single_age(mxF5[ok])], 1), 1e-9))    # (N, 2, 101)
    e0 = e0_both(np.exp(lm[:, 0]), np.exp(lm[:, 1]))
    out = np.empty((len(levels), 2, A))
    for k, lev in enumerate(levels):
        x = e0 - lev
        w = np.exp(-0.5 * (x / bandwidth) ** 2)
        if w.sum() < 5.0:                                    # fuera de los datos: los 50 pares más cercanos
            near = np.argsort(np.abs(x))[:50]
            w = np.zeros_like(x)
            w[near] = 1.0
        s0, s1, s2 = w.sum(), (w * x).sum(), (w * x * x).sum()
        t0 = np.tensordot(w, lm, 1)
        t1 = np.tensordot(w * x, lm, 1)
        out[k] = (s2 * t0 - s1 * t1) / (s0 * s2 - s1 * s1)   # ordenada de la recta local en x = 0
    fam_e0 = e0_both(np.exp(out[:, 0]), np.exp(out[:, 1]))
    keep = np.r_[True, np.diff(np.maximum.accumulate(fam_e0)) > 1e-6]      # e0 estrictamente creciente
    return Family(levels=np.asarray(levels)[keep], logmx=out[keep], e0=fam_e0[keep])


@lru_cache(maxsize=1)
def wpp_family() -> Family:
    """La familia estimada con las estimaciones de WPP 2024 (periodos cerrados en 2023 o antes)."""
    from world4.data import wpp
    d = wpp.load()
    est = d["periods"] + 2 <= wpp.YEAR1
    return fit(d["mxM"][:, est].reshape(-1, d["mxM"].shape[-1]).astype(float),
               d["mxF"][:, est].reshape(-1, d["mxF"].shape[-1]).astype(float))


def aggregate_family(base: Family, logmx, labels) -> Family:
    """Familia de un AGREGADO de países (el mundo o un grupo). ``logmx``: log de sus tasas por periodo, sexo y
    edad simple ``(P, 2, 101)``; ``labels``: la e0 con la que se pedirá cada periodo. Dentro del rango de
    ``labels`` devuelve las tablas del agregado; fuera, las prolonga con la forma de ``base`` y el residuo del
    borde."""
    lm = np.asarray(logmx, float)
    lab = np.asarray(labels, float)
    order = np.argsort(lab)
    lab, lm = lab[order], lm[order]
    keep = np.r_[True, np.diff(np.maximum.accumulate(lab)) > 1e-6]              # un año de crisis no hace retroceder
    lab, lm = lab[keep], lm[keep]
    own = e0_both(np.exp(lm[:, 0]), np.exp(lm[:, 1]))                           # la e0 que dan esas tasas aquí
    lo, hi = lab[0], lab[-1]
    levels = np.unique(np.r_[base.e0[(base.e0 < lo) | (base.e0 > hi)], lab])
    out = np.empty((len(levels), 2, A))
    for k, lev in enumerate(levels):
        if lo <= lev <= hi:
            j = int(np.argmin(np.abs(lab - lev)))                               # los niveles interiores son nodos
            out[k] = lm[j]
        else:
            j = 0 if lev < lo else -1                                           # borde: residuo constante
            out[k] = lm[j] + np.log(base.mx(lev + (own[j] - lab[j]))) - np.log(base.mx(own[j]))
    return Family(levels=levels, logmx=out, e0=levels)


def world_family(base: Family, mxM5, mxF5, e0_official) -> Family:
    """Familia del agregado mundial. ``mxM5``, ``mxF5``: tasas quinquenales del mundo ``(P, 22)``;
    ``e0_official``: la e0 de ambos sexos que WPP publica para cada uno de esos periodos."""
    lm = np.log(np.maximum(np.stack([single_age(np.asarray(mxM5, float)), single_age(np.asarray(mxF5, float))], 1),
                           1e-9))                                              # (P, 2, 101)
    return aggregate_family(base, lm, e0_official)


@lru_cache(maxsize=1)
def wpp_world_family() -> Family:
    """La familia del mundo con las tablas mundiales de WPP 2024 (``world4.data.world``)."""
    from world4.data import world
    w = world.load()
    le = dict(zip(w["years"].tolist(), w["le"].tolist()))
    official = [float(np.mean([le[y] for y in range(int(p) - 2, int(p) + 3)])) for p in w["mx_periods"]]
    return world_family(wpp_family(), w["mxM"], w["mxF"], official)
