"""Grupos de países para la demografía del L0 vigente (fase V2).

World3 es un solo mundo medio; la fecundidad y la mortalidad dependen del ingreso de forma no lineal, y la media
de una función no lineal no es la función de la media. Aquí el mundo se parte en grupos FIJOS de países y de
cada grupo se saca, de los datos (``world4.data.world`` y ``world4.data.wpp``):

* su población por edad simple y sexo, sus nacimientos, muertes y migración neta (residuo contable);
* su fecundidad total y su patrón por edad (nacimientos de los miembros / mujeres de los miembros);
* su familia de tablas de vida (``lifetable.aggregate_family`` sobre las tasas agregadas de sus miembros) y su
  esperanza de vida: la de esas tasas agregadas, que es MENOR que la media de la de sus miembros (2,8 años en el
  mundo de 1950), llevada a año con la media de los miembros y puesta en la escala de la e0 oficial de WPP con
  el desfase que tiene el agregado mundial en cada quinquenio;
* su ingreso relativo ``rho`` (PIB por persona del grupo / el del mundo, Maddison) y su alimento relativo
  ``phi`` (kilocalorías por persona del grupo / las del mundo, FAO).

Particiones: ``income`` (umbrales de PIB por persona en un año de referencia; los países sin dato toman el de
su subregión de la ONU), ``regions`` (las 10 regiones de ``world4.core.l0r.regions``) y ``world`` (un solo
grupo: debe reproducir la fase 1).

Años: los de WPP, 1950-2023. ``rho`` acaba en 2022 y ``phi`` cubre 1961-2013; fuera de sus años se mantienen
en el último valor conocido [S].
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from world4.core.l0v import lifetable
from world4.core.l0v.cohorts import A
from world4.sectors.population import asfr_single, single_age

INCOME_YEAR = 1970
# umbrales de PIB por persona en INCOME_YEAR, dólares internacionales de 2011 (Maddison): cuatro grupos
INCOME_CUTS = (1500.0, 4000.0, 10000.0)
INCOME_NAMES = ("low", "lower-middle", "upper-middle", "high")


@dataclass
class GroupInputs:
    names: tuple
    member: np.ndarray           # (países,) índice de grupo, -1 fuera
    years: np.ndarray            # (Y,)
    popM: np.ndarray             # (Y, G, 101) personas a 31 de diciembre
    popF: np.ndarray
    births: np.ndarray           # (Y, G) personas en el año
    deaths: np.ndarray
    migration: np.ndarray        # (Y, G) migrantes netos en el año (suman 0 en el mundo)
    tfr: np.ndarray              # (Y, G)
    pattern: np.ndarray          # (Y, G, 101) suma 1
    srb: np.ndarray              # (Y, G)
    e0: np.ndarray               # (Y, G)
    rho: np.ndarray              # (Y, G) ingreso por persona relativo al mundo
    phi: np.ndarray              # (Y, G) alimento por persona relativo al mundo
    families: tuple              # una lifetable.Family por grupo
    offset_t: np.ndarray         # (P,) centros de quinquenio
    offset: np.ndarray           # (P,) e0 oficial del mundo - e0 de sus tasas agregadas calculada aquí

    @property
    def G(self) -> int:
        return len(self.names)


def _panel():
    from world4.data import world, wpp
    return wpp.load(), world.load_countries()


def gdp_pc(year: int = INCOME_YEAR) -> np.ndarray:
    """PIB por persona de cada país en ``year``; sin dato, el de su subregión de la ONU (o el de su continente)."""
    d, c = _panel()
    t = int(year - c["years"][0])
    pop = (np.nansum(d["popM"][:, t], 1, dtype=float) + np.nansum(d["popF"][:, t], 1, dtype=float)) * 1e3
    y = c["gdp"][:, t] / pop
    for key in ("subregion", "area"):
        for name in np.unique(c[key]):
            m = c[key] == name
            ok = m & np.isfinite(y)
            if ok.any():
                fill = np.nansum(c["gdp"][ok, t]) / pop[ok].sum()
                y = np.where(m & ~np.isfinite(y), fill, y)
    return y


def partition(kind: str = "income"):
    """``(nombres, índice de grupo por país)`` de una partición con nombre."""
    d, c = _panel()
    if kind == "world":
        return ("world",), np.zeros(len(c["codes"]), int)
    if kind == "income":
        return INCOME_NAMES, np.searchsorted(np.asarray(INCOME_CUTS), gdp_pc(INCOME_YEAR), side="right")
    if kind == "regions":
        from world4.core.l0r import regions
        idx = np.array([regions.CODES.index(regions.OVERRIDES.get(i) or regions.BY_SUBREGION.get(s, "SEA"))
                        for i, s in zip(c["iso3"], c["subregion"])])
        return regions.CODES, idx
    raise ValueError(f"partition must be 'income', 'regions' or 'world', got {kind!r}")


def _sum(a, member, G):
    """Suma por grupo a lo largo del eje de países (el primero)."""
    return np.stack([np.nansum(a[member == g], 0, dtype=float) for g in range(G)], 0)


def _ratio_to_world(num, den, member, G):
    """``(Σ num / Σ den)`` del grupo sobre el del mundo, usando sólo los países-año con dato; (Y, G)."""
    ok = np.isfinite(num) & np.isfinite(den)
    n, dn = np.where(ok, num, 0.0), np.where(ok, den, 0.0)
    world = n.sum(0) / np.where(dn.sum(0) > 0, dn.sum(0), np.nan)
    grp = _sum(n, member, G) / np.where(_sum(dn, member, G) > 0, _sum(dn, member, G), np.nan)
    r = (grp / world).T                                                  # (Y, G)
    for g in range(G):                                                   # huecos: el valor conocido más cercano
        col = r[:, g]
        ok1 = np.isfinite(col)
        if ok1.any():
            r[:, g] = np.interp(np.arange(len(col)), np.where(ok1)[0], col[ok1])
        else:
            r[:, g] = 1.0
    return r


def build(names, member) -> GroupInputs:
    from world4.data import wpp
    d, c = _panel()
    years = c["years"]
    G, Y = len(names), len(years)
    member = np.asarray(member, int)
    PM, PF = d["popM"].astype(float) * 1e3, d["popF"].astype(float) * 1e3          # (C, Y, 101)
    PM, PF = np.nan_to_num(PM), np.nan_to_num(PF)
    popM = _sum(PM, member, G).transpose(1, 0, 2)                                   # (Y, G, 101)
    popF = _sum(PF, member, G).transpose(1, 0, 2)
    births = _sum(c["births"], member, G).T
    deaths = _sum(c["deaths"], member, G).T
    tot = popM.sum(2) + popF.sum(2)
    migration = np.zeros((Y, G))
    migration[1:] = tot[1:] - tot[:-1] - births[1:] + deaths[1:]

    # fecundidad: nacimientos por edad de cada país = mujeres a mitad de año x TFR x patrón del quinquenio
    est = d["periods"] + 2 <= wpp.YEAR1
    periods = d["periods"][est]
    pidx = np.clip((years - (periods[0] - 2)) // 5, 0, len(periods) - 1)
    pat_c = asfr_single(np.nan_to_num(d["pasfr"][:, est].astype(float)))            # (C, P, 101)
    Wmid = PF.copy()
    Wmid[:, 1:] = 0.5 * (PF[:, 1:] + PF[:, :-1])
    b_age = Wmid * np.nan_to_num(c["tfr"])[:, :, None] * pat_c[:, pidx]             # (C, Y, 101)
    asfr_g = _sum(b_age, member, G) / np.maximum(_sum(Wmid, member, G), 1.0)        # (G, Y, 101)
    tfr = asfr_g.sum(2).T
    pattern = (asfr_g / np.maximum(asfr_g.sum(2, keepdims=True), 1e-12)).transpose(1, 0, 2)
    srb_c = np.nan_to_num(d["srb"].astype(float), nan=1.05)
    srb = (_sum(c["births"] * srb_c, member, G) / np.maximum(_sum(c["births"], member, G), 1.0)).T

    # mortalidad: tasas del grupo por quinquenio = muertes de sus miembros / población de sus miembros
    e0c = c["e0"]
    ptot = PM.sum(2) + PF.sum(2)                                                    # (C, Y)
    base = lifetable.wpp_family()
    mM, mF = single_age(d["mxM"][:, est].astype(float)), single_age(d["mxF"][:, est].astype(float))   # (C, P, 101)
    mid = np.clip(periods - 1 - years[0], 0, Y - 1)                                 # 31 de diciembre del año c-1

    def pooled(m):
        """log de las tasas agregadas de los países ``m`` por quinquenio: (P, 2, 101)."""
        lm = np.empty((len(periods), 2, A))
        for s, (mx, P) in enumerate(((mM, PM), (mF, PF))):
            w = np.where(np.isfinite(mx[m]), P[m][:, mid], 0.0)                     # (c, P, 101)
            lm[:, s] = np.log(np.maximum((w * np.nan_to_num(mx[m])).sum(0) / np.maximum(w.sum(0), 1e-9), 1e-9))
        return lm

    def members_mean(m):
        """Media de la e0 oficial de los países ``m`` con su población: anual (Y,) y por quinquenio (P,)."""
        ok = np.isfinite(e0c[m])
        yearly = (np.nan_to_num(e0c[m]) * ptot[m]).sum(0) / np.maximum((ptot[m] * ok).sum(0), 1.0)
        return yearly, np.array([yearly[max(0, int(p) - 2 - years[0]): int(p) + 3 - years[0]].mean() for p in periods])

    everyone = member >= 0
    lw = pooled(everyone)
    wy, wp = members_mean(everyone)
    from world4.data import world as _world
    wobs = _world.load()
    official = np.array([np.nanmean(wobs["le"][int(p) - 2 - wobs["years"][0]: int(p) + 3 - wobs["years"][0]])
                         for p in periods])
    to_official = official - lifetable.e0_both(np.exp(lw[:, 0]), np.exp(lw[:, 1]))  # (P,) desfase del mundo
    tp = periods.astype(float)
    fams, e0 = [], np.empty((Y, G))
    for g in range(G):
        lm = pooled(member == g)
        yearly, per = members_mean(member == g)
        node = lifetable.e0_both(np.exp(lm[:, 0]), np.exp(lm[:, 1])) + to_official  # e0 del grupo en cada quinquenio
        e0[:, g] = yearly + np.interp(years, tp, node - per)
        fams.append(lifetable.aggregate_family(base, lm, node))

    pmid = ptot.copy()
    pmid[:, 1:] = 0.5 * (ptot[:, 1:] + ptot[:, :-1])
    rho = _ratio_to_world(c["gdp"], np.where(np.isfinite(c["gdp"]), pmid, np.nan), member, G)
    phi = _ratio_to_world(c["kcal"] * pmid, np.where(np.isfinite(c["kcal"]), pmid, np.nan), member, G)
    return GroupInputs(names=tuple(names), member=member, years=years, popM=popM, popF=popF, births=births,
                       deaths=deaths, migration=migration, tfr=tfr, pattern=pattern, srb=srb, e0=e0, rho=rho,
                       phi=phi, families=tuple(fams), offset_t=tp, offset=to_official)


@lru_cache(maxsize=1)
def world_drivers() -> dict:
    """Índices mundiales observados (1970 = 1) con los que se puede mover la demografía en lugar de con la economía
    de World3: PIB por persona (suma de los países con dato de Maddison, 1950-2022) y kilocalorías por persona
    (FAO, 1961-2013). Fuera de sus años se quedan en el último valor [S]. ``{"years", "gdp_pc", "kcal"}``."""
    from world4.data import world
    d, c = _panel()
    years = c["years"]
    ptot = (np.nansum(d["popM"], 2, dtype=float) + np.nansum(d["popF"], 2, dtype=float)) * 1e3
    pmid = ptot.copy()
    pmid[:, 1:] = 0.5 * (ptot[:, 1:] + ptot[:, :-1])
    ok = np.isfinite(c["gdp"])
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(ok, c["gdp"], 0.0).sum(0) / np.where(ok, pmid, 0.0).sum(0)

    def hold(v):
        good = np.isfinite(v)
        return np.interp(np.arange(len(v)), np.where(good)[0], v[good])

    w = world.load()
    kcal = w["kcal"][np.searchsorted(w["years"], years)]
    i70 = int(1970 - years[0])
    y, kcal = hold(y), hold(kcal)
    return {"years": years, "gdp_pc": y / y[i70], "kcal": kcal / kcal[i70]}


@lru_cache(maxsize=4)
def inputs(kind: str = "income") -> GroupInputs:
    """Los datos de una partición con nombre (``income``, ``regions``, ``world``), en caché."""
    return build(*partition(kind))
