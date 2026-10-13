"""Panel regional del L0 (10 regiones × años 1950-2019, y 2020-2023 donde hay dato) a partir de los MISMOS
datos que usa el L1: PWT/Atlas (población, PIB, capital, inversión, depreciación, tasas brutas, migración,
deuda, temperatura local), F3 (energía primaria y cuota no fósil), F5 (cereales, tierras, N mineral), F2
(extracción fósil acumulada y reservas), F4 (CO2 y GMST).

Dos tipos de serie:

* **Niveles** sobre un conjunto FIJO de países (``members``: los observados en el año base), para las
  condiciones iniciales y el candado P2: así el agregado no salta cuando entra un país (la antigua URSS,
  Europa del Este y varios africanos entran en 1990 en la PWT).
* **Tasas encadenadas** (crecimiento entre t-1 y t sólo con los países presentes en los dos años), para
  estimar las ecuaciones con toda la historia disponible.
"""
from __future__ import annotations

import numpy as np

from world4.core.l0r import regions

NR = len(regions.CODES)
BASE_DEFAULT = 1992          # el primer año con todas las regiones en PWT, FAOSTAT y energía


def _wsum(x, w, m):
    """Suma de x*w (o de x si w es None) sobre los países de m, NaN si no hay ninguno con dato."""
    v = x if w is None else x * w
    ok = m & np.isfinite(v)
    s = np.where(ok, v, 0.0).sum(0)
    return np.where(ok.any(0), s, np.nan)


_CACHE = {}


def country_panel(W) -> dict:
    """Series (N, T) por país en los años del L1, con las de los sectores."""
    key = id(W)
    if key in _CACHE:
        return _CACHE[key]
    from world4.data import food as fdata
    from world4.sectors import energy
    obs = W.observed
    pop = np.where(obs, W.pop, np.nan)
    Y = np.where(obs, W.Y, np.nan)
    K = np.where(obs, W.K, np.nan)
    ep = energy.panel(W)
    fa = fdata.aligned(W.iso3, W.isonum, W.years)
    nf = sum(ep["src"][f"src_{s}"] for s in ("nuclear", "hydro", "vre", "other"))
    out = {"pop": pop, "Y": Y, "K": K, "csh_i": W.csh_i, "delta": W.delta,
           "cbr": W.extra.get("crude_birth_rate"), "cdr": W.extra.get("crude_death_rate"),
           "mig": W.extra.get("net_migration_rate"), "debt": W.locf("debt_gdp"),
           "tmp": W.locf("tmp_level"), "beta_t": W.locf("tmp_pattern_beta", 1.33),
           "pe": ep["pe"], "nonfossil": nf, "cereal": fa["cereal_prod"], "cropland": fa["cropland"],
           "cereal_area": fa["cereal_area"], "nfert": fa["n_fert"]}
    _CACHE[key] = out
    return out


def regional(W, base_year: int = BASE_DEFAULT) -> dict:
    """Niveles (10, T) sobre los países observados en ``base_year`` y tasas encadenadas (10, T)."""
    key = ("reg", id(W), base_year)
    if key in _CACHE:
        return _CACHE[key]
    c = country_panel(W)
    ridx = regions.region_index(W.iso3, W.isonum)
    tb = base_year - 1950
    T = W.T
    members = W.observed[:, tb] & (ridx >= 0)
    lev, chain = {}, {}
    pop, Y, K = c["pop"], c["Y"], c["K"]
    for name, x, w in (("pop", pop, None), ("Y", Y, None), ("K", K, None), ("pe", c["pe"], None),
                       ("nonfossil", c["nonfossil"], None), ("cereal", c["cereal"], None),
                       ("cropland", c["cropland"], None), ("cereal_area", c["cereal_area"], None),
                       ("nfert", c["nfert"], None)):
        lev[name] = np.array([_wsum(x, w, members[:, None] & (ridx == r)[:, None]) for r in range(NR)])
    # promedios ponderados (niveles sobre el conjunto fijo)
    for name, x, wname in (("cbr", c["cbr"], "pop"), ("cdr", c["cdr"], "pop"), ("mig", c["mig"], "pop"),
                           ("csh_i", c["csh_i"], "Y"), ("delta", c["delta"], "K"), ("debt", c["debt"], "Y"),
                           ("tmp", c["tmp"], "pop"), ("beta_t", c["beta_t"], "pop")):
        w = {"pop": pop, "Y": Y, "K": K}[wname]
        num = np.array([_wsum(x, w, members[:, None] & (ridx == r)[:, None]) for r in range(NR)])
        den = np.array([_wsum(np.where(np.isfinite(x), 1.0, np.nan), w, members[:, None] & (ridx == r)[:, None])
                        for r in range(NR)])
        lev[name] = num / den
    # tasas encadenadas: crecimiento del agregado con los países presentes en t-1 y t
    for name, x in (("pop", pop), ("Y", Y), ("K", K), ("pe", c["pe"]), ("cereal", c["cereal"]),
                    ("cropland", c["cropland"]), ("nfert", c["nfert"]), ("nonfossil", c["nonfossil"]),
                    ("cereal_area", c["cereal_area"])):
        g = np.full((NR, T), np.nan)
        for r in range(NR):
            m = ridx == r
            for t in range(1, T):
                both = m & np.isfinite(x[:, t]) & np.isfinite(x[:, t - 1]) & (x[:, t - 1] > 0)
                if both.sum() >= 1:
                    a, b = x[both, t].sum(), x[both, t - 1].sum()
                    if a > 0 and b > 0:
                        g[r, t] = np.log(a / b)
        chain[name] = g
    # índices encadenados (nivel del año base = 1), para ajustar ecuaciones en niveles relativos
    idx = {}
    for name, g in chain.items():
        a = np.full((NR, T), np.nan)
        a[:, tb] = 0.0
        for t in range(tb + 1, T):
            a[:, t] = a[:, t - 1] + g[:, t]
        for t in range(tb - 1, -1, -1):
            a[:, t] = a[:, t + 1] - g[:, t + 1]
        idx[name] = a
    out = {"levels": lev, "chain": chain, "logindex": idx, "members": members, "ridx": ridx, "base": tb}
    _CACHE[key] = out
    return out


def world(W) -> dict:
    """Series mundiales (T,): extracción fósil acumulada y producción (petróleo + gas + carbón, TWh),
    reservas, CO2 (CMIP7, GtCO2) y GMST (HadCRUT5)."""
    key = ("world", id(W))
    if key in _CACHE:
        return _CACHE[key]
    from world4.data import climate as cdata
    from world4.sectors import resources
    D = resources.data(W)
    yrs = 1950 + np.arange(W.T)
    X = sum(np.asarray(D[c]["X"], float) for c in ("oil", "gas", "coal"))
    Q = sum(np.asarray(D[c]["Q"], float) for c in ("oil", "gas", "coal"))
    res = {c: np.asarray(D[c]["reserves"], float) for c in ("oil", "gas", "coal")}
    ffi = cdata.co2_ffi_observed(yrs)
    d = cdata.load()
    ey = d["emis_years"]
    afolu = np.array([d["emis"]["CO2 AFOLU"][1][y - ey[0]] for y in yrs])
    cum = np.array([np.sum(d["emis"]["CO2 FFI"][1][:y - ey[0] + 1]) + np.sum(d["emis"]["CO2 AFOLU"][1][:y - ey[0] + 1])
                    for y in yrs])
    out = {"X": X, "Q": Q, "reserves": res, "co2_ffi": ffi, "co2_afolu": afolu, "cum_co2": cum,
           "gmst": cdata.gmst_observed(yrs)}
    _CACHE[key] = out
    return out
