"""Panel país-año para predecir la densidad de eventos de violencia organizada con el Atlas y World3.

Una fila por país del Atlas (ISO3) y año 1989-2025:

* **Resultado** (UCDP GED 26.1; 2026 queda fuera, sólo tiene enero-septiembre): eventos por tipo, muertes, celdas
  con eventos. El país de un evento es el del lugar donde ocurrió: el de su celda PRIO-GRID según Natural Earth
  (``reference/priogrid_isonum.npz``, ``scripts/make_cell_iso.mjs``), no el código Gleditsch-Ward de UCDP; los
  eventos sin celda fiable (``where_prec`` 5-7) usan el país ISO dominante de su código Gleditsch-Ward.
* **Exposición**: superficie terrestre del país (suma de ``landarea`` de PRIO-GRID de sus celdas). La densidad es
  eventos por 10 000 km².
* **Predictores** con la información del año ``t`` (``FACTORS``, por grupo): historia del conflicto (con los
  vecinos por contigüidad de celdas), economía, demografía, instituciones, desigualdad, sociedad y geografía,
  geopolítica, clima y ambiente, los análogos de World3 por país (los de ``reports/acople/investigacion.py``) y el
  estado mundial (la corrida de World3 y precios mundiales observados).
* Países que el Atlas no tiene pero tienen eventos (``NOT_IN_ATLAS``: Afganistán, Somalia, Sudán del Sur, Libia...)
  entran con su historia de GED y las variables del Atlas vacías (``in_atlas`` = 0).
* Después de 2019 el Atlas no tiene datos: sus variables quedan en su último valor (``atlas_frozen`` = años desde
  2019) y sólo la historia de GED se actualiza. Así se puede predecir desde cualquier año hasta 2025.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from world4 import home
from world4.conflict.grid import TYPES

FIRST_YEAR, LAST_GED_YEAR, LAST_ATLAS_YEAR = 1989, 2025, 2019
AREA_UNIT = 1e4                       # km²: densidad = eventos por 10 000 km²

# grupo -> {predictor: descripción}. Las transformaciones están en ``_features``.
FACTORS = {
    "historia": {
        "h_ev": "ln(1 + eventos en t)", "h_ev_l1": "ln(1 + eventos en t-1)", "h_ev_5y": "ln(1 + media de eventos t-4..t)",
        "h_dens": "ln(1 + eventos por 10 000 km² en t)", "h_deaths": "ln(1 + muertes en t)",
        "h_since": "0,5^(años desde el último evento / 2)", "h_extent": "parte de las celdas con eventos en t",
        "h_sb": "ln(1 + eventos estatales en t)", "h_ns": "ln(1 + eventos no estatales en t)",
        "h_os": "ln(1 + eventos contra civiles en t)", "h_nb": "ln(1 + eventos de los países vecinos en t)",
        "h_acd": "conflicto civil activo (UCDP/PRIO, >= 25 muertes) en t",
    },
    "economía": {
        "lny": "ln PIB per cápita (PWT)", "growth": "crecimiento del PIB per cápita", "infl": "inflación (log con signo)",
        "debt": "deuda pública / PIB", "crisis": "crisis bancaria, cambiaria o de deuda", "res_rents": "rentas de recursos / PIB",
        "oil_rents": "rentas del petróleo / PIB", "trade": "comercio / PIB", "mineral_depl": "agotamiento de minerales / RNB",
    },
    "demografía": {
        "ln_pop": "ln población", "pop_growth": "crecimiento de la población", "youth": "jóvenes 15-24 / adultos",
        "urban": "población urbana", "urban_growth": "crecimiento urbano", "tfr": "fecundidad", "life_exp": "esperanza de vida",
        "dle3": "cambio de la esperanza de vida en 3 años", "median_age": "edad mediana", "migration": "tasa de migración neta",
    },
    "instituciones": {
        "polyarchy": "democracia electoral (V-Dem)", "anocracy": "régimen intermedio: polyarchy × (1 - polyarchy)",
        "corruption": "corrupción (V-Dem)", "rule_of_law": "estado de derecho (V-Dem)", "fiscal_cap": "capacidad fiscal (V-Dem)",
        "state_cap": "capacidad estatal (índice del Atlas)", "excluded": "población excluida (EPR)",
        "discriminated": "población discriminada (EPR)", "coups10": "golpes en 10 años", "regime_change": "cambio de régimen",
        "leader_tenure": "ln años del líder en el poder", "state_age": "ln años desde la independencia",
        "excl_soc": "exclusión por grupo social (V-Dem)",
    },
    "desigualdad": {"gini": "Gini del ingreso disponible", "top10": "parte del 10 % más rico"},
    "sociedad y geografía": {
        "rugged": "rugosidad del terreno", "mountains": "ln % montañoso", "ethfrac": "fraccionalización étnica",
        "al_ethnic": "fraccionalización étnica (Alesina)", "cohesion": "cohesión (índice del Atlas)", "colony": "excolonia",
        "indep_violent": "independencia violenta", "n_neighbors": "vecinos terrestres", "ln_area": "ln superficie",
    },
    "geopolítica": {
        "mids": "disputas militarizadas", "rivalries": "rivalidades estratégicas", "major_power": "gran potencia",
        "oda": "ayuda oficial / RNB", "sanctions": "sanciones", "milex": "gasto militar / PIB",
        "foreign_interv": "guerra civil con intervención extranjera", "unga_us": "distancia a EE. UU. en la ONU",
        "nb_democracy": "democracias entre los vecinos", "ln_cinc": "ln poder material (CINC)",
    },
    "clima y ambiente": {
        "temp_anom": "anomalía de temperatura", "disasters": "ln(1 + desastres climáticos)", "water_stress": "estrés hídrico",
        "ln_yield": "ln rendimiento de cereales", "yield_shock": "rendimiento frente a su media de 5 años",
        "pm25": "PM2,5", "eco_deficit": "déficit ecológico per cápita", "vulnerability": "vulnerabilidad ND-GAIN",
        "ln_co2pc": "ln CO₂ per cápita", "forest": "bosque",
    },
    "World3 mundial": {
        "w3_iopc": "World3: producción industrial per cápita (ln)", "w3_fpc": "World3: alimento per cápita (ln)",
        "w3_nrfr": "World3: recursos no renovables restantes", "w3_ppolx": "World3: contaminación persistente (ln)",
        "w3_le": "World3: esperanza de vida", "w3_hwi": "World3: bienestar humano",
        "w3_fpc_g": "World3: crecimiento del alimento per cápita (5 años)", "w3_iopc_g": "World3: crecimiento industrial (5 años)",
        "w3_fpc_g_x_poor": "World3: crecimiento del alimento × país pobre", "w3_nrfr_x_rents": "World3: recursos × rentas del país",
        "oil_price": "ln precio real del petróleo", "oil_shock_x_exporter": "Δ ln precio del petróleo × exportador",
        "cereal_price": "ln precio real de cereales", "cereal_shock_x_poor": "Δ ln precio de cereales × país pobre",
        "year_trend": "tendencia lineal del año",
    },
}
# análogos por país de las variables de World3 (reports/acople/investigacion.py)
WORLD3_ANALOGS = {"lny": "iopc", "yield_shock": "fpc", "res_rents": "nrfr", "ln_co2pc": "ppolx", "pm25": "ppolx",
                  "dle3": "le", "eco_deficit": "hef", "pop_growth": "pop", "youth": "pop"}
ALL = [f for g in FACTORS.values() for f in g]
# columnas que no entran en el cribado pero usa el modo escenario (variables que simula el L1)
EXTRA = {"democ": "democracia (V-Dem Regimes of the World >= 2), la variable de régimen del L1"}
# países con eventos en GED que el Atlas no tiene (ISO 3166-1): entran con su historia y sin variables del Atlas
NOT_IN_ATLAS = {4: "AFG", 90: "SLB", 232: "ERI", 328: "GUY", 434: "LBY", 598: "PNG", 626: "TLS", 706: "SOM",
                728: "SSD", 732: "ESH"}
GROUP_OF = {f: g for g, fs in FACTORS.items() for f in fs}


# ---------------------------------------------------------------- celdas -> países
@lru_cache(maxsize=1)
def _cells():
    from world4.data import conflict_grid as cdata
    d = cdata.load()
    ref = np.load(home.PKG / "data" / "reference" / "priogrid_isonum.npz")
    if not np.array_equal(ref["gid"], d["gid"]):
        raise ValueError("priogrid_isonum.npz does not match the installed PRIO-GRID cells")
    iso = ref["isonum"].astype(int)
    gw = np.nan_to_num(d["pg_gwno"], nan=-1).astype(int)
    # país Gleditsch-Ward -> ISO dominante entre sus celdas con país
    df = pd.DataFrame({"gw": gw, "iso": iso})
    gw2iso = df[df.iso > 0].groupby("gw").iso.agg(lambda s: s.value_counts().index[0]).to_dict()
    iso = np.where(iso > 0, iso, [gw2iso.get(g, 0) for g in gw])
    return d, iso, gw2iso


def country_geometry() -> pd.DataFrame:
    """isonum -> superficie (km²), número de celdas."""
    d, iso, _ = _cells()
    df = pd.DataFrame({"isonum": iso, "area": np.nan_to_num(d["pg_landarea"])})
    g = df[df.isonum > 0].groupby("isonum").agg(area_km2=("area", "sum"), n_cells=("area", "size"))
    return g.reset_index()


def neighbours() -> dict[int, set[int]]:
    """Países que comparten un borde de celda (anillo 1 de PRIO-GRID)."""
    from world4.conflict.grid import Grid
    d, iso, _ = _cells()
    g = Grid(d)
    A = g.ring1.tocoo()
    a, b = iso[A.row], iso[A.col]
    keep = (a > 0) & (b > 0) & (a != b)
    out: dict[int, set[int]] = {}
    for x, y in zip(a[keep], b[keep]):
        out.setdefault(int(x), set()).add(int(y))
    return out


def events_country_year() -> pd.DataFrame:
    """(isonum, year) -> eventos por tipo, muertes y celdas con eventos, 1989-2025."""
    d, iso, gw2iso = _cells()
    cell = d["ev_cell"]
    ev_iso = np.where(cell >= 0, iso[np.maximum(cell, 0)], [gw2iso.get(int(g), 0) for g in d["ev_country"]])
    year = 1989 + d["ev_month"] // 12
    df = pd.DataFrame({"isonum": ev_iso, "year": year, "type": d["ev_type"], "deaths": d["ev_deaths"], "cell": cell})
    df = df[(df.isonum > 0) & (df.year <= LAST_GED_YEAR)]
    out = df.groupby(["isonum", "year"]).agg(ev_all=("type", "size"), deaths_all=("deaths", "sum"),
                                             cells=("cell", lambda s: s[s >= 0].nunique()))
    for k, v in enumerate(TYPES):
        out[f"ev_{v}"] = df[df.type == k].groupby(["isonum", "year"]).size()
    out["deaths_sb"] = df[df.type == 0].groupby(["isonum", "year"]).deaths.sum()
    return out.fillna(0).reset_index()


# ---------------------------------------------------------------- World3 y mundo
@lru_cache(maxsize=1)
def world_series() -> pd.DataFrame:
    """Año -> la corrida estándar de World3-03 (escenario 1) y precios mundiales reales observados (Atlas)."""
    from world4.core import l0
    from world4.data import atlas
    r = l0.run(scenario=1)
    t = np.asarray(r.time)
    keep = np.isclose(t % 1, 0)
    w3 = pd.DataFrame({"year": t[keep].astype(int),
                       **{f"w3_{k}": np.asarray(r.series[k])[keep, 0] for k in ("iopc", "fpc", "nrfr", "ppolx", "le", "hwi")}})
    for k in ("iopc", "fpc", "ppolx"):
        w3[f"w3_{k}"] = np.log(w3[f"w3_{k}"])
    w3["w3_fpc_g"] = w3.w3_fpc.diff(5) / 5
    w3["w3_iopc_g"] = w3.w3_iopc.diff(5) / 5
    w = pd.read_csv(atlas.root() / "data" / "processed" / "world_full.csv")
    w = w[["year", "recursos__petroleo_precio_real_usd2019_bbl", "recursos__jacks_wheat", "recursos__jacks_corn",
           "recursos__jacks_rice"]].copy()
    w["oil_price"] = np.log(w.pop("recursos__petroleo_precio_real_usd2019_bbl"))
    w["cereal_price"] = np.log(w[["recursos__jacks_wheat", "recursos__jacks_corn", "recursos__jacks_rice"]]).mean(axis=1)
    w = w[["year", "oil_price", "cereal_price"]]
    w["oil_shock"] = w.oil_price.diff()
    w["cereal_shock"] = w.cereal_price.diff()
    return w3.merge(w, on="year", how="left")


# ---------------------------------------------------------------- el panel
def _slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


@lru_cache(maxsize=1)
def build() -> pd.DataFrame:
    """El panel completo con resultados y predictores (sin imputar: ``factors.prepare`` imputa por pliegue)."""
    from world4.data import atlas
    atlas.require()
    p = pd.read_csv(atlas.root() / "data" / "processed" / "panel_full.csv", low_memory=False)
    p = p[p.year >= FIRST_YEAR - 5].sort_values(["iso3", "year"])
    geo = country_geometry()
    countries = p[["iso3", "isonum"]].drop_duplicates().merge(geo, on="isonum")      # países con celdas
    extra = pd.DataFrame({"isonum": list(NOT_IN_ATLAS), "iso3": list(NOT_IN_ATLAS.values())}).merge(geo, on="isonum")
    countries = pd.concat([countries.assign(in_atlas=1), extra.assign(in_atlas=0)], ignore_index=True)
    years = np.arange(FIRST_YEAR - 5, LAST_GED_YEAR + 1)
    grid = countries.assign(_k=1).merge(pd.DataFrame({"year": years, "_k": 1}), on="_k").drop(columns="_k")
    # Atlas: hasta 2019; después, último valor (congelado)
    a = grid.merge(p.drop(columns=["isonum"]), on=["iso3", "year"], how="left")
    a = a.sort_values(["iso3", "year"])
    atlas_cols = [c for c in p.columns if c not in ("iso3", "isonum", "year", "country")]
    late = a.year > LAST_ATLAS_YEAR
    a.loc[late, atlas_cols] = np.nan
    a[atlas_cols] = a.groupby("iso3")[atlas_cols].ffill()
    a = a.copy()
    a["atlas_frozen"] = np.maximum(a.year - LAST_ATLAS_YEAR, 0)
    ev = events_country_year()
    a = a.merge(ev, on=["isonum", "year"], how="left")
    for c in ("ev_all", "ev_sb", "ev_ns", "ev_os", "deaths_all", "deaths_sb", "cells"):
        a[c] = a[c].fillna(0.0)
    a = a.merge(world_series(), on="year", how="left")
    return _features(a.reset_index(drop=True))


def _features(a: pd.DataFrame) -> pd.DataFrame:
    g = a.groupby("iso3")
    area10 = a.area_km2 / AREA_UNIT
    f = pd.DataFrame(index=a.index)
    # historia (GED y UCDP/PRIO)
    f["h_ev"] = np.log1p(a.ev_all)
    f["h_ev_l1"] = np.log1p(g.ev_all.shift(1))
    f["h_ev_5y"] = np.log1p(g.ev_all.transform(lambda s: s.rolling(5, min_periods=1).mean()))
    f["h_dens"] = np.log1p(a.ev_all / area10)
    f["h_deaths"] = np.log1p(a.deaths_all)
    last = a.year.where(a.ev_all > 0).groupby(a.iso3).ffill()
    f["h_since"] = np.where(last.notna(), 0.5 ** ((a.year - last) / 2.0), 0.0)
    f["h_extent"] = a.cells / a.n_cells
    for v in ("sb", "ns", "os"):
        f[f"h_{v}"] = np.log1p(a[f"ev_{v}"])
    nb = neighbours()
    iso_of = dict(zip(a.iso3, a.isonum))
    evmap = a.set_index(["isonum", "year"]).ev_all
    f["h_nb"] = np.log1p([sum(evmap.get((n, y), 0.0) for n in nb.get(int(iso_of[c]), ())) for c, y in zip(a.iso3, a.year)])
    f["h_acd"] = a.conflict
    # economía
    f["lny"] = np.log(a.gdppc)
    f["growth"] = g.gdppc.transform(lambda s: np.log(s).diff())
    f["infl"] = _slog(a.infl)
    f["debt"] = a.debt_gdp
    f["crisis"] = a.crisis_any
    f["res_rents"] = a.renta_total_pct_pib
    f["oil_rents"] = a.renta_petroleo_pct_pib
    f["trade"] = a.trade_gdp_wdi
    f["mineral_depl"] = a.agotamiento_minerales_pct_rnb
    # demografía
    f["ln_pop"] = np.log(a["pop"])
    f["pop_growth"] = a.pop_growth
    f["youth"] = a.youth_bulge_15_24_adult
    f["urban"] = a.urban_share
    f["urban_growth"] = a.urban_pop_growth
    f["tfr"] = a.tfr
    f["life_exp"] = a.life_expectancy
    f["dle3"] = g.life_expectancy.diff(3)
    f["median_age"] = a.median_age
    f["migration"] = a.net_migration_rate
    # instituciones
    f["polyarchy"] = a.v2x_polyarchy
    f["anocracy"] = a.v2x_polyarchy * (1 - a.v2x_polyarchy)
    f["corruption"] = a.vdem_corr
    f["rule_of_law"] = a.vdem_rule
    f["fiscal_cap"] = a.v2stfisccap
    f["state_cap"] = a.state_capacity_index
    f["excluded"] = a.epr_excluded_share
    f["discriminated"] = a.epr_discriminated_share
    f["coups10"] = a.coups_past10
    f["regime_change"] = a.regime_change
    f["leader_tenure"] = np.log1p(a.leader_tenure)
    f["state_age"] = np.log1p(a.years_since_indep.clip(lower=0))
    f["excl_soc"] = a.v2xpe_exlsocgr
    # desigualdad
    f["gini"] = a.gini_disp
    f["top10"] = a.top10_share_ptinc
    # sociedad y geografía
    f["rugged"] = a.rugged
    f["mountains"] = a.log_mountainous
    f["ethfrac"] = a.creg_ethfrac
    f["al_ethnic"] = a.al_ethnic
    f["cohesion"] = a.cohesion_index
    f["colony"] = a.former_colony
    f["indep_violent"] = a.indep_violent
    f["n_neighbors"] = a.n_land_neighbors
    f["ln_area"] = np.log(a.area_km2)
    # geopolítica
    f["mids"] = np.log1p(a.mid_n)
    f["rivalries"] = a.n_rivalries
    f["major_power"] = a.major_power
    f["oda"] = _slog(a.oda_gni)
    f["sanctions"] = np.log1p(a.sanctions_n)
    f["milex"] = a.mil_exp_wdi
    f["foreign_interv"] = a.civil_war_foreign_intervened
    f["unga_us"] = a.unga_dist_us
    f["nb_democracy"] = a.neighbor_dem_share
    f["ln_cinc"] = a.log_cinc
    # clima y ambiente
    f["temp_anom"] = a.tmp_anom_fao
    f["disasters"] = np.log1p(a.dis_n_climate)
    f["water_stress"] = _slog(a.water_stress_pct)
    lyld = np.log(a.cereal_yield)
    f["ln_yield"] = lyld
    f["yield_shock"] = lyld - lyld.groupby(a.iso3).transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    f["pm25"] = a.pm25
    f["eco_deficit"] = a.eco_deficit_pc
    f["vulnerability"] = a.ndgain_vulnerability
    f["ln_co2pc"] = np.log(a.co2_pc.clip(lower=1e-3))
    f["forest"] = a.forest_pct
    # mundo: World3 y precios observados; interacciones con la exposición del país
    for c in ("w3_iopc", "w3_fpc", "w3_nrfr", "w3_ppolx", "w3_le", "w3_hwi", "w3_fpc_g", "w3_iopc_g", "oil_price",
              "cereal_price"):
        f[c] = a[c]
    poor = (f.lny < 8.0).astype(float)                 # ~3000 USD 2017 por persona
    exporter = (a.renta_petroleo_pct_pib > 5).astype(float)
    f["w3_fpc_g_x_poor"] = a.w3_fpc_g * poor
    f["w3_nrfr_x_rents"] = a.w3_nrfr * a.renta_total_pct_pib
    f["oil_shock_x_exporter"] = a.oil_shock * exporter
    f["cereal_shock_x_poor"] = a.cereal_shock * poor
    f["year_trend"] = a.year - 2000.0
    f["democ"] = np.where(a.regime_row.notna(), (a.regime_row >= 2).astype(float), np.nan)
    f = f.replace([np.inf, -np.inf], np.nan)
    keep = ["iso3", "isonum", "year", "area_km2", "n_cells", "in_atlas", "atlas_frozen", "ev_all", "ev_sb", "ev_ns", "ev_os",
            "deaths_all", "deaths_sb", "cells"]
    out = pd.concat([a[keep], f[ALL + list(EXTRA)]], axis=1)
    return out[out.year >= FIRST_YEAR].reset_index(drop=True)


def with_targets(df: pd.DataFrame, h: int, outcome: str = "ev_all") -> pd.DataFrame:
    """Añade ``y`` = el resultado en t + h (NaN si ese año no existe)."""
    nxt = df[["iso3", "year", outcome]].copy()
    nxt["year"] = nxt.year - h
    return df.merge(nxt.rename(columns={outcome: "y"}), on=["iso3", "year"], how="left")
