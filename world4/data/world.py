"""Series mundiales observadas para medir el L0 (World3) contra la realidad.

El L0 es un modelo del mundo agregado: su verdad son series MUNDIALES, no el panel de países del Atlas.
Este módulo las instala, las comprueba contra ``world_manifest.json`` (SHA-256) y las convierte una vez a
``data_cache/world/world_obs.npz``. Nada de terceros viaja en el repo.

| Serie              | Unidad                            | Años      | Fuente                                         |
|--------------------|-----------------------------------|-----------|------------------------------------------------|
| ``pop``            | millones de personas (1 de julio) | 1900-2025 | OWID co2-data (HYDE, Gapminder, ONU)           |
| ``le``             | años, ambos sexos                 | 1950-2023 | ONU WPP 2024 (``e01dt``)                       |
| ``tfr``            | hijos por mujer                   | 1950-2023 | ONU WPP 2024 (``tfr1dt``)                      |
| ``cbr``, ``cdr``   | por 1000 habitantes               | 1950-2023 | ONU WPP 2024 (``misc1dt``)                     |
| ``births``, ``deaths`` | millones por año              | 1950-2023 | ONU WPP 2024 (``misc1dt``)                     |
| ``gdp``            | 1e9 dólares internacionales 2011  | sueltos   | Maddison Project vía OWID co2-data             |
| ``fossil_co2``     | Mt CO2/año (carbón+petróleo+gas+antorcha) | 1900-2025 | Global Carbon Budget vía OWID          |
| ``primary_energy`` | TWh/año                           | 1965-2024 | Energy Institute / EIA vía OWID                |
| ``co2_fi``, ``luc_co2`` | Mt CO2/año: fósil e industria (con cemento); cambio de uso del suelo | 1900-2025, 1900-2024 | Global Carbon Budget vía OWID |
| ``cum_co2_fi``, ``cum_luc_co2`` | Mt CO2 acumuladas desde 1750 y 1850 | 1900-2025, 1900-2024 | Global Carbon Budget vía OWID |
| ``co2_ppm``        | ppm, media anual en Mauna Loa     | 1959-2025 | NOAA GML (espejo ``datasets/co2-ppm``)         |
| ``gmst``           | °C, anomalía global sobre 1951-1980 | 1900-2024 | NASA GISTEMP v4 (espejo ``datasets/global-temp``) |
| ``kcal``           | kcal por persona y día            | 1961-2013 | FAO Food Balance Sheets vía owid-datasets      |
| ``food_veq``       | kcal en equivalente vegetal por persona y día | 1961-2013 | calculada con la misma hoja de FAO  |
| ``cereal_yield``   | índice, 1961 = 100                | 1961-2014 | Banco Mundial WDI vía owid-datasets            |
| ``cropland``       | millones de hectáreas             | 1900-2016 | HYDE 3.2 vía owid-datasets (decenal hasta 2000)|

Además, del mundo (código ONU 900) y para el modelo de cohortes de ``world4.core.l0v``: el patrón de
fecundidad por edad por quinquenio (``asfr7``, ``asfr_periods``), la razón de sexos al nacer (``srb``) y las
tasas de mortalidad por edad y sexo por quinquenio (``mxM``, ``mxF``, ``mx_periods``; grupos de ``wpp.MX_GROUPS``).
Salen de los ficheros crudos de ``world4.data.wpp`` (``--install-wpp``), que ``install`` trae si faltan.

**Recursos fósiles** (para ``world4.core.l0v.resources``; se leen crudos, son pequeños): las curvas de oferta de
GCAM (``gcam-fos-curves.csv``: EJ disponibles y coste de extracción por región, combustible y grado, derivadas de
Rogner 1997), sus coeficientes de carbono (``gcam-primary-fuel-ccoef.csv``), la relación reservas/producción de
BP 2016 y la producción por combustible. En ``world_obs.npz`` van además las emisiones acumuladas por
combustible (``cum_coal_co2``, ``cum_oil_co2``, ``cum_gas_co2``) y las anuales (``coal_co2``, ``oil_co2``, ``gas_co2``).

**Panel de países** (``world_countries.npz``, para los grupos de países de ``world4.core.l0v``): por país de
WPP (236) y año 1950-2023, fecundidad total, esperanza de vida, nacimientos y muertes (WPP), PIB (Maddison vía
OWID, 164 países, 95-99 % de la población mundial) y kilocalorías por persona (FAO, 1961-2013), con el código
ISO3 y la subregión de la ONU (``UNlocations.txt``). Los huecos se dejan como NaN: quien agrupa decide.
La conversión necesita ``pycountry`` además de ``rdata`` (extra ``world4[world]``).

``food_veq`` es el alimento como lo cuenta World3: cada kilocaloría de origen animal vale ``ANIMAL_FACTOR`` (7)
vegetales. Las kilocalorías animales no vienen en la hoja de FAO que se descarga; se calculan con los kilos por
persona de carne, leche, huevos y pescado y un contenido energético medio de cada uno (``KCAL_PER_KG``) [S]: dan
el 13 % de las kilocalorías en 1961 y el 16 % en 2013.

El PIB mundial de Maddison sólo trae años sueltos antes de 2015 (1950, 1960, ..., 2010) [declarado]: las series
anuales del Banco Mundial y FAOSTAT posterior a 2013 quedan fuera hasta poder descargarlas.

Convención temporal: las series anuales son del año natural (o del 1 de julio); el L0 las compara en ``t = año``.
"""
from __future__ import annotations

import hashlib
import json
import urllib.parse
import urllib.request
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

WPP_COMMIT = "2da7768ae64fc74105d3f9e98f9a74d37b62f99a"
CO2_COMMIT = "22d9a01321ddec951b36f2bfc66e05dc6ad50ad8"
MLO_COMMIT = "52f2be52ee6145eaad67367959af9b1b0a0bd83e"
TEMP_COMMIT = "b90e8a349acb508d382d40d1cfa393c723bd82ef"
GCAM_COMMIT = "11e128fb7ce3e14e9c4daf3903ba73123046a7aa"
OWIDDS_COMMIT = "6155d4ca1ea14ef30e753010a25521eeb416e8a2"
_RAW = "https://raw.githubusercontent.com/"
_GCAM = f"{_RAW}JGCRI/gcam-core/{GCAM_COMMIT}/input/gcamdata/inst/extdata/"
_FOOD = "Food supply (FAO, 2020)"
_YIELD = "Cereal yield index - (World Bank (2017) & OWID)"
_LAND = "Land Use Data - HYDE (2017)"
_RP = "Reserves-Production Ratio - BP Statistics (2016)"
_PROD = "Fossil fuel production BP & Shift (2022)"


def _owidds(name: str) -> str:
    q = urllib.parse.quote(name)
    return f"{_RAW}owid/owid-datasets/{OWIDDS_COMMIT}/datasets/{q}/{q}.csv"


URLS = {
    "e01dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/e01dt.rda",
    "tfr1dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/tfr1dt.rda",
    "misc1dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/misc1dt.rda",
    "owid-co2-data.csv": f"{_RAW}owid/co2-data/{CO2_COMMIT}/owid-co2-data.csv",
    "co2-annmean-mlo.csv": f"{_RAW}datasets/co2-ppm/{MLO_COMMIT}/data/co2-annmean-mlo.csv",
    "global-temp-annual.csv": f"{_RAW}datasets/global-temp/{TEMP_COMMIT}/data/annual.csv",
    "food-supply-fao-2020.csv": _owidds(_FOOD),
    "cereal-yield-index.csv": _owidds(_YIELD),
    "land-use-hyde-2017.csv": _owidds(_LAND),
    "bp-reserves-production-ratio-2016.csv": _owidds(_RP),
    "fossil-fuel-production-2022.csv": _owidds(_PROD),
    "gcam-fos-curves.csv": _GCAM + "energy/A11.fos_curves.csv",
    "gcam-primary-fuel-ccoef.csv": _GCAM + "emissions/A_PrimaryFuelCCoef.csv",
    "UNlocations.txt": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/UNlocations.txt",
}
# nombres de OWID/FAO que no están en la tabla de países de co2-data, y el código de Kosovo (sin ISO oficial)
FAO_ALIASES = {"Czech Republic": "CZE", "Macedonia": "MKD", "Swaziland": "SWZ", "Sudan (former)": "SDN",
               "Ethiopia PDR": "ETH"}
ISO_EXTRA = {412: "XKX"}
OWID_ISO_FIX = {"OWID_KOS": "XKX"}
MANIFEST = Path(__file__).with_name("world_manifest.json")
ANIMAL_FACTOR = 7.0                           # kg vegetales por kg de producto animal en World3
KCAL_PER_KG = {"Meat food supply quantity (kg/capita/yr)": 2000.0,
               "Milk - Excluding Butter - Food supply quantity (kg/capita/yr)": 600.0,
               "Eggs - Food supply quantity (kg/capita/yr)": 1450.0,
               "Fish, Seafood- Food supply quantity (kg/capita/yr)": 1000.0}
YEAR0, YEAR1 = 1900, 2025
WORLD = 900                                   # código de la ONU para el agregado mundial
GMST_LAST = 2024                              # el espejo trae años en curso: sólo años completos
SERIES = ("pop", "le", "tfr", "cbr", "cdr", "births", "deaths", "gdp", "fossil_co2", "primary_energy", "co2_fi", "luc_co2",
          "cum_co2_fi", "cum_luc_co2", "coal_co2", "oil_co2", "gas_co2", "cum_coal_co2", "cum_oil_co2", "cum_gas_co2",
          "co2_ppm", "gmst", "kcal",
          "food_veq", "cereal_yield", "cropland")
UNITS = {"pop": "million persons", "le": "years", "tfr": "children per woman", "cbr": "per 1000", "cdr": "per 1000",
         "births": "million per year", "deaths": "million per year",
         "gdp": "1e9 international-$ 2011", "fossil_co2": "Mt CO2 per year", "primary_energy": "TWh per year",
         "co2_ppm": "ppm", "kcal": "kcal per person per day", "cereal_yield": "index, 1961 = 100",
         "cropland": "million hectares", "food_veq": "vegetal-equivalent kcal per person per day",
         "co2_fi": "Mt CO2 per year", "luc_co2": "Mt CO2 per year", "cum_co2_fi": "Mt CO2", "cum_luc_co2": "Mt CO2",
         "gmst": "degC above 1951-1980",
         **{k: "Mt CO2 per year" for k in ("coal_co2", "oil_co2", "gas_co2")},
         **{k: "Mt CO2" for k in ("cum_coal_co2", "cum_oil_co2", "cum_gas_co2")}}
SOURCE_OF = {"pop": "owid_co2_world", "le": "wpp", "tfr": "wpp", "cbr": "wpp", "cdr": "wpp", "births": "wpp", "deaths": "wpp", "gdp": "owid_co2_world",
             "fossil_co2": "owid_co2_world", "primary_energy": "owid_co2_world", "co2_ppm": "noaa_mlo",
             "kcal": "owid_food_supply", "food_veq": "owid_food_supply", "cereal_yield": "owid_cereal_yield", "cropland": "owid_hyde_land",
             "co2_fi": "owid_co2_world", "luc_co2": "owid_co2_world", "cum_co2_fi": "owid_co2_world",
             "cum_luc_co2": "owid_co2_world", "gmst": "gistemp",
             **{k: "owid_co2_world" for k in ("coal_co2", "oil_co2", "gas_co2", "cum_coal_co2", "cum_oil_co2",
                                              "cum_gas_co2")}}


class WorldDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "world"


def npz_path() -> Path:
    return root() / "world_obs.npz"


def countries_path() -> Path:
    return root() / "world_countries.npz"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    raw = root() / "raw"
    missing, bad = [], []
    for name, meta in manifest()["files"].items():
        p = raw / name
        if not p.is_file():
            missing.append(name)
        elif _sha(p) != meta["sha256"]:
            bad.append(name)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(raw)}


def install(log=print) -> dict:
    """Descarga lo que falte (también los ficheros por edad de WPP), verifica SHA-256 y escribe ``world_obs.npz``."""
    from world4.data import wpp
    if not wpp.available() or not wpp.verify_raw()["ok"]:
        log("WPP 2024 by age and sex is needed too: installing it")
        wpp.install(log=log)
    raw = root() / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        if not (raw / name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(url, raw / name)
    v = verify_raw()
    if not v["ok"]:
        raise WorldDataError(f"world files missing {v['missing']} or with another SHA-256 {v['mismatched']}")
    convert()
    convert_countries()
    load.cache_clear()
    load_countries.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path()), "countries_npz": str(countries_path())}


def _rda(path: Path):
    import pandas as pd
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(path)
    df = pd.DataFrame(obj[list(obj)[0]]).reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    df["country_code"] = pd.to_numeric(df.country_code).astype(int)
    df["year"] = df.year.astype(int)
    return df


def _rda_world(path: Path):
    """Las filas del mundo (código 900) de un ``.rda`` de WPP, indexadas por año."""
    df = _rda(path)
    return df[df.country_code == WORLD].copy()


def convert() -> Path:
    import pandas as pd
    from world4.data import wpp
    raw = root() / "raw"
    years = np.arange(YEAR0, YEAR1 + 1)
    out = {k: np.full(len(years), np.nan) for k in SERIES}

    def put(name, yr, val, scale=1.0):
        yr, val = np.asarray(yr, int), np.asarray(val, float) * scale
        ok = (yr >= YEAR0) & (yr <= YEAR1) & np.isfinite(val)
        out[name][yr[ok] - YEAR0] = val[ok]

    for name, fn, col, scale in (("le", "e01dt.rda", "e0B", 1.0), ("tfr", "tfr1dt.rda", "tfr", 1.0),
                                 ("cbr", "misc1dt.rda", "cbr", 1.0), ("cdr", "misc1dt.rda", "cdr", 1.0),
                                 ("births", "misc1dt.rda", "births", 1e-3), ("deaths", "misc1dt.rda", "deaths", 1e-3)):
        d = _rda_world(raw / fn)
        d = d[d.year <= wpp.YEAR1]                               # estimaciones; lo demás es proyección
        put(name, d.year, d[col], scale)

    w = pd.read_csv(raw / "owid-co2-data.csv")
    w = w[w.country == "World"]
    put("pop", w.year, w.population, 1e-6)
    put("gdp", w.year, w.gdp, 1e-9)
    put("primary_energy", w.year, w.primary_energy_consumption)
    fossil = w[["coal_co2", "oil_co2", "gas_co2", "flaring_co2"]].fillna(0.0).sum(axis=1)
    put("fossil_co2", w.year[fossil > 0], fossil[fossil > 0])
    for name, col in (("co2_fi", "co2"), ("luc_co2", "land_use_change_co2"), ("cum_co2_fi", "cumulative_co2"),
                      ("cum_luc_co2", "cumulative_luc_co2"), ("coal_co2", "coal_co2"), ("oil_co2", "oil_co2"),
                      ("gas_co2", "gas_co2"), ("cum_coal_co2", "cumulative_coal_co2"),
                      ("cum_oil_co2", "cumulative_oil_co2"), ("cum_gas_co2", "cumulative_gas_co2")):
        put(name, w.year, w[col])
    g = pd.read_csv(raw / "global-temp-annual.csv")
    g = g[(g.Source == "GISTEMP") & (g.Year <= GMST_LAST)]
    put("gmst", g.Year, g.Mean)

    m = pd.read_csv(raw / "co2-annmean-mlo.csv")
    put("co2_ppm", m["Year"], m["Mean"])
    f = pd.read_csv(raw / "food-supply-fao-2020.csv")
    f = f[f.Entity == "World"]
    put("kcal", f.Year, f.iloc[:, 2])
    animal = sum(f[col] * k for col, k in KCAL_PER_KG.items()) / 365.0
    put("food_veq", f.Year, f.iloc[:, 2] + (ANIMAL_FACTOR - 1) * animal)
    c = pd.read_csv(raw / "cereal-yield-index.csv")
    c = c[c.Entity == "World"]
    put("cereal_yield", c.Year, c.iloc[:, 2])
    h = pd.read_csv(raw / "land-use-hyde-2017.csv", usecols=[0, 1, 2])
    h = h[h.Entity == "World"]
    put("cropland", h.Year, h.iloc[:, 2], 1e-6)

    wraw = wpp.root() / "raw"
    fa = _rda_world(wraw / "percentASFR5dt.rda")
    periods = np.array(sorted(p for p in fa.year.unique() if p + 2 <= wpp.YEAR1))
    asfr7 = np.array([[float(fa[(fa.year == p) & (fa.age.astype(str) == g)].pasfr.iloc[0]) for g in wpp.ASFR_GROUPS]
                      for p in periods])
    sr = _rda_world(wraw / "sexRatio1dt.rda")
    sr = sr[sr.year.between(wpp.YEAR0, wpp.YEAR1)].sort_values("year")
    mx = _rda_world(wraw / "mx5dt.rda")
    mx["age"] = mx.age.astype(str)
    mxw = {s: np.array([[float(mx[(mx.year == p) & (mx.age == g)][s].iloc[0]) for g in wpp.MX_GROUPS]
                        for p in periods]) for s in ("mxM", "mxF")}

    p = npz_path()
    np.savez_compressed(p, years=years, asfr_periods=periods, asfr7=asfr7, srb_years=sr.year.to_numpy(),
                        srb=sr.srb.to_numpy(float), mx_periods=periods, mxM=mxw["mxM"], mxF=mxw["mxF"], **out)
    return p


def convert_countries() -> Path:
    """El panel de países: ``(país, año)`` con los países en el orden de ``wpp.load()["codes"]``."""
    import pandas as pd
    import pycountry
    from world4.data import wpp
    raw = root() / "raw"
    codes = wpp.load()["codes"].astype(int)
    years = np.arange(wpp.YEAR0, wpp.YEAR1 + 1)
    ci = {int(c): i for i, c in enumerate(codes)}
    C, Y = len(codes), len(years)
    num2iso = {int(c.numeric): c.alpha_3 for c in pycountry.countries}
    num2iso.update(ISO_EXTRA)
    iso3 = np.array([num2iso.get(int(c), "") for c in codes])
    iso2i = {s: i for i, s in enumerate(iso3) if s}

    def grid():
        return np.full((C, Y), np.nan)

    def fill(arr, rows, yr, val, scale=1.0):
        rows, yr, val = np.asarray(rows), np.asarray(yr, int), np.asarray(val, float) * scale
        ok = (rows >= 0) & (yr >= years[0]) & (yr <= years[-1]) & np.isfinite(val)
        arr[rows[ok], yr[ok] - years[0]] = val[ok]

    out = {k: grid() for k in ("tfr", "e0", "births", "deaths", "gdp", "kcal")}
    for name, fn, col, scale in (("e0", "e01dt.rda", "e0B", 1.0), ("tfr", "tfr1dt.rda", "tfr", 1.0),
                                 ("births", "misc1dt.rda", "births", 1e3), ("deaths", "misc1dt.rda", "deaths", 1e3)):
        d = _rda(raw / fn)
        fill(out[name], d.country_code.map(ci).fillna(-1).astype(int), d.year, d[col], scale)

    o = pd.read_csv(raw / "owid-co2-data.csv", usecols=["country", "year", "iso_code", "gdp"])
    o["iso_code"] = o.iso_code.replace(OWID_ISO_FIX)
    name2iso = {**dict(zip(o.country, o.iso_code)), **FAO_ALIASES}
    g = o[o.gdp.notna() & o.iso_code.notna()]
    fill(out["gdp"], g.iso_code.map(iso2i).fillna(-1).astype(int), g.year, g.gdp)
    f = pd.read_csv(raw / "food-supply-fao-2020.csv", usecols=[0, 1, 2])
    f.columns = ["entity", "year", "kcal"]
    fill(out["kcal"], f.entity.map(name2iso).map(iso2i).fillna(-1).astype(int), f.year, f.kcal)

    u = pd.read_csv(raw / "UNlocations.txt", sep="\t")
    u = u[u.location_type == 4]
    sub = {int(c): str(r) for c, r in zip(u.country_code, u.reg_name)}
    area = {int(c): str(r) for c, r in zip(u.country_code, u.area_name)}
    p = countries_path()
    np.savez_compressed(p, codes=codes, iso3=iso3, years=years,
                        subregion=np.array([sub.get(int(c), "") for c in codes]),
                        area=np.array([area.get(int(c), "") for c in codes]), **out)
    return p


@lru_cache(maxsize=1)
def load_countries() -> dict:
    p = countries_path()
    if not p.is_file():
        raise WorldDataError(f"no country panel at {p}: python world4_cli.py --install-world")
    z = np.load(p)
    return {k: z[k] for k in z.files}


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise WorldDataError(f"no world series at {p}: python world4_cli.py --install-world")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file() and countries_path().is_file()


def series(name: str):
    """``(años, valores)`` de una serie, sólo donde hay dato."""
    d = load()
    ok = np.isfinite(d[name])
    return d["years"][ok], d[name][ok]
