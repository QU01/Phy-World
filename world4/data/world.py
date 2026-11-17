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
| ``co2_ppm``        | ppm, media anual en Mauna Loa     | 1959-2025 | NOAA GML (espejo ``datasets/co2-ppm``)         |
| ``kcal``           | kcal por persona y día            | 1961-2013 | FAO Food Balance Sheets vía owid-datasets      |
| ``cereal_yield``   | índice, 1961 = 100                | 1961-2014 | Banco Mundial WDI vía owid-datasets            |

Además, del mundo (código ONU 900) y para el modelo de cohortes de ``world4.core.l0v``: el patrón de
fecundidad por edad por quinquenio (``asfr7``, ``asfr_periods``), la razón de sexos al nacer (``srb``) y las
tasas de mortalidad por edad y sexo por quinquenio (``mxM``, ``mxF``, ``mx_periods``; grupos de ``wpp.MX_GROUPS``).
Salen de los ficheros crudos de ``world4.data.wpp`` (``--install-wpp``), que ``install`` trae si faltan.

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
OWIDDS_COMMIT = "6155d4ca1ea14ef30e753010a25521eeb416e8a2"
_RAW = "https://raw.githubusercontent.com/"
_FOOD = "Food supply (FAO, 2020)"
_YIELD = "Cereal yield index - (World Bank (2017) & OWID)"


def _owidds(name: str) -> str:
    q = urllib.parse.quote(name)
    return f"{_RAW}owid/owid-datasets/{OWIDDS_COMMIT}/datasets/{q}/{q}.csv"


URLS = {
    "e01dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/e01dt.rda",
    "tfr1dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/tfr1dt.rda",
    "misc1dt.rda": f"{_RAW}PPgp/wpp2024/{WPP_COMMIT}/data/misc1dt.rda",
    "owid-co2-data.csv": f"{_RAW}owid/co2-data/{CO2_COMMIT}/owid-co2-data.csv",
    "co2-annmean-mlo.csv": f"{_RAW}datasets/co2-ppm/{MLO_COMMIT}/data/co2-annmean-mlo.csv",
    "food-supply-fao-2020.csv": _owidds(_FOOD),
    "cereal-yield-index.csv": _owidds(_YIELD),
}
MANIFEST = Path(__file__).with_name("world_manifest.json")
YEAR0, YEAR1 = 1900, 2025
WORLD = 900                                   # código de la ONU para el agregado mundial
SERIES = ("pop", "le", "tfr", "cbr", "cdr", "births", "deaths", "gdp", "fossil_co2", "primary_energy", "co2_ppm", "kcal",
          "cereal_yield")
UNITS = {"pop": "million persons", "le": "years", "tfr": "children per woman", "cbr": "per 1000", "cdr": "per 1000",
         "births": "million per year", "deaths": "million per year",
         "gdp": "1e9 international-$ 2011", "fossil_co2": "Mt CO2 per year", "primary_energy": "TWh per year",
         "co2_ppm": "ppm", "kcal": "kcal per person per day", "cereal_yield": "index, 1961 = 100"}
SOURCE_OF = {"pop": "owid_co2_world", "le": "wpp", "tfr": "wpp", "cbr": "wpp", "cdr": "wpp", "births": "wpp", "deaths": "wpp", "gdp": "owid_co2_world",
             "fossil_co2": "owid_co2_world", "primary_energy": "owid_co2_world", "co2_ppm": "noaa_mlo",
             "kcal": "owid_food_supply", "cereal_yield": "owid_cereal_yield"}


class WorldDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "world"


def npz_path() -> Path:
    return root() / "world_obs.npz"


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
    load.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path())}


def _rda_world(path: Path):
    """Las filas del mundo (código 900) de un ``.rda`` de WPP, indexadas por año."""
    import pandas as pd
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(path)
    df = pd.DataFrame(obj[list(obj)[0]]).reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    df = df[pd.to_numeric(df.country_code) == WORLD].copy()
    df["year"] = df.year.astype(int)
    return df


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

    m = pd.read_csv(raw / "co2-annmean-mlo.csv")
    put("co2_ppm", m["Year"], m["Mean"])
    f = pd.read_csv(raw / "food-supply-fao-2020.csv")
    f = f[f.Entity == "World"]
    put("kcal", f.Year, f.iloc[:, 2])
    c = pd.read_csv(raw / "cereal-yield-index.csv")
    c = c[c.Entity == "World"]
    put("cereal_yield", c.Year, c.iloc[:, 2])

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


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise WorldDataError(f"no world series at {p}: python world4_cli.py --install-world")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file()


def series(name: str):
    """``(años, valores)`` de una serie, sólo donde hay dato."""
    d = load()
    ok = np.isfinite(d[name])
    return d["years"][ok], d[name][ok]
