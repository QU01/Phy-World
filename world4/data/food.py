"""Datos de F5: alimentos, suelo, nitrógeno y agua.

* **FAOSTAT** (FAO, descarga masiva; CC BY 4.0 con los términos adicionales de la base estadística de
  FAO, leídos en la sesión de F5: no usar los datos para promocionar una empresa). Conjuntos: QCL
  (cereales: producción, superficie, rendimiento), RL (uso del suelo: cultivo, pastos, bosque,
  superficie de riego), RFN (N mineral), ESB (balance de nutrientes de las tierras de cultivo), GT
  (emisiones agrícolas y de conversión de bosques), FBSH + FBS (suministro de calorías por persona).
* **WDI** (Banco Mundial, CC BY 4.0, a partir de AQUASTAT): extracción de agua dulce total y la parte
  agrícola (ER.H2O.FWTL.K3, ER.H2O.FWAG.ZS).

FAOSTAT revisa su historia y los ficheros masivos se reemplazan sin versión: el manifiesto fija el
SHA-256 de los ficheros descargados el 2026-10-12 (con la fecha de actualización de cada conjunto),
y una descarga con otro SHA es otra *vintage* y se rechaza.

Empalme de calorías: FBSH (1961-2013, metodología antigua) y FBS (2010-). Para cada país, la serie
antigua se reescala por el cociente medio FBS/FBSH de 2010-2013 [declarado]; desde 2010 se usa FBS.

Países: por código M49 (``Area Code (M49)``), que es el ``isonum`` del L1. Los Estados desaparecidos
(URSS, Yugoslavia, Checoslovaquia, Etiopía PDR, Sudán antiguo...) no tienen país en el L1: sus
sucesores entran cuando FAOSTAT empieza a darlos (1992-2012).
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

FAO = "https://bulks-faostat.fao.org/production/{}"
WDI = "https://api.worldbank.org/v2/country/all/indicator/{}?format=json&per_page=20000&date=1960:2024"
FAO_FILES = {
    "QCL": "Production_Crops_Livestock_E_All_Data_(Normalized).zip",
    "RL": "Inputs_LandUse_E_All_Data_(Normalized).zip",
    "RFN": "Inputs_FertilizersNutrient_E_All_Data_(Normalized).zip",
    "ESB": "Environment_Cropland_nutrient_budget_E_All_Data_(Normalized).zip",
    "GT": "Emissions_Totals_E_All_Data_(Normalized).zip",
    "FBSH": "FoodBalanceSheetsHistoric_E_All_Data_(Normalized).zip",
    "FBS": "FoodBalanceSheets_E_All_Data_(Normalized).zip",
}
WDI_FILES = {"ER.H2O.FWAG.ZS": "wdi_ER.H2O.FWAG.ZS.json", "ER.H2O.FWTL.K3": "wdi_ER.H2O.FWTL.K3.json"}
MANIFEST = Path(__file__).with_name("food_manifest.json")
YEAR0, YEAR1 = 1961, 2024
# (conjunto, código de producto, código de elemento) -> serie del panel
SERIES = {
    "cereal_prod": ("QCL", 1717, 5510),          # t
    "cereal_area": ("QCL", 1717, 5312),          # ha
    "land": ("RL", 6601, 5110),                  # 1000 ha
    "cropland": ("RL", 6620, 5110),
    "pasture": ("RL", 6655, 5110),
    "forest": ("RL", 6646, 5110),                # desde 1990 (FRA)
    "irrigated": ("RL", 6690, 5110),             # equipada para riego
    "n_fert": ("RFN", 3102, 5157),               # t N, uso agrícola
    "nb_mineral": ("ESB", 5087, 7275),           # t N en tierras de cultivo
    "nb_manure": ("ESB", 5089, 7275),
    "nb_fixation": ("ESB", 5078, 7275),
    "nb_deposition": ("ESB", 5076, 7275),
    "nb_input": ("ESB", 5079, 7275),
    "nb_removal": ("ESB", 5077, 7275),
    "nb_residue": ("ESB", 5088, 7275),
    "nb_output": ("ESB", 5080, 7275),
    "nb_balance": ("ESB", 5081, 7275),
    "ch4_farm": ("GT", 6996, 7225),              # kt CH4, explotación agrícola (farm gate)
    "n2o_farm": ("GT", 6996, 7230),              # kt N2O
    "co2_forest_conv": ("GT", 6750, 7273),       # kt CO2, conversión neta de bosque (1990-)
}
KCAL = {"FBSH": (2901, 664), "FBS": (2901, 664)}
WORLD_M49 = 1                                    # "World" en FAOSTAT


class FoodDataError(RuntimeError):
    pass


def root() -> Path:
    from world4.data import atlas
    return atlas.root().parent / "food"


def raw(name: str) -> Path:
    return root() / "raw" / name


def npz_path() -> Path:
    return root() / "food.npz"


def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_raw() -> dict:
    missing, bad = [], []
    for name, meta in manifest()["files"].items():
        p = raw(name)
        if not p.is_file():
            missing.append(name)
        elif _sha(p) != meta["sha256"]:
            bad.append(name)
    return {"ok": not missing and not bad, "missing": missing, "mismatched": bad, "root": str(root() / "raw")}


def install(log=print) -> dict:
    (root() / "raw").mkdir(parents=True, exist_ok=True)
    urls = {**{f: FAO.format(f) for f in FAO_FILES.values()}, **{f: WDI.format(i) for i, f in WDI_FILES.items()}}
    for name in manifest()["files"]:
        if not raw(name).is_file():
            log(f"downloading {name}")
            urllib.request.urlretrieve(urls[name], raw(name))
    v = verify_raw()
    if not v["ok"]:
        raise FoodDataError(f"food files missing {v['missing']} or with another SHA-256 {v['mismatched']} (FAOSTAT "
                            "replaces its bulk files: a different SHA is a new vintage, see food_manifest.json)")
    convert()
    load.cache_clear()
    return {**v, "npz": str(npz_path()), "npz_sha256": _sha(npz_path())}


def _read(code: str, items: set):
    import pandas as pd
    z = zipfile.ZipFile(raw(FAO_FILES[code]))
    main = [f for f in z.namelist() if f.endswith("(Normalized).csv")][0]
    cols = ["Area Code (M49)", "Item Code", "Element Code", "Year", "Value"]
    out = []
    for c in pd.read_csv(z.open(main), encoding="latin1", usecols=cols, chunksize=2_000_000, low_memory=False):
        keep = c[[(i, e) in items for i, e in zip(c["Item Code"], c["Element Code"])]]
        out.append(keep)
    d = pd.concat(out)
    d["m49"] = pd.to_numeric(d["Area Code (M49)"].astype(str).str.lstrip("'"), errors="coerce")
    return d[d.m49.notna() & d.Year.between(YEAR0, YEAR1)]


def convert() -> Path:
    years = np.arange(YEAR0, YEAR1 + 1)
    frames = {}
    for code in set(v[0] for v in SERIES.values()) | set(KCAL):
        items = {(i, e) for k, (c, i, e) in SERIES.items() if c == code}
        if code in KCAL:
            items.add(KCAL[code])
        frames[code] = _read(code, items)
    codes = sorted(set(int(x) for d in frames.values() for x in d.m49.unique()))
    pos = {c: k for k, c in enumerate(codes)}
    C, T = len(codes), len(years)

    def grid(code, item, elem):
        d = frames[code]
        d = d[(d["Item Code"] == item) & (d["Element Code"] == elem)]
        g = np.full((C, T), np.nan)
        g[[pos[int(m)] for m in d.m49], d.Year.to_numpy() - YEAR0] = d.Value.to_numpy(float)
        return g

    arr = {k: grid(*v) for k, v in SERIES.items()}
    old, new = grid("FBSH", *KCAL["FBSH"]), grid("FBS", *KCAL["FBS"])
    ov = (years >= 2010) & (years <= 2013)
    ratio = np.nanmean(np.where(np.isfinite(old[:, ov]) & np.isfinite(new[:, ov]), new[:, ov] / old[:, ov], np.nan), 1)
    ratio = np.where(np.isfinite(ratio), ratio, 1.0)
    kcal = np.where(years[None] >= 2010, new, old * ratio[:, None])
    kcal = np.where(np.isfinite(kcal), kcal, np.where(years[None] < 2010, np.nan, old * ratio[:, None]))
    arr["kcal"] = kcal
    arr["kcal_splice_ratio"] = np.repeat(ratio[:, None], T, 1)
    # agua (WDI): extracción agrícola = total x parte agrícola, km3
    wd = {}
    for ind, fname in WDI_FILES.items():
        js = json.loads(raw(fname).read_text(encoding="utf-8"))[1]
        wd[ind] = {(r["countryiso3code"], int(r["date"])): r["value"] for r in js if r["value"] is not None}
    out = npz_path()
    iso_w = sorted({k[0] for k in wd["ER.H2O.FWTL.K3"]} | {k[0] for k in wd["ER.H2O.FWAG.ZS"]})
    wpos = {c: k for k, c in enumerate(iso_w)}
    tot = np.full((len(iso_w), T), np.nan)
    ag = np.full((len(iso_w), T), np.nan)
    for (c, y), v in wd["ER.H2O.FWTL.K3"].items():
        if YEAR0 <= y <= YEAR1:
            tot[wpos[c], y - YEAR0] = v
    for (c, y), v in wd["ER.H2O.FWAG.ZS"].items():
        if YEAR0 <= y <= YEAR1:
            ag[wpos[c], y - YEAR0] = v
    np.savez_compressed(out, m49=np.array(codes), years=years, water_iso3=np.array(iso_w),
                        water_total_km3=tot, water_ag_share=ag, **arr)
    return out


@lru_cache(maxsize=1)
def load() -> dict:
    p = npz_path()
    if not p.is_file():
        raise FoodDataError(f"no food data at {p}: python world4_cli.py --install-food")
    z = np.load(p, allow_pickle=False)
    return {k: z[k] for k in z.files}


def available() -> bool:
    return npz_path().is_file()


def aligned(iso3, isonum, years) -> dict:
    """Arrays (N, T) en el orden y los años del L1 (NaN fuera de 1961-2024) y las series mundiales (T,)."""
    d = load()
    pos = {int(c): k for k, c in enumerate(d["m49"])}
    k = np.array([pos.get(int(c), -1) for c in isonum])
    years = np.asarray(years)
    ty = years - int(d["years"][0])
    okt = (ty >= 0) & (ty < len(d["years"]))
    out, world = {}, {}
    names = list(SERIES) + ["kcal"]
    wk = pos.get(WORLD_M49, -1)
    for name in names:
        a = d[name]
        b = np.full((len(isonum), len(years)), np.nan)
        rows = k >= 0
        b[np.ix_(rows, okt)] = a[np.ix_(k[rows], ty[okt])]
        out[name] = b
        w = np.full(len(years), np.nan)
        if wk >= 0:
            w[okt] = a[wk, ty[okt]]
        world[name] = w
    wpos = {c: j for j, c in enumerate(d["water_iso3"])}
    kw = np.array([wpos.get(c, -1) for c in iso3])
    wag = np.full((len(iso3), len(years)), np.nan)
    rows = kw >= 0
    wag[np.ix_(rows, okt)] = (d["water_total_km3"] * d["water_ag_share"] / 100.0)[np.ix_(kw[rows], ty[okt])]
    out["water_ag_km3"] = wag
    out["world"] = world
    return out
