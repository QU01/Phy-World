"""Elites, desigualdad e indicadores estructural-demográficos (fuente "elites").

Construye un panel país-año (iso3, year) con:
  * WID (World Inequality Database): participaciones del 1 %, 10 %, 50 % inferior y
    40 % medio en la renta nacional antes de impuestos; participaciones de riqueza;
    Gini de renta y riqueza; renta media por adulto de grupos (PPA constante).
  * SWIID 9.6 (Solt): Gini de renta de mercado y disponible (+ errores estándar).
  * Barro-Lee v3 (2021): proporción con estudios terciarios (25-64, 25-34), años de
    escolarización y peso de la cohorte 15-24 sobre 15-64 (youth bulge).
  * WDI (Banco Mundial): matrícula terciaria bruta, consumo privado per cápita,
    desempleo juvenil, urbanización.
  * PWT 10.01 (ya en data/raw): salario relativo de Turchin w = labsh * pop / emp.

Todas las participaciones se expresan como fracciones 0-1.

Uso:
    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from sources import elites; elites.build()"
"""
from __future__ import annotations

import io
import json
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "elites"
LARGE = RAW / "large"
OUT_DIR = ROOT / "data" / "processed" / "sources"
OUT_CSV = OUT_DIR / "elites.csv"
OUT_DICT = OUT_DIR / "elites_dict.json"

YEARS = range(1950, 2020)
GH = "https://raw.githubusercontent.com/"

# --------------------------------------------------------------------------- #
# Registro de ficheros brutos (todos alojados en raw.githubusercontent.com)
# --------------------------------------------------------------------------- #
FILES = {
    # SWIID 9.6 summary (Solt 2020), copia en un repositorio público
    "swiid9_6_summary.csv": "57777-bit/5609finalProject/HEAD/static/data/swiid9_6_summary.csv",
    # Barro-Lee v3 (1950-2015, grupos de edad de 10 años), repositorio oficial
    "BL_v3_MF.csv": "barrolee/BarroLeeDataSet/HEAD/BLData/BL_v3_MF.csv",
    # WDI (formato API del Banco Mundial), copias recientes
    "wdi_SE.TER.ENRR.csv": "gabbolus/Replication_Guaitoli_Pancrazi_2026/HEAD/Data/WorldBank/"
                           "Education_enrollment/API_SE.TER.ENRR_DS2_en_csv_v2_1423.csv",
    "wdi_NE.CON.PRVT.PC.KD.csv": "ajaitly11/Beauty-Consumption-Analysis/HEAD/data/raw/worldbank/"
                                 "Consumption/API_NE.CON.PRVT.PC.KD_DS2_en_csv_v2_30892.csv",
    "wdi_SL.UEM.1524.ZS.csv": "Awonke03/Capstone_Project/HEAD/API_SL.UEM.1524.ZS_DS2_en_csv_v2_5996635.csv",
    "wdi_SP.URB.TOTL.IN.ZS.csv": "hugohe3/ppt-master-examples/HEAD/examples/ppt169_russia_demography_ru/"
                                 "sources/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_334241.csv",
    # Lista de países WID
    "WID_countries.csv": "sandravizz/Global-Inequality-Data/HEAD/data/WID_countries.csv",
}
# Ficheros WID completos por país (volcado masivo de wid.world, copia sin git-lfs)
WID_COUNTRY_URL = GH + "sandravizz/Global-Inequality-Data/HEAD/data/countries/WID_data_{code}.csv"
WID_FILTERED = RAW / "wid_filtered.csv"

# (variable WID, percentil) -> nombre de salida y factor
WID_KEEP = {
    ("sptincj992", "p99p100"): "top1_share_ptinc",
    ("sptincj992", "p90p100"): "top10_share_ptinc",
    ("sptincj992", "p50p90"): "mid40_share_ptinc",
    ("sptincj992", "p0p50"): "bot50_share_ptinc",
    ("shwealj992", "p99p100"): "top1_share_wealth",
    ("shwealj992", "p90p100"): "top10_share_wealth",
    ("shwealj992", "p0p50"): "bot50_share_wealth",
    ("gptincj992", "p0p100"): "gini_ptinc_wid",
    ("ghwealj992", "p0p100"): "gini_wealth_wid",
    ("aptincj992", "p0p50"): "avg_inc_bot50",
    ("aptincj992", "p99p100"): "avg_inc_top1",
    ("aptincj992", "p90p100"): "avg_inc_top10",
    ("aptincj992", "p0p100"): "avg_inc_all",
    ("anninci992", "p0p100"): "nni_per_adult",
    ("xlcuspi999", "p0p100"): "_ppp_lcu_per_usd",
}
WID_VARS = sorted({k[0] for k in WID_KEEP})


def _get(url: str, retries: int = 3) -> bytes:
    last = None
    for i in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"no se pudo descargar {url}: {last}")


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    gi = LARGE / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    for name, path in FILES.items():
        dst = RAW / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"[elites] descargando {name}")
        dst.write_bytes(_get(GH + urllib.parse.quote(path)))
    if not WID_FILTERED.exists():
        _download_wid()


def _wid_one(code: str) -> str:
    """Descarga el fichero WID de un país y conserva solo las series relevantes."""
    try:
        txt = _get(WID_COUNTRY_URL.format(code=code)).decode("utf-8", "replace")
    except RuntimeError:
        return ""
    keep = []
    for line in txt.splitlines()[1:]:
        parts = line.split(";")
        if len(parts) < 5:
            continue
        if (parts[1], parts[2]) in WID_KEEP:
            keep.append(";".join(parts[:5]))
    return "\n".join(keep)


def _download_wid() -> None:
    lines = (RAW / "WID_countries.csv").read_text(encoding="utf-8", errors="replace").splitlines()[1:]
    alpha2 = [l.split(",")[0].strip('"') for l in lines if l]
    codes = [c for c in alpha2 if len(c) == 2 and c.isalpha() and not c.startswith(("Q", "X", "O"))]
    print(f"[elites] descargando WID para {len(codes)} países (filtrado en streaming)")
    with ThreadPoolExecutor(max_workers=8) as ex:
        chunks = list(ex.map(_wid_one, codes))
    body = "\n".join(c for c in chunks if c)
    WID_FILTERED.write_text("country;variable;percentile;year;value\n" + body + "\n")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _iso2_to_iso3(codes):
    import pycountry
    extra = {"XK": "XKX", "KS": "XKX", "CS": "SCG", "YU": "YUG", "SU": "SUN", "DD": "DDR", "ZZ": None}
    out = {}
    for c in codes:
        if c in extra:
            out[c] = extra[c]
            continue
        rec = pycountry.countries.get(alpha_2=c)
        out[c] = rec.alpha_3 if rec else None
    return out


def _names_to_iso3(names):
    import logging
    import country_converter as coco
    logging.getLogger("country_converter").setLevel(logging.ERROR)
    manual = {"Kosovo": "XKX", "Micronesia": "FSM", "Soviet Union": "SUN", "Yugoslavia": "YUG",
              "Czechoslovakia": "CSK", "Serbia and Montenegro": "SCG", "German Democratic Republic": "DDR"}
    conv = coco.convert(list(names), to="ISO3", not_found=None)
    res = dict(zip(names, conv))
    res.update({k: v for k, v in manual.items() if k in res})
    return res


def _read_wdi(name: str, var: str, scale: float = 1.0) -> pd.DataFrame:
    raw = (RAW / name).read_text(encoding="utf-8-sig")
    lines = raw.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith('"Country Name"') or l.startswith("Country Name"))
    df = pd.read_csv(io.StringIO("\n".join(lines[start:])))
    df = df.rename(columns={"Country Code": "iso3"})
    ycols = [c for c in df.columns if str(c).isdigit()]
    long = df.melt(id_vars="iso3", value_vars=ycols, var_name="year", value_name=var)
    long["year"] = long["year"].astype(int)
    long[var] = pd.to_numeric(long[var], errors="coerce") * scale
    return long.dropna(subset=[var])


# --------------------------------------------------------------------------- #
# Constructores por fuente
# --------------------------------------------------------------------------- #
def _wid() -> pd.DataFrame:
    df = pd.read_csv(WID_FILTERED, sep=";", dtype={"country": str}, keep_default_na=False, na_values=[""])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    # Los importes WID están en moneda local constante del último año del volcado;
    # se convierten a USD PPA de ese año con el factor PPA (xlcuspi999) de ese año.
    ref = df[df["variable"] == "aptincj992"].groupby("country")["year"].max()
    ppp = df[df["variable"] == "xlcuspi999"].set_index(["country", "year"])["value"]
    ppp_ref = {c: ppp.get((c, y), np.nan) for c, y in ref.items()}
    df = df[df["year"].between(1950, 2019) & (df["variable"] != "xlcuspi999")]
    money = df["variable"].isin(["aptincj992", "anninci992"])
    df.loc[money, "value"] = df.loc[money, "value"] / df.loc[money, "country"].map(ppp_ref)
    df["var"] = [WID_KEEP[(v, p)] for v, p in zip(df["variable"], df["percentile"])]
    m = _iso2_to_iso3(df["country"].unique())
    df["iso3"] = df["country"].map(m)
    df = df.dropna(subset=["iso3"])
    wide = df.pivot_table(index=["iso3", "year"], columns="var", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    # En este volcado aptinc p0p50 equivale a share*media (no a la media del grupo; ratio
    # constante 0.50 frente a share/0.5*media), mientras que p99p100 y p90p100 sí son
    # medias de grupo. Se recalcula la media del 50 % inferior a partir de la participación.
    if {"bot50_share_ptinc", "avg_inc_all"} <= set(wide.columns):
        wide["avg_inc_bot50"] = wide["bot50_share_ptinc"] / 0.5 * wide["avg_inc_all"]
    # Marca de series planas (WID extrapola manteniendo constante la distribución cuando
    # no hay encuestas ni datos fiscales): 1 si top1 es idéntico al año anterior o siguiente.
    wide = wide.sort_values(["iso3", "year"])
    g = wide.groupby("iso3")["top1_share_ptinc"]
    flat = (g.diff() == 0) | (g.diff(-1) == 0)
    wide["wid_flat_flag"] = np.where(wide["top1_share_ptinc"].notna(), flat.astype(int), np.nan)
    # ratio de renta media top1 / bottom50 (renta relativa de élite vs masas)
    if {"avg_inc_top1", "avg_inc_bot50"} <= set(wide.columns):
        wide["top1_bot50_income_ratio"] = wide["avg_inc_top1"] / wide["avg_inc_bot50"]
    # Salario relativo "WID": renta media del 50 % inferior / renta media nacional
    if {"avg_inc_bot50", "avg_inc_all"} <= set(wide.columns):
        wide["bot50_rel_income"] = wide["avg_inc_bot50"] / wide["avg_inc_all"]
    return wide


def _swiid() -> pd.DataFrame:
    df = pd.read_csv(RAW / "swiid9_6_summary.csv")
    m = _names_to_iso3(df["country"].unique())
    df["iso3"] = df["country"].map(m)
    df = df.dropna(subset=["iso3"])
    out = pd.DataFrame({
        "iso3": df["iso3"], "year": df["year"].astype(int),
        "gini_disp": df["gini_disp"] / 100, "gini_disp_se": df["gini_disp_se"] / 100,
        "gini_mkt": df["gini_mkt"] / 100, "gini_mkt_se": df["gini_mkt_se"] / 100,
        "redist_rel": df["rel_red"] / 100,
    })
    return out.groupby(["iso3", "year"], as_index=False).mean()


def _barro_lee() -> pd.DataFrame:
    bl = pd.read_csv(RAW / "BL_v3_MF.csv")
    bl = bl.rename(columns={"WBcode": "iso3"})
    bl = bl[(bl["ageto"] - bl["agefrom"]) == 9]  # solo grupos de 10 años (15-24 ... 55-64)
    bl["iso3"] = bl["iso3"].replace({"ROM": "MDA", "SER": "SRB"})
    rows = []
    for (iso, yr), g in bl.groupby(["iso3", "year"]):
        g = g.set_index("agefrom")
        pop = g["pop"]
        a = g.loc[g.index >= 25]
        w = a["pop"] / a["pop"].sum()
        rows.append({
            "iso3": iso, "year": int(yr),
            "tert_share_2564": float((a["lh"] * w).sum() / 100),
            "tert_compl_share_2564": float((a["lhc"] * w).sum() / 100),
            "tert_share_2534": float(g.loc[25, "lh"] / 100) if 25 in g.index else np.nan,
            "yr_sch_2564": float((a["yr_sch"] * w).sum()),
            "youth_share_1524_of_1564": float(pop.get(15, np.nan) / pop.sum()),
        })
    df = pd.DataFrame(rows)
    # interpolación lineal quinquenal -> anual (sin extrapolar fuera de 1950-2015)
    out = []
    for iso, g in df.groupby("iso3"):
        g = g.set_index("year").sort_index()
        full = g.reindex(range(g.index.min(), g.index.max() + 1))
        full = full.drop(columns="iso3").interpolate(method="index", limit_area="inside")
        full["iso3"] = iso
        out.append(full.reset_index().rename(columns={"index": "year"}))
    return pd.concat(out, ignore_index=True)


def _wdi() -> pd.DataFrame:
    parts = [
        _read_wdi("wdi_SE.TER.ENRR.csv", "tert_enrol_gross", 0.01),
        _read_wdi("wdi_NE.CON.PRVT.PC.KD.csv", "hh_cons_pc_2015usd"),
        _read_wdi("wdi_SL.UEM.1524.ZS.csv", "youth_unemp", 0.01),
        _read_wdi("wdi_SP.URB.TOTL.IN.ZS.csv", "urban_share", 0.01),
    ]
    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on=["iso3", "year"], how="outer")
    return out


def _pwt() -> pd.DataFrame:
    p = ROOT / "data" / "raw" / "pwt1001.csv"
    if not p.exists():
        return pd.DataFrame(columns=["iso3", "year"])
    d = pd.read_csv(p, usecols=["isocode", "year", "pop", "emp", "labsh", "rgdpna"])
    d = d.rename(columns={"isocode": "iso3"})
    # Salario relativo de Turchin: (labsh*Y/emp)/(Y/pop) = labsh*pop/emp
    d["rel_wage_pwt"] = d["labsh"] * d["pop"] / d["emp"]
    # Salario medio real por trabajador (USD PPA 2017, precios nacionales constantes)
    d["wage_per_worker_pwt"] = d["labsh"] * d["rgdpna"] / d["emp"] * 1e0
    d["emp_pop_ratio"] = d["emp"] / d["pop"]
    return d[["iso3", "year", "rel_wage_pwt", "wage_per_worker_pwt", "emp_pop_ratio"]]


# --------------------------------------------------------------------------- #
# Diccionario
# --------------------------------------------------------------------------- #
WID_SRC = ("World Inequality Database (wid.world), volcado masivo por país; copia en "
           "github.com/sandravizz/Global-Inequality-Data. Adultos 20+, reparto igualitario (j).")
DESC = {
    "top1_share_ptinc": ("Participación del 1 % superior en la renta nacional antes de impuestos (sptinc992j p99p100)", WID_SRC, "fracción 0-1"),
    "top10_share_ptinc": ("Participación del 10 % superior en la renta antes de impuestos (sptinc992j p90p100)", WID_SRC, "fracción 0-1"),
    "mid40_share_ptinc": ("Participación del 40 % medio (p50p90) en la renta antes de impuestos", WID_SRC, "fracción 0-1"),
    "bot50_share_ptinc": ("Participación del 50 % inferior en la renta antes de impuestos (sptinc992j p0p50)", WID_SRC, "fracción 0-1"),
    "top1_share_wealth": ("Participación del 1 % superior en la riqueza personal neta (shweal992j p99p100)", WID_SRC, "fracción 0-1"),
    "top10_share_wealth": ("Participación del 10 % superior en la riqueza personal neta", WID_SRC, "fracción 0-1"),
    "bot50_share_wealth": ("Participación del 50 % inferior en la riqueza personal neta", WID_SRC, "fracción 0-1"),
    "gini_ptinc_wid": ("Gini de la renta antes de impuestos (gptinc992j)", WID_SRC, "0-1"),
    "gini_wealth_wid": ("Gini de la riqueza personal neta (ghweal992j)", WID_SRC, "0-1"),
    "avg_inc_bot50": ("Renta media antes de impuestos del 50 % inferior por adulto (aptinc992j p0p50) — proxy de renta de masas", WID_SRC, "USD PPA constantes del año de referencia WID"),
    "avg_inc_top1": ("Renta media antes de impuestos del 1 % superior por adulto — renta de élite", WID_SRC, "USD PPA constantes (ref. WID)"),
    "avg_inc_top10": ("Renta media antes de impuestos del 10 % superior por adulto", WID_SRC, "USD PPA constantes (ref. WID)"),
    "avg_inc_all": ("Renta media antes de impuestos por adulto (aptinc992j p0p100)", WID_SRC, "USD PPA constantes (ref. WID)"),
    "nni_per_adult": ("Renta nacional neta media por adulto (anninc992i)", WID_SRC, "USD PPA constantes (ref. WID)"),
    "top1_bot50_income_ratio": ("Cociente renta media top 1 % / 50 % inferior (renta relativa élite/masas)", "Derivado de WID", "ratio"),
    "bot50_rel_income": ("Renta media del 50 % inferior / renta media nacional (= 2 * bot50_share); salario relativo alternativo", "Derivado de WID", "ratio"),
    "wid_flat_flag": ("1 si la participación top1 WID es idéntica al año anterior/siguiente (extrapolación plana, baja calidad)", "Derivado de WID", "0/1"),
    "gini_disp": ("Gini de renta disponible (tras impuestos y transferencias), media de 100 imputaciones", "SWIID 9.6 (Solt 2020), swiid9_6_summary.csv", "0-1"),
    "gini_disp_se": ("Error estándar de gini_disp", "SWIID 9.6", "0-1"),
    "gini_mkt": ("Gini de renta de mercado (antes de impuestos y transferencias)", "SWIID 9.6", "0-1"),
    "gini_mkt_se": ("Error estándar de gini_mkt", "SWIID 9.6", "0-1"),
    "redist_rel": ("Redistribución relativa (gini_mkt-gini_disp)/gini_mkt", "SWIID 9.6", "fracción"),
    "tert_share_2564": ("Proporción de la población 25-64 con algún estudio terciario (lh), interpolado anual", "Barro & Lee (2013) v3 2021 (BL_v3_MF.csv)", "fracción 0-1"),
    "tert_compl_share_2564": ("Proporción 25-64 con estudios terciarios completos (lhc)", "Barro-Lee v3", "fracción 0-1"),
    "tert_share_2534": ("Proporción 25-34 con estudios terciarios (aspirantes a élite jóvenes)", "Barro-Lee v3", "fracción 0-1"),
    "yr_sch_2564": ("Años medios de escolarización, población 25-64", "Barro-Lee v3", "años"),
    "youth_share_1524_of_1564": ("Población 15-24 / población 15-64 (youth bulge)", "Barro-Lee v3 (columna pop)", "fracción 0-1"),
    "tert_enrol_gross": ("Tasa bruta de matrícula terciaria (SE.TER.ENRR)", "WDI (vintage 2026-04)", "fracción (puede superar 1)"),
    "hh_cons_pc_2015usd": ("Gasto de consumo final de los hogares per cápita (NE.CON.PRVT.PC.KD)", "WDI (vintage 2025-07)", "USD constantes 2015"),
    "youth_unemp": ("Desempleo juvenil 15-24, estimación modelada OIT (SL.UEM.1524.ZS)", "WDI (vintage 2023-10)", "fracción 0-1"),
    "urban_share": ("Población urbana / total (SP.URB.TOTL.IN.ZS), componente N_urb/N del MMP", "WDI (vintage 2026-07)", "fracción 0-1"),
    "rel_wage_pwt": ("Salario relativo de Turchin w = (labsh*Y/emp)/(Y/pop) = labsh*pop/emp", "Penn World Table 10.01", "ratio"),
    "wage_per_worker_pwt": ("Salario medio real por trabajador = labsh*rgdpna/emp", "Penn World Table 10.01", "millones USD2017 / millones de trabajadores = USD2017"),
    "emp_pop_ratio": ("Empleo / población", "Penn World Table 10.01", "fracción 0-1"),
}


def build() -> pd.DataFrame:
    download()
    frames = [_wid(), _swiid(), _barro_lee(), _wdi(), _pwt()]
    panel = frames[0]
    for f in frames[1:]:
        panel = panel.merge(f, on=["iso3", "year"], how="outer")
    panel = panel[panel["year"].between(1950, 2019)]
    # conservar solo códigos ISO3 de países (descartar agregados WDI)
    import pycountry
    valid = {c.alpha_3 for c in pycountry.countries} | {"XKX", "SUN", "YUG", "CSK", "SCG", "DDR"}
    panel = panel[panel["iso3"].isin(valid)]
    cols = ["iso3", "year"] + [c for c in DESC if c in panel.columns]
    panel = panel[cols].sort_values(["iso3", "year"]).reset_index(drop=True)
    panel = panel.dropna(how="all", subset=cols[2:])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_csv(OUT_CSV, index=False, float_format="%.6g")

    meta = {}
    for c in cols[2:]:
        s = panel[["iso3", "year", c]].dropna()
        d, src, units = DESC[c]
        meta[c] = {
            "description": d, "source": src, "units": units,
            "coverage": {
                "n_obs": int(len(s)), "n_countries": int(s["iso3"].nunique()),
                "year_min": int(s["year"].min()) if len(s) else None,
                "year_max": int(s["year"].max()) if len(s) else None,
                "n_countries_1960": int(s.loc[s["year"] == 1960, "iso3"].nunique()),
                "n_countries_1990": int(s.loc[s["year"] == 1990, "iso3"].nunique()),
                "n_countries_2015": int(s.loc[s["year"] == 2015, "iso3"].nunique()),
            },
        }
    OUT_DICT.write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"[elites] {OUT_CSV.relative_to(ROOT)}: {len(panel)} filas, {panel['iso3'].nunique()} países, "
          f"{len(cols) - 2} variables")
    return panel


if __name__ == "__main__":
    build()
