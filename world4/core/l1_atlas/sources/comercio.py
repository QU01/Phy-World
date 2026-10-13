"""Comercio, IED, intercambio desigual y tecnología (fuente "comercio").

build() descarga (si faltan) los ficheros brutos a data/raw/comercio/ y escribe:

* data/processed/sources/comercio.csv         panel país-año (iso3, year, variables snake_case)
* data/processed/sources/comercio_dyadic.csv.gz  flujos bilaterales dirigidos (iso3_o, iso3_d, year, flow)
* data/processed/sources/comercio_dict.json    diccionario de variables + estimaciones auxiliares

Fuentes (todas vía raw.githubusercontent.com, espejos de los ficheros originales):
* COW Dyadic Trade v4.0 (Barbieri & Keshk 2016), 1870-2014, millones USD corrientes.
* CEPII GeoDist (Mayer & Zignago 2011) vía el paquete CRAN cepiigeodist.
* countrycode (CRAN) para la tabla COW -> ISO3.
* WDI (Banco Mundial): NE.TRD.GNFS.ZS, BX/BM.KLT.DINV.WD.GD.ZS, TX.VAL.TECH.MF.ZS,
  TX.VAL.MANF.ZS.UN, IP.PAT.RESD, SP.POP.SCIE.RD.P6, GB.XPD.RSDV.GD.ZS, TT.PRI.MRCH.XD.WD.
* Índice de Complejidad Económica: Atlas (Harvard Growth Lab) SITC 1962-2017 (Gapminder DDF)
  y Growth Lab rankings 1995-2023 (SITC, HS92).
* PWT 10.01 (data/raw/pwt1001.csv, ya en el repo) para niveles de precios y PIB nominal.
"""
import gzip
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "comercio"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"
GH = "https://raw.githubusercontent.com/"

# nombre local -> (url, grande?)
SOURCES = {
    "Dyadic_COW_4.0.csv": (GH + "PacktPublishing/Superset-Quick-Start-Guide/HEAD/Chapter06/"
                           "Dyadic_COW_4.0.csv", True),
    "countrycode_codelist.rda": (GH + "cran/countrycode/master/data/codelist.rda", False),
    "cepii_dist.rda": (GH + "cran/cepiigeodist/master/data/dist_cepii.rda", False),
    "cepii_geo.rda": (GH + "cran/cepiigeodist/master/data/geo_cepii.rda", False),
    "atlas_sitc_eci.csv": (GH + "open-numbers/ddf--gapminder--atlas_of_economic_complexity/HEAD/"
                           "ddf--datapoints--sitc_eci--by--location--year.csv", False),
    "atlas_hs_eci.csv": (GH + "open-numbers/ddf--gapminder--atlas_of_economic_complexity/HEAD/"
                         "ddf--datapoints--hs_eci--by--location--year.csv", False),
    "growthlab_eci_rankings.csv": (GH + "graebnerc/DevelopmentEconomicsFall26/HEAD/content/tutorials/"
                                   "MeasuringDevelopment/growth_proj_eci_rankings.csv", False),
    "wdi_NE.TRD.GNFS.ZS.csv": (GH + "MatthewTsang0213/Empirical-Project/HEAD/"
                               "API_NE.TRD.GNFS.ZS_DS2_en_csv_v2_101.csv", False),
    "wdi_BX.KLT.DINV.WD.GD.ZS.csv": (GH + "MatthewTsang0213/Empirical-Project/HEAD/"
                                     "API_BX.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_13.csv", False),
    "wdi_BM.KLT.DINV.WD.GD.ZS.csv": (GH + "dufourlorenzo/replication-fdi-growth-panel/HEAD/data/"
                                     "API_BM.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_316.csv", False),
    "wdi_TX.VAL.TECH.MF.ZS.csv": (GH + "wiilldii/courseworkHSE/HEAD/data/"
                                  "API_TX.VAL.TECH.MF.ZS_DS2_en_csv_v2_1859.csv", False),
    "wdi_TX.VAL.MANF.ZS.UN.csv": (GH + "yomisamadesu/Manufacturing_AI_Prediction_Model/HEAD/DSML1/"
                                  "Datasets/API_TX.VAL.MANF.ZS.UN_DS2_en_csv_v2_7563.csv", False),
    "wdi_IP.PAT.RESD.csv": (GH + "Wei-Seng/group30_week4teamassignment/HEAD/TASK1.4/"
                            "API_IP.PAT.RESD_DS2_en_csv_v2_226.csv", False),
    "wdi_SP.POP.SCIE.RD.P6.csv": (GH + "Zavvnr/Zavier-ISS-Presentation-Data/HEAD/presentation_2/data/"
                                  "API_SP.POP.SCIE.RD.P6_DS2_en_csv_v2_37695.csv", False),
    "wdi_GB.XPD.RSDV.GD.ZS.csv": (GH + "Zavvnr/Zavier-ISS-Presentation-Data/HEAD/presentation_2/data/"
                                  "API_GB.XPD.RSDV.GD.ZS_DS2_en_csv_v2_232.csv", False),
    "wdi_TT.PRI.MRCH.XD.WD.csv": (GH + "ronnywang/worldbank/HEAD/WDI_bundle/parsed/"
                                  "TT.PRI.MRCH.XD.WD_WDI.csv", False),
}

# códigos COW sin ISO3 en countrycode: sucesor territorial usado por PWT
COW_EXTRA = {260: "DEU", 315: "CZE", 345: "SRB", 347: "XKX", 678: "YEM", 730: "KOR"}

# (variable, código WDI, escala, descripción, unidad)
WDI = [
    ("trade_gdp_wdi", "NE.TRD.GNFS.ZS", 0.01, "Comercio (X+M de bienes y servicios) / PIB", "fracción"),
    ("fdi_in_gdp", "BX.KLT.DINV.WD.GD.ZS", 0.01, "IED, entradas netas / PIB", "fracción"),
    ("fdi_out_gdp", "BM.KLT.DINV.WD.GD.ZS", 0.01, "IED, salidas netas / PIB", "fracción"),
    ("hightech_x_share", "TX.VAL.TECH.MF.ZS", 0.01,
     "Exportaciones de alta tecnología / exportaciones de manufacturas", "fracción"),
    ("manuf_x_share", "TX.VAL.MANF.ZS.UN", 0.01, "Manufacturas / exportaciones de mercancías", "fracción"),
    ("patents_resident", "IP.PAT.RESD", 1.0, "Solicitudes de patente de residentes", "número"),
    ("researchers_per_mn", "SP.POP.SCIE.RD.P6", 1.0, "Investigadores en I+D por millón de habitantes",
     "por millón"),
    ("rd_gdp", "GB.XPD.RSDV.GD.ZS", 0.01, "Gasto en I+D / PIB", "fracción"),
    ("tot_wdi", "TT.PRI.MRCH.XD.WD", 1.0, "Términos de intercambio de trueque neto (2000=100)", "índice"),
]

Y0, Y1 = 1950, 2019
CORE_Q = 0.80  # "centro" = 20 % superior del PIB per cápita PWT de cada año


def _path(name):
    return (LARGE if SOURCES[name][1] else RAW) / name


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    gi = LARGE / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    for name, (url, _) in SOURCES.items():
        dst = _path(name)
        if not dst.exists():
            print("descargando", name)
            tmp = dst.with_suffix(dst.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.rename(dst)


def _rda(name):
    import warnings

    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(_path(name))
    return obj[list(obj)[0]]


def _cow_iso():
    cl = _rda("countrycode_codelist.rda")
    cl = cl[cl.cown.notna() & cl.iso3c.notna()]
    m = dict(zip(cl.cown.astype(int), cl.iso3c.astype(str)))
    m.update(COW_EXTRA)
    return m


def _wdi(name, var, scale):
    df = pd.read_csv(_path(name), skiprows=4)
    years = [c for c in df.columns if c.strip().isdigit()]
    long = df.melt(id_vars=["Country Code"], value_vars=years, var_name="year", value_name=var)
    long = long.rename(columns={"Country Code": "iso3"}).dropna()
    long["year"] = long.year.astype(int)
    long[var] = long[var].astype(float) * scale
    return long


def _wdi_plain(name, var, scale):  # formato sin las 4 líneas de cabecera
    df = pd.read_csv(_path(name))
    years = [c for c in df.columns if c.strip().isdigit()]
    long = df.melt(id_vars=["Country Code"], value_vars=years, var_name="year", value_name=var)
    long = long.rename(columns={"Country Code": "iso3"}).dropna()
    long["year"] = long.year.astype(int)
    long[var] = long[var].astype(float) * scale
    return long


# ---------------------------------------------------------------------------------------------
def dyadic():
    """COW dyadic -> flujos dirigidos exportador (o) -> importador (d), millones USD corrientes."""
    iso = _cow_iso()
    cols = ["ccode1", "ccode2", "year", "flow1", "flow2"]
    t = pd.read_csv(_path("Dyadic_COW_4.0.csv"), usecols=cols)
    t = t[(t.year >= Y0) & (t.year <= Y1)]
    # flow1 = importaciones de ccode1 desde ccode2  => o = ccode2, d = ccode1
    a = t[["ccode2", "ccode1", "year", "flow1"]].set_axis(["o", "d", "year", "flow"], axis=1)
    b = t[["ccode1", "ccode2", "year", "flow2"]].set_axis(["o", "d", "year", "flow"], axis=1)
    f = pd.concat([a, b], ignore_index=True)
    f = f[f.flow > 0]
    f["iso3_o"] = f.o.map(iso)
    f["iso3_d"] = f.d.map(iso)
    f = f.dropna(subset=["iso3_o", "iso3_d"])
    f = f[f.iso3_o != f.iso3_d]
    f = f.groupby(["iso3_o", "iso3_d", "year"], as_index=False).flow.sum()
    f["flow"] = f.flow.round(3)
    return f


def _pwt():
    p = pd.read_csv(ROOT / "data" / "raw" / "pwt1001.csv",
                    usecols=["isocode", "year", "cgdpo", "pl_gdpo", "pl_x", "pl_m", "csh_x", "csh_m",
                             "rgdpna", "pop"])
    p = p.rename(columns={"isocode": "iso3"})
    # niveles de precios no positivos o extremos (outliers conocidos de PWT) -> NaN
    for c in ["pl_gdpo", "pl_x", "pl_m"]:
        p.loc[(p[c] <= 0.01) | (p[c] > 10), c] = np.nan
    p.loc[(p.csh_x < 0) | (p.csh_m > 0) | (p.csh_x - p.csh_m > 5), ["csh_x", "csh_m"]] = np.nan
    p["gdp_usd"] = p.cgdpo * p.pl_gdpo  # PIB nominal aprox. (millones USD corrientes)
    p["gdppc"] = p.rgdpna / p["pop"]
    return p


def _core(pwt):
    """Centro = cuantil 80 del PIB pc (PWT rgdpna/pop) de cada año."""
    c = pwt.dropna(subset=["gdppc"]).copy()
    thr = c.groupby("year").gdppc.transform(lambda s: s.quantile(CORE_Q))
    c["is_core"] = (c.gdppc >= thr).astype(int)
    return c[["iso3", "year", "is_core"]]


def network_indicators(f, core, pwt):
    f = f.merge(core.rename(columns={"iso3": "iso3_d", "is_core": "core_d"}), how="left")
    f = f.merge(core.rename(columns={"iso3": "iso3_o", "is_core": "core_o"}), how="left")
    f[["core_o", "core_d"]] = f[["core_o", "core_d"]].fillna(0)

    ex = f.groupby(["iso3_o", "year"]).agg(exports_cow=("flow", "sum"),
                                           n_export_partners=("flow", "size"))
    ex["x_core"] = f.assign(v=f.flow * f.core_d).groupby(["iso3_o", "year"]).v.sum()
    f["sx"] = f.flow / f.groupby(["iso3_o", "year"]).flow.transform("sum")
    ex["hhi_export_partners"] = (f.sx ** 2).groupby([f.iso3_o, f.year]).sum()
    ex.index.names = ["iso3", "year"]

    im = f.groupby(["iso3_d", "year"]).agg(imports_cow=("flow", "sum"))
    im["m_core"] = f.assign(v=f.flow * f.core_o).groupby(["iso3_d", "year"]).v.sum()
    f["sm"] = f.flow / f.groupby(["iso3_d", "year"]).flow.transform("sum")
    im["hhi_import_partners"] = (f.sm ** 2).groupby([f.iso3_d, f.year]).sum()
    im.index.names = ["iso3", "year"]

    d = ex.join(im, how="outer").fillna({"exports_cow": 0, "imports_cow": 0, "x_core": 0, "m_core": 0})
    d["share_x_to_core"] = d.x_core / d.exports_cow.replace(0, np.nan)
    d["share_m_from_core"] = d.m_core / d.imports_cow.replace(0, np.nan)
    d["share_trade_core"] = (d.x_core + d.m_core) / (d.exports_cow + d.imports_cow).replace(0, np.nan)
    d = d.drop(columns=["x_core", "m_core"]).reset_index()

    # --- drenaje por intercambio desigual, método ERDI (Hickel, Sullivan & Zoomkawala 2021):
    # cada dólar exportado por la periferia i al centro j "vale" p_j/p_i dólares a precios del
    # centro (p = nivel de precios PWT pl_gdpo, EE.UU.=1). Drenaje_ij = X_ij * (p_j/p_i - 1)_+.
    # Variante conservadora: precios de exportación PWT (pl_x) en lugar del nivel de precios del PIB.
    g0 = f[(f.core_d == 1) & (f.core_o == 0)][["iso3_o", "iso3_d", "year", "flow"]]
    for tag, col in [("erdi", "pl_gdpo"), ("px", "pl_x")]:
        pl = pwt[["iso3", "year", col]].dropna()
        g = g0.merge(pl.rename(columns={"iso3": "iso3_o", col: "p_o"}))
        g = g.merge(pl.rename(columns={"iso3": "iso3_d", col: "p_d"}))
        g["drain"] = g.flow * np.clip(g.p_d / g.p_o - 1, 0, None)
        out = g.groupby(["iso3_o", "year"]).drain.sum().rename_axis(["iso3", "year"]).rename(f"drain_{tag}_usd")
        inn = g.groupby(["iso3_d", "year"]).drain.sum().rename_axis(["iso3", "year"]).rename(f"drain_{tag}_in_usd")
        d = d.merge(out.reset_index(), how="left").merge(inn.reset_index(), how="left")
    return d


def eci():
    a = pd.read_csv(_path("atlas_sitc_eci.csv")).rename(columns={"location": "iso3", "sitc_eci": "eci_sitc_atlas"})
    h = pd.read_csv(_path("atlas_hs_eci.csv")).rename(columns={"location": "iso3", "hs_eci": "eci_hs_atlas"})
    for x in (a, h):
        x["iso3"] = x.iso3.str.upper()
    g = pd.read_csv(_path("growthlab_eci_rankings.csv"))
    g = g.rename(columns={"country_iso3_code": "iso3", "eci_sitc": "eci_sitc_gl", "eci_hs92": "eci_hs92_gl"})
    g = g[["iso3", "year", "eci_sitc_gl", "eci_hs92_gl"]]
    e = a.merge(h, how="outer").merge(g, how="outer")
    # serie empalmada: Growth Lab (1995+) y, antes, Atlas SITC reescalado por MCO en el solape
    ov = e.dropna(subset=["eci_sitc_atlas", "eci_sitc_gl"])
    b, a0 = np.polyfit(ov.eci_sitc_atlas, ov.eci_sitc_gl, 1) if len(ov) > 50 else (1.0, 0.0)
    e["eci"] = e.eci_sitc_gl.fillna(a0 + b * e.eci_sitc_atlas)
    return e, dict(splice_slope=float(b), splice_intercept=float(a0), overlap_n=int(len(ov)),
                   overlap_corr=float(ov.eci_sitc_atlas.corr(ov.eci_sitc_gl)) if len(ov) else None)


# ---------------------------------------------------------------------------------------------
def _ols(y, X):
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    n, k = X.shape
    cov = np.linalg.inv(X.T @ X) * (res @ res) / (n - k)
    return beta, np.sqrt(np.diag(cov)), 1 - res.var() / y.var(), n


def gravity(f, pwt):
    """Gravedad log-lineal con efectos fijos de año (datos 1950-2014, flujos > 0):
    ln X_odt = a_t + b_o ln Y_o + b_d ln Y_d - delta ln dist + g contig + l comlang + c colony."""
    dist = _rda("cepii_dist.rda")[["iso_o", "iso_d", "dist", "contig", "comlang_off", "colony"]]
    dist = dist.rename(columns={"iso_o": "iso3_o", "iso_d": "iso3_d"})
    dist["iso3_o"] = dist.iso3_o.replace({"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "YUG": "SRB", "PAL": "PSE"})
    dist["iso3_d"] = dist.iso3_d.replace({"ROM": "ROU", "ZAR": "COD", "TMP": "TLS", "YUG": "SRB", "PAL": "PSE"})
    y = pwt[["iso3", "year", "gdp_usd"]].dropna()
    g = f.merge(dist).merge(y.rename(columns={"iso3": "iso3_o", "gdp_usd": "y_o"})) \
         .merge(y.rename(columns={"iso3": "iso3_d", "gdp_usd": "y_d"}))
    g = g[g.flow > 0.01]
    lx = np.log(g.flow.values)
    X = np.column_stack([np.log(g.y_o), np.log(g.y_d), np.log(g.dist), g.contig, g.comlang_off, g.colony])
    # efectos fijos de año por "within" (desviación respecto a la media del año)
    yr = g.year.values
    def dm(v):
        s = pd.Series(v).groupby(yr).transform("mean").values
        return v - s
    Xd = np.column_stack([dm(X[:, j]) for j in range(X.shape[1])])
    b, se, r2, n = _ols(dm(lx), Xd)
    names = ["ln_gdp_o", "ln_gdp_d", "ln_dist", "contig", "comlang_off", "colony"]
    return {"method": "MCO log-lineal, EF de año (within), flujos>0.01 M USD, COW 1950-2014",
            "n": int(n), "r2_within": round(float(r2), 3),
            "coef": {k: round(float(v), 3) for k, v in zip(names, b[1:])},
            "se_naive": {k: round(float(v), 4) for k, v in zip(names, se[1:])}}


def eci_growth(e, pwt):
    """Crecimiento medio anual del PIB pc en 10 años ~ ECI_t + ln PIBpc_t (+EF década)."""
    p = pwt[["iso3", "year", "gdppc"]].dropna()
    p = p.merge(e[["iso3", "year", "eci"]].dropna())
    fut = pwt[["iso3", "year", "gdppc"]].rename(columns={"gdppc": "g10"})
    fut["year"] -= 10
    p = p.merge(fut)
    p = p[p.year.isin(range(1965, 2010, 5))]
    p["gr"] = np.log(p.g10 / p.gdppc) / 10
    D = pd.get_dummies(p.year, drop_first=True).values.astype(float)
    b, se, r2, n = _ols(p.gr.values, np.column_stack([p.eci, np.log(p.gdppc), D]))
    return {"method": "MCO; crec. anual medio ln(PIBpc) t->t+10 ~ ECI_t + ln PIBpc_t + EF año; t=1965,...,2005",
            "n": int(n), "r2": round(float(r2), 3), "coef_eci": round(float(b[1]), 4),
            "se_eci": round(float(se[1]), 4), "coef_lny": round(float(b[2]), 4),
            "se_lny": round(float(se[2]), 4), "sd_eci": round(float(p.eci.std()), 3)}


def drain_summary(d, tag="erdi"):
    x = d.dropna(subset=[f"drain_{tag}_gdp"])
    x = x[x.is_core == 0].copy()
    x["decade"] = (x.year // 10) * 10
    s = x.groupby("decade").apply(lambda z: pd.Series({
        "agg_drain_over_periphery_gdp": z[f"drain_{tag}_usd"].sum() / z.gdp_usd.sum(),
        "median_drain_gdp": z[f"drain_{tag}_gdp"].median(),
        "agg_drain_over_periphery_exports": z[f"drain_{tag}_usd"].sum() / z.exports_cow.sum(),
        "agg_drain_usd_bn_per_year": z[f"drain_{tag}_usd"].sum() / 1e3 / z.year.nunique(),
        "n_countries": z.iso3.nunique()}), include_groups=False)
    return {int(k): {c: round(float(v), 4) for c, v in r.items()} for k, r in s.iterrows()}


def diffusion(d, pwt):
    """Convergencia condicionada (reduced form): crec. anual PIBpc t->t+10 ~ gap + gap*m_core + gap*fdi
    (+EF año), gap = ln(y_USA / y_i), m_core = importaciones desde el centro / PIB (COW/PWT)."""
    p = pwt.dropna(subset=["gdppc"]).copy()
    us = p[p.iso3 == "USA"].set_index("year").gdppc
    p["gap"] = np.log(p.year.map(us) / p.gdppc)
    z = p.merge(d[["iso3", "year", "imports_cow", "share_m_from_core", "fdi_in_gdp"]], how="left")
    z["mcore"] = z.imports_cow * z.share_m_from_core / z.gdp_usd
    fut = p[["iso3", "year", "gdppc"]].rename(columns={"gdppc": "g10"})
    fut["year"] -= 10
    z = z.merge(fut)
    z["gr"] = np.log(z.g10 / z.gdppc) / 10
    z = z[z.year.isin(range(1970, 2010, 5))].dropna(subset=["gr", "gap", "mcore", "fdi_in_gdp"])
    z = z[(z.mcore < 1.0) & (z.fdi_in_gdp.abs() < 0.5)]
    D = pd.get_dummies(z.year, drop_first=True).values.astype(float)
    b, se, r2, n = _ols(z.gr.values, np.column_stack([z.gap, z.gap * z.mcore, z.gap * z.fdi_in_gdp, D]))
    k = ["gap", "gap_x_mcore", "gap_x_fdi"]
    return {"method": " ".join(diffusion.__doc__.split()), "n": int(n), "r2": round(float(r2), 3),
            "coef": {a: round(float(v), 4) for a, v in zip(k, b[1:4])},
            "se": {a: round(float(v), 4) for a, v in zip(k, se[1:4])},
            "mean_mcore": round(float(z.mcore.mean()), 3), "mean_fdi": round(float(z.fdi_in_gdp.mean()), 4)}


def penn_effect(pwt):
    """ln p_i = a_t + beta * ln(y_i / y_USA): elasticidad del nivel de precios al ingreso relativo
    (efecto Balassa-Samuelson / 'Penn'), que fija cómo escala el drenaje ERDI con la brecha."""
    p = pwt.dropna(subset=["gdppc"]).copy()
    us = p[p.iso3 == "USA"].set_index("year").gdppc
    p["lrel"] = np.log(p.gdppc / p.year.map(us))
    res = {}
    for col in ["pl_gdpo", "pl_x"]:
        z = p.dropna(subset=[col, "lrel"])
        z = z[z.year >= 1960]
        D = pd.get_dummies(z.year, drop_first=True).values.astype(float)
        b, se, r2, n = _ols(np.log(z[col].values), np.column_stack([z.lrel, D]))
        res[col] = {"beta": round(float(b[1]), 4), "se": round(float(se[1]), 4), "r2": round(float(r2), 3), "n": int(n)}
    return res


# ---------------------------------------------------------------------------------------------
DICT = {
    "exports_cow": ("Exportaciones totales (suma diádica COW)", "millones USD corrientes", "COW Dyadic Trade 4.0"),
    "imports_cow": ("Importaciones totales (suma diádica COW)", "millones USD corrientes", "COW Dyadic Trade 4.0"),
    "n_export_partners": ("Número de destinos con exportaciones > 0", "número", "COW Dyadic Trade 4.0"),
    "hhi_export_partners": ("Herfindahl de concentración de exportaciones por socio (0-1)", "índice", "COW"),
    "hhi_import_partners": ("Herfindahl de concentración de importaciones por socio (0-1)", "índice", "COW"),
    "share_x_to_core": ("Exportaciones hacia el centro (20 % sup. PIB pc PWT del año) / exportaciones", "fracción", "COW+PWT"),
    "share_m_from_core": ("Importaciones desde el centro / importaciones", "fracción", "COW+PWT"),
    "share_trade_core": ("Comercio (X+M) con el centro / comercio total", "fracción", "COW+PWT"),
    "is_core": ("1 si el país está en el 20 % superior del PIB pc (rgdpna/pop) de ese año", "0/1", "PWT 10.01"),
    "drain_erdi_usd": ("Drenaje por intercambio desigual (ERDI): sum_j X_ij (p_j/p_i - 1)+ hacia el centro; sólo periferia",
                       "millones USD corrientes", "COW + PWT pl_gdpo; método Hickel et al. (2021)"),
    "drain_erdi_gdp": ("drain_erdi_usd / PIB nominal (cgdpo*pl_gdpo)", "fracción", "COW+PWT"),
    "drain_erdi_x": ("drain_erdi_usd / exportaciones totales COW", "fracción", "COW+PWT"),
    "drain_erdi_in_usd": ("Drenaje recibido por el país del centro desde la periferia (ERDI)", "millones USD corrientes", "COW+PWT"),
    "drain_erdi_in_gdp": ("drain_erdi_in_usd / PIB nominal", "fracción", "COW+PWT"),
    "drain_px_usd": ("Drenaje conservador: como drain_erdi_usd pero con precios de exportación PWT pl_x",
                     "millones USD corrientes", "COW + PWT pl_x"),
    "drain_px_gdp": ("drain_px_usd / PIB nominal", "fracción", "COW+PWT"),
    "drain_px_x": ("drain_px_usd / exportaciones totales COW", "fracción", "COW+PWT"),
    "drain_px_in_usd": ("Drenaje (pl_x) recibido por el centro", "millones USD corrientes", "COW+PWT"),
    "drain_px_in_gdp": ("drain_px_in_usd / PIB nominal", "fracción", "COW+PWT"),
    "openness_pwt": ("Apertura PWT: csh_x - csh_m (precios PPA corrientes)", "fracción", "PWT 10.01"),
    "openness_cow": ("(exports_cow + imports_cow) / PIB nominal PWT (sólo bienes)", "fracción", "COW+PWT"),
    "tot_pwt": ("Términos de intercambio PWT: pl_x / pl_m", "ratio", "PWT 10.01"),
    "price_level": ("Nivel de precios del PIB PWT pl_gdpo (EE.UU. 2017 = 1); ERDI = 1/price_level", "ratio", "PWT 10.01"),
    "eci_sitc_atlas": ("ECI SITC (Atlas, Hausmann-Hidalgo), 1962-2017", "índice estandarizado", "Harvard Growth Lab vía Gapminder DDF"),
    "eci_hs_atlas": ("ECI HS (Atlas), 1995-2017", "índice estandarizado", "Harvard Growth Lab vía Gapminder DDF"),
    "eci_sitc_gl": ("ECI SITC, Growth Lab rankings 1995-2023", "índice estandarizado", "Harvard Growth Lab"),
    "eci_hs92_gl": ("ECI HS92, Growth Lab rankings 1995-2023", "índice estandarizado", "Harvard Growth Lab"),
    "eci": ("ECI empalmado: eci_sitc_gl, y antes de 1995 eci_sitc_atlas reescalado (MCO en solape)", "índice", "Growth Lab"),
    "patents_res_per_mn": ("Solicitudes de patente de residentes por millón de habitantes", "por millón", "WDI + PWT pop"),
    "fdi_in_stock_proxy": ("Stock aproximado de IED: suma de entradas netas/PIB con depreciación 5 %/año desde 1970", "fracción del PIB", "WDI (derivado)"),
}


def build():
    download()
    OUT.mkdir(parents=True, exist_ok=True)
    pwt = _pwt()
    core = _core(pwt)

    f = dyadic()
    with gzip.open(OUT / "comercio_dyadic.csv.gz", "wt", compresslevel=9) as fh:
        f.to_csv(fh, index=False)

    d = network_indicators(f, core, pwt)
    e, splice = eci()
    tabs = [d, e]
    for var, code, scale, _, _ in WDI:
        name = f"wdi_{code}.csv"
        tabs.append((_wdi_plain if code == "TT.PRI.MRCH.XD.WD" else _wdi)(name, var, scale))

    base = pwt[["iso3", "year", "gdp_usd", "pl_gdpo", "pl_x", "pl_m", "csh_x", "csh_m", "pop"]].merge(core, how="left")
    df = base
    for t in tabs:
        df = df.merge(t, on=["iso3", "year"], how="outer")
    df = df[(df.year >= Y0) & (df.year <= Y1)]
    df = df[df.iso3.str.fullmatch(r"[A-Z]{3}")]
    # sólo países (quitar agregados WDI): mantener los iso3 presentes en PWT, COW o ECI
    keep = set(pwt.iso3) | set(f.iso3_o) | set(e.iso3)
    df = df[df.iso3.isin(keep)].copy()

    for tag, col in [("erdi", "pl_gdpo"), ("px", "pl_x")]:
        v = f"drain_{tag}_usd"
        df[v] = np.where(df.is_core == 0, df[v].fillna(0), np.nan)
        df.loc[df.exports_cow.isna() | df[col].isna(), v] = np.nan
        df[f"drain_{tag}_gdp"] = df[v] / df.gdp_usd
        df[f"drain_{tag}_x"] = df[v] / df.exports_cow.replace(0, np.nan)
        df[f"drain_{tag}_in_gdp"] = df[f"drain_{tag}_in_usd"] / df.gdp_usd
    df["openness_pwt"] = df.csh_x - df.csh_m
    df["openness_cow"] = (df.exports_cow + df.imports_cow) / df.gdp_usd
    df["tot_pwt"] = df.pl_x / df.pl_m
    df["price_level"] = df.pl_gdpo
    df["patents_res_per_mn"] = df.patents_resident / df["pop"]
    df = df.sort_values(["iso3", "year"])
    fs = df[["iso3", "year", "fdi_in_gdp"]].copy()
    fs = fs[fs.year >= 1970]
    first = fs.dropna().groupby("iso3").year.min()
    fs = fs[fs.year >= fs.iso3.map(first)]  # desde la primera observación de cada país
    fs["fdi_in_gdp"] = fs.fdi_in_gdp.fillna(0)
    stock = []
    for _, z in fs.groupby("iso3"):
        s, acc = 0.0, []
        for v in z.fdi_in_gdp:
            s = 0.95 * s + v
            acc.append(s)
        stock.append(pd.Series(acc, index=z.index))
    df["fdi_in_stock_proxy"] = pd.concat(stock) if stock else np.nan

    cols = (["iso3", "year", "is_core", "exports_cow", "imports_cow", "openness_cow", "openness_pwt",
             "trade_gdp_wdi", "share_trade_core", "share_x_to_core", "share_m_from_core",
             "hhi_export_partners", "hhi_import_partners", "n_export_partners",
             "drain_erdi_usd", "drain_erdi_gdp", "drain_erdi_x", "drain_erdi_in_usd", "drain_erdi_in_gdp",
             "drain_px_usd", "drain_px_gdp", "drain_px_x", "drain_px_in_usd", "drain_px_in_gdp",
             "price_level", "tot_pwt", "tot_wdi", "fdi_in_gdp", "fdi_out_gdp", "fdi_in_stock_proxy",
             "eci", "eci_sitc_atlas", "eci_hs_atlas", "eci_sitc_gl", "eci_hs92_gl",
             "hightech_x_share", "manuf_x_share", "patents_resident", "patents_res_per_mn",
             "researchers_per_mn", "rd_gdp"])
    out = df[cols].copy()
    num = out.columns.drop(["iso3", "year"])
    out = out.dropna(subset=num, how="all")
    out[num] = out[num].astype(float).round(6)
    out.to_csv(OUT / "comercio.csv", index=False)

    # diccionario + estimaciones
    meta = {k: {"desc": v[0], "unit": v[1], "source": v[2]} for k, v in DICT.items()}
    for var, code, _, desc, unit in WDI:
        meta[var] = {"desc": desc, "unit": unit, "source": f"WDI {code}"}
    for c in cols[2:]:
        s = out[["year", c]].dropna()
        meta[c]["coverage"] = {"years": [int(s.year.min()), int(s.year.max())] if len(s) else None,
                               "n_obs": int(len(s)), "n_countries": int(out.loc[s.index, "iso3"].nunique())}
    est = {"eci_splice": splice}
    try:
        est["gravity"] = gravity(f, pwt)
    except Exception as exc:  # noqa: BLE001
        est["gravity"] = {"error": str(exc)}
    est["eci_growth"] = eci_growth(e.assign(eci=e.eci), pwt)
    est["drain_erdi_by_decade"] = drain_summary(df, "erdi")
    est["drain_px_by_decade"] = drain_summary(df, "px")
    est["penn_effect"] = penn_effect(pwt)
    est["diffusion"] = diffusion(df, pwt)
    meta_all = {"variables": meta, "estimates": est,
                "dyadic": {"file": "comercio_dyadic.csv.gz",
                           "columns": {"iso3_o": "exportador", "iso3_d": "importador", "year": "año",
                                       "flow": "importaciones de d desde o, millones USD corrientes (COW flow1/flow2)"},
                           "years": [int(f.year.min()), int(f.year.max())], "n_rows": int(len(f))}}
    with open(OUT / "comercio_dict.json", "w") as fh:
        json.dump(meta_all, fh, ensure_ascii=False, indent=1)
    print(f"comercio.csv: {len(out)} filas, {out.iso3.nunique()} países; diádico: {len(f)} filas")
    return out


if __name__ == "__main__":
    build()
