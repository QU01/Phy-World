"""Recursos naturales desagregados: panel pais-anio y series mundiales.

Uso:
    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from sources import recursos; recursos.build()"

Descarga (si faltan) los ficheros crudos a data/raw/recursos/ (los >20 MB a data/raw/recursos/large/,
ignorados por git) y escribe:
    data/processed/sources/recursos.csv          panel iso3 x anio (1950-2019)
    data/processed/sources/recursos_world.csv    series mundiales por anio (1850-2025)
    data/processed/sources/recursos_hubbert.csv  ajustes Hubbert por pais y combustible
    data/processed/sources/recursos_dict.json    diccionario de variables, fuentes y constantes

Unidades energeticas: TWh de energia primaria (convencion OWID/Energy Institute, contenido
energetico directo). Conversiones usadas (aprox.): 1 bep = 1,7 MWh; 1 tep = 11,63 MWh;
1 t de hulla/antracita = 0,60 tep; 1 t de subbituminoso/lignito = 0,33 tep.
Ver docs/research/recursos.md para la discusion y la especificacion recomendada.
"""
from __future__ import annotations

import json
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "recursos"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"

GH = "https://raw.githubusercontent.com"
LFS = "https://media.githubusercontent.com/media"
BS = f"{GH}/ChampionApe/BergAndSorensen2026_CE/HEAD/data"
GAP = f"{GH}/open-numbers/ddf--gapminder--systema_globalis/HEAD/countries-etc-datapoints"
BP14 = f"{GH}/kiln/carbonmap.org/HEAD/data/Raw/BP/BP-Statistical_Review_of_world_energy_2014_workbook"

# fichero local -> (url, descripcion)
SOURCES = {
    "owid-energy-data.csv": (f"{GH}/owid/energy-data/master/owid-energy-data.csv",
                             "OWID energy-data (Energy Institute Statistical Review + Shift/EIA), produccion TWh 1900-"),
    "gapminder_oil_proved_reserves_total.csv": (
        f"{GAP}/ddf--datapoints--oil_proved_reserves_total--by--geo--time.csv",
        "Reservas probadas de petroleo (tep), Energy Institute via Gapminder, 1980-2020"),
    "gapminder_natural_gas_proved_reserves_total.csv": (
        f"{GAP}/ddf--datapoints--natural_gas_proved_reserves_total--by--geo--time.csv",
        "Reservas probadas de gas (tep), Energy Institute via Gapminder, 1980-2020"),
    "bp2014_coal_reserves.csv": (f"{BP14}/Coal%20-%20Reserves.csv",
                                 "BP Statistical Review 2014: reservas probadas de carbon a fin de 2013 (Mt)"),
    "bp2014_oil_prices_since_1861.csv": (f"{BP14}/Oil%20-%20Crude%20prices%20since%201861.csv",
                                         "BP Statistical Review 2014: precio del crudo 1861-2013 (US$ corrientes y de 2013)"),
    "oil_price_nominal_1861_2021_owid.csv": (f"{GH}/malrepos/FirstGroup/HEAD/crude-oil-prices.csv",
                                             "OWID (EI Statistical Review): precio del crudo 1861-2021, US$/bbl corrientes"),
    "giant_fields_2018.csv": (
        f"{GH}/alexis-ribal/giant-oil-and-gas-field-discoveries/HEAD/giant_fields_2018.csv",
        "Cust, Mihalyi & Rivera-Ballesteros: extension de Horn (2014), campos gigantes 1868-2019"),
    "gycpi_2011.csv": (f"{GH}/christophe-gouel/storage-estimation/HEAD/Data/gycpi-2011-01.csv",
                       "Grilli-Yang commodity price index (actualizacion Pfaffenzeller et al. 2007), 1900-2011"),
    "jacks2019_real_commodity_prices_1850_2025.xlsx": (
        f"{BS}/raw/jacks2019/Real-commodity-prices-1850-2025.xlsx",
        "Jacks (2019) precios reales de 42 materias primas 1850-2025 (1900=100)"),
    "worldbank_cmo_historical_annual.xlsx": (f"{BS}/raw/worldbank_cmo/CMO-Historical-Data-Annual.xlsx",
                                             "Banco Mundial Pink Sheet anual 1960-2025 (nominal y real 2010 US$)"),
    "gmfd_unep_irp_1970_2019.csv": (f"{BS}/raw/unep_irp/global-material-flows-database.csv",
                                    "UN IRP Global Material Flows Database, extraccion domestica (t), 1970-2019"),
    "usgs_ds140_long.csv": (f"{BS}/interim/b3_usgs_ds140.csv",
                            "USGS Data Series 140 (procesado por Berg & Sorensen 2026): produccion mundial de minerales 1900-2022"),
    "urr_gea2012_bergsorensen.csv": (f"{BS}/interim/b3_urr.csv",
                                     "Rogner et al. (2012) GEA cap. 7 tabla 7.1 + USGS MCS: reservas/recursos mundiales"),
    "oregrades_bergsorensen.csv": (f"{BS}/interim/b3_oregrades.csv", "Leyes de mineral (Calvo 2016, Mudd 2010)"),
    "exploration_bergsorensen.csv": (f"{BS}/interim/b3_exploration.csv", "Costes/descubrimientos mineros (Schodde)"),
    "usgs_mcs2026_bergsorensen.csv": (f"{BS}/interim/b3_usgs_mcs.csv", "USGS Mineral Commodity Summaries 2026 (transcrito)"),
}
WDI_BULK = (f"{LFS}/tejas-rawal/DATS6101-Project1/HEAD/data/WDIData.csv",
            "WDI bulk (edicion 2022, 1960-2021), 208 MB, git-lfs")
WDI_SUBSET = RAW / "wdi_recursos_subset.csv"
NAC_FILE = RAW / "nacionalizaciones_curado.csv"

WDI_VARS = {
    "NY.GDP.PETR.RT.ZS": ("renta_petroleo_pct_pib", "Rentas del petroleo (% PIB)"),
    "NY.GDP.NGAS.RT.ZS": ("renta_gas_pct_pib", "Rentas del gas natural (% PIB)"),
    "NY.GDP.COAL.RT.ZS": ("renta_carbon_pct_pib", "Rentas del carbon (% PIB)"),
    "NY.GDP.MINR.RT.ZS": ("renta_minerales_pct_pib", "Rentas minerales (% PIB)"),
    "NY.GDP.FRST.RT.ZS": ("renta_forestal_pct_pib", "Rentas forestales (% PIB)"),
    "NY.GDP.TOTL.RT.ZS": ("renta_total_pct_pib", "Rentas totales de recursos naturales (% PIB)"),
    "TX.VAL.FUEL.ZS.UN": ("export_combustibles_pct_merc", "Exportaciones de combustibles (% exportaciones de mercancias)"),
    "TX.VAL.MMTL.ZS.UN": ("export_minerales_metales_pct_merc", "Exportaciones de minerales y metales (% exportaciones de mercancias)"),
    "NY.ADJ.DNGY.GN.ZS": ("agotamiento_energia_pct_rnb", "Ahorro ajustado: agotamiento energetico (% RNB)"),
    "NY.ADJ.DMIN.GN.ZS": ("agotamiento_minerales_pct_rnb", "Ahorro ajustado: agotamiento mineral (% RNB)"),
    "NY.ADJ.DFOR.GN.ZS": ("agotamiento_forestal_pct_rnb", "Ahorro ajustado: agotamiento forestal neto (% RNB)"),
    "NY.GDP.MKTP.CD": ("pib_usd_corr", "PIB (US$ corrientes)"),
    "FP.CPI.TOTL": ("ipc_2010", "IPC (2010=100); solo se usa el de EE.UU. como deflactor"),
}

BOE_TWH = 1.7e-6          # TWh por barril equivalente de petroleo
TOE_TWH = 11.63e-6        # TWh por tonelada equivalente de petroleo
HARD_COAL_TOE = 0.60      # tep por tonelada de hulla/antracita
SOFT_COAL_TOE = 0.33      # tep por tonelada de subbituminoso/lignito
Y0, Y1 = 1950, 2019
# precios 2019 por TWh (M US$): crudo 64,2 $/bbl / 1,7 MWh; gas ~5 $/MMBtu; carbon ~78 $/t / 6,98 MWh
PRICE_MUSD_TWH = {"oil": 37.8, "gas": 17.1, "coal": 11.2}

PREDECESSORS = {
    "USSR": ["RUS", "UKR", "KAZ", "AZE", "TKM", "UZB", "BLR", "GEO", "KGZ", "TJK", "LTU", "LVA", "EST",
             "MDA", "ARM"],
    "Czechoslovakia": ["CZE", "SVK"],
    "Yugoslavia": ["SRB", "HRV", "BIH", "SVN", "MKD", "MNE"],
}

BP_COAL_NAMES = {
    "US": "USA", "Canada": "CAN", "Mexico": "MEX", "Brazil": "BRA", "Colombia": "COL", "Venezuela": "VEN",
    "Bulgaria": "BGR", "Czech Republic": "CZE", "Germany": "DEU", "Greece": "GRC", "Hungary": "HUN",
    "Kazakhstan": "KAZ", "Poland": "POL", "Romania": "ROU", "Russian Federation": "RUS", "Spain": "ESP",
    "Turkey": "TUR", "Ukraine": "UKR", "United Kingdom": "GBR", "South Africa": "ZAF", "Zimbabwe": "ZWE",
    "Australia": "AUS", "China": "CHN", "India": "IND", "Indonesia": "IDN", "Japan": "JPN",
    "New Zealand": "NZL", "North Korea": "PRK", "Pakistan": "PAK", "South Korea": "KOR", "Thailand": "THA",
    "Vietnam": "VNM",
}


# ---------------------------------------------------------------------------------------------
# descarga
# ---------------------------------------------------------------------------------------------
def _get(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    head = tmp.read_bytes()[:200]
    if head.startswith(b"version https://git-lfs") or head.startswith(b"404"):
        tmp.unlink()
        raise RuntimeError(f"descarga invalida (lfs/404): {url}")
    tmp.rename(dst)


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    gi = LARGE / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    for name, (url, _) in SOURCES.items():
        if not (RAW / name).exists():
            print("descargando", name)
            _get(url, RAW / name)
    if not WDI_SUBSET.exists():
        bulk = LARGE / "WDIData.csv"
        if not bulk.exists():
            print("descargando WDI bulk (208 MB)")
            _get(WDI_BULK[0], bulk)
        keep = []
        for ch in pd.read_csv(bulk, chunksize=50_000, encoding="utf-8-sig"):
            keep.append(ch[ch["Indicator Code"].isin(WDI_VARS)])
        sub = pd.concat(keep)
        sub = sub.loc[:, ~sub.columns.astype(str).str.startswith("Unnamed")]
        sub.to_csv(WDI_SUBSET, index=False)
    if not NAC_FILE.exists():
        raise FileNotFoundError(f"falta {NAC_FILE} (tabla curada, versionada en git)")


# ---------------------------------------------------------------------------------------------
# utilidades
# ---------------------------------------------------------------------------------------------
def _iso(names) -> list:
    import logging

    import country_converter as coco
    logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    out = coco.convert(list(names), to="ISO3", not_found=None)
    return out if isinstance(out, list) else [out]


def _country_universe() -> set:
    pwt = pd.read_csv(ROOT / "data" / "raw" / "pwt1001.csv", usecols=["isocode"]) \
        if (ROOT / "data" / "raw" / "pwt1001.csv").exists() else pd.DataFrame({"isocode": []})
    owid = pd.read_csv(RAW / "owid-energy-data.csv", usecols=["iso_code"])
    s = set(pwt.isocode.dropna()) | set(owid.iso_code.dropna())
    return {c for c in s if isinstance(c, str) and len(c) == 3 and not c.startswith("OWID")}


# ---------------------------------------------------------------------------------------------
# bloques del panel
# ---------------------------------------------------------------------------------------------
def _production() -> pd.DataFrame:
    """Produccion (TWh) 1900-2019 con relleno hacia delante 2017-2019 donde la serie termina en 2016."""
    d = pd.read_csv(RAW / "owid-energy-data.csv",
                    usecols=["country", "iso_code", "year", "oil_production", "gas_production",
                             "coal_production"])
    d = d[(d.year >= 1900) & (d.year <= Y1)]
    pred = d[d.country.isin(PREDECESSORS)]
    d = d[d.iso_code.notna() & ~d.iso_code.astype(str).str.startswith("OWID")]
    d = d.rename(columns={"iso_code": "iso3"}).drop(columns="country")
    full = pd.MultiIndex.from_product([d.iso3.unique(), range(1900, Y1 + 1)], names=["iso3", "year"])
    d = d.set_index(["iso3", "year"]).reindex(full).sort_index()
    out = pd.DataFrame(index=d.index)
    for f in ["oil", "gas", "coal"]:
        s = d[f"{f}_production"].copy()
        s_orig = s.copy()
        # reparto de la produccion de estados predecesores (URSS, Checoslovaquia, Yugoslavia) entre
        # sucesores para los anios previos al inicio de la serie del sucesor, con la cuota media de
        # los 3 primeros anios del sucesor.
        for pname, succ in PREDECESSORS.items():
            ps = pred[pred.country == pname].set_index("year")[f"{f}_production"]
            if ps.dropna().empty:
                continue
            for iso3 in succ:
                if iso3 not in s.index.get_level_values(0):
                    continue
                si = s_orig.loc[iso3]
                pos = si[si > 0]
                if pos.empty:
                    continue
                first = int(pos.index[0])
                yrs = [y for y in range(first, first + 3) if y in ps.index and ps.get(y, 0) > 0]
                if yrs:
                    share = float(np.nansum([si.get(y, np.nan) for y in yrs]) / np.nansum([ps[y] for y in yrs]))
                else:
                    tot_succ = sum(float(s_orig.loc[c].get(first, 0) or 0) for c in succ
                                   if c in s.index.get_level_values(0))
                    share = float(si.get(first, 0) or 0) / tot_succ if tot_succ > 0 else 0.0
                for y in range(1900, first):
                    cur = s.loc[(iso3, y)]
                    if y in ps.index and ps[y] == ps[y] and not (cur > 0):
                        s.loc[(iso3, y)] = share * ps[y]
        last = s.groupby(level=0).transform(lambda x: x.last_valid_index()[1] if x.notna().any() else np.nan)
        yr = s.index.get_level_values(1)
        fill = s.groupby(level=0).ffill()
        filled = s.isna() & (last >= 2014) & (yr > last) & (yr <= Y1)
        s2 = s.where(~filled, fill)
        out[f"{f}_prod_twh"] = s2
        out[f"{f}_prod_relleno"] = filled.astype(int)
        # acumulada desde 1900 (faltantes = 0: casi siempre paises sin produccion en ese momento)
        out[f"{f}_cum_twh"] = s2.fillna(0).groupby(level=0).cumsum()
    out["fosil_prod_twh"] = out[["oil_prod_twh", "gas_prod_twh", "coal_prod_twh"]].sum(axis=1, min_count=1)
    out["fosil_cum_twh"] = out[["oil_cum_twh", "gas_cum_twh", "coal_cum_twh"]].sum(axis=1)
    out["oil_prod_mbep"] = out.oil_prod_twh / BOE_TWH / 1e6
    out["oil_cum_gbep"] = out.oil_cum_twh / BOE_TWH / 1e9
    return out.reset_index()


def _reserves() -> pd.DataFrame:
    frames = []
    for f, var in [("gapminder_oil_proved_reserves_total.csv", "oil_reservas_twh"),
                   ("gapminder_natural_gas_proved_reserves_total.csv", "gas_reservas_twh")]:
        g = pd.read_csv(RAW / f)
        g.columns = ["geo", "year", "v"]
        g = g[g.geo != "ussr"]
        g["iso3"] = g.geo.str.upper()
        g[var] = g.v * TOE_TWH
        frames.append(g.set_index(["iso3", "year"])[[var]])
    r = pd.concat(frames, axis=1).reset_index()
    return r


def _coal_reserves() -> pd.DataFrame:
    raw = pd.read_csv(RAW / "bp2014_coal_reserves.csv", header=None, skiprows=4, dtype=str)
    rows = []
    for _, r in raw.iterrows():
        name = str(r[0]).strip()
        if name in BP_COAL_NAMES:
            hard = pd.to_numeric(r[1], errors="coerce")
            soft = pd.to_numeric(r[2], errors="coerce")
            hard = 0.0 if np.isnan(hard) else hard
            soft = 0.0 if np.isnan(soft) else soft
            twh = (hard * HARD_COAL_TOE + soft * SOFT_COAL_TOE) * 1e6 * TOE_TWH
            rows.append({"iso3": BP_COAL_NAMES[name], "coal_reservas_2013_mt": hard + soft,
                         "coal_reservas_2013_twh": twh})
    return pd.DataFrame(rows)


def _giants() -> tuple[pd.DataFrame, pd.DataFrame]:
    g = pd.read_csv(RAW / "giant_fields_2018.csv", low_memory=False)
    g = g[g.ISO.notna() & (g.ISO.astype(str).str.len() == 3)]
    g["eur"] = pd.to_numeric(g.EUR_MMBOE, errors="coerce")
    g["npv_gdp"] = pd.to_numeric(g.NPV_GDP_n, errors="coerce")
    g["is_oil"] = g.FIELD_TYPE.astype(str).str.lower().str.startswith("oil")
    agg = g.groupby(["ISO", "year"]).agg(
        gigantes_n=("FIELD_ID", "count"),
        gigantes_eur_mbep=("eur", "sum"),
        gigantes_eur_petroleo_mbep=("eur", lambda s: s[g.loc[s.index, "is_oil"]].sum()),
        gigantes_npv_pct_pib=("npv_gdp", "sum"),
    ).reset_index().rename(columns={"ISO": "iso3"})
    agg["gigantes_eur_twh"] = agg.gigantes_eur_mbep * 1e6 * BOE_TWH
    # mundo (incluye descubrimientos conjuntos sin ISO)
    ga = pd.read_csv(RAW / "giant_fields_2018.csv", low_memory=False)
    ga["eur"] = pd.to_numeric(ga.EUR_MMBOE, errors="coerce")
    ga["is_oil"] = ga.FIELD_TYPE.astype(str).str.lower().str.startswith("oil")
    w = ga.groupby("year").agg(mundo_gigantes_n=("FIELD_ID", "count"),
                               mundo_gigantes_eur_mbep=("eur", "sum"))
    w["mundo_gigantes_eur_petroleo_mbep"] = ga[ga.is_oil].groupby("year").eur.sum()
    return agg, w.reset_index()


def _wdi() -> tuple[pd.DataFrame, pd.Series]:
    w = pd.read_csv(WDI_SUBSET)
    years = [c for c in w.columns if c.isdigit()]
    long = w.melt(id_vars=["Country Code", "Indicator Code"], value_vars=years,
                  var_name="year", value_name="v")
    long["year"] = long.year.astype(int)
    p = long.pivot_table(index=["Country Code", "year"], columns="Indicator Code", values="v")
    uscpi = p.loc["USA", "FP.CPI.TOTL"] if "FP.CPI.TOTL" in p else pd.Series(dtype=float)
    p = p.drop(columns=["FP.CPI.TOTL"], errors="ignore")
    p = p.rename(columns={k: v[0] for k, v in WDI_VARS.items()})
    p.index.names = ["iso3", "year"]
    return p.reset_index(), uscpi


def _gmfd() -> pd.DataFrame:
    g = pd.read_csv(RAW / "gmfd_unep_irp_1970_2019.csv")
    g = g[g["Flow code"] == "DE"]
    cats = {"Fossil fuels": "de_fosiles_t", "Metal ores": "de_metales_t",
            "Non-metallic minerals": "de_no_metalicos_t", "Biomass": "de_biomasa_t"}
    g = g[g.Category.isin(cats)]
    names = g.Country.unique()
    skip = {"USSR", "Czechoslovakia", "Yugoslavia", "Ethiopia (Former)", "Sudan (Former)", "North Yemen",
            "South Yemen", "Netherlands Antilles", "World", "Africa", "Europe", "North America", "EECCA",
            "West Asia", "Asia + Pacific", "Latin America + Caribbean"}
    iso = dict(zip(names, _iso(names)))
    g["iso3"] = g.Country.map(lambda n: None if (n in skip or n.startswith("Rest of")) else iso.get(n))
    g = g[g.iso3.notna()]
    years = [c for c in g.columns if c.isdigit()]
    long = g.melt(id_vars=["iso3", "Category"], value_vars=years, var_name="year", value_name="v")
    long["year"] = long.year.astype(int)
    long["v"] = pd.to_numeric(long.v, errors="coerce")
    p = long.pivot_table(index=["iso3", "year"], columns="Category", values="v", aggfunc="sum")
    return p.rename(columns=cats).reset_index()


def _nationalizations(universe_years: pd.DataFrame) -> pd.DataFrame:
    n = pd.read_csv(NAC_FILE)
    n = n[n.tipo != "privatizacion"]
    ev = n.groupby(["iso3", "year", "sector"]).participacion_estatal_post.max().unstack("sector")
    ev.columns = [f"nac_{c}" for c in ev.columns]
    ev = ev.reset_index()
    p = universe_years.merge(ev, on=["iso3", "year"], how="left")
    for c in ["nac_petroleo", "nac_mineria", "nac_carbon"]:
        if c not in p:
            p[c] = np.nan
    p["nac_evento_petroleo"] = p.nac_petroleo.notna().astype(int)
    p["nac_evento_mineria"] = p[["nac_mineria", "nac_carbon"]].notna().any(axis=1).astype(int)
    # participacion estatal acumulada (maximo historico, arrastrado; 1938+ incluidos via eventos previos)
    allev = n[n.sector == "petroleo"].sort_values("year")
    share = {}
    for iso3, grp in allev.groupby("iso3"):
        share[iso3] = list(zip(grp.year, grp.participacion_estatal_post))

    def st(row):
        s = 0.0
        for y, v in share.get(row.iso3, []):
            if y <= row.year:
                s = v  # el ultimo evento manda (permite reversiones, p.ej. Iran 1954)
        return s

    p["estado_share_petroleo_eventos"] = p.apply(st, axis=1)
    p = p.drop(columns=["nac_carbon"]).rename(columns={"nac_petroleo": "nac_petroleo_share_post",
                                                       "nac_mineria": "nac_mineria_share_post"})
    return p


# ---------------------------------------------------------------------------------------------
# series mundiales
# ---------------------------------------------------------------------------------------------
def _world(uscpi: pd.Series, giants_w: pd.DataFrame) -> pd.DataFrame:
    idx = pd.Index(range(1850, 2026), name="year")
    W = pd.DataFrame(index=idx)
    # petroleo nominal (EI via OWID) y real
    o = pd.read_csv(RAW / "oil_price_nominal_1861_2021_owid.csv")
    W["petroleo_precio_nominal_usd_bbl"] = o.set_index("Year").iloc[:, -1]
    bp = pd.read_csv(RAW / "bp2014_oil_prices_since_1861.csv", skiprows=4, header=None,
                     names=["year", "nom", "real2013"])
    bp = bp[pd.to_numeric(bp.year, errors="coerce").notna()].astype(float)
    bp["year"] = bp.year.astype(int)
    bp = bp.set_index("year")
    defl = (bp.nom / bp.real2013)  # nivel de precios EE.UU. relativo a 2013 (BP usa IPC EE.UU.)
    cpi = uscpi.dropna()
    if len(cpi):
        cpi_rel = cpi / cpi.loc[2013]
        defl = pd.concat([defl[defl.index <= 2013], cpi_rel[cpi_rel.index > 2013]])
    defl = defl / defl.loc[2019]
    W["deflactor_eeuu_2019"] = defl
    W["petroleo_precio_real_usd2019_bbl"] = W.petroleo_precio_nominal_usd_bbl / W.deflactor_eeuu_2019
    # Jacks (2019)
    jx = RAW / "jacks2019_real_commodity_prices_1850_2025.xlsx"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = pd.read_excel(jx, sheet_name="Commodities", header=1)
        ind = pd.read_excel(jx, sheet_name="Indices", header=None, skiprows=2)
        sub = pd.read_excel(jx, sheet_name="Sub-indices", header=0)
    c = c.rename(columns={"year": "year"}).set_index("year")
    c.columns = ["jacks_" + str(k).strip().lower().replace(" ", "_").replace(".", "") for k in c.columns]
    W = W.join(c.apply(pd.to_numeric, errors="coerce"))
    ind = ind.iloc[:, :4]
    ind.columns = ["year", "jacks_indice_vp1975", "jacks_indice_vp2019", "jacks_indice_igual"]
    ind = ind[pd.to_numeric(ind.year, errors="coerce").notna()].astype(float)
    ind["year"] = ind.year.astype(int)
    W = W.join(ind.set_index("year"))
    sub = sub.iloc[:, :4]
    sub.columns = ["year", "jacks_sub_cultivados", "jacks_sub_subsuelo", "jacks_sub_subsuelo_sin_energia"]
    sub = sub[pd.to_numeric(sub.year, errors="coerce").notna()].astype(float)
    sub["year"] = sub.year.astype(int)
    W = W.join(sub.set_index("year"))
    # Grilli-Yang
    gy = pd.read_csv(RAW / "gycpi_2011.csv")
    gy = gy[pd.to_numeric(gy.Year, errors="coerce").notna()].copy()
    gy["year"] = gy.Year.astype(int)
    gy = gy.set_index("year")
    for col, name in [("GYCPI", "gycpi_nominal"), ("GYCPIM", "gycpi_metales_nominal"), ("MUV", "muv_g5")]:
        W[name] = pd.to_numeric(gy[col], errors="coerce")
    W["gycpi_real"] = W.gycpi_nominal / W.muv_g5 * 100
    W["gycpi_metales_real"] = W.gycpi_metales_nominal / W.muv_g5 * 100
    # Pink Sheet (real, US$ de 2010)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cm = pd.read_excel(RAW / "worldbank_cmo_historical_annual.xlsx", sheet_name="Annual Prices (Real)",
                           header=None)
    hdr = cm.iloc[6].astype(str).str.strip()
    body = cm.iloc[8:].copy()
    body = body[pd.to_numeric(body[0], errors="coerce").notna()]
    body.index = body[0].astype(int)
    want = {"Crude oil, average": "cmo_petroleo_real2010", "Coal, Australian": "cmo_carbon_aus_real2010",
            "Natural gas, US": "cmo_gas_eeuu_real2010", "Natural gas, Europe": "cmo_gas_europa_real2010",
            "Copper": "cmo_cobre_real2010", "Iron ore, cfr spot": "cmo_hierro_real2010",
            "Aluminum": "cmo_aluminio_real2010", "Gold": "cmo_oro_real2010", "Tin": "cmo_estano_real2010",
            "Nickel": "cmo_niquel_real2010", "Zinc": "cmo_zinc_real2010"}
    for k, v in want.items():
        j = np.where(hdr.values == k)[0]
        if len(j):
            W[v] = pd.to_numeric(body[j[0]], errors="coerce")
    # USGS DS140: produccion mundial de metales clave
    u = pd.read_csv(RAW / "usgs_ds140_long.csv")
    metals = {"copper": "cobre", "iron_ore": "hierro", "bauxite_alumina__bauxite": "bauxita",
              "aluminum": "aluminio", "gold": "oro", "silver": "plata", "tin": "estano", "zinc": "zinc",
              "lead": "plomo", "nickel": "niquel", "phosphate__phosphate_rock": "fosfato"}
    for series in ["world_mine_production", "world_production"]:
        uu = u[(u.series == series) & u.commodity.isin(metals)]
        for com, grp in uu.groupby("commodity"):
            col = f"usgs_prod_mundial_{metals[com]}_t"
            if col not in W:
                W[col] = grp.groupby("year").value.first()
    # produccion fosil mundial (OWID) y acumulada
    d = pd.read_csv(RAW / "owid-energy-data.csv",
                    usecols=["country", "year", "oil_production", "gas_production", "coal_production"])
    wd = d[d.country == "World"].set_index("year")
    for f in ["oil", "gas", "coal"]:
        W[f"mundo_{f}_prod_twh"] = wd[f"{f}_production"]
        W[f"mundo_{f}_cum_twh"] = wd[f"{f}_production"].fillna(0).cumsum().where(wd[f"{f}_production"].notna())
    # descubrimientos gigantes
    W = W.join(giants_w.set_index("year"))
    for c in ["mundo_gigantes_n", "mundo_gigantes_eur_mbep", "mundo_gigantes_eur_petroleo_mbep"]:
        W.loc[(W.index >= 1868) & (W.index <= 2019), c] = W.loc[(W.index >= 1868) & (W.index <= 2019), c].fillna(0)
    W["mundo_gigantes_eur_cum_mbep"] = W.mundo_gigantes_eur_mbep.fillna(0).cumsum().where(W.index <= 2019)
    return W.reset_index()


# ---------------------------------------------------------------------------------------------
# ajustes Hubbert
# ---------------------------------------------------------------------------------------------
def _hubbert(prod: pd.DataFrame, res: pd.DataFrame, coal_res: pd.DataFrame) -> pd.DataFrame:
    """Dos estimadores por pais y combustible:
    (a) linealizacion de Hubbert P/Q = r (1 - Q/URR) por MCO sobre anios con Q >= 20% de Q_2019;
    (b) r condicional al URR 'proxy' = Q_2019 + reservas_2019: r = mediana de P/[Q(1-Q/URR)], 1965-2019.
    """
    from scipy import stats
    rows = []
    r19 = res[res.year == 2019].set_index("iso3")
    cr = coal_res.set_index("iso3")
    for f in ["oil", "gas", "coal"]:
        for iso3, g in prod.groupby("iso3"):
            g = g.set_index("year")
            P, Q = g[f"{f}_prod_twh"], g[f"{f}_cum_twh"]
            q19 = Q.get(2019, np.nan)
            if not (q19 > 500):  # ~0,3 Gbep; ignorar productores menores
                continue
            reserves = (r19[f"{f}_reservas_twh"].get(iso3, np.nan) if f != "coal"
                        else cr["coal_reservas_2013_twh"].get(iso3, np.nan))
            urr_proxy = q19 + reserves if reserves == reserves else np.nan
            m = (Q >= 0.2 * q19) & P.notna() & (Q > 0) & (Q.index <= 2019)
            r_lin = urr_lin = r2 = np.nan
            if m.sum() >= 10:
                y, x = (P[m] / Q[m]).values, Q[m].values
                lr = stats.linregress(x, y)
                if lr.intercept > 0 and lr.slope < 0:
                    r_lin, urr_lin, r2 = lr.intercept, -lr.intercept / lr.slope, lr.rvalue ** 2
            r_cond = np.nan
            if urr_proxy == urr_proxy:
                mm = (Q.index >= 1965) & (Q.index <= 2019) & (Q > 0.02 * urr_proxy) & P.notna()
                if mm.sum() >= 10:
                    r_cond = float(np.median(P[mm] / (Q[mm] * (1 - Q[mm] / urr_proxy))))
            pk = P.loc[:2019].idxmax() if P.notna().any() else np.nan
            rows.append({"iso3": iso3, "recurso": f, "q_1950_twh": Q.get(1950, np.nan), "q_2019_twh": q19,
                         "reservas_twh": reserves, "urr_proxy_twh": urr_proxy,
                         "x_1950_proxy": Q.get(1950, np.nan) / urr_proxy if urr_proxy == urr_proxy else np.nan,
                         "x_2019_proxy": q19 / urr_proxy if urr_proxy == urr_proxy else np.nan,
                         "r_linealizacion": r_lin, "urr_linealizacion_twh": urr_lin, "r2_linealizacion": r2,
                         "r_condicional_urr_proxy": r_cond, "anio_pico_produccion": pk,
                         "prod_pico_twh": P.max()})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------
def build() -> pd.DataFrame:
    download()
    OUT.mkdir(parents=True, exist_ok=True)
    universe = sorted(_country_universe())
    base = pd.DataFrame([(c, y) for c in universe for y in range(Y0, Y1 + 1)], columns=["iso3", "year"])

    prod = _production()
    res = _reserves()
    coal_res = _coal_reserves()
    giants, giants_w = _giants()
    wdi, uscpi = _wdi()
    gmfd = _gmfd()

    p = base.merge(prod, on=["iso3", "year"], how="left")
    p = p.merge(res, on=["iso3", "year"], how="left")
    p = p.merge(coal_res, on="iso3", how="left")
    p = p.merge(giants, on=["iso3", "year"], how="left")
    for c in ["gigantes_n", "gigantes_eur_mbep", "gigantes_eur_petroleo_mbep", "gigantes_eur_twh"]:
        p[c] = p[c].fillna(0)
    # EUR acumulado de gigantes desde 1868 (incluye pre-1950)
    gcum = giants.sort_values("year").copy()
    gcum["cum"] = gcum.groupby("iso3").gigantes_eur_mbep.cumsum()
    p = pd.merge_asof(p.sort_values("year"), gcum[["iso3", "year", "cum"]].sort_values("year"),
                      on="year", by="iso3", direction="backward").rename(
        columns={"cum": "gigantes_eur_cum_mbep"})
    p["gigantes_eur_cum_mbep"] = p.gigantes_eur_cum_mbep.fillna(0)
    p = p.merge(wdi, on=["iso3", "year"], how="left")
    p = p.merge(gmfd, on=["iso3", "year"], how="left")
    p = _nationalizations(p)

    # derivados: composicion de rentas, R/P, URR proxy y fraccion agotada
    comps = ["renta_petroleo_pct_pib", "renta_gas_pct_pib", "renta_carbon_pct_pib",
             "renta_minerales_pct_pib", "renta_forestal_pct_pib"]
    tot = p[comps].sum(axis=1, min_count=1)
    for c in comps:
        p[c.replace("_pct_pib", "_share")] = (p[c] / tot).where(tot > 0)
    p["renta_no_renovable_pct_pib"] = p[comps[:4]].sum(axis=1, min_count=1)
    for f in ["oil", "gas"]:
        p[f"{f}_rp_anios"] = (p[f"{f}_reservas_twh"] / p[f"{f}_prod_twh"]).where(p[f"{f}_prod_twh"] > 0)
    last = p[p.year == Y1].set_index("iso3")
    r19 = res[res.year == 2019].set_index("iso3")
    urr = pd.DataFrame(index=last.index)
    urr["oil_urr_proxy_twh"] = last.oil_cum_twh + r19.oil_reservas_twh.reindex(last.index)
    urr["gas_urr_proxy_twh"] = last.gas_cum_twh + r19.gas_reservas_twh.reindex(last.index)
    urr["coal_urr_proxy_twh"] = last.coal_cum_twh + last.coal_reservas_2013_twh
    urr["fosil_urr_proxy_twh"] = urr[["oil_urr_proxy_twh", "gas_urr_proxy_twh", "coal_urr_proxy_twh"]].sum(
        axis=1, min_count=1)
    p = p.merge(urr.reset_index(), on="iso3", how="left")
    for f in ["oil", "gas", "coal", "fosil"]:
        p[f"{f}_x_proxy"] = (p[f"{f}_cum_twh"] / p[f"{f}_urr_proxy_twh"]).where(p[f"{f}_urr_proxy_twh"] > 0)
    # agregado ponderado por valor (precios de 2019 por TWh): recurso generico del modelo
    val_urr = sum(p[f"{f}_urr_proxy_twh"].fillna(0) * PRICE_MUSD_TWH[f] for f in PRICE_MUSD_TWH)
    val_cum = sum(p[f"{f}_cum_twh"].fillna(0) * PRICE_MUSD_TWH[f] for f in PRICE_MUSD_TWH)
    p["fosil_valor_urr_proxy_musd2019"] = val_urr.where(val_urr > 0)
    p["fosil_valor_cum_musd2019"] = val_cum
    p["fosil_x_valor"] = (val_cum / val_urr).where(val_urr > 0)

    p = p.sort_values(["iso3", "year"]).reset_index(drop=True)
    keep_countries = p.groupby("iso3")[["oil_prod_twh", "renta_total_pct_pib", "de_fosiles_t"]].apply(
        lambda g: g.notna().any().any())
    p = p[p.iso3.isin(keep_countries[keep_countries].index)]
    p.to_csv(OUT / "recursos.csv", index=False)

    W = _world(uscpi, giants_w)
    W.to_csv(OUT / "recursos_world.csv", index=False)

    hub = _hubbert(prod, res, coal_res)
    hub.to_csv(OUT / "recursos_hubbert.csv", index=False)

    _write_dict(p, W, hub)
    print(f"recursos.csv: {p.shape}, paises={p.iso3.nunique()}; recursos_world.csv: {W.shape}; "
          f"recursos_hubbert.csv: {hub.shape}")
    return p


def _cov(df: pd.DataFrame, c: str, key: str = "iso3") -> dict:
    s = df[df[c].notna()]
    if s.empty:
        return {"n": 0}
    out = {"n": int(len(s)), "anio_min": int(s.year.min()), "anio_max": int(s.year.max())}
    if key in df:
        out["paises"] = int(s[key].nunique())
    return out


def _write_dict(p: pd.DataFrame, W: pd.DataFrame, hub: pd.DataFrame) -> None:
    desc = {
        "iso3": "Codigo ISO3", "year": "Anio",
        "oil_prod_twh": "Produccion de petroleo (crudo+LGN), TWh [OWID/EI; 1900-1964 Shift/Etemad-Luciani]",
        "gas_prod_twh": "Produccion de gas natural, TWh [OWID/EI]",
        "coal_prod_twh": "Produccion de carbon, TWh [OWID/EI]",
        "oil_prod_relleno": "1 si oil_prod_twh 2017-2019 es arrastre del ultimo dato (serie Shift termina en 2016)",
        "gas_prod_relleno": "idem gas", "coal_prod_relleno": "idem carbon",
        "oil_cum_twh": "Produccion acumulada de petroleo desde 1900, TWh (pre-1900 omitido: <1% del total mundial)",
        "gas_cum_twh": "Produccion acumulada de gas desde 1900, TWh",
        "coal_cum_twh": "Produccion acumulada de carbon desde 1900, TWh (omite ~40 Gt pre-1900, sobre todo GBR/DEU/USA/FRA/BEL)",
        "fosil_prod_twh": "Suma petroleo+gas+carbon, TWh", "fosil_cum_twh": "Acumulada fosil desde 1900, TWh",
        "oil_prod_mbep": "Produccion de petroleo, millones de bep (1 bep = 1,7 MWh)",
        "oil_cum_gbep": "Acumulada de petroleo, miles de millones de bep",
        "oil_reservas_twh": "Reservas probadas de petroleo, TWh (EI via Gapminder, 1980-2020; incluye arenas bituminosas CAN y Orinoco VEN)",
        "gas_reservas_twh": "Reservas probadas de gas, TWh (EI via Gapminder, 1980-2020)",
        "coal_reservas_2013_mt": "Reservas probadas de carbon a fin de 2013, Mt (BP 2014 / WEC 2013), invariante en el tiempo",
        "coal_reservas_2013_twh": "Idem en TWh (0,60 tep/t hulla; 0,33 tep/t lignito)",
        "oil_rp_anios": "Ratio reservas/produccion petroleo (anios)", "gas_rp_anios": "Ratio R/P gas (anios)",
        "oil_urr_proxy_twh": "URR proxy = acumulada 1900-2019 + reservas probadas 2019 (cota inferior: sin crecimiento de reservas ni descubrimientos futuros)",
        "gas_urr_proxy_twh": "idem gas", "coal_urr_proxy_twh": "idem carbon (reservas 2013)",
        "fosil_urr_proxy_twh": "Suma de URR proxy fosiles",
        "oil_x_proxy": "Fraccion agotada x = acumulada/URR proxy (petroleo)", "gas_x_proxy": "idem gas",
        "coal_x_proxy": "idem carbon", "fosil_x_proxy": "idem agregado fosil (TWh)",
        "gigantes_n": "Numero de descubrimientos de campos gigantes (>=500 Mbep EUR) en el anio [Horn 2014; Cust-Mihalyi]",
        "gigantes_eur_mbep": "EUR descubierto en campos gigantes, Mbep", "gigantes_eur_petroleo_mbep": "idem solo campos de petroleo",
        "gigantes_eur_twh": "EUR gigantes en TWh", "gigantes_npv_pct_pib": "VPN del descubrimiento, % PIB (1960-2017)",
        "gigantes_eur_cum_mbep": "EUR acumulado de gigantes desde 1868, Mbep",
        "de_fosiles_t": "Extraccion domestica de combustibles fosiles, t [UN IRP GMFD 1970-2019]",
        "de_metales_t": "Extraccion domestica de minerales metalicos (mena bruta), t [UN IRP]",
        "de_no_metalicos_t": "Extraccion domestica de minerales no metalicos, t [UN IRP]",
        "de_biomasa_t": "Extraccion domestica de biomasa, t [UN IRP]",
        "nac_petroleo_share_post": "Evento de nacionalizacion/participacion en petroleo en el anio: participacion estatal posterior (tabla curada)",
        "nac_mineria_share_post": "idem mineria (metales)",
        "nac_evento_petroleo": "1 si hay evento de nacionalizacion petrolera en el anio",
        "nac_evento_mineria": "1 si hay evento en mineria/carbon en el anio",
        "estado_share_petroleo_eventos": "Participacion estatal en la produccion petrolera implicita por los eventos (ultimo evento; 0 = sin evento registrado, NO implica propiedad extranjera)",
        "renta_no_renovable_pct_pib": "Rentas petroleo+gas+carbon+minerales (% PIB)",
        "fosil_valor_urr_proxy_musd2019": "URR proxy fosil valorado a precios brutos de 2019 (M US$; 37,8/17,1/11,2 M$ por TWh de petroleo/gas/carbon)",
        "fosil_valor_cum_musd2019": "Produccion fosil acumulada desde 1900 valorada a precios de 2019 (M US$)",
        "fosil_x_valor": "Fraccion agotada ponderada por valor: candidato directo para x_i del recurso generico del modelo",
    }
    for k, (v, d) in WDI_VARS.items():
        desc.setdefault(v, f"{d} [WDI {k}]")
        desc[v.replace("_pct_pib", "_share")] = f"Participacion en rentas totales: {d}" if "renta_" in v else None
    desc = {k: v for k, v in desc.items() if v}
    wdesc = {
        "petroleo_precio_nominal_usd_bbl": "Precio del crudo US$/bbl corrientes (1861-1944 media EE.UU., 1945-1983 Arabian Light, 1984- Brent) [EI Statistical Review via OWID]",
        "deflactor_eeuu_2019": "Nivel de precios EE.UU. (2019=1): implicito en BP 2014 hasta 2013, IPC WDI despues",
        "petroleo_precio_real_usd2019_bbl": "Precio real del crudo, US$ de 2019/bbl",
        "gycpi_nominal": "Grilli-Yang commodity price index (1977-79=100), nominal", "gycpi_metales_nominal": "GYCPI metales",
        "muv_g5": "Indice de valor unitario de manufacturas (MUV-G5)", "gycpi_real": "GYCPI/MUV (terminos de intercambio materias primas/manufacturas)",
        "gycpi_metales_real": "GYCPI metales / MUV",
        "jacks_indice_vp1975": "Jacks (2019) indice real, ponderaciones valor de produccion 1975 (1900=100)",
        "jacks_indice_vp2019": "idem ponderaciones 2019", "jacks_indice_igual": "idem ponderacion igual (42 bienes)",
        "jacks_sub_cultivados": "Jacks subindice bienes cultivados", "jacks_sub_subsuelo": "Jacks subindice bienes del subsuelo (energia+metales+minerales)",
        "jacks_sub_subsuelo_sin_energia": "Jacks subindice subsuelo sin energia",
        "mundo_oil_prod_twh": "Produccion mundial de petroleo TWh (OWID)", "mundo_oil_cum_twh": "Acumulada mundial desde 1900",
        "mundo_gas_prod_twh": "Produccion mundial de gas TWh", "mundo_gas_cum_twh": "Acumulada mundial de gas",
        "mundo_coal_prod_twh": "Produccion mundial de carbon TWh", "mundo_coal_cum_twh": "Acumulada mundial de carbon",
        "mundo_gigantes_n": "Descubrimientos mundiales de campos gigantes (incl. conjuntos)",
        "mundo_gigantes_eur_mbep": "EUR mundial descubierto en gigantes, Mbep", "mundo_gigantes_eur_petroleo_mbep": "idem petroleo",
        "mundo_gigantes_eur_cum_mbep": "EUR acumulado mundial de gigantes desde 1868",
    }
    for c in W.columns:
        if c.startswith("jacks_") and c not in wdesc:
            wdesc[c] = f"Jacks (2019) precio real de {c[6:]} (1900=100)"
        if c.startswith("cmo_"):
            wdesc[c] = f"Banco Mundial Pink Sheet, precio real (US$ 2010) de {c[4:-9]}"
        if c.startswith("usgs_prod_mundial_"):
            wdesc[c] = f"USGS DS140 produccion mundial (mina) de {c[18:-2]}, t"
    urr = pd.read_csv(RAW / "urr_gea2012_bergsorensen.csv")
    urr = urr[urr.unit == "EJ"]
    const = {f"{r.commodity}__{r.series}_EJ": r.value for r in urr.itertuples()}
    d = {
        "descripcion": "Recursos naturales desagregados; ver docs/research/recursos.md",
        "conversiones": {"TWh_por_bep": BOE_TWH, "TWh_por_tep": TOE_TWH, "tep_por_t_hulla": HARD_COAL_TOE,
                         "tep_por_t_lignito": SOFT_COAL_TOE, "TWh_por_EJ": 277.78},
        "fuentes": {k: {"url": v[0], "descripcion": v[1]} for k, v in SOURCES.items()}
        | {"WDIData.csv (large/)": {"url": WDI_BULK[0], "descripcion": WDI_BULK[1]},
           "nacionalizaciones_curado.csv": {"url": "curado a mano", "descripcion":
                                            "Eventos de nacionalizacion petrolera/minera 1937-2012 compilados de Kobrin (1984), Guriev-Kolotilin-Sonin (2011), Mahdavi (2020), Yergin (1991); verificar contra los apendices originales"}},
        "recursos.csv": {c: {"descripcion": desc.get(c, ""), **_cov(p, c)} for c in p.columns
                         if c not in ("iso3", "year")},
        "recursos_world.csv": {c: {"descripcion": wdesc.get(c, ""), **_cov(W, c, key="__")} for c in W.columns
                               if c != "year"},
        "recursos_hubbert.csv": "Ajustes Hubbert por pais/recurso: linealizacion P/Q=r(1-Q/URR) (Q>=20% de Q2019) y r condicional al URR proxy",
        "constantes_mundiales_gea2012": const,
    }
    (OUT / "recursos_dict.json").write_text(json.dumps(d, ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
    build()
