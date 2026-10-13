"""Clima y ecologia: descarga fuentes crudas y construye paneles pais-anio y mundiales.

Uso:
    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from sources import clima; clima.build()"

Salidas
    data/processed/sources/clima.csv        (iso3, year, variables snake_case; 1950-2019)
    data/processed/sources/clima_world.csv  (year, series mundiales; 1850-2025)
    data/processed/sources/clima_dict.json  (diccionario de variables + estimaciones auxiliares)

Todas las fuentes se descargan de raw.githubusercontent.com (unico host accesible): repositorios
oficiales (OWID, datasets/, open-numbers = WDI/Gapminder) o espejos de replicacion (BHM 2015,
FAOSTAT, GFN, EM-DAT). Ver docs/research/clima.md para detalles y advertencias.
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "clima"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"
GH = "https://raw.githubusercontent.com/"
Y0, Y1 = 1950, 2019

# --------------------------------------------------------------------------------------
# Fuentes crudas (archivo local -> URL)
# --------------------------------------------------------------------------------------
_OECD = GH + "hdesaioecd/oecd-sof-2022-public/HEAD/data/2020%20sfr%20model%20data/nd-gain/"
_BASITH = (GH + "Basith5075/Data_Science_Submission/HEAD/Certification/Practice/python-practice/"
           "Project_Info_Visualization/")
SOURCES = {
    # Global Carbon Project + PRIMAP + Jones et al. (2023) via Our World in Data
    "owid-co2-data.csv": GH + "owid/co2-data/master/owid-co2-data.csv",
    "owid-energy-data.csv": GH + "owid/energy-data/master/owid-energy-data.csv",
    # Burke, Hsiang & Miguel (2015) replication dataset (UDel pop-weighted T & P, 1960-2010)
    "bhm_GrowthClimateDataset.dta": GH + ("UW-MSDS-DATA-598-Reproducibility-WI20/"
                                          "battini-garizio-lee-zlavog-replication-project/HEAD/"
                                          "data/input/GrowthClimateDataset.dta"),
    # FAOSTAT "Temperature change on land" (GISTEMP-based, anomaly vs 1951-1980), 1961-2021
    "fao_temperature_change.csv": _BASITH + "Temperature_Change_Data.csv",
    "fao_country_codes.csv": _BASITH + "FAOSTAT_data_11-24-2020.csv",
    # OWID / Copernicus ERA5 decadal mean surface temperature (area-weighted), 1940s-2020s
    "era5_decadal_temperature.csv": GH + ("GunHoJang/ETO5513Assessment2/HEAD/Data/"
                                          "decadal-average-surface-temperature.csv"),
    # Global mean surface temperature (GISTEMP, HadCRUT5) and CO2 concentration
    "global_temp_annual.csv": GH + "datasets/global-temp/HEAD/data/annual.csv",
    "co2_mlo_annmean.csv": GH + "datasets/co2-ppm/HEAD/data/co2-annmean-mlo.csv",
    "co2_longterm_owid.csv": GH + ("willwin4sure/CSC_630_Data_Visualization/HEAD/climate/data/"
                                   "co2-concentration-long-term.csv"),
    # Global Footprint Network, National Footprint Accounts 2018 edition (1961-2014)
    "nfa_2018.csv": GH + "jpmckeown/eo_data/HEAD/data/NFA_2018.csv",
    # Living Planet Index (WWF/ZSL 2022 via OWID)
    "lpi_owid.csv": GH + "kflisikowsky/sad/HEAD/data/viz/global-living-planet-index.csv",
    # ND-GAIN climate vulnerability / readiness (1995-2019)
    "ndgain_gain.csv": _OECD + "gain/gain.csv",
    "ndgain_vulnerability.csv": _OECD + "vulnerability/vulnerability.csv",
    "ndgain_readiness.csv": _OECD + "readiness/readiness.csv",
    "ndgain_food.csv": _OECD + "vulnerability/food.csv",
    "ndgain_water.csv": _OECD + "vulnerability/water.csv",
}
# EM-DAT (CRED/UCLouvain) public extract, natural disasters 1900-2025. Licence forbids
# redistribution -> stored under large/ (git-ignored).
EMDAT = ("large/emdat_natural_1900_2025.csv",
         GH + "bobromero/climate_change_analyzer/HEAD/data/NaturalDisastersEmDat1900-2025.csv")

# World Development Indicators via open-numbers DDF (World Bank, CC-BY 4.0)
WDI_URL = (GH + "open-numbers/ddf--open_numbers--world_development_indicators/master/datapoints/"
           "ddf--datapoints--{}--by--geo--time.csv")
WDI = {
    "ag_lnd_frst_zs": ("forest_pct", "Superficie forestal (% del territorio)", "%"),
    "ag_lnd_frst_k2": ("forest_km2", "Superficie forestal", "km2"),
    "ag_lnd_agri_zs": ("agri_land_pct", "Tierra agricola (% del territorio)", "%"),
    "ag_lnd_arbl_zs": ("arable_land_pct", "Tierra arable (% del territorio)", "%"),
    "ag_yld_crel_kg": ("cereal_yield", "Rendimiento de cereales", "kg/ha"),
    "ag_con_fert_zs": ("fertilizer_kg_ha", "Consumo de fertilizantes", "kg/ha tierra arable"),
    "ag_lnd_prcp_mm": ("pre_clim_wdi", "Precipitacion media de largo plazo (climatologia, constante)",
                       "mm/anio"),
    "er_h2o_fwst_zs": ("water_stress_pct", "Estres hidrico: extraccion/recursos disponibles (ODS 6.4.2)",
                       "%"),
    "er_h2o_fwtl_zs": ("fw_withdrawal_pct", "Extraccion de agua dulce (% recursos internos)", "%"),
    "en_pop_el5m_zs": ("pop_below5m_pct", "Poblacion en tierras < 5 m s.n.m.", "%"),
    "en_clc_mdat_zs": ("drought_flood_pop_pct",
                       "Poblacion afectada por sequias/inundaciones/temperaturas extremas (media 1990-2009)",
                       "% poblacion"),
    "ny_gdp_frst_rt_zs": ("forest_rents_pct", "Rentas forestales (% PIB)", "% PIB"),
    "ny_adj_dfor_gn_zs": ("depl_forest_pct_gni", "Agotamiento neto de bosques (% INB)", "% INB"),
    "ny_adj_dres_gn_zs": ("depl_natres_pct_gni", "Agotamiento de recursos naturales (% INB)", "% INB"),
    "ny_adj_dngy_gn_zs": ("depl_energy_pct_gni", "Agotamiento energetico (% INB)", "% INB"),
    "ny_adj_dmin_gn_zs": ("depl_mineral_pct_gni", "Agotamiento mineral (% INB)", "% INB"),
    "ny_adj_dco2_gn_zs": ("dmg_co2_pct_gni", "Dano por emisiones de CO2 (% INB, ahorro ajustado)", "% INB"),
    "ny_adj_dpem_gn_zs": ("dmg_pm_pct_gni", "Dano por material particulado (% INB)", "% INB"),
    "er_lnd_ptld_zs": ("protected_land_pct", "Areas terrestres protegidas", "%"),
    "en_atm_pm25_mc_m3": ("pm25", "Exposicion media a PM2.5", "ug/m3"),
    "er_fsh_capt_mt": ("fish_capture_t", "Captura pesquera", "t"),
    "en_mam_thrd_no": ("threatened_mammals", "Especies de mamiferos amenazadas", "n"),
}
# Gapminder fasttrack via open-numbers (material footprint = UNEP IRP; SDI = Hickel 2020)
GAP_URL = (GH + "open-numbers/ddf--gapminder--fasttrack/master/countries_etc_datapoints/"
           "ddf--datapoints--{}--by--country--time.csv")
GAP = {
    "matfootp": ("material_footprint_t", "Huella material (extraccion de materias primas atribuida al "
                 "consumo final; UNEP-IRP)", "t"),
    "matfootp_cap": ("material_footprint_pc", "Huella material per capita", "t/persona"),
    "sdi": ("sdi", "Sustainable Development Index (Hickel 2020)", "indice 0-100"),
}
# Global Footprint Network, NFA 2025 edition (1961-2024), per-capita consumption EF and
# biocapacity, one file per country in eldho-se/overshoot-app
GFN_URL = GH + "eldho-se/overshoot-app/HEAD/backend/data/countries/Countries/{}.csv"
GFN_FILES = """Peru Cuba Mali Chad Iraq Fiji Oman Togo Kenya Japan Egypt World China Samoa Chile Gabon
Congo Haiti Sudan Italy Yemen Niger Ghana Benin Spain Tonga India Qatar Malta Nepal Greece Poland Israel
Gambia Serbia Brazil Sweden Panama Guinea Kuwait Mexico Rwanda Angola Uganda Norway Bhutan Cyprus Latvia
Canada Zambia France Malawi Guyana Jordan Nigeria Bolivia Algeria Lesotho Burundi Namibia Ecuador Bahamas
Finland Armenia Morocco Somalia Denmark Romania Germany Georgia Croatia Hungary Albania Lebanon Bahrain
Ukraine Estonia Belgium Tunisia Eritrea Myanmar Senegal Grenada Liberia Ireland Austria Belarus Viet_Nam
Paraguay Colombia Djibouti Cameroon Slovenia Slovakia Pakistan Bulgaria Dominica Suriname Mongolia
Honduras Barbados Malaysia Eswatini Ethiopia Cambodia Thailand Botswana Zimbabwe Portugal Indonesia
Mauritius Sri_Lanka Nicaragua Argentina Lithuania Australia Guatemala Singapore Bangladesh Tajikistan
Mozambique Mauritania Cabo_Verde Azerbaijan Madagascar Kazakhstan Uzbekistan Kyrgyzstan Luxembourg
Montenegro Costa_Rica South_Sudan Switzerland Timor-Leste El_Salvador Philippines Netherlands New_Zealand
Saint_Lucia Afghanistan Burkina_Faso Saudi_Arabia Turkmenistan South_Africa Tu╠êrkiye Sierra_Leone
Guinea-Bissau United_Kingdom Czech_Republic Sudan_(former) Solomon_Islands French_Polynesia
Papua_New_Guinea Equatorial_Guinea Dominican_Republic Russian_Federation State_of_Palestine
Korea,_Republic_of Co╠éte_d_Ivoire Antigua_and_Barbuda Trinidad_and_Tobago United_Arab_Emirates
Sao_Tome_and_Principe Bosnia_and_Herzegovina United_States_of_America Central_African_Republic
Iran,_Islamic_Republic_of Republic_of_North_Macedonia Saint_Vincent_and_Grenadines
Tanzania,_United_Republic_of Congo,_Democratic_Republic_of Lao_People_s_Democratic_Republic
Venezuela,_Bolivarian_Republic_of Korea,_Democratic_People_s_Republic_of""".split()

OWID_CO2_VARS = {
    "co2": ("co2_mt", "Emisiones territoriales de CO2 fosil + cemento", "MtCO2"),
    "co2_per_capita": ("co2_pc", "CO2 territorial per capita", "tCO2/persona"),
    "co2_per_gdp": ("co2_per_gdp", "Intensidad de CO2 del PIB", "kgCO2/$ PPA 2011"),
    "co2_including_luc": ("co2_incl_luc_mt", "CO2 incluyendo cambio de uso del suelo", "MtCO2"),
    "land_use_change_co2": ("co2_luc_mt", "CO2 por cambio de uso del suelo (deforestacion)", "MtCO2"),
    "coal_co2": ("co2_coal_mt", "CO2 del carbon", "MtCO2"),
    "oil_co2": ("co2_oil_mt", "CO2 del petroleo", "MtCO2"),
    "gas_co2": ("co2_gas_mt", "CO2 del gas", "MtCO2"),
    "cement_co2": ("co2_cement_mt", "CO2 del cemento", "MtCO2"),
    "flaring_co2": ("co2_flaring_mt", "CO2 por quema en antorcha", "MtCO2"),
    "consumption_co2": ("co2_cons_mt", "CO2 basado en consumo (1990-)", "MtCO2"),
    "consumption_co2_per_capita": ("co2_cons_pc", "CO2 de consumo per capita", "tCO2/persona"),
    "trade_co2": ("co2_trade_mt", "CO2 neto importado incorporado en comercio (consumo - territorial)",
                  "MtCO2"),
    "trade_co2_share": ("co2_trade_share", "CO2 neto importado en % de las territoriales", "%"),
    "cumulative_co2": ("co2_cum_mt", "CO2 fosil acumulado desde 1750", "MtCO2"),
    "cumulative_co2_including_luc": ("co2_cum_incl_luc_mt", "CO2 acumulado incl. uso del suelo (1850-)",
                                     "MtCO2"),
    "cumulative_luc_co2": ("co2_cum_luc_mt", "CO2 acumulado por uso del suelo", "MtCO2"),
    "share_global_co2": ("co2_share_world", "Participacion en CO2 mundial", "%"),
    "share_global_cumulative_co2": ("co2_cum_share_world", "Participacion en CO2 acumulado mundial", "%"),
    "methane": ("ch4_mtco2e", "Emisiones de metano", "MtCO2e"),
    "nitrous_oxide": ("n2o_mtco2e", "Emisiones de oxido nitroso", "MtCO2e"),
    "total_ghg": ("ghg_mtco2e", "GEI totales incl. uso del suelo", "MtCO2e"),
    "total_ghg_excluding_lucf": ("ghg_excl_lucf_mtco2e", "GEI totales excl. uso del suelo", "MtCO2e"),
    "ghg_per_capita": ("ghg_pc", "GEI per capita", "tCO2e/persona"),
    "temperature_change_from_ghg": ("dT_from_ghg", "Calentamiento global atribuible a GEI del pais "
                                    "(Jones et al. 2023)", "degC"),
    "temperature_change_from_co2": ("dT_from_co2", "Calentamiento global atribuible al CO2 del pais",
                                    "degC"),
    "share_of_temperature_change_from_ghg": ("dT_share_world", "Participacion en el calentamiento global",
                                             "%"),
    "primary_energy_consumption": ("primary_energy_twh", "Consumo de energia primaria", "TWh"),
}
OWID_ENERGY_VARS = {
    "coal_production": ("coal_prod_twh", "Produccion de carbon", "TWh"),
    "oil_production": ("oil_prod_twh", "Produccion de petroleo", "TWh"),
    "gas_production": ("gas_prod_twh", "Produccion de gas", "TWh"),
    "fossil_fuel_consumption": ("fossil_cons_twh", "Consumo de combustibles fosiles", "TWh"),
    "fossil_share_energy": ("fossil_share_energy", "Participacion fosil en energia primaria", "%"),
}
CLIMATE_SUBGROUPS = {"Hydrological", "Meteorological", "Climatological"}


# --------------------------------------------------------------------------------------
# Descarga
# --------------------------------------------------------------------------------------
def _get(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    parts = urllib.parse.urlsplit(url)
    url = urllib.parse.urlunsplit(parts._replace(path=urllib.parse.quote(parts.path, safe="/%(),'-_.~")))
    tmp = dst.with_suffix(dst.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    head = tmp.read_bytes()[:64]
    if head.startswith(b"version https://git-lfs"):
        tmp.unlink()
        raise RuntimeError(f"git-lfs pointer: {url}")
    tmp.replace(dst)


def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    gi = LARGE / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    for name, url in SOURCES.items():
        if force or not (RAW / name).exists():
            print("descargando", name)
            _get(url, RAW / name)
    if force or not (RAW / EMDAT[0]).exists():
        print("descargando EM-DAT")
        _get(EMDAT[1], RAW / EMDAT[0])
    for code in WDI:
        f = RAW / "wdi" / f"{code}.csv"
        if force or not f.exists():
            _get(WDI_URL.format(code), f)
    for code in GAP:
        f = RAW / "gapminder" / f"{code}.csv"
        if force or not f.exists():
            _get(GAP_URL.format(code), f)
    gfn = RAW / "gfn_nfa2025_percap.csv"
    if force or not gfn.exists():
        print("descargando GFN NFA-2025 (", len(GFN_FILES), "archivos )")
        frames = []
        tmp = RAW / "_gfn_tmp.csv"
        for n in GFN_FILES:
            try:
                _get(GFN_URL.format(n), tmp)
                frames.append(pd.read_csv(tmp))
            except Exception as e:  # noqa: BLE001
                print("  GFN fallo", n, e)
        tmp.unlink(missing_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(gfn, index=False)


# --------------------------------------------------------------------------------------
# Lectores
# --------------------------------------------------------------------------------------
def _valid_iso(s: pd.Series) -> pd.Series:
    return s.astype(str).str.fullmatch(r"[A-Z]{3}")


def _owid_co2() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RAW / "owid-co2-data.csv")
    ren = {k: v[0] for k, v in OWID_CO2_VARS.items()}
    world = d[d.country == "World"][["year", *ren]].rename(columns=ren)
    c = d[_valid_iso(d.iso_code.fillna(""))][["iso_code", "year", *ren]]
    c = c.rename(columns={"iso_code": "iso3", **ren})
    return c, world


def _owid_energy() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RAW / "owid-energy-data.csv")
    ren = {k: v[0] for k, v in OWID_ENERGY_VARS.items()}
    world = d[d.country == "World"][["year", *ren]].rename(columns=ren)
    c = d[_valid_iso(d.iso_code.fillna(""))][["iso_code", "year", *ren]]
    c = c.rename(columns={"iso_code": "iso3", **ren})
    c["fossil_prod_twh"] = c[["coal_prod_twh", "oil_prod_twh", "gas_prod_twh"]].sum(axis=1, min_count=1)
    world["fossil_prod_twh"] = world[["coal_prod_twh", "oil_prod_twh", "gas_prod_twh"]].sum(
        axis=1, min_count=1)
    return c, world


def _bhm() -> pd.DataFrame:
    d = pd.read_stata(RAW / "bhm_GrowthClimateDataset.dta",
                      columns=["iso", "year", "UDel_temp_popweight", "UDel_precip_popweight"])
    d = d.rename(columns={"iso": "iso3", "UDel_temp_popweight": "tmp_pw_udel",
                          "UDel_precip_popweight": "pre_pw_udel"})
    d["iso3"] = d.iso3.replace({"ROM": "ROU", "ZAR": "COD", "TMP": "TLS"})
    d["year"] = d.year.astype(int)
    return d.dropna(subset=["tmp_pw_udel"])


def _fao_codes() -> pd.DataFrame:
    m = pd.read_csv(RAW / "fao_country_codes.csv", encoding="utf-8-sig", dtype=str)
    m = m.rename(columns={"Country Code": "fao", "ISO3 Code": "iso3"})[["fao", "iso3"]]
    m = m[_valid_iso(m.iso3.fillna(""))]
    m["fao"] = m.fao.astype(int)
    return m


def _fao_temp() -> pd.DataFrame:
    d = pd.read_csv(RAW / "fao_temperature_change.csv", encoding="latin1")
    d = d[d.Months == "Meteorological year"]
    yc = [c for c in d.columns if c.startswith("Y") and c[1:].isdigit()]
    long = d.melt(id_vars=["Area Code", "Element"], value_vars=yc, var_name="year")
    long["year"] = long.year.str[1:].astype(int)
    long = long.pivot_table(index=["Area Code", "year"], columns="Element", values="value").reset_index()
    long = long.rename(columns={"Area Code": "fao", "Temperature change": "tmp_anom_fao",
                                "Standard Deviation": "tmp_anom_fao_sd"})
    long = long.merge(_fao_codes(), on="fao", how="inner").drop(columns="fao")
    long["iso3"] = long.iso3.replace({"SUD": "SDN"})
    return long.groupby(["iso3", "year"], as_index=False).mean(numeric_only=True)


def _era5_decadal() -> pd.DataFrame:
    d = pd.read_csv(RAW / "era5_decadal_temperature.csv")
    d = d[_valid_iso(d.Code.fillna(""))].rename(
        columns={"Code": "iso3", "Year": "decade", d.columns[-1]: "tmp_era5_decadal"})
    return d[["iso3", "decade", "tmp_era5_decadal"]]


def _emdat() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RAW / EMDAT[0], low_memory=False, encoding="latin1")
    d = d[d["Disaster Group"] == "Natural"].rename(columns={"ISO": "iso3", "Start Year": "year"})
    d["iso3"] = d.iso3.replace({"SCG": "SRB", "YUG": "SRB", "SUN": "RUS", "CSK": "CZE",
                                "DDR": "DEU", "DFR": "DEU", "YMN": "YEM", "YMD": "YEM",
                                "AZO": "PRT", "SPI": "ESP"})
    dt = d["Disaster Type"]
    d["clim"] = d["Disaster Subgroup"].isin(CLIMATE_SUBGROUPS)
    flags = {
        "dis_n_total": pd.Series(True, index=d.index),
        "dis_n_climate": d.clim,
        "dis_n_flood": dt.eq("Flood"),
        "dis_n_storm": dt.eq("Storm"),
        "dis_n_drought": dt.eq("Drought"),
        "dis_n_extreme_temp": dt.eq("Extreme temperature"),
        "dis_n_wildfire": dt.eq("Wildfire"),
        "dis_n_geo": d["Disaster Subgroup"].eq("Geophysical"),
    }
    for k, v in flags.items():
        d[k] = v.astype(int)
    num = lambda c: pd.to_numeric(d[c], errors="coerce").fillna(0)  # noqa: E731
    d["dis_deaths"] = num("Total Deaths")
    d["dis_deaths_climate"] = d.dis_deaths * d.clim
    d["dis_affected"] = num("Total Affected")
    d["dis_damage_kusd"] = num("Total Damage, Adjusted ('000 US$)")
    d["dis_damage_climate_kusd"] = d.dis_damage_kusd * d.clim
    cols = [*flags, "dis_deaths", "dis_deaths_climate", "dis_affected", "dis_damage_kusd",
            "dis_damage_climate_kusd"]
    c = d.groupby(["iso3", "year"], as_index=False)[cols].sum()
    w = d.groupby("year", as_index=False)[cols].sum()
    return c, w


def _nfa2018() -> pd.DataFrame:
    d = pd.read_csv(RAW / "nfa_2018.csv").rename(columns={"ISO alpha-3 code": "iso3"})
    d = d[_valid_iso(d.iso3.fillna(""))]
    rec = {"EFConsPerCap": "ef_cons_pc_2018ed", "EFProdPerCap": "ef_prod_pc",
           "BiocapPerCap": "biocap_pc_2018ed", "EFImportsPerCap": "ef_imports_pc",
           "EFExportsPerCap": "ef_exports_pc", "EFConsTotGHA": "ef_cons_gha",
           "BiocapTotGHA": "biocap_gha", "EFImportsTotGHA": "ef_imports_gha",
           "EFExportsTotGHA": "ef_exports_gha"}
    d = d[d.record.isin(rec)]
    w = d.pivot_table(index=["iso3", "year"], columns="record", values="total").rename(columns=rec)
    carb = d[d.record == "EFConsPerCap"].set_index(["iso3", "year"])["carbon"].rename("ef_carbon_pc")
    w = w.join(carb).reset_index()
    w["ef_net_imports_gha"] = w.ef_imports_gha - w.ef_exports_gha
    return w


def _gfn2025() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = pd.read_csv(RAW / "gfn_nfa2025_percap.csv")
    p = d.pivot_table(index=["countryCode", "countryName", "year"], columns="record",
                      values="total").reset_index()
    p = p.rename(columns={"EFConsPerCap": "ef_cons_pc_2025ed", "BiocapPerCap": "biocap_pc_2025ed"})
    world = p[p.countryName == "World"][["year", "ef_cons_pc_2025ed", "biocap_pc_2025ed"]]
    p = p.merge(_fao_codes(), left_on="countryCode", right_on="fao", how="inner")
    p = p[["iso3", "year", "ef_cons_pc_2025ed", "biocap_pc_2025ed"]]
    return p.groupby(["iso3", "year"], as_index=False).mean(), world


def _ddf(folder: str, spec: dict, key: str) -> pd.DataFrame:
    out = None
    for code, (name, *_rest) in spec.items():
        d = pd.read_csv(RAW / folder / f"{code}.csv")
        d = d.rename(columns={key: "iso3", "time": "year", code: name})
        d["iso3"] = d.iso3.str.upper()
        out = d if out is None else out.merge(d, on=["iso3", "year"], how="outer")
    return out


def _ndgain() -> pd.DataFrame:
    out = None
    for f, name in [("gain", "ndgain_index"), ("vulnerability", "ndgain_vulnerability"),
                    ("readiness", "ndgain_readiness"), ("food", "ndgain_food"),
                    ("water", "ndgain_water")]:
        d = pd.read_csv(RAW / f"ndgain_{f}.csv").drop(columns="Name")
        d = d.melt(id_vars="ISO3", var_name="year", value_name=name).rename(columns={"ISO3": "iso3"})
        d["year"] = d.year.astype(int)
        out = d if out is None else out.merge(d, on=["iso3", "year"], how="outer")
    return out


def _world_climate() -> pd.DataFrame:
    g = pd.read_csv(RAW / "global_temp_annual.csv")
    g = g.pivot_table(index="Year", columns="Source", values="Mean").rename(
        columns={"GISTEMP": "gmst_gistemp", "GCAG": "gmst_hadcrut5"})
    g.index.name = "year"
    g["gmst_preind"] = g.gmst_hadcrut5 - g.loc[1850:1900, "gmst_hadcrut5"].mean()
    mlo = pd.read_csv(RAW / "co2_mlo_annmean.csv").set_index("Year")["Mean"].rename("co2_ppm_mlo")
    lt = pd.read_csv(RAW / "co2_longterm_owid.csv")
    lt = lt[lt.Entity == "World"].set_index("Year").iloc[:, -1].rename("co2_ppm_icecore")
    years = pd.Index(range(1850, 2026), name="year")
    w = pd.DataFrame(index=years).join(g).join(mlo).join(lt)
    ice = lt[lt.index >= 1700].reindex(range(1700, 2026)).interpolate(limit_area="inside")
    w["co2_ppm"] = w.co2_ppm_mlo.fillna(ice.reindex(years))
    return w.reset_index()


# --------------------------------------------------------------------------------------
# Construccion
# --------------------------------------------------------------------------------------
def _pattern_scaling(fao: pd.DataFrame, world: pd.DataFrame) -> pd.DataFrame:
    """OLS por pais: anomalia local (FAO) = a + beta * anomalia global (GISTEMP), 1961-2019."""
    g = world.set_index("year")["gmst_gistemp"]
    rows = []
    for iso, d in fao[(fao.year <= Y1)].groupby("iso3"):
        x = g.reindex(d.year).values
        y = d.tmp_anom_fao.values
        ok = ~(np.isnan(x) | np.isnan(y))
        if ok.sum() < 30:
            continue
        b, a = np.polyfit(x[ok], y[ok], 1)
        rows.append({"iso3": iso, "tmp_pattern_beta": b})
    return pd.DataFrame(rows)


def _temperature_level(bhm, fao, era) -> pd.DataFrame:
    """Temperatura media anual (nivel, degC), ponderada por poblacion cuando es posible.

    src 1: UDel pop-weighted (BHM 2015), 1960(1)-2010
    src 2: climatologia UDel 1981-2010 + (anomalia FAO_t - anomalia FAO media 1981-2010), 2011-2019
           y anios 1961-2010 faltantes
    src 3: 1950-1960: climatologia UDel 1961-1970 + (ERA5 decada 1950s - ERA5 decada 1960s)
    src 4: sin UDel: nivel ERA5 (area) 1951-1980 + anomalia FAO (area); 1950-60 = ERA5 decadal
    """
    grid = pd.MultiIndex.from_product([sorted(set(bhm.iso3) | set(fao.iso3) | set(era.iso3)),
                                       range(Y0, Y1 + 1)], names=["iso3", "year"]).to_frame(index=False)
    grid["decade"] = grid.year // 10 * 10
    t = (grid.merge(bhm[["iso3", "year", "tmp_pw_udel"]], how="left")
         .merge(fao[["iso3", "year", "tmp_anom_fao"]], how="left")
         .merge(era, on=["iso3", "decade"], how="left"))

    def per_country(d):
        d = d.sort_values("year").copy()
        lvl = d.tmp_pw_udel.copy()
        src = pd.Series(np.where(lvl.notna(), 1, np.nan), index=d.index)
        yrs = d.year
        if d.tmp_pw_udel.notna().sum() >= 20:
            clim = d.loc[yrs.between(1981, 2010), "tmp_pw_udel"].mean()
            fclim = d.loc[yrs.between(1981, 2010), "tmp_anom_fao"].mean()
            fill = clim + d.tmp_anom_fao - fclim
            m = lvl.isna() & fill.notna()
            lvl[m], src[m] = fill[m], 2
            e50 = d.loc[yrs.between(1950, 1959), "tmp_era5_decadal"].mean()
            e60 = d.loc[yrs.between(1960, 1969), "tmp_era5_decadal"].mean()
            c60 = d.loc[yrs.between(1961, 1970), "tmp_pw_udel"].mean()
            m = lvl.isna() & (yrs < 1961)
            lvl[m], src[m] = c60 + (e50 - e60), 3
        else:
            base = d.loc[yrs.between(1950, 1979), "tmp_era5_decadal"].mean()
            fill = base + d.tmp_anom_fao
            m = lvl.isna() & fill.notna()
            lvl[m], src[m] = fill[m], 4
            m = lvl.isna() & (yrs < 1961)
            lvl[m], src[m] = d.loc[m, "tmp_era5_decadal"], 4
        d["tmp_level"], d["tmp_level_src"] = lvl, src.where(lvl.notna())
        return d

    t = pd.concat([per_country(d) for _, d in t.groupby("iso3")], ignore_index=True)
    return t[["iso3", "year", "tmp_level", "tmp_level_src", "tmp_era5_decadal"]]


def _bhm_replication(bhm_path: Path, panel_df: pd.DataFrame | None) -> dict:
    """Replica BHM (2015): g_it = b1 T + b2 T^2 + c1 P + c2 P^2 + FE pais + FE anio + tendencias
    pais (lineal+cuadratica); errores estandar agrupados por pais."""
    import statsmodels.formula.api as smf

    res = {}
    d = pd.read_stata(bhm_path, columns=["iso", "year", "growthWDI", "UDel_temp_popweight",
                                         "UDel_precip_popweight"]).dropna()
    d = d.rename(columns={"growthWDI": "g", "UDel_temp_popweight": "T", "UDel_precip_popweight": "P"})

    def fit(df, label):
        df = df.copy()
        df["T2"], df["P"], df["P2"] = df["T"] ** 2, df["P"] / 1000, (df["P"] / 1000) ** 2
        df["t"] = df.year - df.year.min()
        df["t2"] = df.t ** 2
        f = "g ~ T + T2 + P + P2 + C(iso) + C(year) + C(iso):t + C(iso):t2"
        m = smf.ols(f, data=df).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(df.iso)[0]})
        b1, b2 = m.params["T"], m.params["T2"]
        res[label] = {"b1_T": float(b1), "b2_T2": float(b2), "se_b1": float(m.bse["T"]),
                      "se_b2": float(m.bse["T2"]), "T_opt": float(-b1 / (2 * b2)),
                      "n": int(m.nobs), "years": [int(df.year.min()), int(df.year.max())]}

    fit(d, "bhm_original_udel_1961_2010")
    if panel_df is not None:
        fit(panel_df, "pwt_panel_tmp_level_1951_2019")
    return res


def build(force_download: bool = False) -> pd.DataFrame:
    download(force_download)
    OUT.mkdir(parents=True, exist_ok=True)

    co2, w_co2 = _owid_co2()
    en, w_en = _owid_energy()
    bhm = _bhm()
    fao = _fao_temp()
    era = _era5_decadal()
    dis, w_dis = _emdat()
    nfa = _nfa2018()
    gfn, w_gfn = _gfn2025()
    wdi = _ddf("wdi", WDI, "geo")
    gap = _ddf("gapminder", GAP, "country")
    ndg = _ndgain()
    world = _world_climate()

    tlev = _temperature_level(bhm, fao, era)
    ps = _pattern_scaling(fao, world)

    frames = [co2, en, bhm, fao, tlev, dis, nfa, gfn, wdi, gap, ndg]
    df = None
    for f in frames:
        f = f[(f.year >= Y0) & (f.year <= Y1)]
        df = f if df is None else df.merge(f, on=["iso3", "year"], how="outer")
    df = df.merge(ps, on="iso3", how="left")
    df = df[_valid_iso(df.iso3)]

    # EM-DAT: ausencia de registro = 0 eventos (para paises en el panel de clima)
    dcols = [c for c in dis.columns if c.startswith("dis_")]
    df[dcols] = df[dcols].fillna(0)
    # Huella ecologica combinada: edicion 2025 si existe, si no 2018
    df["ef_cons_pc"] = df.ef_cons_pc_2025ed.fillna(df.ef_cons_pc_2018ed)
    df["biocap_pc"] = df.biocap_pc_2025ed.fillna(df.biocap_pc_2018ed)
    df["eco_deficit_pc"] = df.ef_cons_pc - df.biocap_pc
    df["ef_net_imports_pc"] = df.ef_imports_pc - df.ef_exports_pc
    # Deforestacion anual (% de la superficie forestal del anio previo), desde 1990
    df = df.sort_values(["iso3", "year"])
    lag = df.groupby("iso3").forest_km2.shift(1)
    df["forest_loss_pct"] = -(df.forest_km2 - lag) / lag * 100
    # Anomalia de temperatura respecto a la media propia 1951-1980
    base = df[df.year.between(1951, 1980)].groupby("iso3").tmp_level.mean().rename("_b")
    df = df.merge(base, on="iso3", how="left")
    df["tmp_anom_level"] = df.tmp_level - df._b
    df = df.drop(columns="_b")
    df["tmp_level_sq"] = df.tmp_level ** 2
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    df.to_csv(OUT / "clima.csv", index=False)

    # ------------------------------- mundo ---------------------------------------------
    w = world.merge(w_co2, on="year", how="left").merge(w_en, on="year", how="left")
    w = w.merge(w_dis, on="year", how="left").merge(w_gfn, on="year", how="left")
    w["overshoot_ratio"] = w.ef_cons_pc_2025ed / w.biocap_pc_2025ed
    lpi = pd.read_csv(RAW / "lpi_owid.csv")
    lpi = lpi[lpi.Entity == "World"].iloc[:, 2:]
    lpi.columns = ["year", "lpi", "lpi_hi", "lpi_lo"]
    w = w.merge(lpi, on="year", how="left")
    mf = pd.read_csv(RAW / "gapminder" / "matfootp.csv")
    mf = mf.groupby("time", as_index=False).matfootp.sum(min_count=1).rename(
        columns={"time": "year", "matfootp": "material_footprint_t_sum"})
    w = w.merge(mf, on="year", how="left")
    w["co2_gtc"] = w.co2_mt / 3664.0
    w["co2_cum_incl_luc_gtco2"] = w.co2_cum_incl_luc_mt / 1000
    w = w.rename(columns={c: "world_" + c for c in w.columns if c.startswith("dis_")})
    w.to_csv(OUT / "clima_world.csv", index=False)

    # ------------------------------- regresion BHM ---------------------------------------
    est = {}
    try:
        pnl = pd.read_csv(ROOT / "data" / "processed" / "panel.csv", usecols=["iso3", "year", "gdppc"])
        pnl = pnl.sort_values(["iso3", "year"])
        pnl["g"] = np.log(pnl.gdppc).groupby(pnl.iso3).diff()
        pr = df[["iso3", "year", "tmp_level", "pre_pw_udel"]].copy()
        # precipitacion: UDel hasta 2010 (se usa climatologia del pais despues)
        pr["P"] = pr.pre_pw_udel.fillna(pr.groupby("iso3").pre_pw_udel.transform("mean"))
        pd_ = pnl.merge(pr, on=["iso3", "year"]).rename(columns={"iso3": "iso", "tmp_level": "T"})
        pd_ = pd_.dropna(subset=["g", "T", "P"])
        pd_ = pd_[pd_.g.abs() < 0.5]
        est = _bhm_replication(RAW / "bhm_GrowthClimateDataset.dta", pd_[["iso", "year", "g", "T", "P"]])
    except Exception as e:  # noqa: BLE001
        est = {"error": repr(e)}
    ps_d = ps.tmp_pattern_beta.describe()
    est["pattern_scaling_beta_summary"] = {k: float(v) for k, v in ps_d.items()}

    _write_dict(df, w, est)
    print("clima.csv", df.shape, "| clima_world.csv", w.shape)
    return df


def _cov(s: pd.Series, years: pd.Series) -> str:
    y = years[s.notna()]
    return f"{int(y.min())}-{int(y.max())}" if len(y) else "n/a"


def _write_dict(df: pd.DataFrame, w: pd.DataFrame, est: dict) -> None:
    meta = {}
    for spec, src in [(OWID_CO2_VARS, "OWID co2-data (Global Carbon Project 2024, Jones et al. 2023)"),
                      (OWID_ENERGY_VARS, "OWID energy-data (Energy Institute Statistical Review; EIA)")]:
        for _, (n, desc, unit) in spec.items():
            meta[n] = (desc, unit, src)
    for _, (n, desc, unit) in WDI.items():
        meta[n] = (desc, unit, "World Bank WDI via open-numbers DDF")
    for _, (n, desc, unit) in GAP.items():
        meta[n] = (desc, unit, "Gapminder fasttrack via open-numbers (UNEP-IRP; Hickel 2020)")
    bhm_src = "Burke, Hsiang & Miguel (2015) replication data; U. Delaware v3.01, ponderado por poblacion"
    fao_src = "FAOSTAT Temperature change on land (NASA GISTEMP), base 1951-1980, anio meteorologico"
    emdat = "EM-DAT (CRED/UCLouvain) extracto publico, desastres naturales por anio de inicio"
    nfa = "Global Footprint Network NFA 2018 (1961-2014)"
    extra = {
        "tmp_pw_udel": ("Temperatura media anual ponderada por poblacion", "degC", bhm_src),
        "pre_pw_udel": ("Precipitacion anual ponderada por poblacion", "mm/anio", bhm_src),
        "tmp_anom_fao": ("Anomalia de temperatura terrestre vs 1951-1980 (area)", "degC", fao_src),
        "tmp_anom_fao_sd": ("Desv. estandar climatologica de la anomalia (1951-1980)", "degC", fao_src),
        "tmp_era5_decadal": ("Temperatura media decadal (ERA5, area)", "degC", "OWID/Copernicus ERA5"),
        "tmp_level": ("Temperatura media anual (nivel) empalmada; ponderada por poblacion salvo src=4",
                      "degC", "Construida: UDel + FAO + ERA5 (ver tmp_level_src)"),
        "tmp_level_src": ("Fuente de tmp_level: 1 UDel; 2 clim. UDel 1981-2010 + anomalia FAO; "
                          "3 UDel+ERA5 decadal (1950-60); 4 ERA5 area + FAO", "codigo", "Construida"),
        "tmp_level_sq": ("tmp_level al cuadrado (para funcion BHM)", "degC^2", "Construida"),
        "tmp_anom_level": ("tmp_level menos su media 1951-1980", "degC", "Construida"),
        "tmp_pattern_beta": ("Coeficiente de escalamiento de patron: dT_pais/dT_global (OLS 1961-2019)",
                             "degC/degC", "Construida: FAO vs GISTEMP"),
        "fossil_prod_twh": ("Produccion de combustibles fosiles (carbon+petroleo+gas)", "TWh",
                            "OWID energy-data"),
        "ef_cons_pc_2018ed": ("Huella ecologica de consumo per capita", "gha/persona", nfa),
        "biocap_pc_2018ed": ("Biocapacidad per capita", "gha/persona", nfa),
        "ef_prod_pc": ("Huella ecologica de produccion per capita", "gha/persona", nfa),
        "ef_imports_pc": ("Huella incorporada en importaciones per capita", "gha/persona", nfa),
        "ef_exports_pc": ("Huella incorporada en exportaciones per capita", "gha/persona", nfa),
        "ef_cons_gha": ("Huella ecologica de consumo total", "gha", nfa),
        "biocap_gha": ("Biocapacidad total", "gha", nfa),
        "ef_imports_gha": ("Huella en importaciones total", "gha", nfa),
        "ef_exports_gha": ("Huella en exportaciones total", "gha", nfa),
        "ef_carbon_pc": ("Componente carbono de la huella de consumo", "gha/persona", nfa),
        "ef_net_imports_gha": ("Huella neta importada (proxy de intercambio ecologico desigual)", "gha", nfa),
        "ef_net_imports_pc": ("Huella neta importada per capita", "gha/persona", nfa),
        "ef_cons_pc_2025ed": ("Huella de consumo per capita (NFA 2025)", "gha/persona", "GFN NFA 2025"),
        "biocap_pc_2025ed": ("Biocapacidad per capita (NFA 2025)", "gha/persona", "GFN NFA 2025"),
        "ef_cons_pc": ("Huella de consumo per capita (2025ed, si falta 2018ed)", "gha/persona", "GFN"),
        "biocap_pc": ("Biocapacidad per capita (2025ed, si falta 2018ed)", "gha/persona", "GFN"),
        "eco_deficit_pc": ("Deficit ecologico = huella - biocapacidad", "gha/persona", "GFN"),
        "forest_loss_pct": ("Perdida anual de superficie forestal (% del anio previo)", "%", "WDI"),
        "ndgain_index": ("ND-GAIN indice (alto = mejor preparado)", "0-100", "Notre Dame ND-GAIN"),
        "ndgain_vulnerability": ("ND-GAIN vulnerabilidad", "0-1", "Notre Dame ND-GAIN"),
        "ndgain_readiness": ("ND-GAIN preparacion", "0-1", "Notre Dame ND-GAIN"),
        "ndgain_food": ("ND-GAIN vulnerabilidad alimentaria", "0-1", "Notre Dame ND-GAIN"),
        "ndgain_water": ("ND-GAIN vulnerabilidad hidrica", "0-1", "Notre Dame ND-GAIN"),
        "dis_n_total": ("Numero de desastres naturales", "eventos", emdat),
        "dis_n_climate": ("Desastres hidrologicos+meteorologicos+climatologicos", "eventos", emdat),
        "dis_n_flood": ("Inundaciones", "eventos", emdat),
        "dis_n_storm": ("Tormentas (incl. ciclones)", "eventos", emdat),
        "dis_n_drought": ("Sequias", "eventos", emdat),
        "dis_n_extreme_temp": ("Temperaturas extremas", "eventos", emdat),
        "dis_n_wildfire": ("Incendios forestales", "eventos", emdat),
        "dis_n_geo": ("Geofisicos (sismos, volcanes; placebo no climatico)", "eventos", emdat),
        "dis_deaths": ("Muertes por desastres naturales", "personas", emdat),
        "dis_deaths_climate": ("Muertes por desastres climaticos", "personas", emdat),
        "dis_affected": ("Personas afectadas", "personas", emdat),
        "dis_damage_kusd": ("Danos totales ajustados por IPC", "miles USD", emdat),
        "dis_damage_climate_kusd": ("Danos por desastres climaticos, ajustados", "miles USD", emdat),
    }
    meta.update(extra)
    cvars = {}
    for c in df.columns:
        if c in ("iso3", "year"):
            continue
        desc, unit, src = meta.get(c, ("", "", ""))
        cvars[c] = {"desc": desc, "unit": unit, "source": src, "coverage": _cov(df[c], df.year),
                    "n_obs": int(df[c].notna().sum()), "n_countries": int(df.loc[df[c].notna(), "iso3"].nunique())}
    wmeta = {
        "gmst_gistemp": ("Anomalia de temperatura media global, NASA GISTEMP v4 (base 1951-1980)", "degC"),
        "gmst_hadcrut5": ("Anomalia global HadCRUT5 (base media siglo XX, NOAA/GCAG etiqueta)", "degC"),
        "gmst_preind": ("HadCRUT5 rebasada a 1850-1900 (preindustrial)", "degC"),
        "co2_ppm_mlo": ("CO2 atmosferico Mauna Loa (NOAA/Scripps)", "ppm"),
        "co2_ppm_icecore": ("CO2 de nucleos de hielo + instrumental (OWID/NOAA)", "ppm"),
        "co2_ppm": ("CO2 atmosferico: Mauna Loa desde 1959, nucleos de hielo interpolados antes", "ppm"),
        "co2_gtc": ("Emisiones fosiles mundiales en carbono", "GtC"),
        "co2_cum_incl_luc_gtco2": ("Emisiones acumuladas incl. uso del suelo desde 1850", "GtCO2"),
        "overshoot_ratio": ("Huella ecologica mundial / biocapacidad (NFA 2025)", "ratio"),
        "lpi": ("Living Planet Index mundial (1970=100)", "indice"),
        "lpi_hi": ("LPI IC 95% superior", "indice"), "lpi_lo": ("LPI IC 95% inferior", "indice"),
        "material_footprint_t_sum": ("Suma de huellas materiales nacionales", "t"),
    }
    wvars = {}
    for c in w.columns:
        if c == "year":
            continue
        base = c.replace("world_", "")
        desc, unit = wmeta.get(c, meta.get(base, ("", ""))[:2])
        wvars[c] = {"desc": desc, "unit": unit, "coverage": _cov(w[c], w.year)}
    out = {"clima.csv": cvars, "clima_world.csv": wvars, "_estimates": est,
           "_notes": ["Ver docs/research/clima.md", "EM-DAT: ceros = sin eventos registrados; "
                      "subregistro fuerte antes de ~1990", "OWID emisiones en MtCO2 (no GtC)"]}
    (OUT / "clima_dict.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    build()
