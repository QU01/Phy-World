"""Observed decision variables ("acciones") for behavioral cloning / inverse RL.

Builds a country-year panel (ISO3, 1950-2019) of observable proxies for the six
budget channels of the model (k, r, m, x, f, w) plus military spending, all as
fractions of GDP (0-1) unless the variable name says otherwise (``*_share_*``
variables are fractions of another aggregate, e.g. of merchandise imports).

Raw sources (all fetched from raw.githubusercontent.com, the only data host
reachable from the build environment):

* World Bank WDI per-indicator bulk CSVs (``API_<code>_DS2_en_csv_v2_*.csv``)
  re-hosted in public repositories. For each indicator a *recent* vintage
  (2022-2025 releases) is layered on top of an older full bundle
  (ronnywang/worldbank ``WDI_bundle/parsed``, 1960-2013) which fills the gaps.
* Our World in Data grapher exports (SIPRI military spending 1949-, IMF
  "Public Finances in Modern History" government expenditure 1880-2011,
  ICTD/UNU-WIDER tax revenue 1980-, OECD SOCX + Lindert long-run social
  spending 1880-, Tanzi & Schuknecht public education spending 1870-1993).
* SWIID 9.92 (Solt 2020) market vs disposable Gini -> redistribution.
* Penn World Table 10.01 (``data/raw/pwt1001.csv``, produced by
  ``src/fetch_data.py``): csh_i, csh_g, csh_m, csh_x for 1950-2019 coverage.

Usage::

    import sys; sys.path.insert(0, 'src')
    from . import acciones; acciones.build()
"""
from __future__ import annotations

import io
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "acciones"
OUT = ROOT / "data" / "processed" / "sources"
GH = "https://raw.githubusercontent.com/"
RONNY = "ronnywang/worldbank/HEAD/WDI_bundle/parsed/{}_WDI.csv"
YEARS = (1950, 2019)
# Assumed share of natural-resource rents re-invested in extraction capacity (upstream
# capex / rents). Global upstream oil & gas capex (~0.4-0.5 tn USD/yr, IEA WEI) vs. WDI
# oil+gas+mineral rents (~1.5-3 tn USD/yr) gives ~0.2-0.35. See docs/research/acciones.md.
X_THETA = 0.3

# --------------------------------------------------------------------------
# WDI indicators: variable -> (indicator code, [repo paths, highest priority first])
# The ronnywang bundle (vintage ~2014, data 1960-2013) is always the last fallback.
# --------------------------------------------------------------------------
WDI = {
    "k_gfcf_wdi": ("NE.GDI.FTOT.ZS", []),
    "k_gcf_wdi": ("NE.GDI.TOTL.ZS", []),
    "r_rd_gerd": ("GB.XPD.RSDV.GD.ZS", [
        "IlyaMich/country-compare/HEAD/data/raw/rnd_expenditure_pct_gdp/wb_rnd_expenditure_pct_gdp.csv",
        "zjleee/Data-Analysis-and-Visualization-CA1/HEAD/Datasets/API_GB.XPD.RSDV.GD.ZS_DS2_en_csv_v2_85148.csv",
    ]),
    "r_edu_pub": ("SE.XPD.TOTL.GD.ZS", [
        "hayrullahalper/NumPie/HEAD/Dataset/API_SE.XPD.TOTL.GD.ZS_DS2_en_csv_v2_216037.csv",
    ]),
    "r_hitech_share_mexp": ("TX.VAL.TECH.MF.ZS", [
        "thanhqtran/tohoku_bootcamp/HEAD/2023_spring/computation/Day01/Data01/API_TX.VAL.TECH.MF.ZS_DS2_en_csv_v2_4771883.csv",
    ]),
    "m_imports_wdi": ("NE.IMP.GNFS.ZS", [
        "SaptakBhoumik/stat_assignment/HEAD/data/import_percent/API_NE.IMP.GNFS.ZS_DS2_en_csv_v2_192.csv",
    ]),
    "m_manuf_share_imp": ("TM.VAL.MANF.ZS.UN", []),
    "x_exports_wdi": ("NE.EXP.GNFS.ZS", []),
    "x_fuel_share_exp": ("TX.VAL.FUEL.ZS.UN", [
        "pehls/gp27_techchallenge_4/HEAD/data/raw/fuel_exports/API_TX.VAL.FUEL.ZS.UN_DS2_en_csv_v2_6302702.csv",
    ]),
    "x_ores_share_exp": ("TX.VAL.MMTL.ZS.UN", [
        "QMSS-G5063-2022/Group_J_Climate_Economics/HEAD/data_prep/worldbank/original_from_wb/ore_metal_exports_perc_merchexport.csv",
    ]),
    "x_resource_rents": ("NY.GDP.TOTL.RT.ZS", [
        "hdesaioecd/oecd-sof-2022-public/HEAD/data/2020 sfr model data/API_NY-2/API_NY.GDP.TOTL.RT.ZS_DS2_en_csv_v2_3470550.csv",
    ]),
    "f_fdi_out": ("BM.KLT.DINV.WD.GD.ZS", [
        "dufourlorenzo/replication-fdi-growth-panel/HEAD/data/API_BM.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_316.csv",
        RONNY.format("BM.KLT.DINV.GD.ZS"),  # pre-2014 code of the same series
    ]),
    "f_fdi_in": ("BX.KLT.DINV.WD.GD.ZS", [
        "dufourlorenzo/replication-fdi-growth-panel/HEAD/data/API_BX.KLT.DINV.WD.GD.ZS_DS2_en_csv_v2_397.csv",
    ]),
    "w_gov_cons_wdi": ("NE.CON.GOVT.ZS", [
        "ilovecaffeine/country-data-fingerprint/HEAD/data/raw/API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789/API_NE.CON.GOVT.ZS_DS2_en_csv_v2_38789.csv",
    ]),
    "w_tax_wdi": ("GC.TAX.TOTL.GD.ZS", [
        "SaptakBhoumik/stat_assignment/HEAD/data/tax_percent/API_GC.TAX.TOTL.GD.ZS_DS2_en_csv_v2_962.csv",
    ]),
    "w_gov_exp_wdi": ("GC.XPN.TOTL.GD.ZS", [
        "QMSS-G5063-2022/Group_J_Climate_Economics/HEAD/data_prep/worldbank/original_from_wb/expense_perc_gdp.csv",
    ]),
    "w_transfers_share_exp": ("GC.XPN.TRFT.ZS", []),
    "mil_exp_wdi": ("MS.MIL.XPND.GD.ZS", [
        "S1attaro/Fighting-Illini/HEAD/Final-Project-Military-Spending-Conflict/data/API_MS.MIL.XPND.GD.ZS_DS2_en_csv_v2_211.csv",
    ]),
}

# OWID grapher exports (Entity, Code, Year, value): variable -> (repo path, description, source)
OWID = {
    "mil_exp_sipri": (
        "morettogSonova/archimedes/HEAD/data/in/victor/military-spending-as-a-share-of-gdp-sipri.csv",
        "Military expenditure, % of GDP", "SIPRI Military Expenditure Database via OWID"),
    "w_gov_exp_hist": (
        "mikepsinn/economic-data/HEAD/data/historical-gov-spending-gdp.csv",
        "General government total expenditure, % of GDP (long run)",
        "IMF Public Finances in Modern History (Mauro et al. 2015) via OWID"),
    "w_tax_ictd": (
        "DataScienceLiam/DataScienceLiam.github.io/HEAD/total-tax-revenues-gdp.csv",
        "Total tax revenue incl. social contributions, % of GDP",
        "ICTD/UNU-WIDER Government Revenue Dataset 2021 via OWID"),
    "w_social_exp": (
        "AMGrobelnik/ai-invention-90d6bf-the-triple-shield-revisited-education-we/HEAD/round-2/dataset-1/src/owid_social_spending.csv",
        "Public social expenditure (cash + in-kind: pensions, health, family, unemployment...), % of GDP",
        "OECD SOCX (1980-) + Lindert (2004) historical series via OWID"),
    "r_edu_pub_hist": (
        "Sriharsha-Pamidi/Programming-for-Data-Analysis/HEAD/project/untitled folder/public-education-expenditure-as-share-of-gdp.csv",
        "Public expenditure on education, % of GDP (historical, 17 advanced economies)",
        "Tanzi & Schuknecht (2000) via OWID"),
}

SWIID = "aaravgupta10/Neutrosophic-human-development-index/HEAD/data/inputs/swiid9_92_summary.csv"
PWT_RDA = GH + "cran/pwt10/master/data/pwt10.01.rda"

# World Bank aggregate (non-country) codes to drop
WB_AGG = set("""AFE AFW ARB CEB CSS EAP EAR EAS ECA ECS EMU EUU FCS HIC HPC IBD IBT IDA IDB IDX INX
LAC LCN LDC LIC LMC LMY LTE MEA MIC MNA NAC OED OSS PRE PSS PST SAS SSA SSF SST TEA TEC TLA TMN
TSA TSS UMC WLD EAR LTE""".split())


# --------------------------------------------------------------------------
def _fname(path: str) -> str:
    owner = path.split("/")[0]
    return f"{owner}__{Path(path).name}".replace(" ", "_").replace("%", "pct")


def _fetch(path: str) -> Path:
    """Download ``GH + path`` into data/raw/acciones (cached). Returns local path."""
    RAW.mkdir(parents=True, exist_ok=True)
    dst = RAW / _fname(path)
    if not dst.exists():
        url = GH + urllib.parse.quote(path)
        data = urllib.request.urlopen(url, timeout=120).read()
        if data[:40].startswith(b"version https://git-lfs"):
            raise RuntimeError(f"git-lfs pointer, not data: {url}")
        if len(data) > 20e6:
            (RAW / "large").mkdir(exist_ok=True)
            dst = RAW / "large" / dst.name
        dst.write_bytes(data)
    return dst


def _read_wdi(path: Path, code: str) -> pd.DataFrame:
    """Read a WDI wide CSV (with or without the 4-line API preamble) -> long iso3, year, value."""
    txt = path.read_text(encoding="utf-8-sig", errors="replace")
    lines = txt.splitlines()
    skip = next(i for i, l in enumerate(lines[:10]) if l.lstrip('"').startswith("Country Name"))
    d = pd.read_csv(io.StringIO(txt), skiprows=skip)
    if "Indicator Code" in d.columns:
        d = d[d["Indicator Code"].isin([code, code.replace(".WD.", ".")])]
    yrs = [c for c in d.columns if str(c).strip().isdigit()]
    d = d.melt(id_vars=["Country Code"], value_vars=yrs, var_name="year", value_name="v")
    d = d.rename(columns={"Country Code": "iso3"})
    d["year"] = d["year"].astype(int)
    d["v"] = pd.to_numeric(d["v"], errors="coerce")
    return d.dropna(subset=["v"])


def _wdi_series(code: str, paths: list[str]) -> tuple[pd.Series, list[str]]:
    chain = list(paths)
    if not any(p.startswith("ronnywang/") for p in chain):
        chain.append(RONNY.format(code))
    out, used = None, []
    for p in chain:
        try:
            d = _read_wdi(_fetch(p), code).set_index(["iso3", "year"])["v"]
        except Exception as e:  # noqa: BLE001 - a missing mirror must not break the build
            print(f"  [warn] {code}: {p}: {e}")
            continue
        used.append(p)
        out = d if out is None else out.combine_first(d)
    return out, used


def _read_owid(path: Path) -> pd.Series:
    d = pd.read_csv(path)
    d = d[d["Code"].astype(str).str.len() == 3]
    col = [c for c in d.columns if c not in ("Entity", "Code", "Year") and "region" not in c.lower()][0]
    d = d.rename(columns={"Code": "iso3", "Year": "year", col: "v"})
    return d.set_index(["iso3", "year"])["v"].astype(float)


def _read_swiid(path: Path) -> pd.DataFrame:
    import country_converter as coco
    d = pd.read_csv(path)
    names = d["country"].unique()
    iso = dict(zip(names, coco.convert(list(names), to="ISO3", not_found=None)))
    iso["Kosovo"] = "XKX"
    d["iso3"] = d["country"].map(iso)
    d = d[d["iso3"].notna() & (d["iso3"].str.len() == 3)]
    d = d.groupby(["iso3", "year"]).mean(numeric_only=True)
    out = pd.DataFrame(index=d.index)
    out["w_redist_abs"] = (d["abs_red"]).where(d["abs_red"].notna()) / 100
    out["w_redist_abs_all"] = (d["gini_mkt"] - d["gini_disp"]) / 100
    out["w_redist_rel"] = d["rel_red"] / 100
    out["gini_mkt"] = d["gini_mkt"] / 100
    out["gini_disp"] = d["gini_disp"] / 100
    return out


def _read_pwt() -> pd.DataFrame:
    csv = ROOT / "data" / "raw" / "pwt1001.csv"
    if csv.exists():
        p = pd.read_csv(csv, usecols=["isocode", "year", "csh_i", "csh_g", "csh_m", "csh_x"])
    else:
        import rdata
        tmp = RAW / "pwt10.01.rda"
        if not tmp.exists():
            urllib.request.urlretrieve(PWT_RDA, tmp)
        obj = rdata.read_rda(tmp)
        p = obj[list(obj)[0]][["isocode", "year", "csh_i", "csh_g", "csh_m", "csh_x"]]
    p = p.rename(columns={"isocode": "iso3"})
    p["year"] = p["year"].astype(int)
    p = p.set_index(["iso3", "year"]).astype(float)
    return pd.DataFrame({
        "k_csh_i_pwt": p["csh_i"],
        "w_csh_g_pwt": p["csh_g"],
        "m_csh_m_pwt": -p["csh_m"],  # PWT stores imports as a negative share
        "x_csh_x_pwt": p["csh_x"],
    })


def _interp(s: pd.Series, limit: int = 5) -> pd.Series:
    """Within-country linear interpolation over gaps <= limit years (no extrapolation)."""
    return s.groupby(level="iso3", group_keys=False).apply(
        lambda g: g.interpolate(limit=limit, limit_area="inside"))


def _splice(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Fill gaps in ``primary`` with ``secondary`` rescaled by the country-specific median
    ratio primary/secondary over the overlap (global median if no overlap). Removes the
    level break between e.g. WDI national-price shares and PWT PPP shares."""
    ratio = (primary / secondary).replace([np.inf, -np.inf], np.nan)
    ratio = ratio.where((ratio > 0.1) & (ratio < 10))
    r_c = ratio.groupby(level="iso3").median()
    r = pd.Series(secondary.index.get_level_values("iso3").map(r_c), index=secondary.index)
    r = r.fillna(ratio.median())
    return primary.fillna(secondary * r)


def _ffill(s: pd.Series, limit: int = 6) -> pd.Series:
    return s.groupby(level="iso3", group_keys=False).apply(lambda g: g.ffill(limit=limit))


# --------------------------------------------------------------------------
DESC = {
    "k_gfcf_wdi": "Gross fixed capital formation, share of GDP (current LCU)",
    "k_gcf_wdi": "Gross capital formation (GFCF + inventories), share of GDP",
    "k_csh_i_pwt": "Share of gross capital formation at current PPPs (PWT csh_i)",
    "r_rd_gerd": "Gross domestic expenditure on R&D (GERD, all sectors), share of GDP",
    "r_edu_pub": "Government expenditure on education, share of GDP",
    "r_edu_pub_hist": "Public expenditure on education, share of GDP (historical)",
    "r_hitech_share_mexp": "High-technology exports, share of manufactured exports (OUTCOME of upgrading, not budget)",
    "m_imports_wdi": "Imports of goods and services, share of GDP",
    "m_manuf_share_imp": "Manufactures imports, share of merchandise imports",
    "m_csh_m_pwt": "Imports share at current PPPs (PWT -csh_m)",
    "x_exports_wdi": "Exports of goods and services, share of GDP",
    "x_csh_x_pwt": "Exports share at current PPPs (PWT csh_x)",
    "x_fuel_share_exp": "Fuel exports, share of merchandise exports",
    "x_ores_share_exp": "Ores and metals exports, share of merchandise exports",
    "x_resource_rents": "Total natural resource rents (oil, gas, coal, minerals, forest), share of GDP",
    "f_fdi_out": "Foreign direct investment, net outflows (BoP), share of GDP (can be negative)",
    "f_fdi_in": "Foreign direct investment, net inflows (BoP), share of GDP (foreign extraction/ownership pressure)",
    "w_gov_cons_wdi": "General government final consumption expenditure, share of GDP",
    "w_csh_g_pwt": "Government consumption share at current PPPs (PWT csh_g)",
    "w_tax_wdi": "Tax revenue (central government), share of GDP",
    "w_gov_exp_wdi": "Central government expense, share of GDP",
    "w_transfers_share_exp": "Subsidies and other transfers, share of central government expense",
    "w_redist_abs": "SWIID absolute redistribution (Gini market - Gini disposable)/100, only where SWIID reports it",
    "w_redist_abs_all": "Gini market - Gini disposable (/100), all SWIID imputations",
    "w_redist_rel": "SWIID relative redistribution (abs_red / gini_mkt)",
    "gini_mkt": "SWIID Gini of market income (0-1)",
    "gini_disp": "SWIID Gini of disposable income (0-1)",
    "mil_exp_wdi": "Military expenditure, share of GDP (WDI, from SIPRI)",
    "m_manuf_imports": "DERIVED: imports of manufactures, share of GDP = m_imports_wdi * m_manuf_share_imp (share carried forward <=6y)",
    "x_primary_exports": "DERIVED: fuel+ores exports, share of GDP = exports/GDP (WDI spliced with rescaled PWT csh_x) * (fuel+ores share of merchandise exports)",
    "w_transfers": "DERIVED: subsidies & transfers, share of GDP = w_gov_exp_wdi * w_transfers_share_exp",
    "w_civil_cons": "DERIVED: non-military government consumption = gov. consumption (WDI, else PWT) - military spending",
    "acc_k": "COMPOSITE action k: GFCF (WDI), gaps filled with PWT csh_i rescaled to WDI level (country median ratio)",
    "acc_r": "COMPOSITE action r: R&D (interp., missing->0) + public education (WDI else historical; interp. gaps<=10y and carried +-10y)",
    "acc_m": "COMPOSITE action m: imports/GDP (WDI, spliced with rescaled PWT csh_m) x manufactures share of imports (carried forward, else country median)",
    "acc_x": "COMPOSITE action x: X_THETA(=0.3) x natural resource rents/GDP (assumed re-investment in extraction)",
    "acc_f": "COMPOSITE action f: max(FDI net outflows/GDP, 0)",
    "acc_w": "COMPOSITE action w: social expenditure (OECD/Lindert, interp.) else central-gov subsidies & transfers else civil (non-military) gov. consumption",
    "a_k": "Budget share of k among the six composite actions (row sums to 1 when all six present)",
    "a_r": "Budget share of r", "a_m": "Budget share of m", "a_x": "Budget share of x",
    "a_f": "Budget share of f", "a_w": "Budget share of w",
    "budget_total": "Sum of the six composite actions, share of GDP",
    "n_acc_obs": "Number of composite actions observed (0-6)",
}


def build(verbose: bool = True) -> pd.DataFrame:
    OUT.mkdir(parents=True, exist_ok=True)
    cols, meta = {}, {}

    for var, (code, paths) in WDI.items():
        s, used = _wdi_series(code, paths)
        if s is None:
            continue
        cols[var] = s / 100.0
        meta[var] = {"source": f"World Bank WDI {code}", "files": used,
                     "units": "fraction of GDP" if "_share_" not in var else "fraction"}

    for var, (path, desc, src) in OWID.items():
        try:
            cols[var] = _read_owid(_fetch(path)) / 100.0
            meta[var] = {"source": src, "files": [path], "units": "fraction of GDP"}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {var}: {e}")

    sw = _read_swiid(_fetch(SWIID))
    for c in sw.columns:
        cols[c] = sw[c]
        meta[c] = {"source": "SWIID 9.92 (Solt 2020)", "files": [SWIID],
                   "units": "Gini points / 100" if c != "w_redist_rel" else "fraction of market Gini"}

    pwt = _read_pwt()
    for c in pwt.columns:
        cols[c] = pwt[c]
        meta[c] = {"source": "Penn World Table 10.01 (Feenstra et al. 2015)",
                   "files": ["data/raw/pwt1001.csv"], "units": "fraction of CGDPo"}

    df = pd.DataFrame(cols)
    df.index.names = ["iso3", "year"]
    df = df.reset_index()
    df = df[~df["iso3"].isin(WB_AGG) & df["iso3"].str.fullmatch(r"[A-Z]{3}")]
    df = df[(df["year"] >= YEARS[0]) & (df["year"] <= YEARS[1])]
    df = df.set_index(["iso3", "year"]).sort_index()
    # keep only rows with at least one observed WDI/OWID/SWIID/PWT value
    df = df.dropna(how="all")

    # ---------------- derived variables ----------------
    manuf_share = _ffill(df["m_manuf_share_imp"])
    df["m_manuf_imports"] = df["m_imports_wdi"] * manuf_share
    prim = df[["x_fuel_share_exp", "x_ores_share_exp"]].sum(axis=1, min_count=1)
    df["x_primary_exports"] = _splice(df["x_exports_wdi"], df["x_csh_x_pwt"]) * prim
    df["w_transfers"] = df["w_gov_exp_wdi"] * df["w_transfers_share_exp"]
    gcons = df["w_gov_cons_wdi"].fillna(df["w_csh_g_pwt"])
    mil = df["mil_exp_sipri"].fillna(df["mil_exp_wdi"])
    df["w_civil_cons"] = (gcons - mil.fillna(0)).clip(lower=0)
    for v in ["m_manuf_imports", "x_primary_exports", "w_transfers", "w_civil_cons"]:
        meta[v] = {"source": "derived (see description)", "files": [], "units": "fraction of GDP"}

    # ---------------- composite actions ----------------
    acc = pd.DataFrame(index=df.index)
    acc["acc_k"] = _splice(df["k_gfcf_wdi"], df["k_csh_i_pwt"])
    rd = _interp(df["r_rd_gerd"])
    edu = _interp(df["r_edu_pub"].fillna(df["r_edu_pub_hist"]), limit=10)
    edu = edu.groupby(level="iso3", group_keys=False).apply(
        lambda g: g.ffill(limit=10).bfill(limit=10))
    acc["acc_r"] = edu + rd.fillna(0.0)
    imp = _splice(df["m_imports_wdi"], df["m_csh_m_pwt"])
    share = _ffill(manuf_share, limit=70).fillna(
        manuf_share.groupby(level="iso3").transform("median")).fillna(manuf_share.median())
    acc["acc_m"] = imp * share
    acc["acc_x"] = X_THETA * _interp(df["x_resource_rents"])
    acc["acc_f"] = df["f_fdi_out"].clip(lower=0)
    acc["acc_w"] = (_interp(df["w_social_exp"])
                    .fillna(df["w_transfers"])
                    .fillna(df["w_civil_cons"]))
    # sanity bounds: tiny economies / PWT outliers produce shares <0 or >1
    for c in ["acc_k", "acc_r", "acc_w"]:
        acc[c] = acc[c].where((acc[c] >= 0) & (acc[c] <= 1))
    acc["acc_m"] = acc["acc_m"].where(acc["acc_m"] >= 0).clip(upper=1.0)
    acc["acc_f"] = acc["acc_f"].where(acc["acc_f"] <= 1)
    tot = acc.sum(axis=1, min_count=6)
    df = df.join(acc)
    df["budget_total"] = tot
    for c in acc.columns:
        df["a_" + c[4:]] = acc[c] / tot
    df["n_acc_obs"] = acc.notna().sum(axis=1)
    for v in list(acc.columns) + ["a_k", "a_r", "a_m", "a_x", "a_f", "a_w", "budget_total", "n_acc_obs"]:
        meta[v] = {"source": "composite (see docs/research/acciones.md)", "files": [],
                   "units": "fraction of GDP" if not v.startswith(("a_", "n_")) else
                   ("fraction of budget" if v.startswith("a_") else "count")}

    df = df.reset_index()
    df.to_csv(OUT / "acciones.csv", index=False, float_format="%.6g")

    # ---------------- dictionary with coverage ----------------
    info = {}
    for c in df.columns[2:]:
        s = df[["iso3", "year", c]].dropna()
        info[c] = {
            "description": DESC.get(c, OWID.get(c, (None, c))[1]),
            **meta.get(c, {}),
            "coverage": {
                "n_obs": int(len(s)),
                "n_countries": int(s["iso3"].nunique()),
                "year_min": int(s["year"].min()) if len(s) else None,
                "year_max": int(s["year"].max()) if len(s) else None,
                "n_countries_by_decade": {
                    str(d): int(s[(s.year >= d) & (s.year < d + 10)]["iso3"].nunique())
                    for d in range(1950, 2020, 10)},
            },
        }
    (OUT / "acciones_dict.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    if verbose:
        print(f"acciones.csv: {df.shape[0]} rows, {df['iso3'].nunique()} countries, "
              f"{df.shape[1] - 2} variables")
        for c, v in info.items():
            cv = v["coverage"]
            print(f"  {c:24s} n={cv['n_obs']:6d} countries={cv['n_countries']:3d} "
                  f"{cv['year_min']}-{cv['year_max']}")
    return df


if __name__ == "__main__":
    build()
