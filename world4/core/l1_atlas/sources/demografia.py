"""Demografia y proxies de asabiya / cohesion -> data/processed/sources/demografia.csv

Panel pais-anio (ISO3, 1950-2030). 1950-2023 = estimaciones; 2024-2030 = proyecciones
UN WPP 2024 variante media (columna ``wpp_projection``=1).

Fuentes (todas descargables desde raw.githubusercontent.com):
  * UN World Population Prospects 2024 (paquete R PPgp/wpp2024): estructura por edad simple,
    TFR, esperanza de vida, migracion neta, tasas brutas, proyecciones con intervalos.
  * World Bank WDI SP.URB.TOTL.IN.ZS (urbanizacion, 1960-2023; release jul-2026).
  * CREG (Nardulli et al.) y HIEF (Drazanova) via cran/peacesciencer: fraccionalizacion y
    polarizacion etnica/religiosa anual 1945-2013; terreno rugoso (Nunn-Puga);
    rivalidades estrategicas Thompson-Dreyer (proxy de frontera metaetnica).
  * EPR 2021 (Vogt et al.): poblacion etnica excluida del poder, 1946-2021.
  * Alesina et al. (2003): fraccionalizacion etnica, linguistica y religiosa (constante).
  * WVS/EVS via Our World in Data: confianza generalizada ("most people can be trusted").
  * V-Dem v14 (vdeminstitute/vdemdata): capacidad estatal (fiscal, territorial, administracion
    impartcial), polarizacion politica, poder por grupo social, legitimacion nacionalista.
  * ICOW Colonial History 1.1 (Hensel): metropoli colonial, anio y violencia de independencia.
  * Edad del estado en el sistema Gleditsch-Ward (andybega/ds-external-data).

Uso:
    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from sources import demografia; demografia.build()"
"""
from __future__ import annotations

import json
import logging
import urllib.request
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from .._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RAW = ROOT / "data" / "raw" / "demografia"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"
YEARS = range(1950, 2031)
LAST_EST = 2023  # ultimo anio de estimaciones WPP 2024

GH = "https://raw.githubusercontent.com/"
WPP = GH + "PPgp/wpp2024/main/data/"
PS = GH + "cran/peacesciencer/master/data/"
SOURCES = {  # nombre local -> URL  (los >20 MB van a large/, ignorado por git)
    "wpp2024_popAge1dt.rda": WPP + "popAge1dt.rda",
    "large/wpp2024_popprojAge1dt.rda": WPP + "popprojAge1dt.rda",
    "wpp2024_popproj1dt.rda": WPP + "popproj1dt.rda",
    "wpp2024_tfr1dt.rda": WPP + "tfr1dt.rda",
    "wpp2024_tfrproj1dt.rda": WPP + "tfrproj1dt.rda",
    "wpp2024_e01dt.rda": WPP + "e01dt.rda",
    "wpp2024_e0proj1dt.rda": WPP + "e0proj1dt.rda",
    "wpp2024_mig1dt.rda": WPP + "mig1dt.rda",
    "wpp2024_migproj1dt.rda": WPP + "migproj1dt.rda",
    "wpp2024_misc1dt.rda": WPP + "misc1dt.rda",
    "wpp2024_miscproj1dt.rda": WPP + "miscproj1dt.rda",
    "wpp2024_UNlocations.txt": WPP + "UNlocations.txt",
    "wdi_urban_share.csv": GH + "hugohe3/ppt-master-examples/HEAD/examples/ppt169_russia_demography_ru/"
                                "sources/API_SP.URB.TOTL.IN.ZS_DS2_en_csv_v2_334241.csv",
    "ps_creg.rda": PS + "creg.rda",
    "ps_hief.rda": PS + "hief.rda",
    "ps_rugged.rda": PS + "rugged.rda",
    "ps_td_rivalries.rda": PS + "td_rivalries.rda",
    "ps_cow_states.rda": PS + "cow_states.rda",
    "ps_gw_states.rda": PS + "gw_states.rda",
    "epr_2021.csv": GH + "andybega/ds-external-data/HEAD/epr/data-raw/EPR-2021.csv",
    "gwstate_age.csv": GH + "andybega/ds-external-data/HEAD/gw-state-age/output/gwstate-age.csv",
    "alesina2003_fractionalization.csv": GH + "lukesonnet/foreign_fighters/HEAD/data/fractionalization.csv",
    "owid_wvs_trust.csv": GH + "adamgreen1708/chart-templates/HEAD/archive/projects/2026-04-trust-in-others/"
                               "data/self-reported-trust-attitudes.csv",
    "icow_coldata110.csv": GH + "globalgov/manystates/HEAD/data-raw/states/ICOW/coldata110.csv",
    "large/vdem.RData": GH + "vdeminstitute/vdemdata/master/data/vdem.RData",
}
VDEM_VARS = ["v2stfisccap", "v2svstterr", "v2clrspct", "v2stcritrecadm", "v2x_rule",
             "v2cacamps", "v2pepwrsoc", "v2xpe_exlsocgr", "v2exl_legitideolcr_1"]

# COW/GW -> ISO3 donde country_converter falla o se equivoca
GW_FIX = {255: "DEU", 260: "DEU", 265: None, 315: "CZE", 316: "CZE", 317: "SVK", 340: "SRB",
          345: "SRB", 347: "XKX", 364: "RUS", 365: "RUS", 678: "YEM", 679: "YEM", 680: None,
          816: "VNM", 817: None, 511: None, 947: "TUV", 6: None, 54: "DMA", 55: "GRD", 56: "LCA",
          57: "VCT", 58: "ATG", 60: "KNA", 221: "MCO", 223: "LIE", 232: "AND", 331: "SMR",
          396: None, 397: None, 835: "BRN", 935: "VUT", 940: "SLB", 946: "KIR", 950: "FJI",
          955: "TON", 970: "KIR", 971: "NRU", 983: "MHL", 986: "PLW", 987: "FSM", 990: "WSM"}


def _coco():
    import country_converter as coco
    logging.getLogger("country_converter").setLevel(logging.CRITICAL)
    return coco.CountryConverter()


def _download():
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        dst = RAW / name
        if not dst.exists():
            print("descargando", name)
            urllib.request.urlretrieve(url, dst)


def _rda(name):
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(RAW / name)
    df = pd.DataFrame(obj[list(obj)[0]]).reset_index(drop=True)
    df.columns = [str(c) for c in df.columns]
    for c in ("country_code", "year", "age"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _gw_iso3(codes, cc):
    codes = pd.Series(codes).astype(float).astype("Int64")
    uniq = [int(c) for c in codes.dropna().unique()]
    conv = dict(zip(uniq, cc.convert(uniq, src="GWcode", to="ISO3", not_found=None)))
    conv = {k: (v if isinstance(v, str) and len(v) == 3 and v.isalpha() else None) for k, v in conv.items()}
    conv.update(GW_FIX)
    return codes.map(lambda c: conv.get(int(c)) if pd.notna(c) else None)


# ----------------------------------------------------------------------------- WPP
def _age_aggregates(df):
    """df: country_code, year, age, pop, popM (miles). Devuelve agregados por pais-anio."""
    df = df.sort_values(["country_code", "year", "age"])
    g = df.groupby(["country_code", "year"], sort=False)
    a = df["age"].to_numpy()

    def s(mask, col="pop"):
        return df[col].where(mask, 0.0).groupby([df.country_code, df.year], sort=False).sum()

    tot = g["pop"].sum()
    out = pd.DataFrame({
        "pop_wpp": tot / 1e3,  # millones
        "share_0_14": s(a <= 14) / tot,
        "share_15_24": s((a >= 15) & (a <= 24)) / tot,
        "share_15_29": s((a >= 15) & (a <= 29)) / tot,
        "share_15_64": s((a >= 15) & (a <= 64)) / tot,
        "share_65p": s(a >= 65) / tot,
        "youth_bulge_15_24_adult": s((a >= 15) & (a <= 24)) / s(a >= 15),
        "youth_bulge_15_29_adult": s((a >= 15) & (a <= 29)) / s(a >= 15),
        "male_share_15_29": s((a >= 15) & (a <= 29), "popM") / tot,
    })
    out["dependency_ratio"] = (out.share_0_14 + out.share_65p) / out.share_15_64
    # mediana: interpolacion lineal dentro del anio de edad donde el acumulado cruza 1/2
    cum = g["pop"].cumsum()
    half = df.groupby(["country_code", "year"], sort=False)["pop"].transform("sum") / 2
    before = cum - df["pop"]
    hit = (before < half) & (cum >= half)
    med = df.loc[hit, "age"] + (half[hit] - before[hit]) / df.loc[hit, "pop"]
    med = df.loc[hit, ["country_code", "year"]].assign(median_age=med.to_numpy())
    med = med.drop_duplicates(["country_code", "year"]).set_index(["country_code", "year"])
    out = out.join(med)
    return out.reset_index()


def _wpp(cc):
    est = _rda("wpp2024_popAge1dt.rda")
    est = est[(est.year >= 1950) & (est.year <= LAST_EST)]
    prj = _rda("large/wpp2024_popprojAge1dt.rda")
    prj = prj[(prj.year > LAST_EST) & (prj.year <= max(YEARS))].copy()
    prj["pop"] = prj.popM + prj.popF
    cols = ["country_code", "year", "age", "pop", "popM"]
    ages = pd.concat([est[cols], prj[cols]])
    ages = ages[ages.country_code < 900]  # >=900 son agregados regionales
    w = _age_aggregates(ages)

    def both(e, p, keep):
        e, p = _rda(e), _rda(p)
        return pd.concat([e[["country_code", "year"] + keep[0]].rename(columns=keep[1]),
                          p[["country_code", "year"] + keep[0]].rename(columns=keep[1])])

    parts = [
        both("wpp2024_tfr1dt.rda", "wpp2024_tfrproj1dt.rda", (["tfr"], {})),
        both("wpp2024_e01dt.rda", "wpp2024_e0proj1dt.rda", (["e0B"], {"e0B": "life_expectancy"})),
        both("wpp2024_mig1dt.rda", "wpp2024_migproj1dt.rda", (["mig"], {"mig": "net_migration"})),
        both("wpp2024_misc1dt.rda", "wpp2024_miscproj1dt.rda",
             (["cbr", "cdr", "growthrate", "cnmr"],
              {"cbr": "crude_birth_rate", "cdr": "crude_death_rate", "growthrate": "pop_growth",
               "cnmr": "net_migration_rate"})),
    ]
    pp = _rda("wpp2024_popproj1dt.rda")[["country_code", "year", "pop_80l", "pop_80u", "pop_low", "pop_high"]]
    pp[["pop_80l", "pop_80u", "pop_low", "pop_high"]] /= 1e3
    pp = pp.rename(columns={"pop_80l": "pop_wpp_lo80", "pop_80u": "pop_wpp_hi80",
                            "pop_low": "pop_wpp_low_var", "pop_high": "pop_wpp_high_var"})
    parts.append(pp)
    for p in parts:
        p = p.drop_duplicates(["country_code", "year"])
        w = w.merge(p, on=["country_code", "year"], how="left")
    w["net_migration"] = w["net_migration"] / 1e3  # millones de personas
    w["pop_growth"] = w["pop_growth"] / 100  # fraccion anual
    w["net_migration_rate"] = w["net_migration_rate"] / 1e3  # por persona
    w["crude_birth_rate"] /= 1e3
    w["crude_death_rate"] /= 1e3
    codes = w.country_code.astype(int).unique().tolist()
    iso = dict(zip(codes, cc.convert(codes, src="ISOnumeric", to="ISO3", not_found=None)))
    iso.update({412: "XKX", 158: "TWN", 830: None, 680: None})  # 830 Channel Islands, 680 Sark
    w["iso3"] = w.country_code.astype(int).map(iso)
    w = w[w.iso3.apply(lambda x: isinstance(x, str) and len(x) == 3 and x.isalpha())]
    w["wpp_projection"] = (w.year > LAST_EST).astype(int)
    return w.drop(columns="country_code")


# ----------------------------------------------------------------------------- urbanizacion
def _logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _urban(iso_years):
    wdi = pd.read_csv(RAW / "wdi_urban_share.csv", skiprows=4)
    yrs = [c for c in wdi.columns if c.isdigit()]
    u = wdi.melt(id_vars="Country Code", value_vars=yrs, var_name="year", value_name="urban_share")
    u = u.rename(columns={"Country Code": "iso3"}).assign(year=lambda d: d.year.astype(int))
    u["urban_share"] /= 100
    u = iso_years.merge(u.dropna(), on=["iso3", "year"], how="left")
    u["urban_share_imputed"] = u.urban_share.isna().astype(int)

    def fill(g):
        g = g.sort_values("year").copy()
        obs = g.dropna(subset=["urban_share"])
        if len(obs) < 5:
            return g
        first, last = obs.year.min(), obs.year.max()
        head = obs[obs.year <= first + 19]  # tendencia 1960-79 (mas estable que 1960-69)
        tail = obs[obs.year >= last - 9]
        b0 = np.polyfit(head.year, _logit(head.urban_share), 1)
        b1 = np.polyfit(tail.year, _logit(tail.urban_share), 1)
        pre, post = g.year < first, g.year > last
        g.loc[pre, "urban_share"] = 1 / (1 + np.exp(-np.polyval(b0, g.loc[pre, "year"])))
        g.loc[post, "urban_share"] = 1 / (1 + np.exp(-np.polyval(b1, g.loc[post, "year"])))
        return g

    u = pd.concat([fill(g) for _, g in u.groupby("iso3")])
    return u


# ----------------------------------------------------------------------------- cohesion
def _creg_hief(cc):
    c = _rda("ps_creg.rda")
    c["iso3"] = _gw_iso3(c.gwcode, cc)
    c = c.rename(columns={k: "creg_" + k for k in ["ethfrac", "ethpol", "relfrac", "relpol"]})
    c = c.dropna(subset=["iso3"]).groupby(["iso3", "year"], as_index=False)[
        ["creg_ethfrac", "creg_ethpol", "creg_relfrac", "creg_relpol"]].first()
    h = _rda("ps_hief.rda")
    h["iso3"] = _gw_iso3(h.gwcode, cc)
    h = h.dropna(subset=["iso3"]).groupby(["iso3", "year"], as_index=False)["efindex"].first()
    h = h.rename(columns={"efindex": "hief_efindex"})
    out = c.merge(h, on=["iso3", "year"], how="outer")
    out["year"] = out.year.astype(int)
    return out


def _epr(cc):
    e = pd.read_csv(RAW / "epr_2021.csv")
    e = e.loc[e.index.repeat(e["to"] - e["from"] + 1)].copy()
    e["year"] = e["from"] + e.groupby(level=0).cumcount()
    excl = e.status.isin(["POWERLESS", "DISCRIMINATED", "SELF-EXCLUSION"])
    e["x"] = e["size"].where(excl, 0)
    e["d"] = e["size"].where(e.status == "DISCRIMINATED", 0)
    e["m"] = e.status.isin(["MONOPOLY", "DOMINANT"]).astype(int)
    e["r"] = (e.status != "IRRELEVANT").astype(int)
    g = e.groupby(["gwid", "year"]).agg(epr_excluded_share=("x", "sum"),
                                         epr_discriminated_share=("d", "sum"),
                                         epr_monopoly_dominant=("m", "max"),
                                         epr_n_groups=("r", "sum"),
                                         epr_largest_group=("size", "max")).reset_index()
    g["iso3"] = _gw_iso3(g.gwid, cc)
    return g.dropna(subset=["iso3"]).groupby(["iso3", "year"], as_index=False).first().drop(columns="gwid")


def _alesina(cc):
    a = pd.read_csv(RAW / "alesina2003_fractionalization.csv").iloc[1:]
    a = a[a.Country.notna() & (a.Country.str.strip() != "")]
    a["Country"] = a.Country.str.replace(r"\s*\(.*\)", "", regex=True).str.strip()
    a["iso3"] = cc.convert(a.Country.tolist(), to="ISO3", not_found=None)
    a = a[a.iso3.apply(lambda x: isinstance(x, str))]
    a = a.rename(columns={"Ethnic": "al_ethnic", "Language": "al_language", "Religion": "al_religion"})
    for c in ["al_ethnic", "al_language", "al_religion"]:
        a[c] = pd.to_numeric(a[c], errors="coerce")
    return a.groupby("iso3", as_index=False)[["al_ethnic", "al_language", "al_religion"]].first()


def _trust():
    t = pd.read_csv(RAW / "owid_wvs_trust.csv")
    t = t[t.Code.str.len() == 3].rename(columns={"Code": "iso3", "Year": "year", "Trust in others": "trust_wvs"})
    t["trust_wvs"] /= 100
    return t[["iso3", "year", "trust_wvs"]]


def _vdem():
    sub = RAW / "vdem_subset.csv"
    if not sub.exists():
        d = _rda("large/vdem.RData")
        d = d[["country_text_id", "year"] + VDEM_VARS]
        d = d[d.year >= 1900].rename(columns={"country_text_id": "iso3"})
        d.to_csv(sub, index=False)
    d = pd.read_csv(sub)
    d["year"] = d.year.astype(int)
    d["v2svstterr"] /= 100
    return d.rename(columns={"v2exl_legitideolcr_1": "v2exl_legit_nationalist"})


def _icow(cc):
    i = pd.read_csv(RAW / "icow_coldata110.csv", encoding="utf-8-sig")
    i["iso3"] = _gw_iso3(i.State, cc)
    ruler = _gw_iso3(i.ColRuler.where(i.ColRuler > 0), cc)
    i["colonial_ruler"] = ruler.fillna("none")
    i["indep_year"] = (i.IndDate // 100).where(i.IndDate > 0)
    i["indep_violent"] = i.IndViol
    i["former_colony"] = (i.ColRuler > 0).astype(int)
    return i.dropna(subset=["iso3"]).groupby("iso3", as_index=False)[
        ["colonial_ruler", "indep_year", "indep_violent", "former_colony"]].first()


def _rivalries(cc):
    r = _rda("ps_td_rivalries.rda")
    rows = []
    for _, x in r.iterrows():
        sp = "spatial" in {x.type1, x.type2, x.type3}
        for code in (x.ccode1, x.ccode2):
            end = max(YEARS) if x.endyear >= 2010 else int(x.endyear)  # 2010 = en curso al cierre
            for y in range(int(max(x.styear, 1816)), end + 1):
                rows.append((code, y, sp))
    d = pd.DataFrame(rows, columns=["ccode", "year", "spatial"])
    d["iso3"] = _gw_iso3(d.ccode, cc)
    g = d.dropna(subset=["iso3"]).groupby(["iso3", "year"]).agg(
        rivalries_active=("spatial", "size"), rivalries_spatial=("spatial", "sum")).reset_index()
    return g


def _statics(cc):
    rg = _rda("ps_rugged.rda")
    rg["iso3"] = _gw_iso3(rg.ccode, cc)
    rg = rg.dropna(subset=["iso3"]).groupby("iso3", as_index=False)[["rugged", "newlmtnest"]].first()
    rg = rg.rename(columns={"newlmtnest": "log_mountainous"})
    return rg


def _asabiya_proxy(df):
    """Proxy compuesto (0.2-0.8) de asabiya: media de z-scores disponibles de
    cohesion (1-frac. etnica, 1-poblacion excluida, confianza) y capacidad estatal V-Dem."""
    base = df[(df.year >= 1950) & (df.year <= 2019)]
    comps = {
        "coh_eth": 1 - df.hief_efindex.fillna(df.creg_ethfrac),
        "coh_excl": 1 - df.epr_excluded_share,
        "coh_trust": df.trust_wvs_interp,
        "cap_fisc": df.v2stfisccap,
        "cap_terr": df.v2svstterr,
        "cap_adm": df.v2clrspct,
    }
    z = {}
    for k, v in comps.items():
        vb = v.loc[base.index]
        z[k] = (v - vb.mean()) / vb.std()
    z = pd.DataFrame(z)
    coh = z[["coh_eth", "coh_excl", "coh_trust"]].mean(axis=1)
    cap = z[["cap_fisc", "cap_terr", "cap_adm"]].mean(axis=1)
    df["state_capacity_index"] = cap
    df["cohesion_index"] = coh
    zbar = pd.concat([coh, cap], axis=1).mean(axis=1).where(z.notna().sum(axis=1) >= 3)
    from scipy.stats import norm
    sd = zbar.loc[base.index].std()
    df["asabiya_proxy"] = 0.2 + 0.6 * norm.cdf(zbar / sd)
    return df


# ----------------------------------------------------------------------------- build
def build():
    _download()
    cc = _coco()
    df = _wpp(cc)
    iso_years = pd.MultiIndex.from_product([sorted(df.iso3.unique()), YEARS], names=["iso3", "year"]).to_frame(index=False)
    df = iso_years.merge(df, on=["iso3", "year"], how="left")
    df = df.merge(_urban(iso_years), on=["iso3", "year"], how="left")
    df = df.sort_values(["iso3", "year"])
    df["urban_pop_growth"] = np.log(df.urban_share * df.pop_wpp).groupby(df.iso3).diff()

    # variables anuales con arrastre hacia adelante (ffill) mas alla de su ultimo anio
    for part, cols in [(_creg_hief(cc), None), (_epr(cc), None), (_vdem(), None)]:
        cols = [c for c in part.columns if c not in ("iso3", "year")]
        df = df.merge(part, on=["iso3", "year"], how="left")
        df[cols] = df.groupby("iso3")[cols].ffill()
    df = df.merge(_rivalries(cc), on=["iso3", "year"], how="left")
    df[["rivalries_active", "rivalries_spatial"]] = df[["rivalries_active", "rivalries_spatial"]].fillna(0)
    # exposicion historica acumulada a rivalidades (anios-rivalidad desde 1816)
    riv = _rivalries(cc)
    riv = riv.sort_values(["iso3", "year"])
    riv["rivalry_years_cum"] = riv.groupby("iso3").rivalries_active.cumsum()
    df = pd.merge_asof(df.sort_values("year"), riv[["iso3", "year", "rivalry_years_cum"]].sort_values("year"),
                       on="year", by="iso3", direction="backward").sort_values(["iso3", "year"])
    df["rivalry_years_cum"] = df.rivalry_years_cum.fillna(0)

    # confianza: observada por ola + interpolada
    tr = _trust()
    df = df.merge(tr, on=["iso3", "year"], how="left")
    df["trust_wvs_interp"] = df.groupby("iso3").trust_wvs.transform(
        lambda s: s.interpolate(limit_area="inside").ffill().bfill())
    df["trust_wvs_mean"] = df.groupby("iso3").trust_wvs.transform("mean")

    # estaticas
    df = df.merge(_alesina(cc), on="iso3", how="left")
    df = df.merge(_icow(cc), on="iso3", how="left")
    df = df.merge(_statics(cc), on="iso3", how="left")
    df["years_since_indep"] = df.year - df.indep_year
    age = pd.read_csv(RAW / "gwstate_age.csv")
    age["iso3"] = _gw_iso3(age.gwcode, cc)
    age = age.dropna(subset=["iso3"]).groupby(["iso3", "year"], as_index=False).state_age.max()
    df = df.merge(age, on=["iso3", "year"], how="left")
    last = df.dropna(subset=["state_age"]).groupby("iso3").apply(
        lambda g: g.loc[g.year.idxmax(), ["year", "state_age"]], include_groups=False)
    m = df.state_age.isna() & df.iso3.isin(last.index)
    ref = last.reindex(df.loc[m, "iso3"])
    df.loc[m, "state_age"] = np.where(df.loc[m, "year"].to_numpy() > ref.year.to_numpy(),
                                      ref.state_age.to_numpy() + df.loc[m, "year"].to_numpy() - ref.year.to_numpy(),
                                      np.nan)
    df = df.reset_index(drop=True)
    df = _asabiya_proxy(df)

    order = ["iso3", "year", "wpp_projection"]
    df = df[order + [c for c in df.columns if c not in order]]
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "demografia.csv", index=False, float_format="%.6g")
    d = _dictionary(df)
    (OUT / "demografia_dict.json").write_text(json.dumps(d, ensure_ascii=False, indent=1))
    print("demografia.csv", df.shape, "paises", df.iso3.nunique())
    return df


DESC = {
    "wpp_projection": ("1 si el anio es proyeccion WPP 2024 variante media (2024-2030)", "0/1", "WPP2024"),
    "pop_wpp": ("Poblacion total 1 julio", "millones", "WPP2024"),
    "share_0_14": ("Proporcion 0-14 anios", "fraccion", "WPP2024"),
    "share_15_24": ("Proporcion 15-24 anios sobre poblacion total", "fraccion", "WPP2024"),
    "share_15_29": ("Proporcion 15-29 anios sobre poblacion total", "fraccion", "WPP2024"),
    "share_15_64": ("Proporcion en edad de trabajar 15-64", "fraccion", "WPP2024"),
    "share_65p": ("Proporcion 65+", "fraccion", "WPP2024"),
    "youth_bulge_15_24_adult": ("Bulto juvenil de Urdal (2006): 15-24 / poblacion 15+", "fraccion", "WPP2024"),
    "youth_bulge_15_29_adult": ("15-29 / poblacion 15+", "fraccion", "WPP2024"),
    "male_share_15_29": ("Varones 15-29 / poblacion total", "fraccion", "WPP2024"),
    "dependency_ratio": ("(0-14 + 65+) / 15-64", "ratio", "WPP2024"),
    "median_age": ("Edad mediana (interpolada por edad simple)", "anios", "WPP2024"),
    "tfr": ("Tasa global de fecundidad", "hijos/mujer", "WPP2024"),
    "life_expectancy": ("Esperanza de vida al nacer, ambos sexos", "anios", "WPP2024"),
    "net_migration": ("Migrantes netos en el anio", "millones de personas", "WPP2024"),
    "crude_birth_rate": ("Tasa bruta de natalidad", "por persona-anio", "WPP2024"),
    "crude_death_rate": ("Tasa bruta de mortalidad", "por persona-anio", "WPP2024"),
    "pop_growth": ("Tasa de crecimiento poblacional", "fraccion anual", "WPP2024"),
    "net_migration_rate": ("Tasa neta de migracion", "por persona-anio", "WPP2024"),
    "pop_wpp_lo80": ("Proyeccion probabilistica: limite inferior 80%", "millones", "WPP2024 (solo 2024+)"),
    "pop_wpp_hi80": ("Proyeccion probabilistica: limite superior 80%", "millones", "WPP2024 (solo 2024+)"),
    "pop_wpp_low_var": ("Variante baja (-0.5 hijos)", "millones", "WPP2024 (solo 2024+)"),
    "pop_wpp_high_var": ("Variante alta (+0.5 hijos)", "millones", "WPP2024 (solo 2024+)"),
    "urban_share": ("Poblacion urbana / total", "fraccion", "WDI SP.URB.TOTL.IN.ZS (UN WUP); 1950-59 y 2024+ extrapolacion logit-lineal"),
    "urban_share_imputed": ("1 si urban_share es extrapolado (fuera de 1960-2023) o falta", "0/1", "calculado"),
    "urban_pop_growth": ("Crecimiento log de la poblacion urbana", "log-dif anual", "calculado"),
    "creg_ethfrac": ("Fraccionalizacion etnica CREG (ffill tras 2013)", "0-1", "CREG via peacesciencer"),
    "creg_ethpol": ("Polarizacion etnica CREG (Montalvo-Reynal-Querol)", "0-1", "CREG via peacesciencer"),
    "creg_relfrac": ("Fraccionalizacion religiosa CREG", "0-1", "CREG via peacesciencer"),
    "creg_relpol": ("Polarizacion religiosa CREG", "0-1", "CREG via peacesciencer"),
    "hief_efindex": ("Indice de fraccionalizacion etnica historica HIEF (Drazanova) (ffill tras 2013)", "0-1", "HIEF via peacesciencer"),
    "epr_excluded_share": ("Poblacion en grupos etnicos excluidos (powerless+discriminated+self-exclusion)", "fraccion", "EPR 2021 (ffill tras 2021)"),
    "epr_discriminated_share": ("Poblacion en grupos discriminados", "fraccion", "EPR 2021"),
    "epr_monopoly_dominant": ("1 si un grupo tiene monopolio o dominancia", "0/1", "EPR 2021"),
    "epr_n_groups": ("Numero de grupos politicamente relevantes", "n", "EPR 2021"),
    "epr_largest_group": ("Tamanio del grupo mayor", "fraccion", "EPR 2021"),
    "v2stfisccap": ("Capacidad fiscal del estado (V-Dem, escala latente ~-3..3)", "z", "V-Dem v14 (ffill)"),
    "v2svstterr": ("Territorio bajo control estatal", "fraccion", "V-Dem v14"),
    "v2clrspct": ("Administracion rigurosa e imparcial", "z", "V-Dem v14"),
    "v2stcritrecadm": ("Criterios de reclutamiento meritocratico de la administracion", "z", "V-Dem v14"),
    "v2x_rule": ("Indice de estado de derecho", "0-1", "V-Dem v14"),
    "v2cacamps": ("Polarizacion politica (campos antagonicos)", "z", "V-Dem v14"),
    "v2pepwrsoc": ("Poder distribuido por grupo social (alto=igualitario)", "z", "V-Dem v14"),
    "v2xpe_exlsocgr": ("Exclusion por grupo social", "0-1", "V-Dem v14"),
    "v2exl_legit_nationalist": ("Legitimacion ideologica nacionalista del regimen (proxy de asabiya ideologica)", "0-1", "V-Dem v14"),
    "rivalries_active": ("Rivalidades estrategicas activas (Thompson-Dreyer; las vigentes en 2010 se suponen en curso hasta 2030)", "n", "peacesciencer td_rivalries"),
    "rivalries_spatial": ("Rivalidades territoriales (espaciales) activas", "n", "peacesciencer td_rivalries"),
    "rivalry_years_cum": ("Anios-rivalidad acumulados desde 1816 (exposicion historica a frontera)", "anios", "calculado"),
    "trust_wvs": ("Confianza generalizada observada ('se puede confiar en la mayoria')", "fraccion", "WVS/EVS via OWID"),
    "trust_wvs_interp": ("Confianza interpolada entre olas, constante fuera", "fraccion", "calculado"),
    "trust_wvs_mean": ("Media por pais de las olas WVS", "fraccion", "calculado"),
    "al_ethnic": ("Fraccionalizacion etnica Alesina et al. 2003", "0-1", "Alesina et al. 2003"),
    "al_language": ("Fraccionalizacion linguistica Alesina et al. 2003", "0-1", "Alesina et al. 2003"),
    "al_religion": ("Fraccionalizacion religiosa Alesina et al. 2003", "0-1", "Alesina et al. 2003"),
    "colonial_ruler": ("ISO3 de la principal metropoli colonial ('none' si ninguna)", "codigo", "ICOW Colonial History 1.1"),
    "indep_year": ("Anio de independencia", "anio", "ICOW"),
    "indep_violent": ("1 si la independencia fue violenta", "0/1", "ICOW"),
    "former_colony": ("1 si tuvo metropoli colonial o imperial (ColRuler>0; incluye otomana, rusa)", "0/1", "ICOW"),
    "years_since_indep": ("Anios desde la independencia (negativo = antes)", "anios", "calculado"),
    "rugged": ("Indice de terreno rugoso (Nunn-Puga)", "indice", "peacesciencer"),
    "log_mountainous": ("log % terreno montanioso (Fearon-Laitin)", "log", "peacesciencer"),
    "state_age": ("Anios en el sistema de estados Gleditsch-Ward (desde 1816; extendido tras 2017)", "anios", "ds-external-data"),
    "state_capacity_index": ("Media de z-scores de v2stfisccap, v2svstterr, v2clrspct (base 1950-2019)", "z", "calculado"),
    "cohesion_index": ("Media de z-scores de 1-frac.etnica (HIEF/CREG), 1-excluidos EPR, confianza", "z", "calculado"),
    "asabiya_proxy": ("Proxy compuesto de asabiya en [0.2,0.8]: 0.2+0.6*Phi(media(cohesion,capacidad)/sd)", "0-1", "calculado"),
}


def _dictionary(df):
    d = {}
    for c in df.columns:
        if c in ("iso3", "year"):
            continue
        desc, unit, src = DESC.get(c, ("", "", ""))
        s = df.loc[df[c].notna(), ["iso3", "year"]]
        d[c] = {"descripcion": desc, "unidad": unit, "fuente": src,
                "cobertura_anios": [int(s.year.min()), int(s.year.max())] if len(s) else None,
                "n_paises": int(s.iso3.nunique()), "n_obs": int(len(s))}
    return d


if __name__ == "__main__":
    build()
