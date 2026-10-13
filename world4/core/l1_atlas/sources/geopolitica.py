"""Instituciones y geopolitica: panel pais-anio (ISO3) + series mundiales.

Fuentes (todas descargables via raw.githubusercontent.com):
  * peacesciencer (CRAN mirror, Miller 2022): COW NMC 6.0 (CINC), COW major powers, Maoz powers,
    ATOP 5.1 alianzas (diadas), COW inter-state war 4.0, GML MID 2.2.1 (diadas dirigidas),
    Archigos 4.1 (lideres), Thompson-Dreyer rivalidades estrategicas, ccode_democracy
    (Polity2, V-Dem polyarchy, UDS xm_qudsest), cow_sdp_gdp (Anders et al. PIB/SDP), cow_gw_years.
  * Powell & Thyne (2011) golpes de Estado, version 2023-09-08.
  * Bailey, Strezhnev & Voeten (2017) puntos ideales en la AGNU (Apr 2020, sesiones 1-74).
  * Global Sanctions Data Base (Felbermayr et al. 2020; Kirikakha et al. 2021) v4.
  * WDI DT.ODA.ODAT.GN.ZS: AOD neta recibida (% del INB), release dic-2021.
  * V-Dem v15 (vdemdata, >20 MB -> data/raw/geopolitica/large/, ignorado por git):
    regimes of the world, indices de democracia, capacidad fiscal, autonomia internacional,
    BMR (Boix-Miller-Rosato) transiciones.
  * UCDP/PRIO ACD (ya en data/raw/ucdp_acd.csv via fetch_data.py): conflicto interestatal e
    intervenciones externas (conflictos intraestatales internacionalizados).

Uso:
    .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from sources import geopolitica; geopolitica.build()"
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
RAW = ROOT / "data" / "raw" / "geopolitica"
LARGE = RAW / "large"
OUT = ROOT / "data" / "processed" / "sources"
Y0, Y1 = 1946, 2019

PS = "https://raw.githubusercontent.com/cran/peacesciencer/master/data/"
PS_FILES = ["cow_nmc", "cow_majors", "maoz_powers", "atop_alliance", "cow_alliance", "cow_war_inter",
            "gml_dirdisp", "gml_part", "archigos", "leader_codes", "td_rivalries", "ccode_democracy",
            "cow_sdp_gdp", "cow_gw_years", "cow_states", "gw_states", "cow_mid_ddydisps",
            "gml_mid_ddydisps", "cow_contdir"]
URLS = {f"{f}.rda": PS + f + ".rda" for f in PS_FILES}
URLS.update({
    "powell_thyne_coups_final.txt":
        "https://raw.githubusercontent.com/reddylee/PhD-Coup-Survival/HEAD/data/powell_thyne_coups_final.txt",
    "IdealpointestimatesAll_Apr2020.csv":
        "https://raw.githubusercontent.com/valeriafarinola/UNIdealPointsJulia/HEAD/data/"
        "IdealpointestimatesAll_Apr2020.csv",
    "GSDB_V4.csv": "https://raw.githubusercontent.com/xintongfolio/civil-wars-research-paper/HEAD/GSDB_V4.csv",
    "wdi_oda_gni.csv":
        "https://raw.githubusercontent.com/hdesaioecd/oecd-sof-2022-public/HEAD/data/2020%20sfr%20model%20data/"
        "API_DT/API_DT.ODA.ODAT.GN.ZS_DS2_en_csv_v2_3470450.csv",
    "large/vdem.RData": "https://raw.githubusercontent.com/vdeminstitute/vdemdata/master/data/vdem.RData",
})

# ---------------------------------------------------------------- codigos de pais
# Excepciones comunes COW/GW (continuidad del estado sucesor; entidades extintas -> codigos no ISO)
_COMMON = {255: "DEU", 260: "DEU", 265: "DDR", 315: "CZE", 316: "CZE", 317: "SVK", 340: "SRB",
           345: "SRB", 347: "XKX", 365: "RUS", 678: "YEM", 679: "YEM", 680: "YMD", 816: "VNM",
           817: "RVN", 730: "KOR", 525: "SSD", 529: "ETH", 530: "ETH", 626: "SSD", 713: "TWN",
           396: None, 397: None, 511: None, 1: None, 99: None}
_COW = {970: "NRU", 946: "KIR", 947: "TUV", 955: "TON", 935: "VUT", 591: "SYC", 403: "STP",
        54: "DMA", 55: "GRD", 56: "LCA", 57: "VCT", 58: "ATG", 60: "KNA", 221: "MCO", 223: "LIE",
        232: "AND", 331: "SMR", 812: "LAO", 983: "MHL", 986: "PLW", 987: "FSM", 990: "WSM"}
_GW = {970: "KIR", 971: "NRU", 972: "TON", 973: "TUV", 935: "VUT", 812: "LAO", 591: "SYC"}
NON_ISO = {"DDR", "YMD", "RVN"}
_ISO3: set | None = None


def valid_iso3() -> set:
    """Conjunto de ISO3 validos (country_converter) + Kosovo + estados extintos usados internamente."""
    global _ISO3
    if _ISO3 is None:
        import country_converter as coco
        _ISO3 = set(coco.CountryConverter().data.ISO3.dropna()) | {"XKX"} | NON_ISO
    return _ISO3  # estados extintos: se conservan hasta la agregacion y luego se excluyen


def code_to_iso3(codes, system: str = "cow") -> pd.Series:
    """Convierte codigos COW ('cow') o Gleditsch-Ward ('gw') a ISO3 (con excepciones manuales)."""
    import country_converter as coco
    s = pd.Series(codes).astype("float").round()
    uniq = [int(c) for c in s.dropna().unique()]
    extra = dict(_COMMON)
    extra.update(_COW if system == "cow" else _GW)
    todo = [c for c in uniq if c not in extra]
    logger = logging.getLogger()
    lvl = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        res = coco.CountryConverter().convert(todo, src="GWcode", to="ISO3", not_found=None) if todo else []
    finally:
        logger.setLevel(lvl)
    if isinstance(res, str):
        res = [res]
    m = dict(zip(todo, res))
    m.update(extra)
    ok = valid_iso3()
    m = {k: (v if isinstance(v, str) and v in ok else None) for k, v in m.items()}
    return s.map(lambda c: m.get(int(c)) if pd.notna(c) else None)


# ---------------------------------------------------------------- descargas
def download(force: bool = False) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    LARGE.mkdir(parents=True, exist_ok=True)
    gi = LARGE / ".gitignore"
    if not gi.exists():
        gi.write_text("*\n!.gitignore\n")
    for name, url in URLS.items():
        dst = RAW / name
        if dst.exists() and not force:
            continue
        print("descargando", name)
        urllib.request.urlretrieve(url, dst)
        if dst.stat().st_size < 400 and b"git-lfs" in dst.read_bytes():
            raise RuntimeError(f"{name}: puntero git-lfs, no datos")


def _rda(name: str) -> pd.DataFrame:
    import rdata
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = rdata.read_rda(RAW / f"{name}.rda")
    return pd.DataFrame(obj[name]).reset_index(drop=True)


def _days_to_date(x):
    return pd.to_datetime(pd.Series(x, dtype="float"), unit="D", origin="1970-01-01")


# ---------------------------------------------------------------- bloques pais-anio
def _base() -> pd.DataFrame:
    g = _rda("cow_gw_years")
    g["iso3"] = code_to_iso3(g["ccode"].fillna(g["gwcode"]), "cow")
    g = g[(g.year >= Y0) & g.iso3.notna()][["iso3", "year"]]
    last = g[g.year == g.year.max()].iso3.unique()
    ext = pd.DataFrame([(i, y) for i in last for y in range(int(g.year.max()) + 1, Y1 + 1)],
                       columns=["iso3", "year"])
    b = pd.concat([g, ext]).drop_duplicates()
    b["year"] = b.year.astype(int)
    return b[b.year <= Y1]


def _coups() -> pd.DataFrame:
    d = pd.read_csv(RAW / "powell_thyne_coups_final.txt", sep="\t")
    d["iso3"] = code_to_iso3(d.ccode, "cow")
    d = d[d.iso3.notna()]
    d["succ"] = (d.coup == 2).astype(int)
    d["fail"] = (d.coup == 1).astype(int)
    out = d.groupby(["iso3", "year"]).agg(coup_attempts=("coup", "size"), coup_success=("succ", "sum"),
                                          coup_failed=("fail", "sum")).reset_index()
    return out


def _vdem() -> pd.DataFrame:
    import rdata
    keep = {"v2x_polyarchy": "vdem_polyarchy", "v2x_libdem": "vdem_libdem", "v2x_regime": "regime_row",
            "v2x_ex_military": "vdem_ex_military", "v2x_neopat": "vdem_neopat", "v2x_corr": "vdem_corr",
            "v2x_rule": "vdem_rule", "v2x_jucon": "vdem_jucon", "v2stfisccap": "vdem_fisccap",
            "v2svinlaut": "vdem_intl_autonomy", "v2x_civlib": "vdem_civlib",
            "e_boix_regime": "bmr_democracy", "e_democracy_trans": "bmr_transition",
            "e_democracy_breakdowns": "bmr_breakdowns", "e_chga_demo": "cheibub_democracy"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = rdata.read_rda(LARGE / "vdem.RData")["vdem"]
    v = v[(v.year >= Y0 - 1) & (v.year <= Y1)][["country_text_id", "year"] + list(keep)].rename(columns=keep)
    v = v.rename(columns={"country_text_id": "iso3"})
    v = v[~v.iso3.isin(["PSB", "PSG", "SML", "ZZB"])]
    v["iso3"] = v.iso3.replace({"VDR": "RVN"})
    v["year"] = v.year.astype(int)
    v = v.sort_values(["iso3", "year"])
    prev = v.groupby("iso3").regime_row.shift()
    v["regime_change"] = np.where(prev.notna() & v.regime_row.notna(), (v.regime_row != prev).astype(float), np.nan)
    v["dem_transition"] = np.where(prev.notna() & v.regime_row.notna(),
                                   ((prev <= 1) & (v.regime_row >= 2)).astype(float), np.nan)
    v["dem_breakdown"] = np.where(prev.notna() & v.regime_row.notna(),
                                  ((prev >= 2) & (v.regime_row <= 1)).astype(float), np.nan)
    v["polyarchy_change"] = v.vdem_polyarchy - v.groupby("iso3").vdem_polyarchy.shift()
    return v[v.year >= Y0]


def _democracy_ps() -> pd.DataFrame:
    d = _rda("ccode_democracy")
    d["iso3"] = code_to_iso3(d.ccode, "cow")
    d = d[d.iso3.notna() & (d.year >= Y0)]
    d = d.groupby(["iso3", "year"])[["polity2", "xm_qudsest"]].mean().reset_index()
    return d.rename(columns={"xm_qudsest": "uds_democracy"})


def _leaders() -> pd.DataFrame:
    a = _rda("archigos")
    a["iso3"] = code_to_iso3(a.gwcode, "gw")
    a = a[a.iso3.notna()].copy()
    a["start"] = _days_to_date(a.startdate)
    a["end"] = _days_to_date(a.enddate)
    a["sy"], a["ey"] = a.start.dt.year, a.end.dt.year
    ymax = int(a.ey.max())  # Archigos 4.1 termina 2015
    rows = []
    for iso, g in a.groupby("iso3"):
        g = g.sort_values("start")
        first = g.start.min()
        for y in range(max(Y0, int(g.sy.min())), min(ymax, int(g.ey.max())) + 1):
            dec31 = pd.Timestamp(y, 12, 31)
            ent = g[(g.sy == y) & (g.start > first)]
            ex = g[(g.ey == y) & (g.exit != "Still in Office")]
            cur = g[(g.start <= dec31)].iloc[-1]
            rows.append((iso, y, len(ent), int((ex.exit == "Irregular").sum()), int((ex.exit == "Foreign").sum()),
                         int(ex.exitcode.astype(str).str.contains("Military").sum()),
                         int(ex.exitcode.astype(str).str.contains("Popular Protest").sum()),
                         (min(dec31, cur.end) - cur.start).days / 365.25,
                         1 if str(cur.entry) == "Irregular" else 0))
    return pd.DataFrame(rows, columns=["iso3", "year", "leader_entries", "leader_irregular_exit",
                                       "leader_foreign_removal", "leader_removed_military",
                                       "leader_removed_protest", "leader_tenure", "leader_irregular_entry"])


def _nmc() -> pd.DataFrame:
    m = _rda("cow_nmc")
    m["iso3"] = code_to_iso3(m.ccode, "cow")
    m = m[m.iso3.notna() & (m.year >= Y0)]
    m = m.groupby(["iso3", "year"]).agg(cinc=("cinc", "sum"), milex=("milex", "sum"), milper=("milper", "sum"),
                                        ccode=("ccode", "first")).reset_index()
    m.loc[m.milex < 0, "milex"] = np.nan
    m.loc[m.milper < 0, "milper"] = np.nan
    m["cinc_rank"] = m.groupby("year").cinc.rank(ascending=False, method="min")
    return m


def _majors(base: pd.DataFrame) -> pd.DataFrame:
    mj = _rda("cow_majors")
    mj["iso3"] = code_to_iso3(mj.ccode, "cow")
    out = base[["iso3", "year"]].copy()
    out["major_power"] = 0
    for _, r in mj.iterrows():
        end = Y1 if r.endyear >= 2016 else r.endyear
        out.loc[(out.iso3 == r.iso3) & out.year.between(r.styear, end), "major_power"] = 1
    mz = _rda("maoz_powers")
    mz["iso3"] = code_to_iso3(mz.ccode, "cow")
    out["global_power_maoz"] = 0
    out["regional_power_maoz"] = 0
    for kind, st, en in [("global", "globstdate", "globenddate"), ("regional", "regstdate", "regenddate")]:
        for _, r in mz.dropna(subset=[st]).iterrows():
            y0, y1 = _days_to_date([r[st]]).dt.year[0], _days_to_date([r[en]]).dt.year[0]
            y1 = Y1 if y1 >= 2019 else y1
            out.loc[(out.iso3 == r.iso3) & out.year.between(y0, y1), f"{kind}_power_maoz"] = 1
    return out


def _alliances() -> pd.DataFrame:
    a = _rda("atop_alliance")
    a = a[a.year >= Y0]
    a["iso1"] = code_to_iso3(a.ccode1, "cow")
    a["iso2"] = code_to_iso3(a.ccode2, "cow")
    a = a[a.iso1.notna() & a.iso2.notna() & (a.iso1 != a.iso2)]
    # la base es no dirigida pero incluye ambas direcciones (verificado)
    d = a[a.atop_defense == 1]
    anyal = a[(a[["atop_defense", "atop_offense", "atop_neutral", "atop_nonagg", "atop_consul"]] == 1).any(axis=1)]
    out = d.groupby(["iso1", "year"]).iso2.nunique().rename("n_defense_allies").to_frame()
    out["n_allies_any"] = anyal.groupby(["iso1", "year"]).iso2.nunique()
    for tag, iso in [("us", "USA"), ("rus", "RUS"), ("chn", "CHN")]:
        out[f"ally_{tag}_defense"] = d[d.iso2 == iso].groupby(["iso1", "year"]).size().clip(upper=1)
    out = out.reset_index().rename(columns={"iso1": "iso3"})
    return out


def _unga() -> pd.DataFrame:
    u = pd.read_csv(RAW / "IdealpointestimatesAll_Apr2020.csv", encoding="latin-1")
    u["year"] = u.session + 1945
    u["iso3"] = code_to_iso3(u.ccode, "cow")
    u = u[u.iso3.notna()].groupby(["iso3", "year"]).IdealPoint.mean().reset_index()
    piv = u.pivot(index="year", columns="iso3", values="IdealPoint")
    for tag, iso in [("us", "USA"), ("rus", "RUS"), ("chn", "CHN")]:
        u[f"unga_dist_{tag}"] = (u.IdealPoint - u.year.map(piv[iso])).abs()
    return u.rename(columns={"IdealPoint": "unga_ideal_point"})


def _mids() -> pd.DataFrame:
    g = _rda("gml_dirdisp")
    g["iso3"] = code_to_iso3(g.ccode1, "cow")
    g = g[g.iso3.notna() & (g.year >= Y0)]
    g["fatal"] = (g.fatality1 > 0).astype(int)
    g["init"] = ((g.sidea1 == 1) & (g.orig1 == 1) & (g.midonset == 1)).astype(int)
    g["use_force"] = (g.hostlev1 >= 4).astype(int)
    out = g.groupby(["iso3", "year"]).agg(mid_n=("dispnum", "nunique"), mid_onset=("midonset", "max"),
                                          mid_fatal=("fatal", "max"), mid_initiated=("init", "max"),
                                          mid_hostlev_max=("hostlev1", "max"), mid_use_force=("use_force", "max"))
    out["mid_n_onsets"] = g[g.midonset == 1].groupby(["iso3", "year"]).dispnum.nunique()
    out = out.reset_index()
    out["mid_ongoing"] = 1
    return out


def _wars() -> pd.DataFrame:
    w = _rda("cow_war_inter")
    w["iso3"] = code_to_iso3(w.ccode1, "cow")
    w = w[w.iso3.notna() & (w.year >= Y0)]
    last = w.groupby(["warnum", "iso3"]).year.transform("max")
    w["won"] = ((w.year == last) & (w.outcome1 == 1)).astype(int)
    w["lost"] = ((w.year == last) & (w.outcome1 == 2)).astype(int)
    w["deaths_onset"] = np.where(w.cowinteronset == 1, w.batdeath1.where(w.batdeath1 >= 0), 0)
    w = w.drop_duplicates(["warnum", "iso3", "year"])
    return w.groupby(["iso3", "year"]).agg(cow_war_ongoing=("cowinterongoing", "max"),
                                           cow_war_onset=("cowinteronset", "max"),
                                           cow_war_batdeaths=("deaths_onset", "sum"),
                                           cow_war_won=("won", "max"), cow_war_lost=("lost", "max")).reset_index()


def _ucdp_inter() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "data" / "raw" / "ucdp_acd.csv")
    d = d[(d.year >= Y0) & (d.year <= Y1)]
    it = d[d.type_of_conflict == "interstate"]
    parts = pd.concat([it[["conflict_id", "year", "gwno_a", "intensity_level"]].rename(columns={"gwno_a": "gw"}),
                       it[["conflict_id", "year", "gwno_b", "intensity_level"]].rename(columns={"gwno_b": "gw"})])
    parts["iso3"] = code_to_iso3(parts.gw, "gw")
    inter = parts.dropna(subset=["iso3"]).groupby(["iso3", "year"]).agg(
        ucdp_interstate=("conflict_id", "nunique"), ucdp_interstate_war=("intensity_level", "max")).reset_index()
    inter["ucdp_interstate_war"] = (inter.ucdp_interstate_war == 2).astype(int)
    ii = d[d.type_of_conflict == "II"]
    sec = pd.concat([ii[["conflict_id", "year", "gwno_a_2nd"]].rename(columns={"gwno_a_2nd": "gw"}),
                     ii[["conflict_id", "year", "gwno_b_2nd"]].rename(columns={"gwno_b_2nd": "gw"})]).dropna()
    sec["iso3"] = code_to_iso3(sec.gw, "gw")
    abroad = sec.dropna(subset=["iso3"]).groupby(["iso3", "year"]).conflict_id.nunique().rename(
        "intervention_abroad_n").reset_index()
    host = ii[["conflict_id", "year", "gwno_a"]].copy()
    host["iso3"] = code_to_iso3(host.gwno_a, "gw")
    host = host.dropna(subset=["iso3"]).groupby(["iso3", "year"]).conflict_id.nunique().rename(
        "civil_war_foreign_intervened").reset_index()
    host["civil_war_foreign_intervened"] = 1
    return inter.merge(abroad, how="outer").merge(host, how="outer")


def _rivalries(base, nmc) -> pd.DataFrame:
    t = _rda("td_rivalries")
    t = t[t.endyear >= Y0]
    t["i1"] = code_to_iso3(t.ccode1, "cow")
    t["i2"] = code_to_iso3(t.ccode2, "cow")
    rows = []
    for _, r in t.dropna(subset=["i1", "i2"]).iterrows():
        for y in range(int(max(r.styear, Y0)), int(min(r.endyear, Y1)) + 1):
            rows += [(r.i1, r.i2, y), (r.i2, r.i1, y)]
    rv = pd.DataFrame(rows, columns=["iso3", "rival", "year"])
    c = nmc.set_index(["iso3", "year"]).cinc
    rv["c_own"] = [c.get((a, y), np.nan) for a, y in zip(rv.iso3, rv.year)]
    rv["c_riv"] = [c.get((a, y), np.nan) for a, y in zip(rv.rival, rv.year)]
    rv["parity"] = np.minimum(rv.c_own, rv.c_riv) / np.maximum(rv.c_own, rv.c_riv)
    rv["share"] = rv.c_own / (rv.c_own + rv.c_riv)
    out = rv.groupby(["iso3", "year"]).agg(n_rivalries=("rival", "nunique"), rival_parity_max=("parity", "max"),
                                           rival_power_share_min=("share", "min")).reset_index()
    return out


def _contiguity(nmc: pd.DataFrame, vdem: pd.DataFrame) -> pd.DataFrame:
    """Vecinos contiguos (COW Direct Contiguity 3.2): conteos, paridad CINC maxima y democracias vecinas."""
    c = _rda("cow_contdir")
    c = c[c.conttype.between(1, 3)].copy()  # tierra, o agua <=150 millas
    c["i1"] = code_to_iso3(c.ccode1, "cow")
    c["i2"] = code_to_iso3(c.ccode2, "cow")
    c["y0"] = _days_to_date(c.stdate).dt.year
    c["y1"] = _days_to_date(c.enddate).dt.year
    c.loc[c.y1 >= c.y1.max(), "y1"] = Y1
    rows = []
    for _, r in c.dropna(subset=["i1", "i2"]).iterrows():
        if r.i1 == r.i2:
            continue
        for y in range(int(max(r.y0, Y0)), int(min(r.y1, Y1)) + 1):
            rows.append((r.i1, r.i2, y, int(r.conttype == 1)))
    d = pd.DataFrame(rows, columns=["iso3", "nb", "year", "land"]).drop_duplicates(["iso3", "nb", "year"])
    cinc = nmc.set_index(["iso3", "year"]).cinc
    reg = vdem.set_index(["iso3", "year"]).regime_row
    idx = pd.MultiIndex.from_arrays([d.iso3, d.year])
    nidx = pd.MultiIndex.from_arrays([d.nb, d.year])
    d["c_own"] = cinc.reindex(idx).values
    d["c_nb"] = cinc.reindex(nidx).values
    d["parity"] = np.minimum(d.c_own, d.c_nb) / np.maximum(d.c_own, d.c_nb)
    d["nb_dem"] = (reg.reindex(nidx).values >= 2).astype(float)
    d.loc[np.isnan(reg.reindex(nidx).values), "nb_dem"] = np.nan
    return d.groupby(["iso3", "year"]).agg(n_contig_neighbors=("nb", "nunique"), n_land_neighbors=("land", "sum"),
                                           neighbor_parity_max=("parity", "max"),
                                           neighbor_dem_share=("nb_dem", "mean")).reset_index()


def _oda() -> pd.DataFrame:
    w = pd.read_csv(RAW / "wdi_oda_gni.csv", skiprows=4)
    yrs = [c for c in w.columns if c.strip().isdigit()]
    w = w.melt(id_vars=["Country Code"], value_vars=yrs, var_name="year", value_name="oda_gni")
    w["year"] = w.year.astype(int)
    w = w.rename(columns={"Country Code": "iso3"}).dropna(subset=["oda_gni"])
    return w[w.iso3.isin(valid_iso3())]  # excluye agregados regionales/de ingreso del WDI


def _sanctions() -> pd.DataFrame:
    import country_converter as coco
    s = pd.read_csv(RAW / "GSDB_V4.csv")
    cc = coco.CountryConverter()
    logger = logging.getLogger()
    logger.setLevel(logging.CRITICAL)
    special = {"Soviet Union": "RUS", "Yugoslavia": "SRB", "Serbia and Montenegro": "SRB", "Yemen, North": "YEM",
               "South Vietnam": "RVN", "German Democratic Republic": "DDR", "Yemen, South": "YMD",
               "Palestine": "PSE", "Czechoslovakia": "CZE", "North Vietnam": "VNM", "Rhodesia": "ZWE",
               "West Germany": "DEU", "Germany, West": "DEU", "Germany, East": "DDR", "Zaire": "COD"}

    ok = valid_iso3()

    def conv(x):
        r = cc.convert(x, to="ISO3", not_found=None)
        return r if isinstance(r, str) and r in ok else None

    def targets(name):
        name = str(name).strip()
        if name in special:
            return [special[name]]
        r = conv(name)
        if r:
            return [r]
        out = []
        for p in name.split(","):
            p = p.strip()
            if p in special:
                out.append(special[p])
                continue
            r = conv(p) if p else None
            if r:
                out.append(r)
        return out

    rows = []
    for _, r in s.iterrows():
        us = "United States" in str(r.sanctioning_state)
        eu = any(k in str(r.sanctioning_state) for k in ("EU", "European"))
        un = "United Nations" in str(r.sanctioning_state)
        end = int(r.end) if pd.notna(r.end) else Y1
        for iso in set(targets(r.sanctioned_state)):
            for y in range(int(r.begin), min(end, Y1) + 1):
                rows.append((iso, y, int(y == r.begin), int(us), int(eu), int(un), int(r.trade), int(r.financial)))
    d = pd.DataFrame(rows, columns=["iso3", "year", "onset", "us", "eu", "un", "trade", "fin"])
    return d.groupby(["iso3", "year"]).agg(sanctions_n=("onset", "size"), sanctions_onset=("onset", "max"),
                                           sanctions_us=("us", "max"), sanctions_eu=("eu", "max"),
                                           sanctions_un=("un", "max"), sanctions_trade=("trade", "max"),
                                           sanctions_financial=("fin", "max")).reset_index()


def _gdp() -> pd.DataFrame:
    g = _rda("cow_sdp_gdp")
    g["iso3"] = code_to_iso3(g.ccode, "cow")
    g = g[g.iso3.notna() & (g.year >= Y0)]
    g["gdp"] = np.exp(g.wbgdp2011est)
    return g.groupby(["iso3", "year"]).gdp.sum(min_count=1).reset_index()


# ---------------------------------------------------------------- construccion
def _since_last(df: pd.DataFrame, col: str, name: str, start: int) -> pd.Series:
    """Anios desde el ultimo evento (censurado: cuenta desde `start` o desde la entrada del pais)."""
    out = pd.Series(np.nan, index=df.index)
    for iso, g in df.groupby("iso3"):
        c = 0.0
        vals = []
        for y, v in zip(g.year, g[col]):
            if y < start or pd.isna(v):
                vals.append(np.nan)
                continue
            c = 0.0 if v > 0 else c + 1
            vals.append(c)
        out.loc[g.index] = vals
    return out


def build(force_download: bool = False) -> pd.DataFrame:
    download(force_download)
    OUT.mkdir(parents=True, exist_ok=True)
    base = _base()
    nmc = _nmc()
    vd = _vdem()
    blocks = [_coups(), vd, _democracy_ps(), _leaders(), nmc.drop(columns="ccode"), _majors(base),
              _alliances(), _unga(), _mids(), _wars(), _ucdp_inter(), _rivalries(base, nmc), _contiguity(nmc, vd), _oda(),
              _sanctions()]
    df = base.copy()
    for b in blocks:
        b = b[(b.year >= Y0) & (b.year <= Y1)]
        df = df.merge(b, on=["iso3", "year"], how="outer")
    df = df[df.year.between(Y0, Y1)]
    df = df.drop_duplicates(["iso3", "year"]).sort_values(["iso3", "year"]).reset_index(drop=True)

    # relleno con ceros de eventos dentro de la ventana de cobertura de cada fuente
    windows = {("coup_attempts", "coup_success", "coup_failed"): (1950, 2019),
               ("leader_entries", "leader_irregular_exit", "leader_foreign_removal", "leader_removed_military",
                "leader_removed_protest"): (Y0, 2015),
               ("mid_ongoing", "mid_onset", "mid_n", "mid_n_onsets", "mid_fatal", "mid_initiated",
                "mid_use_force"): (Y0, 2010),
               ("cow_war_ongoing", "cow_war_onset", "cow_war_batdeaths", "cow_war_won", "cow_war_lost"): (Y0, 2003),
               ("ucdp_interstate", "ucdp_interstate_war", "intervention_abroad_n",
                "civil_war_foreign_intervened"): (Y0, 2019),
               ("n_defense_allies", "n_allies_any", "ally_us_defense", "ally_rus_defense",
                "ally_chn_defense"): (Y0, 2018),
               ("n_rivalries",): (Y0, 2010),
               ("n_contig_neighbors", "n_land_neighbors"): (Y0, 2019),  # Thompson-Dreyer termina en 2010
               ("sanctions_n", "sanctions_onset", "sanctions_us", "sanctions_eu", "sanctions_un",
                "sanctions_trade", "sanctions_financial"): (1949, 2019)}
    for cols, (a, b) in windows.items():
        inwin = df.year.between(a, b)
        for c in cols:
            df.loc[inwin, c] = df.loc[inwin, c].fillna(0)
            df.loc[~inwin, c] = np.nan
    df["mid_hostlev_max"] = df.mid_hostlev_max.fillna(0).where(df.year <= 2010)
    df["coup_any"] = (df.coup_attempts > 0).astype(float).where(df.coup_attempts.notna())
    df["coup_success_any"] = (df.coup_success > 0).astype(float).where(df.coup_success.notna())
    df["years_since_coup"] = _since_last(df, "coup_attempts", "years_since_coup", 1950)
    df["coups_past10"] = df.groupby("iso3").coup_attempts.transform(lambda s: s.shift().rolling(10, 1).sum())
    df["log_cinc"] = np.log(df.cinc.where(df.cinc > 0))
    df["bloc"] = (df.ally_us_defense.fillna(0) - df.ally_rus_defense.fillna(0)).where(df.ally_us_defense.notna())
    df.loc[df.iso3 == "USA", "bloc"] = 1
    df.loc[(df.iso3 == "RUS") & df.year.between(Y0, 1991), "bloc"] = -1
    df["cold_war"] = df.year.between(1947, 1991).astype(int)
    df = df[df.iso3.isin(valid_iso3() - NON_ISO)]
    df = df.dropna(axis=0, how="all", subset=[c for c in df.columns if c not in ("iso3", "year", "cold_war")])
    df["year"] = df.year.astype(int)
    df.to_csv(OUT / "geopolitica.csv", index=False)

    world = _world(df, nmc)
    world.to_csv(OUT / "geopolitica_world.csv", index=False)
    calib = _calibration(df)
    (OUT / "geopolitica_dict.json").write_text(json.dumps(
        {"panel": "data/processed/sources/geopolitica.csv", "world": "data/processed/sources/geopolitica_world.csv",
         "years": [Y0, Y1], "variables": VARS, "world_variables": WORLD_VARS, "sources": SOURCES,
         "coverage": _coverage(df), "calibration_targets": calib}, indent=1, ensure_ascii=False))
    print("geopolitica.csv", df.shape, "| world", world.shape)
    return df


def _world(df: pd.DataFrame, nmc: pd.DataFrame) -> pd.DataFrame:
    gdp = _gdp()
    rows = []
    for y in range(Y0, Y1 + 1):
        r = {"year": y}
        c = nmc[nmc.year == y].set_index("iso3").cinc
        if len(c):
            c = c / c.sum()
            s = c.sort_values(ascending=False)
            r.update(cinc_share_usa=c.get("USA"), cinc_share_rus=c.get("RUS"), cinc_share_chn=c.get("CHN"),
                     cinc_top1_iso=s.index[0], cinc_top1_share=s.iloc[0], cinc_top2_share=s.iloc[1],
                     cinc_lead_ratio=s.iloc[0] / s.iloc[1], cinc_hhi=float((c ** 2).sum()),
                     cinc_con=float(np.sqrt(((c ** 2).sum() - 1 / len(c)) / (1 - 1 / len(c)))),
                     chn_usa_cinc_ratio=c.get("CHN") / c.get("USA"),
                     n_poles_10pct=int((c >= 0.10).sum()))
        g = gdp[gdp.year == y].set_index("iso3").gdp.dropna()
        if len(g) and "USA" in g:
            r.update(gdp_share_usa=g["USA"] / g.sum(), gdp_share_chn=g.get("CHN", np.nan) / g.sum(),
                     gdp_share_rus=g.get("RUS", np.nan) / g.sum(),
                     chn_usa_gdp_ratio=g.get("CHN", np.nan) / g["USA"])
        d = df[df.year == y]
        r.update(cold_war=int(1947 <= y <= 1991),
                 bipolar=int(1947 <= y <= 1991),
                 n_states=int(d.iso3.nunique()),
                 share_democracies_row=(d.regime_row >= 2).mean() if d.regime_row.notna().any() else np.nan,
                 mean_polyarchy=d.vdem_polyarchy.mean(),
                 coups_world=d.coup_attempts.sum(min_count=1), coups_success_world=d.coup_success.sum(min_count=1),
                 dem_transitions_world=d.dem_transition.sum(min_count=1),
                 dem_breakdowns_world=d.dem_breakdown.sum(min_count=1),
                 mid_onsets_world=d.mid_n_onsets.sum(min_count=1),
                 states_in_interstate_conflict=d.ucdp_interstate.gt(0).sum() if y <= 2019 else np.nan,
                 states_sanctioned=d.sanctions_n.gt(0).sum(), n_major_powers=int(d.major_power.sum()),
                 mean_oda_gni=d.oda_gni.mean())
        rows.append(r)
    w = pd.DataFrame(rows)
    w.loc[w.year > 2016, [c for c in w.columns if c.startswith(("cinc", "chn_usa_cinc", "n_poles"))]] = np.nan
    w.loc[w.year > 2016, [c for c in w.columns if c.startswith(("gdp_share", "chn_usa_gdp"))]] = np.nan
    return w


def _coverage(df: pd.DataFrame) -> dict:
    cov = {}
    for c in df.columns:
        if c in ("iso3", "year"):
            continue
        s = df.loc[df[c].notna(), ["iso3", "year"]]
        if len(s):
            cov[c] = {"years": [int(s.year.min()), int(s.year.max())], "countries": int(s.iso3.nunique()),
                      "obs": int(len(s))}
    return cov


def _calibration(df: pd.DataFrame) -> dict:
    """Momentos empiricos (1950-2019) para calibrar hazards anuales del modelo."""
    out = {}
    pan = pd.read_csv(ROOT / "data" / "processed" / "panel.csv", usecols=["iso3", "year", "gdppc"])
    d = df.merge(pan, on=["iso3", "year"], how="inner")
    d = d[d.year >= 1950].sort_values(["iso3", "year"])
    d["lgdppc"] = np.log(d.gdppc)
    d["lgdppc_q"] = pd.qcut(d.lgdppc, 4, labels=["q1", "q2", "q3", "q4"])
    c = d[d.coup_attempts.notna()]
    out["coup_attempt_rate"] = float(c.coup_any.mean())
    out["coup_success_share"] = float(c.coup_success.sum() / c.coup_attempts.sum())
    out["coup_rate_by_decade"] = {str(k): round(float(v), 4) for k, v in
                                  c.groupby(c.year // 10 * 10).coup_any.mean().items()}
    out["coup_rate_by_gdppc_quartile"] = {str(k): round(float(v), 4) for k, v in
                                          c.groupby("lgdppc_q", observed=True).coup_any.mean().items()}
    past = c.coups_past10.fillna(0) > 0
    out["coup_rate_if_coup_past10y"] = float(c[past].coup_any.mean())
    out["coup_rate_if_no_coup_past10y"] = float(c[~past].coup_any.mean())
    d["prev_reg"] = d.groupby("iso3").regime_row.shift()
    aut = d[(d.prev_reg <= 1) & d.dem_transition.notna()]
    dem = d[(d.prev_reg >= 2) & d.dem_breakdown.notna()]
    out["dem_transition_rate_autocracies"] = float(aut.dem_transition.mean())
    out["dem_breakdown_rate_democracies"] = float(dem.dem_breakdown.mean())
    out["dem_transition_by_gdppc_quartile"] = {str(k): round(float(v), 4) for k, v in
                                               aut.groupby("lgdppc_q", observed=True).dem_transition.mean().items()}
    out["dem_breakdown_by_gdppc_quartile"] = {str(k): round(float(v), 4) for k, v in
                                              dem.groupby("lgdppc_q", observed=True).dem_breakdown.mean().items()}
    out["gdppc_quartile_cutoffs_usd2017"] = [round(float(np.exp(x)), 0) for x in
                                             d.lgdppc.quantile([0.25, 0.5, 0.75])]
    m = d[d.mid_onset.notna()]
    out["mid_onset_rate"] = float(m.mid_onset.mean())
    out["mid_fatal_rate"] = float(m.mid_fatal.mean())
    out["interstate_conflict_rate_ucdp"] = float((d.ucdp_interstate > 0).mean())
    w = d[d.cow_war_onset.notna()]
    out["interstate_war_onset_rate_cow"] = float(w.cow_war_onset.mean())
    r = d[d.rival_parity_max.notna() & d.mid_onset.notna()]
    if len(r):
        r = r.assign(par_bin=pd.cut(r.rival_parity_max, [0, .2, .5, .8, 1.0001]))
        out["mid_onset_by_rival_parity"] = {str(k): round(float(v), 4) for k, v in
                                            r.groupby("par_bin", observed=True).mid_onset.mean().items()}
    out["share_sanctioned_country_years"] = float((d.sanctions_n > 0).mean())
    out["oda_gni_median_recipients"] = float(d.oda_gni[d.oda_gni > 0].median())
    le = d[d.leader_irregular_exit.notna()]
    out["irregular_leader_exit_rate"] = float((le.leader_irregular_exit > 0).mean())
    # logits de hazard anuales (regresores rezagados; sin efectos fijos; errores estandar no agrupados)
    try:
        import statsmodels.formula.api as smf
        wsh = d.groupby("year").regime_row.apply(lambda s: (s >= 2).mean()).rename("world_dem_share")
        e = pan.merge(df, on=["iso3", "year"]).sort_values(["iso3", "year"])
        e["lg"] = np.log(e.gdppc)
        e["g"] = e.groupby("iso3").lg.diff()
        for v in ("lg", "g", "regime_row", "vdem_ex_military", "neighbor_dem_share"):
            e[v + "_l"] = e.groupby("iso3")[v].shift()
        e = e[e.year >= 1951].join(wsh, on="year")
        e["past"] = (e.coups_past10.fillna(0) > 0).astype(float)
        e["dem_l"] = (e.regime_row_l >= 2).astype(float)
        e["ea_l"] = (e.regime_row_l == 1).astype(float)
        e["cw"] = e.year.between(1947, 1991).astype(float)
        specs = {
            "coup_logit": ("coup_any ~ lg_l + g_l + past + dem_l + cw", e),
            "coup_logit_full": ("coup_any ~ lg_l + g_l + past + dem_l + ea_l + cw + vdem_ex_military_l", e),
            "dem_transition_logit": ("dem_transition ~ lg_l + g_l + ea_l + world_dem_share + cw",
                                     e[e.regime_row_l <= 1]),
            "dem_transition_logit_neighbors": ("dem_transition ~ lg_l + g_l + ea_l + nbdem + cw",
                                               e[e.regime_row_l <= 1].assign(
                                                   nbdem=lambda z: z.neighbor_dem_share_l.fillna(0))),
            "dem_breakdown_logit": ("dem_breakdown ~ lg_l + g_l + world_dem_share + cw", e[e.regime_row_l >= 2]),
        }
        for k, (f, data) in specs.items():
            fit = smf.logit(f, data).fit(disp=0)
            out[k] = {"formula": f, "coef": {a: round(float(b), 3) for a, b in fit.params.items()},
                      "se": {a: round(float(b), 3) for a, b in fit.bse.items()}, "n": int(fit.nobs)}
    except Exception as ex:  # pragma: no cover
        out["logit_error"] = str(ex)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


# ---------------------------------------------------------------- diccionario
VARS = {
    "coup_attempts": "Intentos de golpe (exitosos+fallidos), Powell & Thyne 2011 (v2023-09); 1950-2019",
    "coup_success": "Golpes exitosos (Powell & Thyne)",
    "coup_failed": "Golpes fallidos (Powell & Thyne)",
    "coup_any": "1 si hubo >=1 intento de golpe en el anio",
    "coup_success_any": "1 si hubo >=1 golpe exitoso",
    "years_since_coup": "Anios desde el ultimo intento (censurado a la izquierda en 1950 o entrada del pais)",
    "coups_past10": "Intentos de golpe en los 10 anios previos (t-10..t-1) - 'trampa del golpe'",
    "vdem_polyarchy": "V-Dem v15 indice de democracia electoral (0-1)",
    "vdem_libdem": "V-Dem indice de democracia liberal (0-1)",
    "regime_row": "Regimes of the World (0 autocracia cerrada, 1 autoc. electoral, 2 dem. electoral, 3 dem. liberal)",
    "regime_change": "1 si regime_row cambia respecto de t-1",
    "dem_transition": "1 si pasa de autocracia (RoW<=1) a democracia (RoW>=2)",
    "dem_breakdown": "1 si pasa de democracia (RoW>=2) a autocracia (RoW<=1)",
    "polyarchy_change": "Variacion anual de vdem_polyarchy",
    "vdem_ex_military": "V-Dem dimension militar del ejecutivo (0-1; alto=regimen militar)",
    "vdem_neopat": "V-Dem neopatrimonialismo (0-1)",
    "vdem_corr": "V-Dem corrupcion politica (0-1)",
    "vdem_rule": "V-Dem estado de derecho (0-1)",
    "vdem_jucon": "V-Dem restricciones judiciales al ejecutivo (0-1)",
    "vdem_fisccap": "V-Dem capacidad fiscal del Estado (v2stfisccap, escala latente ~ -3..3)",
    "vdem_intl_autonomy": "V-Dem autonomia internacional del Estado (v2svinlaut; bajo = dominado por potencia extranjera)",
    "vdem_civlib": "V-Dem libertades civiles (0-1)",
    "bmr_democracy": "Boix-Miller-Rosato democracia dicotomica (via V-Dem, hasta 2020)",
    "bmr_transition": "BMR: +1 transicion a democracia, -1 quiebre, 0 nada",
    "bmr_breakdowns": "BMR: numero acumulado de quiebres democraticos previos",
    "cheibub_democracy": "Cheibub-Gandhi-Vreeland democracia dicotomica (hasta 2008)",
    "polity2": "Polity2 (-10..10) via peacesciencer ccode_democracy",
    "uds_democracy": "Unified Democracy Scores extendidos (Marquez xm_qudsest)",
    "leader_entries": "Numero de nuevos lideres que asumen en el anio (Archigos 4.1, hasta 2015)",
    "leader_irregular_exit": "Salidas irregulares de lideres (golpe, revuelta, asesinato...)",
    "leader_foreign_removal": "Lideres depuestos por fuerza extranjera",
    "leader_removed_military": "Lideres removidos por militares (exitcode contiene 'Military')",
    "leader_removed_protest": "Lideres removidos por protesta popular",
    "leader_tenure": "Anios en el cargo del lider vigente al 31-dic",
    "leader_irregular_entry": "1 si el lider vigente llego por via irregular",
    "cinc": "COW NMC 6.0 Composite Index of National Capability (cuota mundial, hasta 2016)",
    "log_cinc": "log(cinc)",
    "milex": "Gasto militar (miles de USD corrientes, COW NMC)",
    "milper": "Personal militar (miles)",
    "cinc_rank": "Ranking mundial por CINC",
    "major_power": "Gran potencia COW (USA, GBR, FRA, RUS, CHN, DEU 1991+, JPN 1991+)",
    "global_power_maoz": "Potencia global segun Maoz",
    "regional_power_maoz": "Potencia regional segun Maoz",
    "n_defense_allies": "Numero de aliados con pacto de defensa (ATOP 5.1, hasta 2018)",
    "n_allies_any": "Numero de aliados con cualquier tipo de compromiso ATOP",
    "ally_us_defense": "1 si tiene pacto de defensa con EE.UU.",
    "ally_rus_defense": "1 si tiene pacto de defensa con URSS/Rusia",
    "ally_chn_defense": "1 si tiene pacto de defensa con China",
    "bloc": "Bloque: +1 aliado de defensa de EE.UU., -1 de URSS/Rusia, 0 ninguno (EE.UU.=+1, URSS=-1 hasta 1991)",
    "cold_war": "1 si 1947-1991",
    "unga_ideal_point": "Punto ideal AGNU (Bailey, Strezhnev & Voeten 2017; anio = sesion+1945)",
    "unga_dist_us": "|punto ideal - punto ideal de EE.UU.|",
    "unga_dist_rus": "|punto ideal - punto ideal de URSS/Rusia|",
    "unga_dist_chn": "|punto ideal - punto ideal de China|",
    "mid_ongoing": "1 si participa en >=1 disputa militarizada (GML MID 2.2.1, hasta 2010)",
    "mid_onset": "1 si alguna MID comienza en el anio",
    "mid_n": "Numero de MIDs en curso",
    "mid_n_onsets": "Numero de MIDs iniciadas",
    "mid_fatal": "1 si alguna MID con muertes del pais",
    "mid_initiated": "1 si el pais origino una MID nueva del lado A (iniciador)",
    "mid_hostlev_max": "Nivel maximo de hostilidad (1-5)",
    "mid_use_force": "1 si hostilidad >=4 (uso de fuerza)",
    "cow_war_ongoing": "1 si participa en guerra interestatal COW (hasta 2003 en esta version)",
    "cow_war_onset": "1 si entra en guerra interestatal",
    "cow_war_batdeaths": "Muertes en combate del pais en la guerra (total, asignado al anio de inicio)",
    "cow_war_won": "1 si termina una guerra ganada",
    "cow_war_lost": "1 si termina una guerra perdida",
    "ucdp_interstate": "Numero de conflictos armados interestatales UCDP (>=25 muertes) en que participa",
    "ucdp_interstate_war": "1 si algun conflicto interestatal UCDP alcanza >=1000 muertes",
    "intervention_abroad_n": "Numero de conflictos intraestatales internacionalizados donde interviene como 2do actor",
    "civil_war_foreign_intervened": "1 si su conflicto interno esta internacionalizado (tropas extranjeras)",
    "n_rivalries": "Rivalidades estrategicas activas (Thompson & Dreyer 2012; hasta 2010)",
    "rival_parity_max": "Max. paridad CINC min/max frente a sus rivales estrategicos",
    "rival_power_share_min": "Min. cuota propia CINC_i/(CINC_i+CINC_rival)",
    "n_contig_neighbors": "Vecinos contiguos por tierra o agua <=150 millas (COW Direct Contiguity 3.2)",
    "n_land_neighbors": "Vecinos con frontera terrestre",
    "neighbor_parity_max": "Max. paridad CINC min/max frente a un vecino contiguo (hasta 2016)",
    "neighbor_dem_share": "Proporcion de vecinos contiguos democraticos (RoW>=2) - difusion regional",
    "oda_gni": "AOD neta recibida % del INB (WDI DT.ODA.ODAT.GN.ZS, 1960-2020)",
    "sanctions_n": "Numero de casos de sancion activos contra el pais (GSDB v4)",
    "sanctions_onset": "1 si comienza un nuevo caso",
    "sanctions_us": "1 si EE.UU. es emisor de alguna sancion activa",
    "sanctions_eu": "1 si la UE/CE es emisora",
    "sanctions_un": "1 si la ONU es emisora",
    "sanctions_trade": "1 si alguna sancion comercial activa",
    "sanctions_financial": "1 si alguna sancion financiera activa",
}
WORLD_VARS = {
    "cinc_share_usa": "Cuota de EE.UU. en el CINC mundial (hegemonia material)",
    "cinc_share_rus": "Cuota URSS/Rusia",
    "cinc_share_chn": "Cuota China",
    "cinc_top1_iso": "Estado con mayor CINC",
    "cinc_top1_share": "Cuota del lider",
    "cinc_top2_share": "Cuota del segundo",
    "cinc_lead_ratio": "Cociente lider/segundo (>1)",
    "cinc_hhi": "Herfindahl de CINC",
    "cinc_con": "Concentracion de Singer-Bremer-Stuckey (0-1)",
    "chn_usa_cinc_ratio": "CINC China / CINC EE.UU. (transicion de poder)",
    "n_poles_10pct": "Numero de estados con >=10% del CINC (polaridad)",
    "gdp_share_usa": "Cuota de EE.UU. en PIB mundial (Anders et al., PPA 2011, hasta 2015/16)",
    "gdp_share_chn": "Cuota de China en PIB mundial",
    "gdp_share_rus": "Cuota URSS/Rusia en PIB mundial",
    "chn_usa_gdp_ratio": "PIB China / PIB EE.UU.",
    "cold_war": "1 si 1947-1991", "bipolar": "Indicador de bipolaridad (= guerra fria)",
    "n_states": "Numero de estados en el panel",
    "share_democracies_row": "Proporcion de democracias (RoW>=2)",
    "mean_polyarchy": "Polyarchy promedio (no ponderado)",
    "coups_world": "Intentos de golpe en el mundo", "coups_success_world": "Golpes exitosos en el mundo",
    "dem_transitions_world": "Transiciones democraticas (RoW)", "dem_breakdowns_world": "Quiebres democraticos (RoW)",
    "mid_onsets_world": "MIDs iniciadas (conteo pais-disputa)",
    "states_in_interstate_conflict": "Estados en conflicto interestatal UCDP",
    "states_sanctioned": "Estados bajo sanciones (GSDB)", "n_major_powers": "Numero de grandes potencias COW",
    "mean_oda_gni": "AOD/INB promedio (receptores)",
}
SOURCES = {
    "powell_thyne": "Powell, J. & Thyne, C. (2011) Global instances of coups from 1950 to 2010. JPR 48(2). v2023-09-08.",
    "vdem": "Coppedge et al. V-Dem Dataset v15 (vdemdata R package, GitHub vdeminstitute/vdemdata).",
    "peacesciencer": "Miller, S. (2022) peacesciencer: An R package for quantitative peace science research. CMPS.",
    "cow_nmc": "Singer, Bremer & Stuckey (1972); COW National Material Capabilities v6.0 (Greig & Enterline).",
    "atop": "Leeds et al. (2002) Alliance Treaty Obligations and Provisions v5.1.",
    "gml_mid": "Gibler, Miller & Little (2016) GML MID dataset v2.2.1.",
    "cow_war": "Sarkees & Wayman (2010) Resort to War; COW inter-state war v4.0.",
    "archigos": "Goemans, Gleditsch & Chiozza (2009) Archigos v4.1.",
    "td_rivalries": "Thompson & Dreyer (2012) Handbook of International Rivalries.",
    "unga": "Bailey, Strezhnev & Voeten (2017) Estimating dynamic state preferences from UN voting data. JCR 61(2).",
    "gsdb": "Felbermayr et al. (2020) The Global Sanctions Data Base. EER; Syropoulos et al. (2023) v3/v4.",
    "wdi_oda": "World Bank WDI DT.ODA.ODAT.GN.ZS (OECD DAC), release 2021-12-16.",
    "ucdp_acd": "Gleditsch et al. (2002); Davies et al. UCDP/PRIO Armed Conflict Dataset.",
    "cow_contdir": "Stinnett et al. (2002) COW Direct Contiguity v3.2 (via peacesciencer).",
    "sdp_gdp": "Anders, Fariss & Markowitz (2020) Bread before guns or butter. ISQ (PIB/SDP estimado).",
}

if __name__ == "__main__":
    build()
