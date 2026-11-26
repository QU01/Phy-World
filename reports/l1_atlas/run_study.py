"""Estudio completo de la capa L1 (Atlas del Sistema-Mundo):
  1. Validación histórica fuera de muestra desde 2000 (2001-2025) y desde 2010 (15 años: 2011-2025).
  2. Prueba ciega post-panel 2019 -> 2025 frente a los datos observados más recientes (2020-2025).
  3. Proyección de los próximos 15 años (2026-2041, anclada en 2019, R=100 mundos) para los 38 parámetros
     de salida y los 6 canales de conducta, a escala mundial, por zona (centro/semiperiferia/periferia)
     y para 12 países representativos.
  4. Ablaciones de los 7 mecanismos teóricos en el horizonte 2019-2041 y sensibilidad demográfica post-2030.

Uso:
    python reports/l1_atlas/run_study.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.simplefilter("ignore")

from world4.core.l1_atlas import elites
from world4.core.l1_atlas.behavior import BehaviorModel, BehaviorPolicy
from world4.core.l1_atlas.calibrate import CONF, DEBT, LY, RR, TOP10, W
from world4.core.l1_atlas.model import CHANNEL_NAMES, CHANNELS, Params, Simulator, build_world
from world4.core.l1_atlas.stress import StressConfig
from world4.core.l1_atlas.validate import ABLATIONS
from world4.validate.metrics import auc as auc_safe, crps_ensemble

HERE = Path(__file__).parent
OBS_DIR = HERE / "recent_obs"
ATLAS_ROOT = Path("data_cache/atlas")
RES_DIR = ATLAS_ROOT / "results"
OUT_JSON = HERE / "study_results.json"

HORIZON_END = 2041   # 15 años desde 2026 (2026 -> 2041)
YEARS = list(range(1950, HORIZON_END + 1))
T_FULL = len(YEARS)
SHOWCASE = ["USA", "CHN", "DEU", "KOR", "RUS", "BRA", "MEX", "IND", "ZAF", "ARG", "EGY", "NGA"]


def load_params(path: Path) -> Params:
    d = json.loads(path.read_text(encoding="utf-8"))["params"]
    return Params().with_vector(list(d.keys()), list(d.values()))


def gini_w(x: np.ndarray, w: np.ndarray) -> float:
    ok = np.isfinite(x) & np.isfinite(w) & (x > 0) & (w > 0)
    if ok.sum() < 5:
        return float("nan")
    x, w = x[ok], w[ok]
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    cx = np.cumsum(x * w) / (x * w).sum()
    return float(1.0 - np.sum((cx[1:] + cx[:-1]) * np.diff(cw)) - cx[0] * cw[0])


def zone_of_1d(ly: np.ndarray) -> np.ndarray:
    ok = np.isfinite(ly)
    if ok.sum() < 5:
        return np.full_like(ly, np.nan)
    q80 = np.nanquantile(ly, 0.80)
    q50 = np.nanquantile(ly, 0.50)
    z = np.where(ly >= q80, 2.0, np.where(ly >= q50, 1.0, 0.0))
    z[~ok] = np.nan
    return z


def build_pop_extended_world(W41):
    """Copia de W41 extendiendo la tasa media anual WPP 2025-2030 para 2031-2041 (prueba de sensibilidad)."""
    pop = W41.pop.copy()
    # index 75 = 2025, index 80 = 2030
    g_pop = np.log(np.maximum(pop[:, 80], 1e-6) / np.maximum(pop[:, 75], 1e-6)) / 5.0
    g_pop = np.clip(np.where(np.isfinite(g_pop), g_pop, 0.0), -0.025, 0.035)
    for t in range(81, W41.T):
        # ligera desaceleración natural (0.97 por año hacia 0) como aproximación suave de WPP
        decay = 0.97 ** (t - 80)
        pop[:, t] = pop[:, t - 1] * np.exp(g_pop * decay)
    return dataclasses.replace(W41, pop=pop, _locf=dict(W41._locf))


def load_recent_observations(W41):
    """Carga y alinea series observadas recientes 2010-2025 sobre los 180 países de W41."""
    iso_idx = {c: i for i, c in enumerate(W41.iso3)}
    N = W41.N
    T_obs = 2026 - 1950  # 1950..2025 (indices 0..75)

    # 1. GDP per capita real (PWT 1950-2019 + WDI PPP spliced at 2019 for 2020-2025)
    ly_ext = np.full((N, T_obs), np.nan)
    ly_ext[:, :70] = LY
    d_gdp = json.loads((OBS_DIR / "wdi_gdp_pc_ppp.json").read_text(encoding="utf-8"))[1]
    wdi_y = np.full((N, 16), np.nan)  # 2010..2025
    for r in d_gdp:
        c, y, v = r.get("countryiso3code"), int(r["date"]), r["value"]
        if c in iso_idx and v is not None and 2010 <= y <= 2025:
            wdi_y[iso_idx[c], y - 2010] = float(v)
    # Splice 2020-2025 onto 2019 PWT level using exact WDI PPP growth ratios
    base19_wdi = wdi_y[:, 9]
    base19_pwt = LY[:, 69]
    for yr in range(2020, 2026):
        ratio = wdi_y[:, yr - 2010] / base19_wdi
        ok = np.isfinite(ratio) & (ratio > 0) & np.isfinite(base19_pwt)
        ly_ext[ok, yr - 1950] = base19_pwt[ok] + np.log(ratio[ok])
    # Overlap check 2010->2019
    g_pwt = LY[:, 69] - LY[:, 60]
    g_wdi = np.log(wdi_y[:, 9] / wdi_y[:, 0])
    ok_g = np.isfinite(g_pwt) & np.isfinite(g_wdi)
    ov_gdp = dict(n=int(ok_g.sum()), corr=float(np.corrcoef(g_pwt[ok_g], g_wdi[ok_g])[0, 1]),
                  mae_9y=float(np.mean(np.abs(g_pwt[ok_g] - g_wdi[ok_g]))), last_year=2025)

    # 2. Natural resource rents / GDP (Atlas WDI 1970-2019 + WDI 2020-2021)
    rr_ext = np.full((N, T_obs), np.nan)
    rr_ext[:, :70] = RR
    d_rr = json.loads((OBS_DIR / "wdi_res_rents.json").read_text(encoding="utf-8"))[1]
    wdi_rr = np.full((N, 16), np.nan)
    for r in d_rr:
        c, y, v = r.get("countryiso3code"), int(r["date"]), r["value"]
        if c in iso_idx and v is not None and 2010 <= y <= 2025:
            wdi_rr[iso_idx[c], y - 2010] = float(v) / 100.0
    for yr in (2020, 2021):
        rr_ext[:, yr - 1950] = wdi_rr[:, yr - 2010]
    ok_rr = np.isfinite(RR[:, 69]) & np.isfinite(wdi_rr[:, 9])
    ov_rr = dict(n=int(ok_rr.sum()), corr=float(np.corrcoef(RR[ok_rr, 69], wdi_rr[ok_rr, 9])[0, 1]),
                 mae=float(np.mean(np.abs(RR[ok_rr, 69] - wdi_rr[ok_rr, 9]))), last_year=2021)

    # 3. Public debt / GDP (already in Atlas finanzas.csv through 2024!)
    debt_ext = np.full((N, T_obs), np.nan)
    debt_ext[:, :70] = DEBT
    fin = pd.read_csv(ATLAS_ROOT / "data/processed/sources/finanzas.csv", usecols=["iso3", "year", "debt_gdp"])
    fin = fin[fin.iso3.isin(iso_idx) & fin.year.between(2020, 2024) & fin.debt_gdp.notna()]
    for r in fin.itertuples(index=False):
        if r.debt_gdp < 3.5:
            debt_ext[iso_idx[r.iso3], int(r.year) - 1950] = float(r.debt_gdp)
    ov_debt = dict(n_2024=int(np.isfinite(debt_ext[:, 74]).sum()), last_year=2024, source="finanzas.csv (GMD/IMF)")

    # 4. Democracy (V-Dem Regimes of the World >= 2, through 2025 via OWID)
    demo_ext = np.full((N, T_obs), np.nan)
    raw_row = W.locf("regime_row", np.nan)
    demo_at = np.where(W.observed & np.isfinite(raw_row), (raw_row >= 2).astype(float), np.nan)
    demo_ext[:, :70] = demo_at
    df_reg = pd.read_csv(OBS_DIR / "owid_political_regime.csv")
    dem_labels = {"Electoral democracy", "Liberal democracy"}
    owid_demo = np.full((N, 16), np.nan)
    for r in df_reg.itertuples(index=False):
        if r.code in iso_idx and 2010 <= r.year <= 2025 and pd.notna(r.regime_row_owid):
            owid_demo[iso_idx[r.code], int(r.year) - 2010] = 1.0 if r.regime_row_owid in dem_labels else 0.0
    for yr in range(2020, 2026):
        demo_ext[:, yr - 1950] = owid_demo[:, yr - 2010]
    ok_d = np.isfinite(demo_at[:, 69]) & np.isfinite(owid_demo[:, 9])
    ov_demo = dict(n=int(ok_d.sum()), agree_2019=float(np.mean(demo_at[ok_d, 69] == owid_demo[ok_d, 9])),
                   atlas_2019=float(np.mean(demo_at[ok_d, 69])), owid_2019=float(np.mean(owid_demo[ok_d, 9])),
                   last_year=2025)

    # 5. Civil conflict (UCDP/PRIO ACD v24.1 intrastate type 3 & 4 through 2023)
    conf_ext = np.full((N, T_obs), np.nan)
    conf_ext[:, :70] = CONF
    with zipfile.ZipFile(OBS_DIR / "ucdp_prio_acd_241.zip") as zf:
        udf = pd.read_csv(zf.open("UcdpPrioConflict_v24_1.csv"))
    intra = udf[udf.type_of_conflict.isin([3, 4]) & udf.year.between(2010, 2023)].copy()
    # Map UCDP side_a government names to W.iso3
    name_to_iso = {n.lower(): c for c, n in zip(W.iso3, W.names)}
    manual_map = {
        "afghanistan": "AFG", "algeria": "DZA", "angola": "AGO", "azerbaijan": "AZE",
        "bangladesh": "BGD", "benin": "BEN", "burkina faso": "BFA", "burundi": "BDI",
        "cameroon": "CMR", "central african republic": "CAF", "chad": "TCD", "colombia": "COL",
        "congo": "COG", "dr congo (zaire)": "COD", "egypt": "EGY", "ethiopia": "ETH",
        "india": "IND", "indonesia": "IDN", "iran": "IRN", "iraq": "IRQ", "israel": "ISR",
        "ivory coast": "CIV", "kenya": "KEN", "lebanon": "LBN", "libya": "LBY",
        "mali": "MLI", "mauritania": "MRT", "mexico": "MEX", "mozambique": "MOZ",
        "myanmar (burma)": "MMR", "niger": "NER", "nigeria": "NGA", "pakistan": "PAK",
        "philippines": "PHL", "russia (soviet union)": "RUS", "rwanda": "RWA", "senegal": "SEN",
        "somalia": "SOM", "south sudan": "SSD", "sudan": "SDN", "syria": "SYR",
        "tajikistan": "TJK", "thailand": "THA", "togo": "TGO", "turkey": "TUR",
        "uganda": "UGA", "ukraine": "UKR", "united states of america": "USA",
        "yemen (north yemen)": "YEM", "peru": "PER", "malaysia": "MYS", "jordan": "JOR",
        "tunisia": "TUN", "tanzania": "TZA", "china": "CHN", "france": "FRA",
        "united kingdom": "GBR", "armenia": "ARM", "georgia": "GEO", "serbia (yugoslavia)": "SRB",
        "sri lanka": "LKA", "nepal": "NPL", "cambodia (kampuchea)": "KHM", "laos": "LAO",
        "venezuela": "VEN", "guatemala": "GTM", "el salvador": "SLV", "nicaragua": "NIC",
        "djibouti": "DJI", "eritrea": "ERI", "guinea": "GIN", "sierra leone": "SLE",
        "liberia": "LBR", "comoros": "COM", "madagascar": "MDG", "papua new guinea": "PNG",
        "uzbekistan": "UZB", "kyrgyzstan": "KGZ", "bosnia-herzegovina": "BIH",
        "north macedonia": "MKD", "moldova": "MDA", "croatia": "HRV",
    }
    ucdp_arr = np.zeros((N, 14), dtype=float)  # 2010..2023
    unmatched = set()
    for r in intra.itertuples(index=False):
        sa = str(r.side_a).replace("Government of ", "").strip()
        parts = [p.strip() for p in sa.split(",")]
        for part in parts:
            key = part.lower()
            c = manual_map.get(key) or name_to_iso.get(key)
            if c in iso_idx:
                ucdp_arr[iso_idx[c], int(r.year) - 2010] = 1.0
            else:
                unmatched.add(part)
    # Match active mask from 2019 for 2020-2023
    act19 = W.observed[:, 69]
    for yr in range(2020, 2024):
        conf_ext[act19, yr - 1950] = ucdp_arr[act19, yr - 2010]
    # Check 2010-2019 overlap agreement
    ok_c = np.isfinite(CONF[:, 60:70])
    agree_c = float(np.mean(CONF[:, 60:70][ok_c] == ucdp_arr[:, :10][ok_c]))
    ov_conf = dict(agree_2010_2019=agree_c, atlas_2019=float(np.nanmean(CONF[:, 69])),
                   ucdp_2019=float(ucdp_arr[act19, 9].mean()), ucdp_2023=float(ucdp_arr[act19, 13].mean()),
                   unmatched=sorted(unmatched), last_year=2023)

    # 6. Global series from world_full.csv (already up to 2024-2025!)
    wf = pd.read_csv(ATLAS_ROOT / "data/processed/world_full.csv").set_index("year").reindex(range(1950, 2026))
    gmst_obs = pd.to_numeric(wf["clima__gmst_preind"], errors="coerce").to_numpy(float)
    co2_obs = pd.to_numeric(wf["clima__co2_mt"], errors="coerce").to_numpy(float) / 1000.0  # GtCO2
    price_raw = pd.to_numeric(wf["recursos__jacks_indice_vp2019"], errors="coerce").to_numpy(float)
    price_obs = price_raw / price_raw[20]  # 1970 = 1
    overshoot_obs = pd.to_numeric(wf["clima__overshoot_ratio"], errors="coerce").to_numpy(float)
    debt_w_med = pd.to_numeric(wf["finanzas__world_debt_gdp_median"], errors="coerce").to_numpy(float)
    debt_w_gdpw = pd.to_numeric(wf["finanzas__world_debt_gdp_gdpw"], errors="coerce").to_numpy(float)
    us_rstar = pd.to_numeric(wf["finanzas__us_real_strate"], errors="coerce").to_numpy(float)
    dis_clim = pd.to_numeric(wf["clima__world_dis_n_climate"], errors="coerce").to_numpy(float)

    # Also compute world aggregates from extended country arrays
    pop_obs = W41.pop[:, :T_obs]
    gdp_pc_world = np.full(T_obs, np.nan)
    gdp_pc_med = np.full(T_obs, np.nan)
    gini_obs = np.full(T_obs, np.nan)
    rr_mean_obs = np.full(T_obs, np.nan)
    rr_gdpw_obs = np.full(T_obs, np.nan)
    debt_med_obs = np.full(T_obs, np.nan)
    demo_share_obs = np.full(T_obs, np.nan)
    conf_rate_obs = np.full(T_obs, np.nan)
    elite_mean_obs = np.full(T_obs, np.nan)
    elite_mean_full = np.full(T_obs, np.nan)
    elite_popw_full = np.full(T_obs, np.nan)
    raw_top10 = W.extra.get("top10_share_ptinc", np.full((W.N, W.T), np.nan))
    ridx = elites.region_index(W.iso3)
    MACRO_MASKS = {
        "WEST": np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.WEST_CODES]),
        "ASIA_EM": np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.ASIA_EMERG_CODES]),
        "SOUTH": np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.GLOBAL_SOUTH_CODES]),
        "REF": (ridx == elites.R10_CODES.index("REF")),
    }
    elite_obs_macro = {k: np.full(T_obs, np.nan) for k in MACRO_MASKS}
    # Fixed balanced cohort of countries with full WID series 1990..2019 (avoids entry/exit jumps)
    bal90 = np.isfinite(raw_top10[:, 40]) & np.isfinite(raw_top10[:, 69])
    for t in range(T_obs):
        y = np.exp(ly_ext[:, t])
        ok = np.isfinite(y) & np.isfinite(pop_obs[:, t])
        if ok.sum() > 30:
            gdp_pc_world[t] = float(np.sum(y[ok] * pop_obs[ok, t]) / np.sum(pop_obs[ok, t]))
            gdp_pc_med[t] = float(np.median(y[ok]))
            gini_obs[t] = gini_w(y[ok], pop_obs[ok, t])
        if np.isfinite(rr_ext[:, t]).sum() > 30:
            rr_mean_obs[t] = float(np.nanmean(rr_ext[:, t]))
            ok_rg = np.isfinite(rr_ext[:, t]) & ok
            if ok_rg.sum() > 30:
                Y_tot = y[ok_rg] * pop_obs[ok_rg, t]
                rr_gdpw_obs[t] = float(np.sum(rr_ext[ok_rg, t] * Y_tot) / np.sum(Y_tot))
        if np.isfinite(debt_ext[:, t]).sum() > 30:
            debt_med_obs[t] = float(np.nanmedian(debt_ext[:, t]))
        if np.isfinite(demo_ext[:, t]).sum() > 30:
            demo_share_obs[t] = float(np.nanmean(demo_ext[:, t]))
        if np.isfinite(conf_ext[:, t]).sum() > 30:
            conf_rate_obs[t] = float(np.nanmean(conf_ext[:, t]))
        if t < 70 and np.isfinite(TOP10[:, t]).sum() > 10:
            elite_mean_obs[t] = float(np.nanmean(TOP10[:, t]))
        if t < 70:
            ok_f = np.isfinite(raw_top10[:, t]) & (bal90 if t >= 40 else np.ones(W.N, bool))
            if ok_f.sum() > 30:
                elite_mean_full[t] = float(np.nanmean(raw_top10[ok_f, t]))
                ok_fp = ok_f & np.isfinite(pop_obs[:, t])
                if ok_fp.sum() > 30:
                    elite_popw_full[t] = float(
                        np.sum(raw_top10[ok_fp, t] * pop_obs[ok_fp, t]) / np.sum(pop_obs[ok_fp, t])
                    )
            for mk, mm in MACRO_MASKS.items():
                ok_m = mm & np.isfinite(raw_top10[:, t]) & (bal90 if t >= 40 else np.ones(W.N, bool))
                if ok_m.sum() >= 3:
                    elite_obs_macro[mk][t] = float(np.nanmean(raw_top10[ok_m, t]))

    # Observed channels (1950-2019) from hist_actions
    return {
        "country": dict(ly=ly_ext, rr=rr_ext, debt=debt_ext, demo=demo_ext, conf=conf_ext, top10=TOP10),
        "world": dict(
            years=list(range(1950, 2026)),
            gdp_pc_world=gdp_pc_world, gdp_pc_med=gdp_pc_med, gini=gini_obs,
            rr_mean=rr_mean_obs, rr_gdpw=rr_gdpw_obs,
            debt_med=debt_med_obs, debt_w_med=debt_w_med, debt_w_gdpw=debt_w_gdpw,
            demo_share=demo_share_obs, conf_rate=conf_rate_obs,
            elite_mean=elite_mean_obs, elite_mean_full=elite_mean_full,
            elite_popw_full=elite_popw_full, elite_obs_macro=elite_obs_macro,
            gmst=gmst_obs, co2_gt=co2_obs, price_idx=price_obs, overshoot=overshoot_obs,
            us_rstar=us_rstar, dis_clim=dis_clim,
        ),
        "overlaps": dict(gdp=ov_gdp, rr=ov_rr, debt=ov_debt, demo=ov_demo, conf=ov_conf),
    }


def run_forecast(Wx, params: Params, t0: int, t1: int, R: int = 100, seed: int = 5,
                 overrides: dict | None = None, sim_kwargs: dict | None = None):
    if overrides:
        d = dataclasses.asdict(params)
        d.update(overrides)
        params = Params(**d)
    bm = BehaviorModel(cutoff=t0).fit(W, LY)
    bc = BehaviorPolicy(bm, params, noise=1.0, freeze=t0, seed=seed + 7)
    pol = lambda s, t: s.hist_actions(t) if t <= t0 else bc(s, t)
    sim = Simulator(Wx, params, R=R, seed=seed, **(sim_kwargs or {}))
    out = sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0)
    return out, bm, sim


def summarise_all_38(o: dict, Wx, t_slice: slice):
    """Calcula p10/p50/p90 mundiales por año para las 38 variables de salida del L1."""
    R, N, T = o["ly"][:, :, t_slice].shape
    pop = Wx.pop[:, t_slice]  # (N, T)
    out = {}

    def q3(arr_rt: np.ndarray):
        # arr_rt: (R, T)
        return {
            "p10": np.nanpercentile(arr_rt, 10, axis=0).tolist(),
            "p50": np.nanpercentile(arr_rt, 50, axis=0).tolist(),
            "p90": np.nanpercentile(arr_rt, 90, axis=0).tolist(),
            "mean": np.nanmean(arr_rt, axis=0).tolist(),
        }

    # Special weighted / aggregate metrics
    ly = o["ly"][:, :, t_slice]
    y = np.exp(ly)
    ok = np.isfinite(y)
    # World population-weighted GDP per capita & median GDP pc & Gini
    num = np.nansum(np.where(ok, y * pop[None], 0.0), axis=1)
    den = np.nansum(np.where(ok, pop[None], 0.0), axis=1)
    out["gdp_pc_world"] = q3(num / np.maximum(den, 1e-9))
    out["gdp_pc_med"] = q3(np.nanmedian(y, axis=1))
    gini_rt = np.zeros((R, T))
    for r in range(R):
        for ti in range(T):
            gini_rt[r, ti] = gini_w(y[r, :, ti], pop[:, ti])
    out["gini"] = q3(gini_rt)

    # Direct world scalars
    out["Tg"] = q3(o["Tg"][:, t_slice])
    p0_1970 = o["price"][:, 20:21]
    out["price"] = q3(o["price"][:, t_slice] / np.maximum(p0_1970, 1e-9))

    # CO2 world sum (GtCO2/yr: o['co2'] is MtCO2 per country)
    co2_gt = np.nansum(o["co2"][:, :, t_slice], axis=1) / 1000.0
    out["co2"] = q3(co2_gt)

    # All other country-level REC variables + hazard
    # Choose semantically appropriate cross-country aggregation for each
    MEDIAN_VARS = {"psi", "debt", "dclim", "natcap", "reserves", "gni_pc", "cons_pc", "ly"}
    POPW_VARS = {"gni_pc", "cons_pc"}
    for k in Simulator.REC + ["hazard"]:
        if k in ("co2",):
            continue
        arr = o[k][:, :, t_slice]
        if k in POPW_VARS:
            ok_k = np.isfinite(arr)
            v = np.nansum(np.where(ok_k, arr * pop[None], 0.0), axis=1) / np.maximum(
                np.nansum(np.where(ok_k, pop[None], 0.0), axis=1), 1e-9)
        elif k in MEDIAN_VARS:
            v = np.nanmedian(arr, axis=1)
        else:
            v = np.nanmean(arr, axis=1)
        out[k] = q3(v)
        if k == "E":
            ok_e = np.isfinite(arr)
            out["E_popw"] = q3(
                np.nansum(np.where(ok_e, arr * pop[None], 0.0), axis=1)
                / np.maximum(np.nansum(np.where(ok_e, pop[None], 0.0), axis=1), 1e-9)
            )
        if k == "debt":
            # Also track GDP-weighted debt/GDP for direct comparison with IMF/GMD world series
            Y_rt = np.where(ok, y * pop[None], 0.0)
            ok_d = np.isfinite(arr) & ok
            out["debt_gdpw"] = q3(
                np.nansum(np.where(ok_d, arr * Y_rt, 0.0), axis=1)
                / np.maximum(np.nansum(np.where(ok_d, Y_rt, 0.0), axis=1), 1e-9)
            )
        if k == "res_share":
            Y_rt = np.where(ok, y * pop[None], 0.0)
            ok_r = np.isfinite(arr) & ok
            out["res_share_gdpw"] = q3(
                np.nansum(np.where(ok_r, arr * Y_rt, 0.0), axis=1)
                / np.maximum(np.nansum(np.where(ok_r, Y_rt, 0.0), axis=1), 1e-9)
            )
    # Nuevos observables por defecto: Democracia V2 (poly) y capa biofísica/colas (stress.py)
    if "poly" in o:
        out["poly"] = q3(np.nanmean(o["poly"][:, :, t_slice], axis=1))
    if "kcal" in o:
        kc = o["kcal"][:, :, t_slice]
        ok_kc = np.isfinite(kc)
        out["kcal"] = q3(
            np.nansum(np.where(ok_kc, kc * pop[None], 0.0), axis=1)
            / np.maximum(np.nansum(np.where(ok_kc, pop[None], 0.0), axis=1), 1e-9)
        )
        out["pop_under_2300_m"] = q3(np.nansum(np.where(ok_kc & (kc < 2300.0), pop[None], 0.0), axis=1))
        out["pop_under_2000_m"] = q3(np.nansum(np.where(ok_kc & (kc < 2000.0), pop[None], 0.0), axis=1))
    if "p_food" in o:
        out["p_food"] = q3(o["p_food"][:, t_slice])
    if "yield_idx" in o:
        out["yield_idx"] = q3(np.nanmean(o["yield_idx"][:, :, t_slice], axis=1))
    if "excess_deaths" in o:
        ex_yr = np.nansum(o["excess_deaths"][:, :, t_slice], axis=1)
        out["excess_deaths_yr_m"] = q3(ex_yr)
        out["excess_deaths_cum_m"] = q3(np.cumsum(np.nan_to_num(ex_yr), axis=1))
    if "d_glob" in o:
        out["d_glob"] = q3(o["d_glob"][:, t_slice])
    if "global_shock" in o:
        out["global_shock"] = q3(o["global_shock"][:, t_slice])
    return out


def score_oos(o: dict, obs: dict, origin_year: int, eval_start: int, eval_end: int):
    """Evalúa un pronóstico L1 fuera de muestra frente a datos reales (país-año y mundo)."""
    t0 = origin_year - 1950
    t_start = eval_start - 1950
    t_end = eval_end - 1950
    oc = obs["country"]
    ow = obs["world"]
    res = {}

    # 1. Log GDP per capita (country-year)
    pred_ens = o["ly"][:, :, t_start:t_end + 1]
    truth = oc["ly"][:, t_start:t_end + 1]
    base = np.broadcast_to(oc["ly"][:, t0:t0 + 1], truth.shape)
    ok = np.isfinite(truth) & np.isfinite(base) & np.all(np.isfinite(pred_ens), axis=0)
    if ok.sum() > 20:
        pmean = np.mean(pred_ens, axis=0)
        p10, p90 = np.percentile(pred_ens, 10, axis=0), np.percentile(pred_ens, 90, axis=0)
        err = pmean[ok] - truth[ok]
        err_pers = base[ok] - truth[ok]
        crps = float(np.mean(crps_ensemble(pred_ens[:, ok], truth[ok])))
        # End-year cross-country Spearman & growth correlation
        t_last = t_end - t_start
        ok_end = ok[:, t_last]
        sp = float(spearmanr(pmean[ok_end, t_last], truth[ok_end, t_last])[0]) if ok_end.sum() > 20 else None
        gp = pmean[ok_end, t_last] - oc["ly"][ok_end, t0]
        gt = truth[ok_end, t_last] - oc["ly"][ok_end, t0]
        cg = float(np.corrcoef(gp, gt)[0, 1]) if ok_end.sum() > 20 and np.std(gp) > 0 else None
        # World GDP pc MAPE
        y_sim = np.exp(pred_ens)
        pop_s = W.pop[:, :70] if t_end < 70 else None
        res["gdp_pc"] = dict(
            n=int(ok.sum()), mae_log=float(np.mean(np.abs(err))), rmse_log=float(np.sqrt(np.mean(err ** 2))),
            mae_pct=float(np.mean(np.abs(np.exp(pmean[ok]) - np.exp(truth[ok])) / np.exp(truth[ok])) * 100),
            persist_mae_log=float(np.mean(np.abs(err_pers))),
            persist_rmse_log=float(np.sqrt(np.mean(err_pers ** 2))),
            crps=crps, cover_80=float(np.mean((truth[ok] >= p10[ok]) & (truth[ok] <= p90[ok]))),
            spearman_end=sp, corr_growth_end=cg,
        )

    # 2. Public debt / GDP (country-year)
    t_end_d = min(t_end, 2024 - 1950)
    if t_end_d >= t_start:
        ens_d = o["debt"][:, :, t_start:t_end_d + 1]
        tr_d = oc["debt"][:, t_start:t_end_d + 1]
        bs_d = np.broadcast_to(oc["debt"][:, t0:t0 + 1], tr_d.shape)
        ok_d = np.isfinite(tr_d) & np.isfinite(bs_d) & (tr_d < 3.0) & np.all(np.isfinite(ens_d), axis=0)
        if ok_d.sum() > 20:
            pm = np.mean(ens_d, axis=0)
            p10, p90 = np.percentile(ens_d, 10, axis=0), np.percentile(ens_d, 90, axis=0)
            res["debt_gdp"] = dict(
                n=int(ok_d.sum()), mae=float(np.mean(np.abs(pm[ok_d] - tr_d[ok_d]))),
                rmse=float(np.sqrt(np.mean((pm[ok_d] - tr_d[ok_d]) ** 2))),
                persist_mae=float(np.mean(np.abs(bs_d[ok_d] - tr_d[ok_d]))),
                persist_rmse=float(np.sqrt(np.mean((bs_d[ok_d] - tr_d[ok_d]) ** 2))),
                crps=float(np.mean(crps_ensemble(ens_d[:, ok_d], tr_d[ok_d]))),
                cover_80=float(np.mean((tr_d[ok_d] >= p10[ok_d]) & (tr_d[ok_d] <= p90[ok_d]))),
                last_year=1950 + t_end_d,
            )

    # 3. Resource rents / GDP (country-year)
    t_end_r = min(t_end, 2021 - 1950)
    if t_end_r >= t_start:
        ens_r = o["res_share"][:, :, t_start:t_end_r + 1]
        tr_r = oc["rr"][:, t_start:t_end_r + 1]
        bs_r = np.broadcast_to(oc["rr"][:, t0:t0 + 1], tr_r.shape)
        ok_r = np.isfinite(tr_r) & np.isfinite(bs_r) & np.all(np.isfinite(ens_r), axis=0)
        if ok_r.sum() > 20:
            pm = np.mean(ens_r, axis=0)
            p10, p90 = np.percentile(ens_r, 10, axis=0), np.percentile(ens_r, 90, axis=0)
            res["res_share"] = dict(
                n=int(ok_r.sum()), mae=float(np.mean(np.abs(pm[ok_r] - tr_r[ok_r]))),
                rmse=float(np.sqrt(np.mean((pm[ok_r] - tr_r[ok_r]) ** 2))),
                persist_mae=float(np.mean(np.abs(bs_r[ok_r] - tr_r[ok_r]))),
                persist_rmse=float(np.sqrt(np.mean((bs_r[ok_r] - tr_r[ok_r]) ** 2))),
                crps=float(np.mean(crps_ensemble(ens_r[:, ok_r], tr_r[ok_r]))),
                cover_80=float(np.mean((tr_r[ok_r] >= p10[ok_r]) & (tr_r[ok_r] <= p90[ok_r]))),
                last_year=1950 + t_end_r,
            )

    # 4. Civil conflict (country-year, tie-safe AUC + Brier)
    t_end_c = min(t_end, 2023 - 1950)
    if t_end_c >= t_start:
        hz = np.nanmean(o["hazard"][:, :, t_start:t_end_c + 1], axis=0)
        tr_c = oc["conf"][:, t_start:t_end_c + 1]
        bs_c = np.broadcast_to(np.nanmean(oc["conf"][:, max(0, t0 - 20):t0 + 1], axis=1, keepdims=True), tr_c.shape)
        ok_c = np.isfinite(tr_c) & np.isfinite(hz) & np.isfinite(bs_c)
        if ok_c.sum() > 20:
            res["conflict"] = dict(
                n=int(ok_c.sum()),
                auc=auc_safe(tr_c[ok_c], hz[ok_c]),
                brier=float(np.mean((hz[ok_c] - tr_c[ok_c]) ** 2)),
                pred_rate=float(np.mean(hz[ok_c])),
                obs_rate=float(np.mean(tr_c[ok_c])),
                persist_auc=auc_safe(tr_c[ok_c], bs_c[ok_c]),
                persist_brier=float(np.mean((bs_c[ok_c] - tr_c[ok_c]) ** 2)),
                last_year=1950 + t_end_c,
            )

    # 5. Democracy (country-year, tie-safe AUC + Brier + share)
    ens_dm = o["demo"][:, :, t_start:t_end + 1]
    dm = np.nanmean(ens_dm, axis=0)
    tr_dm = oc["demo"][:, t_start:t_end + 1]
    bs_dm = np.broadcast_to(oc["demo"][:, t0:t0 + 1], tr_dm.shape)
    ok_dm = np.isfinite(tr_dm) & np.isfinite(dm) & np.isfinite(bs_dm)
    if ok_dm.sum() > 20:
        res["democracy"] = dict(
            n=int(ok_dm.sum()),
            auc=auc_safe(tr_dm[ok_dm], dm[ok_dm]),
            brier=float(np.mean((dm[ok_dm] - tr_dm[ok_dm]) ** 2)),
            pred_share=float(np.mean(dm[ok_dm])),
            obs_share=float(np.mean(tr_dm[ok_dm])),
            persist_auc=auc_safe(tr_dm[ok_dm], bs_dm[ok_dm]),
            persist_brier=float(np.mean((bs_dm[ok_dm] - tr_dm[ok_dm]) ** 2)),
            last_year=1950 + t_end,
        )

    # 6. Global Temperature GMST (°C over 1850-1900)
    tg_ens = o["Tg"][:, t_start:t_end + 1]
    tg_obs = ow["gmst"][t_start:t_end + 1]
    ok_t = np.isfinite(tg_obs)
    if ok_t.sum() > 0:
        tm = np.mean(tg_ens[:, ok_t], axis=0)
        p10, p90 = np.percentile(tg_ens[:, ok_t], 10, axis=0), np.percentile(tg_ens[:, ok_t], 90, axis=0)
        res["gmst"] = dict(
            n=int(ok_t.sum()), mae=float(np.mean(np.abs(tm - tg_obs[ok_t]))),
            rmse=float(np.sqrt(np.mean((tm - tg_obs[ok_t]) ** 2))),
            persist_mae=float(np.mean(np.abs(ow["gmst"][t0] - tg_obs[ok_t]))),
            cover_80=float(np.mean((tg_obs[ok_t] >= p10) & (tg_obs[ok_t] <= p90))),
            pred_end=float(tm[-1]), obs_end=float(tg_obs[ok_t][-1]),
            last_year=1950 + t_end,
        )

    # 7. Global CO2 emissions (GtCO2/yr)
    t_end_co2 = min(t_end, 2024 - 1950)
    if t_end_co2 >= t_start:
        co2_ens = np.nansum(o["co2"][:, :, t_start:t_end_co2 + 1], axis=1) / 1000.0
        co2_tr = ow["co2_gt"][t_start:t_end_co2 + 1]
        ok_co2 = np.isfinite(co2_tr)
        if ok_co2.sum() > 0:
            cm = np.mean(co2_ens[:, ok_co2], axis=0)
            p10, p90 = np.percentile(co2_ens[:, ok_co2], 10, axis=0), np.percentile(co2_ens[:, ok_co2], 90, axis=0)
            res["co2"] = dict(
                n=int(ok_co2.sum()), mae_gt=float(np.mean(np.abs(cm - co2_tr[ok_co2]))),
                mape_pct=float(np.mean(np.abs(cm - co2_tr[ok_co2]) / co2_tr[ok_co2]) * 100),
                persist_mape_pct=float(np.mean(np.abs(ow["co2_gt"][t0] - co2_tr[ok_co2]) / co2_tr[ok_co2]) * 100),
                cover_80=float(np.mean((co2_tr[ok_co2] >= p10) & (co2_tr[ok_co2] <= p90))),
                pred_end=float(cm[-1]), obs_end=float(co2_tr[ok_co2][-1]),
                last_year=1950 + t_end_co2,
            )

    # 8. Elite income share (Top 10%, only up to 2019)
    t_end_e = min(t_end, 2019 - 1950)
    if t_end_e >= t_start:
        ens_e = o["E"][:, :, t_start:t_end_e + 1]
        tr_e = oc["top10"][:, t_start:t_end_e + 1]
        ok_e = np.isfinite(tr_e) & np.all(np.isfinite(ens_e), axis=0)
        if ok_e.sum() > 20:
            pm = np.mean(ens_e, axis=0)
            p10, p90 = np.percentile(ens_e, 10, axis=0), np.percentile(ens_e, 90, axis=0)
            res["elite_share"] = dict(
                n=int(ok_e.sum()), mae=float(np.mean(np.abs(pm[ok_e] - tr_e[ok_e]))),
                rmse=float(np.sqrt(np.mean((pm[ok_e] - tr_e[ok_e]) ** 2))),
                cover_80=float(np.mean((tr_e[ok_e] >= p10[ok_e]) & (tr_e[ok_e] <= p90[ok_e]))),
                last_year=1950 + t_end_e,
            )

    return res


def zone_breakdown(o: dict, Wx, year_indices: list[int]):
    """Indicadores por zona del sistema-mundo (periferia/semiperiferia/centro) en los años dados."""
    # Fix country zone assignment at 2019 (or track dynamic zone at each year)
    ly19 = np.nanmean(o["ly"][:, :, 69], axis=0)
    z19 = zone_of_1d(ly19)
    out = {}
    VARS = ["ly", "E", "ne", "psi", "asab", "debt", "crisis", "demo", "coup", "hazard",
            "res_share", "reserves", "fshare", "ims", "drain", "rent_in", "dclim", "natcap", "co2"] + CHANNELS
    for ti in year_indices:
        yr = str(1950 + ti)
        pop_t = Wx.pop[:, ti]
        out[yr] = {}
        for zi, zname in ((0, "periferia"), (1, "semiperiferia"), (2, "centro")):
            mask = (z19 == zi)
            row = {"n_countries": int(mask.sum()), "pop_m": float(np.nansum(pop_t[mask]))}
            y_r = np.exp(o["ly"][:, mask, ti])
            p_m = pop_t[mask]
            ok = np.isfinite(y_r)
            gdp_w = np.nansum(np.where(ok, y_r * p_m[None], 0.0), axis=1) / np.maximum(
                np.nansum(np.where(ok, p_m[None], 0.0), axis=1), 1e-9)
            row["gdp_pc_world"] = [float(np.nanpercentile(gdp_w, q)) for q in (10, 50, 90)]
            for k in VARS:
                arr = o[k][:, mask, ti]
                if k == "co2":
                    v = np.nansum(arr, axis=1) / 1000.0
                elif k in ("psi", "debt", "dclim", "natcap", "reserves"):
                    v = np.nanmedian(arr, axis=1)
                else:
                    v = np.nanmean(arr, axis=1)
                row[k] = [float(np.nanpercentile(v, q)) for q in (10, 50, 90)]
            out[yr][zname] = row
    # Also compute mobility between 2026 (idx 76) and 2041 (idx 91)
    R = o["ly"].shape[0]
    up, dn, stay = [], [], []
    for r in range(R):
        z26 = zone_of_1d(o["ly"][r, :, 76])
        z41 = zone_of_1d(o["ly"][r, :, 91])
        ok = np.isfinite(z26) & np.isfinite(z41)
        up.append(float(np.mean(z41[ok] > z26[ok])))
        dn.append(float(np.mean(z41[ok] < z26[ok])))
        stay.append(float(np.mean(z41[ok] == z26[ok])))
    out["mobility_2026_2041"] = dict(
        up_p50=float(np.median(up)), down_p50=float(np.median(dn)), stay_p50=float(np.median(stay)),
        up_p10_p90=[float(np.percentile(up, 10)), float(np.percentile(up, 90))],
    )
    return out


def regional_breakdown(o_hist: dict, o_2000: dict, o_2019: dict, Wx):
    """Desglose sociopolítico regional (10 regiones R10 + 4 macrobloques) en 2019->2041."""
    ridx = elites.region_index(Wx.iso3)
    groups = [(c, ridx == i, elites.R10_NAMES[c]) for i, c in enumerate(elites.R10_CODES)]
    groups += [
        ("WEST", np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.WEST_CODES]), "Occidente (NAM+EUR+OPA)"),
        ("ASIA_EM", np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.ASIA_EMERG_CODES]), "Asia Emergente (CHN+SAS+SEA)"),
        ("SOUTH", np.isin(ridx, [elites.R10_CODES.index(c) for c in elites.GLOBAL_SOUTH_CODES]), "Sur Global (LAM+MEA+SSA)"),
    ]
    SOC_VARS = ("E", "ne", "psi", "asab", "demo", "hazard")

    def _agg(o_dict, mask, var, t_sl):
        arr = o_dict[var][:, mask, t_sl]
        if var == "psi":
            v = np.nanmedian(arr, axis=1)
        else:
            v = np.nanmean(arr, axis=1)
        return {
            "p10": np.nanpercentile(v, 10, axis=0).tolist(),
            "p50": np.nanpercentile(v, 50, axis=0).tolist(),
            "p90": np.nanpercentile(v, 90, axis=0).tolist(),
        }

    res = {}
    for code, mask, label in groups:
        entry = {
            "label": label,
            "n_countries": int(mask.sum()),
            "pop_2019_m": float(np.nansum(Wx.pop[mask, 69])),
            "snap": {},
            "proj_2019_2041": {},
            "hist_1950_2019": {},
            "test_2000_2025": {},
        }
        for ti, tag in ((69, "2019"), (76, "2026"), (91, "2041")):
            row = {}
            for k in SOC_VARS:
                arr = o_2019[k][:, mask, ti]
                v = np.nanmedian(arr, axis=1) if k == "psi" else np.nanmean(arr, axis=1)
                row[k] = [float(np.nanpercentile(v, q)) for q in (10, 50, 90)]
            entry["snap"][tag] = row
        if code in ("WEST", "ASIA_EM", "SOUTH", "REF", "NAM", "EUR", "CHN", "LAM", "SSA"):
            for k in SOC_VARS:
                entry["proj_2019_2041"][k] = _agg(o_2019, mask, k, slice(69, 92))
            entry["hist_1950_2019"]["E"] = _agg(o_hist, mask, "E", slice(0, 70))
            entry["test_2000_2025"]["E"] = _agg(o_2000, mask, "E", slice(50, 76))
        res[code] = entry
    return res


def main():
    t_start = time.time()
    print("1. Construyendo mundos W41 (1950-2041) y cargando observaciones recientes...")
    W41 = build_world(horizon=HORIZON_END)
    W41_ext = build_pop_extended_world(W41)
    obs = load_recent_observations(W41)
    print("   Overlaps 2010-2019:", json.dumps(obs["overlaps"], indent=None))

    p_2019 = load_params(RES_DIR / "calibration.json")
    p_2000 = load_params(RES_DIR / "rolling/params_2000.json")
    p_2010 = load_params(RES_DIR / "rolling/params_2010.json")

    print("2. Corriendo repetición histórica (1950-2019, R=64)...")
    sim_h = Simulator(W, p_2019, R=64, seed=21)
    o_hist = sim_h.run(0, W.T)

    print("3. Corriendo prueba fuera de muestra desde 2000 (2000-2025, R=100)...")
    o_2000, bm_2000, _ = run_forecast(W41, p_2000, t0=50, t1=76, R=100, seed=11)

    print("4. Corriendo prueba fuera de muestra de 15 años desde 2010 (2010-2025, R=100)...")
    o_2010, bm_2010, _ = run_forecast(W41, p_2010, t0=60, t1=76, R=100, seed=13)

    print("5. Corriendo proyección principal desde 2019 hasta 2041 (R=100) [L1 defecto: Stress M1-M3 + Democracia V2]...")
    o_2019, bm_2019, sim_2019 = run_forecast(W41, p_2019, t0=69, t1=W41.T, R=100, seed=5)

    print("6. Corriendo sensibilidad demográfica post-2030 (2019-2041, R=64)...")
    o_2019_ext, _, _ = run_forecast(W41_ext, p_2019, t0=69, t1=W41_ext.T, R=64, seed=5)

    print("7. Corriendo ablaciones de las 7 teorías + módulos nuevos M1-M3 y Democracia V2 (2019-2041, R=48)...")
    ab_base, _, _ = run_forecast(W41, p_2019, t0=69, t1=W41.T, R=48, seed=19)
    ab_base_s = summarise_all_38(ab_base, W41, slice(69, W41.T))
    ablations_2041 = {}
    AB_KEYS = ["gdp_pc_world", "gini", "E", "ne", "psi", "asab", "debt", "crisis",
               "demo", "hazard", "res_share", "reserves", "Tg", "dclim", "natcap", "co2"]
    for name, ov in ABLATIONS.items():
        if name == "completo":
            continue
        o_ab, _, _ = run_forecast(W41, p_2019, t0=69, t1=W41.T, R=48, seed=19, overrides=ov)
        s_ab = summarise_all_38(o_ab, W41, slice(69, W41.T))
        ablations_2041[name] = {
            k: dict(
                base_2041=ab_base_s[k]["p50"][-1],
                abl_2041=s_ab[k]["p50"][-1],
                delta=s_ab[k]["p50"][-1] - ab_base_s[k]["p50"][-1],
                delta_pct=((s_ab[k]["p50"][-1] / ab_base_s[k]["p50"][-1] - 1.0) * 100.0
                           if abs(ab_base_s[k]["p50"][-1]) > 1e-9 else None),
            )
            for k in AB_KEYS
        }
    EXTRA_ABLATIONS = {
        "sin_clima_co2_M0": {"stress_model": StressConfig(climate_co2=False, climate_severe=True, food_brakes=True, heavy_tails=True)},
        "sin_clima_severo_M1": {"stress_model": StressConfig(climate_co2=True, climate_severe=False, food_brakes=True, heavy_tails=True)},
        "sin_frenos_alimentos_M2": {"stress_model": StressConfig(climate_co2=True, climate_severe=True, food_brakes=False, heavy_tails=True)},
        "sin_colas_pesadas_M3": {"stress_model": StressConfig(climate_co2=True, climate_severe=True, food_brakes=True, heavy_tails=False)},
        "sin_democracia_v2": {"democracy_model": None},
        "l1_legado_9e96f6b": {"legacy": True},
    }
    for name, skw in EXTRA_ABLATIONS.items():
        o_ab, _, _ = run_forecast(W41, p_2019, t0=69, t1=W41.T, R=48, seed=19, sim_kwargs=skw)
        s_ab = summarise_all_38(o_ab, W41, slice(69, W41.T))
        ablations_2041[name] = {
            k: dict(
                base_2041=ab_base_s[k]["p50"][-1],
                abl_2041=s_ab[k]["p50"][-1],
                delta=s_ab[k]["p50"][-1] - ab_base_s[k]["p50"][-1],
                delta_pct=((s_ab[k]["p50"][-1] / ab_base_s[k]["p50"][-1] - 1.0) * 100.0
                           if abs(ab_base_s[k]["p50"][-1]) > 1e-9 else None),
            )
            for k in AB_KEYS
        }

    print("8. Calculando tablas, métricas fuera de muestra, zonas y perfiles por país...")
    s_hist = summarise_all_38(o_hist, W, slice(0, 70))
    s_2000 = summarise_all_38(o_2000, W41, slice(50, 76))
    s_2010 = summarise_all_38(o_2010, W41, slice(60, 76))
    s_2019 = summarise_all_38(o_2019, W41, slice(69, W41.T))
    s_2019_ext = summarise_all_38(o_2019_ext, W41_ext, slice(69, W41_ext.T))

    # Compare 2041 with vs without post-2030 demographic extension
    pop_sens = {
        "pop_world_2030_m": float(np.nansum(W41.pop[:, 80])),
        "pop_world_2041_raw_m": float(np.nansum(W41.pop[:, 91])),
        "pop_world_2041_ext_m": float(np.nansum(W41_ext.pop[:, 91])),
        "metrics_2041": {
            k: dict(raw=s_2019[k]["p50"][-1], ext=s_2019_ext[k]["p50"][-1],
                    diff_pct=(s_2019_ext[k]["p50"][-1] / s_2019[k]["p50"][-1] - 1) * 100
                    if abs(s_2019[k]["p50"][-1]) > 1e-9 else 0.0)
            for k in ("gdp_pc_world", "gdp_pc_med", "gini", "debt", "demo", "hazard", "Tg", "co2", "psi", "E", "natcap")
        },
    }

    scorecards = {
        "from_2000_to_2019": score_oos(o_2000, obs, 2000, 2001, 2019),
        "from_2000_to_2025": score_oos(o_2000, obs, 2000, 2001, 2025),
        "from_2010_to_2025_15y": score_oos(o_2010, obs, 2010, 2011, 2025),
        "from_2019_to_2025_recent": score_oos(o_2019, obs, 2019, 2020, 2025),
    }

    # Global world-level accuracy 2020-2025 (proj_2019 vs actual world series!)
    ow = obs["world"]
    recent_world_check = {}
    for key, obs_key, last_yr in (
        ("gdp_pc_world", "gdp_pc_world", 2025),
        ("gdp_pc_med", "gdp_pc_med", 2025),
        ("gini", "gini", 2025),
        ("debt", "debt_med", 2024),
        ("debt_gdpw", "debt_w_gdpw", 2024),
        ("res_share", "rr_mean", 2021),
        ("demo", "demo_share", 2025),
        ("hazard", "conf_rate", 2023),
        ("Tg", "gmst", 2025),
        ("co2", "co2_gt", 2024),
        ("price", "price_idx", 2025),
    ):
        idx_obs = last_yr - 1950
        idx_sim = last_yr - 2019
        obs_19 = float(ow[obs_key][69])
        obs_last = float(ow[obs_key][idx_obs])
        p10 = float(s_2019[key]["p10"][idx_sim])
        p50 = float(s_2019[key]["p50"][idx_sim])
        p90 = float(s_2019[key]["p90"][idx_sim])
        # Also compute 2020..last_yr mean absolute error
        sim_slice = np.array(s_2019[key]["p50"][1:idx_sim + 1])
        obs_slice = np.array(ow[obs_key][70:idx_obs + 1])
        lo_slice = np.array(s_2019[key]["p10"][1:idx_sim + 1])
        hi_slice = np.array(s_2019[key]["p90"][1:idx_sim + 1])
        ok = np.isfinite(obs_slice) & np.isfinite(sim_slice)
        recent_world_check[key] = dict(
            last_year=last_yr, obs_2019=obs_19, obs_last=obs_last,
            pred_p10=p10, pred_p50=p50, pred_p90=p90,
            in_band_last=bool(p10 <= obs_last <= p90),
            in_band_share=float(np.mean((obs_slice[ok] >= lo_slice[ok]) & (obs_slice[ok] <= hi_slice[ok]))),
            mae_2020_last=float(np.mean(np.abs(sim_slice[ok] - obs_slice[ok]))),
            rel_err_last_pct=float((p50 - obs_last) / abs(obs_last) * 100.0) if abs(obs_last) > 1e-9 else None,
        )

    # Zones evolution (2019, 2026, 2031, 2036, 2041)
    zones_evol = zone_breakdown(o_2019, W41, [69, 76, 81, 86, 91])

    # 12 Country profiles (2019, 2024/2025 actual vs L1, 2026, 2031, 2036, 2041)
    iso_idx = {c: i for i, c in enumerate(W41.iso3)}
    countries_12 = {}
    for code in SHOWCASE:
        i = iso_idx[code]
        oc = obs["country"]
        # Series 2000..2041 for key variables
        c_entry = {
            "name": W41.names[i],
            "iso3": code,
            "pop_2019": float(W41.pop[i, 69]),
            "pop_2026": float(W41.pop[i, 76]),
            "pop_2041_raw": float(W41.pop[i, 91]),
            "pop_2041_ext": float(W41_ext.pop[i, 91]),
            "zone_2019": int(zone_of_1d(np.nanmean(o_2019["ly"][:, :, 69], 0))[i]),
            "obs_2019": {
                "gdp_pc": float(np.exp(oc["ly"][i, 69])),
                "debt": float(oc["debt"][i, 69]) if np.isfinite(oc["debt"][i, 69]) else None,
                "demo": float(oc["demo"][i, 69]) if np.isfinite(oc["demo"][i, 69]) else None,
                "conf": float(oc["conf"][i, 69]) if np.isfinite(oc["conf"][i, 69]) else None,
                "rr": float(oc["rr"][i, 69]) if np.isfinite(oc["rr"][i, 69]) else None,
            },
            "obs_recent": {
                "gdp_pc_2025": float(np.exp(oc["ly"][i, 75])) if np.isfinite(oc["ly"][i, 75]) else None,
                "gdp_pc_2024": float(np.exp(oc["ly"][i, 74])) if np.isfinite(oc["ly"][i, 74]) else None,
                "debt_2024": float(oc["debt"][i, 74]) if np.isfinite(oc["debt"][i, 74]) else None,
                "demo_2025": float(oc["demo"][i, 75]) if np.isfinite(oc["demo"][i, 75]) else None,
                "conf_2023": float(oc["conf"][i, 73]) if np.isfinite(oc["conf"][i, 73]) else None,
            },
            "sim": {},
        }
        for yr_i, tag in ((69, "2019"), (74, "2024"), (75, "2025"), (76, "2026"), (81, "2031"), (86, "2036"), (91, "2041")):
            c_entry["sim"][tag] = {
                "gdp_pc": [float(np.nanpercentile(np.exp(o_2019["ly"][:, i, yr_i]), q)) for q in (10, 50, 90)],
                "debt": [float(np.nanpercentile(o_2019["debt"][:, i, yr_i], q)) for q in (10, 50, 90)],
                "demo": float(np.nanmean(o_2019["demo"][:, i, yr_i])),
                "poly": float(np.nanmean(o_2019["poly"][:, i, yr_i])) if "poly" in o_2019 else None,
                "kcal": float(np.nanmedian(o_2019["kcal"][:, i, yr_i])) if "kcal" in o_2019 else None,
                "hazard": float(np.nanmean(o_2019["hazard"][:, i, yr_i])),
                "E": float(np.nanmedian(o_2019["E"][:, i, yr_i])),
                "psi": float(np.nanmedian(o_2019["psi"][:, i, yr_i])),
                "asab": float(np.nanmedian(o_2019["asab"][:, i, yr_i])),
                "dclim": float(np.nanmedian(o_2019["dclim"][:, i, yr_i])),
                "reserves": float(np.nanmedian(o_2019["reserves"][:, i, yr_i])),
                "core": float(np.nanmedian(o_2019["core"][:, i, yr_i])),
            }
        # Yearly trajectories 2000-2041 for charts
        c_entry["traj_gdp_obs_2000_2025"] = [
            float(np.exp(oc["ly"][i, t])) if np.isfinite(oc["ly"][i, t]) else None for t in range(50, 76)
        ]
        c_entry["traj_gdp_sim_2019_2041"] = {
            "p10": [float(np.nanpercentile(np.exp(o_2019["ly"][:, i, t]), 10)) for t in range(69, 92)],
            "p50": [float(np.nanpercentile(np.exp(o_2019["ly"][i, i, t]) if False else np.exp(o_2019["ly"][:, i, t]), 50)) for t in range(69, 92)],
            "p90": [float(np.nanpercentile(np.exp(o_2019["ly"][:, i, t]), 90)) for t in range(69, 92)],
        }
        countries_12[code] = c_entry

    # Parameter stability across rolling cutoffs (1970, 1980, 1990, 2000, 2010, 2019)
    cutoff_params = {}
    for yr in (1970, 1980, 1990, 2000, 2010):
        d = json.loads((RES_DIR / f"rolling/params_{yr}.json").read_text(encoding="utf-8"))
        cutoff_params[str(yr)] = d["params"]
    cal19 = json.loads((RES_DIR / "calibration.json").read_text(encoding="utf-8"))
    cutoff_params["2019"] = cal19["params"]

    # Official 49-origin rolling validation (from validation.json)
    val_49 = json.loads((RES_DIR / "validation.json").read_text(encoding="utf-8"))

    # Save JSON cleanly (converting nan -> None)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return clean(obj.tolist())
        if isinstance(obj, (np.floating, float)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    payload = clean({
        "meta": {
            "report_date": "2026-11",
            "n_countries": W41.N,
            "last_panel_year": 2019,
            "horizon_start": 2026,
            "horizon_end": HORIZON_END,
            "R_main": 100,
            "R_hist": 64,
            "R_ablations": 48,
            "atlas_commit": "9e96f6b+v2_stress_dem_elites",
            "default_mode": "L1 V2 (Stress M1-M3 + Democracia V2 + Élites Regionales R10)",
            "stress_config": dataclasses.asdict(StressConfig()),
            "democracy_v2": sim_2019.democracy_model.summary() if sim_2019.democracy_model is not None else None,
            "elites_v2": sim_2019.elites_model.summary() if sim_2019.elites_model is not None else None,
            "seconds": round(time.time() - t_start, 1),
        },
        "overlaps": obs["overlaps"],
        "observed_world": obs["world"],
        "series": {
            "hist_1950_2019": s_hist,
            "test_2000_2025": s_2000,
            "test_2010_2025": s_2010,
            "proj_2019_2041": s_2019,
        },
        "scorecards": scorecards,
        "recent_world_check": recent_world_check,
        "pop_sensitivity": pop_sens,
        "zones": zones_evol,
        "regions_r10": regional_breakdown(o_hist, o_2000, o_2019, W41),
        "countries": countries_12,
        "ablations_2041": ablations_2041,
        "behavior_2019": bm_2019.summary(),
        "calibration_2019": {"fit": cal19["fit"], "etapa1": cal19["etapa1"], "libres": cal19["libres"]},
        "cutoff_params": cutoff_params,
        "validation_49_origins": val_49,
    })

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo en {time.time() - t_start:.1f} s -> {OUT_JSON}")

    # Print executive summary to stdout so we can analyze EVERY finding carefully!
    print("\n=== RESUMEN POST-PANEL 2019 -> 2025 (MUNDO) ===")
    for k, v in payload["recent_world_check"].items():
        print(f"  {k:16s} ({v['last_year']}): obs19={v['obs_2019']:.4g} | obs={v['obs_last']:.4g} | "
              f"L1=[{v['pred_p10']:.4g}, {v['pred_p50']:.4g}, {v['pred_p90']:.4g}] | "
              f"err={v['rel_err_last_pct']:+.1f}% | banda={v['in_band_share']*100:.0f}%")

    print("\n=== RESUMEN PROYECCION 15 ANOS (2026 -> 2041, MEDIANA [P10, P90]) ===")
    for k in ["gdp_pc_world", "gdp_pc_med", "gini", "core", "E", "ne", "psi", "asab",
              "debt", "debt_gdpw", "crisis", "demo", "coup", "hazard", "conflict",
              "res_share", "reserves", "fshare", "ims", "price",
              "Tg", "temp", "dclim", "natcap", "co2", "drain", "rent_in", "eco_ue",
              "k", "r", "m", "x", "f", "w"]:
        s = payload["series"]["proj_2019_2041"][k]
        # indices relative to 2019: 2019=0, 2026=7, 2031=12, 2036=17, 2041=22
        print(f"  {k:14s}: 2019={s['p50'][0]:.4g} | 2026={s['p50'][7]:.4g} | "
              f"2041={s['p50'][22]:.4g} [{s['p10'][22]:.4g}, {s['p90'][22]:.4g}]")


if __name__ == "__main__":
    main()
