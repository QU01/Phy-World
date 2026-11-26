"""Estudio completo de la capa L2 (F12: eventos de violencia organizada y desplazamiento a escala mundial):
  1. Registro histórico 1989-01 a 2026-09 (191 países, 64.859 celdas 0,5°, 428.892 eventos, 2.059 díadas).
  2. Dos pruebas ciegas retrospectivas sin fuga (P.truncated) en las 5 regiones de UCDP:
       - Prueba A (36 meses, 2023-01 -> 2025-12): agentes vs hurdle vs last12 vs zero, país y celda + desplazamiento.
       - Prueba B (12 meses recientes, 2025-01 -> 2025-12): agentes vs hurdle vs last12 en el último año cerrado.
  3. Proyección en vivo a 36 meses (2026-10 -> 2029-09, R=64) para las 5 regiones, 191 países, 64.859 celdas,
     desplazamiento (refugiados y desplazados internos) y puente L2 -> L1; además, prueba de estrés de deriva
     a 60 meses (2026-10 -> 2031-09, R=32).
  4. Auditoría completa de todos los parámetros estimados en las 5 regiones (11 rasgos de díada, 3 de nacimiento,
     4 de huella espacial, 8 de desplazamiento) y 4 ablaciones estructurales a 36 meses.

Uso:
    python reports/l2_conflict/run_study.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.simplefilter("ignore")

import numpy as np

from world4.core.l2 import agents, baseline, displacement, forecast, panel, regions
from world4.validate import metrics

HERE = Path(__file__).parent
OUT_JSON = HERE / "study_results.json"

REGIONS = list(regions.WORLD)  # ('africa', 'americas', 'asia', 'europe', 'mideast')
REG_NAMES = {
    "africa": "África",
    "americas": "Américas",
    "asia": "Asia y Oceanía",
    "europe": "Europa",
    "mideast": "Oriente Medio",
}
SHOWCASE_ISO = ["UKR", "SDN", "ISR", "MMR", "MEX", "NGA", "BFA", "MLI", "COD", "SOM", "ETH", "SYR"]


def q3(arr_rt: np.ndarray) -> dict:
    """(R, ...) -> dict con p10, p50, p90 y mean."""
    return {
        "p10": np.percentile(arr_rt, 10, axis=0).tolist(),
        "p50": np.percentile(arr_rt, 50, axis=0).tolist(),
        "p90": np.percentile(arr_rt, 90, axis=0).tolist(),
        "mean": np.mean(arr_rt, axis=0).tolist(),
    }


def score_step(ens: np.ndarray, obs: np.ndarray, steps: tuple[int, ...], with_brier25: bool = True) -> dict:
    """ens (R, U, H), obs (U, H) -> métricas por paso s."""
    out = {}
    for s in steps:
        if s > obs.shape[1] or s > ens.shape[2]:
            continue
        e, o = ens[:, :, s - 1], obs[:, s - 1]
        lo, hi = np.percentile(e, 10, axis=0), np.percentile(e, 90, axis=0)
        row = {
            "n": int(len(o)),
            "crps_log": float(np.mean(metrics.crps_ensemble(np.log1p(e), np.log1p(o)))),
            "crps": float(np.mean(metrics.crps_ensemble(e, o))),
            "brier1": float(np.mean((np.mean(e >= 1, axis=0) - (o >= 1)) ** 2)),
            "cover_80_pos": float(np.mean((o[o > 0] >= lo[o > 0]) & (o[o > 0] <= hi[o > 0]))) if (o > 0).any() else None,
            "cover_80_all": float(np.mean((o >= lo) & (o <= hi))),
        }
        if with_brier25:
            row["brier25"] = float(np.mean((np.mean(e >= 25, axis=0) - (o >= 25)) ** 2))
        out[str(s)] = row
    return out


def extract_panel_history(P: panel.Panel) -> dict:
    """Series históricas anuales (1989-2025 + 2026 parcial) y mensuales recientes (2018-2026) por región."""
    d_all = P.deaths("country", "all")              # (K, T)
    d_sb = P.deaths("country", "state_based")
    d_ns = P.deaths("country", "non_state")
    d_os = P.deaths("country", "one_sided")
    d_civ = P.to_countries(P.site_deaths("civ"))
    n_dy = P.dyad_series("events")                  # (D, T)
    first_dy = np.where((n_dy > 0).any(1), (n_dy > 0).argmax(1), P.T + 10)

    # Annual 1989..2025 (full years) + 2026 (Jan-Sep provisional)
    yrs = list(range(1989, 2027))
    ann = {k: [] for k in ("deaths", "state_based", "non_state", "one_sided", "civ",
                           "active_countries_1", "active_countries_25", "active_dyads", "births",
                           "ref_stock", "idp_stock")}
    for y in yrs:
        m0 = P.month(y, 1)
        m1 = min(P.month(y, 12) + 1, P.T)
        sl = slice(m0, m1)
        ann["deaths"].append(float(d_all[:, sl].sum()))
        ann["state_based"].append(float(d_sb[:, sl].sum()))
        ann["non_state"].append(float(d_ns[:, sl].sum()))
        ann["one_sided"].append(float(d_os[:, sl].sum()))
        ann["civ"].append(float(d_civ[:, sl].sum()))
        yk = d_all[:, sl].sum(1)
        ann["active_countries_1"].append(int((yk >= 1).sum()))
        ann["active_countries_25"].append(int((yk >= 25).sum()))
        ann["active_dyads"].append(int((n_dy[:, sl].sum(1) > 0).sum()))
        ann["births"].append(int(((first_dy >= m0) & (first_dy < m1)).sum()) if y >= 1991 else 0)
        j = y - int(P.years[0])
        if 0 <= j < len(P.years):
            ann["ref_stock"].append(float(np.nansum(P.ref_stock[:, :, j])))
            ann["idp_stock"].append(float(np.nansum(P.idp_stock[:, j])))
        else:
            ann["ref_stock"].append(None)
            ann["idp_stock"].append(None)

    # Monthly 2018-01 .. 2026-09 (105 months)
    t_m0 = P.month(2018, 1)
    monthly = {
        "labels": [P.label(t) for t in range(t_m0, P.T)],
        "deaths": d_all[:, t_m0:].sum(0).tolist(),
        "state_based": d_sb[:, t_m0:].sum(0).tolist(),
        "non_state": d_ns[:, t_m0:].sum(0).tolist(),
        "one_sided": d_os[:, t_m0:].sum(0).tolist(),
        "active_dyads": (n_dy[:, t_m0:] > 0).sum(0).tolist(),
        "active_countries_1": (d_all[:, t_m0:] >= 1).sum(0).tolist(),
    }

    return {
        "summary": {
            "K": P.K, "C": P.C, "S": P.S, "D": P.D, "actors": int(len(P.actor_kind)),
            "events": int(len(P.ev["t0"])), "deaths_total": float(P.ev["best"].sum()),
            "deaths_2025": float(d_all[:, P.month(2025, 1):P.month(2025, 12) + 1].sum()),
            "deaths_2026_ytd": float(d_all[:, P.month(2026, 1):P.T].sum()),
            "last_final": P.label(P.last_final_month), "last_month": P.label(P.T - 1),
        },
        "years": yrs,
        "annual": ann,
        "monthly_2018_2026": monthly,
    }


def run_region_study(reg: str) -> dict:
    t_r = time.time()
    print(f"\n=== Región: {reg} ===", flush=True)
    P = panel.load(reg)
    hist = extract_panel_history(P)

    # ── 1. Backtest A: 36 meses desde 2023-01 hasta 2025-12 (sin fuga, R=48)
    print(f"  [1/5] Backtest 36m (2023-01 -> 2025-12, R=48)...", flush=True)
    t0_23 = P.month(2023, 1)
    H23 = 36
    Q23 = P.truncated(t0_23)
    y23_c = P.deaths("country")[:, t0_23:t0_23 + H23]
    y23_g = P.deaths("cell")[:, t0_23:t0_23 + H23]

    th23 = displacement.fit_cached(Q23, t0_23, maxiter=140)
    f23 = forecast.run(Q23, t0_23, H23, members=48, seed=23, displacement=True, disp_params=th23)
    ag23_c, ag23_g = f23.deaths("country"), f23.deaths("cell")
    hm23_c = baseline.fit(Q23, "country", t0_23)
    hu23_c = baseline.forecast(Q23, hm23_c, t0_23, H23, members=48, seed=23)
    l12_23_c = baseline.last12(Q23, "country", t0_23, H23)
    z23_c = baseline.zero(Q23, "country", H23)
    # Cell-level baselines for h=1,3,6,12
    hm23_g = baseline.fit(Q23, "cell", t0_23, steps=(1, 3, 6, 12))
    hu23_g = baseline.forecast(Q23, hm23_g, t0_23, 12, members=48, seed=23)
    l12_23_g = baseline.last12(Q23, "cell", t0_23, 12)

    bt23_scores = {
        "country": {
            "agents": score_step(ag23_c, y23_c, (1, 3, 6, 12, 24, 36)),
            "hurdle": score_step(hu23_c, y23_c, (1, 3, 6, 12, 24, 36)),
            "last12": score_step(l12_23_c, y23_c, (1, 3, 6, 12, 24, 36)),
            "zero": score_step(z23_c, y23_c, (1, 3, 6, 12, 24, 36)),
        },
        "cell": {
            "agents": score_step(ag23_g[:, :, :12], y23_g[:, :12], (1, 3, 6, 12), with_brier25=False),
            "hurdle": score_step(hu23_g, y23_g[:, :12], (1, 3, 6, 12), with_brier25=False),
            "last12": score_step(l12_23_g, y23_g[:, :12], (1, 3, 6, 12), with_brier25=False),
        },
        "monthly_totals": {
            "labels": [P.label(t0_23 + h) for h in range(H23)],
            "obs": y23_c.sum(0).tolist(),
            "agents": q3(ag23_c.sum(1)),
            "hurdle": q3(hu23_c.sum(1)),
            "last12": float(l12_23_c.mean(0).sum(0)[0]),
        },
        "cumulative_36m": {
            "obs": float(y23_c.sum()),
            "agents": q3(ag23_c.sum((1, 2))),
            "hurdle": q3(hu23_c.sum((1, 2))),
            "last12": float(l12_23_c.mean(0).sum()),
        },
    }
    # Displacement 2023->2025 evaluation
    j22 = 2022 - int(P.years[0])
    j25 = 2025 - int(P.years[0])
    obs_ref25 = np.nan_to_num(P.ref_stock[:, :, j25].sum(1))
    obs_ref22 = np.nan_to_num(P.ref_stock[:, :, j22].sum(1))
    obs_idp25 = np.nan_to_num(P.idp_stock[:, j25])
    obs_idp22 = np.nan_to_num(P.idp_stock[:, j22])
    sim_ref25 = f23.displaced["ref"][:, :, -1]
    sim_idp25 = f23.displaced["idp"][:, :, -1]
    bt23_disp = {
        "ref_2022_obs": float(obs_ref22.sum()),
        "ref_2025_obs": float(obs_ref25.sum()),
        "ref_2025_sim": q3(sim_ref25.sum(1)),
        "idp_2022_obs": float(obs_idp22.sum()),
        "idp_2025_obs": float(obs_idp25.sum()),
        "idp_2025_sim": q3(sim_idp25.sum(1)),
        "ref_mae_asinh_sim": float(np.mean(np.abs(np.arcsinh(np.median(sim_ref25, 0) / 1e4) - np.arcsinh(obs_ref25 / 1e4)))),
        "ref_mae_asinh_pers": float(np.mean(np.abs(np.arcsinh(obs_ref22 / 1e4) - np.arcsinh(obs_ref25 / 1e4)))),
    }
    bt23_scores["displacement"] = bt23_disp

    # ── 2. Backtest B: 12 meses recientes (2025-01 -> 2025-12, R=48)
    print(f"  [2/5] Backtest 12m reciente (2025-01 -> 2025-12, R=48)...", flush=True)
    t0_25 = P.month(2025, 1)
    H25 = 12
    Q25 = P.truncated(t0_25)
    y25_c = P.deaths("country")[:, t0_25:t0_25 + H25]
    f25 = forecast.run(Q25, t0_25, H25, members=48, seed=25, displacement=False, foot=f23.model.foot)
    ag25_c = f25.deaths("country")
    hm25_c = baseline.fit(Q25, "country", t0_25, steps=(1, 3, 6, 12))
    hu25_c = baseline.forecast(Q25, hm25_c, t0_25, H25, members=48, seed=25)
    l12_25_c = baseline.last12(Q25, "country", t0_25, H25)

    bt25_scores = {
        "country": {
            "agents": score_step(ag25_c, y25_c, (1, 3, 6, 12)),
            "hurdle": score_step(hu25_c, y25_c, (1, 3, 6, 12)),
            "last12": score_step(l12_25_c, y25_c, (1, 3, 6, 12)),
        },
        "total_2025": {
            "obs": float(y25_c.sum()),
            "agents": q3(ag25_c.sum((1, 2))),
            "hurdle": q3(hu25_c.sum((1, 2))),
            "last12": float(l12_25_c.mean(0).sum()),
        },
        "by_country_2025": {
            iso: {
                "obs": float(y25_c[k].sum()),
                "agents_p10": float(np.percentile(ag25_c[:, k].sum(1), 10)),
                "agents_p50": float(np.percentile(ag25_c[:, k].sum(1), 50)),
                "agents_p90": float(np.percentile(ag25_c[:, k].sum(1), 90)),
                "hurdle_p50": float(np.percentile(hu25_c[:, k].sum(1), 50)),
                "last12": float(l12_25_c[:, k].mean(0).sum()),
            }
            for k, iso in enumerate(P.iso3)
            if y25_c[k].sum() >= 25 or np.percentile(ag25_c[:, k].sum(1), 50) >= 25 or iso in SHOWCASE_ISO
        },
    }

    # ── 3. Proyección principal en vivo: 36 meses (2026-10 -> 2029-09, R=64) con desplazamiento
    print(f"  [3/5] Proyección principal 36m (2026-10 -> 2029-09, R=64) + desplazamiento...", flush=True)
    t0_live = P.T
    H_LIVE = 36
    th_live = displacement.fit_cached(P, t0_live, maxiter=160)
    fl = forecast.run(P, t0_live, H_LIVE, members=64, seed=42, displacement=True, disp_params=th_live)
    agl_c = fl.deaths("country")               # (R, K, 36)
    agl_g = fl.deaths("cell")                  # (R, C, 36)
    agl_civ = P.to_countries(fl.run.site_civ)  # (R, K, 36)
    ct = fl.run.country_type                   # (R, K, 36, 3)

    hml_c = baseline.fit(P, "country", t0_live)
    hul_c = baseline.forecast(P, hml_c, t0_live, H_LIVE, members=64, seed=42)
    l12l_c = baseline.last12(P, "country", t0_live, H_LIVE)

    # Active cells per month (R, H)
    act_cells = (agl_g >= 1).sum(1)
    act_c1 = (agl_c >= 1).sum(1)
    act_c25 = (agl_c >= 25).sum(1)
    ev_tot = fl.run.dyad_events.sum(1)         # (R, H)
    ad_tot = fl.run.active_dyads.sum(1)        # (R, H)
    br_tot = fl.run.births.sum(1)              # (R, H)

    # Top 25 0.5° hotspot cells in the next 12 months for this region
    cell_12m_p50 = np.percentile(agl_g[:, :, :12].sum(2), 50, axis=0)
    cell_12m_mean = np.mean(agl_g[:, :, :12].sum(2), axis=0)
    cell_12m_pany = np.mean((agl_g[:, :, :12].sum(2) >= 1), axis=0)
    top_idx = np.argsort(cell_12m_mean)[::-1][:25]
    hotspots = []
    for ci in top_idx:
        if cell_12m_mean[ci] < 1.0:
            continue
        ck = int(P.cell_country[ci])
        hotspots.append({
            "gid": int(P.gid[ci]),
            "lon": round(float(P.lon[ci]), 2),
            "lat": round(float(P.lat[ci]), 2),
            "iso3": P.iso3[ck] if ck >= 0 else "?",
            "region": reg,
            "mean_12m": float(cell_12m_mean[ci]),
            "p50_12m": float(cell_12m_p50[ci]),
            "p_any_12m": float(cell_12m_pany[ci]),
        })

    # Land mask sample (every ~1.5° for lightweight crisp SVG world map!)
    land_pts = []
    for ci in range(P.C):
        if int(round(P.lon[ci] * 2)) % 3 == 0 and int(round(P.lat[ci] * 2)) % 3 == 0:
            land_pts.append([round(float(P.lon[ci]), 1), round(float(P.lat[ci]), 1)])

    # L2 -> L1 bridge (annual conflict >=25 state-based deaths in 2027, 2028)
    l2_l1_2027 = forecast.conflict_members(fl, 2027).mean(0)
    l2_l1_2028 = forecast.conflict_members(fl, 2028).mean(0)

    # Country-level 12m & 36m forecasts + displacement + L2->L1 bridge
    d_hist_25 = P.deaths("country")[:, P.month(2025, 1):P.month(2025, 12) + 1].sum(1)
    d_hist_26 = P.deaths("country")[:, P.month(2026, 1):P.T].sum(1)
    j25_idx = 2025 - int(P.years[0])
    countries = {}
    for k, iso in enumerate(P.iso3):
        m12 = float(np.mean(agl_c[:, k, :12].sum(1)))
        if m12 >= 15.0 or d_hist_25[k] >= 25.0 or iso in SHOWCASE_ISO:
            asylum = []
            rs_k = np.nan_to_num(P.ref_stock[k, :, j25_idx])
            rs_tot = float(rs_k.sum())
            if rs_tot > 0:
                for ai in np.argsort(-rs_k)[:3]:
                    if rs_k[ai] > 0:
                        asylum.append([str(P.asylum_iso3[ai]), round(float(rs_k[ai] / rs_tot), 3)])
            countries[iso] = {
                "region": reg,
                "deaths_2025": float(d_hist_25[k]),
                "deaths_2026_ytd": float(d_hist_26[k]),
                "agents_12m": q3(agl_c[:, k, :12].sum(1)),
                "agents_36m": q3(agl_c[:, k, :36].sum(1)),
                "hurdle_12m": q3(hul_c[:, k, :12].sum(1)),
                "hurdle_36m": q3(hul_c[:, k, :36].sum(1)),
                "last12_12m": float(l12l_c[:, k, :12].mean(0).sum()),
                "p_any_12m": float(np.mean(agl_c[:, k, :12].sum(1) >= 1)),
                "p_l1_2027": float(l2_l1_2027[k]) if np.isfinite(l2_l1_2027[k]) else None,
                "p_l1_2028": float(l2_l1_2028[k]) if np.isfinite(l2_l1_2028[k]) else None,
                "civ_share_12m": float(agl_civ[:, k, :12].sum() / max(agl_c[:, k, :12].sum(), 1.0)),
                "type_share_12m": [float(ct[:, k, :12, ti].sum() / max(ct[:, k, :12].sum(), 1.0)) for ti in range(3)],
                "ref_2025": float(np.nansum(P.ref_stock[k, :, j25_idx])),
                "idp_2025": float(np.nan_to_num(P.idp_stock[k, j25_idx])),
                "ref_2029": q3(fl.displaced["ref"][:, k, -1]),
                "idp_2029": q3(fl.displaced["idp"][:, k, -1]),
                "asylum_top": asylum,
                "monthly_obs_2023_2026": P.deaths("country")[k, P.month(2023, 1):P.T].tolist(),
                "monthly_agents_36m": q3(agl_c[:, k, :]),
                "monthly_hurdle_36m": q3(hul_c[:, k, :]),
            }

    # ── 4. Prueba de estrés de deriva a 60 meses (5 años: 2026-10 -> 2031-09, R=32)
    print(f"  [4/5] Estrés de deriva a 60 meses (2026-10 -> 2031-09, R=32)...", flush=True)
    r60 = agents.simulate(P, fl.model, 60, members=32, seed=99, births=True)
    d60_c = P.to_countries(r60.site_deaths)
    drift_60m = {
        "labels": [P.label(t0_live + h) for h in range(60)],
        "deaths_monthly": q3(d60_c.sum(1)),
        "active_dyads": q3(r60.active_dyads.sum(1)),
        "births_monthly": q3(r60.births.sum(1)),
        "events_monthly": q3(r60.dyad_events.sum(1)),
        "annual_deaths": {
            f"Y{yr_i + 1}": q3(d60_c[:, :, yr_i * 12:(yr_i + 1) * 12].sum((1, 2)))
            for yr_i in range(5)
        },
    }

    # ── 5. Cuatro ablaciones estructurales a 36 meses (R=32, misma semilla!)
    print(f"  [5/5] Ablaciones estructurales (36m, R=32)...", flush=True)
    r_base = agents.simulate(P, fl.model, 36, members=32, seed=77, births=True)
    r_nobirth = agents.simulate(P, fl.model, 36, members=32, seed=77, births=False)
    M_nodiff = agents.fit(P, t0_live, seed=42, foot={**fl.model.foot, "diffuse": 0.0})
    r_nodiff = agents.simulate(P, M_nodiff, 36, members=32, seed=77, births=True)
    M_nofloor = agents.fit(P, t0_live, seed=42, foot={**fl.model.foot, "floor": 0.0})
    r_nofloor = agents.simulate(P, M_nofloor, 36, members=32, seed=77, births=True)
    M_recent = agents.fit(P, t0_live, seed=42, foot=fl.model.foot, mark_halflife=24.0)
    r_recent = agents.simulate(P, M_recent, 36, members=32, seed=77, births=True)

    def ab_stat(run_obj):
        dc = P.to_countries(run_obj.site_deaths)
        dg = P.to_cells(run_obj.site_deaths)
        return {
            "deaths_36m_p50": float(np.median(dc.sum((1, 2)))),
            "deaths_y3_p50": float(np.median(dc[:, :, 24:36].sum((1, 2)))),
            "active_cells_y3": float(np.median((dg[:, :, 24:36].sum(2) >= 1).sum(1))),
            "active_dyads_y3": float(np.median(run_obj.active_dyads[:, :, 24:36].mean(2).sum(1))),
            "births_36m": float(np.median(run_obj.births.sum((1, 2)))),
        }

    ablations = {
        "completo": ab_stat(r_base),
        "sin_nacimientos": ab_stat(r_nobirth),
        "sin_difusion": ab_stat(r_nodiff),
        "sin_suelo_espacial": ab_stat(r_nofloor),
        "letalidad_reciente_24m": ab_stat(r_recent),
    }

    # All estimated parameters for this region
    M = fl.model
    feat_names = ["const"] + list(M.names)
    params_audit = {
        "D0_alive": int(M.D0),
        "n_cap": float(M.n_cap),
        "resid_sd": float(np.std(M.resid)),
        "b_act": {n: round(float(v), 4) for n, v in zip(feat_names, M.b_act) if n != "l1"},
        "b_int": {n: round(float(v), 4) for n, v in zip(feat_names, M.b_int) if n != "l1"},
        "b_birth": {n: round(float(v), 4) for n, v in zip(["const", "log_ev12", "log_pop"], M.b_birth)},
        "foot": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in M.foot.items()},
        "displacement": {k: float(v) for k, v in th_live.items() if k in displacement.NAMES},
        "disp_mult_range": [float(np.min(th_live["mult"])), float(np.median(th_live["mult"])), float(np.max(th_live["mult"]))],
    }

    live_36m = {
        "labels": fl.months(),
        "deaths_monthly": q3(agl_c.sum(1)),
        "hurdle_monthly": q3(hul_c.sum(1)),
        "last12_monthly": float(l12l_c.mean(0).sum(0)[0]),
        "state_based_monthly": q3(ct[:, :, :, 0].sum(1)),
        "non_state_monthly": q3(ct[:, :, :, 1].sum(1)),
        "one_sided_monthly": q3(ct[:, :, :, 2].sum(1)),
        "civ_monthly": q3(agl_civ.sum(1)),
        "events_monthly": q3(ev_tot),
        "active_dyads_monthly": q3(ad_tot),
        "births_monthly": q3(br_tot),
        "active_countries_1": q3(act_c1),
        "active_countries_25": q3(act_c25),
        "active_cells": q3(act_cells),
        "ref_monthly": q3(fl.displaced["ref"].sum(1)),
        "idp_monthly": q3(fl.displaced["idp"].sum(1)),
        "totals_12m": {
            "agents": q3(agl_c[:, :, :12].sum((1, 2))),
            "hurdle": q3(hul_c[:, :, :12].sum((1, 2))),
            "last12": float(l12l_c[:, :, :12].mean(0).sum()),
        },
        "totals_36m": {
            "agents": q3(agl_c.sum((1, 2))),
            "hurdle": q3(hul_c.sum((1, 2))),
            "last12": float(l12l_c.mean(0).sum()),
        },
    }

    print(f"  -> {reg} listo en {time.time() - t_r:.1f}s | 2025 real={bt25_scores['total_2025']['obs']:,.0f}, "
          f"ag25={bt25_scores['total_2025']['agents']['p50']:,.0f}, "
          f"live Y1={live_36m['totals_12m']['agents']['p50']:,.0f}, "
          f"Y3={drift_60m['annual_deaths']['Y3']['p50']:,.0f}, "
          f"Y5={drift_60m['annual_deaths']['Y5']['p50']:,.0f}", flush=True)

    return {
        "region": reg,
        "name": REG_NAMES[reg],
        "history": hist,
        "backtest_2023_36m": bt23_scores,
        "backtest_2025_12m": bt25_scores,
        "live_36m": live_36m,
        "drift_60m": drift_60m,
        "ablations_36m": ablations,
        "params": params_audit,
        "hotspots": hotspots,
        "countries": countries,
        "land_pts": land_pts,
    }


def main():
    t0 = time.time()
    reg_out = {}
    for reg in REGIONS:
        reg_out[reg] = run_region_study(reg)

    # Aggregate world-level views cleanly (noting independence across regions!)
    yrs = reg_out["mideast"]["history"]["years"]
    world_ann = {k: [] for k in ("deaths", "state_based", "non_state", "one_sided", "civ",
                                 "active_countries_1", "active_countries_25", "active_dyads", "births",
                                 "ref_stock", "idp_stock")}
    for i in range(len(yrs)):
        for k in world_ann:
            vals = [reg_out[r]["history"]["annual"][k][i] for r in REGIONS]
            world_ann[k].append(None if any(v is None for v in vals) else float(sum(vals)))

    # Pooled country & cell backtest scores across all 5 regions (weighted by unit count N!)
    def pool_bt(bt_key, level, steps):
        out = {}
        for m in ("agents", "hurdle", "last12", "zero"):
            if m not in reg_out["mideast"][bt_key][level]:
                continue
            out[m] = {}
            for s in steps:
                sk = str(s)
                ns = [reg_out[r][bt_key][level][m][sk]["n"] for r in REGIONS]
                ntot = sum(ns)
                row = {"n": ntot}
                for met in ("crps_log", "crps", "brier1", "brier25", "cover_80_all"):
                    if met not in reg_out["mideast"][bt_key][level][m][sk]:
                        continue
                    row[met] = float(sum(reg_out[r][bt_key][level][m][sk][met] * n for r, n in zip(REGIONS, ns)) / ntot)
                out[m][sk] = row
        return out

    # Collect top 80 global hotspots and world land points
    all_hot = sorted([h for r in REGIONS for h in reg_out[r]["hotspots"]], key=lambda x: -x["mean_12m"])[:80]
    land_world = [pt for r in REGIONS for pt in reg_out[r]["land_pts"]]
    all_countries = {}
    for r in REGIONS:
        all_countries.update(reg_out[r]["countries"])
    # Remove duplicated land_pts inside each region to keep JSON compact
    for r in REGIONS:
        del reg_out[r]["land_pts"]

    # Official W13 20-origin rolling validation (2015-2024) from docs/F12-informe.md §12
    w13_official = {
        "period": "2015-2024 (20 orígenes por región · 100 corridas)",
        "world_country": {
            "1": {"n": 3820, "agents": 0.1550, "hurdle": 0.1579, "last12": 0.1541, "zero": 0.759},
            "3": {"n": 3820, "agents": 0.1814, "hurdle": 0.1838, "last12": 0.1726, "zero": 0.761},
            "6": {"n": 3820, "agents": 0.1933, "hurdle": 0.1923, "last12": 0.1826, "zero": 0.748},
            "12": {"n": 3820, "agents": 0.2226, "hurdle": 0.2124, "last12": 0.2041, "zero": 0.755},
            "24": {"n": 3629, "agents": 0.3113, "hurdle": 0.2525, "last12": 0.2384, "zero": 0.763},
            "36": {"n": 3247, "agents": 0.4452, "hurdle": 0.2863, "last12": 0.2800, "zero": 0.770},
        },
        "world_cell": {
            "1": {"n": 1297180, "agents": 0.0094, "hurdle": 0.0088, "last12": 0.0089, "zero": 0.0134},
            "3": {"n": 1297180, "agents": 0.0103, "hurdle": 0.0095, "last12": 0.0094, "zero": 0.0133},
            "6": {"n": 1297180, "agents": 0.0111, "hurdle": 0.0095, "last12": 0.0095, "zero": 0.0129},
            "12": {"n": 1297180, "agents": 0.0132, "hurdle": 0.0102, "last12": 0.0103, "zero": 0.0130},
        },
        "by_region": {
            "africa": {"c1": [0.249, 0.252], "c12": [0.373, 0.334], "c36": [0.931, 0.414], "g1": [0.0219, 0.0211], "g12": [0.0321, 0.0239]},
            "americas": {"c1": [0.138, 0.134], "c12": [0.177, 0.156], "c36": [0.255, 0.217], "g1": [0.0051, 0.0045], "g12": [0.0057, 0.0049]},
            "asia": {"c1": [0.103, 0.099], "c12": [0.134, 0.115], "c36": [0.215, 0.174], "g1": [0.0099, 0.0096], "g12": [0.0138, 0.0107]},
            "europe": {"c1": [0.053, 0.046], "c12": [0.071, 0.077], "c36": [0.115, 0.121], "g1": [0.0017, 0.0014], "g12": [0.0046, 0.0026]},
            "mideast": {"c1": [0.318, 0.305], "c12": [0.500, 0.457], "c36": [0.816, 0.634], "g1": [0.0382, 0.0356], "g12": [0.0476, 0.0411]},
        },
        "displacement_world": {
            "ref_1y": {"n": 955, "pers": 0.084, "obs_d": 0.138, "full": 0.142, "cov80": 0.25},
            "ref_2y": {"n": 955, "pers": 0.159, "obs_d": 0.232, "full": 0.251, "cov80": 0.33},
            "ref_3y": {"n": 955, "pers": 0.210, "obs_d": 0.290, "full": 0.327, "cov80": 0.38},
            "idp_1y": {"n": 282, "pers": 0.319, "obs_d": 0.467, "full": 0.481, "cov80": 0.22},
            "idp_2y": {"n": 296, "pers": 0.546, "obs_d": 0.671, "full": 0.708, "cov80": 0.28},
            "idp_3y": {"n": 298, "pers": 0.723, "obs_d": 0.816, "full": 0.948, "cov80": 0.27},
        },
        "bridge_mideast": {"l2_jan": 0.084, "pers": 0.108, "l1_3y": 0.256, "n": 120},
    }

    payload = {
        "meta": {
            "report_date": "2026-11",
            "regions": REGIONS,
            "K_total": sum(reg_out[r]["history"]["summary"]["K"] for r in REGIONS),
            "C_total": sum(reg_out[r]["history"]["summary"]["C"] for r in REGIONS),
            "D_total": sum(reg_out[r]["history"]["summary"]["D"] for r in REGIONS),
            "actors_total": sum(reg_out[r]["history"]["summary"]["actors"] for r in REGIONS),
            "events_total": sum(reg_out[r]["history"]["summary"]["events"] for r in REGIONS),
            "deaths_total": sum(reg_out[r]["history"]["summary"]["deaths_total"] for r in REGIONS),
            "last_final": "2025-12",
            "last_candidate": "2026-09",
            "live_horizon_months": 36,
            "stress_horizon_months": 60,
            "seconds": round(time.time() - t0, 1),
        },
        "world_annual": {"years": yrs, **world_ann},
        "world_backtest_2023": {
            "country": pool_bt("backtest_2023_36m", "country", (1, 3, 6, 12, 24, 36)),
            "cell": pool_bt("backtest_2023_36m", "cell", (1, 3, 6, 12)),
        },
        "world_backtest_2025": {
            "country": pool_bt("backtest_2025_12m", "country", (1, 3, 6, 12)),
        },
        "w13_official": w13_official,
        "regions": reg_out,
        "countries": all_countries,
        "top_hotspots": all_hot,
        "land_world": land_world,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nEstudio L2 completado en {time.time() - t0:.1f} s -> {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
