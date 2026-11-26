"""Estudio especulativo de largo plazo (2019 -> 2100) sobre Phy-World L1 V2.

Compara 6 bifurcaciones del siglo XXI (R=64 mundos por escenario, números aleatorios comunes,
demografía extendida estilo ONU WPP 2024 mediana con pico ~9,54 B en 2080):

  S0_legado     : L1 legado 9e96f6b (referencia sin M0-M3, sin Democracia V2, sin Élites R10)
  S5_adapt_verde: Transición Verde + Adaptación Tecnológica (Net-Zero <2 °C, sustitución
                  renovable del cuello Hubbert, adaptación térmica 40 % y techo agrícola 1,65x)
  S2_netzero    : Transición Verde Pura (Net-Zero ~2060, +1,93 °C, con Hubbert y M1-M3 intactos)
  S1_endogeno   : L1 V2 Endógeno (defecto actual: bucle CO2 <-> PIB, estabiliza en ~+2,8 °C)
  S3_alta_4c    : Alta Trayectoria Climática (+4,25 °C en 2100 por feedbacks carbono/fósiles)
  S4_policrisis : Policrisis Compuesta (+4,25 °C + choques globales x2 + techo agrícola 1,25x)

Salida: reports/l1_atlas/study_2100.json
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from world4.core.l1_atlas import elites
from world4.core.l1_atlas.behavior import BehaviorModel, BehaviorPolicy
from world4.core.l1_atlas.calibrate import LY, W
from world4.core.l1_atlas.model import Params, Simulator, build_world
from world4.core.l1_atlas.stress import StressConfig, StressModel

HERE = Path(__file__).parent
RES_DIR = Path("data_cache/atlas/results")
OUT_JSON = HERE / "study_2100.json"
HORIZON_END = 2100
SHOWCASE = ["USA", "DEU", "KOR", "CHN", "IND", "RUS", "BRA", "MEX", "EGY", "NGA", "ZAF", "ARG"]


def extend_wpp_demography_2100(W100):
    """Convergencia demográfica estilo WPP 2024 mediana (2031-2100): pico mundial ~9,54 B en 2080."""
    pop = W100.pop.copy()
    g_pop = np.log(np.maximum(pop[:, 80], 1e-6) / np.maximum(pop[:, 75], 1e-6)) / 5.0
    g_pop = np.clip(np.where(np.isfinite(g_pop), g_pop, 0.0), -0.015, 0.032)
    for t in range(81, W100.T):
        dt = t - 80
        g_t = np.where(
            g_pop > 0,
            g_pop * (0.965 ** dt) - 0.0025 * (1.0 - 0.965 ** dt),
            g_pop * (0.985 ** dt),
        )
        pop[:, t] = np.maximum(pop[:, t - 1] * np.exp(g_t), 0.05)
    return dataclasses.replace(W100, pop=pop, _locf=dict(W100._locf))


class ScenarioStressModel(StressModel):
    """StressModel con horquillas climáticas y opción de sustitución energética renovable."""

    def __init__(
        self,
        world,
        config: StressConfig,
        climate_path: str = "endog",
        clean_energy_decouple: bool = False,
    ):
        super().__init__(world, config)
        self.climate_path = climate_path
        self.clean_energy_decouple = clean_energy_decouple

    def pre_step(self, sim, t: int, obs_mode: bool):
        ov = super().pre_step(sim, t, obs_mode)
        # Sustitución energética limpia tras 2025: evita que el agotamiento Hubbert de fósiles
        # estrangule la oferta energética total (mantiene x = X/U <= 0.40 via energía renovable)
        if not obs_mode and self.clean_energy_decouple and (1950 + t >= 2026):
            sim.X = np.minimum(sim.X, 0.40 * sim.U)
        return ov

    def post_step(self, sim, rec, t, obs_mode, dclim_prev, act, orig_p, rng=None):
        super().post_step(sim, rec, t, obs_mode, dclim_prev, act, orig_p, rng=rng)
        if obs_mode or self.climate_path == "endog":
            return
        yr = 1950 + t
        tcre_ratio = np.clip(self.tcre_r / max(self.tcre_hat, 1e-6), 0.65, 1.45)
        if self.climate_path == "netzero":
            x = np.clip((yr - 2019) / 46.0, 0.0, 1.0)
            tg_target = 1.24 + 0.68 * (1.0 - (1.0 - x) ** 2.2)
            co2_target = max(37.0 * (1.0 - np.clip((yr - 2024) / 38.0, 0.0, 1.0)) ** 1.6, 0.5)
        elif self.climate_path == "high4c":
            x = np.clip((yr - 2019) / 81.0, 0.0, 1.0)
            tg_target = 1.24 + 2.96 * (x ** 1.15)
            co2_target = 37.0 + 38.0 * (x ** 0.9) * (1.0 - 0.15 * x)
        else:
            return
        self.tg_forced = 1.24 + (tg_target - 1.24) * tcre_ratio
        sim.Tg = np.maximum(self.tg_forced + self.tg_anom, 0.5)
        sim.last_Tg = sim.Tg[:, 0].copy()
        cur_sum = np.maximum(np.nansum(rec["co2"], axis=1, keepdims=True), 1e-6)
        rec["co2"] = rec["co2"] * (co2_target * 1000.0 / cur_sum)


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


def q3_rt(arr_rt: np.ndarray) -> dict:
    return {
        "p10": np.nanpercentile(arr_rt, 10, axis=0).tolist(),
        "p50": np.nanpercentile(arr_rt, 50, axis=0).tolist(),
        "p90": np.nanpercentile(arr_rt, 90, axis=0).tolist(),
    }


def summarise_scenario(o: dict, Wx, ridx: np.ndarray) -> dict:
    """Extrae trayectorias anuales 2019-2100 (idx 69..150), probabilidades de cola,
    desglose regional R10 y países representativos."""
    sl = slice(69, 151)  # 2019..2100 (82 años)
    R, N, T = o["ly"][:, :, sl].shape
    pop = Wx.pop[:, sl]  # (N, T)
    ly = o["ly"][:, :, sl]
    y = np.exp(ly)
    ok = np.isfinite(y)

    # Series mundiales 2019..2100
    gdp_w = np.nansum(np.where(ok, y * pop[None], 0.0), axis=1) / np.maximum(
        np.nansum(np.where(ok, pop[None], 0.0), axis=1), 1e-9
    )
    gdp_med = np.nanmedian(y, axis=1)
    gini_rt = np.zeros((R, T))
    for r in range(R):
        for ti in range(T):
            gini_rt[r, ti] = gini_w(y[r, :, ti], pop[:, ti])

    co2_gt = np.nansum(o["co2"][:, :, sl], axis=1) / 1000.0
    p0_1970 = o["price"][:, 20:21]
    price_idx = o["price"][:, sl] / np.maximum(p0_1970, 1e-9)

    world = {
        "gdp_pc_world": q3_rt(gdp_w),
        "gdp_pc_med": q3_rt(gdp_med),
        "gini": q3_rt(gini_rt),
        "Tg": q3_rt(o["Tg"][:, sl]),
        "co2": q3_rt(co2_gt),
        "price": q3_rt(price_idx),
        "E": q3_rt(np.nanmean(o["E"][:, :, sl], axis=1)),
        "E_popw": q3_rt(
            np.nansum(np.where(np.isfinite(o["E"][:, :, sl]), o["E"][:, :, sl] * pop[None], 0.0), axis=1)
            / np.maximum(np.nansum(np.where(np.isfinite(o["E"][:, :, sl]), pop[None], 0.0), axis=1), 1e-9)
        ),
        "ne": q3_rt(np.nanmean(o["ne"][:, :, sl], axis=1)),
        "psi": q3_rt(np.nanmedian(o["psi"][:, :, sl], axis=1)),
        "asab": q3_rt(np.nanmean(o["asab"][:, :, sl], axis=1)),
        "demo": q3_rt(np.nanmean(o["demo"][:, :, sl], axis=1)),
        "hazard": q3_rt(np.nanmean(o["hazard"][:, :, sl], axis=1)),
        "dclim": q3_rt(np.nanmedian(o["dclim"][:, :, sl], axis=1)),
        "natcap": q3_rt(np.nanmedian(o["natcap"][:, :, sl], axis=1)),
        "debt": q3_rt(np.nanmedian(o["debt"][:, :, sl], axis=1)),
    }
    if "poly" in o:
        world["poly"] = q3_rt(np.nanmean(o["poly"][:, :, sl], axis=1))
    if "p_food" in o:
        kc = o["kcal"][:, :, sl]
        ok_k = np.isfinite(kc)
        world["p_food"] = q3_rt(o["p_food"][:, sl])
        world["kcal"] = q3_rt(
            np.nansum(np.where(ok_k, kc * pop[None], 0.0), axis=1)
            / np.maximum(np.nansum(np.where(ok_k, pop[None], 0.0), axis=1), 1e-9)
        )
        world["pop_under_2000_m"] = q3_rt(np.nansum(np.where(ok_k & (kc < 2000.0), pop[None], 0.0), axis=1))
        world["pop_under_2300_m"] = q3_rt(np.nansum(np.where(ok_k & (kc < 2300.0), pop[None], 0.0), axis=1))
        ex_yr = np.nansum(np.nan_to_num(o["excess_deaths"][:, :, sl]), axis=1)
        ex_yr[:, 0] = 0.0  # empezar acumulación en 2020
        world["excess_deaths_yr_m"] = q3_rt(ex_yr)
        world["excess_deaths_cum_m"] = q3_rt(np.cumsum(ex_yr, axis=1))
        world["pop_dyn_b"] = q3_rt(np.nansum(o["pop_dyn"][:, :, sl], axis=1) / 1000.0)

    # Probabilidades de cola hacia 2100 (relativas a 2026 = índice 7 dentro de sl)
    rel_gdp = (gdp_w[:, -1] / np.maximum(gdp_w[:, 7], 1e-9)) - 1.0
    haz_end = np.nanmean(o["hazard"][:, :, 150], axis=1)
    dem_end = np.nanmean(o["demo"][:, :, 150], axis=1)
    tg_end = o["Tg"][:, 150]
    west_m = np.isin(ridx, [0, 1, 2])
    west_dem_end = np.nanmean(o["demo"][:, west_m, 150], axis=1)
    tails = {
        "gdp_drop_30": float(np.mean(rel_gdp < -0.30)),
        "gdp_drop_50": float(np.mean(rel_gdp < -0.50)),
        "haz_gt_25": float(np.mean(haz_end > 0.25)),
        "haz_gt_35": float(np.mean(haz_end > 0.35)),
        "demo_lt_35": float(np.mean(dem_end < 0.35)),
        "west_demo_lt_70": float(np.mean(west_dem_end < 0.70)),
        "tg_gt_30": float(np.mean(tg_end > 3.0)),
        "tg_gt_40": float(np.mean(tg_end > 4.0)),
    }
    if "kcal" in o:
        kc_end = o["kcal"][:, :, 150]
        p2k_end = np.nansum(np.where(np.isfinite(kc_end) & (kc_end < 2000.0), Wx.pop[:, 150][None], 0.0), axis=1)
        ex_tot = np.nansum(np.nan_to_num(o["excess_deaths"][:, :, 70:151]), axis=(1, 2))
        tails.update({
            "famine_500m": float(np.mean(p2k_end > 500.0)),
            "famine_1000m": float(np.mean(p2k_end > 1000.0)),
            "famine_3000m": float(np.mean(p2k_end > 3000.0)),
            "exd_gt_200m": float(np.mean(ex_tot > 200.0)),
            "exd_gt_500m": float(np.mean(ex_tot > 500.0)),
            "exd_gt_1000m": float(np.mean(ex_tot > 1000.0)),
            "exd_gt_2000m": float(np.mean(ex_tot > 2000.0)),
        })

    # Desglose regional R10 + 3 macrobloques en 2026, 2040, 2060, 2080, 2100
    reg_groups = [(c, ridx == i, elites.R10_NAMES[c]) for i, c in enumerate(elites.R10_CODES)] + [
        ("WEST", np.isin(ridx, [0, 1, 2]), "Occidente (NAM+EUR+OPA)"),
        ("ASIA_EM", np.isin(ridx, [4, 5, 6]), "Asia Emergente (CHN+SAS+SEA)"),
        ("SOUTH", np.isin(ridx, [7, 8, 9]), "Sur Global (LAM+MEA+SSA)"),
    ]
    regions = {}
    for code, m, label in reg_groups:
        snaps = {}
        for yr in (2019, 2026, 2040, 2060, 2080, 2100):
            ti = yr - 1950
            pt = Wx.pop[m, ti]
            yt = np.exp(o["ly"][:, m, ti])
            gw = np.nansum(np.where(np.isfinite(yt), yt * pt[None], 0.0), axis=1) / np.maximum(pt.sum(), 1e-9)
            row = {
                "gdp_pc": [float(np.nanpercentile(gw, q)) for q in (10, 50, 90)],
                "E": [float(np.nanpercentile(np.nanmean(o["E"][:, m, ti], 1), q)) for q in (10, 50, 90)],
                "ne": float(np.nanmedian(np.nanmean(o["ne"][:, m, ti], 1))),
                "psi": float(np.nanmedian(np.nanmedian(o["psi"][:, m, ti], 1))),
                "asab": float(np.nanmedian(np.nanmean(o["asab"][:, m, ti], 1))),
                "demo": float(np.nanmedian(np.nanmean(o["demo"][:, m, ti], 1))),
                "hazard": float(np.nanmedian(np.nanmean(o["hazard"][:, m, ti], 1))),
            }
            if "kcal" in o:
                kct = o["kcal"][:, m, ti]
                row["kcal"] = float(np.nanmedian(np.nansum(np.where(np.isfinite(kct), kct * pt[None], 0.0), 1) / np.maximum(pt.sum(), 1e-9)))
                row["pop_under_2000_m"] = float(np.nanmedian(np.nansum(np.where(np.isfinite(kct) & (kct < 2000.0), pt[None], 0.0), 1)))
                row["exd_cum_m"] = float(np.nanmedian(np.nansum(np.nan_to_num(o["excess_deaths"][:, m, 70:ti + 1]), axis=(1, 2)))) if ti >= 70 else 0.0
            snaps[str(yr)] = row
        # Trayectorias completas 2019..2100 para macrobloques y NAM/EUR/CHN/SSA/SAS/LAM
        traj = {}
        if code in ("WEST", "ASIA_EM", "SOUTH", "REF", "NAM", "EUR", "CHN", "SAS", "LAM", "SSA", "MEA"):
            pm = Wx.pop[m, sl]
            ym = np.exp(o["ly"][:, m, sl])
            gwm = np.nansum(np.where(np.isfinite(ym), ym * pm[None], 0.0), axis=1) / np.maximum(pm.sum(axis=0)[None], 1e-9)
            traj["gdp_pc"] = q3_rt(gwm)
            traj["E"] = q3_rt(np.nanmean(o["E"][:, m, sl], axis=1))
            traj["psi"] = q3_rt(np.nanmedian(o["psi"][:, m, sl], axis=1))
            traj["hazard"] = q3_rt(np.nanmean(o["hazard"][:, m, sl], axis=1))
            traj["demo"] = q3_rt(np.nanmean(o["demo"][:, m, sl], axis=1))
        regions[code] = {"label": label, "n": int(m.sum()), "snap": snaps, "traj": traj}

    # Países representativos
    iso2i = {c: i for i, c in enumerate(Wx.iso3)}
    countries = {}
    for c in SHOWCASE:
        i = iso2i[c]
        cs = {}
        for yr in (2026, 2060, 2100):
            ti = yr - 1950
            cs[str(yr)] = {
                "gdp_pc": [float(np.nanpercentile(np.exp(o["ly"][:, i, ti]), q)) for q in (10, 50, 90)],
                "E": float(np.nanmedian(o["E"][:, i, ti])),
                "psi": float(np.nanmedian(o["psi"][:, i, ti])),
                "asab": float(np.nanmedian(o["asab"][:, i, ti])),
                "demo": float(np.nanmean(o["demo"][:, i, ti])),
                "hazard": float(np.nanmean(o["hazard"][:, i, ti])),
                "kcal": float(np.nanmedian(o["kcal"][:, i, ti])) if "kcal" in o else None,
            }
        countries[c] = cs

    return {"world": world, "tails": tails, "regions": regions, "countries": countries}


def main():
    t0 = time.time()
    print("1. Construyendo mundo 1950-2100 con demografía extendida estilo WPP...")
    cal = json.loads((RES_DIR / "calibration.json").read_text(encoding="utf-8"))
    p = Params().with_vector(list(cal["params"].keys()), list(cal["params"].values()))
    W100_flat = build_world(horizon=HORIZON_END)
    W100 = extend_wpp_demography_2100(W100_flat)
    bm = BehaviorModel(cutoff=69).fit(W, LY)
    ridx = elites.region_index(W100.iso3)
    R = 64

    specs = [
        ("S0_legado", "L1 Legado (9e96f6b)", W100_flat, dict(legacy=True)),
        (
            "S5_adapt_verde",
            "Transición Verde + Adaptación Tecnológica (<2 °C)",
            W100,
            dict(
                stress_model=ScenarioStressModel(
                    W100,
                    StressConfig(
                        glob_lin=0.048,        # 40 % de adaptación térmica/macro
                        glob_quad=0.021,
                        yield_ceiling=1.65,    # mejora biotecnológica / agrícola
                        yield_gain_rate=0.018,
                    ),
                    climate_path="netzero",
                    clean_energy_decouple=True,
                )
            ),
        ),
        (
            "S2_netzero",
            "Transición Verde Pura (Net-Zero ~2060, +1,9 °C)",
            W100,
            dict(stress_model=ScenarioStressModel(W100, StressConfig(), "netzero")),
        ),
        (
            "S1_endogeno",
            "L1 V2 Endógeno + WPP (+2,8 °C, defecto)",
            W100,
            dict(stress_model=ScenarioStressModel(W100, StressConfig(), "endog")),
        ),
        (
            "S3_alta_4c",
            "Alta Trayectoria Climática (+4,2 °C)",
            W100,
            dict(stress_model=ScenarioStressModel(W100, StressConfig(), "high4c")),
        ),
        (
            "S4_policrisis",
            "Policrisis Compuesta (+4,2 °C + Choques ×2)",
            W100,
            dict(
                stress_model=ScenarioStressModel(
                    W100,
                    StressConfig(global_shock_prob=0.07, yield_ceiling=1.25),
                    "high4c",
                )
            ),
        ),
    ]

    scenarios_out = {}
    for key, label, Wx, kwargs in specs:
        ts = time.time()
        bc = BehaviorPolicy(bm, p, noise=1.0, freeze=69, seed=12)
        pol = lambda s, t: s.hist_actions(t) if t <= 69 else bc(s, t)
        sim = Simulator(Wx, p, R=R, seed=5, **kwargs)
        o = sim.run(0, Wx.T, policy=pol, freeze_from=69, assimilate_at=69)
        res = summarise_scenario(o, Wx, ridx)
        res["label"] = label
        scenarios_out[key] = res
        w2100 = res["world"]
        print(
            f"  {key:16s} ({time.time()-ts:4.1f}s): Tg={w2100['Tg']['p50'][-1]:.2f}C | "
            f"PIB=${w2100['gdp_pc_world']['p50'][-1]:6.0f} | E={w2100['E']['p50'][-1]*100:.1f}% | "
            f"Demo={w2100['demo']['p50'][-1]*100:.1f}% | Haz={w2100['hazard']['p50'][-1]*100:.1f}% | "
            f"ExD={w2100.get('excess_deaths_cum_m', {'p50':[0]})['p50'][-1]:.0f}M",
            flush=True,
        )

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
            "R": R,
            "years": list(range(2019, HORIZON_END + 1)),
            "pop_wpp_b": (W100.pop[:, 69:151].sum(axis=0) / 1000.0).tolist(),
            "pop_flat_b": (W100_flat.pop[:, 69:151].sum(axis=0) / 1000.0).tolist(),
            "seconds": round(time.time() - t0, 1),
        },
        "scenarios": scenarios_out,
    })
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Listo en {time.time()-t0:.1f}s -> {OUT_JSON}")


if __name__ == "__main__":
    main()
