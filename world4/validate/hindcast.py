"""W4 (parte de F1): hindcast libre de la población mundial 1990 -> 2023.

Origen 1990, con el modelo vigente (F1 + F2): la calibración del L1 es la de corte 1990, la regla de conducta y las ecuaciones
demográficas se estiman con datos <= 1990 y, desde ahí, el L1 corre hasta 2023 SIN reanclar a
ningún dato. La verdad es la población del mundo del L1 (``build_world(horizon=2023)``): PWT hasta
2019 y WPP 2024 reescalado a la PWT de 2019 en 2020-2023, sumada sobre los mismos 180 países.

Criterio (hoja de ruta, salida de F1 y W4): el dato dentro de la banda p10-p90 del ensemble en
>= 80 % de los años 1991-2023. Se informa también la extrapolación de F0.5 (tendencia de 5 años),
que no tiene banda.

Aviso de vintage: la estructura por edad inicial y las tasas por edad de 1990 son las de WPP 2024
(historia revisada); no hay ediciones antiguas cargadas para medir esa fuga.
"""
from __future__ import annotations

import json

import numpy as np

ORIGIN, END = 1990, 2023
MIN_INSIDE = 0.80


def run(R: int = 64, seed: int | None = None, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import model
    W23 = model.build_world(horizon=END)
    t0 = ORIGIN - 1950
    H = END - ORIGIN
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W23, horizon=H, f2=coupling.F2Config())
    tr = coupling.forecast(t0, R=4, seed=seed, pop_mode="trend", world=W23, horizon=H)
    years = np.arange(ORIGIN, END + 1)
    truth = np.nansum(W23.pop[:, t0:t0 + H + 1], 0) / 1e3                       # miles de millones
    ens = np.nansum(f["pop"], 1) / 1e3                                          # (R, H+1)
    p10, p50, p90 = (np.percentile(ens, q, 0) for q in (10, 50, 90))
    trend = np.nansum(tr["pop"][0], 0) / 1e3
    inside = (truth >= p10) & (truth <= p90)
    k = slice(1, None)
    out = {"origin": ORIGIN, "end": END, "R": R, "unit": "billion persons (180 L1 countries)",
           "years": years.tolist(), "truth": truth.tolist(), "p10": p10.tolist(), "p50": p50.tolist(),
           "p90": p90.tolist(), "trend_f05": trend.tolist(),
           "inside_share": float(inside[k].mean()), "n_years": int(inside[k].size),
           "rel_err_p50_end": float(p50[-1] / truth[-1] - 1), "rel_err_trend_end": float(trend[-1] / truth[-1] - 1),
           "max_abs_rel_err_p50": float(np.max(np.abs(p50[k] / truth[k] - 1))),
           "max_abs_rel_err_trend": float(np.max(np.abs(trend[k] / truth[k] - 1)))}
    out["ok"] = out["inside_share"] >= MIN_INSIDE
    log(f"hindcast {ORIGIN}->{END}: inside p10-p90 in {out['inside_share']:.0%} of years; "
        f"2023 p50 {out['rel_err_p50_end']:+.2%} vs trend {out['rel_err_trend_end']:+.2%}")
    return out


# ------------------------------------------------------------------ F2: psi_t como salida
PSI_ORIGINS = (1980, 1990)
PSI_MIN_INSIDE = 0.70


def run_psi(origin: int = 1980, R: int = 64, seed: int | None = None, log=print) -> dict:
    """``psi_t`` (rentas de recursos del mundo / PIB) en hindcast libre desde ``origin`` hasta 2019:
    mercados de F2 estimados con datos <= origen, población de F1, sin reanclar. La verdad es la
    ``psi_t`` del Atlas (WDI), que desde F2 ya no entra en el modelo después del origen.

    1970-1979 no puede ser fuera de muestra: las rentas por clase del WDI empiezan en 1970. Se informa
    el origen 1980 (criterio) y el 1990; la referencia es la media de 10 años antes del origen (lo que
    hacía el Atlas)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    t0 = origin - 1950
    H = W.T - 1 - t0
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", f2=coupling.F2Config(), horizon=H)
    years = np.arange(origin, origin + H + 1)
    truth = W.psi_t[t0:t0 + H + 1].astype(float)
    ens = f["psi"]
    p10, p50, p90 = (np.percentile(ens, q, 0) for q in (10, 50, 90))
    base = float(np.mean(W.psi_t[t0 - 9:t0 + 1]))
    k = slice(1, None)
    inside = (truth >= p10) & (truth <= p90)
    out = {"origin": origin, "end": int(years[-1]), "R": R, "unit": "world resource rents / world GDP (fraction)",
           "years": years.tolist(), "truth": truth.tolist(), "p10": p10.tolist(), "p50": p50.tolist(),
           "p90": p90.tolist(), "atlas_10y_mean": base,
           "inside_share": float(inside[k].mean()), "n_years": int(inside[k].size),
           "mae_p50": float(np.mean(np.abs(p50[k] - truth[k]))), "mae_atlas": float(np.mean(np.abs(base - truth[k]))),
           "prices": {c: {"p50": np.percentile(f[f"price_{c}"], 50, 0).tolist()} for c in ("oil", "gas", "coal", "metals")
                      if f"price_{c}" in f}}
    out["ok"] = out["inside_share"] >= PSI_MIN_INSIDE
    log(f"psi hindcast {origin}->{out['end']}: inside p10-p90 in {out['inside_share']:.0%} of years; MAE p50 "
        f"{out['mae_p50']:.4f} vs Atlas 10-year mean {out['mae_atlas']:.4f}")
    return out


# ------------------------------------------------------------------ F3: mezcla de energía primaria
ENERGY_ORIGINS = (1990, 2005)
ENERGY_END = 2023


def run_energy(origin: int = 2005, R: int = 64, seed: int | None = None, log=print) -> dict:
    """Mezcla de energía primaria por región (6 áreas de la ONU) y fuente, y energía primaria mundial,
    en hindcast libre desde ``origin`` hasta 2023 con el modelo F1+F2+F3 (todo estimado con datos <=
    origen, sin reanclar). Verdad: OWID / Energy Institute (con el desglose imputado declarado en
    ``world4.data.energy``). Referencia: la mezcla del origen congelada (persistencia)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import model
    from world4.data import energy as edata
    from world4.sectors import energy
    W23 = model.build_world(horizon=ENERGY_END)
    t0 = origin - 1950
    H = ENERGY_END - origin
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W23, horizon=H,
                          f2=coupling.F2Config(), f3=coupling.F3Config())
    years = np.arange(origin, ENERGY_END + 1)
    a = edata.aligned(W23.iso3, W23.isonum, years)
    region = a["region"]
    ok = np.isfinite(a["pe"][:, 0])                      # países con energía en el origen (los que simula F3)
    out = {"origin": origin, "end": ENERGY_END, "R": R, "years": years.tolist(), "regions": {}, "sources": list(energy.SOURCES)}
    inside, err_m, err_p = [], [], []
    for r, name in enumerate(edata.REGIONS):
        m = (region == r) & ok
        tot_obs = np.nansum(np.where(m[:, None], a["pe"], 0), 0)
        tot_mod = np.nansum(np.where(m[None, :, None], f["energy_PE"], 0), 1)          # (R, T)
        reg = {"countries": int(m.sum()), "shares": {}}
        for s in energy.SOURCES:
            obs = np.nansum(np.where(m[:, None], a[f"src_{s}"], 0), 0) / np.maximum(tot_obs, 1e-12)
            mod = np.nansum(np.where(m[None, :, None], f[f"energy_gen_{s}"], 0), 1) / np.maximum(tot_mod, 1e-12)
            p10, p50, p90 = (np.percentile(mod, q, 0) for q in (10, 50, 90))
            ins = (obs >= p10 - 1e-12) & (obs <= p90 + 1e-12)
            inside.append(ins[1:])
            err_m.append(np.abs(p50[1:] - obs[1:]))
            err_p.append(np.abs(obs[0] - obs[1:]))
            reg["shares"][s] = {"obs": obs.tolist(), "p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist(),
                                "inside_share": float(ins[1:].mean())}
        out["regions"][name] = reg
    tot_obs = np.nansum(np.where(ok[:, None], a["pe"], 0), 0)
    tot_mod = np.nansum(np.where(ok[None, :, None], f["energy_PE"], 0), 1)
    q10, q50, q90 = (np.percentile(tot_mod, q, 0) for q in (10, 50, 90))
    ins_w = (tot_obs >= q10) & (tot_obs <= q90)
    out["world_pe"] = {"unit": "TWh (substitution method)", "obs": tot_obs.tolist(), "p10": q10.tolist(),
                       "p50": q50.tolist(), "p90": q90.tolist(), "inside_share": float(ins_w[1:].mean()),
                       "rel_err_p50_end": float(q50[-1] / tot_obs[-1] - 1)}
    out["mix_inside_share"] = float(np.mean(np.concatenate(inside)))
    out["mix_mae_p50"] = float(np.mean(np.concatenate(err_m)))
    out["mix_mae_persistence"] = float(np.mean(np.concatenate(err_p)))
    out["materials_p50_end"] = {m: float(np.median(f[f"material_{m}"][:, -1])) for m in energy.MATERIALS_T_PER_MW}
    out["ok"] = out["world_pe"]["inside_share"] >= MIN_INSIDE
    log(f"energy hindcast {origin}->{ENERGY_END}: world primary energy inside p10-p90 in "
        f"{out['world_pe']['inside_share']:.0%} of years (p50 in {ENERGY_END} {out['world_pe']['rel_err_p50_end']:+.1%}); "
        f"regional mix inside in {out['mix_inside_share']:.0%} of region x source x year cells; mix MAE p50 "
        f"{out['mix_mae_p50']:.4f} vs persistence {out['mix_mae_persistence']:.4f}")
    return out


# ------------------------------------------------------------------ F4: CO2 y GMST
CLIMATE_ORIGINS = (1990, 2005)
CLIMATE_END = 2023


def run_climate(origin: int = 1990, R: int = 64, seed: int | None = None, log=print) -> dict:
    """CO2 fósil mundial, concentración de CO2 y GMST en hindcast libre desde ``origin`` hasta 2023 con
    F1+F2+F3+F4 (emisiones generadas por el L1, FaIR calibrado, sin reanclar). Verdad: CMIP7 (CO2
    fósil), NOAA vía el Atlas (ppm, hasta 2019) y HadCRUT5 (GMST). Referencias: la GMST del Atlas (TCRE
    sobre el CO2 de su extracción, con el mismo L1 F1-F3), la persistencia y la tendencia de 10 años.

    Fuga declarada: el posterior de FaIR se restringió con la GMST observada hasta 2023; lo que este
    hindcast mide fuera de muestra es la cadena economía -> energía -> emisiones -> clima, no los
    parámetros del clima."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, model
    from world4.data import climate as cdata
    W23 = model.build_world(horizon=CLIMATE_END)
    t0 = origin - 1950
    H = CLIMATE_END - origin
    seed = t0 if seed is None else seed
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W23, horizon=H,
                          f2=coupling.F2Config(), f3=coupling.F3Config(), f4=coupling.F4Config())
    a = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W23, horizon=H,
                          f2=coupling.F2Config(), f3=coupling.F3Config())
    years = np.arange(origin, CLIMATE_END + 1)
    k = slice(1, None)
    out = {"origin": origin, "end": CLIMATE_END, "R": R, "years": years.tolist(),
           "leak": "FaIR posterior constrained with observed GMST up to 2023 (climate parameters in sample)"}

    def band(ens, truth, name, unit):
        p10, p50, p90 = (np.percentile(ens, q, 0) for q in (10, 50, 90))
        ok = np.isfinite(truth)
        ins = (truth >= p10) & (truth <= p90)
        last = int(np.where(ok)[0][-1])
        out[name] = {"unit": unit, "obs": [None if not np.isfinite(x) else float(x) for x in truth],
                     "p10": p10.tolist(), "p50": p50.tolist(), "p90": p90.tolist(),
                     "inside_share": float(ins[k][ok[k]].mean()), "n_years": int(ok[k].sum()),
                     "mae_p50": float(np.mean(np.abs(p50[k][ok[k]] - truth[k][ok[k]]))),
                     "end_year": int(years[last]), "err_p50_end": float(p50[last] - truth[last])}
        return out[name]

    g = band(f["Tg"], cdata.gmst_observed(years), "gmst", "degC above 1850-1900 (HadCRUT5)")
    c = band(f["climate_co2_ffi"], cdata.co2_ffi_observed(years), "co2_ffi", "GtCO2/year (CMIP7 history)")
    c["rel_err_p50_end"] = float(np.array(c["p50"])[-1] / c["obs"][-1] - 1)
    ppm = np.array([calibrate.W.wlocf("clima__co2_ppm", int(y) - 1950, np.nan) if y <= 2019 else np.nan
                    for y in years])
    band(f["climate_co2_conc"], ppm, "co2_ppm", "ppm (NOAA via the Atlas, to 2019)")
    obs = cdata.gmst_observed(years)
    hist = cdata.gmst_observed(np.arange(origin - 9, origin + 1))
    slope = np.polyfit(np.arange(10), hist, 1)[0]
    tgA = np.median(a["Tg"], 0)
    g["mae_persistence"] = float(np.mean(np.abs(obs[0] - obs[k])))
    g["mae_trend_10y"] = float(np.mean(np.abs(obs[0] + slope * np.arange(1, H + 1) - obs[k])))
    g["mae_atlas_tcre"] = float(np.mean(np.abs(tgA[k] - obs[k])))
    g["atlas_tcre_p50"] = tgA.tolist()
    out["ok"] = g["inside_share"] >= MIN_INSIDE and c["inside_share"] >= MIN_INSIDE
    log(f"climate hindcast {origin}->{CLIMATE_END}: GMST inside p10-p90 in {g['inside_share']:.0%} of years (MAE p50 "
        f"{g['mae_p50']:.3f} vs Atlas TCRE {g['mae_atlas_tcre']:.3f}, 10-year trend {g['mae_trend_10y']:.3f}); fossil CO2 "
        f"inside in {c['inside_share']:.0%} (p50 in {CLIMATE_END} {c['rel_err_p50_end']:+.1%})")
    return out


def run_climate_observed(log=print) -> dict:
    """FaIR solo con las emisiones OBSERVADAS 1750-2023 y las 841 configuraciones (con su variabilidad
    interna): GMST 1950-2023 frente a HadCRUT5. Es EN MUESTRA (el posterior se restringió con esa misma
    GMST): comprueba el montaje, no la capacidad de pronóstico."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.data import climate as cdata
    from world4.sectors import climate
    ids = list(cdata.load()["params"].index)
    f = climate.build_fair(ids, CLIMATE_END + 1)
    f.run(progress=False)
    T = f.temperature.loc[dict(layer=0, scenario="historical")].data
    yrs = f.timebounds[:-1]
    ann = 0.5 * (T[1:] + T[:-1])
    ann = ann - ann[(yrs >= 1850) & (yrs <= 1900)].mean(0)
    m = (yrs >= 1950) & (yrs <= CLIMATE_END)
    ens = ann[m]
    obs = cdata.gmst_observed(yrs[m])
    p10, p50, p90 = (np.percentile(ens, q, 1) for q in (10, 50, 90))
    ins = (obs >= p10) & (obs <= p90)
    out = {"years": yrs[m].astype(int).tolist(), "obs": obs.tolist(), "p10": p10.tolist(), "p50": p50.tolist(),
           "p90": p90.tolist(), "inside_share": float(ins.mean()), "mae_p50": float(np.mean(np.abs(p50 - obs))),
           "configs": len(ids), "in_sample": True}
    log(f"FaIR with observed emissions 1950-{CLIMATE_END} (in sample): GMST inside p10-p90 in {out['inside_share']:.0%} "
        f"of years, MAE p50 {out['mae_p50']:.3f} degC")
    return out


def run_carbon_budget(log=print) -> dict:
    """Coherencia con AR6: TCRE del conjunto (1pctCO2) y presupuesto restante desde 2020 con las reglas
    de emisión de F4 y con la variante de no-CO2 «como AR6»."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.sectors import climate
    t = climate.tcre()
    lo, hi = climate.AR6_TCRE_LIKELY
    out = {"tcre": {"p": t["p"], "ar6_likely": [lo, hi], "ar6_best": t["ar6_best"],
                    "median_inside_likely": bool(lo <= t["p"][50] <= hi)},
           "budget_f4": climate.remaining_budget(non_co2="f4"),
           "budget_ar6_like": climate.remaining_budget(non_co2="ar6_like")}
    b = out["budget_f4"]["budgets"]
    out["budget_coherent"] = bool(all(
        AR6 <= ours_hi and ours_lo <= AR6 for lim in ("1.5", "2.0")
        for AR6, ours_lo, ours_hi in [(climate.AR6_BUDGET[float(lim)][50], b[lim]["gtco2_by_probability"][67],
                                       b[lim]["gtco2_by_probability"][33])]))
    fmt = lambda x: f"{x:.0f}" if np.isfinite(x) else ("below the fastest pathway" if x < 0 else "above the slowest")  # noqa: E731
    log(f"TCRE p17/p50/p83 {t['p'][17]:.3f}/{t['p'][50]:.3f}/{t['p'][83]:.3f} degC per 1000 GtCO2 (AR6 likely "
        f"{lo:.2f}-{hi:.2f}); budget from 2020 at 50 %: 1.5 degC {fmt(b['1.5']['gtco2_by_probability'][50])} GtCO2 "
        f"(AR6 500), 2 degC {fmt(b['2.0']['gtco2_by_probability'][50])} (AR6 1350)")
    return out


# ------------------------------------------------------------------ F5: cereales, superficie, calorías, N
FOOD_ORIGINS = (1990, 2005)
FOOD_END = 2023


def run_food(origin: int = 1990, R: int = 64, seed: int | None = None, log=print) -> dict:
    """Producción de cereales, superficie cultivada (cropland), calorías por persona y balance de N de las
    tierras de cultivo, en hindcast libre desde ``origin`` hasta 2023 con F1-F5 (todo estimado con datos
    <= origen, sin reanclar). Verdad: FAOSTAT sumado sobre los MISMOS países que el modelo simula (los
    que tienen datos en el origen: en 1990 faltan los Estados sucesores de la URSS, Yugoslavia y
    Checoslovaquia, el 13 % de los cereales del mundo). Referencias: persistencia y tendencia de 10 años."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import model
    from world4.data import food as fdata
    W23 = model.build_world(horizon=FOOD_END)
    t0 = origin - 1950
    H = FOOD_END - origin
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W23, horizon=H, f5=coupling.F5Config())
    ok = f["sim"].fd.ok
    years = np.arange(origin, FOOD_END + 1)
    a = fdata.aligned(W23.iso3, W23.isonum, years)
    k = slice(1, None)
    out = {"origin": origin, "end": FOOD_END, "R": R, "years": years.tolist(), "countries": int(ok.sum())}

    def band(ens, truth, name, unit):
        p10, p50, p90 = (np.percentile(ens, q, 0) for q in (10, 50, 90))
        good = np.isfinite(truth)
        ins = (truth >= p10) & (truth <= p90)
        hist = truth[: 1]
        g = {"unit": unit, "obs": [float(x) if np.isfinite(x) else None for x in truth], "p10": p10.tolist(),
             "p50": p50.tolist(), "p90": p90.tolist(), "inside_share": float(ins[k][good[k]].mean()),
             "n_years": int(good[k].sum())}
        last = int(np.where(good)[0][-1])
        g["rel_err_p50_end"] = float(p50[last] / truth[last] - 1)
        g["end_year"] = int(years[last])
        g["mae_rel_p50"] = float(np.mean(np.abs(p50[k][good[k]] / truth[k][good[k]] - 1)))
        g["mae_rel_persistence"] = float(np.mean(np.abs(hist[0] / truth[k][good[k]] - 1)))
        out[name] = g
        return g

    def tot(key):
        return np.array([np.nansum(np.where(ok & np.isfinite(a[key][:, j]), a[key][:, j], 0))
                         if np.isfinite(a[key][ok, j]).sum() > 0.8 * ok.sum() else np.nan for j in range(len(years))])

    c = band(f["foodw_cereal_prod"], tot("cereal_prod"), "cereal_prod", "t of cereals (FAOSTAT, simulated countries)")
    cl = band(f["foodw_cropland"], tot("cropland"), "cropland", "1000 ha cropland (FAOSTAT, simulated countries)")
    popd = np.where(ok[:, None] & np.isfinite(a["kcal"]), W23.pop[:, origin - 1950:FOOD_END - 1950 + 1], 0)
    kc_obs = np.nansum(np.nan_to_num(a["kcal"]) * popd, 0) / np.maximum(popd.sum(0), 1e-9)
    band(f["foodw_kcal"], np.where(popd.sum(0) > 0, kc_obs, np.nan), "kcal", "kcal/person/day, population-weighted")
    band(f["foodw_n_balance"], tot("nb_balance"), "n_balance", "t N, cropland nitrogen balance (FAOSTAT)")
    out["n_intentional_tg_p50_end"] = float(np.median(f["foodw_n_intentional_tg"][:, -1]))
    out["water_ag_km3_p50_end"] = float(np.median(f["foodw_water_ag_km3"][:, -1]))
    out["ok"] = c["inside_share"] >= MIN_INSIDE and cl["inside_share"] >= MIN_INSIDE
    log(f"food hindcast {origin}->{FOOD_END} ({int(ok.sum())} countries): cereals inside p10-p90 in {c['inside_share']:.0%} "
        f"of years (p50 in {c['end_year']} {c['rel_err_p50_end']:+.1%}); cropland inside in {cl['inside_share']:.0%} "
        f"({cl['rel_err_p50_end']:+.1%}); kcal inside in {out['kcal']['inside_share']:.0%}; N balance inside in "
        f"{out['n_balance']['inside_share']:.0%}")
    return out


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/w4_pop_hindcast.json")
    r = run()
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=1), encoding="utf-8")
