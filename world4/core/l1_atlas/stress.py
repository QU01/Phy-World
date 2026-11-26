"""Capa biofísica y de colas pesadas (V1) para Phy-World L1 (Atlas del Sistema-Mundo).

Cierra los tres lazos estructurales ausentes o débiles del Atlas 9e96f6b (modo por defecto de
``Simulator`` desde esta revisión; ``legacy=True`` conserva la paridad bit a bit W2):

  M1 · Daños climáticos persistentes y correlacionados [M]:
       - Persistencia ``rho_D = 0.94`` (frente a 0.80), tope al bono en países fríos (+0.004/año)
         y suelo ampliado a ``-3.5`` (frente a ``-1.5``).
       - Arrastre macro global por encima de +1.5 °C (estrés laboral húmedo, comercio, cadenas de
         suministro): ``d_glob = -0.08 * max(Tg - 1.5, 0) - 0.035 * max(Tg - 2.5, 0)^2``.
       - Desastres climáticos más frecuentes y severos con el calentamiento (``dis_kappa = 0.35``,
         ``dis_mu = -5.8``).

  M2 · Frenos biofísicos alimentarios y mercado mundial de cereales [M]:
       - Inicializado estrictamente con datos <= ``obs_until`` (sin fuga temporal).
       - Techo logístico de rendimiento agrícola (``1.45x`` sobre la frontera del origen) y daño
         térmico/hídrico no lineal incremental sobre el clima del origen (−5 %/°C hasta +1.5 °C
         locales adicionales; −14 %/°C por encima), más choques correlacionados de graneros.
       - Equilibrio del mercado mundial de cereales (elasticidad −0.35) que fija el precio real
         ``p_food`` y las calorías por país ``kcal`` según pobreza y dependencia importadora.
       - Lazos cerrados: ``p_food`` y déficit calórico (< 2300 kcal) elevan el riesgo de conflicto
         civil; déficit agudo (< 2000 kcal) genera mortalidad en exceso y reduce la PTF laboral.

  M3 · Colas pesadas Student-t(3), choques globales sincronizados y mortalidad por desastres [M]:
       - Exceso de colas pesadas idiosincráticas en PTF (Student-t, df=3, asimétricas para crisis)
         que corrigen la subcobertura p10-p90 (D7) sin distorsionar los años normales.
       - Choques globales sincronizados (pandemias / crisis globales: 3.5 %/año, −3 % a −10 % PTF
         en todos los países a la vez + exceso de mortalidad global).
       - Mortalidad directa por desastres climáticos extremos escalada por calentamiento y
         vulnerabilidad periférica.

Todos los parámetros añadidos son supuestos estructurales declarados [M] para evitar el sesgo de
optimismo por omisión fuera de muestra; sólo actúan para ``t > obs_until``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


@dataclass(frozen=True)
class StressConfig:
    """Interruptores y parámetros [M] de la capa biofísica y de colas pesadas."""
    climate_co2: bool = True       # M0: CO2 ligado al PIB (intensidad observada) + TCRE efectiva OLS + ENSO (sustituye Hubbert roto)
    climate_severe: bool = True    # M1: daños climáticos persistentes (rho_D=0.94), daño macro global >1.5C y desastres x
    food_brakes: bool = True       # M2: frenos biofísicos alimentarios y mercado mundial -> conflicto y mortalidad
    heavy_tails: bool = True       # M3: colas pesadas Student-t(3), choques globales sincronizados y mortalidad por desastres
    # M0: CO2 y clima sin colapso artificial de Hubbert (estimados con datos <= obs_until)
    tcre_log_sd: float = 0.22      # incertidumbre paramétrica TCRE (~rango probable AR6)
    ci_trend_sd: float = 0.006     # incertidumbre entre miembros sobre descarbonización anual
    enso_rho: float = 0.30         # persistencia anual de la variabilidad interna climática
    gcam_carbon_cap_gt: float = 47500.0  # techo físico total de carbono fósil (GCAM, V6)
    # M1: clima severo
    rho_D: float = 0.94
    cold_bonus_cap: float = 0.004
    dclim_floor: float = -3.5
    glob_lin: float = 0.08
    glob_quad: float = 0.035
    dis_kappa: float = 0.35
    dis_mu: float = -5.8
    # M2: frenos alimentarios
    yield_ceiling: float = 1.45
    yield_gain_rate: float = 0.015
    food_elast: float = 0.35
    # M3: colas pesadas y choques globales
    tail_df: float = 3.0
    tail_thresh: float = 2.2
    tail_scale: float = 0.016
    global_shock_prob: float = 0.035


class StressModel:
    """Estado dinámico (R, N) acoplado al ``Simulator`` de L1 para ``t > obs_until``."""

    def __init__(self, world, config: StressConfig | None = None):
        self.w = world
        self.cfg = config if config is not None else StressConfig()
        self.obs_until = world.T - 1
        self._climate_ablated = False
        # atributos estáticos inicializados en reset(t0, obs_until)
        self.cy0 = None
        self.ndf = None
        self.import_dep = None
        self.kcal0 = None
        self.Tbar_ref = None
        self.T_origin = None
        self.pop_origin = None
        # estado dinámico (R, N)
        self.pop_dyn = None
        self.excess_deaths = None
        self.cum_excess_deaths = None
        self.yield_idx = None
        self._tech_idx = None
        self.kcal = None
        self.p_food = None
        self.d_glob = None
        self.d_food_tfp = None
        self.global_shock_flag = None
        self._rng_clim = None
        self._rng_food = None
        self._rng_tails = None
        # M0: estado climático y de emisiones CO2 (sin colapso Hubbert)
        self.co2_0 = 37.0
        self.luc_0 = 5.5
        self.Y_world_0 = 125e6
        self.g_ci_hat = -0.0125
        self.tcre_hat = 0.62
        self.enso_sd = 0.095
        self.tg_forced_0 = 1.24
        self.tg_anom_0 = 0.0
        self.co2_country_share = None
        self.tcre_r = None
        self.g_ci_r = None
        self.tg_forced = None
        self.tg_anom = None
        self.cum_co2_gt = None

    def _init_origin_data(self, obs_until: int):
        """Extrae condiciones iniciales por país usando estrictamente datos <= ``obs_until``."""
        w = self.w
        t0 = min(max(int(obs_until), 0), w.T - 1)
        cy = w.locf("cereal_yield")[:, t0]
        med_cy = float(np.nanmedian(cy[cy > 0])) if np.any(np.isfinite(cy) & (cy > 0)) else 3000.0
        self.cy0 = np.where(np.isfinite(cy) & (cy > 0), cy, med_cy)

        ndf = w.locf("ndgain_food")[:, t0]
        self.ndf = np.where(np.isfinite(ndf), np.clip(ndf, 0.1, 0.9), 0.45)

        imp = w.locf("ef_net_imports_pc")[:, t0]
        ef = w.locf("ef_cons_pc")[:, t0]
        self.import_dep = np.where(
            np.isfinite(imp) & np.isfinite(ef) & (ef > 0.1),
            np.clip(imp / ef, 0.0, 0.75),
            0.15,
        )

        # media mundial de calorías según época del origen (FAOSTAT: ~2720 en 2000, ~2850 en 2010, ~2970 en 2019)
        yr0 = 1950 + t0
        world_kcal_ref = float(np.clip(2720.0 + 13.0 * (yr0 - 2000), 2400.0, 3050.0))
        ly0 = np.log(np.maximum(w.Y[:, t0] / np.maximum(w.pop[:, t0], 1e-6), 200.0))
        med_ly0 = float(np.nanmedian(ly0)) if np.any(np.isfinite(ly0)) else 8.8
        ly0 = np.where(np.isfinite(ly0), ly0, med_ly0)
        raw_k = world_kcal_ref + 360.0 * (ly0 - med_ly0) - 450.0 * (self.ndf - 0.45)
        raw_k = np.clip(raw_k, 2100.0, 3600.0)
        pop0 = np.nan_to_num(w.pop[:, t0])
        denom = float(np.sum(pop0))
        mean_k = float(np.sum(raw_k * pop0) / denom) if denom > 0 else world_kcal_ref
        self.kcal0 = np.clip(raw_k * (world_kcal_ref / max(mean_k, 1.0)), 2050.0, 3650.0)

        tclim = w.locf("tmp_level")
        tbar = np.nanmean(tclim[:, :min(31, t0 + 1)], axis=1)
        self.Tbar_ref = np.where(np.isfinite(tbar), tbar, 15.0)
        t_ori = np.nanmean(tclim[:, max(0, t0 - 9):t0 + 1], axis=1)
        self.T_origin = np.where(np.isfinite(t_ori), t_ori, self.Tbar_ref)
        self.pop_origin = np.maximum(np.nan_to_num(w.pop[:, t0]), 0.1)

        # M0: estimar intensidad de carbono (20 años <= t0) y TCRE efectiva OLS (1970..t0) sin fuga temporal
        co2_s = w.world.get("clima__co2_mt", np.full(w.T, 30000.0)) / 1000.0
        luc_s = w.world.get("clima__co2_luc_mt", np.full(w.T, 5500.0)) / 1000.0
        gmst_s = w.world.get("clima__gmst_preind", np.full(w.T, 1.0))
        Y_w_hist = np.nansum(np.where(w.observed, w.Y, 0.0), axis=0)
        self.co2_0 = float(co2_s[t0]) if np.isfinite(co2_s[t0]) and co2_s[t0] > 0 else 30.0
        luc_win = luc_s[max(0, t0 - 9):t0 + 1]
        self.luc_0 = float(np.nanmean(luc_win)) if np.any(np.isfinite(luc_win)) else 5.5
        self.Y_world_0 = float(Y_w_hist[t0]) if np.isfinite(Y_w_hist[t0]) and Y_w_hist[t0] > 0 else 100e6

        i_start = min(20, t0)  # 1970
        j0 = max(i_start, t0 - 20)
        ok_ci = (np.isfinite(co2_s[j0:t0 + 1]) & (co2_s[j0:t0 + 1] > 0)
                 & np.isfinite(Y_w_hist[j0:t0 + 1]) & (Y_w_hist[j0:t0 + 1] > 0))
        if ok_ci.sum() >= 5:
            ci_log = np.log(co2_s[j0:t0 + 1][ok_ci] / (Y_w_hist[j0:t0 + 1][ok_ci] / 1e6))
            x_ci = np.arange(len(ok_ci), dtype=float)[ok_ci]
            self.g_ci_hat = float(np.clip(np.polyfit(x_ci, ci_log, 1)[0], -0.025, 0.005))
        else:
            self.g_ci_hat = -0.0125

        ok_t = (np.isfinite(gmst_s[i_start:t0 + 1])
                & np.isfinite(co2_s[i_start:t0 + 1]) & np.isfinite(luc_s[i_start:t0 + 1]))
        if ok_t.sum() >= 10:
            cum_t = np.cumsum(np.nan_to_num(co2_s[i_start:t0 + 1]) + np.nan_to_num(luc_s[i_start:t0 + 1])) / 1000.0
            g_win = gmst_s[i_start:t0 + 1]
            slope, icpt = np.polyfit(cum_t[ok_t], g_win[ok_t], 1)
            self.tcre_hat = float(np.clip(slope, 0.35, 0.90))
            fit_g = slope * cum_t + icpt
            self.enso_sd = float(np.clip(np.std(g_win[ok_t] - fit_g[ok_t]), 0.06, 0.14))
            self.tg_forced_0 = float(fit_g[-1])
            self.tg_anom_0 = float(g_win[-1] - fit_g[-1]) if np.isfinite(g_win[-1]) else 0.0
        else:
            self.tcre_hat = 0.62
            self.enso_sd = 0.095
            self.tg_forced_0 = float(gmst_s[t0]) if np.isfinite(gmst_s[t0]) else 1.0
            self.tg_anom_0 = 0.0

        # cuota de CO2 por país en el origen (para repartir las emisiones mundiales entre los 180 países)
        co2_pc0 = w.locf("co2_pc")[:, t0]
        c_em = np.where(np.isfinite(co2_pc0) & (co2_pc0 > 0), co2_pc0 * self.pop_origin, np.nan)
        y0_safe = np.where(np.isfinite(w.Y[:, t0]) & (w.Y[:, t0] > 0), w.Y[:, t0], 0.0)
        c_em = np.where(np.isfinite(c_em), c_em, y0_safe * 1e-4)
        tot_c = float(np.sum(c_em))
        self.co2_country_share = c_em / max(tot_c, 1e-12)

    def reset(self, R: int, t0: int, obs_until: int, params, seed: int = 0):
        self.obs_until = obs_until
        self._rng_clim = np.random.default_rng(int(seed) + 1511)
        self._rng_food = np.random.default_rng(int(seed) + 2029)
        self._rng_tails = np.random.default_rng(int(seed) + 3037)
        self._init_origin_data(obs_until)
        # si la ablación sin_clima_ecologia apagó el bloque climático, respetar la ablación
        self._climate_ablated = (
            abs(params.bhm1) < 1e-12 and abs(params.bhm2) < 1e-12 and abs(params.dis_p) < 1e-12
        )
        N = self.w.N
        self.pop_dyn = np.broadcast_to(self.w.pop[:, t0], (R, N)).astype(float).copy()
        self.excess_deaths = np.zeros((R, N), dtype=float)
        self.cum_excess_deaths = np.zeros((R, N), dtype=float)
        self.yield_idx = np.ones((R, N), dtype=float)
        self._tech_idx = np.ones((R, N), dtype=float)
        self.kcal = np.broadcast_to(self.kcal0, (R, N)).copy()
        self.p_food = np.ones((R, 1), dtype=float)
        self.d_glob = np.zeros((R, 1), dtype=float)
        self.d_food_tfp = np.zeros((R, N), dtype=float)
        self.global_shock_flag = np.zeros(R, dtype=float)
        # M0: sorteos paramétricos por miembro (TCRE con dispersión AR6 y tendencia de intensidad de carbono)
        self.tcre_r = self.tcre_hat * np.exp(self._rng_clim.normal(0.0, self.cfg.tcre_log_sd, size=(R, 1)))
        self.tcre_r = np.clip(self.tcre_r, 0.30, 1.25)
        self.g_ci_r = np.clip(
            self.g_ci_hat + self._rng_clim.normal(0.0, self.cfg.ci_trend_sd, size=(R, 1)),
            -0.032, 0.005,
        )
        self.tg_forced = np.full((R, 1), self.tg_forced_0, dtype=float)
        self.tg_anom = np.full((R, 1), self.tg_anom_0, dtype=float)
        self.cum_co2_gt = np.zeros((R, 1), dtype=float)

    @property
    def climate_active(self) -> bool:
        return self.cfg.climate_severe and not self._climate_ablated

    def pre_step(self, sim, t: int, obs_mode: bool):
        """Actualiza población dinámica y daño macro global antes del cálculo de producción."""
        w = self.w
        R, N = sim.R, w.N
        if obs_mode:
            self.pop_dyn = np.broadcast_to(w.pop[:, t], (R, N)).astype(float).copy()
            self.d_glob.fill(0.0)
            self.d_food_tfp.fill(0.0)
            self.excess_deaths.fill(0.0)
            self.global_shock_flag.fill(0.0)
            return None

        g_base = w.pop[:, t] / np.maximum(w.pop[:, max(t - 1, 0)], 1e-6)
        self.pop_dyn = np.maximum(self.pop_dyn * g_base[None, :], 1e-4)

        if self.climate_active:
            tg = sim.Tg
            self.d_glob = (
                -self.cfg.glob_lin * np.maximum(tg - 1.5, 0.0)
                - self.cfg.glob_quad * np.maximum(tg - 2.5, 0.0) ** 2
            )
            return dict(
                rho_D=self.cfg.rho_D,
                dis_kappa=self.cfg.dis_kappa,
                dis_mu=self.cfg.dis_mu,
            )
        self.d_glob.fill(0.0)
        return None

    def effective_lnA_shift(self, t: int) -> np.ndarray | float:
        if t <= self.obs_until:
            return 0.0
        shift = 0.0
        if self.climate_active:
            shift = shift + self.d_glob
        if self.cfg.food_brakes:
            shift = shift + self.d_food_tfp
        return shift

    def post_step(self, sim, rec: dict, t: int, obs_mode: bool, dclim_prev: np.ndarray,
                  act: np.ndarray, orig_p, rng: np.random.Generator = None):
        """Aplica los lazos post-paso de M1, M2 y M3 para ``t > obs_until`` (RNGs independientes por módulo para CRN exacto)."""
        R, N = sim.R, self.w.N
        rng_f = self._rng_food if self._rng_food is not None else rng
        rng_t = self._rng_tails if self._rng_tails is not None else rng
        ex_deaths = np.zeros((R, N), dtype=float)
        g_shock = np.zeros(R, dtype=float)

        if obs_mode:
            self.excess_deaths = ex_deaths
            self.global_shock_flag = g_shock
            return

        Tloc = rec["temp"]
        Tloc_safe = np.where(np.isfinite(Tloc), Tloc, self.T_origin[None, :])
        dt_inc = np.maximum(Tloc_safe - self.T_origin[None, :], 0.0)
        ly = np.where(np.isfinite(rec["ly"]), rec["ly"], 8.5)
        poor = np.clip((9.5 - ly) / 2.5, 0.0, 1.0)

        # M0: CO2 ligado al PIB con intensidad de carbono estimada <= origen + TCRE efectiva OLS + ENSO
        # Sustituye el colapso artificial de CO2 por Hubbert (2,7 Gt en 2041) y la banda nula de Tg (0 % cob.)
        if self.cfg.climate_co2:
            rng_c = self._rng_clim
            h_yr = max(t - self.obs_until, 1)
            Y_i = np.where(act & np.isfinite(rec["ly"]), np.exp(ly) * self.w.pop[:, t][None, :], 0.0)
            Y_w = np.maximum(np.sum(Y_i, axis=1, keepdims=True), 1e-6)
            # factor de agotamiento físico solo a escala total de recursos fósiles GCAM (~47.500 GtCO2)
            rem_frac = np.clip(1.0 - self.cum_co2_gt / self.cfg.gcam_carbon_cap_gt, 0.02, 1.0)
            co2_w = self.co2_0 * (Y_w / self.Y_world_0) * np.exp(self.g_ci_r * h_yr) * (rem_frac ** 0.5)
            co2_w = np.clip(co2_w, 1.0, 150.0)
            self.cum_co2_gt = self.cum_co2_gt + co2_w + self.luc_0
            # actualizar rec["co2"] (MtCO2 por país) preservando suma mundial exacta
            w_c = np.where(act, self.co2_country_share[None, :] * np.maximum(Y_i / np.maximum(self.w.Y[:, self.obs_until][None, :], 1e-3), 0.1), 0.0)
            w_c = w_c / np.maximum(np.sum(w_c, axis=1, keepdims=True), 1e-12)
            rec["co2"] = np.where(act, co2_w * 1000.0 * w_c, np.nan)
            # actualizar Tg (GMST) con respuesta forzada + variabilidad interna AR(1) (ENSO)
            self.tg_forced = self.tg_forced + self.tcre_r * (co2_w + self.luc_0) / 1000.0
            rho = self.cfg.enso_rho
            eps_t = rng_c.normal(0.0, self.enso_sd * np.sqrt(max(1.0 - rho ** 2, 0.01)), size=(R, 1))
            self.tg_anom = rho * self.tg_anom + eps_t
            sim.Tg = np.maximum(self.tg_forced + self.tg_anom, 0.5)
            sim.last_Tg = sim.Tg[:, 0].copy()

        # M1: persistencia climática con tope al bono cálido en países fríos y suelo ampliado
        if self.climate_active:
            h_loc = orig_p.bhm1 * Tloc_safe + orig_p.bhm2 * Tloc_safe ** 2
            h_bar = orig_p.bhm1 * self.Tbar_ref + orig_p.bhm2 * self.Tbar_ref ** 2
            dh = np.minimum(h_loc - h_bar[None, :], self.cfg.cold_bonus_cap)
            sim.Dclim = np.where(act, np.maximum(self.cfg.rho_D * dclim_prev + dh, self.cfg.dclim_floor), 0.0)

        # M2: frenos biofísicos alimentarios + mercado mundial -> conflicto y mortalidad aguda
        if self.cfg.food_brakes:
            ceil = self.cfg.yield_ceiling
            tech_gain = self.cfg.yield_gain_rate * np.maximum(1.0 - self._tech_idx / ceil, 0.0)
            self._tech_idx = np.clip(self._tech_idx * (1.0 + tech_gain), 0.5, ceil)

            if self._climate_ablated:
                heat_pen = 0.0
                water_pen = 0.0
                wb_sd = np.full(R, 0.025)
            else:
                heat_pen = 0.05 * np.minimum(dt_inc, 1.5) + 0.14 * np.maximum(dt_inc - 1.5, 0.0)
                water_pen = 0.012 * np.maximum(dt_inc - 1.0, 0.0) ** 1.4
                wb_sd = 0.025 + 0.020 * np.maximum(sim.Tg[:, 0] - 1.5, 0.0)

            glob_h = rng_f.standard_t(df=4, size=(R, 1)) * (wb_sd[:, None] * 0.55)
            idio_h = rng_f.normal(0.0, 0.05, size=(R, N))
            clim_mult = np.exp(-heat_pen - water_pen + np.clip(glob_h + idio_h, -0.45, 0.20))
            self.yield_idx = np.clip(self._tech_idx * clim_mult, 0.15, 1.8)

            base_prod = self.cy0[None, :] * self.pop_origin[None, :]
            world_supply = np.sum(base_prod * self.yield_idx, axis=1, keepdims=True)
            pop_ratio = np.sum(self.pop_dyn, axis=1, keepdims=True) / max(float(np.sum(self.pop_origin)), 1e-6)
            world_demand = np.sum(base_prod, axis=1, keepdims=True) * pop_ratio * (1.0 + 0.0015 * max(t - self.obs_until, 0))
            ratio_ds = np.maximum(world_demand / np.maximum(world_supply, 1e-6), 0.35)
            self.p_food = np.clip(ratio_ds ** (1.0 / self.cfg.food_elast), 0.65, 6.5)

            lp_food = np.maximum(np.log(self.p_food), 0.0)
            own_eff = (1.0 - self.import_dep[None, :]) * np.log(np.maximum(self.yield_idx, 0.15))
            price_eff = -(0.10 + 0.36 * poor + 0.22 * self.import_dep[None, :]) * lp_food
            inc_eff = 0.05 * np.clip(ly - 9.0, -2.5, 2.0)
            target_k = self.kcal0[None, :] * np.exp(np.clip(own_eff * 0.40 + price_eff + inc_eff, -0.85, 0.25))
            self.kcal = np.clip(0.55 * self.kcal + 0.45 * target_k, 1100.0, 3800.0)

            shortfall = np.maximum(np.log(2300.0 / self.kcal), 0.0)
            famine_def = np.maximum(2000.0 - self.kcal, 0.0) / 800.0
            mort_food = np.where(act, 0.08 * (famine_def ** 1.8), 0.0)
            ex_deaths = ex_deaths + self.pop_dyn * np.minimum(mort_food, 0.15)
            self.d_food_tfp = -0.60 * shortfall

            extra_conf_logit = 1.1 * poor * lp_food + 3.2 * shortfall
            p_food_conf = _sig(-3.9 + extra_conf_logit) - _sig(-3.9)
            u_fc = rng_f.random((R, N))
            new_fc = (u_fc < np.clip(p_food_conf, 0.0, 0.45)) & act
            sim.conf = np.where(new_fc, 1.0, sim.conf)
            sim.hazard = np.where(act, np.clip(sim.hazard + np.clip(p_food_conf, 0.0, 0.45), 0.0, 1.0), np.nan)

        # M3: colas pesadas Student-t(3), choques globales sincronizados y mortalidad por desastres
        if self.cfg.heavy_tails:
            t_draw = rng_t.standard_t(df=self.cfg.tail_df, size=(R, N))
            tail_shock = np.where(
                np.abs(t_draw) > self.cfg.tail_thresh,
                np.clip(t_draw * self.cfg.tail_scale, -0.22, 0.09),
                0.0,
            )
            tail_shock = np.where(tail_shock > 0, tail_shock * 0.35, tail_shock)
            sim.lnA = np.where(act, sim.lnA + tail_shock, sim.lnA)

            u_glob = rng_t.random(R)
            hit_glob = u_glob < self.cfg.global_shock_prob
            g_shock = hit_glob.astype(float)
            if hit_glob.any():
                mag = rng_t.uniform(0.03, 0.10, size=R) * hit_glob
                sim.lnA = np.where(act, sim.lnA - mag[:, None], sim.lnA)
                mort_g = rng_t.uniform(0.0015, 0.008, size=R) * hit_glob
                ex_deaths = ex_deaths + self.pop_dyn * mort_g[:, None]

            if not self._climate_ablated:
                tg = sim.Tg
                p_dis_m = 0.08 * np.exp(0.30 * np.maximum(tg - 1.2, 0.0))
                u_dm = rng_t.random((R, N))
                hit_dm = (u_dm < p_dis_m) & act
                vuln = np.clip(1.0 - np.nan_to_num(rec["core"]), 0.05, 1.0)
                sev = np.abs(rng_t.standard_t(df=3, size=(R, N))) * 0.0007 * vuln
                ex_deaths = ex_deaths + np.where(hit_dm, self.pop_dyn * np.minimum(sev, 0.04), 0.0)

        ex_deaths = np.minimum(ex_deaths, self.pop_dyn * 0.25)
        self.pop_dyn = np.maximum(self.pop_dyn - ex_deaths, 1e-4)
        self.excess_deaths = ex_deaths
        self.cum_excess_deaths = self.cum_excess_deaths + ex_deaths
        self.global_shock_flag = g_shock
