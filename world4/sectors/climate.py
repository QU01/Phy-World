"""F4: clima y carbono con FaIR v2.2.4 (conjunto calibrado y restringido 1.6.0) en lugar de la TCRE.

Cada miembro del ensemble del L1 lleva una configuración distinta del posterior de FaIR (841, sin
reemplazo). FaIR corre desde 1750 con las emisiones observadas (CMIP7) hasta el origen y, después, año
a año con las emisiones que produce el modelo (``_fair_step.Stepper``, paridad exacta con ``FAIR.run``).

Emisiones después del origen (por miembro; todo lo ajustado usa datos <= origen):

* **CO2 fósil e industrial:** el valor del origen × F(t)/F(t0) × [(1 - s_o)·I(t) + s_o]. F es el
  consumo de energía fósil de F3 (la cantidad); I(t) = Σ_c s_c r_c / Σ_c w_c r_c es la intensidad de
  carbono relativa de la mezcla de combustibles, con r_c = Q_c(t)/Q_c(t0) la producción de petróleo,
  gas y carbón de F2 (el reparto), s_c la parte de cada combustible en el CO2 del origen y w_c su parte
  en la energía fósil del origen (Global Carbon Budget y Energy Institute vía el Atlas). El resto s_o
  (cemento, antorchas, otros) sigue a F [S]. Sin mejora de intensidad dentro de cada combustible (es
  química). F2 no reparte el fósil de F3 (cada clase sigue su tendencia), así que sus cantidades
  sólo se usan como reparto: desde 1990 suman bastante menos que el fósil de F3 (declarado en el
  informe de F4).
* **CO2 de uso del suelo (AFOLU):** media de los 10 años previos más un AR(1) con la persistencia y
  la escala de los 30 años previos [S, hasta F5].
* **Especies ligadas a la energía** (SO2, BC, OC, NOx, CO, COV): el valor del origen × el CO2 fósil
  relativo × exp(g_s·h); g_s es la tendencia de ln(E_s / CO2 fósil) en los 20 años previos, acotada a
  [-6 %, +2 %] anual [S]. Es la «intensidad de contaminación» del sistema energético, que cae con
  los filtros y las normas.
* **N2O y NH3:** ídem frente a la población mundial (F1) [S, hasta F5]. **CH4:** frente a una mezcla
  35 % fósil y 65 % población [S].
* **Halogenados y gases fluorados:** el valor del origen × exp(g_s·h), con la tendencia de 10 años
  acotada a [-15 %, +5 %] [S]. El Protocolo de Montreal está en la tendencia, no en una regla.
* **Forzamientos naturales:** solar = media de los 11 años previos (un ciclo); volcánico = 0 (no se
  conocen erupciones futuras). Uso del suelo y riego: el valor del origen [S, hasta F5].

Temperatura del L1 (``Tg``, °C respecto de 1850-1900): la respuesta forzada del miembro más la
diferencia con la observación en el origen, partida en una parte lenta (media de 10 años, se
mantiene) y una rápida (el año del origen, decae con la persistencia anual estimada). Con ``noise``,
la variabilidad interna es la estocástica calibrada de FaIR (sólo después del origen: la historia es
determinista).

Elementos de inflexión (``tipping``, apagados por defecto, exploratorios y separables por ablación):
umbrales de Armstrong McKay et al. (2022) como distribuciones triangulares por miembro [S]; los
impactos en carbono son ILUSTRATIVOS [M, sin verificar ni elicitar].
"""
from __future__ import annotations

import warnings

import numpy as np

FUELS = ("oil", "gas", "coal")
ENERGY_SLCF = ("Sulfur", "BC", "OC", "NOx", "CO", "VOC")
POP_SPECIES = ("N2O", "NH3")
CH4_FOSSIL_SHARE = 0.35            # [S]
CH4_AGRI_SHARE = 0.35              # con F5: parte agrícola del CH4 antropogénico (FAOSTAT farm gate ~ 1/3) [S]
G_ENERGY = (-0.06, 0.02)           # tendencia anual de intensidad, especies energéticas [S]
G_POP = (-0.03, 0.02)
G_HALO = (-0.15, 0.05)
FIT_YEARS = 20
RHO_FAST_MAX = 0.8
# Armstrong McKay et al. (2022), Science 377:eabn7950: umbral (mínimo, central, máximo) en °C [S];
# impactos ilustrativos [M]: CO2 en GtCO2/año y CH4 en MtCH4/año una vez cruzado, con rampa de 20 años
TIPPING = {"permafrost_abrupt_thaw": {"threshold": (1.0, 1.5, 2.3), "co2": 1.0, "ch4": 10.0},
           "amazon_dieback": {"threshold": (2.0, 3.5, 6.0), "co2": 0.9, "ch4": 0.0}}
TIPPING_RAMP = 20


def _slope(y, x=None):
    y = np.asarray(y, float)
    ok = np.isfinite(y)
    if ok.sum() < 5:
        return 0.0
    x = np.arange(len(y), dtype=float) if x is None else np.asarray(x, float)
    return float(np.polyfit(x[ok], y[ok], 1)[0])


def build_fair(config_ids, end_year: int, seeds=None):
    """Objeto ``FAIR`` con el montaje de la calibración 1.6.0 (como el ejemplo oficial del conjunto
    calibrado): emisiones históricas CMIP7, forzamientos solar/volcánico/uso del suelo/riego escalados
    por miembro, propiedades de las especies y parámetros posteriores. ``seeds`` (una por config)
    sustituye la semilla de la variabilidad interna."""
    import pandas as pd
    from fair import FAIR
    from fair.interface import fill, initialise
    from fair.io import read_properties
    from world4.data import climate as cdata
    d = cdata.load()
    P = d["params"]
    ids = [int(i) for i in config_ids]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = FAIR(ch4_method="Thornhill2021")
        f.define_time(1750, end_year, 1)
        f.define_scenarios(["historical"])
        f.define_configs(ids)
        spec_file = str(cdata.raw("species_configs_properties.csv"))
        sp, pr = read_properties(filename=spec_file)
        f.define_species(sp, pr)
        f.allocate()
        em = pd.read_csv(cdata.raw("historical_emissions_1750-2023_cmip7.csv"))
        em = em.rename(columns={k: f"{int(k) + 0.5}" for k in em.columns if k.isdigit()})
        f.fill_from_pandas(mode="emissions", df=em)
        # después de 2023 no hay dato: marcador 0 que el L1 (o el experimento) sobrescribe antes de avanzar
        e = f.emissions.data
        e[(f.timepoints > cdata.EMIS_END + 1)[:, None, None, None] & np.isnan(e)] = 0.0
        for key, (yrs, v) in d["forcing"].items():
            s = np.interp(f.timebounds, yrs, v)
            fill(f.forcing, s[:, None, None] * P.loc[ids, f"forcing_scale[{key}]"].to_numpy()[None, None, :], specie=key)
        f.fill_species_configs(spec_file)
        _override(f, P.loc[ids])
        if seeds is not None:
            fill(f.climate_configs["seed"], np.asarray(seeds, int))
            fill(f.climate_configs["use_seed"], True)
        initialise(f.concentration, f.species_configs["baseline_concentration"])
        initialise(f.forcing, 0)
        initialise(f.temperature, 0)
        initialise(f.cumulative_emissions, 0)
        initialise(f.airborne_emissions, 0)
        initialise(f.ocean_heat_content_change, 0)
    return f


def _override(f, df):
    """``FAIR.override_defaults`` vectorizado (mismo resultado; la prueba de paridad lo compara): una
    asignación por columna para todas las configuraciones a la vez en lugar de una por celda."""
    from fair.io.param_sets import energy_balance_parameters
    for col in df.columns:
        name, idx = (col.split("[")[0], col.split("[")[1][:-1]) if "[" in col else (col, None)
        vals = df[col].to_numpy()
        if name in energy_balance_parameters:
            da = f.climate_configs[name]
            if idx is not None:
                da.loc[dict(layer=int(idx))] = vals
            else:
                da.loc[:] = vals.astype(da.dtype) if da.dtype == bool else vals
        else:
            if idx is not None:
                if idx not in f.species:
                    continue
                f.species_configs[name].loc[dict(specie=idx)] = vals
            else:
                f.species_configs[name].loc[:] = vals


class Climate:
    """FaIR por miembro, avanzado año a año por el L1. Índice de año ``t`` desde 1950, como el L1."""

    def __init__(self, W, t0: int, R: int, seed: int, end_t: int, noise: bool = True, tipping: bool = False):
        from world4.data import climate as cdata
        from world4.sectors._fair_step import Stepper
        self.W, self.t0, self.R, self.noise, self.tipping = W, t0, R, noise, tipping
        self.rng = np.random.default_rng(seed)
        d = cdata.load()
        P = d["params"]
        if R > len(P):
            raise ValueError(f"F4 has {len(P)} FaIR configurations; R = {R} is larger")
        self.config_ids = self.rng.choice(P.index.to_numpy(), size=R, replace=False)
        self.year0 = 1950 + t0
        end_year = 1950 + end_t + 1
        seeds = self.rng.integers(1, 2 ** 31 - 1, R)
        self.f = build_fair(self.config_ids, end_year, seeds=seeds)
        if not noise:
            from fair.interface import fill
            fill(self.f.climate_configs["stochastic_run"], False)
        self.st = Stepper(self.f)
        tp = self.f.timepoints - 0.5                     # año natural de cada punto temporal
        self.tp_year = tp.astype(int)
        self.i0 = int(np.where(self.tp_year == self.year0)[0][0])
        # la historia es determinista: variabilidad interna sólo después del origen
        self.st.stochastic_d[:self.i0 + 2] = 0.0
        sp = list(self.f.species)
        self.k = {s: sp.index(s) for s in sp}
        self.species = sp
        self.st.run_to(self.i0 + 1)                      # emisiones observadas hasta el año del origen
        T = self.st.temperature()[:, 0, :]               # (límites, R)
        yrs = self.tp_year
        ann = 0.5 * (T[1:] + T[:-1])                     # GMST del año natural = media de sus dos límites
        self.base = ann[(yrs >= 1850) & (yrs <= 1900)].mean(0)
        self._ann = lambda i: 0.5 * (self.st.temperature()[i, 0, :] + self.st.temperature()[i + 1, 0, :]) - self.base
        forced = ann[: self.i0 + 1] - self.base          # años 1750..origen
        hy = yrs[: self.i0 + 1]
        obs = cdata.gmst_observed(hy)
        resid = obs[:, None] - forced                    # (años, R)
        last10 = (hy > self.year0 - 10) & (hy <= self.year0)
        self.slow = np.nanmean(resid[last10], 0)
        self.fast = resid[-1] - self.slow
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rm = np.nanmedian(resid, 1)
        win = (hy >= max(1950, self.year0 - 39)) & (hy <= self.year0)
        x = rm[win]
        trail = np.array([np.mean(x[max(0, j - 9):j + 1]) for j in range(len(x))])
        e = x - trail
        self.rho = float(np.clip(np.corrcoef(e[1:], e[:-1])[0, 1], 0.0, RHO_FAST_MAX)) if len(e) > 5 else 0.5
        self.agri, self.k_forest, self.loss_ref = None, None, None
        self._fit_emission_rules(d)
        self.hist = {t0: {"gmst": forced[-1] + self.slow + self.fast, "co2_ffi": self._e("CO2 FFI", self.i0),
                          "co2_afolu": self._e("CO2 AFOLU", self.i0), "co2_conc": self._conc_co2(self.i0 + 1),
                          "erf": self.st.forcing_sum[self.i0 + 1, 0].copy()}}
        if tipping:
            self.thr = {k: self.rng.triangular(*v["threshold"], R) for k, v in TIPPING.items()}
            self.crossed = {k: np.full(R, -1) for k in TIPPING}

    # ------------------------------------------------------------------ reglas de emisión
    def _e(self, specie, i):
        return self.st.emis[i, 0, :, self.k[specie]].copy()

    def _conc_co2(self, ib):
        return self.st.conc[ib, 0, :, self.k["CO2"]].copy()

    def _fit_emission_rules(self, d):
        W = self.W
        y0 = self.year0
        ey = d["emis_years"]
        col = lambda s: d["emis"][s][1]  # noqa: E731
        ix = lambda y: int(y - ey[0])  # noqa: E731
        win = np.arange(y0 - FIT_YEARS + 1, y0 + 1)
        ffi = col("CO2 FFI")
        popw = np.nansum(W.pop, 0)                       # población mundial del panel (dato, <= origen)
        pidx = lambda yy: np.array([popw[y - 1950] if 1950 <= y <= y0 else np.nan for y in yy])  # noqa: E731
        self.E0 = {s: float(col(s)[ix(y0)]) for s in self.species if s in d["emis"]}
        # las reglas mueven sólo la parte ANTROPOGÉNICA: emisiones por encima de las naturales de 1750
        # (``baseline_emissions`` de FaIR); bajar de ahí sería quitar volcanes, incendios o suelos
        bl = self.st.baseline_emis
        self.b = {s: bl[:, self.k[s]].copy() for s in self.E0}
        bm = {s: float(np.mean(self.b[s])) for s in self.E0}
        anth = lambda s, yy: np.maximum(col(s)[[ix(y) for y in yy]] - bm[s], 1e-12)  # noqa: E731
        self.A0 = {s: np.maximum(self.E0[s] - self.b[s], 0.0) for s in self.E0}
        self.g = {}
        for s in ENERGY_SLCF:
            self.g[s] = float(np.clip(_slope(np.log(anth(s, win) / ffi[[ix(y) for y in win]])), *G_ENERGY))
        P = pidx(win)
        for s in POP_SPECIES:
            self.g[s] = float(np.clip(_slope(np.log(anth(s, win) / P)), *G_POP))
        Dch4 = CH4_FOSSIL_SHARE * ffi[[ix(y) for y in win]] / ffi[ix(y0)] + (1 - CH4_FOSSIL_SHARE) * P / pidx([y0])[0]
        self.g["CH4"] = float(np.clip(_slope(np.log(anth("CH4", win) / Dch4)), *G_POP))
        w10 = np.arange(y0 - 9, y0 + 1)
        self.halo = [s for s in self.species if s in d["emis"] and s not in ENERGY_SLCF + POP_SPECIES
                     + ("CH4", "CO2 FFI", "CO2 AFOLU")]
        for s in self.halo:
            v = col(s)[[ix(y) for y in w10]] - bm[s]
            self.g[s] = float(np.clip(_slope(np.log(v)), *G_HALO)) if np.all(v > 0) else 0.0
        # AFOLU: media de 10 años + AR(1) con la persistencia y la escala de 30 años (<= origen)
        af = col("CO2 AFOLU")
        a30 = af[[ix(y) for y in range(y0 - 29, y0 + 1)]]
        self.afolu_mean = float(np.mean(af[[ix(y) for y in w10]]))
        res = a30 - np.polyval(np.polyfit(np.arange(30), a30, 1), np.arange(30))
        self.afolu_rho = float(np.clip(np.corrcoef(res[1:], res[:-1])[0, 1], 0.0, 0.9))
        self.afolu_sd = float(np.std(res) * np.sqrt(1 - self.afolu_rho ** 2))
        self.afolu_dev = np.zeros(self.R)
        # partes del CO2 fósil por combustible en el origen (GCB vía el Atlas)
        wx = lambda k: float(W.wlocf(f"clima__co2_{k}_mt", y0 - 1950, np.nan))  # noqa: E731
        tot = float(W.wlocf("clima__co2_mt", y0 - 1950, np.nan))
        parts = {c: wx(c) / tot for c in FUELS}
        if not all(np.isfinite(list(parts.values()))) or not np.isfinite(tot):
            parts = {"oil": 0.4, "gas": 0.2, "coal": 0.35}
        self.s_fuel = parts
        self.s_other = max(0.0, 1.0 - sum(parts.values()))
        en = {c: float(W.wlocf(f"clima__{c}_prod_twh", y0 - 1950, np.nan)) for c in FUELS}
        tot_e = sum(en.values())
        self.w_fuel = ({c: en[c] / tot_e for c in FUELS} if np.isfinite(tot_e) and tot_e > 0
                       else {"oil": 0.38, "gas": 0.24, "coal": 0.38})
        self.P0 = float(pidx([y0])[0])
        # solar: media del último ciclo; volcánico 0; uso del suelo y riego: valor del origen
        fs = self.st.forcing
        ib0 = self.i0 + 1
        for name in ("Solar", "Volcanic", "Land use", "Irrigation"):
            if name not in self.k:
                continue
            k = self.k[name]
            if name == "Solar":
                v = fs[ib0 - 10:ib0 + 1, 0, :, k].mean(0)
            elif name == "Volcanic":
                v = np.zeros(self.R)
            else:
                v = fs[ib0, 0, :, k]
            fs[ib0 + 1:, 0, :, k] = v[None, :]

    def emissions(self, t, Q_ratio, F_ratio, P_world, extra=None):
        """Emisiones del año ``t`` (R,) por especie, desde los estados del L1 del mismo año."""
        h = t - self.t0
        ffi0 = self.E0["CO2 FFI"]
        r = {c: np.where(np.isfinite(Q_ratio[c]), Q_ratio[c], 1.0) for c in FUELS}
        carbon = sum(self.s_fuel[c] * r[c] for c in FUELS) / max(sum(self.s_fuel.values()), 1e-12)
        energy_mix = sum(self.w_fuel[c] * r[c] for c in FUELS)
        intensity = carbon / np.maximum(energy_mix, 1e-12)
        ffi = ffi0 * F_ratio * ((1 - self.s_other) * intensity + self.s_other)
        self.last_intensity = intensity
        out = {"CO2 FFI": ffi}
        X = ffi / ffi0
        PR = P_world / self.P0
        for s in ENERGY_SLCF:
            out[s] = self.b[s] + self.A0[s] * X * np.exp(self.g[s] * h)
        ag = self.agri
        for s in POP_SPECIES:
            drv = ag["nin_ratio"] if ag is not None else PR
            out[s] = self.b[s] + self.A0[s] * drv * np.exp(self.g[s] * h)
        if ag is not None:                     # F5: CH4 35 % fósil, 35 % agrícola (calorías), 30 % población
            mix = CH4_FOSSIL_SHARE * X + CH4_AGRI_SHARE * ag["food_ratio"] + (1 - CH4_FOSSIL_SHARE - CH4_AGRI_SHARE) * PR
        else:
            mix = CH4_FOSSIL_SHARE * X + (1 - CH4_FOSSIL_SHARE) * PR
        out["CH4"] = self.b["CH4"] + self.A0["CH4"] * mix * np.exp(self.g["CH4"] * h)
        for s in self.halo:
            out[s] = self.b[s] + self.A0[s] * np.exp(self.g[s] * h)
        z = self.rng.standard_normal(self.R) if self.noise else np.zeros(self.R)
        self.afolu_dev = self.afolu_rho * self.afolu_dev + self.afolu_sd * z
        out["CO2 AFOLU"] = self.afolu_mean + self.afolu_dev
        if ag is not None and ag.get("forest_loss") is not None and self.k_forest is not None:
            out["CO2 AFOLU"] = out["CO2 AFOLU"] + self.k_forest * (ag["forest_loss"] - self.loss_ref)
        if extra:
            for s, v in extra.items():
                out[s] = out[s] + v
        return out

    def set_agri(self, food_world: dict, forest: dict | None = None):
        """F5: N2O y NH3 siguen a las entradas de N de las tierras de cultivo del mundo y el CH4 agrícola al
        suministro de calorías (FAOSTAT, mundo); se reestiman sus tendencias de intensidad con datos <=
        origen. Con ``forest`` (pérdida de bosque y CO2 de conversión de bosque de FAOSTAT, 1991-origen),
        el CO2 de uso del suelo suma k·(pérdida - pérdida media de 10 años) [S]."""
        y0 = self.year0
        d = __import__("world4.data.climate", fromlist=["load"]).load()
        ey = d["emis_years"]
        col = lambda s: d["emis"][s][1]  # noqa: E731
        win = np.arange(y0 - FIT_YEARS + 1, y0 + 1)
        bm = {s: float(np.mean(self.b[s])) for s in self.b}
        anth = lambda s: np.maximum(col(s)[[int(y - ey[0]) for y in win]] - bm[s], 1e-12)  # noqa: E731
        nin, kc = food_world["n_input"], food_world["food_kcal_total"]          # arrays por año 1950+
        idx = lambda a: np.array([a[y - 1950] for y in win])  # noqa: E731
        for s in POP_SPECIES:
            self.g[s] = float(np.clip(_slope(np.log(anth(s) / idx(nin))), *G_POP))
        ffi = col("CO2 FFI")
        X = np.array([ffi[int(y - ey[0])] for y in win]) / ffi[int(y0 - ey[0])]
        popw = food_world["pop"]
        D = (CH4_FOSSIL_SHARE * X + CH4_AGRI_SHARE * idx(kc) / kc[y0 - 1950]
             + (1 - CH4_FOSSIL_SHARE - CH4_AGRI_SHARE) * idx(popw) / popw[y0 - 1950])
        self.g["CH4"] = float(np.clip(_slope(np.log(anth("CH4") / D)), *G_POP))
        self.agri = {"nin_ratio": 1.0, "food_ratio": 1.0, "forest_loss": None}
        self.k_forest = self.loss_ref = None
        if forest is not None and len(forest["loss"]) >= 8 and np.sum(forest["loss"]) > 0:
            self.k_forest = float(np.sum(forest["co2_gt"]) / np.sum(forest["loss"]))      # GtCO2 por 1000 ha
            self.loss_ref = float(np.mean(forest["loss"][-10:]))

    def _tipping_extra(self, t):
        T = self.hist[max(self.hist)]["gmst"]
        co2, ch4 = np.zeros(self.R), np.zeros(self.R)
        for k, v in TIPPING.items():
            new = (self.crossed[k] < 0) & (T >= self.thr[k])
            self.crossed[k] = np.where(new, t, self.crossed[k])
            ramp = np.where(self.crossed[k] >= 0, np.clip((t - self.crossed[k]) / TIPPING_RAMP, 0, 1), 0.0)
            co2 += v["co2"] * ramp
            ch4 += v["ch4"] * ramp
        return {"CO2 AFOLU": co2, "CH4": ch4}

    def step(self, t, Q_ratio, F_ratio, P_world, agri=None):
        """Avanza el año ``t`` (> origen) y devuelve la GMST (R,) respecto de 1850-1900."""
        i = self.i0 + (t - self.t0)
        extra = self._tipping_extra(t) if self.tipping else None
        if agri is not None and self.agri is not None:
            self.agri.update(agri)
        em = self.emissions(t, Q_ratio, F_ratio, P_world, extra)
        for s, v in em.items():
            self.st.emis[i, 0, :, self.k[s]] = v
        self.st.step(i)
        h = t - self.t0
        g = self._ann(i) + self.slow + self.fast * self.rho ** h
        self.hist[t] = {"gmst": g, "co2_ffi": em["CO2 FFI"].copy(), "co2_afolu": em["CO2 AFOLU"].copy(),
                        "co2_conc": self._conc_co2(i + 1), "erf": self.st.forcing_sum[i + 1, 0].copy()}
        return g

    def carbon_accounts(self):
        """Cierre de masa del CO2 (W7): acumulado emitido = en la atmósfera + absorbido, y concentración
        sobre la base = carga atmosférica × concentración por unidad emitida."""
        k = self.k["CO2"]
        i = self.st.i
        cum = self.st.cum[i, 0, :, k]
        air = self.st.airborne[i, 0, :, k]
        parts = self.st.partitions[0, :, k, :].sum(-1)
        conc = self.st.conc[i, 0, :, k]
        base = self.st.baseline_conc[:, k]
        cpe = self.st.conc_per_emis[k]
        return {"cumulative": cum, "airborne": air, "uptake": cum - air, "boxes": parts,
                "conc_excess": conc - base, "airborne_as_conc": air * cpe}


# ---------------------------------------------------------------------------------- AR6 (coherencia)
# AR6 WG1 (Cap. 5, tabla 5.8; SPM D.1): TCRE probable 1,0-2,3 °C por 1000 PgC (0,27-0,63 por 1000 GtCO2),
# mejor estimación 1,65 (0,45). Presupuestos restantes desde el 1 de enero de 2020, en GtCO2, por
# probabilidad de quedar por debajo (17, 33, 50, 67, 83 %).
AR6_TCRE_LIKELY = (1.0 / 3.664, 2.3 / 3.664)
AR6_BUDGET = {1.5: {17: 900, 33: 650, 50: 500, 67: 400, 83: 300},
              2.0: {17: 2300, 33: 1700, 50: 1350, 67: 1150, 83: 900}}


def tcre(config_ids=None) -> dict:
    """TCRE de cada configuración en el experimento 1pctCO2 (concentración +1 %/año, sin otros
    forzamientos): calentamiento en el año 70 (duplicación) / CO2 acumulado diagnosticado, °C por 1000
    GtCO2. Corre con ``FAIR.run`` (el modo dirigido por concentración no está en ``Stepper``)."""
    from fair import FAIR
    from fair.interface import fill, initialise
    from fair.io import read_properties
    from world4.data import climate as cdata
    P = cdata.load()["params"]
    ids = [int(i) for i in (P.index if config_ids is None else config_ids)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = FAIR()
        f.define_time(0, 80, 1)
        f.define_scenarios(["1pctCO2"])
        f.define_configs(ids)
        sp, pr = read_properties(filename=str(cdata.raw("species_configs_properties.csv")),
                                 species=["CO2", "CH4", "N2O"])
        for s in sp:
            pr[s]["input_mode"] = "concentration"
        f.define_species(sp, pr)
        f.allocate()
        f.fill_species_configs(str(cdata.raw("species_configs_properties.csv")))
        _override(f, P.loc[ids])
        fill(f.climate_configs["stochastic_run"], False)
        c0 = f.species_configs["baseline_concentration"]
        for s in sp:
            base = c0.loc[dict(specie=s)].data
            grow = 1.01 ** f.timebounds if s == "CO2" else np.ones_like(f.timebounds)
            fill(f.concentration, grow[:, None, None] * base[None, None, :], specie=s)
        initialise(f.forcing, 0)
        initialise(f.temperature, 0)
        initialise(f.cumulative_emissions, 0)
        initialise(f.airborne_emissions, 0)
        f.run(progress=False)
    T = f.temperature.loc[dict(scenario="1pctCO2", layer=0)].data
    cum = f.cumulative_emissions.loc[dict(scenario="1pctCO2", specie="CO2")].data
    k = int(np.where(f.timebounds == 70)[0][0])
    v = T[k] / cum[k] * 1000.0
    return {"tcre": v, "p": {q: float(np.percentile(v, q)) for q in (5, 17, 50, 83, 95)},
            "ar6_likely": AR6_TCRE_LIKELY, "ar6_best": 0.45}


# variante «como las trayectorias de AR6» del experimento de presupuesto [S, ilustrativa]: en el año de
# CO2 neto cero el CH4 antropogénico ha caído un 50 % y el N2O y el NH3 un 20 %
AR6_LIKE_CUTS = {"CH4": 0.5, "N2O": 0.2, "NH3": 0.2}


def remaining_budget(config_ids=None, years=(10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130),
                     non_co2: str = "f4") -> dict:
    """Presupuesto de carbono restante desde 2020 con las reglas de emisión de F4: historia observada
    hasta 2019; desde 2020 el CO2 fósil y el de uso del suelo caen en línea recta a 0 en N años; las
    especies energéticas siguen al CO2 fósil y la población queda en la de 2019 (``non_co2="f4"``, las
    reglas del modelo) o, con ``"ar6_like"``, el CH4, el N2O y el NH3 caen además como en
    ``AR6_LIKE_CUTS``. Para cada configuración
    se interpola el CO2 acumulado desde 2020 con el que el pico de calentamiento toca 1,5 y 2 °C; el
    presupuesto con probabilidad p de quedar por debajo es el cuantil (100 - p) de esas cifras."""
    from world4.data import climate as cdata
    from world4.sectors._fair_step import Stepper
    P = cdata.load()["params"]
    ids = list(P.index if config_ids is None else config_ids)
    end = 2200
    peaks, cums = [], []
    for N in years:
        f = build_fair(ids, end)
        from fair.interface import fill
        fill(f.climate_configs["stochastic_run"], False)
        st = Stepper(f)
        yrs = (f.timepoints - 0.5).astype(int)
        i19 = int(np.where(yrs == 2019)[0][0])
        st.run_to(i19 + 1)
        sp = list(f.species)
        k = {s: sp.index(s) for s in sp}
        fs = st.forcing
        for name in ("Solar", "Volcanic", "Land use", "Irrigation"):
            v = fs[i19 - 9:i19 + 2, 0, :, k[name]].mean(0) if name == "Solar" else \
                (np.zeros(len(ids)) if name == "Volcanic" else fs[i19 + 1, 0, :, k[name]])
            fs[i19 + 2:, 0, :, k[name]] = v[None, :]
        E19 = {s: st.emis[i19, 0, :, k[s]].copy() for s in sp if f.properties_df.loc[s, "input_mode"] == "emissions"}
        halo = [s for s in E19 if s not in ENERGY_SLCF + POP_SPECIES + ("CH4", "CO2 FFI", "CO2 AFOLU")]
        B = {s: st.baseline_emis[:, k[s]] for s in E19}                  # emisiones naturales (1750)
        cum = np.zeros(len(ids))
        for i in range(i19 + 1, len(yrs)):
            h = yrs[i] - 2019
            x = max(0.0, 1.0 - h / N)
            st.emis[i, 0, :, k["CO2 FFI"]] = E19["CO2 FFI"] * x
            st.emis[i, 0, :, k["CO2 AFOLU"]] = E19["CO2 AFOLU"] * x
            for s in ENERGY_SLCF:
                st.emis[i, 0, :, k[s]] = B[s] + (E19[s] - B[s]) * x
            cut = (lambda sp_: 1.0 - AR6_LIKE_CUTS[sp_] * (1.0 - x)) if non_co2 == "ar6_like" else (lambda sp_: 1.0)
            for s in POP_SPECIES:
                st.emis[i, 0, :, k[s]] = B[s] + (E19[s] - B[s]) * cut(s)
            ch4 = (CH4_FOSSIL_SHARE * x + 1 - CH4_FOSSIL_SHARE)
            if non_co2 == "ar6_like":
                ch4 = min(ch4, cut("CH4"))
            st.emis[i, 0, :, k["CH4"]] = B["CH4"] + (E19["CH4"] - B["CH4"]) * ch4
            for s in halo:
                st.emis[i, 0, :, k[s]] = B[s] + (E19[s] - B[s]) * x
            cum += st.emis[i, 0, :, k["CO2 FFI"]] + st.emis[i, 0, :, k["CO2 AFOLU"]]
            st.step(i)
        T = st.temperature()[:, 0, :]
        ann = 0.5 * (T[1:] + T[:-1])
        ann = ann - ann[(yrs >= 1850) & (yrs <= 1900)].mean(0)
        peaks.append(ann[yrs >= 2020].max(0))
        cums.append(cum.copy())
    peaks, cums = np.array(peaks), np.array(cums)          # (N, configs)
    out = {"pathway_years": list(years), "non_co2": non_co2, "budgets": {}}
    for lim in (1.5, 2.0):
        b = np.full(len(ids), np.nan)
        for j in range(len(ids)):
            pk, cu = peaks[:, j], cums[:, j]
            if pk[0] >= lim:                                # ni la caída más rápida queda por debajo
                b[j] = -np.inf
            elif pk[-1] <= lim:                             # ni la más lenta lo alcanza
                b[j] = np.inf
            else:
                b[j] = float(np.interp(lim, pk, cu))
        qs = {p: float(np.percentile(b, 100 - p, method="nearest")) for p in (17, 33, 50, 67, 83)}
        out["budgets"][str(lim)] = {"gtco2_by_probability": qs, "ar6": AR6_BUDGET[lim],
                                    "share_below_fastest": float(np.mean(b == -np.inf)),
                                    "share_above_slowest": float(np.mean(b == np.inf))}
    return out
