"""Acoplamiento de los sectores endógenos al L1 sin tocar el port del Atlas (W2 sigue bit a bit).

El ``Simulator`` del Atlas lee la población y el capital humano de ``world.pop[:, t]`` y
``world.hc[:, t]`` (arrays exógenos (N, T)). ``CoupledSimulator`` le da en su lugar un
``MemberSeries``: hasta el origen devuelve el dato; después, el estado (R, N) de cada miembro que
producen los sectores. Todas las operaciones del Atlas sobre esas series son elemento a elemento
y difunden (N,) -> (R, N), salvo una que cambia de significado y se declara:

    * la frontera de capital humano ``hq`` (cuantil 90 de ``hc`` entre países) pasa a calcularse
      sobre todos los miembros juntos en lugar de sobre un único vector (misma escala).

Los sectores usan su propio generador aleatorio: los números comunes del L1 no cambian, así que
con los sectores apagados ``CoupledSimulator`` es idéntico al ``Simulator`` (prueba P3 en tests).

Orden temporal: el estado del año ``t`` se calcula con los resultados del L1 del año ``t-1``
(``sim.last``) al final del paso ``t-1`` (antes de activar países nuevos) o, si no, al empezar ``t``.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from world4.sectors import human_capital, population, resources

POP_MODES = ("observed", "trend", "endogenous")


@dataclass(frozen=True)
class F1Config:
    population: bool = True        # cohortes endógenas (si no, población observada: la fuga D1)
    human_capital: bool = True     # hc endógeno (si no, hc observado: la fuga D2)
    youth_bulge: bool = True       # el bulto juvenil del MMP sale de las cohortes
    noise: bool = True             # incertidumbre paramétrica y residual por miembro
    drivers: str = "model"         # "observed": TFR/e0/migración del dato (sólo pruebas de contabilidad)


class MemberSeries:
    """Sustituto de un array (N, T): dato hasta ``origin``; estado (R, N) por miembro después."""

    def __init__(self, hist: np.ndarray, origin: int, states: dict):
        self.hist, self.origin, self.states = hist, origin, states
        self.shape = hist.shape

    def __getitem__(self, key):
        rows, t = key
        t = int(t)
        if t <= self.origin:
            return self.hist[rows, t]
        return self.states[t][:, rows]

    def series(self, R: int) -> np.ndarray:
        """(R, N, T) con el dato hasta el origen y el estado después (NaN donde no se calculó)."""
        out = np.broadcast_to(self.hist[None], (R,) + self.hist.shape).astype(float).copy()
        out[:, :, self.origin + 1:] = np.nan
        for t, v in self.states.items():
            if t > self.origin:
                out[:, :, t] = v
        return out


@dataclass(frozen=True)
class F2Config:
    markets: bool = True           # psi_t de los mercados por clase (si no, la del Atlas: congelada, D3)
    noise: bool = True
    u_scale: float = 1.0           # multiplica el recurso último (relaciones metamórficas, W6)
    persist_psi: bool = False      # ablación: psi congelada en su valor del origen (sin mercados), normalizada


@dataclass(frozen=True)
class F3Config:
    energy: bool = True            # mezcla energética, EROI y materiales (requiere F2)
    noise: bool = True


@dataclass(frozen=True)
class F4Config:
    climate: bool = True           # FaIR calibrado con emisiones endógenas en lugar de la TCRE (requiere F3)
    noise: bool = True             # variabilidad interna estocástica de FaIR después del origen
    tipping: bool = False          # elementos de inflexión (exploratorio; ablación)


@dataclass(frozen=True)
class F5Config:
    food: bool = True              # alimentos, suelo, N y agua; huella y biocapacidad calculadas (requiere F4)
    noise: bool = True
    mortality: bool = True         # el déficit calórico entra en la e0 de F1 (si F1 está activo)


# F7 (calib.l1): perillas del L1 calibrado. Cada una desplaza el intercepto de una o varias ecuaciones
# sectoriales en una cantidad por miembro (R,): el ``offset`` (N,) pasa a (R, N). Con todas a 0 el
# resultado es idéntico al del L1 sin calibrar (prueba en tests).
L1_KNOBS = {
    "tfr": ("pop_s", ("tfr",), "d ln TFR per year"),
    "e0": ("pop_s", ("e0",), "d e0 per year (years)"),
    "intensity": ("en", ("intensity",), "d ln(energy / GDP) per year"),
    "nonfossil": ("en", ("nuclear", "hydro", "vre", "other"), "d logit(non-fossil source shares) per year"),
    "yield": ("fd", ("yield",), "d ln(cereal yield) per year"),
    "cropland": ("fd", ("cropland",), "d ln(cropland) per year"),
}


def _shifted(eq, d):
    return dataclasses.replace(eq, offset=np.asarray(eq.offset, float)[None, :] + np.asarray(d, float)[:, None])


_FITS: dict = {}
_MFITS: dict = {}
_EFITS: dict = {}
_FFITS: dict = {}


def food_fits(Wd, LYd, t0: int) -> dict:
    """Ecuaciones de F5 y la e0 con déficit calórico, con datos <= t0 (en caché por origen)."""
    from world4.sectors import food
    key = (id(Wd), t0)
    if key not in _FFITS:
        eqs = food.fit(Wd, LYd, t0)
        p = food.panel(Wd, LYd)
        short = np.maximum(np.log(food.KCAL_MIN) - p["lk"], 0.0)
        eqs["_e0_food"] = population.fit_e0_food(Wd, LYd, t0, short)
        _FFITS[key] = eqs
    return _FFITS[key]


def food_world_history(Wd, LYd):
    """Series mundiales (T,) para las emisiones agrícolas de F4 con F5 y la pérdida de bosque por año."""
    from world4.sectors import food
    p = food.panel(Wd, LYd)
    w = p["src"]["world"]
    popw = np.nansum(Wd.pop, 0)
    F = p["F"]
    Fl = food._lag(F)
    both = np.isfinite(F) & np.isfinite(Fl)
    loss = np.nansum(np.where(both, np.maximum(Fl - F, 0.0), 0.0), 0)
    co2 = np.nansum(p["src"]["co2_forest_conv"], 0) / 1e6          # kt -> Gt
    has = both.sum(0) > 50
    return {"n_input": w["nb_input"], "food_kcal_total": w["kcal"] * popw, "pop": popw,
            "forest_loss": np.where(has, loss, np.nan),
            "forest_co2_gt": np.where(np.isfinite(p["src"]["co2_forest_conv"]).sum(0) > 50, co2, np.nan)}


def fits(Wd, LYd, t0: int) -> dict:
    """Ecuaciones de F1 estimadas con datos <= t0 (en caché por origen)."""
    key = (id(Wd), t0)
    if key not in _FITS:
        eqs = population.fit(Wd, LYd, t0)
        eqs["schooling"] = human_capital.fit(Wd, LYd, t0)
        _FITS[key] = eqs
    return _FITS[key]


def fossil_driver(Wd):
    """Crecimiento del consumo fósil mundial (T,) del panel de energía de F3."""
    from world4.sectors import energy
    fw = energy.panel(Wd)["fossil_world"]
    g = np.full(len(fw), np.nan)
    g[1:] = np.log(fw[1:] / fw[:-1])
    return g


def market_fits(Wd, t0: int, f3: bool = False) -> dict:
    """Ecuaciones de F2 (mercados por clase) estimadas con datos <= t0 (en caché por origen). Con
    ``f3``, la demanda fósil se escribe sobre el consumo de energía fósil (F3) en lugar del PIB."""
    key = (id(Wd), t0, f3)
    if key not in _MFITS:
        _MFITS[key] = resources.fit(Wd, t0, fossil_driver=fossil_driver(Wd) if f3 else None)
    return _MFITS[key]


def energy_fits(Wd, t0: int) -> dict:
    from world4.sectors import energy
    key = (id(Wd), t0)
    if key not in _EFITS:
        _EFITS[key] = energy.fit(Wd, t0)
    return _EFITS[key]


def make_simulator(model):
    """Clase ``CoupledSimulator`` sobre el ``model.Simulator`` del Atlas portado."""

    class CoupledSimulator(model.Simulator):
        def __init__(self, world, params, R=8, seed=0, f1: F1Config | None = None, data=None,
                     f2: F2Config | None = None, f3: F3Config | None = None, f4: F4Config | None = None,
                     f5: F5Config | None = None, calib: dict | None = None):
            super().__init__(world, params, R=R, seed=seed)
            calib = {k: np.broadcast_to(np.asarray(v, float), (R,)) for k, v in (calib or {}).items()}
            bad = set(calib) - set(L1_KNOBS)
            if bad:
                raise KeyError(f"unknown L1 calibration knobs {sorted(bad)}")
            self.calib = calib
            if f3 is not None and f3.energy and (f2 is None or not f2.markets):
                raise ValueError("F3 (energy) needs F2 (markets): the fossil fuels are F2's")
            if f4 is not None and f4.climate and (f3 is None or not f3.energy):
                raise ValueError("F4 (climate) needs F3 (energy): the emissions come from the energy mix")
            if f5 is not None and f5.food and (f4 is None or not f4.climate):
                raise ValueError("F5 (food) needs F4 (climate): agricultural emissions and yields need its climate")
            self.f1, self.f2, self.f3, self.f4, self.f5 = f1, f2, f3, f4, f5
            self.en = self.cl = self.fd = None
            self.data = data                      # (W, LY) del panel para ajustar (por defecto, world)
            self.world0 = world
            self.pop_s = self.hc_s = self.mk = None
            self.origin = None
            self._G = {}
            self.last_psi = None

        # --------------------------------------------------------- sectores
        def _apply_calib(self, attr):
            """Desplaza las ecuaciones del sector ``attr`` según las perillas de F7."""
            sector = getattr(self, attr)
            for k, d in self.calib.items():
                where, names, _ = L1_KNOBS[k]
                if where != attr or sector is None:
                    continue
                sector.eqs = {**sector.eqs, **{n: _shifted(sector.eqs[n], d) for n in names}}
                if k == "e0" and getattr(sector, "food_eq", None) is not None:
                    sector.food_eq = _shifted(sector.food_eq, d)

        def _sector_seed(self):
            return int(self.rng.bit_generator.state["state"]["state"] % (2 ** 31)) + 1_000_003

        def _setup(self, t0):
            from world4.data import wpp
            Wd, LYd = self._data()
            f1 = self.f1
            seed = self._sector_seed()
            eqs = fits(Wd, LYd, t0)
            wppa = wpp.aligned(Wd.isonum)
            self.pop_s = population.Population(Wd, LYd, t0, eqs, wppa, self.R, seed, drivers=f1.drivers,
                                               noise=f1.noise)
            self.hc_s = (human_capital.HumanCapital(Wd, LYd, t0, eqs["schooling"], self.pop_s, self.R, seed + 1,
                                                    noise=f1.noise) if f1.human_capital else None)
            self.eqs = eqs
            self._apply_calib("pop_s")
            p = population.panel(Wd, LYd)
            self._tbar = np.where(np.isfinite(p["tbar"]), p["tbar"], 15.0)
            self._sig = np.clip(Wd.locf("tmp_anom_fao_sd", 0.31)[:, t0], 0.15, 1.0)
            self._ly_fb = np.where(np.isfinite(LYd[:, t0]), LYd[:, t0], np.nanmedian(LYd[:, t0]))
            prev = np.where(np.isfinite(LYd[:, t0 - 1]), LYd[:, t0 - 1], self._ly_fb)
            self._ly_hist = {t0 - 1: np.broadcast_to(prev, (self.R, Wd.N)).copy()}
            w = self.world0
            pop = MemberSeries(w.pop, t0, self.pop_s.hist_pop) if f1.population else w.pop
            hc = MemberSeries(w.hc, t0, self.hc_s.hist) if self.hc_s is not None else w.hc
            self.w = dataclasses.replace(w, pop=pop, hc=hc, _locf=w._locf)

        def _setup_markets(self, t0):
            Wd, _ = self._data()
            f2 = self.f2
            with_f3 = self.f3 is not None and self.f3.energy
            self.mk = resources.Markets(Wd, t0, market_fits(Wd, t0, with_f3), self.R, self._sector_seed() + 7,
                                        noise=f2.noise, u_scale=f2.u_scale,
                                        fossil_driver=fossil_driver(Wd) if with_f3 else None)
            if f2.persist_psi:                    # ablación: sin mercados, psi del origen para siempre
                mk = self.mk
                mk.step = lambda t, gY, mk=mk: mk.hist.setdefault(t, mk.hist[mk.t0].copy())

        def _setup_energy(self, t0):
            from world4.sectors import energy
            Wd, _ = self._data()
            self.en = energy.Energy(Wd, t0, energy_fits(Wd, t0), self.R, self._sector_seed() + 11,
                                    noise=self.f3.noise)
            self._apply_calib("en")
            self.en.set_depletion(self.mk.depletion())
            # precio de cobre y níquel en el año base del índice de metales (unidades de volumen)
            D = resources.data(Wd)
            m = resources.metals_at(D, t0)
            b = m["base"] - 1950
            raw = D["metals_raw"]
            self._metal_price = {"copper": float(raw["cobre"][1][b]), "nickel": float(raw["niquel"][1][b])}

        def _setup_climate(self, t0):
            from world4.sectors import climate
            Wd, _ = self._data()
            self.cl = climate.Climate(Wd, t0, self.R, self._sector_seed() + 13, end_t=self.w.T, noise=self.f4.noise,
                                      tipping=self.f4.tipping)
            self._F0 = np.maximum(self.en.fossil_world()[:, 0], 1e-9)
            self._Q0 = {c: self.mk.quant[t0][c] for c in climate.FUELS}

        def _setup_food(self, t0):
            from world4.sectors import food
            Wd, LYd = self._data()
            eqs = food_fits(Wd, LYd, t0)
            self.fd = food.Food(Wd, LYd, t0, {k: v for k, v in eqs.items() if k != "_e0_food"}, self.R,
                                self._sector_seed() + 17, noise=self.f5.noise)
            self._apply_calib("fd")
            if self.pop_s is not None and self.f5.mortality:
                self.pop_s.enable_food(eqs["_e0_food"])
                if "e0" in self.calib:
                    self.pop_s.food_eq = _shifted(self.pop_s.food_eq, self.calib["e0"])
                self.pop_s.food_short = self.fd.shortfall()
            fw = food_world_history(Wd, LYd)
            yrs = np.arange(1991 - 1950, t0 + 1)
            ok = [t for t in yrs if np.isfinite(fw["forest_loss"][t]) and np.isfinite(fw["forest_co2_gt"][t])]
            forest = ({"loss": fw["forest_loss"][ok], "co2_gt": fw["forest_co2_gt"][ok]}
                      if self.fd.forest_on and len(ok) >= 8 else None)
            if self.cl is not None:
                self.cl.set_agri(fw, forest)
            w0 = self.fd.world()
            self._fd0 = {"n_input": w0["n_input"], "food": w0["food_kcal_total"]}
            self._ly_food = None

        def _food_step(self, t, rec):
            ly = rec["ly"]
            prev = self._ly_food if self._ly_food is not None else ly
            self._ly_food = np.where(np.isfinite(ly), ly, prev)
            pop = np.broadcast_to(self.w.pop[:, t], (self.R, self.w.N))
            fossil_pc = None
            if self.en is not None:
                fossil_pc = self.en.generation()["fossil"] / np.maximum(pop, 1e-9)
            F_before = np.nan_to_num(self.fd.F).copy()
            self.fd.step(t, ly, ly - prev, rec["temp"], pop, fossil_pc)
            if self.pop_s is not None and self.f5.mortality:
                self.pop_s.food_short = self.fd.shortfall()
            w = self.fd.world()
            loss = np.sum(np.where(self.fd.ok[None], np.maximum(F_before - np.nan_to_num(self.fd.F), 0.0), 0.0), 1)
            return {"nin_ratio": w["n_input"] / np.maximum(self._fd0["n_input"], 1e-9),
                    "food_ratio": w["food_kcal_total"] / np.maximum(self._fd0["food"], 1e-9),
                    "forest_loss": loss if self.fd.forest_on else None}

        def _climate_step(self, t, agri=None):
            """Emisiones del año t desde F1-F3 -> FaIR -> GMST del año t (la usa el L1 desde t+1)."""
            from world4.sectors import climate
            Q = {c: self.mk.quant[t][c] / self._Q0[c] for c in climate.FUELS}
            F = np.maximum(self.en.fossil_world()[:, 0], 1e-9) / self._F0
            pop = self.w.pop[:, t]
            P = np.nansum(np.broadcast_to(pop, (self.R, self.w.N)), 1)
            g = self.cl.step(t, Q, F, P, agri=agri)
            self.Tg = g[:, None].copy()
            self.last_Tg = g.copy()

        def _gdp(self, t):
            """PIB (R, N) del año t según el L1: exp(ly) * población."""
            return np.exp(self.last["ly"]) * self.w.pop[:, t]

        def _advance(self, t):
            if self.origin is None or t <= self.origin:
                return
            if self.mk is not None and t not in self.mk.hist:
                G1 = self._gdp(t - 1)
                if t - 2 < self.origin:
                    # El PIB del origen está reanclado al dato y el del año anterior es historia simulada
                    # sin reanclar: su cociente es el salto del reanclaje, no crecimiento. Para ese año se
                    # usa el crecimiento OBSERVADO de cada país (dato <= origen).
                    Yd = self._data()[0].Y
                    gobs = Yd[:, self.origin - 1] / Yd[:, self.origin]
                    self._G[t - 2] = np.where(np.isfinite(gobs), G1 * gobs[None], G1)
                G0 = self._G.get(t - 2)
                if G0 is None:
                    gY = np.zeros(self.R)
                else:
                    both = np.isfinite(G1) & np.isfinite(G0)
                    gY = np.log(np.sum(np.where(both, G1, 0), 1)
                                / np.maximum(np.sum(np.where(both, G0, 0), 1), 1e-12))
                self._G[t - 1] = G1
                gY = np.clip(gY, -0.2, 0.2)
                if self.en is not None:
                    gYi = np.where(np.isfinite(G1) & np.isfinite(G0) & (G0 > 0), np.log(np.maximum(G1, 1e-12)
                                   / np.maximum(G0, 1e-12)), gY[:, None]) if G0 is not None else \
                        np.broadcast_to(gY[:, None], G1.shape)
                    F_prev = self.en.fossil_world()
                    self.en.step(t, gYi, self.mk.log_fossil_price(), self.mk.depletion())
                    gF = np.log(np.maximum(self.en.fossil_world(), 1e-9) / np.maximum(F_prev, 1e-9))[:, 0]
                    mat = self.en.materials[t]
                    extra = (mat["copper"] * self._metal_price["copper"] + mat["nickel"] * self._metal_price["nickel"])[:, 0]
                    self.mk.step(t, gY, gF=np.clip(gF, -0.3, 0.3), extra_metals=extra)
                else:
                    self.mk.step(t, gY)
            if self.pop_s is None or t in self.pop_s.hist_pop:
                return
            last = self.last
            ly = last["ly"]
            ly = np.where(np.isfinite(ly), ly, self._ly_fb[None])
            prev = self._ly_hist.get(t - 2)
            dly = ly - prev if prev is not None else np.zeros_like(ly)
            dly = np.where(np.isfinite(dly), np.clip(dly, -0.5, 0.5), 0.0)
            self._ly_hist[t - 1] = ly
            conf = np.nan_to_num(last["conflict"])
            temp = last["temp"]
            heat = np.where(np.isfinite(temp), np.maximum(temp - self._tbar[None], 0) / self._sig[None], 0.0)
            P = self.pop_s.total()
            act = np.isfinite(last["ly"])
            wgt = np.where(act, P, 0.0)
            lyw = np.sum(ly * wgt, 1, keepdims=True) / np.maximum(wgt.sum(1, keepdims=True), 1e-12)
            school = self.hc_s.adult() if self.hc_s is not None else \
                np.broadcast_to(np.nan_to_num(population.panel(*self._data())["s"][:, self.origin]), ly.shape)
            self.pop_s.advance(t, ly, dly, conf, heat, ly - lyw, school)
            if self.hc_s is not None:
                r = np.nan_to_num(last["r"]) / max(self.p.r_scale, 1e-9)
                self.hc_s.advance(t, ly, r)

        def _data(self):
            return self.data if self.data is not None else (self.world0, np.log(self.world0.Y / self.world0.pop))

        # --------------------------------------------------------- ganchos del Atlas
        def _activate(self, t):
            self._advance(t)
            return super()._activate(t)

        def _normalise_psi(self, t):
            """Desplazamiento de TFP (R, N) que neutraliza el efecto de UNIDADES de cambiar el exponente
            psi de la Cobb-Douglas del Atlas (insumos en niveles absolutos: +0.01 de psi subía el PIB
            per cápita de todos los países ~5 %). Con él, a insumos dados la producción no cambia
            respecto de la psi del Atlas (psi_bar) y psi sólo actúa por el precio del recurso: más
            renta para quien lo extrae, más coste para quien lo compra (normalización local de la
            función de producción, De La Grandville). Es 0 cuando psi == psi_bar."""
            psi = self.mk.hist[t]                                   # (R, 1)
            pbar = float(super()._psi(t))
            w = self.w
            hL = np.log(np.maximum(w.hc[:, t] * w.pop[:, t], 1e-12))
            lnB0 = (self._effective_lnA() + model.ALPHA * np.log(np.maximum(self.K, 1e-6)) + (1 - model.ALPHA) * hL
                    + np.log(np.maximum(1 - self.p.gam_c * self.conf, 1e-6)))
            psi_prev = self.last_psi if self.last_psi is not None else float(super()._psi(t - 1))
            lnk = np.log(np.maximum(psi_prev / np.maximum(self.price * self.mR, 1e-12), 1e-12))
            ref = (lnB0 - pbar * hL + pbar * lnk) / (1 - pbar)
            d = (1 - psi) * ref - lnB0 + psi * hL - psi * lnk
            return np.where(self.active, np.nan_to_num(d), 0.0)

        def step(self, actions):
            t = self.t
            self._advance(t)
            d = None
            if self.mk is not None and t > self.origin:
                d = self._normalise_psi(t)
                if self.en is not None:          # F3: coste real del sistema energético y escasez física
                    d = d + self.en.tfp_shift() + self.en.shortage_shift(self.mk.shortage())
                self.lnA = self.lnA + d
                F_old = self.lnA_F.copy()
            rec = super().step(actions)
            if d is not None:
                self.lnA = self.lnA - d
                self.lnA_F = np.maximum(F_old + self.p.g0, self._q_lnA())
                self.last_psi = self.mk.hist[t]
            if self.mk is not None:
                self._G[t] = self._gdp(t)
            agri = None
            if self.fd is not None and t > self.origin:
                agri = self._food_step(t, rec)
            if self.cl is not None and t > self.origin:
                self._climate_step(t, agri)
            return rec

        def _psi(self, t):
            if self.mk is not None and self.origin is not None and t > self.origin:
                return self.mk.hist[t]
            return super()._psi(t)

        def ex(self, name, t, default):
            if (self.fd is not None and name in ("ef_cons_pc", "biocap_pc") and self.origin is not None
                    and t > self.origin):
                v = self.fd.footprint()[name]               # D4: calculadas desde los flujos del modelo (año t-1)
                return np.where(np.isfinite(v), v, super().ex(name, t, default))
            if (name == "youth_bulge_15_24_adult" and self.pop_s is not None and self.f1.youth_bulge
                    and self.f1.population and t > self.origin):
                return self.pop_s.youth[t]
            return super().ex(name, t, default)

        def run(self, t0=0, t1=None, policy="hist", freeze_from=None, assimilate_at=None, record=True):
            self.w = self.world0
            self.pop_s = self.hc_s = self.mk = self.en = self.cl = self.fd = None
            self._G = {}
            self.last_psi = None
            self.origin = freeze_from
            if self.f1 is not None and freeze_from is not None:
                self._setup(freeze_from)
            if self.f2 is not None and self.f2.markets and freeze_from is not None:
                self._setup_markets(freeze_from)
            if self.f3 is not None and self.f3.energy and freeze_from is not None:
                self._setup_energy(freeze_from)
            if self.f4 is not None and self.f4.climate and freeze_from is not None:
                self._setup_climate(freeze_from)
            if self.f5 is not None and self.f5.food and freeze_from is not None:
                self._setup_food(freeze_from)
            out = super().run(t0, t1, policy=policy, freeze_from=freeze_from, assimilate_at=assimilate_at,
                              record=record)
            t1 = self.w.T if t1 is None else t1
            for name, arr in (("pop", self.w.pop), ("hc", self.w.hc)):
                full = arr.series(self.R) if isinstance(arr, MemberSeries) else \
                    np.broadcast_to(arr[None], (self.R,) + arr.shape).astype(float)
                out[name] = full[:, :, t0:t1]
            psi = np.full((self.R, t1 - t0), np.nan)
            for t in range(t0, t1):
                if self.mk is not None and t > self.origin:
                    psi[:, t - t0] = self.mk.hist[t][:, 0]
                else:
                    psi[:, t - t0] = float(super()._psi(t))
            out["psi"] = psi
            if self.mk is not None:
                for c in resources.CLASSES:
                    a = np.full((self.R, t1 - t0), np.nan)
                    for t, d in self.mk.prices.items():
                        if t0 <= t < t1:
                            a[:, t - t0] = d[c]
                    out[f"price_{c}"] = a
            if self.en is not None:
                from world4.sectors import energy
                for key in ("PE",) + tuple(f"gen_{x}" for x in energy.SOURCES):
                    a = np.full((self.R, self.w.N, t1 - t0), np.nan)
                    for t, snap in self.en.hist.items():
                        if t0 <= t < t1:
                            a[:, :, t - t0] = snap[key]
                    out[f"energy_{key}"] = a
                for mname in energy.MATERIALS_T_PER_MW:
                    a = np.full((self.R, t1 - t0), np.nan)
                    for t, mats in self.en.materials.items():
                        if t0 <= t < t1:
                            a[:, t - t0] = mats[mname][:, 0]
                    out[f"material_{mname}"] = a
            if self.cl is not None:
                for key in ("gmst", "co2_ffi", "co2_afolu", "co2_conc", "erf"):
                    a = np.full((self.R, t1 - t0), np.nan)
                    for t, snap in self.cl.hist.items():
                        if t0 <= t < t1:
                            a[:, t - t0] = snap[key]
                    out[f"climate_{key}"] = a
            if self.fd is not None:
                for key in ("prod", "cropland", "kcal", "forest"):
                    a = np.full((self.R, self.w.N, t1 - t0), np.nan)
                    for t, snap in self.fd.hist.items():
                        if t0 <= t < t1:
                            a[:, :, t - t0] = snap[key]
                    out[f"food_{key}"] = a
                for key in self.fd.hist[self.origin]["world"]:
                    a = np.full((self.R, t1 - t0), np.nan)
                    for t, snap in self.fd.hist.items():
                        if t0 <= t < t1:
                            a[:, t - t0] = snap["world"][key]
                    out[f"foodw_{key}"] = a
            return out

    return CoupledSimulator


# ---------------------------------------------------------------------------------- pronóstico
def forecast(t0: int, R: int = 32, seed: int | None = None, pop_mode: str = "endogenous", f1: F1Config | None = None,
             world=None, horizon: int = 10, overrides=None, f2: F2Config | None = None,
             f3: F3Config | None = None, f4: F4Config | None = None, f5: F5Config | None = None,
             calib: dict | None = None, offsets: dict | None = None, raw: bool = False):
    """Pronóstico del L1 desde el origen ``t0`` (índice desde 1950), como ``validate.forecast`` del
    Atlas (calibración de corte <= origen, regla de conducta estimada hasta el origen), con la
    población y el capital humano según ``pop_mode``:

        observed    los observados (el Atlas tal cual: fuga D1/D2, el mutante del test de fuga)
        trend       extrapolados con la tendencia de 5 años (F0.5)
        endogenous  los sectores de F1

    y, con ``f2``, la ``psi_t`` de los mercados por clase de F2 en lugar de la del Atlas (congelada
    en su media de 10 años).

    Devuelve arrays (R, N, horizon+1) para t0..t0+horizon: ly, hazard, res_share, debt, demo, pop, hc;
    (R, horizon+1): Tg, psi y, con F2, price_<clase>."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.data import atlas
    atlas.require()
    from world4.core.l1_atlas import behavior, calibrate, model, validate
    if pop_mode not in POP_MODES:
        raise ValueError(f"pop_mode must be one of {POP_MODES}")
    W = calibrate.W if world is None else world
    params, _ = validate.params_for(1950 + t0)
    if overrides:
        params = model.Params(**{**dataclasses.asdict(params), **overrides})
    seed = t0 if seed is None else seed
    t1 = min(t0 + horizon + 1, W.T)
    bm = behavior.BehaviorModel(cutoff=t0).fit(calibrate.W, calibrate.LY)
    # ``offsets`` (F8, scenarios.conduct): desplazamiento del logit de cada canal de la regla desde t0 + 1
    bc = behavior.BehaviorPolicy(bm, params, noise=1.0, freeze=t0, seed=seed + 7, offsets=offsets,
                                 offset_from=t0 + 1)
    pol = lambda s, t: s.hist_actions(t) if t <= t0 else bc(s, t)  # noqa: E731
    if pop_mode == "trend":
        from world4.structural.engine import honest_world
        W = honest_world(W, t0)
    if f5 is not None and f4 is None:
        f4 = F4Config()
    if f4 is not None and f3 is None:
        f3 = F3Config()
    if f3 is not None and f2 is None:
        f2 = F2Config()
    if pop_mode == "endogenous" or f2 is not None:
        sim = make_simulator(model)(W, params, R=R, seed=seed,
                                    f1=(f1 or F1Config()) if pop_mode == "endogenous" else None,
                                    data=(calibrate.W, calibrate.LY), f2=f2, f3=f3, f4=f4, f5=f5, calib=calib)
        o = sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0)
    else:
        if calib:
            raise ValueError("L1 calibration knobs need the endogenous sectors (pop_mode='endogenous' or f2)")
        sim = model.Simulator(W, params, R=R, seed=seed)
        o = sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0)
        o["pop"] = np.broadcast_to(W.pop[None], (R,) + W.pop.shape)
        o["hc"] = np.broadcast_to(W.hc[None], (R,) + W.hc.shape)
        o["psi"] = np.broadcast_to(np.array([sim._psi(t) for t in range(t1)])[None], (R, t1))
    out = {k: np.asarray(o[k][:, :, t0:t1], float) for k in ("ly", "hazard", "res_share", "debt", "demo", "pop", "hc")}
    for k in ("Tg", "psi") + tuple(f"price_{c}" for c in resources.CLASSES):
        if k in o:
            out[k] = np.asarray(o[k][:, t0:t1], float)
    for k in o:
        if k.startswith("energy_") or k.startswith("food_"):
            out[k] = np.asarray(o[k][:, :, t0:t1], float)
        elif k.startswith("material_") or k.startswith("climate_") or k.startswith("foodw_"):
            out[k] = np.asarray(o[k][:, t0:t1], float)
    if raw:                                   # las series del Atlas (para la recompensa revelada del IRL)
        out["raw"] = {k: np.asarray(v[..., t0:t1], float) for k, v in o.items()
                      if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[2] >= t1}
    out["sim"] = sim
    return out
