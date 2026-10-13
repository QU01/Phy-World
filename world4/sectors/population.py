"""Población endógena por cohortes (F1): edad simple 0-100+ y sexo, paso anual, por país y miembro.

Cierra D1: después del origen la población que ve el L1 ya no es la observada, sino la que
producen estas cohortes con nacimientos, defunciones y migración que responden al estado del L1.

Estructura
    * Estado inicial en el origen ``t0``: estructura por edad y sexo de WPP 2024 en ``t0``, escalada
      para que el total coincida con la población que usa el L1 (PWT) en ``t0``.
    * Fecundidad: TFR con dinámica de ajuste parcial (estilo Wittgenstein Centre: la transición
      la empujan ingreso, educación y urbanización)
          dlnTFR_t = b0 + b1 lnTFR_{t-1} + b2 dlnTFR_{t-1} + b3 ly_{t-1} + b4 s_{t-1} + b5 urb_{t-1}
      repartida por edad con el patrón ``percentASFR`` de WPP del último quinquenio cerrado.
    * Mortalidad: e0 con
          de0_t = c0 + c1 e0_{t-1} + c2 de0_{t-1} + c3 ly_{t-1} + c4 dly_{t-1} + c5 conflicto_{t-1}
                  + c6 calor_{t-1}
      (c5 <= 0 y c6 <= 0 declarados). Las tasas por edad son las de WPP en el origen escaladas por
      un factor común exp(kappa) que reproduce la e0 (riesgos proporcionales, Newton por miembro).
    * Migración neta (tasa por persona y año):
          m_t = d0 + d1 m_{t-1} + d2 (ly_{t-1} - ly_mundo_{t-1}) + d3 conflicto_{t-1}
      (d3 <= 0) con techo |m| <= percentil 99 de los datos de ajuste, suma mundial forzada a cero y
      perfil de edad fijo (adultos jóvenes; supuesto [S]).
    * Urbanización: dlogit(u)_t = e0 + e1 logit(u)_{t-1} + e2 dlogit(u)_{t-1} + e3 ly_{t-1}.

Cada ecuación se estima con el panel del Atlas (WPP, PWT, UCDP, temperatura) sólo con años <= el
origen, con covarianza por país; cada miembro del ensemble toma sus coeficientes de esa
distribución y un ruido anual con la desviación del residuo. Todo lo que no se pudo poner:

    * alimento -> mortalidad: no hay sector de alimentos hasta F5 (término ausente, no a cero
      fingido: lo dice ``MISSING_TERMS``);
    * desastres -> mortalidad: el L1 no expone sus desastres por miembro (F4);
    * mortalidad distinta por educación, migración por edad estimada, fecundidad por educación
      de la madre: fuera de F1.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from world4.sectors import _panel

A = 101                                   # edades 0..99 y 100+
MISSING_TERMS = {
    "food": "food -> mortality: no food sector until F5",
    "disasters": "disasters -> mortality: the L1 does not expose its disasters per member (F4)",
}
OBSERVABLE_DRIVERS = ("tfr", "e0", "mig")
SIGNS = {"e0": {"conflict": -1, "heat": -1}, "mig": {"conflict": -1}}
HEAT_BASE_YEARS = 31                      # climatología 1950-1980, como el L1


def _expit(x):
    return 1 / (1 + np.exp(-x))


def _logit(u):
    u = np.clip(u, 1e-3, 1 - 1e-3)
    return np.log(u / (1 - u))


def phi_hc(s):
    """Rendimiento de la escolaridad de la PWT (Psacharopoulos): ln hc = phi(s)."""
    s = np.asarray(s, float)
    return np.where(s <= 4, 0.134 * s, np.where(s <= 8, 0.536 + 0.101 * (s - 4), 0.94 + 0.068 * (s - 8)))


def phi_inv(lnhc):
    x = np.asarray(lnhc, float)
    return np.where(x <= 0.536, x / 0.134, np.where(x <= 0.94, 4 + (x - 0.536) / 0.101, 8 + (x - 0.94) / 0.068))


# ------------------------------------------------------------------------------ panel y ajuste
_PANEL = {}


def panel(W, LY) -> dict:
    """Series (N, T) del panel que usan las ecuaciones, del dato observado (sin rellenar hacia atrás)."""
    key = id(W)
    if key in _PANEL:
        return _PANEL[key]
    ex = W.extra
    nan = np.full((W.N, W.T), np.nan)
    tfr = ex.get("tfr", nan)
    e0 = ex.get("life_expectancy", nan)
    mig = ex.get("net_migration_rate", nan)
    urb = W.locf("urban_share")
    tmp = W.locf("tmp_level")
    tbar = np.nanmean(tmp[:, :HEAT_BASE_YEARS], 1)
    sig = np.clip(W.locf("tmp_anom_fao_sd", 0.31), 0.15, 1.0)
    heat = np.maximum(tmp - tbar[:, None], 0) / sig
    hc = np.where(W.hc > 0, W.hc, np.nan)
    s = phi_inv(np.log(hc))
    r = W.locf("acc_r")
    popw = np.where(np.isfinite(LY), W.pop, 0.0)
    lyw = np.nansum(np.nan_to_num(LY) * popw, 0) / np.maximum(popw.sum(0), 1e-9)
    p = dict(ltfr=np.log(np.where(tfr > 0, tfr, np.nan)), e0=e0, mig=mig, lu=_logit(urb), ly=LY,
             lyrel=LY - lyw[None], conf=W.conflict.astype(float), heat=heat, s=s, r=r, tbar=tbar)
    _PANEL[key] = p
    return p


def _lag(a, k=1):
    out = np.full_like(a, np.nan)
    out[:, k:] = a[:, :-k]
    return out


def _stack(cols, t_lo, t_hi):
    """Filas país-año para t en [t_lo, t_hi]."""
    sl = slice(t_lo, t_hi + 1)
    N = cols[0].shape[0]
    n_t = t_hi - t_lo + 1
    country = np.repeat(np.arange(N), n_t)
    year = np.tile(np.arange(t_lo, t_hi + 1), N)
    return [c[:, sl].ravel() for c in cols], country, year


def fit(W, LY, cutoff_t: int, t_min: int = 2) -> dict:
    """Ajusta las cuatro ecuaciones demográficas con años <= ``cutoff_t`` (índice desde 1950)."""
    p = panel(W, LY)
    eqs = {}
    cy = 1950 + cutoff_t

    def go(name, y, regs, terms, signs=None):
        cols, country, year = _stack([y] + regs, t_min, cutoff_t)
        X = np.column_stack([np.ones_like(cols[0])] + cols[1:])
        recent = year > cutoff_t - 10
        eqs[name] = _panel.fit(name, cols[0], X, country, ("const",) + terms, cy, signs=signs,
                               n_countries=W.N, recent=recent, year=year)

    lt, e0, lu = p["ltfr"], p["e0"], p["lu"]
    dlt, de0, dlu = lt - _lag(lt), e0 - _lag(e0), lu - _lag(lu)
    ly1 = _lag(p["ly"])
    go("tfr", dlt, [_lag(lt), _lag(dlt), ly1, _lag(p["s"]), _lag(_expit(lu))],
       ("lnTFR", "dlnTFR", "ly", "schooling", "urban"))
    go("e0", de0, [_lag(e0), _lag(de0), ly1, ly1 - _lag(p["ly"], 2), _lag(p["conf"]), _lag(p["heat"])],
       ("e0", "de0", "ly", "dly", "conflict", "heat"), SIGNS["e0"])
    go("mig", p["mig"], [_lag(p["mig"]), _lag(p["lyrel"]), _lag(p["conf"])],
       ("mig", "ly_rel", "conflict"), SIGNS["mig"])
    go("urban", dlu, [_lag(lu), _lag(dlu), ly1], ("logit_u", "dlogit_u", "ly"))
    m = p["mig"][:, :cutoff_t + 1]
    eqs["mig"].meta["cap"] = float(np.nanpercentile(np.abs(m), 99)) if np.isfinite(m).any() else 0.02
    return eqs


def fit_e0_food(W, LY, cutoff_t: int, shortfall, t_min: int = 2):
    """F5: la ecuación de e0 con el déficit calórico del año anterior (``shortfall`` (N, T) =
    max(0, ln KCAL_MIN - ln kcal), FAOSTAT) como regresor más, con signo <= 0. Sólo la usa el modelo
    con F5; sin F5 la e0 es la de ``fit`` (F1-F4 no cambian)."""
    p = panel(W, LY)
    e0 = p["e0"]
    de0 = e0 - _lag(e0)
    ly1 = _lag(p["ly"])
    regs = [_lag(e0), _lag(de0), ly1, ly1 - _lag(p["ly"], 2), _lag(p["conf"]), _lag(p["heat"]), _lag(shortfall)]
    cols, country, year = _stack([de0] + regs, t_min, cutoff_t)
    X = np.column_stack([np.ones_like(cols[0])] + cols[1:])
    return _panel.fit("e0_food", cols[0], X, country, ("const", "e0", "de0", "ly", "dly", "conflict", "heat", "food_short"),
                      1950 + cutoff_t, signs={**SIGNS["e0"], "food_short": -1}, n_countries=W.N,
                      recent=year > cutoff_t - 10, year=year)


# ------------------------------------------------------------------------------ tablas de vida
def single_age(mx5):
    """(..., 22) quinquenios de WPP (0, 1-4, 5-9, ..., 95-99, 100+) -> (..., 101) edades simples."""
    g = np.empty(A, int)
    g[0], g[1:5] = 0, 1
    g[5:100] = 2 + (np.arange(5, 100) - 5) // 5
    g[100] = 21
    return mx5[..., g]


def life_expectancy(mx):
    """e0 de tasas (..., 101); a0 = 0.2 en la edad 0, 0.5 en el resto, grupo abierto l/m."""
    mx = np.maximum(mx, 1e-9)
    q = 1 - np.exp(-mx[..., :-1])
    l = np.concatenate([np.ones(mx.shape[:-1] + (1,)), np.cumprod(1 - q, -1)], -1)
    Lx = l[..., :-1] - (1 - np.r_[0.2, np.full(A - 2, 0.5)]) * (l[..., :-1] - l[..., 1:])
    return Lx.sum(-1) + l[..., -1] / mx[..., -1]


def asfr_single(pasfr7):
    """(N, 7) porcentajes por quinquenio 15-49 -> (N, 101) fracción por edad simple (suma 1)."""
    f = np.zeros(pasfr7.shape[:-1] + (A,))
    for k in range(7):
        f[..., 15 + 5 * k: 20 + 5 * k] = pasfr7[..., k:k + 1] / 100.0 / 5.0
    s = f.sum(-1, keepdims=True)
    return np.where(s > 0, f / np.where(s > 0, s, 1), 0.0)


def migration_profile():
    """Perfil por edad de los migrantes netos (suma 1): adultos jóvenes y sus hijos [S]."""
    a = np.arange(A, dtype=float)
    w = np.exp(-0.5 * ((a - 27) / 8.0) ** 2) + 0.35 * np.exp(-a / 6.0) + 0.02 * np.exp(-0.5 * ((a - 63) / 5.0) ** 2)
    return w / w.sum()


# ------------------------------------------------------------------------------ simulación
@dataclass
class Init:
    """Lo que el sector lee en el origen; todo con fecha <= origen."""
    P: np.ndarray        # (N, 2, A) millones, hombres/mujeres
    mx0: np.ndarray      # (N, 2, A) tasas de mortalidad del último quinquenio cerrado
    fa: np.ndarray       # (N, A) patrón de fecundidad por edad
    srb: np.ndarray      # (N,)
    ok: np.ndarray       # (N,) país con todos los datos
    state: dict          # (N,) ltfr, dltfr, e0, de0, mig, lu, dlu en el origen


def initial(W, LY, t0: int, wppa: dict) -> Init:
    from world4.data import wpp
    p = panel(W, LY)
    y0 = 1950 + t0
    k_y = y0 - int(wppa["years"][0])
    pm, pf = wppa["popM"][:, k_y].astype(float), wppa["popF"][:, k_y].astype(float)
    tot = pm.sum(1) + pf.sum(1)
    scale = W.pop[:, t0] / np.where(tot > 0, tot, np.nan)          # WPP (miles) -> PWT (millones)
    P = np.stack([pm, pf], 1) * scale[:, None, None]
    kp = wpp.period_for(y0, wppa["periods"])
    mx0 = np.stack([single_age(wppa["mxM"][:, kp].astype(float)), single_age(wppa["mxF"][:, kp].astype(float))], 1)
    fa = asfr_single(wppa["pasfr"][:, kp].astype(float))
    srb = wppa["srb"][:, k_y].astype(float)

    def at(a, t):
        return a[:, t] if t >= 0 else np.full(W.N, np.nan)

    st = {"ltfr": at(p["ltfr"], t0), "dltfr": at(p["ltfr"], t0) - at(p["ltfr"], t0 - 1),
          "e0": at(p["e0"], t0), "de0": at(p["e0"], t0) - at(p["e0"], t0 - 1),
          "mig": at(p["mig"], t0), "lu": at(p["lu"], t0), "dlu": at(p["lu"], t0) - at(p["lu"], t0 - 1)}
    ok = (wppa["has"] & np.isfinite(scale) & np.all(np.isfinite(P.reshape(W.N, -1)), 1)
          & np.all(np.isfinite(mx0.reshape(W.N, -1)), 1) & np.isfinite(srb) & (fa.sum(1) > 0)
          & np.isfinite(st["ltfr"]) & np.isfinite(st["e0"]))
    for k in ("dltfr", "de0", "mig", "dlu"):
        st[k] = np.where(np.isfinite(st[k]), st[k], 0.0)
    st["lu"] = np.where(np.isfinite(st["lu"]), st["lu"], np.nanmedian(st["lu"]))
    return Init(P, mx0, fa, np.where(np.isfinite(srb), srb, 1.05), ok, st)


class Population:
    """Cohortes (R, N, 2, A) que avanzan un año con los resultados del L1 del año anterior.

    ``drivers="model"`` usa las ecuaciones; ``"observed"`` (o un subconjunto de ``OBSERVABLE_DRIVERS``)
    toma esos impulsores del dato: sólo para comprobar la contabilidad de las cohortes y descomponer
    el error. Es una fuga a propósito y nunca entra en un pronóstico.
    """

    def __init__(self, W, LY, t0: int, eqs: dict, wppa: dict, R: int, seed: int, drivers: str = "model",
                 noise: bool = True):
        self.W, self.LY, self.t0, self.R, self.eqs, self.drivers = W, LY, t0, R, eqs, drivers
        self.observed = (frozenset(OBSERVABLE_DRIVERS) if drivers == "observed" else
                         frozenset() if drivers == "model" else frozenset(drivers))
        if not self.observed <= set(OBSERVABLE_DRIVERS):
            raise ValueError(f"drivers: 'model', 'observed' or a subset of {OBSERVABLE_DRIVERS}")
        self.rng = np.random.default_rng(seed)
        self.noise = noise
        ini = initial(W, LY, t0, wppa)
        self.ok, self.fa, self.srb = ini.ok, ini.fa, ini.srb
        self.bmale = self.srb / (1 + self.srb)
        N = W.N
        self.P = np.broadcast_to(np.nan_to_num(ini.P), (R, N, 2, A)).copy()
        self.mx0 = np.nan_to_num(ini.mx0, nan=0.05)
        rep = lambda v: np.broadcast_to(v, (R, N)).copy()  # noqa: E731
        self.s = {k: rep(v) for k, v in ini.state.items()}
        self.B = {k: e.draw(R, self.rng) for k, e in eqs.items()} if noise else \
            {k: np.broadcast_to(e.beta, (R, len(e.beta))).copy() for k, e in eqs.items()}
        self.shock = {k: _panel.shocks(e, R, N, self.rng, on=noise) for k, e in eqs.items() if k != "schooling"}
        self.kappa = np.zeros((R, N))
        self.kappa = self._solve_kappa(self.s["e0"], iters=8)
        self.prof = migration_profile()
        self.trend_g = self._trend_growth()
        self.hist_pop = {t0: np.broadcast_to(W.pop[:, t0], (R, N)).copy()}
        self.youth = {t0: self._youth()}
        self.last_births = np.zeros((R, N))
        self.food_eq, self.B_food, self.food_short = None, None, None

    def enable_food(self, eq):
        """F5: la e0 pasa a la ecuación con déficit calórico (``fit_e0_food``); ``food_short`` (R, N) lo
        escribe el sector de alimentos cada año."""
        self.food_eq = eq
        self.B_food = eq.draw(self.R, self.rng) if self.noise else             np.broadcast_to(eq.beta, (self.R, len(eq.beta))).copy()
        self.food_short = np.zeros((self.R, self.W.N))

    # -------------------------------------------------------------- utilidades
    def _trend_growth(self):
        W, t0 = self.W, self.t0
        lo = max(0, t0 - 5)
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (W.pop[:, t0] / W.pop[:, lo]) ** (1.0 / max(t0 - lo, 1))
        return np.where(np.isfinite(g), np.clip(g, 0.9, 1.1), 1.0)

    def _mx(self, kappa):
        return self.mx0[None] * np.exp(kappa)[..., None, None]

    def _e0(self, kappa):
        mx = self._mx(kappa)
        e = life_expectancy(mx)                              # (R, N, 2)
        return self.bmale * e[..., 0] + (1 - self.bmale) * e[..., 1]

    def _solve_kappa(self, target, iters=3):
        """Newton sobre el factor común de mortalidad; arranca del kappa del año anterior."""
        k = self.kappa.copy()
        tgt = np.where(np.isfinite(target), target, self._e0(k))
        for _ in range(iters):
            f0 = self._e0(k) - tgt
            d = (self._e0(k + 1e-3) - self._e0(k)) / 1e-3
            k = np.clip(k - f0 / np.where(np.abs(d) > 1e-9, d, -1.0), -6.0, 4.0)
        return k

    def _youth(self):
        Pb = self.P.sum(2)
        return Pb[..., 15:25].sum(-1) / np.maximum(Pb[..., 15:].sum(-1), 1e-12)

    def _noise(self, k):
        common, own = self.shock[k]
        return common() + own()

    def total(self):
        return self.P.sum((2, 3))

    # -------------------------------------------------------------- un año
    def advance(self, t: int, ly, dly, conf, heat, lyrel, schooling):
        """Estado del año ``t`` a partir del de ``t-1``. Entradas (R, N) del año t-1."""
        R, N, s, B = self.R, self.W.N, self.s, self.B
        eq = self.eqs
        urb = _expit(s["lu"])
        X = np.stack([np.ones((R, N)), s["ltfr"], s["dltfr"], ly, schooling, urb], -1)
        dlt = np.einsum("rnk,rk->rn", np.nan_to_num(X), B["tfr"]) + eq["tfr"].offset + self._noise("tfr")
        ltfr = np.clip(s["ltfr"] + np.clip(dlt, -0.15, 0.15), np.log(0.7), np.log(9.0))
        cols = [np.ones((R, N)), s["e0"], s["de0"], ly, dly, conf, heat]
        if self.food_eq is not None:                       # F5: con el déficit calórico del año anterior
            cols.append(np.nan_to_num(self.food_short))
            de0 = (np.einsum("rnk,rk->rn", np.nan_to_num(np.stack(cols, -1)), self.B_food)
                   + self.food_eq.offset + self._noise("e0"))
        else:
            de0 = (np.einsum("rnk,rk->rn", np.nan_to_num(np.stack(cols, -1)), B["e0"]) + eq["e0"].offset
                   + self._noise("e0"))
        e0 = np.clip(s["e0"] + np.clip(de0, -15, 3), 15.0, 92.0)
        X = np.stack([np.ones((R, N)), s["mig"], lyrel, conf], -1)
        cap = eq["mig"].meta["cap"]
        mig = np.einsum("rnk,rk->rn", np.nan_to_num(X), B["mig"]) + eq["mig"].offset + self._noise("mig")
        mig = np.clip(mig, -cap, cap)
        X = np.stack([np.ones((R, N)), s["lu"], s["dlu"], ly], -1)
        dlu = np.einsum("rnk,rk->rn", np.nan_to_num(X), B["urban"]) + eq["urban"].offset + self._noise("urban")
        lu = s["lu"] + np.clip(dlu, -0.3, 0.3)
        s["dlu"] = lu - s["lu"]
        if self.observed:                     # descomposición / contabilidad: impulsores del dato
            p = panel(self.W, self.LY)
            obs = lambda k, fb: np.where(np.isfinite(p[k][:, t]), p[k][:, t], fb)  # noqa: E731
            if "tfr" in self.observed:
                ltfr = obs("ltfr", ltfr)
            if "e0" in self.observed:
                e0 = obs("e0", e0)
            if "mig" in self.observed:
                mig = obs("mig", np.zeros_like(mig))
        s["dltfr"], s["de0"] = ltfr - s["ltfr"], e0 - s["e0"]
        s["ltfr"], s["e0"], s["lu"] = ltfr, e0, lu
        # ------------------------------------------------ cohortes
        self.kappa = self._solve_kappa(e0)
        surv = np.exp(-self._mx(self.kappa))                 # (R, N, 2, A)
        P = self.P
        new = np.empty_like(P)
        new[..., 1:A - 1] = P[..., :A - 2] * surv[..., :A - 2]
        new[..., A - 1] = P[..., A - 2] * surv[..., A - 2] + P[..., A - 1] * surv[..., A - 1]
        births = np.exp(ltfr) * np.einsum("rna,na->rn", P[:, :, 1], self.fa)
        new[:, :, 0, 0] = births * self.bmale * np.sqrt(surv[:, :, 0, 0])
        new[:, :, 1, 0] = births * (1 - self.bmale) * np.sqrt(surv[:, :, 1, 0])
        tot = P.sum((2, 3))
        act = self.ok[None] & (tot > 0)
        if "mig" not in self.observed:
            wsum = np.sum(np.where(act, tot, 0.0), 1, keepdims=True)
            mig = mig - np.sum(np.where(act, mig * tot, 0.0), 1, keepdims=True) / np.maximum(wsum, 1e-12)
        s["mig"] = mig
        new += (mig * tot)[..., None, None] * (self.prof / 2.0)
        new = np.maximum(new, 0.0)
        # países sin datos completos: tendencia de 5 años (el método de F0.5), declarado
        trend = self.P * self.trend_g[None, :, None, None]
        self.P = np.where(act[..., None, None], new, trend)
        self.last_births = births
        self.hist_pop[t] = self.total()
        self.youth[t] = self._youth()
        return self.hist_pop[t]
