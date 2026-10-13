"""Energía con EROI (F3): mezcla primaria por país, energía neta y materiales de la transición.

Estado por miembro y país (R, N): energía primaria ``PE`` (TWh, método de sustitución) y la cuota
de cada fuente no fósil; el fósil es el resto (la fuente que ajusta).

Cada año ``t`` después del origen, con el estado del L1 del año anterior:

    intensidad  dln(PE / PIB)_t = a + b dln(PE / PIB)_{t-1} + c dln pF_{t-1}     (c <= 0)
                pF = índice mundial de precio fósil (petróleo, gas y carbón de F2, pesos de energía)
    cuotas      dz_s = a_s + b_s z_s,{t-1} + c_s dz_s,{t-1} [+ d G_{t-1} para vre]    z = logit(cuota)
                nuclear, hidro, otras: difusión estimada; nuclear donde no hay se queda en 0
    brecha vre  G_t = ln pF_t + BETA ln CUM_t - K_INT (cuota vre)^2       (d >= 0)
                CUM: generación eólica + solar acumulada en el mundo (aprendizaje) [S: BETA, K_INT]
    fósil       PE * (1 - sum de cuotas no fósiles)

Todo se estima con el panel país-año (OWID / EI) con datos <= origen, efecto propio de cada país
encogido, choques comunes y propios AR(1) (``_panel``), como F1.

EROI [S, sin verificar: valores de la literatura (Hall et al. 2014; Murphy et al. 2022), con
incertidumbre +-30 % por miembro]: fósil 20 en el origen y cae con el agotamiento mundial de F2,
EROI_f = 20 (1 - x_t) / (1 - x_origen); nuclear 15; hidro 40; otras 4; eólica + solar 10 en el
origen y sube con el aprendizaje, (CUM_t / CUM_origen)^(BETA / 2), con techo 30.

    bruta = sum_s gen_s      autoconsumo = sum_s gen_s / EROI_s      neta = bruta - autoconsumo

Al L1 (por la TFP, normalizado a 0 en el origen, como F2):

    * coste real del sistema energético por país, c = sum_s cuota_s kappa_s / (1 - 1/EROI_mezcla),
      kappa_vre = (CUM/CUM0)^-BETA (1 + K_INT cuota^2) / (1 + K_INT cuota0^2), el resto 1 (el precio
      fósil ya lo lleva F2 por las rentas: no se cuenta dos veces); dlnTFP = -THETA_E ln(c / c0)
    * escasez física: si la capacidad de F2 no cubre la demanda fósil (falta f), el país pierde
      energía en proporción a su cuota fósil y la producción cae según un CES de baja sustitución
      (SIGMA_E = 0.3 [S]): dlnY = (1/rho) ln(1 - THETA_E + THETA_E (1 - cuota_f f)^rho)

Materiales [S, sin verificar: IEA 2021, The Role of Critical Minerals in Clean Energy Transitions]:
por MW nuevo de eólica + solar, cobre 2.85 t, níquel 0.2 t, tierras raras 0.007 t. El cobre y el
níquel entran como demanda adicional en el mercado de metales de F2; las tierras raras sólo como
indicador (sin serie de oferta). El litio no: depende de vehículos y baterías, que no hay.

Lo que no está: electricidad como vector (todo es energía primaria), almacenamiento, vehículos,
biomasa tradicional (el EI no la cuenta), coste absoluto de cada fuente (se supone igual en el
origen [S]; sólo cuentan sus cambios).
"""
from __future__ import annotations

import numpy as np

from world4.sectors import _panel

NONFOSSIL = ("nuclear", "hydro", "vre", "other")
SOURCES = ("fossil",) + NONFOSSIL
BETA = 0.234                     # aprendizaje eólica+solar: tasa 15 % por duplicación (-log2(0.85)) [S]
BETA_RANGE = (0.152, 0.322)      # tasas 10-20 % [S]
K_INT = 2.0                      # coste de integración de la variable: x(1 + K cuota^2) [S]
THETA_E = 0.05                   # cuota del coste energético en el PIB [S]
SIGMA_E = 0.3                    # sustitución energía / resto de factores [S]
EROI0 = {"fossil": 20.0, "nuclear": 15.0, "hydro": 40.0, "vre": 10.0, "other": 4.0}   # [S]
EROI_SD = 0.3                    # incertidumbre log por miembro [S]
EROI_VRE_MAX = 30.0
SHARE_FLOOR = 1e-4
MIN_N = 50                       # observaciones mínimas para que la cuota de una fuente sea dinámica
MW_PER_TWH_YR = 1e6 / (8760 * 0.25)          # factor de capacidad 0.25 [S]
MATERIALS_T_PER_MW = {"copper": 2.85, "nickel": 0.2, "rare_earths": 0.007}   # [S] IEA 2021
SIGNS = {"intensity": {"dlnpF": -1}, "vre": {"G": +1}}
MISSING = {
    "electricity": "no electricity vector, storage or vehicles: primary energy only",
    "lithium": "lithium demand depends on vehicles and batteries, which are not modelled",
    "traditional_biomass": "traditional biomass is not in the Energy Institute data",
    "absolute_costs": "sources are assumed to cost the same at the origin; only cost changes count",
}


def _logit(s):
    s = np.clip(s, SHARE_FLOOR, 1 - SHARE_FLOOR)
    return np.log(s / (1 - s))


def _expit(z):
    return 1 / (1 + np.exp(-z))


def _lag(a, k=1):
    o = np.full_like(a, np.nan)
    o[:, k:] = a[:, :-k]
    return o


# ------------------------------------------------------------------------------ panel
_PANEL = {}


def panel(W) -> dict:
    """Series (N, T) del panel de energía en los años del L1 y series mundiales (T,)."""
    key = id(W)
    if key in _PANEL:
        return _PANEL[key]
    from world4.data import energy as edata
    from world4.sectors import resources
    a = edata.aligned(W.iso3, W.isonum, W.years)
    pe = a["pe"]
    sh = {s: a[f"src_{s}"] / np.where(pe > 0, pe, np.nan) for s in SOURCES}
    Y = np.where(W.observed, W.Y, np.nan)
    D = resources.data(W)
    # índice de precio fósil: petróleo, gas y carbón con pesos de energía mundial de cada año (cadena)
    lp = {c: np.log(D[c]["p"]) for c in ("oil", "gas", "coal")}
    q = {c: D[c]["Q"] for c in ("oil", "gas", "coal")}
    dl = np.zeros(W.T)
    for t in range(1, W.T):
        num = den = 0.0
        for c in lp:
            if np.isfinite(lp[c][t]) and np.isfinite(lp[c][t - 1]) and np.isfinite(q[c][t - 1]):
                num += q[c][t - 1] * (lp[c][t] - lp[c][t - 1])
                den += q[c][t - 1]
        dl[t] = num / den if den > 0 else 0.0
    lpF = np.cumsum(dl)
    vre_w = np.nansum(np.where(np.isfinite(a["src_vre"]), a["src_vre"], 0), 0)
    cum = np.cumsum(vre_w) + 1.0
    fossil_w = np.where(np.isfinite(pe).sum(0) > 50, np.nansum(np.where(np.isfinite(a["src_fossil"]),
                                                                          a["src_fossil"], 0), 0), np.nan)
    out = {"pe": pe, "sh": sh, "lint": np.log(np.where(pe > 0, pe, np.nan) / Y), "lpF": lpF, "cum": cum,
           "fossil_world": fossil_w, "region": a["region"], "imputed": a["imputed"], "src": a}
    _PANEL[key] = out
    return out


def _stack(cols, t_lo, t_hi):
    sl = slice(t_lo, t_hi + 1)
    N = cols[0].shape[0]
    n_t = t_hi - t_lo + 1
    return [c[:, sl].ravel() for c in cols], np.repeat(np.arange(N), n_t), np.tile(np.arange(t_lo, t_hi + 1), N)


def fit(W, cutoff_t: int, beta: float = BETA) -> dict:
    """Ecuaciones de F3 con datos <= ``cutoff_t`` (desde 1966: el panel de energía empieza en 1965)."""
    p = panel(W)
    N, T = W.N, W.T
    t_lo = max(1966 - 1950 + 2, 2)
    cy = 1950 + cutoff_t
    eqs = {}
    rep = lambda v: np.broadcast_to(v, (N, T))  # noqa: E731

    wgt = _stack([_lag(np.where(p["pe"] > 0, p["pe"], np.nan))], t_lo, cutoff_t)[0][0]

    def go(name, y, regs, terms, signs=None, rows=None):
        cols, country, year = _stack([y] + regs, t_lo, cutoff_t)
        X = np.column_stack([np.ones_like(cols[0])] + cols[1:])
        yy = cols[0] if rows is None else np.where(_stack([rows], t_lo, cutoff_t)[0][0] > 0, cols[0], np.nan)
        # ponderadas por la energía del país: la ecuación debe representar el agregado mundial
        eqs[name] = _panel.fit(name, yy, X, country, ("const",) + terms, cy, signs=signs, n_countries=N,
                               recent=year > cutoff_t - 10, year=year, weights=wgt)

    li = p["lint"]
    di = li - _lag(li)
    dpF = np.diff(p["lpF"], prepend=np.nan)
    go("intensity", di, [_lag(di), rep(_lag(dpF[None])[0])], ("dln_intensity", "dlnpF"), SIGNS["intensity"])
    G = p["lpF"][None] + beta * np.log(p["cum"])[None] - K_INT * np.nan_to_num(p["sh"]["vre"]) ** 2
    for s in NONFOSSIL:
        z = _logit(p["sh"][s])
        dz = z - _lag(z)
        present = (_lag(p["sh"][s]) > SHARE_FLOOR).astype(float)          # sólo donde la fuente existe
        if s == "vre":
            go(s, dz, [_lag(z), _lag(dz), _lag(G)], ("z", "dz", "G"), SIGNS["vre"], rows=present)
        else:
            go(s, dz, [_lag(z), _lag(dz)], ("z", "dz"), rows=present)
    # Sin efecto propio en las cuotas: con una reversión a la media débil, el efecto propio de un país
    # que arranca (eólica en Australia, nuclear en Argentina) extrapola su crecimiento exponencial
    # inicial hasta cuotas del 30-90 %. Las cuotas siguen la dinámica común estimada [declarado].
    for s in NONFOSSIL:
        eqs[s].offset = np.zeros_like(eqs[s].offset)
    eqs["meta"] = {"beta": beta}
    return eqs


# ------------------------------------------------------------------------------ simulación
class Energy:
    def __init__(self, W, t0: int, eqs: dict, R: int, seed: int, noise: bool = True):
        self.W, self.t0, self.R, self.eqs, self.noise = W, t0, R, eqs, noise
        self.rng = np.random.default_rng(seed)
        p = panel(W)
        self.p = p
        N = W.N
        rep = lambda v: np.broadcast_to(v, (R, N)).copy()  # noqa: E731
        pe0 = p["pe"][:, t0]
        lo = max(0, t0 - 5)
        self.ok = np.isfinite(pe0) & (pe0 > 0)
        pe0 = np.where(self.ok, pe0, 0.0)
        self.PE = rep(pe0)
        self.sh = {}
        for s in NONFOSSIL:
            v = np.nan_to_num(p["sh"][s][:, t0])
            self.sh[s] = rep(v)
        self.sh0 = {s: self.sh[s].copy() for s in NONFOSSIL}
        self.dz = {s: rep(np.nan_to_num(_logit(p["sh"][s][:, t0]) - _logit(p["sh"][s][:, t0 - 1]))) for s in NONFOSSIL}
        self.has = {s: p["sh"][s][:, t0] > SHARE_FLOOR for s in NONFOSSIL}
        # sólo donde la fuente existe en el origen [declarado: un país sin nuclear o sin eólica + solar en el
        # origen no la adopta en la proyección]; con < MIN_N observaciones la cuota queda congelada
        self.frozen = {s for s in NONFOSSIL if eqs[s].n < MIN_N}
        di0 = p["lint"][:, t0] - p["lint"][:, lo]
        self.di = rep(np.nan_to_num(di0 / max(t0 - lo, 1)))
        self.B = {k: (e.draw(R, self.rng) if noise else np.broadcast_to(e.beta, (R, len(e.beta))).copy())
                  for k, e in eqs.items() if k != "meta"}
        self.shock = {k: _panel.shocks(e, R, N, self.rng, on=noise) for k, e in eqs.items() if k != "meta"}
        self.beta = (np.exp(self.rng.uniform(np.log(BETA_RANGE[0]), np.log(BETA_RANGE[1]), R)) if noise
                     else np.full(R, eqs["meta"]["beta"]))[:, None]
        z = (lambda: np.exp(EROI_SD * self.rng.standard_normal(R))) if noise else (lambda: np.ones(R))
        self.eroi0 = {s: (EROI0[s] * z())[:, None] for s in SOURCES}
        self.cum0 = float(p["cum"][t0])
        self.cum = np.full((R, 1), self.cum0)
        self.lpF0 = float(p["lpF"][t0])
        self.lpF = np.full((R, 1), self.lpF0)
        self.dlpF = np.full((R, 1), float(p["lpF"][t0] - p["lpF"][t0 - 1]))
        self.x0 = self.x = None                       # agotamiento fósil mundial de F2 (``set_depletion``)
        from world4.data import energy as edata
        d = edata.load()
        k = int(np.clip(1950 + t0 - int(d["eff_years"][0]), 0, len(d["eff_years"]) - 1))
        self.eff_vre = float(np.nanmean([d["eff_solar"][k], d["eff_wind"][k]]))
        self.cost0 = None
        self.hist = {t0: self._snapshot()}
        self.cost0 = self._cost()
        self.materials = {}
        self.last_new_vre = np.zeros((R, 1))

    # -------------------------------------------------------------- utilidades
    def _snapshot(self):
        gen = self.generation()
        return {"PE": self.PE.copy(), **{f"gen_{s}": g.copy() for s, g in gen.items()}}

    def generation(self):
        nf = sum(self.sh[s] for s in NONFOSSIL)
        scale = np.where(nf > 0.98, 0.98 / np.maximum(nf, 1e-12), 1.0)
        gen = {s: self.PE * self.sh[s] * scale for s in NONFOSSIL}
        gen["fossil"] = self.PE - sum(gen.values())
        return gen

    def eroi(self):
        e = dict(self.eroi0)
        if self.x is not None and self.x0 is not None:
            e["fossil"] = self.eroi0["fossil"] * np.clip((1 - self.x) / np.maximum(1 - self.x0, 1e-6), 0.05, 1.5)
        e["vre"] = np.minimum(self.eroi0["vre"] * (self.cum / self.cum0) ** (self.beta / 2), EROI_VRE_MAX)
        return e

    def accounts(self):
        """Bruta, autoconsumo y neta (R, N); por construcción bruta = autoconsumo + neta."""
        gen, e = self.generation(), self.eroi()
        gross = sum(gen.values())
        own = sum(gen[s] / e[s] for s in SOURCES)
        return {"gross": gross, "own_use": own, "net": gross - own, "gen": gen, "eroi": e}

    def _cost(self):
        a = self.accounts()
        gross = np.maximum(a["gross"], 1e-12)
        sv = self.sh["vre"]
        k_vre = ((self.cum / self.cum0) ** (-self.beta) * (1 + K_INT * sv ** 2)
                 / (1 + K_INT * self.sh0["vre"] ** 2))
        mix = (a["gen"]["fossil"] + a["gen"]["nuclear"] + a["gen"]["hydro"] + a["gen"]["other"]
               + a["gen"]["vre"] * k_vre) / gross
        return mix / np.maximum(a["net"] / gross, 1e-3)

    def tfp_shift(self):
        """dlnTFP (R, N) por el coste real del sistema energético, 0 en el origen."""
        c = self._cost()
        return np.where(self.ok[None], -THETA_E * np.log(np.maximum(c / self.cost0, 1e-6)), 0.0)

    def shortage_shift(self, short):
        """dlnY (R, N) si falta una fracción ``short`` (R, 1) de la oferta fósil mundial."""
        if short is None or not np.any(short > 0):
            return 0.0
        gen = self.generation()
        fs = gen["fossil"] / np.maximum(self.PE, 1e-12)
        rho = (SIGMA_E - 1) / SIGMA_E
        avail = np.clip(1 - fs * short, 0.05, 1.0)
        return np.where(self.ok[None], np.log(1 - THETA_E + THETA_E * avail ** rho) / rho, 0.0)

    def fossil_world(self):
        return self.generation()["fossil"].sum(1, keepdims=True)

    # -------------------------------------------------------------- un año
    def set_depletion(self, x):
        """Agotamiento fósil mundial (R, 1) en el origen: la EROI fósil cae desde ahí."""
        self.x0 = np.array(x, float).copy()
        self.x = self.x0.copy()

    def step(self, t: int, gY_i, lpF, x_depl=None):
        """Año ``t``: ``gY_i`` (R, N) crecimiento del PIB de cada país (año anterior), ``lpF`` (R, 1)
        log del índice de precio fósil del año anterior (de F2), ``x_depl`` (R, 1) agotamiento fósil."""
        R, N = self.R, self.W.N
        eq, B = self.eqs, self.B
        if x_depl is not None and self.x0 is not None:
            self.x = x_depl
        self.dlpF = lpF - self.lpF
        self.lpF = lpF
        X = np.stack([np.ones((R, N)), self.di, np.broadcast_to(self.dlpF, (R, N))], -1)
        di = np.einsum("rnk,rk->rn", X, B["intensity"]) + eq["intensity"].offset + self._noise("intensity")
        di = np.clip(di, -0.15, 0.15)
        self.PE = np.where(self.ok[None], self.PE * np.exp(np.clip(gY_i, -0.3, 0.3) + di), self.PE)
        self.di = di
        G = self.lpF + self.beta * np.log(self.cum) - K_INT * self.sh["vre"] ** 2
        vre_before = self.generation()["vre"]
        for s in NONFOSSIL:
            z = _logit(self.sh[s])
            cols = [np.ones((R, N)), z, self.dz[s]] + ([np.broadcast_to(G, (R, N))] if s == "vre" else [])
            dz = np.einsum("rnk,rk->rn", np.stack(cols, -1), B[s]) + eq[s].offset + self._noise(s)
            dz = np.clip(dz, -1.0, 1.0)
            if s in self.frozen:
                dz = np.zeros_like(dz)
            new = np.where(self.has[s][None], _expit(z + dz), 0.0)
            self.dz[s] = np.where(self.has[s][None], dz, 0.0)
            self.sh[s] = new
        gen = self.generation()
        self.cum = self.cum + gen["vre"].sum(1, keepdims=True)
        self.last_new_vre = np.maximum(gen["vre"] - vre_before, 0).sum(1, keepdims=True)
        self.materials[t] = self._materials(t, self.last_new_vre)
        self.hist[t] = self._snapshot()
        return self.hist[t]

    def _noise(self, k):
        common, own = self.shock[k]
        return common() + own()

    def _materials(self, t, new_vre_primary):
        """Toneladas (R, 1) de cada material por la eólica + solar nueva del año. De primaria a
        generación con la eficiencia de sustitución del año del origen (nada posterior)."""
        mw = new_vre_primary * self.eff_vre * MW_PER_TWH_YR
        return {m: mw * v for m, v in MATERIALS_T_PER_MW.items()}
