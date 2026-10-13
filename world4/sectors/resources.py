"""Escasez global endógena (F2): mercados mundiales por clase y ``psi_t`` como SALIDA. Cierra D3.

En el Atlas, ``psi_t`` (rentas de recursos del mundo / PIB del mundo) se lee del dato: el modelo
explica cómo se reparten las rentas entre países, no los ciclos globales. Aquí ``psi_t`` sale de
cuatro mercados mundiales, uno por clase, con estado propio en cada miembro del ensemble:

    clase     cantidad (TWh o valor a precios 2010)       precio real                  recurso último U
    oil       producción mundial de petróleo (EI/OWID)     USD 2019 / barril            X0 + reservas0 * g
    gas       producción mundial de gas                    media geométrica EE. UU./UE  X0 + reservas0 * g
    coal      producción mundial de carbón                 Australia (CMO)              X0 + 150 años de Q0 * g
    metals    cobre, hierro, aluminio, zinc, níquel, estaño (USGS) a precios del año base = origen;
              índice de precios CMO con pesos del año base (nunca un año posterior al origen)

Cada año ``t`` (después del origen), por clase:

    demanda   dlnQd_t = gY + di_t,  di_t = a0 + a1 dlnp_{t-1} + a2 di_{t-1}       (a1 <= 0)
              con di = dln(Q / PIB mundial): elasticidad renta 1 impuesta [S] y tendencia de intensidad
              estimada (con la elasticidad libre, el signo salía al revés y la constante absorbía el
              crecimiento de 1950-1990: la demanda de petróleo crecía al 4 % anual)
    capacidad Cap_t = R_MAX_RATE * (U - X_t)        no se puede extraer más del 10 % de lo que queda [S]
    cantidad  Q_t = min(Qd_t, Cap_t);  X_{t+1} = X_t + Q_t
    precio    lnp_t = m + phi (lnp_{t-1} - m) + e_t + KAPPA * ln(Qd_t / Cap_t)^+   (AR(1) + escasez [S])
              estacionario alrededor de la media histórica hasta el origen, sin deriva y phi <= 0.95 [S]:
              con datos <= 1980 salía una raíz unitaria con deriva de +7 % anual (los años setenta)
    renta     dln psi_c = theta dlnp_t + dlnQ_t - gY      (theta >= 0; sin constante: ninguna tendencia
              propia de la renta, que con datos 1971-1990 absorbería los años setenta), más una
              desviación de NIVEL AR(1) estacionaria (no un paseo aleatorio)

con ``gY`` el crecimiento del PIB mundial del L1 de ESE miembro (con un año de retraso: el L1
necesita ``psi_t`` antes de producir el año t). ``psi_t = sum_c psi_c + bosque`` (renta forestal,
renovable: congelada en su media de 10 años, como hacía el Atlas con toda la ``psi``).

Estimación con datos <= origen (demanda: los 20 años previos al origen, para no extrapolar la
intensidad creciente de 1950-1973; precios desde que existen; rentas por clase desde 1971); incertidumbre paramétrica y residual por miembro, como en F1. El ruido del precio usa
la desviación típica COMPLETA (no la robusta): los choques del petróleo son la historia, no colas.

Recurso último: ``g`` ~ log-uniforme [1, 3] por miembro [S] (crecimiento de reservas; con datos <=
1990 la linealización de Hubbert da para el petróleo un U MENOR que lo extraído hasta 2019, así que
no se usa). Sin reservas en el origen (antes de 1980): 30 años de producción (petróleo), 40 (gas) [S].

Lo que no está: metales críticos (litio, tierras raras: sin series), agotamiento de metales (sin
recurso último), la parte de cada país (la sigue repartiendo el Hubbert por país del Atlas), y
cualquier petróleo/gas no convencional como clase propia.
"""
from __future__ import annotations

import numpy as np

from world4.sectors import _panel

CLASSES = ("oil", "gas", "coal", "metals")
RENT_COL = {"oil": "renta_petroleo_pct_pib", "gas": "renta_gas_pct_pib", "coal": "renta_carbon_pct_pib",
            "metals": "renta_minerales_pct_pib", "forest": "renta_forestal_pct_pib"}
METALS = {  # nombre en el panel mundial: (producción USGS, precio CMO real 2010)
    "cobre": ("recursos__usgs_prod_mundial_cobre_t", "recursos__cmo_cobre_real2010"),
    "hierro": ("recursos__usgs_prod_mundial_hierro_t", "recursos__cmo_hierro_real2010"),
    "aluminio": ("recursos__usgs_prod_mundial_aluminio_t", "recursos__cmo_aluminio_real2010"),
    "zinc": ("recursos__usgs_prod_mundial_zinc_t", "recursos__cmo_zinc_real2010"),
    "niquel": ("recursos__usgs_prod_mundial_niquel_t", "recursos__cmo_niquel_real2010"),
    "estano": ("recursos__usgs_prod_mundial_estano_t", "recursos__cmo_estano_real2010"),
}
R_MAX_RATE = 0.10                        # extracción máxima por año / recurso restante [S]
KAPPA = 1.0                              # prima de precio por exceso de demanda sobre capacidad [S]
G_RANGE = (1.0, 3.0)                     # crecimiento de reservas, log-uniforme [S]
RP_FALLBACK = {"oil": 30.0, "gas": 40.0, "coal": 150.0}     # años de producción si no hay reservas [S]
THETA_PRIOR = (2.0, 0.5)
LEVEL_RHO, LEVEL_SD_MAX = 0.5, 0.5
DEMAND_WINDOW = 20                        # años previos al origen para la demanda (estructura reciente)
PHI_MAX = 0.95                            # persistencia máxima del precio real (estacionario) [S]
MIN_YEARS = 10                            # años mínimos de precio y demanda para que una clase sea endógena        # desviación de nivel de la renta por clase [S]                 # palanca de la renta sobre el precio si hay < 8 años de rentas [S]
SIGNS = {"demand": {"dlnp": -1}, "rent": {"dlnp": +1}}
MISSING = {
    "critical_metals": "critical metals (lithium, rare earths) have no series: not modelled",
    "metal_depletion": "metals have no ultimate resource: no depletion, only demand and price",
    "country_split": "the split of rents across countries is still the Atlas's per-country Hubbert",
}


# ------------------------------------------------------------------------------ datos
_DATA = {}


def data(W) -> dict:
    """Series mundiales (T,) por clase, del panel del Atlas: Q, p, psi_c, X, reservas, y gY."""
    key = id(W)
    if key in _DATA:
        return _DATA[key]
    w, ex, T = W.world, W.extra, W.T
    nan = np.full(T, np.nan)
    Y = np.where(W.observed, W.Y, np.nan)
    both = np.isfinite(Y[:, 1:]) & np.isfinite(Y[:, :-1])
    gY = nan.copy()
    gY[1:] = np.log(np.nansum(np.where(both, Y[:, 1:], 0), 0) / np.nansum(np.where(both, Y[:, :-1], 0), 0))

    def share(col):
        a = ex.get(col)
        if a is None:
            return nan.copy()
        a = a / 100.0
        ok = np.isfinite(a) & np.isfinite(Y)
        n = ok.sum(0)
        s = np.nansum(np.where(ok, a * Y, 0), 0) / np.maximum(np.nansum(np.where(ok, Y, 0), 0), 1e-12)
        return np.where(n >= 20, s, np.nan)

    out = {"gY": gY, "psi_obs": W.psi_t.astype(float), "forest": share(RENT_COL["forest"])}
    gas_p = np.exp(0.5 * (np.log(w.get("recursos__cmo_gas_eeuu_real2010", nan))
                          + np.log(w.get("recursos__cmo_gas_europa_real2010", nan))))
    prices = {"oil": w.get("recursos__petroleo_precio_real_usd2019_bbl", nan), "gas": gas_p,
              "coal": w.get("recursos__cmo_carbon_aus_real2010", nan)}
    for c in ("oil", "gas", "coal"):
        res = ex.get(f"{c}_reservas_twh")
        rsum = np.full(T, np.nan)
        if res is not None:
            n = np.isfinite(res).sum(0)
            rsum = np.where(n >= 30, np.nansum(res, 0), np.nan)
        out[c] = {"Q": w.get(f"recursos__mundo_{c}_prod_twh", nan).astype(float),
                  "X": w.get(f"recursos__mundo_{c}_cum_twh", nan).astype(float),
                  "p": prices[c].astype(float), "psi": share(RENT_COL[c]), "reserves": rsum}
    # metales: series en bruto; el índice se construye con el año base = origen (``metals_at``)
    out["metals_raw"] = {k: (w.get(q, nan).astype(float), w.get(pn, nan).astype(float)) for k, (q, pn) in METALS.items()}
    out["metals"] = {"psi": share(RENT_COL["metals"]), "X": nan.copy(), "reserves": nan.copy()}
    _DATA[key] = out
    return out


def metals_at(D, base_t: int) -> dict:
    """Volumen (a precios del año base) e índice de precios (pesos del año base) de los metales, con
    el año base = el último <= ``base_t`` con todos los metales: nada posterior al origen."""
    raw = D["metals_raw"]
    ok_years = np.all([np.isfinite(q) & np.isfinite(p) for q, p in raw.values()], 0)
    cand = np.where(ok_years[:base_t + 1])[0]
    b = int(cand[-1]) if len(cand) else int(np.where(ok_years)[0][0])
    vol = sum(q * p[b] for q, p in raw.values())
    wsum = sum(p[b] * q[b] for q, p in raw.values())
    pidx = sum(p * q[b] for q, p in raw.values()) / wsum
    return {**D["metals"], "Q": np.where(ok_years, vol, np.nan), "p": np.where(ok_years, pidx, np.nan), "base": 1950 + b}


def _cls(D, c, t0):
    return metals_at(D, t0) if c == "metals" else D[c]


def _d(a):
    o = np.full_like(a, np.nan)
    o[1:] = a[1:] - a[:-1]
    return o


def _l(a, k=1):
    o = np.full_like(a, np.nan)
    o[k:] = a[:-k]
    return o


FOSSIL = ("oil", "gas", "coal")


def fit(W, cutoff_t: int, fossil_driver=None) -> dict:
    """Demanda, precio y renta de cada clase con años <= ``cutoff_t``. Con ``fossil_driver`` (T,), el
    crecimiento del consumo fósil mundial (F3), la demanda de petróleo, gas y carbón se escribe como
    intensidad sobre la ENERGÍA fósil y no sobre el PIB: la mezcla energética de F3 la mueve."""
    D = data(W)
    gY = D["gY"]
    cy = 1950 + cutoff_t
    eqs = {}
    for c in CLASSES:
        s = _cls(D, c, cutoff_t)
        lq, lp = np.log(s["Q"]), np.log(s["p"])
        dq, dp = _d(lq), _d(lp)
        t = np.arange(2, cutoff_t + 1)
        one = np.zeros(len(t), int)

        def go(name, y, X, terms, signs=None):
            return _panel.fit(name, y, np.column_stack([np.ones(len(t))] + X), one, ("const",) + terms, cy,
                              signs=signs, n_countries=1, year=None)

        drv = fossil_driver if (fossil_driver is not None and c in FOSSIL) else gY
        di = dq - drv                      # cambio de la intensidad Q / PIB (o Q / energía fósil, F3)
        recent = t[t > cutoff_t - DEMAND_WINDOW]       # estructura reciente: los 20 años previos al origen
        td = np.isin(t, recent)
        Xd = np.column_stack([np.ones(len(t)), _l(dp)[t], _l(di)[t]])
        n_dem = int(np.sum(td & np.isfinite(di[t]) & np.all(np.isfinite(Xd), 1)))
        n_pr = int(np.sum(np.isfinite(lp[t]) & np.isfinite(_l(lp)[t])))
        if n_dem < MIN_YEARS or n_pr < MIN_YEARS:
            eqs[(c, "frozen")] = True            # sin datos suficientes en el origen: renta congelada
            continue
        eqs[(c, "demand")] = _panel.fit(f"{c}/demand", np.where(td, di[t], np.nan), Xd, one,
                                        ("const", "dlnp", "dln_intensity"), cy, signs=SIGNS["demand"], n_countries=1)
        okp = np.isfinite(lp[:cutoff_t + 1])
        m = float(np.nanmean(lp[:cutoff_t + 1])) if okp.any() else 0.0
        # AR(1) estacionario alrededor de la media histórica, sin constante y phi <= PHI_MAX [S]
        e_p = _panel.fit(f"{c}/price", lp[t] - m, (_l(lp)[t] - m)[:, None], one, ("lnp_1",), cy, n_countries=1)
        e_p.beta[0] = float(np.clip(e_p.beta[0], 0.0, PHI_MAX))
        e_p.meta["mean_lnp"] = m
        eqs[(c, "price")] = e_p
        y = _d(np.log(s["psi"]))[t] - dq[t] + gY[t]
        okr = np.isfinite(y) & np.isfinite(dp[t])
        if okr.sum() >= 8:       # sin constante: la renta / PIB sólo se mueve con precio y cantidad / PIB
            e_r = _panel.fit(f"{c}/rent", y, dp[t][:, None], one, ("dlnp",), cy, signs=SIGNS["rent"], n_countries=1)
            # El residuo es una desviación de NIVEL estacionaria (márgenes, medición del WDI), no un paseo
            # aleatorio. Su persistencia no se identifica con 10-40 años de rentas que empiezan en valores
            # minúsculos (gas en 1970: 0,04 % del PIB, x15 en siete años): rho = 0,5 y la escala robusta
            # del residuo, acotada a 0,5 en logaritmos [S].
            e_r.meta.update(level_rho=LEVEL_RHO, level_sd=float(min(e_r.sd, LEVEL_SD_MAX)))
            eqs[(c, "rent")] = e_r
        else:
            eqs[(c, "rent")] = _panel.Equation(f"{c}/rent", ("dlnp",), np.array([THETA_PRIOR[0]]),
                                               np.array([[THETA_PRIOR[1] ** 2]]), 0.05, int(okr.sum()), 1, cy,
                                               meta={"prior": True, "level_rho": LEVEL_RHO, "level_sd": 0.1})
    return eqs


# ------------------------------------------------------------------------------ simulación
class Markets:
    """Estado (R,) por clase; ``step(gY)`` avanza un año y devuelve ``psi`` (R, 1)."""

    def __init__(self, W, t0: int, eqs: dict, R: int, seed: int, noise: bool = True, g=None,
                 u_scale: float = 1.0, fossil_driver=None):
        self.W, self.t0, self.R, self.eqs, self.noise = W, t0, R, eqs, noise
        self.rng = np.random.default_rng(seed)
        D = data(W)
        self.D = D
        lo = max(0, t0 - 9)
        fr = D["forest"][lo:t0 + 1]
        self.forest = float(np.nanmean(fr)) if np.isfinite(fr).any() else 0.003
        if g is None:
            g = np.exp(self.rng.uniform(np.log(G_RANGE[0]), np.log(G_RANGE[1]), R)) if noise else \
                np.full(R, np.sqrt(G_RANGE[0] * G_RANGE[1]))
        self.st = {}
        self.frozen = {c for c in CLASSES if eqs.get((c, "frozen"))}
        for c in CLASSES:
            s = _cls(D, c, t0)
            st = {}
            Q0, p0 = float(s["Q"][t0]), float(s["p"][t0])
            st["lnQ"], st["lnp"] = np.full(R, np.log(Q0)), np.full(R, np.log(p0))
            st["lnp0"] = float(np.log(p0))
            st["dq"] = np.full(R, float(np.nan_to_num(np.log(s["Q"][t0] / s["Q"][t0 - 1]))))
            drv = fossil_driver if (fossil_driver is not None and c in FOSSIL) else D["gY"]
            st["di"] = st["dq"] - float(np.nan_to_num(drv[t0]))
            st["dp"] = np.full(R, float(np.nan_to_num(np.log(s["p"][t0] / s["p"][t0 - 1]))))
            psi0 = s["psi"][max(0, t0 - 2):t0 + 1]
            psi0 = float(psi0[np.isfinite(psi0)][-1]) if np.isfinite(psi0).any() else 0.001
            st["lnpsi"] = np.full(R, np.log(max(psi0, 1e-6)))
            if c in self.frozen:
                lo = max(0, t0 - 9)
                ps = s["psi"][lo:t0 + 1]
                st["lnpsi"] = np.full(R, np.log(max(float(np.nanmean(ps)) if np.isfinite(ps).any() else 0.001, 1e-6)))
                st["lnQ"], st["lnp"] = np.full(R, np.nan), np.full(R, np.nan)
                self.st[c] = st
                continue
            st["core"] = st["lnpsi"].copy()
            m_r = self.eqs[(c, "rent")].meta
            st["dev_shock"] = _panel.Shock(m_r["level_sd"], m_r["level_rho"], (R,), self.rng, on=noise)
            st["dev_shock"].x = np.zeros(R)                  # en el origen la renta es la observada
            if c == "metals":
                st["X"], st["U"] = np.zeros(R), np.full(R, np.inf)
            else:
                X0 = float(s["X"][t0])
                res = s["reserves"][t0]
                if not np.isfinite(res):
                    res = RP_FALLBACK[c] * Q0
                st["X"] = np.full(R, X0)
                st["U"] = X0 + res * g * u_scale
            st["B"] = {k: (self.eqs[(c, k)].draw(R, self.rng) if noise else
                           np.broadcast_to(self.eqs[(c, k)].beta, (R, len(self.eqs[(c, k)].beta))).copy())
                       for k in ("demand", "price", "rent")}
            st["B"]["price"][:, 0] = np.clip(st["B"]["price"][:, 0], 0.0, PHI_MAX)
            self.st[c] = st
        self.hist = {t0: self.psi()}
        self.prices = {t0: {c: np.exp(self.st[c]["lnp"]) for c in CLASSES}}
        self.quant = {t0: {c: np.exp(self.st[c]["lnQ"]) for c in CLASSES}}

    def psi(self):
        return (sum(np.exp(self.st[c]["lnpsi"]) for c in CLASSES) + self.forest)[:, None]

    def _z(self):
        return self.rng.standard_normal(self.R) if self.noise else np.zeros(self.R)

    def step(self, t: int, gY, gF=None, extra_metals=None):
        """Año ``t`` con el crecimiento del PIB mundial ``gY`` (R,) del año anterior. Con F3, ``gF`` (R,)
        es el crecimiento del consumo fósil mundial (mueve petróleo, gas y carbón) y ``extra_metals``
        (R,) la demanda de cobre y níquel de la transición (en unidades de volumen de metales)."""
        for c in CLASSES:
            if c in self.frozen:
                continue
            st, eq = self.st[c], self.eqs
            Bd, Bp, Br = st["B"]["demand"], st["B"]["price"], st["B"]["rent"]
            di = (Bd[:, 0] + Bd[:, 1] * st["dp"] + Bd[:, 2] * st["di"] + eq[(c, "demand")].sd * self._z())
            drv = gF if (gF is not None and c in FOSSIL) else gY
            Qd = np.exp(st["lnQ"] + np.clip(drv + di, -0.3, 0.3))
            if c == "metals" and extra_metals is not None:
                Qd = Qd + extra_metals
            st["Qd"] = Qd
            cap = R_MAX_RATE * np.maximum(st["U"] - st["X"], 0.0)
            Q = np.minimum(Qd, np.maximum(cap, 1e-9))
            m = eq[(c, "price")].meta["mean_lnp"]
            sdp = eq[(c, "price")].meta.get("sd_raw", eq[(c, "price")].sd)
            lnp = (m + Bp[:, 0] * (st["lnp"] - m) + sdp * self._z()
                   + KAPPA * np.log(np.maximum(Qd / np.maximum(cap, 1e-9), 1.0)))
            dp = lnp - st["lnp"]
            dq = np.log(Q) - st["lnQ"]
            st["core"] = st["core"] + Br[:, 0] * dp + dq - gY
            st["dev"] = st["dev_shock"]()
            st["lnpsi"] = np.minimum(st["core"] + st["dev"], np.log(0.5))
            st["X"] = st["X"] + Q
            st["lnQ"], st["lnp"], st["dq"], st["dp"], st["di"] = np.log(Q), lnp, dq, dp, dq - drv
            st["binding"] = Qd > cap
        self.hist[t] = self.psi()
        self.prices[t] = {c: np.exp(self.st[c]["lnp"]) for c in CLASSES}
        self.quant[t] = {c: np.exp(self.st[c]["lnQ"]) for c in CLASSES}
        return self.hist[t]


    def shortage(self):
        """Fracción (R, 1) de la demanda fósil mundial que la capacidad no cubrió este año."""
        qd = sum(self.st[c].get("Qd", np.exp(self.st[c]["lnQ"])) for c in FOSSIL if c not in self.frozen)
        q = sum(np.exp(self.st[c]["lnQ"]) for c in FOSSIL if c not in self.frozen)
        return np.clip(1 - q / np.maximum(qd, 1e-12), 0, 1)[:, None] if np.ndim(qd) else None

    def depletion(self):
        """Agotamiento fósil mundial (R, 1): X / U ponderado por la producción de cada clase."""
        num = den = 0.0
        for c in FOSSIL:
            if c in self.frozen:
                continue
            st = self.st[c]
            q = np.exp(st["lnQ"])
            num = num + q * np.clip(st["X"] / np.maximum(st["U"], 1e-12), 0, 1)
            den = den + q
        if np.ndim(den) == 0:                     # todas las clases fósiles congeladas en el origen
            return np.zeros((self.R, 1))
        return (num / np.maximum(den, 1e-12))[:, None]

    def log_fossil_price(self):
        """Log del índice de precio fósil (R, 1): media de los log-precios con pesos de cantidad."""
        num = den = 0.0
        for c in FOSSIL:
            if c in self.frozen:
                continue
            st = self.st[c]
            q = np.exp(st["lnQ"])
            num = num + q * (st["lnp"] - st["lnp0"])
            den = den + q
        if np.ndim(den) == 0:
            return np.zeros((self.R, 1))
        return (num / np.maximum(den, 1e-12))[:, None]


def standalone(W, t0: int, years: int, g_world: float = 0.02, R: int = 1, noise: bool = False, u_scale=1.0,
               seed: int = 0) -> dict:
    """Los mercados solos con el PIB mundial creciendo a ``g_world``: para relaciones metamórficas (W6)."""
    from world4.core.coupling import market_fits
    mk = Markets(W, t0, market_fits(W, t0), R, seed, noise=noise, u_scale=u_scale)
    for k in range(1, years + 1):
        mk.step(t0 + k, np.full(R, g_world))
    yrs = sorted(mk.quant)
    return {"years": [1950 + t for t in yrs],
            "Q": {c: np.array([mk.quant[t][c].mean() for t in yrs]) for c in CLASSES},
            "p": {c: np.array([mk.prices[t][c].mean() for t in yrs]) for c in CLASSES},
            "psi": np.array([float(mk.hist[t].mean()) for t in yrs])}
