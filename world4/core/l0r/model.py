"""L0 regional de F6: 10 regiones, 92 stocks, paso anual, la MISMA estructura que el L1 agregada.

Cada ecuación es la forma agregada de una del L1 y se estima con los mismos datos agregados por región
(``data.regional``), con datos <= ``cutoff``:

| Stock (por región)        | Flujos                                   | En el L1                                  |
|---------------------------|------------------------------------------|-------------------------------------------|
| P población               | nacimientos, defunciones, migración      | F1 (cohortes; aquí tasas brutas)          |
| ln natalidad bruta        | convergencia a un nivel según el ingreso | F1 (ecuación de la fecundidad)             |
| ln intensidad energética  | mejora autónoma + elasticidad al ingreso | F3 (ecuación de la intensidad)             |
| K capital                 | inversión s·Y, depreciación δK           | Atlas (canal k)                            |
| lnA PTF                   | convergencia a la región líder           | Atlas (difusión de la PTF)                 |
| D daño climático (log)    | h(T_r) - h(T̄_r), persistencia ρ_D        | Atlas (BHM, ``Dclim``)                     |
| z logit cuota no fósil    | difusión con el precio fósil             | F3 (cuotas con dinámica estimada)          |
| lnC tierras de cultivo    | expansión con la población               | F5                                         |
| lnYld rendimiento         | brecha + N + temperatura                 | F5 (coeficientes de temperatura de F5)     |
| lnN N mineral por ha      | convergencia a la frontera               | F5                                         |
| d deuda / PIB             | déficit, dilución por crecimiento        | Atlas (deuda)                              |
| **Mundo:** X extracción fósil acumulada, CO2 acumulado                                            |

Producción: lnY = lnA + D + α lnK + (1-α) lnP + E, con E el efecto de la energía (coste real con la EROI
fósil que cae con el agotamiento, y escasez física CES si la capacidad de extracción no cubre la
demanda; F2/F3). Energía = intensidad(ingreso) × Y; CO2 = κ × fósil; GMST = T_base + TCRE_ef ×
ΔCO2 acumulado (TCRE efectiva estimada en la historia, con la incertidumbre relativa del conjunto de
FaIR de F4). Comida: cereal per cápita = rendimiento × tierras × parte de cereal / P; su déficit sube
la mortalidad (si los datos lo identifican).

La mortalidad bruta depende del ingreso y de la natalidad (indicador de la estructura de edades: con
menos nacimientos la población envejece y la mortalidad bruta sube) [S]. 112 stocks: 11 por región + 2.

Todo se vectoriza sobre (miembros, regiones). Los flujos tienen nombre (``FLOWS``): ``loops.py`` los usa
para puntuar los lazos.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world4.core.l0r import data as rdata
from world4.core.l0r import regions

NR = rdata.NR
REG_STOCKS = ("P", "K", "lnA", "D", "z", "lnC", "lnYld", "lnN", "debt", "lcbr", "lint")
WORLD_STOCKS = ("X", "C")
# flujo -> (stock destino, signo): + entrada, - salida
FLOWS = {
    "births": ("P", 1), "deaths": ("P", -1), "migration": ("P", 1),
    "fertility_change": ("lcbr", 1), "efficiency_change": ("lint", 1),
    "investment": ("K", 1), "depreciation": ("K", -1),
    "tfp_catchup": ("lnA", 1),
    "damage_new": ("D", 1), "damage_decay": ("D", -1),
    "nf_diffusion": ("z", 1),
    "cropland_growth": ("lnC", 1),
    "yield_growth": ("lnYld", 1),
    "fert_growth": ("lnN", 1),
    "deficit": ("debt", 1), "debt_dilution": ("debt", -1),
    "extraction": ("X", 1),
    "emissions": ("C", 1),
}
CPC_MIN = 250.0          # kg de cereal por persona y año: umbral del déficit para la mortalidad [S]
RES_GROWTH = 2.0         # crecimiento de reservas (F2 usa log-uniforme [1, 3]; aquí su mediana) [S]
COAL_RP = 150.0          # años de producción de carbón si no hay reservas (como F2) [S]
R_MAX = 0.10             # tasa máxima de extracción del recurso restante (F2) [S]
KAPPA_P = 1.0            # precio fósil ~ (1-x0)/(1-x) [S]
SUBST = 1.0              # Δz += SUBST × escasez fósil: sustitución bajo escasez FÍSICA, no identificable en la
                         # historia (el precio fósil no movió la mezcla: coeficiente 0) [S]
SHRINK = 5.0             # efecto propio de región: residuo medio de 20 años × n/(n + SHRINK) (como ``_panel``) [S]


def _clip(x, lo, hi):
    """``np.clip`` sin su sobrecoste (unos 2 µs menos por llamada en arrays de 10 elementos)."""
    return np.minimum(np.maximum(x, lo), hi)


def _ols(y, X):
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    b, *_ = np.linalg.lstsq(X[ok], y[ok], rcond=None)
    e = y[ok] - X[ok] @ b
    return b, float(np.std(e)), int(ok.sum())


@dataclass
class Params:
    cutoff: int
    ly_range: tuple            # ingreso observado (min, max) en el panel: las tasas no se extrapolan fuera [S]
    a_b: np.ndarray            # Δln cbr = a_b + b1 ln cbr + b2 ly
    b_b: tuple
    a_d: np.ndarray            # ln cdr = a_d + c1 ly + c2 ly^2 + c3 short + c4 ln cbr
    c_d: tuple
    mig: np.ndarray
    s: np.ndarray
    delta: np.ndarray
    g_A: float
    lam_A: float
    u_A: np.ndarray            # efecto propio de la PTF por región (encogido)
    a_i: np.ndarray            # Δln(PE/Y) = a_i + b_i Δly
    b_i: float
    z_coef: tuple              # dz = a + b z + d lpF (a por región)
    c_coef: tuple              # dlnC = a + b dlnP (a por región)
    n_coef: tuple              # dlnN = b1 (qN - lnN) + b2 dly
    qN: float
    y_coef: tuple              # dlnYld = c + gap + n (c por región)
    yT: tuple                  # temperatura (de F5): dT, dT_hot
    cereal_share: np.ndarray   # superficie de cereal / tierras de cultivo
    deficit: np.ndarray
    alpha: float
    bhm: tuple
    rho_D: float
    tbar: np.ndarray           # temperatura media 1951-1980 por región
    beta_t: np.ndarray
    tcre_eff: float            # °C por GtCO2 (efectiva, historia)
    t_icpt: float              # GMST = t_icpt + tcre_eff * CO2 acumulado (la recta de la historia)
    afolu: float
    kappa_co2: float           # GtCO2 por TWh fósil
    theta_E: float
    sigma_E: float
    eroi0: dict
    fits: dict = field(default_factory=dict)


def calibrate(W, cutoff: int = 2019, base_year: int = rdata.BASE_DEFAULT) -> Params:
    """Estima el L0 con datos <= ``cutoff`` del panel regional (los mismos que el L1)."""
    from world4.core.l1_atlas import model as amodel
    from world4.sectors import energy, food
    R = rdata.regional(W, base_year)
    L, I = R["levels"], R["logindex"]
    tb = R["base"]
    T = W.T
    tc = cutoff - 1950
    t1 = 1971 - 1950
    yrs = slice(t1, tc + 1)
    lpop = np.log(L["pop"][:, tb])[:, None] + I["pop"]
    lY = np.log(L["Y"][:, tb])[:, None] + I["Y"]
    lK = np.log(L["K"][:, tb])[:, None] + I["K"]
    ly = lY - lpop
    alpha = float(amodel.ALPHA)
    lA = lY - alpha * lK - (1 - alpha) * lpop
    fits = {}
    reg = np.repeat(np.arange(NR)[:, None], T, 1)

    def panel(*arrs):
        return [a[:, yrs].ravel() for a in arrs]

    def own(yfull, regs_full, b):
        """Efecto propio de cada región en una ecuación en diferencias (residuo de los 20 años previos)."""
        pred = b[0] + sum(bk * r for bk, r in zip(b[1:], regs_full))
        e = (yfull - pred)[:, tc - 19:tc + 1]
        nn = np.isfinite(e).sum(1)
        return np.where(nn > 0, np.nansum(e, 1) / (nn + SHRINK), 0.0)

    def fe(yv, regs):
        rr = reg[:, yrs].ravel()
        X = np.column_stack([np.eye(NR)[rr]] + regs)
        return _ols(yv, X)

    # natalidad y mortalidad (tasas brutas, efectos de región)
    cereal_pc = np.exp(np.log(np.maximum(L["cereal"][:, tb], 1))[:, None] + I["cereal"] - lpop) * 1e3 / 1e6  # kg/persona
    short = np.maximum(np.log(CPC_MIN) - np.log(cereal_pc), 0.0)
    lcbr = np.log(L["cbr"])
    dcb = np.diff(lcbr, axis=1, prepend=np.nan)
    yv, lc1, ly1 = panel(dcb, np.roll(lcbr, 1, 1), np.roll(ly, 1, 1))
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(lc1), lc1, ly1]))
    a_b, b_b = float(b[0]) + own(dcb, [np.roll(lcbr, 1, 1), np.roll(ly, 1, 1)], b), (float(b[1]), float(b[2]))
    fits["cbr"] = (b, sd, n)
    yv, l1, l2, lcb, sh = panel(np.log(L["cdr"]), ly, ly ** 2, lcbr, short)
    b, sd, n = fe(yv, [l1, l2, lcb, sh])
    if b[NR + 3] < 0:                                   # signo declarado: el déficit no reduce la mortalidad
        b, sd, n = fe(yv, [l1, l2, lcb])
        b = np.append(b, 0.0)
    a_d, c_d = b[:NR], (b[NR], b[NR + 1], b[NR + 3], b[NR + 2])
    fits["cdr"] = (b, sd, n)
    last = lambda a, k: np.nanmean(a[:, tc - k + 1:tc + 1], 1)  # noqa: E731
    mig = np.nan_to_num(last(L["mig"], 10))
    s = last(L["csh_i"], 20)
    delta = last(L["delta"], 20)
    # PTF: convergencia a la región líder
    dA = np.diff(lA, axis=1, prepend=np.nan)
    gap = np.nanmax(lA, 0)[None] - lA
    yv, gp = panel(dA, np.roll(gap, 1, 1))
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(gp), gp]))
    g_A, lam_A = float(b[0]), float(max(b[1], 0.0))
    fits["tfp"] = (b, sd, n)
    # efecto propio de cada región: residuo medio de los 20 años previos, encogido n/(n + 10) (como
    # ``_panel``): Asia oriental y meridional crecen más de lo que da la convergencia común
    u_A = own(dA, [np.roll(gap, 1, 1)], (g_A, lam_A))
    # intensidad energética
    lpe = np.log(np.maximum(L["pe"][:, tb], 1))[:, None] + I["pe"]
    lint = lpe - lY
    dli, dlyf = np.diff(lint, axis=1, prepend=np.nan), np.diff(ly, axis=1, prepend=np.nan)
    yv, l1 = panel(dli, dlyf)
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(l1), l1]))
    a_i, b_i = float(b[0]) + own(dli, [dlyf], b), float(b[1])
    fits["intensity"] = (b, sd, n)
    # cuota no fósil (logit) con el precio fósil (índice del panel de F3)
    nfs = L["nonfossil"] / L["pe"]
    z = np.log(nfs / (1 - nfs))
    lpF = energy.panel(W)["lpF"]
    dz = np.diff(z, axis=1, prepend=np.nan)
    yv, zl, pl = panel(dz, np.roll(z, 1, 1), np.broadcast_to(np.roll(lpF, 1)[None], (NR, T)))
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(zl), zl, pl]))
    z_coef = (float(b[0]) + own(dz, [np.roll(z, 1, 1), np.broadcast_to(np.roll(lpF, 1)[None], (NR, T))], b),
              float(b[1]), float(max(b[2], 0.0)))
    fits["nonfossil"] = (b, sd, n)
    # tierras de cultivo
    yv, dp = panel(R["chain"]["cropland"], R["chain"]["pop"])
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(dp), dp]))
    c_coef = (float(b[0]) + own(R["chain"]["cropland"], [R["chain"]["pop"]], b), float(b[1]))
    fits["cropland"] = (b, sd, n)
    # N mineral por ha y rendimiento
    lC = np.log(np.maximum(L["cropland"][:, tb], 1))[:, None] + I["cropland"]
    lnf = np.log(np.maximum(L["nfert"][:, tb], 1))[:, None] + I["nfert"]
    lN = np.log(np.exp(lnf - lC) + 1.0)                 # ln(kg N por ha + 1): toneladas / miles de ha
    qN = float(np.nanmax(lN[:, tc]))
    dN = np.diff(lN, axis=1, prepend=np.nan)
    dly = np.diff(ly, axis=1, prepend=np.nan)
    yv, gN, dl = panel(dN, qN - np.roll(lN, 1, 1), dly)
    b, sd, n = _ols(yv, np.column_stack([gN, dl]))
    n_coef = (float(max(b[0], 0.0)), float(b[1]))
    fits["nfert"] = (b, sd, n)
    lca = np.log(np.maximum(L["cereal_area"][:, tb], 1))[:, None] + I["cereal_area"]
    lce = np.log(np.maximum(L["cereal"][:, tb], 1))[:, None] + I["cereal"]
    lYld = lce - lca
    dY = np.diff(lYld, axis=1, prepend=np.nan)
    gapY = np.nanmax(lYld, 0)[None] - lYld
    yv, gy, dn = panel(dY, np.roll(gapY, 1, 1), dN)
    b, sd, n = _ols(yv, np.column_stack([np.ones_like(gy), gy, dn]))
    y_coef = (float(b[0]) + own(dY, [np.roll(gapY, 1, 1), dN], b), float(max(b[1], 0.0)), float(max(b[2], 0.0)))
    fits["yield"] = (b, sd, n)
    from world4.core.coupling import food_fits
    fy = food_fits(W, np.log(W.Y / W.pop), cutoff - 1950)["yield"]
    yT = (float(fy.beta[3]), float(fy.beta[4]))
    cereal_share = np.exp(lca[:, tc] - lC[:, tc]) / 1000.0         # ha de cereal / ha de cultivo
    # deuda: déficit medio que reproduce la deuda de los 20 años previos
    d = L["debt"]
    gYn = np.diff(lY, axis=1, prepend=np.nan)
    defi = np.diff(d, axis=1, prepend=np.nan) + gYn * np.roll(d, 1, 1)
    deficit = np.clip(np.nan_to_num(last(defi, 20)), -0.05, 0.05)
    # clima
    p0 = amodel.Params()
    tm = L["tmp"]
    tbar = np.nanmean(tm[:, 1:31], 1)
    tbar = np.where(np.isfinite(tbar), tbar, np.nanmean(tm, 1))
    w = rdata.world(W)
    m = np.arange(1970 - 1950, tc + 1)
    ok = np.isfinite(w["gmst"][m]) & np.isfinite(w["cum_co2"][m])
    tcre_eff, t_icpt = (float(v) for v in np.polyfit(w["cum_co2"][m][ok], w["gmst"][m][ok], 1))
    afolu = float(np.nanmean(w["co2_afolu"][tc - 9:tc + 1]))
    F = L["pe"] - L["nonfossil"]
    kappa = float(w["co2_ffi"][tc] / np.nansum(F[:, tc]))
    ly_obs = ly[:, yrs]
    return Params(cutoff, (float(np.nanmin(ly_obs)), float(np.nanmax(ly_obs))), a_b, b_b, a_d, c_d, mig, s, delta, g_A, lam_A, u_A, a_i, b_i, z_coef, c_coef, n_coef, qN,
                  y_coef, yT, cereal_share, deficit, alpha, (p0.bhm1, p0.bhm2), p0.rho_D, tbar,
                  np.clip(np.nan_to_num(L["beta_t"][:, tc], nan=1.33), 0.5, 2.5), tcre_eff, t_icpt, afolu, kappa,
                  energy.THETA_E, energy.SIGMA_E, {"fossil": energy.EROI0["fossil"], "nonfossil": 15.0}, fits)


@dataclass
class Initial:
    year: int
    reg: dict                   # stock -> (10,)
    world: dict                 # X, C, U, Tg0, C0, lpF0
    tmp0: np.ndarray
    obs: dict = None            # tasas observadas en el año inicial (ly, cbr, cdr, ln intensidad) para anclar


def initial(W, year: int, base_year: int = rdata.BASE_DEFAULT) -> Initial:
    """Stocks en ``year`` desde los datos (los países del conjunto fijo del año base)."""
    R = rdata.regional(W, base_year)
    L, I = R["levels"], R["logindex"]
    tb, t = R["base"], year - 1950
    lev = lambda name: np.log(np.maximum(L[name][:, tb], 1))[:, None] + I[name]  # noqa: E731
    P = np.exp(lev("pop")[:, t])
    lC = lev("cropland")[:, t]
    lN = np.log(np.exp(lev("nfert")[:, t] - lC) + 1.0)
    lYld = lev("cereal")[:, t] - lev("cereal_area")[:, t]
    nfs = L["nonfossil"][:, t] / L["pe"][:, t]
    from world4.core.l1_atlas import model as amodel
    alpha = float(amodel.ALPHA)
    lY, lK = lev("Y")[:, t], lev("K")[:, t]
    reg = {"P": P, "K": np.exp(lK), "lnA": lY - alpha * lK - (1 - alpha) * np.log(P), "D": np.zeros(NR),
           "z": np.log(nfs / (1 - nfs)), "lnC": lC, "lnYld": lYld, "lnN": lN,
           "debt": np.nan_to_num(L["debt"][:, t], nan=0.5), "lcbr": np.log(L["cbr"][:, t]),
           "lint": np.log(np.maximum(L["pe"][:, tb], 1)) + I["pe"][:, t] - lY}
    w = rdata.world(W)
    res = w["reserves"]
    U = (w["X"][t] + RES_GROWTH * sum(np.nan_to_num(res[c][t]) for c in ("oil", "gas"))
         + COAL_RP * float(__import__("world4.sectors.resources", fromlist=["data"]).data(W)["coal"]["Q"][t]))
    from world4.sectors import energy
    world = {"X": float(w["X"][t]), "C": float(w["cum_co2"][t]), "U": float(U), "Tg0": float(w["gmst"][t]),
             "C0": float(w["cum_co2"][t]), "X0": float(w["X"][t]), "lpF0": float(energy.panel(W)["lpF"][t])}
    obs = {"ly": lY - np.log(P), "cbr": L["cbr"][:, t], "cdr": L["cdr"][:, t],
           "lint": np.log(np.maximum(L["pe"][:, tb], 1)) + I["pe"][:, t] - lY}
    return Initial(year, reg, world, L["tmp"][:, t].copy(), obs)


MULTS = ("tcre", "resource", "s", "cbr", "lam_A", "damage", "eroi", "deficit", "yield_T", "subst", "damage_sign", "cdr")
# desplazamientos aditivos (por año) de las ecuaciones en diferencias: las perillas de F7 (calib.l0r)
SHIFTS = ("d_gA", "d_int", "d_cbr", "d_crop", "d_yield", "d_z")
# F8: palancas de política (``scenarios.levers`` les da caja y unidades) y la eficacia incierta de cada grupo
POLICY = ("carbon_price", "renewables", "efficiency", "agri_rd", "aid", "fertility", "consolidation")
EFFECT = ("pe_carbon", "pe_energy", "pe_agri", "pe_social")
# Efectos de las palancas [M]: elegidos, no estimados (la historia 1992-2019 no tiene precio del carbono
# mundial ni programas de esta escala). Su incertidumbre entra como ``EFFECT`` (factor por estado del mundo)
P_REF = 100.0          # USD/tCO2 que duplican el coste fósil efectivo: log(1 + τ / P_REF)
DW = 0.3               # pérdida de PIB por unidad de recaudación del carbono (recaudación reciclada)
EPS_Z = 0.05           # Δ logit cuota no fósil por año por unidad de log(1 + τ / P_REF)
EPS_I = 0.01           # Δ ln intensidad por año por unidad de log(1 + τ / P_REF), sobre la parte fósil
EPS_R = 0.04           # Δ logit cuota no fósil por año por 1 % del PIB en renovables
EPS_E = 0.005          # Δ ln intensidad por año por 1 % del PIB en eficiencia
EPS_Y = 0.003          # Δ ln rendimiento por año por 0,1 % del PIB en I+D agrícola
EPS_F = 0.01           # Δ ln natalidad bruta por año por 1 % del PIB (educación, planificación familiar)
AID_EFF = 0.5          # parte de la ayuda que llega como capital
CONSOL_COST = 0.5      # inversión que se pierde por unidad de déficit recortado
RAMP = 10              # años hasta el efecto pleno (desde el primer año de la corrida)
DONORS = ("NAM", "EUR", "OPA")
RECIPIENTS = ("SSA", "SAS")
FERT_REGIONS = ("SSA", "SAS", "MEA")


class L0R:
    """El modelo, con parámetros ``p`` e inicio ``ini``. ``M`` miembros (los multiplicadores ``mult`` son
    arrays (M,) por parámetro, por defecto 1; los desplazamientos ``SHIFTS`` van en el mismo dict, por
    defecto 0)."""

    def __init__(self, p: Params, ini: Initial, M: int = 1, mult: dict | None = None, policy: dict | None = None):
        self.p, self.ini, self.M = p, ini, M
        mult = dict(mult or {})
        policy = {**{k: mult.pop(k) for k in list(mult) if k in POLICY or k in EFFECT}, **(policy or {})}
        unknown = (set(mult) - set(MULTS) - set(SHIFTS)) | (set(policy) - set(POLICY) - set(EFFECT))
        if unknown:
            raise KeyError(f"unknown L0R parameters {sorted(unknown)}")
        one, zero = np.ones((M, 1)), np.zeros((M, 1))
        self.m = {k: (np.asarray(mult[k], float).reshape(M, 1) if k in mult else one) for k in MULTS}
        self.m.update({k: (np.asarray(mult[k], float).reshape(M, 1) if k in mult else zero) for k in SHIFTS})
        col = lambda v: np.broadcast_to(np.asarray(v, float).reshape(-1, 1), (M, 1)).copy()  # noqa: E731
        self.pol = ({k: col(policy.get(k, 0.0)) for k in POLICY} | {k: col(policy.get(k, 1.0)) for k in EFFECT}
                    if any(np.any(np.asarray(policy.get(k, 0.0)) != 0) for k in POLICY) else None)
        self._ramp = 0.0
        # anclaje: la diferencia entre el dato y la ecuación en el año inicial se mantiene (como el
        # reanclaje del L1 en el origen); GMST en la recta de la historia (sin el año de El Niño o Pinatubo)
        o, ly0 = ini.obs, ini.obs["ly"]
        self.off = {"cdr": np.nan_to_num(np.log(o["cdr"]) - (p.a_d + p.c_d[0] * ly0 + p.c_d[1] * ly0 ** 2
                                                              + p.c_d[3] * np.log(o["cbr"])))}
        self.Tg0 = p.t_icpt + p.tcre_eff * ini.world["C0"]

    # -------------------------------------------------------------- estado
    def x0(self) -> dict:
        M = self.M
        st = {k: np.broadcast_to(v[None], (M, NR)).astype(float).copy() for k, v in self.ini.reg.items()}
        st["X"] = np.full((M, 1), self.ini.world["X"])
        st["C"] = np.full((M, 1), self.ini.world["C"])
        return st

    # -------------------------------------------------------------- auxiliares y flujos
    def _consts(self):
        """Constantes de la corrida precalculadas (W10: el L0 debe correr a 2100 en < 10 ms)."""
        p, m, w = self.p, self.m, self.ini.world
        U = w["U"] * m["resource"] + (w["X0"] - w["X0"] * m["resource"])          # mismo X0, recurso escalado
        x0 = _clip(w["X0"] / np.maximum(U, 1e-9), 0, 0.999)
        rho = (p.sigma_E - 1) / p.sigma_E
        h0 = p.bhm[0] * self.ini.tmp0 + p.bhm[1] * self.ini.tmp0 ** 2
        self.k = dict(U=U, x0=x0, lpF0=w["lpF0"] + KAPPA_P * np.log(1 - x0), eroi_f0=p.eroi0["fossil"] * m["eroi"],
                      inv_nf=1.0 / p.eroi0["nonfossil"], rho=rho, inv_rho=1.0 / rho, a1=1 - p.alpha,
                      tcre=p.tcre_eff * m["tcre"], Tg0=self.Tg0, C0=w["C0"], tmp0=self.ini.tmp0[None],
                      beta_t=p.beta_t[None], share=p.cereal_share[None], lcpc=np.log(CPC_MIN),
                      a_d=(p.a_d + self.off["cdr"])[None] + np.log(m["cdr"]), s=p.s[None] * m["s"], delta=p.delta[None],
                      g_A=p.g_A + p.u_A[None] + m["d_gA"], lam=p.lam_A * m["lam_A"], h0=h0[None],
                      dmg=m["damage"] * m["damage_sign"],
                      decay=1 - p.rho_D, mig=p.mig[None], a_b=p.a_b[None] + m["d_cbr"], a_i=p.a_i[None] + m["d_int"],
                      z0=p.z_coef[0] + m["d_z"], subst=SUBST * m["subst"], c0=p.c_coef[0] + m["d_crop"],
                      y0=p.y_coef[0] + m["d_yield"],
                      yTh=m["yield_T"] * (p.yT[0] + p.yT[1] * (p.tbar[None] - 15) / 10),
                      deficit=p.deficit[None] * m["deficit"], cbr_m=m["cbr"], food=p.c_d[2] != 0.0)
        if self.pol is not None:
            q = self.pol
            mask = lambda codes: np.array([c in codes for c in regions.CODES], float)[None]  # noqa: E731
            self.k["pol"] = dict(
                lcp=np.log1p(np.maximum(q["carbon_price"], 0) / P_REF), tau=q["carbon_price"],
                spend=q["renewables"] + q["efficiency"] + q["agri_rd"] + q["fertility"] * mask(FERT_REGIONS),
                aid_out=q["aid"] * mask(DONORS), recip=mask(RECIPIENTS), fmask=mask(FERT_REGIONS))

    def aux(self, st: dict) -> dict:
        p, k = self.p, getattr(self, "k", None)
        if k is None:
            self._consts()
            k = self.k
        P, K = st["P"], st["K"]                                    # positivos por construcción
        x = np.minimum(st["X"] / k["U"], 0.999)
        lpF = k["lpF0"] - KAPPA_P * np.log(1 - x)
        nfs = 1 / (1 + np.exp(-st["z"]))
        eroi_f = k["eroi_f0"] * np.maximum((1 - x) / (1 - k["x0"]), 0.05)        # (1-x)/(1-x0) <= 1 si x >= x0
        net = 1 - (1 - nfs) / eroi_f - nfs * k["inv_nf"]
        net0 = getattr(self, "_net0", None)
        if net0 is None:
            self._net0 = net0 = net.copy()
        lP = np.log(P)
        lY0 = st["lnA"] + st["D"] + p.alpha * np.log(K) + k["a1"] * lP
        Y0 = np.exp(lY0)
        E = np.exp(st["lint"]) * Y0                                # energía primaria (TWh)
        F = E * (1 - nfs)
        Fw = F.sum(1, keepdims=True)
        cap = R_MAX * np.maximum(k["U"] - st["X"], 0.0)
        short = np.minimum(np.maximum(1 - cap / Fw, 0.0), 0.95)
        if short.any():
            avail = _clip(1 - (1 - nfs) * short, 0.05, 1.0)
            ces = np.log(1 - p.theta_E + p.theta_E * avail ** k["rho"]) * k["inv_rho"]
        else:
            ces = 0.0
        lY = lY0 - p.theta_E * np.log(net0 / np.maximum(net, 1e-3)) + ces
        if self.pol is not None and self._ramp > 0:            # F8: pérdida por el precio del carbono
            q = k["pol"]
            rev = q["tau"] * p.kappa_co2 * F * 1e3 / Y0         # recaudación / PIB (F en TWh, Y en millones USD)
            lY = lY - self._ramp * DW * rev
        Y = np.exp(lY)
        Tg = k["Tg0"] + k["tcre"] * (st["C"] - k["C0"])
        Treg = k["tmp0"] + k["beta_t"] * (Tg - k["Tg0"])
        cpc = np.exp(st["lnYld"] + st["lnC"]) * k["share"] / P          # kg de cereal por persona
        return {"P": P, "K": K, "Y": Y, "ly": lY - lP, "lP": lP, "E": E, "F": F, "Fw": Fw, "cap": cap,
                "short": short, "lpF": lpF, "Tg": Tg, "Treg": Treg, "cpc": cpc,
                "food_short": np.maximum(k["lcpc"] - np.log(cpc), 0.0) if k["food"] else 0.0, "x": x, "nfs": nfs,
                "net": net}

    def flows(self, st: dict, a: dict | None = None, prev: dict | None = None) -> dict:
        """Flujos del año (todos (M, 10) o (M, 1)). ``prev`` lleva lo que depende del año anterior (Δly,
        ΔlnP, Treg) para los flujos en diferencias."""
        p = self.p
        a = self.aux(st) if a is None else a
        k = self.k
        prev = prev or {}
        ly, P, lP = a["ly"], a["P"], a["lP"]
        lyc = np.minimum(np.maximum(ly, p.ly_range[0]), p.ly_range[1])    # sin extrapolar el polinomio
        lcbr = st["lcbr"]
        cdr = np.exp(k["a_d"] + (p.c_d[0] + p.c_d[1] * lyc) * lyc + p.c_d[3] * lcbr
                     + (p.c_d[2] * a["food_short"] if k["food"] else 0.0))
        dly = ly - prev.get("ly", ly)
        dlnP = lP - prev.get("lnP", lP)
        dlnN = p.n_coef[0] * (p.qN - st["lnN"]) + p.n_coef[1] * dly
        Treg = a["Treg"]
        dT = Treg - prev.get("Treg", Treg)
        lnA, lnYld = st["lnA"], st["lnYld"]
        mn, mx = np.minimum, np.maximum
        ext = mn(a["Fw"], a["cap"])
        fl = {
            "births": np.exp(lcbr) * k["cbr_m"] * P, "deaths": cdr * P, "migration": k["mig"] * P,
            "fertility_change": mn(mx(k["a_b"] + p.b_b[0] * lcbr + p.b_b[1] * lyc, -0.1), 0.1),
            "efficiency_change": mn(mx(k["a_i"] + p.b_i * dly, -0.1), 0.1),
            "investment": k["s"] * a["Y"], "depreciation": k["delta"] * a["K"],
            "tfp_catchup": k["g_A"] + k["lam"] * (lnA.max(1, keepdims=True) - lnA),
            "damage_new": k["dmg"] * (p.bhm[0] * Treg + p.bhm[1] * Treg * Treg - k["h0"]),   # 0 en el año inicial
            "damage_decay": k["decay"] * st["D"],
            "nf_diffusion": mn(mx(k["z0"] + p.z_coef[1] * st["z"] + p.z_coef[2] * a["lpF"] + k["subst"] * a["short"],
                                  -0.5), 0.5),
            "cropland_growth": mn(mx(k["c0"] + p.c_coef[1] * dlnP, -0.1), 0.1),
            "yield_growth": mn(mx(k["y0"] + p.y_coef[1] * (lnYld.max(1, keepdims=True) - lnYld) + p.y_coef[2] * dlnN
                                  + k["yTh"] * dT, -0.3), 0.3),
            "fert_growth": mn(mx(dlnN, -0.3), 0.3),
            "deficit": k["deficit"], "debt_dilution": (dly + dlnP) * st["debt"],
            "extraction": ext,
            "emissions": p.kappa_co2 * ext + p.afolu,
        }
        if self.pol is not None and self._ramp > 0:
            self._policy_flows(fl, st, a)
        return fl

    def _policy_flows(self, fl, st, a):
        """F8: los efectos de las palancas (con rampa y con la eficacia incierta de cada grupo)."""
        q, e, r = self.k["pol"], self.pol, self._ramp
        Y = a["Y"]
        spend = r * (q["spend"] + q["aid_out"])
        cons = np.maximum(self.k["deficit"], 0.0) * r * e["consolidation"] * CONSOL_COST
        aid = r * AID_EFF * np.sum(q["aid_out"] * Y, 1, keepdims=True)
        pr = a["P"] * q["recip"]
        aid_in = aid * pr / np.maximum(pr.sum(1, keepdims=True), 1e-12)
        fl["investment"] = np.maximum(fl["investment"] - (spend + cons) * Y + aid_in, 0.0)
        fl["nf_diffusion"] = fl["nf_diffusion"] + r * (e["pe_carbon"] * EPS_Z * q["lcp"]
                                                       + e["pe_energy"] * EPS_R * e["renewables"] / 0.01)
        fl["efficiency_change"] = fl["efficiency_change"] - r * (e["pe_carbon"] * EPS_I * q["lcp"] * (1 - a["nfs"])
                                                                 + e["pe_energy"] * EPS_E * e["efficiency"] / 0.01)
        fl["yield_growth"] = fl["yield_growth"] + r * e["pe_agri"] * EPS_Y * e["agri_rd"] / 0.001
        fl["fertility_change"] = fl["fertility_change"] - r * e["pe_social"] * EPS_F * e["fertility"] / 0.01 * q["fmask"]
        fl["deficit"] = fl["deficit"] * (1 - r * e["consolidation"])

    def _consumption(self, st, a):
        """Consumo (M, 10) = (1 - s) Y: las palancas se pagan con inversión (Y = C + I + gasto + ayuda), así que
        su coste aparece como menos capital y menos PIB después, no como menos consumo hoy."""
        return (1 - self.k["s"]) * a["Y"]

    # -------------------------------------------------------------- corrida
    def run(self, years: int, record=("P", "Y", "ly", "E", "F", "Tg", "cpc", "nfs", "short", "x"),
            keep_states: bool = False) -> dict:
        st = self.x0()
        self._net0 = None
        out = {}
        states = []
        a = self.aux(st)
        prev = {"ly": a["ly"], "lnP": a["lP"], "Treg": a["Treg"]}
        items = tuple(FLOWS.items())
        for k in range(years + 1):
            self._ramp = min(k / RAMP, 1.0) if self.pol is not None else 0.0
            if k:
                a = self.aux(st)
            for name in record:
                v = a[name] if name in a else (self._consumption(st, a) if name == "C" else st[name])
                if k == 0:
                    out[name] = np.empty((years + 1,) + v.shape)
                out[name][k] = v
            if keep_states:
                states.append({kk: v.copy() for kk, v in st.items()})
            if k == years:
                break
            fl = self.flows(st, a, prev)
            prev = {"ly": a["ly"], "lnP": a["lP"], "Treg": a["Treg"]}
            for f, (target, sign) in items:
                st[target] = st[target] + fl[f] if sign > 0 else st[target] - fl[f]
        out["years"] = self.ini.year + np.arange(years + 1)
        if keep_states:
            out["states"] = states
        return out
