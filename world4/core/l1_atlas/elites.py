"""Modelo empírico regionalizado de captura de élites (E = cuota Top-10 % WID) para Phy-World L1.

Sustituye la ecuación heredada de ``model.py`` (que imponía un término fijo ``-0.02 * conf``
sin sustento empírico, reversión hacia un ``E0`` congelado en el año de entrada y varianza nula)
por un panel jerárquico regional (10 regiones IAM: ``NAM, EUR, OPA, REF, CHN, SAS, SEA, LAM,
MEA, SSA``) estimado estrictamente con datos ``<= cutoff`` (sin fuga temporal):

  ΔE_{i,t} = b0 + u_{r(i)} + (b_E + Δλ_{r(i)})·(E_{i,t-1} - 0.45)
             + β' X_{i,t-1} + γ_ref·wave_ref + γ_eeur·wave_eeur
             + u^{reg}_{r(i),t} + u^{id}_{i,t}

donde:
  - ``u_r`` y ``Δλ_r`` tienen regularización jerárquica (Empirical Bayes / Ridge) hacia la media
    global y cota de estabilidad (``b_E + Δλ_r <= -0.008``), capturando derivas regionales
    divergentes (alza estructural en Occidente/NAM/OPA y Asia emergente vs. compresión gradual
    desde extremos en LATAM/SSA/MEA) sin sobreajustar regiones pequeñas.
  - ``X_{i,t-1}`` acopla covariables endógenas del simulador: poliarquía continua (``poly``),
    centralidad (``core``), brecha de Piketty ``r* - g``, rentas de recursos (``res_share``),
    apertura comercial (``open``), esfuerzo redistributivo (``a["w"]``) y conflicto (``conf``).
  - ``wave_ref`` (1989-2000 en ex-URSS) y ``wave_eeur`` (1989-1998 en Europa del Este) absorben
    el choque irrepetible de la transición postsoviética durante la estimación y se anulan en
    modo pronóstico (análogo a ``wave90`` en ``democracy.py``).
  - La incertidumbre combina un **choque común regional** ``u^{reg}_{r,t}`` y un **choque
    idiosincrático por país** ``u^{id}_{i,t}`` con memoria AR(1) ``rho_u``, estimados por
    descomposición de varianza (ANOVA de efectos aleatorios) sobre los residuos por región.

Se integra en ``model.Simulator(..., elites_model="auto")`` preservando paridad bit a bit
cuando ``legacy=True`` o ``elites_model=None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


R10_CODES: tuple[str, ...] = (
    "NAM", "EUR", "OPA", "REF", "CHN", "SAS", "SEA", "LAM", "MEA", "SSA",
)
R10_NAMES: dict[str, str] = {
    "NAM": "Norteamérica",
    "EUR": "Europa",
    "OPA": "OCDE Pacífico",
    "REF": "Ex-URSS",
    "CHN": "China y Asia Or.",
    "SAS": "Asia del Sur",
    "SEA": "Sudeste Asiático",
    "LAM": "América Latina",
    "MEA": "Oriente Medio y N. Áf.",
    "SSA": "África Subsahariana",
}
WEST_CODES: tuple[str, ...] = ("NAM", "EUR", "OPA")
ASIA_EMERG_CODES: tuple[str, ...] = ("CHN", "SAS", "SEA")
GLOBAL_SOUTH_CODES: tuple[str, ...] = ("LAM", "MEA", "SSA")

# Mapeo autocontenido ISO3 -> R10 (coherente con world4/core/l0r/regions.py, sin depender de UNlocations.txt)
R10_MAP: dict[str, str] = {
    # NAM
    "USA": "NAM", "CAN": "NAM", "BMU": "NAM",
    # OPA (OCDE Pacífico)
    "JPN": "OPA", "KOR": "OPA", "AUS": "OPA", "NZL": "OPA",
    # EUR (Europa sin ex-URSS)
    "ALB": "EUR", "AUT": "EUR", "BEL": "EUR", "BGR": "EUR", "BIH": "EUR", "CHE": "EUR",
    "CYP": "EUR", "CZE": "EUR", "DEU": "EUR", "DNK": "EUR", "ESP": "EUR", "EST": "EUR",
    "FIN": "EUR", "FRA": "EUR", "GBR": "EUR", "GRC": "EUR", "HRV": "EUR", "HUN": "EUR",
    "IRL": "EUR", "ISL": "EUR", "ITA": "EUR", "LTU": "EUR", "LUX": "EUR", "LVA": "EUR",
    "MKD": "EUR", "MLT": "EUR", "MNE": "EUR", "NLD": "EUR", "NOR": "EUR", "POL": "EUR",
    "PRT": "EUR", "ROU": "EUR", "SRB": "EUR", "SVK": "EUR", "SVN": "EUR", "SWE": "EUR",
    # REF (Antigua URSS)
    "RUS": "REF", "UKR": "REF", "BLR": "REF", "MDA": "REF", "ARM": "REF", "AZE": "REF",
    "GEO": "REF", "KAZ": "REF", "KGZ": "REF", "TJK": "REF", "TKM": "REF", "UZB": "REF",
    # CHN (China y resto de Asia Oriental)
    "CHN": "CHN", "HKG": "CHN", "MAC": "CHN", "TWN": "CHN", "MNG": "CHN", "PRK": "CHN",
    # SAS (Asia Meridional)
    "IND": "SAS", "PAK": "SAS", "BGD": "SAS", "LKA": "SAS", "NPL": "SAS", "BTN": "SAS",
    "MDV": "SAS", "AFG": "SAS",
    # SEA (Sudeste Asiático y Pacífico)
    "IDN": "SEA", "THA": "SEA", "VNM": "SEA", "PHL": "SEA", "MYS": "SEA", "SGP": "SEA",
    "MMR": "SEA", "KHM": "SEA", "LAO": "SEA", "BRN": "SEA", "TLS": "SEA", "FJI": "SEA",
    "PNG": "SEA", "SLB": "SEA", "VUT": "SEA", "WSM": "SEA", "TON": "SEA",
    # MEA (Oriente Medio y Norte de África)
    "ARE": "MEA", "BHR": "MEA", "DZA": "MEA", "EGY": "MEA", "IRN": "MEA", "IRQ": "MEA",
    "ISR": "MEA", "JOR": "MEA", "KWT": "MEA", "LBN": "MEA", "LBY": "MEA", "MAR": "MEA",
    "OMN": "MEA", "PSE": "MEA", "QAT": "MEA", "SAU": "MEA", "SDN": "MEA", "SYR": "MEA",
    "TUN": "MEA", "TUR": "MEA", "YEM": "MEA",
    # LAM (América Latina y Caribe)
    "ABW": "LAM", "AIA": "LAM", "ARG": "LAM", "ATG": "LAM", "BHS": "LAM", "BLZ": "LAM",
    "BOL": "LAM", "BRA": "LAM", "BRB": "LAM", "CHL": "LAM", "COL": "LAM", "CRI": "LAM",
    "CUB": "LAM", "CUW": "LAM", "CYM": "LAM", "DMA": "LAM", "DOM": "LAM", "ECU": "LAM",
    "GRD": "LAM", "GTM": "LAM", "GUY": "LAM", "HND": "LAM", "HTI": "LAM", "JAM": "LAM",
    "KNA": "LAM", "LCA": "LAM", "MEX": "LAM", "MSR": "LAM", "NIC": "LAM", "PAN": "LAM",
    "PER": "LAM", "PRI": "LAM", "PRY": "LAM", "SLV": "LAM", "SUR": "LAM", "SXM": "LAM",
    "TCA": "LAM", "TTO": "LAM", "URY": "LAM", "VCT": "LAM", "VEN": "LAM", "VGB": "LAM",
    # SSA (África Subsahariana)
    "AGO": "SSA", "BDI": "SSA", "BEN": "SSA", "BFA": "SSA", "BWA": "SSA", "CAF": "SSA",
    "CIV": "SSA", "CMR": "SSA", "COD": "SSA", "COG": "SSA", "COM": "SSA", "CPV": "SSA",
    "DJI": "SSA", "ERI": "SSA", "ETH": "SSA", "GAB": "SSA", "GHA": "SSA", "GIN": "SSA",
    "GMB": "SSA", "GNB": "SSA", "GNQ": "SSA", "KEN": "SSA", "LBR": "SSA", "LSO": "SSA",
    "MDG": "SSA", "MLI": "SSA", "MOZ": "SSA", "MRT": "SSA", "MUS": "SSA", "MWI": "SSA",
    "NAM": "SSA", "NER": "SSA", "NGA": "SSA", "RWA": "SSA", "SEN": "SSA", "SLE": "SSA",
    "SOM": "SSA", "SSD": "SSA", "STP": "SSA", "SWZ": "SSA", "SYC": "SSA", "TCD": "SSA",
    "TGO": "SSA", "TZA": "SSA", "UGA": "SSA", "ZAF": "SSA", "ZMB": "SSA", "ZWE": "SSA",
}

_EEUR_ISO3: frozenset[str] = frozenset({
    "POL", "HUN", "CZE", "SVK", "ROU", "BGR", "EST", "LVA", "LTU",
    "ALB", "HRV", "SVN", "SRB", "BIH", "MKD", "MNE",
})

COV_NAMES: tuple[str, ...] = (
    "poly_c", "core_c", "r_minus_g", "res_share", "open_c", "acc_w", "conf",
)


def region_index(iso3) -> np.ndarray:
    """Devuelve índice entero 0..9 en ``R10_CODES`` para cada código ISO3 (-1 si no mapeado)."""
    out = np.full(len(iso3), -1, dtype=int)
    for i, c in enumerate(iso3):
        code = R10_MAP.get(str(c).upper())
        if code is not None:
            out[i] = R10_CODES.index(code)
    return out


def region_codes(iso3) -> list[str]:
    return [R10_MAP.get(str(c).upper(), "OTH") for c in iso3]


@dataclass
class ElitesModel:
    """Modelo regionalizado R10 de captura de élites (cuota Top-10 % antes de impuestos)."""
    b0: float                   # intercepto global
    b_ec: float                 # reversión media global sobre (E - 0.45)
    u_reg_int: np.ndarray       # (K,) desviaciones regionales de intercepto
    dlam_reg: np.ndarray        # (K,) desviaciones regionales de reversión media
    b_cov: np.ndarray           # (7,) coeficientes de covariables endógenas
    b_wave_ref: float           # choque transitorio ex-URSS (1989-2000; 0 en pronóstico)
    b_wave_eeur: float          # choque transitorio Europa del Este (1989-1998; 0 en pronóstico)
    sd_reg: np.ndarray          # (K,) desviación estándar del choque común regional
    sd_id: np.ndarray           # (K,) desviación estándar del choque idiosincrático por país
    reg_idx: np.ndarray         # (N,) índice de región 0..K-1 (-1 -> media global)
    is_ref: np.ndarray          # (N,) bool ex-URSS
    is_eeur: np.ndarray         # (N,) bool Europa del Este postcomunista
    rho_u: float = 0.30         # memoria AR(1) de choques
    cutoff_year: int = 2019
    n_obs: int = 0
    resid_sd: float = 0.011

    @property
    def K(self) -> int:
        return len(R10_CODES)

    def summary(self) -> dict:
        regions = {}
        for k, code in enumerate(R10_CODES):
            lam = -(self.b_ec + float(self.dlam_reg[k]))
            e_star = 0.45 + (self.b0 + float(self.u_reg_int[k])) / max(lam, 1e-4)
            regions[code] = {
                "u_r_pp_yr": round(float(self.u_reg_int[k]) * 100.0, 3),
                "lam_pct_yr": round(lam * 100.0, 2),
                "E_star_pct": round(float(np.clip(e_star, 0.15, 0.85)) * 100.0, 1),
                "sd_reg_pp": round(float(self.sd_reg[k]) * 100.0, 2),
                "sd_id_pp": round(float(self.sd_id[k]) * 100.0, 2),
            }
        return {
            "cutoff_year": self.cutoff_year,
            "n_obs": self.n_obs,
            "rho_u": self.rho_u,
            "b0_pp_yr": round(self.b0 * 100.0, 4),
            "b_ec_pct_yr": round(self.b_ec * 100.0, 3),
            "b_cov_pp_yr": {k: round(float(v) * 100.0, 3) for k, v in zip(COV_NAMES, self.b_cov)},
            "wave_ref_pp_yr": round(self.b_wave_ref * 100.0, 3),
            "wave_eeur_pp_yr": round(self.b_wave_eeur * 100.0, 3),
            "resid_sd_pp": round(self.resid_sd * 100.0, 2),
            "regions": regions,
        }

    def step(
        self,
        E: np.ndarray,
        u_reg: np.ndarray,
        u_id: np.ndarray,
        *,
        poly: np.ndarray,
        core: np.ndarray,
        rstar: float,
        g_prev: np.ndarray,
        res_share: np.ndarray,
        open_: np.ndarray,
        acc_w: np.ndarray,
        conf: np.ndarray,
        act: np.ndarray,
        year: int,
        obs_mode: bool,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Avanza un año la captura de élites ``E`` (R, N) con deriva regional y choques."""
        R, N = E.shape
        K = self.K
        ridx = self.reg_idx
        mapped = ridx >= 0
        ridx_safe = np.where(mapped, ridx, 0)

        u_r_vec = np.where(mapped, self.u_reg_int[ridx_safe], 0.0)
        dlam_vec = np.where(mapped, self.dlam_reg[ridx_safe], 0.0)

        ec = np.nan_to_num(E, nan=0.45) - 0.45
        p_c = np.clip(np.nan_to_num(poly, nan=0.40), 0.0, 1.0) - 0.5
        c_c = np.clip(np.nan_to_num(core, nan=0.20), 0.0, 1.0) - 0.2
        g_c = np.clip(np.nan_to_num(g_prev, nan=0.02), -0.25, 0.25)
        r_m_g = float(np.clip(rstar, -0.05, 0.10)) - g_c
        rr_c = np.clip(np.nan_to_num(res_share, nan=0.02), 0.0, 0.8)
        op_c = np.clip(np.nan_to_num(open_, nan=0.50), 0.0, 2.5) - 0.5
        aw_c = np.clip(np.nan_to_num(acc_w, nan=0.02), 0.0, 0.4)
        cf_c = np.clip(np.nan_to_num(conf, nan=0.0), 0.0, 1.0)

        bc = self.b_cov
        drift = (
            self.b0
            + u_r_vec[None, :]
            + (self.b_ec + dlam_vec[None, :]) * ec
            + bc[0] * p_c
            + bc[1] * c_c
            + bc[2] * r_m_g
            + bc[3] * rr_c
            + bc[4] * op_c
            + bc[5] * aw_c
            + bc[6] * cf_c
        )
        # En replay histórico (obs_mode), aplicar choque transitorio postsoviético 1989-2000;
        # en pronóstico futuro (not obs_mode), las olas irrepetibles valen 0.
        if obs_mode:
            if 1989 <= year <= 2000:
                drift = drift + self.b_wave_ref * self.is_ref[None, :]
            if 1989 <= year <= 1998:
                drift = drift + self.b_wave_eeur * self.is_eeur[None, :]

        inn_scale = float(np.sqrt(max(1.0 - self.rho_u ** 2, 0.25)))
        sd_id_vec = np.where(mapped, self.sd_id[ridx_safe], self.resid_sd)
        eta_reg = rng.standard_normal((R, K)) * (self.sd_reg[None, :] * inn_scale)
        eps_id = rng.standard_normal((R, N)) * (sd_id_vec[None, :] * inn_scale)

        u_reg_next = self.rho_u * u_reg + eta_reg
        u_id_next = np.where(act, self.rho_u * u_id + eps_id, 0.0)
        reg_shock = np.where(mapped[None, :], u_reg_next[:, ridx_safe], 0.0)

        E_next = np.where(act, np.clip(E + drift + reg_shock + u_id_next, 0.15, 0.85), E)
        return E_next, u_reg_next, u_id_next


def fit(
    W,
    LY: np.ndarray | None = None,
    cutoff: int | None = None,
    *,
    min_year: int = 1980,
    ridge_reg: float = 30.0,
    ridge_cov: float = 25.0,
    rho_u: float = 0.30,
) -> ElitesModel:
    """Estima el modelo regionalizado de captura de élites usando exclusivamente datos <= cutoff."""
    N, T = W.N, W.T
    K = len(R10_CODES)
    cutoff = (T - 1) if cutoff is None else min(int(cutoff), T - 1)
    if LY is None:
        LY = np.log(np.maximum(W.Y, 1e-9) / np.maximum(W.pop, 1e-9))
        LY = np.where(W.observed, LY, np.nan)

    ridx = region_index(W.iso3)
    is_ref = (ridx == R10_CODES.index("REF"))
    is_eeur = np.array([str(c).upper() in _EEUR_ISO3 for c in W.iso3], dtype=bool)

    top10_raw = W.extra.get("top10_share_ptinc", np.full((N, T), np.nan))
    flat = np.nan_to_num(W.extra.get("wid_flat_flag", np.zeros((N, T))))
    E_obs = np.where(W.observed & (flat == 0), top10_raw, np.nan)

    poly = W.locf("vdem_polyarchy", 0.35)
    g1 = np.zeros((N, T))
    g1[:, 1:] = np.clip(np.nan_to_num(np.diff(LY, axis=1), nan=0.0), -0.25, 0.25)
    rstar = np.clip(
        np.nan_to_num(W.world.get("finanzas__us_real_strate", np.full(T, 0.01)), nan=0.01),
        -0.05, 0.10,
    )
    rr = np.clip(np.nan_to_num(W.rr, nan=0.0), 0.0, 0.8)
    opn = np.clip(np.nan_to_num(W.open_, nan=0.5), 0.0, 2.5)
    conf = np.clip(np.nan_to_num(W.conflict, nan=0.0), 0.0, 1.0)
    acc_w = np.clip(np.nan_to_num(W.locf("acc_w", 0.0), nan=0.0), 0.0, 0.4)

    core_arr = np.full((N, T), 0.2)
    for t in range(T):
        ok = np.isfinite(LY[:, t])
        if ok.sum() > 10:
            q80 = float(np.nanquantile(LY[:, t], 0.80))
            core_arr[:, t] = 1.0 / (
                1.0 + np.exp(-np.clip((np.nan_to_num(LY[:, t], nan=q80) - q80) / 0.35, -20.0, 20.0))
            )

    t_start = max(4, min_year - 1950)
    if cutoff - t_start < 8:
        t_start = 4

    rows_X, rows_y, rows_r, rows_t = [], [], [], []
    for t in range(t_start, cutoff + 1):
        yr = 1950 + t
        e_prev = E_obs[:, t - 1]
        e_cur = E_obs[:, t]
        ec = e_prev - 0.45

        R_int = np.zeros((N, K))
        R_slp = np.zeros((N, K))
        mapped = ridx >= 0
        if mapped.any():
            idx_m = np.where(mapped)[0]
            R_int[idx_m, ridx[idx_m]] = 1.0
            R_slp[idx_m, ridx[idx_m]] = ec[idx_m]

        w_ref = np.where(is_ref & (1989 <= yr <= 2000), 1.0, 0.0)
        w_eeur = np.where(is_eeur & (1989 <= yr <= 1998), 1.0, 0.0)

        F = np.column_stack([
            np.ones(N),
            ec,
            R_int,
            R_slp,
            poly[:, t - 1] - 0.5,
            core_arr[:, t - 1] - 0.2,
            rstar[t - 1] - g1[:, t - 1],
            rr[:, t - 1],
            opn[:, t - 1] - 0.5,
            acc_w[:, t - 1],
            conf[:, t - 1],
            w_ref,
            w_eeur,
        ])
        ok = np.isfinite(e_prev) & np.isfinite(e_cur) & np.all(np.isfinite(F), axis=1)
        if ok.any():
            rows_X.append(F[ok])
            rows_y.append(e_cur[ok] - e_prev[ok])
            rows_r.append(ridx[ok])
            rows_t.append(np.full(int(ok.sum()), t, dtype=int))

    if not rows_X:
        return ElitesModel(
            b0=0.0, b_ec=-0.025,
            u_reg_int=np.zeros(K), dlam_reg=np.zeros(K),
            b_cov=np.zeros(len(COV_NAMES)),
            b_wave_ref=0.0, b_wave_eeur=0.0,
            sd_reg=np.full(K, 0.0032), sd_id=np.full(K, 0.010),
            reg_idx=ridx, is_ref=is_ref, is_eeur=is_eeur,
            rho_u=rho_u, cutoff_year=1950 + cutoff, n_obs=0,
        )

    X = np.vstack(rows_X)
    y = np.concatenate(rows_y)
    r_all = np.concatenate(rows_r)
    t_all = np.concatenate(rows_t)

    P = X.shape[1]
    pen = np.zeros(P)
    pen[0] = 0.0
    pen[1] = 1.0
    pen[2:2 + K] = ridge_reg
    pen[2 + K:2 + 2 * K] = ridge_reg * 2.5
    pen[2 + 2 * K:-2] = ridge_cov
    pen[-2:] = 1.0

    beta = np.linalg.solve(X.T @ X + np.diag(pen), X.T @ y)
    # Garantizar reversión media estable en todas las regiones (evita explosión en regiones pequeñas)
    beta[1] = min(float(beta[1]), -0.015)
    for k in range(K):
        if beta[1] + beta[2 + K + k] > -0.008:
            beta[2 + K + k] = -0.008 - beta[1]

    resid = y - X @ beta
    non_wave = (X[:, -2] < 0.5) & (X[:, -1] < 0.5)
    res_nw = resid[non_wave] if non_wave.sum() > 20 else resid
    res_nw = res_nw - float(np.mean(res_nw))
    glob_sd = float(np.clip(np.std(res_nw), 0.004, 0.025))

    # Descomposición de varianza (ANOVA de efectos aleatorios) por región: choque regional vs país
    sd_reg = np.zeros(K)
    sd_id = np.zeros(K)
    for k in range(K):
        mk = non_wave & (r_all == k)
        if mk.sum() >= 25:
            ek = resid[mk]
            tk = t_all[mk]
            uniq_t, inv = np.unique(tk, return_inverse=True)
            counts = np.bincount(inv).astype(float)
            means = np.bincount(inv, weights=ek) / counts
            within = ek - means[inv]
            var_id = float(np.sum(within ** 2) / max(len(within) - len(uniq_t), 1))
            if len(uniq_t) > 3:
                v_bar = float(np.var(means, ddof=1))
                var_reg = max(v_bar - var_id * float(np.mean(1.0 / counts)), 0.0)
            else:
                var_reg = 0.0
            sd_reg[k] = float(np.sqrt(0.70 * var_reg + 0.30 * (0.0032 ** 2)))
            sd_id[k] = float(np.sqrt(0.75 * var_id + 0.25 * (glob_sd ** 2)))
        else:
            sd_reg[k] = 0.0032
            sd_id[k] = glob_sd

    return ElitesModel(
        b0=float(beta[0]),
        b_ec=float(beta[1]),
        u_reg_int=beta[2:2 + K].copy(),
        dlam_reg=beta[2 + K:2 + 2 * K].copy(),
        b_cov=beta[2 + 2 * K:-2].copy(),
        b_wave_ref=float(beta[-2]),
        b_wave_eeur=float(beta[-1]),
        sd_reg=sd_reg,
        sd_id=sd_id,
        reg_idx=ridx,
        is_ref=is_ref,
        is_eeur=is_eeur,
        rho_u=rho_u,
        cutoff_year=1950 + cutoff,
        n_obs=int(len(y)),
        resid_sd=glob_sd,
    )
