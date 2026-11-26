"""Modelo empírico V2 de dinámica democrática para Phy-World L1 (Atlas Mundo).

Sustituye la especificación binaria heredada (ell, g, W_D global con vdem_polyarchy
congelada) por un sistema acoplado continuo + discreto validado fuera de muestra:

  1. Calidad democrática continua (vdem_polyarchy): deriva anual en función de
     nivel actual, anocracia 4p(1-p), inercia/momento de 3 años (captura olas de
     autocratización ejecutiva gradual), difusión vecinal endógena (matriz de
     contigüidad geográfica COW con respaldo comercial para islas), renta per cápita,
     crecimiento, maldición de recursos naturales, edad del régimen, trampa de golpes,
     guerra civil y asabiya — más choques empíricos estratificados con memoria AR(1).
  2. Transiciones de régimen (A->D y D->A): logits acoplados a la poliarquía continua,
     momento de 3 años, democracia vecinal endógena, recursos, edad del régimen,
     trampa de golpes, asabiya, renta y crecimiento — SIN la tendencia global W_D y
     absorbiendo la ola única de 1989-1995 (colapso soviético) durante la estimación.

Se estima estrictamente con datos <= cutoff (sin fuga temporal) y se integra en
``model.Simulator(..., democracy_model=dm)`` preservando paridad bit a bit cuando
``democracy_model=None``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import submodels


def _sig(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def build_neighbor_matrix(W) -> np.ndarray:
    """Matriz fila-estocástica (N, N) de vecinos geográficos (COW contdir) con respaldo
    en socios comerciales simetrizados (S_exp + M_imp) para países insulares."""
    N = W.N
    S_fb = None
    if W.S_exp is not None and W.M_imp is not None:
        S_sym = W.S_exp[-1].astype(float) + W.M_imp[-1].astype(float)
        S_fb = S_sym / np.maximum(S_sym.sum(1, keepdims=True), 1e-9)
    try:
        import rdata
        from .sources import geopolitica as gp

        f = gp.RAW / "cow_contdir.rda"
        if f.exists():
            parsed = rdata.parser.parse_file(f)
            df = list(rdata.conversion.convert(parsed).values())[0]
            df = df[df["year"] >= 2000].copy()
            c1 = gp.code_to_iso3(df["state1no"], "cow")
            c2 = gp.code_to_iso3(df["state2no"], "cow")
            idx = {c: i for i, c in enumerate(W.iso3)}
            A = np.zeros((N, N), float)
            for a, b in zip(c1, c2):
                if a in idx and b in idx and a != b:
                    A[idx[a], idx[b]] = 1.0
                    A[idx[b], idx[a]] = 1.0
            rs = A.sum(1, keepdims=True)
            if S_fb is not None:
                return np.where(rs > 0, A / np.maximum(rs, 1.0), S_fb)
            return np.where(rs > 0, A / np.maximum(rs, 1.0), 1.0 / max(N, 1))
    except Exception:
        pass
    if S_fb is not None:
        return S_fb
    M = np.full((N, N), 1.0 / max(N - 1, 1))
    np.fill_diagonal(M, 0.0)
    return M


def compute_regime_age(demo_arr: np.ndarray, t_end: int) -> np.ndarray:
    """Años consecutivos en el régimen binario actual hasta el índice t_end (inclusive)."""
    N = demo_arr.shape[0]
    age = np.ones((N, t_end + 1), float)
    for i in range(N):
        a = 1.0
        prev = np.nan
        for t in range(t_end + 1):
            v = demo_arr[i, t]
            if np.isfinite(v):
                if np.isfinite(prev) and v == prev:
                    a += 1.0
                else:
                    a = 1.0
                prev = v
            age[i, t] = a
    return age


POLY_NAMES = (
    "const", "poly_c", "anoc", "dp3", "nbr_gap", "ell", "g1",
    "log_res", "lage", "coup10", "conf", "asab_c", "wave90",
)
TR_NAMES = (
    "const", "poly_c", "dp3", "ell", "g1", "nbr_d",
    "log_res", "lage", "coup10", "asab_c", "wave90",
)


@dataclass
class DemocracyModel:
    """Predictor acoplado de poliarquía continua y transiciones de régimen."""
    b_poly: np.ndarray          # (13,) coeficientes deriva anual de poly (último = wave90)
    b_ad: np.ndarray            # (11,) logit A -> D (último = wave90)
    b_da: np.ndarray            # (11,) logit D -> A (último = wave90)
    res_low: np.ndarray         # residuos centrados para poly < 0.30
    res_mid: np.ndarray         # residuos centrados para 0.30 <= poly <= 0.65
    res_high: np.ndarray        # residuos centrados para poly > 0.65
    nbr_w: np.ndarray           # (N, N) matriz de vecinos
    rho_u: float = 0.25         # persistencia AR(1) de choques de poliarquía
    cutoff_year: int = 2019

    def summary(self) -> dict:
        return {
            "cutoff_year": self.cutoff_year,
            "rho_u": self.rho_u,
            "b_poly": {k: round(float(v), 4) for k, v in zip(POLY_NAMES, self.b_poly)},
            "b_ad": {k: round(float(v), 3) for k, v in zip(TR_NAMES, self.b_ad)},
            "b_da": {k: round(float(v), 3) for k, v in zip(TR_NAMES, self.b_da)},
        }

    def step(
        self,
        poly: np.ndarray,
        demo: np.ndarray,
        p_buf: np.ndarray,
        reg_age: np.ndarray,
        u_mem: np.ndarray,
        *,
        ell: np.ndarray,
        g_prev: np.ndarray,
        res_share: np.ndarray,
        coup_trap: np.ndarray,
        conf: np.ndarray,
        asab: np.ndarray,
        succ_coup: np.ndarray,
        act: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Avanza un año el estado político acoplado (R, N) en modo pronóstico (wave90 = 0)."""
        R, N = poly.shape
        dp3 = np.clip(poly - p_buf[:, :, 0], -0.4, 0.4)
        nbr_p = poly @ self.nbr_w.T
        nbr_d = demo @ self.nbr_w.T
        anoc = 4.0 * poly * (1.0 - poly)
        lage = np.log1p(np.maximum(reg_age, 1.0))
        lres = np.log1p(10.0 * np.clip(res_share, 0.0, 0.8))
        g_c = np.clip(g_prev, -0.3, 0.3)
        c10 = (coup_trap > 0).astype(float)

        bp = self.b_poly
        drift = (
            bp[0]
            + bp[1] * (poly - 0.5)
            + bp[2] * anoc
            + bp[3] * dp3
            + bp[4] * (nbr_p - poly)
            + bp[5] * ell
            + bp[6] * g_c
            + bp[7] * lres
            + bp[8] * lage
            + bp[9] * c10
            + bp[10] * conf
            + bp[11] * (asab - 0.5)
        )

        e_lo = self.res_low[rng.integers(0, len(self.res_low), (R, N))]
        e_md = self.res_mid[rng.integers(0, len(self.res_mid), (R, N))]
        e_hi = self.res_high[rng.integers(0, len(self.res_high), (R, N))]
        eps = np.where(poly < 0.30, e_lo, np.where(poly <= 0.65, e_md, e_hi))
        u_next = np.where(act, self.rho_u * u_mem + eps, 0.0)
        p_next = np.where(act, np.clip(poly + drift + u_next, 0.02, 0.94), poly)

        # Golpe de Estado exitoso daña directamente la poliarquía
        p_next = np.where(succ_coup & act, np.minimum(p_next, np.maximum(p_next - 0.18, 0.12)), p_next)

        bad, bda = self.b_ad, self.b_da
        z_ad = (
            bad[0]
            + bad[1] * (p_next - 0.5)
            + bad[2] * dp3
            + bad[3] * ell
            + bad[4] * g_c
            + bad[5] * nbr_d
            + bad[6] * lres
            + bad[7] * lage
            + bad[8] * c10
            + bad[9] * (asab - 0.5)
        )
        z_da = (
            bda[0]
            + bda[1] * (p_next - 0.5)
            + bda[2] * dp3
            + bda[3] * ell
            + bda[4] * g_c
            + bda[5] * nbr_d
            + bda[6] * lres
            + bda[7] * lage
            + bda[8] * c10
            + bda[9] * (asab - 0.5)
        )

        u = rng.random((R, N))
        to_d = (demo == 0) & (u < _sig(z_ad)) & (~succ_coup) & act
        to_a = (((demo == 1) & (u < _sig(z_da))) | (succ_coup & (demo == 1))) & act
        d_next = np.where(to_d, 1.0, np.where(to_a, 0.0, demo))
        # Coherencia suave cuando cambia el régimen binario
        p_next = np.where(to_d, np.maximum(p_next, 0.48), np.where(to_a, np.minimum(p_next, 0.42), p_next))
        age_next = np.where(act, np.where(d_next == demo, reg_age + 1.0, 1.0), reg_age)

        buf_next = np.empty_like(p_buf)
        buf_next[:, :, :-1] = p_buf[:, :, 1:]
        buf_next[:, :, -1] = p_next
        return p_next, d_next, buf_next, age_next, u_next


def fit(
    W,
    LY: np.ndarray | None = None,
    cutoff: int | None = None,
    *,
    ridge_poly: float = 10.0,
    ridge_logit: float = 0.5,
    rho_u: float = 0.25,
) -> DemocracyModel:
    """Estima el modelo V2 usando exclusivamente datos con índice temporal <= cutoff."""
    N, T = W.N, W.T
    cutoff = (T - 1) if cutoff is None else min(int(cutoff), T - 1)
    if LY is None:
        LY = np.log(W.Y / W.pop)
        LY[~W.observed] = np.nan

    poly_raw = W.extra.get("vdem_polyarchy", np.full((N, T), np.nan))
    row_raw = W.extra.get("regime_row", np.full((N, T), np.nan))
    demo_raw = np.where(np.isfinite(row_raw), (row_raw >= 2).astype(float), np.nan)
    poly_locf = W.locf("vdem_polyarchy", np.nan)
    row_locf = W.locf("regime_row", np.nan)
    demo_locf = np.where(np.isfinite(row_locf), (row_locf >= 2).astype(float), np.nan)

    g1 = np.full((N, T), np.nan)
    g1[:, 1:] = np.diff(LY, axis=1)
    ell = LY - np.log(10000.0)
    log_res = np.log1p(10.0 * np.clip(W.rr, 0.0, 0.8))
    coups10 = (W.locf("coups_past10", 0.0) > 0).astype(float)
    asab_obs = W.locf("asabiya_proxy", 0.5)
    conf_obs = W.conflict
    wave90 = np.array([1.0 if 1989 <= 1950 + t <= 1995 else 0.0 for t in range(T)])

    nbr_w = build_neighbor_matrix(W)
    age_arr = compute_regime_age(demo_locf, cutoff)

    Xp_list, yp_list = [], []
    tr_X, tr_y = {0: [], 1: []}, {0: [], 1: []}
    for t in range(4, cutoff + 1):
        p_prev = poly_raw[:, t - 1]
        p_cur = poly_raw[:, t]
        dp3 = p_prev - poly_locf[:, t - 4]
        p_fill = np.where(np.isfinite(poly_locf[:, t - 1]), poly_locf[:, t - 1], np.nanmean(poly_locf[:, t - 1]))
        nbr_p = nbr_w @ p_fill
        d_fill = np.where(np.isfinite(demo_locf[:, t - 1]), demo_locf[:, t - 1], np.nanmean(demo_locf[:, t - 1]))
        nbr_d = nbr_w @ d_fill
        anoc = 4.0 * p_prev * (1.0 - p_prev)
        lage = np.log1p(age_arr[:, t - 1])
        w90 = np.full(N, wave90[t])

        F_p = np.column_stack([
            np.ones(N), p_prev - 0.5, anoc, np.clip(dp3, -0.4, 0.4), nbr_p - p_prev,
            ell[:, t - 1], np.clip(g1[:, t - 1], -0.3, 0.3), log_res[:, t - 1],
            lage, coups10[:, t - 1], conf_obs[:, t - 1], asab_obs[:, t - 1] - 0.5, w90,
        ])
        ok = np.all(np.isfinite(F_p), 1) & np.isfinite(p_cur)
        if ok.any():
            Xp_list.append(F_p[ok])
            yp_list.append(p_cur[ok] - p_prev[ok])

        F_tr = np.column_stack([
            np.ones(N), p_prev - 0.5, np.clip(dp3, -0.4, 0.4), ell[:, t - 1],
            np.clip(g1[:, t - 1], -0.3, 0.3), nbr_d, log_res[:, t - 1],
            lage, coups10[:, t - 1], asab_obs[:, t - 1] - 0.5, w90,
        ])
        for frm, to in ((0, 1), (1, 0)):
            okt = np.all(np.isfinite(F_tr), 1) & (demo_raw[:, t - 1] == frm) & np.isfinite(demo_raw[:, t])
            if okt.any():
                tr_X[frm].append(F_tr[okt])
                tr_y[frm].append((demo_raw[okt, t] == to).astype(float))

    Xp, yp = np.vstack(Xp_list), np.concatenate(yp_list)
    reg = np.eye(Xp.shape[1]) * ridge_poly
    reg[0, 0] = 0.0
    b_poly = np.linalg.solve(Xp.T @ Xp + reg, Xp.T @ yp)
    resid = yp - Xp @ b_poly
    p_tr = Xp[:, 1] + 0.5

    def _centered(mask):
        r = resid[mask]
        return (r - np.mean(r)) if len(r) > 5 else (resid - np.mean(resid))

    res_low = _centered(p_tr < 0.30)
    res_mid = _centered((p_tr >= 0.30) & (p_tr <= 0.65))
    res_high = _centered(p_tr > 0.65)

    b_ad, _, _, _ = submodels.logit_fit(np.vstack(tr_X[0]), np.concatenate(tr_y[0]), ridge=ridge_logit)
    b_da, _, _, _ = submodels.logit_fit(np.vstack(tr_X[1]), np.concatenate(tr_y[1]), ridge=ridge_logit)

    return DemocracyModel(
        b_poly=b_poly,
        b_ad=b_ad,
        b_da=b_da,
        res_low=res_low,
        res_mid=res_mid,
        res_high=res_high,
        nbr_w=nbr_w,
        rho_u=rho_u,
        cutoff_year=1950 + cutoff,
    )
