"""Discrepancia AR1 por variable (Kennedy y O'Hagan 2001), con ``phycore.uq.DiscrepancyGP``:

    realidad(h) = ρ · modelo(h) + δ(h) + ε,   ρ ~ N(1, s_marca²),   δ ~ GP(0, Matérn 5/2)

donde ``modelo(h)`` y ``realidad(h)`` son el CRECIMIENTO desde el inicio de la corrida (log del cociente,
o diferencia en °C para la GMST) y h los años desde ese inicio. ρ corrige un error de escala del
crecimiento (el modelo crece de más o de menos) y δ lo que queda. La anchura de la previa de ρ sale de la
marca de la variable (``phycore.contracts.marks.PRIOR_WIDTH``: [V] 0,05, [S] 0,15, [M] 0,30) y una
variable [?] no tiene discrepancia: queda BLOQUEADA.

Sólo se ajusta con pares dentro de la ventana de calibración (datos <= corte). Fuera de ella, la
predicción del GP vuelve a su previa (varianza σ_f²) y la incertidumbre de ρ crece con el crecimiento
que el modelo proyecta: la banda se ensancha con la distancia a los datos.
"""
from __future__ import annotations

import numpy as np
from phycore.contracts.marks import PRIOR_WIDTH, Mark
from phycore.uq import DiscrepancyGP


def prior_sd(mark: str) -> float | None:
    return PRIOR_WIDTH[Mark(mark)]


def fit(h, model_growth, obs_growth, mark: str = "[S]", seed: int = 0) -> DiscrepancyGP:
    h = np.asarray(h, float).reshape(-1, 1)
    f, y = np.asarray(model_growth, float).ravel(), np.asarray(obs_growth, float).ravel()
    ok = np.isfinite(f) & np.isfinite(y) & np.isfinite(h[:, 0])
    return DiscrepancyGP(prior_rho_sd=prior_sd(mark), seed=seed).fit(h[ok], f[ok], y[ok])


def sample(gp: DiscrepancyGP, h, model_growth, rng, draws: int = 4) -> np.ndarray:
    """Muestras de la realidad (draws·R, H) para cada miembro (R, H) del ensemble: marginales por año
    (es lo que necesita una banda p10-p90 por año)."""
    h = np.asarray(h, float)
    M = np.atleast_2d(np.asarray(model_growth, float))
    R, H = M.shape
    out = np.empty((draws, R, H))
    for r in range(R):
        mu, sd = gp.predict(h.reshape(-1, 1), M[r], with_noise=True)
        out[:, r] = mu[None] + sd[None] * rng.standard_normal((draws, H))
    return out.reshape(draws * R, H)


def report(gp: DiscrepancyGP, band: float) -> dict:
    v = gp.verdict(declared_band=band).as_dict()
    return {k: v[k] for k in ("n_pairs", "rho", "rho_sd", "rho_sigmas_from_prior", "delta_rms", "loo_rmse",
                              "loo_rmse_naive", "loo_coverage", "sigma_f", "sigma_n", "verdict", "reason")}
