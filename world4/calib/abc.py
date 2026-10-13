"""ABC-SMC con núcleo gaussiano y el error del modelo como núcleo final (Wilkinson 2013).

El objetivo de la generación t es

    π_t(θ) ∝ π(θ) · 1[θ ∈ NROY] · K_εt(y(θ) - z),   K_ε(e) = exp(-½ Σ_j e_j² / (ε² σ_j²))

con σ_j² = Var_obs,j + Var_disc,j. ε baja de forma adaptativa (cada paso conserva la mitad del tamaño
muestral efectivo, como Del Moral, Doucet y Jasra 2012) hasta ε = 1: en ese punto el núcleo ES el modelo
de error declarado y el resultado es el posterior exacto bajo ese error (Wilkinson 2013, teorema 1), no
una aproximación con tolerancia. Entre generaciones: remuestreo sistemático y pasos de Metropolis con
paseo aleatorio (covarianza 2.38²/d de la de las partículas).

Con un simulador estocástico (``simulate`` devuelve una realización) los pasos de Metropolis son
pseudo-marginales: la distancia de la partícula se guarda y se reutiliza.

``sbi`` (Apache-2.0) era la alternativa de la hoja de ruta; necesita torch, que el producto no lleva.
"""
from __future__ import annotations

import math

import numpy as np

ESS_FRAC = 0.5
MCMC_STEPS = 3
MAX_GEN = 60


def _systematic(w, rng):
    n = len(w)
    u = (rng.random() + np.arange(n)) / n
    c = np.cumsum(w)
    c[-1] = 1.0
    return np.searchsorted(c, u)


def _ess(logw):
    w = np.exp(logw - logw.max())
    return float(w.sum() ** 2 / np.sum(w * w))


def abc_smc(sample_prior, log_prior, simulate, z, sigma, n: int = 1000, seed: int = 0, ess_frac: float = ESS_FRAC,
            mcmc_steps: int = MCMC_STEPS, log=print) -> dict:
    """``sample_prior(n, rng) -> (n, d)``; ``log_prior(theta) -> (n,)`` (-inf fuera de la NROY o de la caja);
    ``simulate(theta) -> (n, q)``. Devuelve partículas equiponderadas en ε = 1."""
    rng = np.random.default_rng(seed)
    z, sigma = np.asarray(z, float), np.asarray(sigma, float)
    theta = sample_prior(n, rng)
    D2 = np.sum(((simulate(theta) - z) / sigma) ** 2, 1)
    D2 = np.where(np.isfinite(D2), D2, np.inf)
    eps = float(np.sqrt(np.max(D2[np.isfinite(D2)])) * 2)
    hist, n_sims, acc = [], n, []
    for g in range(MAX_GEN):
        # siguiente ε: bisección para que el ESS de los pesos incrementales sea ess_frac * n
        def logw(e):
            return -0.5 * D2 * (1 / e ** 2 - 1 / eps ** 2)
        lo, hi = 1.0, eps
        if _ess(logw(1.0)) >= ess_frac * n:
            new = 1.0
        else:
            for _ in range(60):
                mid = math.sqrt(lo * hi)
                if _ess(logw(mid)) >= ess_frac * n:
                    hi = mid
                else:
                    lo = mid
            new = hi
        lw = logw(new)
        w = np.exp(lw - lw.max())
        w /= w.sum()
        eps = new
        idx = _systematic(w, rng)
        theta, D2 = theta[idx], D2[idx]
        cov = np.atleast_2d(np.cov(theta.T)) * (2.38 ** 2 / theta.shape[1]) + 1e-12 * np.eye(theta.shape[1])
        lp = log_prior(theta)
        a_tot = 0
        for _ in range(mcmc_steps):
            prop = theta + rng.multivariate_normal(np.zeros(theta.shape[1]), cov, n)
            lpp = log_prior(prop)
            ok = np.isfinite(lpp)
            D2p = np.full(n, np.inf)
            if ok.any():
                D2p[ok] = np.sum(((simulate(prop[ok]) - z) / sigma) ** 2, 1)
                n_sims += int(ok.sum())
            D2p = np.where(np.isfinite(D2p), D2p, np.inf)
            la = (lpp - lp) - 0.5 * (D2p - D2) / eps ** 2
            accept = ok & (np.log(rng.random(n)) < np.where(np.isfinite(la), la, -np.inf))
            theta[accept], D2[accept], lp[accept] = prop[accept], D2p[accept], lpp[accept]
            a_tot += int(accept.sum())
        acc.append(a_tot / (n * mcmc_steps))
        hist.append({"eps": eps, "accept": acc[-1]})
        log(f"  ABC-SMC gen {g + 1}: eps {eps:.3f}, MH acceptance {acc[-1]:.0%}")
        if eps <= 1.0 + 1e-9:
            break
    return {"theta": theta, "D2": D2, "generations": hist, "simulations": n_sims, "eps_final": eps,
            "converged": bool(eps <= 1.0 + 1e-9)}
