"""Emulador GP en NumPy para el *history matching*.

Un GP independiente por salida (Vernon et al. 2010 emulan cada salida por separado):

    f(x) = h(x)ᵀβ + u(x),   h(x) = (1, x),   u ~ GP(0, σ_f² k_Matérn5/2,ARD) + nugget σ_n²

* β se integra con una previa plana (krigeado universal): la varianza predictiva incluye la
  incertidumbre de la media.
* Los hiperparámetros (longitudes, σ_f, σ_n) se ajustan por máxima verosimilitud restringida (REML)
  con L-BFGS-B desde varios arranques.
* ``noise_var`` por punto (la varianza estocástica de la mediana de un ensemble de R miembros) entra
  en la diagonal y no se confunde con la señal.
* Las entradas se escalan al cubo unidad de la caja de la previa; la salida se estandariza.

``loo()`` da los errores estandarizados de dejar uno fuera: el diagnóstico de Bastos y O'Hagan (2009)
que decide si un emulador sirve (|z| > 3 en menos del 5 % de los puntos).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

LS_BOUNDS = (0.03, 10.0)       # en unidades del cubo unidad
SF_BOUNDS = (1e-3, 10.0)       # en unidades de la salida estandarizada
SN_BOUNDS = (1e-4, 1.0)
JITTER = 1e-8


def _matern52(D2):
    r = np.sqrt(np.maximum(D2, 0.0))
    s5 = math.sqrt(5.0)
    return (1.0 + s5 * r + 5.0 / 3.0 * r * r) * np.exp(-s5 * r)


def _sqdist(A, B, ls):
    A, B = A / ls, B / ls
    return np.maximum(np.sum(A * A, 1)[:, None] + np.sum(B * B, 1)[None, :] - 2.0 * A @ B.T, 0.0)


class GP:
    """GP de una salida. ``fit(X, y, noise_var)`` con X en [0, 1]^d."""

    def __init__(self, restarts: int = 3, seed: int = 0):
        self.restarts, self.seed = restarts, seed

    # ---------------------------------------------------------------- verosimilitud (REML)
    def _state(self, p, X, y, nv, D2s=None):
        """K, su Cholesky, la matriz de proyección P = K⁻¹ - K⁻¹H A⁻¹ HᵀK⁻¹ y α = P y."""
        d = X.shape[1]
        ls, sf2, sn2 = np.exp(p[:d]), math.exp(2 * p[d]), math.exp(2 * p[d + 1])
        if D2s is None:
            D2s = (X[:, None, :] - X[None, :, :]) ** 2                     # (n, n, d), sin escalar
        D2 = D2s @ (1.0 / ls ** 2)
        Km = _matern52(D2)
        K = sf2 * Km + np.diag(sn2 + nv + JITTER)
        c = cho_factor(K, lower=True)
        H = np.column_stack([np.ones(len(X)), X])
        Ki_H = cho_solve(c, H)
        A = H.T @ Ki_H
        Ai = np.linalg.inv(A)
        Ki = cho_solve(c, np.eye(len(X)))
        P = Ki - Ki_H @ Ai @ Ki_H.T
        return dict(ls=ls, sf2=sf2, sn2=sn2, D2=D2, Km=Km, c=c, H=H, Ki_H=Ki_H, A=A, Ai=Ai, P=P, alpha=P @ y,
                    beta=Ai @ (Ki_H.T @ y))

    def _nll_grad(self, p, X, y, nv, D2s):
        try:
            st = self._state(p, X, y, nv, D2s)
        except (np.linalg.LinAlgError, ValueError):
            return 1e10, np.zeros_like(p)
        sign, logdetA = np.linalg.slogdet(st["A"])
        if sign <= 0:
            return 1e10, np.zeros_like(p)
        nll = 0.5 * float(y @ st["alpha"]) + float(np.sum(np.log(np.diag(st["c"][0])))) + 0.5 * logdetA
        d = X.shape[1]
        Wm = np.outer(st["alpha"], st["alpha"]) - st["P"]               # dnll/dK = -1/2 W
        r = np.sqrt(np.maximum(st["D2"], 1e-300))
        s5 = math.sqrt(5.0)
        dk_dr2 = -5.0 / 6.0 * (1.0 + s5 * r) * np.exp(-s5 * r)
        G = st["sf2"] * dk_dr2 * Wm
        g = np.empty_like(p)
        # d(r²)/d(log ls_i) = -2 (x_i - x_i')² / ls_i²
        g[:d] = 0.5 * 2.0 * np.einsum("ij,ijk->k", G, D2s) / st["ls"] ** 2
        g[d] = -0.5 * float(np.sum(Wm * (2.0 * st["sf2"] * st["Km"])))
        g[d + 1] = -0.5 * float(2.0 * st["sn2"] * np.trace(Wm))
        return nll, g

    def fit(self, X, y, noise_var=None) -> "GP":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        n, d = X.shape
        if n < d + 3:
            raise ValueError(f"{n} runs for {d} inputs: an emulator needs at least d + 3")
        self.y_mu, self.y_sd = float(np.mean(y)), float(np.std(y)) + 1e-12
        ys = (y - self.y_mu) / self.y_sd
        nv = np.zeros(n) if noise_var is None else np.asarray(noise_var, float).ravel() / self.y_sd ** 2
        D2s = (X[:, None, :] - X[None, :, :]) ** 2
        rng = np.random.default_rng(self.seed)
        lo = np.array([math.log(LS_BOUNDS[0])] * d + [math.log(SF_BOUNDS[0]), math.log(SN_BOUNDS[0])])
        hi = np.array([math.log(LS_BOUNDS[1])] * d + [math.log(SF_BOUNDS[1]), math.log(SN_BOUNDS[1])])
        starts = [np.array([math.log(0.5)] * d + [math.log(0.5), math.log(0.05)])]
        for _ in range(self.restarts - 1):
            starts.append(np.concatenate([rng.uniform(math.log(0.15), math.log(2.0), d),
                                          [rng.uniform(math.log(0.1), math.log(1.0))],
                                          [rng.uniform(math.log(1e-3), math.log(0.1))]]))
        best = None
        for s0 in starts:
            r = minimize(self._nll_grad, s0, args=(X, ys, nv, D2s), jac=True, method="L-BFGS-B",
                         bounds=list(zip(lo, hi)), options={"maxiter": 200})
            if best is None or r.fun < best.fun:
                best = r
        self.p = best.x
        self.X, self.ys, self.nv = X, ys, nv
        st = self._state(self.p, X, ys, nv, D2s)
        self.ls, self.sf2, self.sn2 = st["ls"], st["sf2"], st["sn2"]
        self._c, self.H, self.Ki_H, self.Ai, self.beta, self._P = st["c"], st["H"], st["Ki_H"], st["Ai"], st["beta"], st["P"]
        self.Ki_r = cho_solve(self._c, ys - self.H @ self.beta)
        return self

    # ---------------------------------------------------------------- predicción
    def predict(self, Xs):
        """Media y varianza (sin el nugget: la de la función, no la de una corrida nueva)."""
        Xs = np.atleast_2d(np.asarray(Xs, float))
        ks = self.sf2 * _matern52(_sqdist(Xs, self.X, self.ls))
        Hs = np.column_stack([np.ones(len(Xs)), Xs])
        mu = Hs @ self.beta + ks @ self.Ki_r
        v = solve_triangular(self._c[0], ks.T, lower=True)
        R = Hs - ks @ self.Ki_H
        var = self.sf2 - np.sum(v * v, 0) + np.einsum("ij,jk,ik->i", R, self.Ai, R)
        return mu * self.y_sd + self.y_mu, np.maximum(var, 1e-12) * self.y_sd ** 2

    def loo(self):
        """Errores estandarizados de dejar uno fuera en forma cerrada (krigeado universal, Dubrule 1983):
        residuo (Py)_i / P_ii, varianza 1 / P_ii; hiperparámetros fijos."""
        a = self._P @ self.ys
        dg = np.diag(self._P)
        return (a / dg) / np.sqrt(1.0 / dg)


class Emulator:
    """Un GP por salida sobre la caja ``box`` (d, 2): las entradas se escalan al cubo unidad."""

    def __init__(self, box, restarts: int = 3, seed: int = 0):
        self.box = np.asarray(box, float)
        self.restarts, self.seed = restarts, seed

    def unit(self, theta):
        lo, hi = self.box[:, 0], self.box[:, 1]
        return (np.atleast_2d(theta) - lo) / (hi - lo)

    def fit(self, theta, Y, noise_var=None) -> "Emulator":
        U = self.unit(theta)
        Y = np.asarray(Y, float)
        nv = None if noise_var is None else np.asarray(noise_var, float)
        self.gps = [GP(self.restarts, self.seed + j).fit(U, Y[:, j], None if nv is None else nv[:, j])
                    for j in range(Y.shape[1])]
        return self

    def predict(self, theta):
        U = self.unit(theta)
        out = [g.predict(U) for g in self.gps]
        return np.column_stack([m for m, _ in out]), np.column_stack([v for _, v in out])

    def loo_z(self) -> np.ndarray:
        return np.column_stack([g.loo() for g in self.gps])

    def diagnostics(self) -> dict:
        z = self.loo_z()
        return {"loo_abs_z_gt3_share": float(np.mean(np.abs(z) > 3)), "loo_abs_z_gt2_share": float(np.mean(np.abs(z) > 2)),
                "loo_rms_z": float(np.sqrt(np.mean(z ** 2))), "outputs": len(self.gps),
                "lengthscales_min": [float(np.min(g.ls)) for g in self.gps]}
