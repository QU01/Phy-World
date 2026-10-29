"""Los dos modelos del módulo, en NumPy/SciPy, con pesos de muestreo (1 / probabilidad de inclusión).

``Logistic``  P(al menos un evento en la celda en el mes t+h | predictores en t): un modelo DIRECTO por
              horizonte, la parte "umbral" de los modelos hurdle de ViEWS (Fatalities002) y de Fritz et al.
              (2024, PLY_cal_double_hurdle). Regresión logística con penalización L2 por Newton-Raphson.
``Size``      log(1 + muertes) dado que hubo evento: mínimos cuadrados ponderados y su desviación residual
              (la parte "intensidad" del hurdle; normal en log1p, discretizada al muestrear).
``Hawkes``    proceso de Hawkes espacio-temporal en tiempo discreto (Browning et al. 2025; endémico-epidémico
              de Held et al. 2005):  lambda_{i,t+1} = exp(B_i g) + sum_c a_c Z_{i,t,c},  Y ~ BinNeg(lambda, theta),
              con a_c >= 0 (cada evento engendra en promedio sum a_c eventos: la razón de ramificación).
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, special


def _sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


class Logistic:
    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, y, w, max_iter: int = 50, tol: float = 1e-8):
        """Newton-Raphson amortiguado (retroceso sobre la log-verosimilitud penalizada); los pesos se
        normalizan a media 1, así ``l2`` es la penalización frente a una muestra del tamaño de la de entrenamiento."""
        X = np.asarray(X, np.float64)
        w = np.asarray(w, np.float64)
        w = w / w.mean()
        y = np.asarray(y, np.float64)
        sw = w.sum()
        self.mu = (w @ X) / sw
        sd = np.sqrt((w @ (X - self.mu) ** 2) / sw)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd]).astype(np.float32)   # float32: BLAS 2x
        k = Xs.shape[1]
        P = np.eye(k) * self.l2
        P[0, 0] = 1e-8
        ybar = np.clip((w @ y) / sw, 1e-6, 1 - 1e-6)
        b = np.zeros(k)
        b[0] = np.log(ybar / (1 - ybar))

        def objective(b):
            z = (Xs @ b.astype(np.float32)).astype(np.float64)
            return -(w @ (y * z - np.logaddexp(0.0, z))) + 0.5 * b @ P @ b

        f = objective(b)
        for it in range(max_iter):
            p = _sigmoid((Xs @ b.astype(np.float32)).astype(np.float64))
            g = (Xs.T @ (w * (y - p)).astype(np.float32)).astype(np.float64) - P @ b
            H = (Xs.T @ (Xs * (w * p * (1 - p)).astype(np.float32)[:, None])).astype(np.float64) + P
            step = np.linalg.solve(H, g)
            t = 1.0
            while True:
                fn = objective(b + t * step)
                if fn <= f + 1e-12 * abs(f) or t < 1e-6:
                    break
                t *= 0.5
            b = b + t * step
            done = abs(f - fn) <= tol * max(abs(f), 1.0)
            f = fn
            if done:
                break
        self.coef = b
        self.iterations = it + 1
        return self

    def logit(self, X):
        return self.coef[0] + ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.coef[1:]

    def predict(self, X):
        return _sigmoid(self.logit(X))

    def contributions(self, X):
        """Aporte de cada predictor al logit respecto del promedio de entrenamiento (coef x valor estandarizado)."""
        return ((np.asarray(X, np.float64) - self.mu) / self.sd) * self.coef[1:]

    def to_dict(self):
        return {"coef": self.coef.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "l2": self.l2}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"])
        m.coef, m.mu, m.sd = (np.asarray(d[k]) for k in ("coef", "mu", "sd"))
        return m


class Size:
    """log1p(muertes) | evento ~ Normal(X b, s)."""

    def __init__(self, l2: float = 1.0):
        self.l2 = l2

    def fit(self, X, deaths, w):
        X = np.asarray(X, np.float64)
        w = np.asarray(w, np.float64)
        w = w / w.mean()
        sw = w.sum()
        self.mu = (w @ X) / sw
        sd = np.sqrt((w @ (X - self.mu) ** 2) / sw)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Xs = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        y = np.log1p(deaths)
        P = np.eye(Xs.shape[1]) * self.l2
        P[0, 0] = 0.0
        self.coef = np.linalg.solve(Xs.T @ (Xs * w[:, None]) + P, Xs.T @ (w * y))
        r = y - Xs @ self.coef
        self.sigma = float(np.sqrt((w @ r ** 2) / sw))
        return self

    def mean_log(self, X):
        return self.coef[0] + ((np.asarray(X, np.float64) - self.mu) / self.sd) @ self.coef[1:]

    def sample(self, X, rng, S):
        """(n, S) muertes simuladas dado evento."""
        m = self.mean_log(X)
        return np.maximum(np.round(np.expm1(m[:, None] + self.sigma * rng.standard_normal((len(m), S)))), 0.0)

    def to_dict(self):
        return {"coef": self.coef.tolist(), "mu": self.mu.tolist(), "sd": self.sd.tolist(), "sigma": self.sigma,
                "l2": self.l2}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"])
        m.coef, m.mu, m.sd = (np.asarray(d[k]) for k in ("coef", "mu", "sd"))
        m.sigma = d["sigma"]
        return m


class Hawkes:
    """Binomial negativa (NB2: var = lambda + lambda^2 / theta) con media de Hawkes. El fondo ``exp(B g)``
    usa covariables estandarizadas; las excitaciones ``Z`` entran crudas (eventos ponderados por el núcleo)."""

    def __init__(self, l2_background: float = 1e-2):
        self.l2 = l2_background

    def fit(self, B, Z, y, w, max_iter: int = 400, tol: float = 1e-7):
        """EM sobre la estructura de ramificación (Mohler et al. 2011; Veen y Schoenberg 2008): cada evento se
        reparte entre el fondo y cada canal de excitación en proporción a su parte de lambda; el paso M de
        las alfas es cerrado y el del fondo es una regresión de Poisson (Newton). La media se estima por
        cuasi-verosimilitud de Poisson; después ``theta`` (sobredispersión) por máxima verosimilitud BinNeg."""
        B = np.asarray(B, np.float64)
        Z = np.asarray(Z, np.float64)
        y = np.asarray(y, np.float64)
        w = np.asarray(w, np.float64)
        w = w / w.mean()
        sw = w.sum()
        self.mu = (w @ B[:, 1:]) / sw
        sd = np.sqrt((w @ (B[:, 1:] - self.mu) ** 2) / sw)
        self.sd = np.where(sd > 1e-9, sd, 1.0)
        Bs = np.column_stack([B[:, 0], (B[:, 1:] - self.mu) / self.sd])
        kb, kz = Bs.shape[1], Z.shape[1]
        P = np.eye(kb) * self.l2
        P[0, 0] = 0.0
        ybar = max((w @ y) / sw, 1e-8)
        g = np.zeros(kb)
        g[0] = np.log(ybar * 0.2)
        a = np.full(kz, 0.01)
        zden = Z.T @ w
        pos = y > 0
        Bp, Zp, yp, wp = Bs[pos], Z[pos], y[pos], w[pos]
        ll_old = -np.inf
        for it in range(max_iter):
            base_p = np.exp(np.clip(Bp @ g, -40, 20))
            ex_p = Zp * a
            lam_p = base_p + ex_p.sum(1) + 1e-300
            r = (wp * yp / lam_p)
            a = np.where(zden > 0, (ex_p.T @ r) / np.maximum(zden, 1e-300), 0.0)
            rb = np.zeros(len(y))
            rb[pos] = yp * base_p / lam_p                                  # eventos atribuidos al fondo
            for _ in range(2):                                              # Newton del GLM de Poisson del fondo
                mu_all = np.exp(np.clip(Bs @ g, -40, 20))
                grad = Bs.T @ (w * (rb - mu_all)) - P @ g
                Hm = Bs.T @ (Bs * (w * mu_all)[:, None]) + P
                g = g + np.linalg.solve(Hm, grad)
            lam = np.exp(np.clip(Bs @ g, -40, 20)) + Z @ a
            ll = float(w @ (y * np.log(lam + 1e-300) - lam)) / sw
            if abs(ll - ll_old) <= tol * max(abs(ll), 1.0):
                break
            ll_old = ll
        self.gamma, self.alpha = g, a
        self.iterations = it + 1
        self.converged = it + 1 < max_iter
        lam = np.exp(np.clip(Bs @ g, -40, 20)) + Z @ a + 1e-12

        def nb_nll(lt):
            th = np.exp(lt)
            return -float(w @ (special.gammaln(y + th) - special.gammaln(th) + th * np.log(th / (th + lam))
                               + y * np.log(lam / (th + lam)))) / sw

        r = optimize.minimize_scalar(nb_nll, bounds=(np.log(1e-3), np.log(1e3)), method="bounded")
        self.theta = float(np.exp(r.x))
        self.nll = float(r.fun)
        return self

    def background(self, B):
        B = np.asarray(B, np.float64)
        return np.exp(np.clip(self.gamma[0] * B[..., 0] + ((B[..., 1:] - self.mu) / self.sd) @ self.gamma[1:], -40, 20))

    def intensity(self, B, Z):
        return self.background(B) + np.asarray(Z, np.float64) @ self.alpha

    def p_any(self, lam):
        """P(Y >= 1) con la binomial negativa."""
        return 1.0 - (self.theta / (self.theta + lam)) ** self.theta

    def sample(self, lam, rng, small: float = 1e-3):
        """Conteos BinNeg; donde lambda < ``small`` basta un Bernoulli con P(Y >= 1) (P(Y >= 2) es O(lambda^2))."""
        lam = np.maximum(lam, 1e-12)
        out = (rng.random(lam.shape) < self.p_any(lam)).astype(np.float32)
        big = lam >= small
        if big.any():
            lb = lam[big]
            out[big] = rng.negative_binomial(self.theta, self.theta / (self.theta + lb))
        return out

    def to_dict(self):
        return {"gamma": self.gamma.tolist(), "alpha": self.alpha.tolist(), "theta": float(self.theta),
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "l2": self.l2, "converged": self.converged}

    @classmethod
    def from_dict(cls, d):
        m = cls(d["l2"])
        m.gamma, m.alpha, m.mu, m.sd = (np.asarray(d[k]) for k in ("gamma", "alpha", "mu", "sd"))
        m.theta, m.converged = d["theta"], d.get("converged", True)
        return m
