"""Piezas estadísticas pequeñas del L2: logit y Poisson por IRLS con cresta, y mínimos cuadrados.

Se escriben aquí, y no se toman del Atlas portado, para que el L2 no dependa de los datos del Atlas.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def add_const(X: np.ndarray) -> np.ndarray:
    return np.concatenate([np.ones((len(X), 1)), X], 1)


def _newton(Xc, y, w, ridge, iters, mean, dmean, nll):
    """Newton con retroceso sobre la verosimilitud penalizada (la constante no se penaliza)."""
    b = np.zeros(Xc.shape[1])
    pen = ridge * w.sum() * np.eye(Xc.shape[1])
    pen[0, 0] = 0.0
    f = nll(Xc @ b, y, w) + 0.5 * b @ pen @ b
    for _ in range(iters):
        eta = Xc @ b
        mu, v = mean(eta), np.maximum(dmean(eta), 1e-10) * w
        step = np.linalg.solve(Xc.T @ (Xc * v[:, None]) + pen + 1e-10 * np.eye(len(b)),
                               Xc.T @ (w * (y - mu)) - pen @ b)
        a = 1.0
        while a > 1e-6:
            b1 = b + a * step
            f1 = nll(Xc @ b1, y, w) + 0.5 * b1 @ pen @ b1
            if np.isfinite(f1) and f1 <= f + 1e-12:
                break
            a *= 0.5
        else:
            break
        done = abs(f - f1) < 1e-9 * (abs(f) + 1.0)
        b, f = b1, f1
        if done:
            break
    return b


def _scaled(X):
    X = np.asarray(X, float)
    m, s = X.mean(0), X.std(0)
    s = np.where(s > 0, s, 1.0)
    return add_const((X - m) / s), m, s


def _unscale(b, m, s):
    """Coeficientes en la escala original de X a partir de los ajustados sobre (X - m) / s."""
    out = np.empty_like(b)
    out[1:] = b[1:] / s
    out[0] = b[0] - np.sum(b[1:] * m / s)
    return out


def logit_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 60, w=None) -> np.ndarray:
    """Logit con constante (la añade). Los rasgos se estandarizan por dentro (la cresta es sobre esa escala) y
    los coeficientes se devuelven en la escala original: (1 + p,)."""
    Xc, m, s = _scaled(X)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)

    def nll(eta, y, w):
        return float(np.sum(w * (np.logaddexp(0.0, eta) - y * eta)))

    b = _newton(Xc, y, w, ridge, iters, sigmoid, lambda e: sigmoid(e) * (1 - sigmoid(e)), nll)
    return _unscale(b, m, s)


def poisson_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-4, iters: int = 60, w=None) -> np.ndarray:
    """Poisson log-lineal con constante, mismo esquema que ``logit_fit``."""
    Xc, m, s = _scaled(X)
    y = np.asarray(y, float)
    w = np.ones(len(y)) if w is None else np.asarray(w, float)

    def mean(eta):
        return np.exp(np.clip(eta, -30, 30))

    def nll(eta, y, w):
        return float(np.sum(w * (mean(eta) - y * eta)))

    b = _newton(Xc, y, w, ridge, iters, mean, mean, nll)
    return _unscale(b, m, s)


def ols_fit(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Mínimos cuadrados con constante. Devuelve (coeficientes, residuos)."""
    Xc = add_const(np.asarray(X, float))
    y = np.asarray(y, float)
    pen = ridge * len(y) * np.eye(Xc.shape[1])
    pen[0, 0] = 0.0
    b = np.linalg.solve(Xc.T @ Xc + pen, Xc.T @ y)
    return b, y - Xc @ b


def predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return b[0] + np.asarray(X, float) @ b[1:]
