"""Recalibración (D) de lo que sale del L1 antes de entregarlo a nadie.

* Eventos: probabilidad cruda -> isotónica (PAV) con >= 200 pares, Platt con 30-199, identidad
  con menos. Es la misma escalera que el calibrador de Prime Radiant.
* Numéricas: recalibración por cuantiles (Kuleshov et al. 2018). Con los PIT de pronósticos
  pasados se estima C, la CDF de los PIT, y la distribución entregada es G(x) = C(F_ens(x)).
  F_ens es una mezcla de núcleos t de Student (3 g. l., ancho de Silverman) sobre los miembros.
  Con núcleo gaussiano una observación a varias desviaciones del ensemble daba PIT = 0 o 1
  EXACTOS y ninguna C podía volver a repartir esa masa (la deuda tenía el 25 % de los PIT en
  los deciles extremos); la cola t no satura.

Los pares de entrenamiento los decide quien llama: siempre desenlaces publicados antes de as_of.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import t as student_t

KERNEL_DF = 3

MIN_PLATT, MIN_ISOTONIC = 30, 200


# ------------------------------------------------------------------ eventos
def _pav(x, y):
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    vals, wts, starts = [], [], []
    for i, v in enumerate(ys):
        vals.append(v); wts.append(1.0); starts.append(i)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            w = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / w
            wts[-2] = w
            vals.pop(); wts.pop(); starts.pop()
    fitted = np.empty(len(xs))
    for b, (v, s) in enumerate(zip(vals, starts)):
        e = starts[b + 1] if b + 1 < len(starts) else len(xs)
        fitted[s:e] = v
    # nodos únicos de x (media de y en empates) para interpolar
    ux, idx = np.unique(xs, return_index=True)
    uy = np.array([fitted[xs == u].mean() for u in ux])
    return ux, uy


class EventCalibrator:
    def __init__(self, raw=(), outcomes=()):
        raw = np.asarray(raw, float); y = np.asarray(outcomes, float)
        ok = np.isfinite(raw) & np.isfinite(y)
        raw, y = raw[ok], y[ok]
        self.n = int(len(raw))
        self.kind = "identity"
        if self.n >= MIN_ISOTONIC and 0 < y.mean() < 1:
            self.kind = "isotonic"
            self.x, self.y = _pav(raw, y)
            # cuantiles extremos acotados: un recalibrador no promete 0 ni 1
            self.y = np.clip(self.y, 0.5 / (self.n + 1), 1 - 0.5 / (self.n + 1))
        elif self.n >= MIN_PLATT and 0 < y.mean() < 1:
            self.kind = "platt"
            z = np.log(np.clip(raw, 1e-4, 1 - 1e-4) / (1 - np.clip(raw, 1e-4, 1 - 1e-4)))
            a, b = 1.0, 0.0
            for _ in range(100):                                       # Newton
                p = 1 / (1 + np.exp(-(a * z + b)))
                g = np.array([np.sum((p - y) * z), np.sum(p - y)])
                w = p * (1 - p)
                H = np.array([[np.sum(w * z * z), np.sum(w * z)], [np.sum(w * z), np.sum(w)]]) + 1e-9 * np.eye(2)
                step = np.linalg.solve(H, g)
                a, b = a - step[0], b - step[1]
                if np.max(np.abs(step)) < 1e-10:
                    break
            self.a, self.b = float(a), float(b)

    def __call__(self, p):
        p = np.asarray(p, float)
        if self.kind == "isotonic":
            return np.interp(p, self.x, self.y)
        if self.kind == "platt":
            z = np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
            return 1 / (1 + np.exp(-(self.a * z + self.b)))
        return p

    def describe(self) -> dict:
        return {"kind": self.kind, "n": self.n}


# ------------------------------------------------------------------ numéricas
def _bandwidth(m: np.ndarray) -> float:
    m = m[np.isfinite(m)]
    if len(m) < 2:
        return 1.0
    sd = np.std(m, ddof=1)
    iqr = np.subtract(*np.percentile(m, [75, 25])) / 1.349
    s = min(sd, iqr) if iqr > 0 else sd
    return float(max(0.9 * s * len(m) ** (-0.2), 1e-3 * (abs(np.mean(m)) + 1e-3)))


def ens_cdf(members, x) -> np.ndarray:
    m = np.asarray(members, float); m = m[np.isfinite(m)]
    h = _bandwidth(m)
    x = np.atleast_1d(np.asarray(x, float))
    return np.mean(student_t.cdf((x[:, None] - m[None, :]) / h, KERNEL_DF), 1)


def ens_quantile(members, q) -> np.ndarray:
    m = np.asarray(members, float); m = m[np.isfinite(m)]
    h = _bandwidth(m)
    q = np.atleast_1d(np.asarray(q, float))
    lo = np.full(q.shape, m.min() - 1e4 * h); hi = np.full(q.shape, m.max() + 1e4 * h)
    for _ in range(80):                                                   # bisección
        mid = 0.5 * (lo + hi)
        below = np.mean(student_t.cdf((mid[:, None] - m[None, :]) / h, KERNEL_DF), 1) < q
        lo = np.where(below, mid, lo); hi = np.where(below, hi, mid)
    return 0.5 * (lo + hi)


class NumericCalibrator:
    """C = CDF empírica de los PIT pasados, con 1 pseudo-observación uniforme en cada extremo."""

    def __init__(self, pits=()):
        u = np.sort(np.asarray([p for p in pits if np.isfinite(p)], float))
        self.n = int(len(u))
        self.kind = "quantile" if self.n >= MIN_PLATT else "identity"
        n = self.n
        self.u = np.concatenate([[0.0], u, [1.0]]) if n else np.array([0.0, 1.0])
        self.c = np.linspace(0, 1, len(self.u))

    def C(self, u):
        return np.interp(u, self.u, self.c) if self.kind == "quantile" else np.asarray(u, float)

    def Cinv(self, tau):
        return np.interp(tau, self.c, self.u) if self.kind == "quantile" else np.asarray(tau, float)

    def cdf(self, members, x):
        return self.C(ens_cdf(members, x))

    def quantile(self, members, tau):
        u = np.clip(self.Cinv(np.asarray(tau, float)), 1e-6, 1 - 1e-6)
        return ens_quantile(members, u)

    def describe(self) -> dict:
        d = {"kind": self.kind, "n": self.n}
        if getattr(self, "selected", None):
            d["selected"] = self.selected
        return d


def bin_probabilities(cal: NumericCalibrator, members, edges) -> list[float]:
    F = cal.cdf(members, np.asarray(edges, float))
    cuts = np.concatenate([[0.0], np.clip(F, 0, 1), [1.0]])
    p = np.maximum(np.diff(cuts), 0.0)
    return (p / p.sum()).tolist()


def ks_distance(pits) -> float:
    """D de Kolmogorov-Smirnov contra U(0,1): la mayor desviación de cobertura a cualquier nivel."""
    u = np.sort(np.asarray(pits, float)); n = len(u)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    return float(max(np.max(i / n - u), np.max(u - (i - 1) / n)))


def ks_uniform_p(pits) -> float:
    """p-valor de Kolmogorov-Smirnov contra U(0,1) (aproximación asintótica)."""
    u = np.sort(np.asarray(pits, float)); n = len(u)
    if n == 0:
        return float("nan")
    i = np.arange(1, n + 1)
    d = max(np.max(i / n - u), np.max(u - (i - 1) / n))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    s = sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return float(min(1.0, max(0.0, 2 * s)))




# ------------------------------------------------------------------ F7: sesgo reciente
Z_SCALE = 2.5631031310892007          # p90 - p10 de una normal estándar


def ens_scale(members):
    """(p50, s) del ensemble: s = (p90 - p10) / 2,563, la desviación de una normal con esa banda."""
    m = np.asarray(members, float); m = m[np.isfinite(m)]
    p10, p50, p90 = np.percentile(m, [10, 50, 90])
    return float(p50), float(max((p90 - p10) / Z_SCALE, 1e-6))


class ZCalibrator:
    """Recalibración en el espacio del residuo estandarizado z = (y - p50) / s (F7).

    ``bias`` es la mediana de z de los desenlaces publicados en los últimos años (todos los horizontes):
    el sesgo actual del modelo. La distribución entregada es G(x) = Ĉ((x - p50)/s - bias), con Ĉ la CDF
    empírica de los z de entrenamiento, cada uno corregido con el sesgo que se conocía en SU fecha. Ataca la
    deriva por décadas de F0.5: un recalibrador de PIT fijo no puede seguir un sesgo que cambia."""

    def __init__(self, z_train=(), bias: float = 0.0):
        z = np.sort(np.asarray([v for v in z_train if np.isfinite(v)], float))
        self.n, self.bias = int(len(z)), float(bias)
        self.kind = "recent_bias" if self.n >= MIN_PLATT else "identity"
        self.z = z
        self.c = np.arange(1, self.n + 1) / (self.n + 1)

    def cdf_z(self, z):
        z = np.asarray(z, float) - self.bias
        if self.kind == "identity":
            from scipy.stats import norm
            return norm.cdf(z)
        lo, hi = 0.5 / (self.n + 1), 1 - 0.5 / (self.n + 1)
        return np.interp(z, self.z, self.c, left=lo, right=hi)

    def cdf(self, members, x):
        p50, s = ens_scale(members)
        return self.cdf_z((np.asarray(x, float) - p50) / s)

    def quantile(self, members, tau):
        p50, s = ens_scale(members)
        tau = np.asarray(tau, float)
        if self.kind == "identity":
            from scipy.stats import norm
            z = norm.ppf(np.clip(tau, 1e-6, 1 - 1e-6))
        else:
            z = np.interp(tau, self.c, self.z)
        return p50 + s * (z + self.bias)

    def describe(self) -> dict:
        d = {"kind": self.kind, "n": self.n, "bias_z": self.bias}
        if getattr(self, "selected", None):
            d["selected"] = self.selected
        return d
