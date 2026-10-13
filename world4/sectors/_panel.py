"""Estimación de las ecuaciones de conducta de los sectores: MCO agrupado sobre el panel país-año
con covarianza robusta por país (cluster), restricciones de signo declaradas y un efecto propio
de cada país encogido hacia cero. Sólo se usan pares cuyo año final es <= el corte (origen)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Equation:
    name: str
    terms: tuple                     # nombres de las columnas de X (la primera es la constante)
    beta: np.ndarray
    cov: np.ndarray                  # robusta por país
    sd: float                        # desviación típica del residuo
    n: int
    n_countries: int
    cutoff_year: int
    dropped: tuple = ()              # términos a 0 por violar su signo declarado
    offset: np.ndarray = None        # (N,) efecto propio de cada país, encogido
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"terms": list(self.terms), "beta": [round(float(b), 6) for b in self.beta],
                "se": [round(float(s), 6) for s in np.sqrt(np.maximum(np.diag(self.cov), 0))],
                "resid_sd_robust": round(self.sd, 6), "resid_sd": round(self.meta.get("sd_raw", self.sd), 6),
                "common_sd": round(self.meta.get("sd_common", 0.0), 6),
                "common_rho": round(self.meta.get("rho_common", 0.0), 3),
                "idio_rho": round(self.meta.get("rho_idio", 0.0), 3),
                "n": self.n, "countries": self.n_countries,
                "cutoff": self.cutoff_year, "sign_dropped": list(self.dropped)}

    def draw(self, R: int, rng) -> np.ndarray:
        """(R, K) coeficientes por miembro: incertidumbre paramétrica del ajuste."""
        if R == 1:
            return self.beta[None].copy()
        return rng.multivariate_normal(self.beta, self.cov, size=R, method="eigh")


def time_effects(e, year, min_n=20):
    """Componente común a todos los países: media del residuo por año (años con >= ``min_n``
    países), su desviación típica y su autocorrelación de orden 1."""
    yrs = np.unique(year)
    m = np.array([e[year == y].mean() if (year == y).sum() >= min_n else np.nan for y in yrs])
    m = m[np.isfinite(m)]
    if len(m) < 5:
        return 0.0, 0.0
    m = m - m.mean()
    rho = float(np.clip(np.corrcoef(m[:-1], m[1:])[0, 1], 0.0, 0.95)) if np.std(m) > 0 else 0.0
    return float(np.std(m)), rho


def fit(name, y, X, country, terms, cutoff_year, signs=None, n_countries=None, shrink=10.0, recent=None,
        year=None, weights=None):
    """y (n,), X (n, K), country (n,) índice de país. ``signs``: {término: +1 | -1}.
    ``recent`` (n,) bool: filas que entran en el efecto propio (últimos años antes del corte).
    ``year`` (n,): para estimar el choque común a todos los países (efectos de año).
    ``weights`` (n,): mínimos cuadrados ponderados (p. ej. por el tamaño del país, para que la
    ecuación represente el agregado mundial y no al país mediano)."""
    ok = np.isfinite(y) & np.all(np.isfinite(X), 1)
    if weights is not None:
        ok &= np.isfinite(weights) & (weights > 0)
    w = np.ones(int(ok.sum())) if weights is None else weights[ok] / np.mean(weights[ok])
    sw = np.sqrt(w)
    y, X, country = y[ok], X[ok], country[ok]
    recent = None if recent is None else recent[ok]
    year = None if year is None else year[ok]
    keep = list(range(X.shape[1]))
    dropped = []
    while True:
        Xk = X[:, keep]
        beta_k = np.linalg.lstsq(Xk * sw[:, None], y * sw, rcond=None)[0]
        bad = [k for j, k in enumerate(keep) if signs and terms[k] in signs and beta_k[j] * signs[terms[k]] < 0]
        if not bad:
            break
        keep.remove(bad[0])
        dropped.append(terms[bad[0]])
    e = y - Xk @ beta_k
    XtXi = np.linalg.pinv((Xk * w[:, None]).T @ Xk)
    groups = np.unique(country)
    G = len(groups)
    if G == 1:             # una sola serie (mercados mundiales): MCO clásico, corregido por grados de libertad
        cov_k = XtXi * float((w * e) @ e) / max(len(e) - len(keep), 1)
    else:
        meat = np.zeros((len(keep), len(keep)))
        for g in groups:
            s = Xk[country == g].T @ (w * e)[country == g]
            meat += np.outer(s, s)
        cov_k = XtXi @ meat @ XtXi * (G / max(G - 1, 1))
    beta = np.zeros(X.shape[1])
    cov = np.zeros((X.shape[1], X.shape[1]))
    beta[keep] = beta_k
    cov[np.ix_(keep, keep)] = cov_k
    N = int(n_countries if n_countries is not None else country.max() + 1)
    sel = np.ones_like(e, bool) if recent is None else recent
    off = (np.bincount(country[sel], weights=e[sel], minlength=N)
           / (np.bincount(country[sel], minlength=N) + shrink))
    # El ruido por miembro usa una escala robusta (1.4826 MAD): los residuos tienen colas pesadas
    # (genocidios, hambrunas, éxodos) que una normal con la desviación típica repartiría por todos
    # los años. Las colas quedan en ``meta["sd_raw"]`` y no se simulan [declarado].
    sd = 1.4826 * float(np.median(np.abs(e - np.median(e))))
    sd_t, rho_t = time_effects(e, year) if year is not None else (0.0, 0.0)
    rho_i = within_rho(e, country, year, sd) if year is not None else 0.0
    return Equation(name, tuple(terms), beta, cov, sd, int(len(y)), int(G), int(cutoff_year),
                    tuple(dropped), off, meta={"sd_raw": float(np.std(e)), "sd_common": sd_t, "rho_common": rho_t,
                                               "rho_idio": rho_i})


def within_rho(e, country, year, sd):
    """Autocorrelación de orden 1 del residuo dentro de cada país (años consecutivos), con los
    residuos recortados a +-4 escalas robustas para que un año extremo no la decida."""
    ec = np.clip(e, -4 * sd, 4 * sd) if sd > 0 else e
    key = country.astype(np.int64) * 10_000 + year.astype(np.int64)
    pos = {k: i for i, k in enumerate(key)}
    prev = np.array([pos.get(k - 1, -1) for k in key])
    ok = prev >= 0
    if ok.sum() < 30:
        return 0.0
    a, b = ec[ok], ec[prev[ok]]
    return float(np.clip(np.corrcoef(a, b)[0, 1], 0.0, 0.95))


class Shock:
    """Ruido AR(1) estacionario de desviación ``sd`` y persistencia ``rho``, de forma (R, n):
    n = 1 para el choque común a todos los países de un miembro, n = N para el propio de cada país."""

    def __init__(self, sd: float, rho: float, shape, rng, on: bool = True):
        self.sd, self.rho, self.shape, self.rng = float(sd), float(rho), shape, rng
        self.on = on and self.sd > 0
        self.x = self.sd * rng.standard_normal(shape) if self.on else np.zeros(shape)

    def __call__(self):
        if not self.on:
            return 0.0
        self.x = self.rho * self.x + self.sd * np.sqrt(1 - self.rho ** 2) * self.rng.standard_normal(self.shape)
        return self.x


def shocks(eq: Equation, R: int, N: int, rng, on: bool = True):
    """(común, propio) para una ecuación: efectos de año y residuo de cada país, ambos AR(1)."""
    m = eq.meta
    return (Shock(m.get("sd_common", 0.0), m.get("rho_common", 0.0), (R, 1), rng, on),
            Shock(eq.sd, m.get("rho_idio", 0.0), (R, N), rng, on))
