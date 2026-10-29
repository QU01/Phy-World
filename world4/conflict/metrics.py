"""Reglas de puntuación para pronósticos de eventos raros por celda (sin dependencias fuera de NumPy/SciPy).

* Brier y log-loss: propias para probabilidades de evento.
* AUROC y AUPR (precisión media): discriminación; con ~0,5 % de celda-mes con eventos el AUROC sale alto
  casi siempre, la AUPR es la que separa (ViEWS 2019 reporta ambas).
* CRPS de una distribución dada por muestras (la métrica principal del reto ViEWS 2023/24 para muertes):
  ``mean|X - y| - 0.5 mean|X - X'|``, con la fórmula ordenada para el segundo término.
* Calibración por intervalos fijos (en escala casi logarítmica: casi todas las probabilidades son pequeñas).
"""
from __future__ import annotations

import numpy as np
from scipy import stats

CAL_EDGES = np.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 + 1e-9])


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - y) ** 2))


def log_loss(p, y, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def auroc(p, y) -> float:
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = stats.rankdata(p)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(p, y) -> float:
    """AUPR como suma de precisiones en cada umbral distinto ponderadas por el aumento de exhaustividad."""
    y = np.asarray(y, bool)
    n1 = int(y.sum())
    if n1 == 0:
        return float("nan")
    o = np.argsort(-np.asarray(p, float), kind="stable")
    ps, ys = np.asarray(p, float)[o], y[o]
    tp, fp = np.cumsum(ys), np.cumsum(~ys)
    last = np.r_[np.nonzero(np.diff(ps))[0], len(ps) - 1]          # último índice de cada umbral distinto
    tp, fp = tp[last], fp[last]
    prec, rec = tp / (tp + fp), tp / n1
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def crps_samples(samples, y) -> np.ndarray:
    """CRPS por observación de una distribución empírica (n, S) contra y (n,)."""
    x = np.sort(np.asarray(samples, float), axis=1)
    S = x.shape[1]
    t1 = np.abs(x - np.asarray(y, float)[:, None]).mean(1)
    j = np.arange(1, S + 1)
    t2 = (x * (2 * j - S - 1)).sum(1) / S ** 2
    return t1 - t2


def calibration(p, y) -> list:
    """[(p media, frecuencia observada, n)] por intervalo de ``CAL_EDGES`` (aditivo entre orígenes por n)."""
    p = np.asarray(p, float)
    k = np.digitize(p, CAL_EDGES) - 1
    out = []
    for b in range(len(CAL_EDGES) - 1):
        s = k == b
        out.append([float(p[s].sum()), float(np.asarray(y)[s].sum()), int(s.sum())])
    return out


def scores(p, y) -> dict:
    y = np.asarray(y, bool)
    return {"brier": brier(p, y), "log_loss": log_loss(p, y), "auroc": auroc(p, y), "aupr": average_precision(p, y),
            "events": int(y.sum()), "expected": float(np.sum(p)), "n": int(len(y))}
