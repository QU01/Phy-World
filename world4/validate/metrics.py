"""Métricas de validación de World4.

``auc`` corrige el defecto que el port del Atlas conserva a propósito (W2): el ``auc`` del Atlas
ordena con ``argsort`` y, con puntuaciones empatadas, el resultado depende del orden de desempate
que elija la implementación de ordenación de la CPU. En la validación del Atlas, el AUC de la
referencia "persistencia" de conflicto sale 0,93 en la máquina original y 0,81 en ésta con los
mismos datos. Aquí se usan rangos promedio (Mann-Whitney con empates), que no dependen del orden.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def auc(y, s) -> float:
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)                                   # rangos promedio en los empates
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def crps_ensemble(ens, y) -> np.ndarray:
    """CRPS de un ensemble (R, n) contra observaciones (n,), forma de los momentos ordenados."""
    e = np.sort(np.asarray(ens, float), 0)
    R = e.shape[0]
    term1 = np.mean(np.abs(e - np.asarray(y, float)[None]), 0)
    i = np.arange(1, R + 1)[:, None]
    return term1 - np.sum((2 * i - R - 1) * e, 0) / (R * R)
