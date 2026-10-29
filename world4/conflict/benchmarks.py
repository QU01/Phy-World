"""Los modelos de referencia del reto ViEWS 2023/24 (Hegre et al. 2025, JPR 62(6)) al nivel celda-mes, más la
persistencia y la frecuencia histórica. Todos usan sólo meses <= el origen ``T``.

Para probabilidad de evento (por tipo o cualquiera):
  ``persistence``   1 si hubo evento en T, si no 0.
  ``cell12``        fracción de los 12 meses T-11..T con evento en la celda.
  ``neighbors12``   fracción de los 9 x 12 valores (la celda y sus 8 vecinas, 12 meses) con evento: la versión
                    probabilística de VIEWS_bm_ph_conflictology_neighbors12, la mejor referencia en ``pgm``.
  ``history``       fracción de todos los meses desde 1989 con evento en la celda.
Para la distribución de muertes estatales (CRPS):
  ``zero``          VIEWS_bm_exactly_zero (CRPS = y).
  ``last``          VIEWS_bm_last_historical: Poisson con media = muertes en T.
  ``cell12``, ``neighbors12``: los 12 o 108 valores observados como muestra (las vecinas en el mar valen 0).
"""
from __future__ import annotations

import numpy as np

from world4.conflict.grid import Events, Grid

PROB = ("persistence", "cell12", "neighbors12", "history")
DEATHS = ("zero", "last", "cell12", "neighbors12")


def ring1_index(grid: Grid) -> np.ndarray:
    """(n, 8) índices de las vecinas; ``n`` (una columna de ceros) donde no hay celda terrestre."""
    A = grid.ring1
    out = np.full((grid.n, 8), grid.n, np.int64)
    for i in range(8):
        has = np.diff(A.indptr) > i
        out[has, i] = A.indices[A.indptr[:-1][has] + i]
    return out


def probabilities(ev: Events, grid: Grid, T: int, outcome: str) -> dict:
    w12 = ev.window("count", outcome, T - 11, T + 1) > 0                     # (12, n)
    hits12 = w12.sum(0).astype(float)
    hist = (ev.window("count", outcome, 0, T + 1) > 0).sum(0)
    return {"persistence": w12[-1].astype(float), "cell12": hits12 / 12.0,
            "neighbors12": (hits12 + grid.ring1 @ hits12) / 108.0, "history": hist / (T + 1.0)}


def death_samples(ev: Events, grid: Grid, T: int, *, samples: int = 200, seed: int = 0) -> dict:
    """{nombre: (n, k) muestras} de muertes estatales para cualquier mes futuro (no dependen del horizonte)."""
    rng = np.random.default_rng(seed)
    d12 = ev.window("deaths", "sb", T - 11, T + 1).T                         # (n, 12)
    pad = np.vstack([d12, np.zeros((1, 12))])
    nb = pad[ring1_index(grid)].reshape(grid.n, -1)                           # (n, 96)
    return {"zero": np.zeros((grid.n, 1)),
            "last": rng.poisson(np.repeat(d12[:, -1:], samples, axis=1)).astype(float),
            "cell12": d12, "neighbors12": np.hstack([d12, nb])}
