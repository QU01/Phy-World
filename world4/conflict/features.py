"""Predictores por celda-mes, calculados en un barrido mes a mes que en el mes ``t`` sólo ve eventos <= ``t``.

Dos familias, con la misma información:

* ``FEATURES`` para el modelo de umbral (``models.Logistic``): historia del conflicto con decaimiento
  (ViEWS, Hegre et al. 2019, 2021: sumas con vida media y ``0.5 ** (meses desde el último evento / 12)``),
  rezagos espaciales en los anillos 1 y 2 (difusión por escalada, Schutte y Weidmann 2011), distancia al
  conflicto activo, el país (todos sus eventos, también los que no tienen celda fiable) y la geografía de
  PRIO-GRID ("natural and social geography" de ViEWS).
* ``HAWKES`` para el proceso de Hawkes discreto (``models.Hawkes``): los estados de excitación de núcleos
  geométricos ``g(k) = b (1 - b) ** (k - 1)`` con vidas medias de 1, 4 y 24 meses, propios, de los anillos 1 y 2
  y del país, por tipo de violencia (Browning et al. 2025, con mezcla de núcleos y excitación cruzada entre tipos).
"""
from __future__ import annotations

import numpy as np

from world4.conflict.grid import STATIC, TYPES, Events, Grid

HALF_LIVES = (1, 4, 24)                       # meses; b = 1 - 0.5 ** (1 / vida media)
BETA = {hl: 1.0 - 0.5 ** (1.0 / hl) for hl in HALF_LIVES}
COUNTRY_HL = 4
BURN_IN = 60                                  # primeros meses (1989-1993) sólo calientan los estados
NEAR_THRESHOLD = 0.05                         # masa decaída (24 m) en la celda y sus dos anillos: "con historia"

FEATURES = tuple(f"{v}_{k}" for v in TYPES for k in ("e1", "e4", "e24", "since")) + (
    "sb_d3", "sb_d24", "r1_e4", "r1_e24", "r2_e4", "r2_e24", "r1_since", "dist_active",
    "country_e6", "country_d12", "country_share12") + STATIC
HAWKES = tuple(f"{u}_{r}_{hl}" for u in TYPES for r in ("own", "r1", "r2") for hl in HALF_LIVES) + tuple(
    f"{u}_country" for u in TYPES)
BACKGROUND = ("const",) + STATIC


class Sweep:
    """Estados acumulados hasta el último mes procesado (``self.t``)."""

    def __init__(self, ev: Events, grid: Grid):
        self.ev, self.g = ev, grid
        n, C = grid.n, grid.n_countries
        self.E = {v: {hl: np.zeros(n) for hl in HALF_LIVES} for v in TYPES}
        self.D = {hl: np.zeros(n) for hl in (3, 24)}                     # muertes estatales
        self.last = {v: np.full(n, -10 ** 6) for v in TYPES}
        self.last_any = np.full(n, -10 ** 6)
        self.CE = np.zeros(C)                                            # país: todos los eventos, vida media 6
        self.CD = np.zeros(C)                                            # país: muertes, vida media 12
        self.t = -1

    def step(self) -> None:
        t = self.t + 1
        cnt, dth = self.ev.month(t)
        for v in TYPES:
            for hl in HALF_LIVES:
                self.E[v][hl] = self.E[v][hl] * (0.5 ** (1.0 / hl)) + cnt[v]
            self.last[v] = np.where(cnt[v] > 0, t, self.last[v])
        for hl in self.D:
            self.D[hl] = self.D[hl] * (0.5 ** (1.0 / hl)) + dth["sb"]
        anyc = sum(cnt[v] for v in TYPES)
        self.last_any = np.where(anyc > 0, t, self.last_any)
        self.CE = self.CE * 0.5 ** (1 / 6) + sum(self.ev.country_count[v][t] for v in TYPES)
        self.CD = self.CD * 0.5 ** (1 / 12) + self.ev.country_deaths[t]
        self.t = t

    def run_to(self, t: int) -> None:
        while self.t < t:
            self.step()

    def _since(self, last):
        return 0.5 ** ((self.t - last) / 12.0)

    def any_decayed(self, hl):
        return sum(self.E[v][hl] for v in TYPES)

    def near_history(self) -> np.ndarray:
        e = self.any_decayed(24)
        return (e + self.g.ring1 @ e + self.g.ring2 @ e) > NEAR_THRESHOLD

    def features(self) -> np.ndarray:
        g = self.g
        cols = []
        for v in TYPES:
            cols += [np.log1p(self.E[v][1]), np.log1p(self.E[v][4]), np.log1p(self.E[v][24]), self._since(self.last[v])]
        cols += [np.log1p(self.D[3]), np.log1p(self.D[24])]
        e4, e24 = self.any_decayed(4), self.any_decayed(24)
        cols += [np.log1p(g.ring1 @ e4), np.log1p(g.ring1 @ e24), np.log1p(g.ring2 @ e4), np.log1p(g.ring2 @ e24)]
        z = self._since(self.last_any)
        cols.append(np.maximum(z, g.ring1.multiply(z[None, :]).max(axis=1).toarray().ravel()))
        active = (self.t - self.last_any) < 12
        cols.append(np.log1p(g.distance_to(active)))
        share = (g.country_matrix @ active.astype(float)) / g.cells_per_country
        cols += [np.log1p(self.CE[g.country]), np.log1p(self.CD[g.country]), share[g.country]]
        X = np.column_stack(cols + [g.static])
        return X.astype(np.float32)

    def hawkes(self) -> np.ndarray:
        """(n, 30) estados de excitación para el mes t+1 (cada uno multiplicado por su b)."""
        return hawkes_design(self.g, self.E).astype(np.float32)


def hawkes_design(g: Grid, E: dict) -> np.ndarray:
    """``E[v][hl]`` de forma (n,) o (n, S) -> columnas de ``HAWKES`` apiladas en el último eje."""
    cols = []
    for u in TYPES:
        for ring in ("own", "r1", "r2"):
            for hl in HALF_LIVES:
                x = E[u][hl]
                if ring == "r1":
                    x = g.ring1 @ x
                elif ring == "r2":
                    x = g.ring2 @ x
                cols.append(BETA[hl] * x)
    for u in TYPES:
        c = g.country_matrix @ E[u][COUNTRY_HL]
        c = c[g.country] / (g.cells_per_country[g.country] if np.ndim(c) == 1 else g.cells_per_country[g.country][:, None])
        cols.append(BETA[COUNTRY_HL] * c)
    return np.stack(cols, axis=-1)


def build(ev: Events, grid: Grid, *, until: int, p_near: float = 0.25, p_far: float = 0.004,
          snapshots=(), seed: int = 0, log=None) -> dict:
    """Barrido de ``BURN_IN`` a ``until`` (inclusive). Filas de entrenamiento: en cada mes, una muestra de
    celdas estratificada SÓLO por su historia (sin mirar el resultado; nada posterior al mes decide qué
    fila entra): con historia cercana probabilidad ``p_near``, sin ella ``p_far``; peso = 1 / probabilidad
    (ViEWS submuestrea los ceros por la misma razón: 99,5 % de las celda-mes no tienen eventos).
    ``snapshots``: meses en los que se guardan los predictores y estados de TODAS las celdas (orígenes)."""
    rng = np.random.default_rng(seed)
    sw = Sweep(ev, grid)
    snaps = set(int(s) for s in snapshots)
    rows = {"month": [], "cell": [], "weight": [], "X": [], "Z": [], "near": []}
    out_snap = {}
    for t in range(until + 1):
        sw.step()
        if t < BURN_IN and t not in snaps:
            continue
        X = sw.features()
        Z = sw.hawkes()
        if t in snaps:
            out_snap[t] = {"X": X, "Z": Z, "E": {v: {hl: sw.E[v][hl].copy() for hl in HALF_LIVES} for v in TYPES},
                           "near": sw.near_history()}
        if t >= BURN_IN:
            near = sw.near_history()
            p = np.where(near, p_near, p_far)
            take = rng.random(grid.n) < p
            idx = np.nonzero(take)[0]
            rows["month"].append(np.full(len(idx), t, np.int32))
            rows["cell"].append(idx.astype(np.int32))
            rows["weight"].append((1.0 / p[idx]).astype(np.float32))
            rows["X"].append(X[idx])
            rows["Z"].append(Z[idx])
            rows["near"].append(near[idx])
        if log and t % 60 == 0:
            log(f"  sweep {t + 1}/{until + 1} months")
    out = {k: np.concatenate(v) for k, v in rows.items()}
    out["snapshots"] = out_snap
    return out
