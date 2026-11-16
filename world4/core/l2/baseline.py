"""La vara del L2: lo que el simulador de agentes tiene que superar para que se le crea un pronóstico.

Tres referencias, de menos a más exigente, para las muertes mensuales de una unidad (país o celda):

``zero``     nunca pasa nada;
``last12``   la distribución empírica de los últimos 12 meses de la propia unidad (la referencia de
             persistencia que usan los concursos de predicción de conflicto; 12 miembros);
``hurdle``   un modelo estadístico de valla, DIRECTO para cada horizonte ``s``: un logit de «habrá alguna
             muerte dentro de s meses» y una regresión del logaritmo de las muertes cuando las hay, con
             residuos remuestreados (colas pesadas sin suponer una forma). Rasgos: la historia de la
             unidad a varias ventanas, el tiempo desde el último evento, los vecinos (celdas) o el resto de
             la región (países), el país de la celda, la población y la geografía de PRIO-GRID.

Todo se estima con filas cuyo desenlace es anterior al origen. No hay parámetro elegido mirando el periodo
de prueba: ventanas y rasgos se fijaron antes de la primera corrida de W13.
"""
from __future__ import annotations

import numpy as np

from world4.core.l2 import panel as pn
from world4.core.l2 import stats

STEPS = {"country": (1, 3, 6, 12, 24, 36), "cell": (1, 3, 6, 12)}
MIN_HISTORY = 36                 # meses de historia antes de la primera fila de entrenamiento
TRAIN_STRIDE = {"country": 1, "cell": 3}     # las celdas entrenan con un origen de cada tres (coste)
MIN_POSITIVE = 30                # con menos filas positivas, la parte positiva usa la media sin regresión


def features(P: pn.Panel, level: str, kind: str = "all") -> tuple[np.ndarray, list]:
    """(U, T+1, F): la columna ``t`` describe la unidad con datos anteriores a ``t``."""
    y = P.deaths(level, kind)
    f, names = [], []
    for k in pn.WINDOWS:
        f.append(np.log1p(pn.lagged_mean(y, k)))
        names.append(f"own{k}")
    f.append(np.log1p(pn.months_since(y)))
    names.append("since")
    T1 = y.shape[1] + 1
    yr = np.clip(P.year0 + np.arange(T1) // 12 - int(P.pop_years[0]), 0, len(P.pop_years) - 1)
    if level == "country":
        tot = y.sum(0, keepdims=True)
        for k in (3, 12):
            f.append(np.log1p(pn.lagged_mean(tot - y, k)))
            names.append(f"rest{k}")
        f.append(np.log(np.maximum(P.pop[:, yr], 1.0)))
        names.append("lpop")
    else:
        nb = pn.neighbour_sum(y, P.nbr)
        for k in (3, 12):
            f.append(np.log1p(pn.lagged_mean(nb, k)))
            names.append(f"nbr{k}")
        yk = P.deaths("country", kind)
        cc = P.cell_country
        for k in (3, 12):
            lk = np.log1p(pn.lagged_mean(yk, k))
            f.append(np.where((cc >= 0)[:, None], lk[np.maximum(cc, 0)], 0.0))
            names.append(f"country{k}")
        f.append(np.log1p(P.pop_cell[:, yr]))
        names.append("lpop")
        for j, n in enumerate(pn.COV_USED):
            f.append(np.repeat(P.cov[:, j][:, None], T1, 1))
            names.append(n)
    return np.stack(f, -1), names


class Hurdle:
    """Logit de ocurrencia + regresión del log de las muertes positivas, con sus residuos."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Hurdle":
        pos = y > 0
        self.rate = float(pos.mean())
        self.b_occ = stats.logit_fit(X, pos.astype(float)) if 0 < pos.sum() < len(y) else None
        if pos.sum() >= MIN_POSITIVE:
            self.b_pos, self.resid = stats.ols_fit(X[pos], np.log(y[pos]))
        else:
            ly = np.log(y[pos]) if pos.any() else np.zeros(1)
            self.b_pos, self.resid = None, ly - ly.mean()
            self.mean_log = float(ly.mean())
        return self

    def p_any(self, X: np.ndarray) -> np.ndarray:
        if self.b_occ is None:
            return np.full(len(X), self.rate)
        return stats.sigmoid(stats.predict(self.b_occ, X))

    def draw(self, X: np.ndarray, members: int, rng) -> np.ndarray:
        """(members, U) muertes."""
        p = self.p_any(X)
        mu = stats.predict(self.b_pos, X) if self.b_pos is not None else np.full(len(X), self.mean_log)
        u = rng.random((members, len(X)))
        e = self.resid[rng.integers(0, len(self.resid), (members, len(X)))]
        return np.where(u < p[None], np.exp(np.clip(mu[None] + e, -5, 14)), 0.0)


def fit(P: pn.Panel, level: str, t0: int, kind: str = "all", steps=None) -> dict:
    """Un ``Hurdle`` por horizonte, con desenlaces anteriores a ``t0``."""
    steps = STEPS[level] if steps is None else steps
    F, names = features(P, level, kind)
    y = P.deaths(level, kind)
    stride = TRAIN_STRIDE[level]
    out = {"names": names, "level": level, "kind": kind, "t0": int(t0), "models": {}}
    for s in steps:
        taus = np.arange(MIN_HISTORY, t0 - s + 1, stride)      # el desenlace tau + s - 1 es anterior a t0
        if len(taus) < 12:
            continue
        X = F[:, taus].reshape(-1, F.shape[-1])
        out["models"][int(s)] = Hurdle().fit(X, y[:, taus + s - 1].reshape(-1))
    return out


def _model_for(model: dict, s: int) -> Hurdle:
    ks = sorted(model["models"])
    if not ks:
        raise ValueError("no hurdle model could be fitted: not enough history before the origin")
    return model["models"][min(ks, key=lambda k: (abs(k - s), k))]


def forecast(P: pn.Panel, model: dict, t0: int, horizon: int, members: int = 100, seed: int = 0) -> np.ndarray:
    """(members, U, horizon): el paso ``h`` (0 = el mes ``t0``) usa el modelo del horizonte ajustado más próximo.
    Los rasgos son los del origen: no usan nada de ``t0`` en adelante."""
    F, _ = features(P, model["level"], model["kind"])
    X = F[:, min(t0, F.shape[1] - 1)]
    rng = np.random.default_rng(seed)
    return np.stack([_model_for(model, h + 1).draw(X, members, rng) for h in range(horizon)], -1)


def p_any(P: pn.Panel, model: dict, t0: int, horizon: int) -> np.ndarray:
    """(U, horizon) probabilidad de al menos una muerte en cada mes."""
    F, _ = features(P, model["level"], model["kind"])
    X = F[:, min(t0, F.shape[1] - 1)]
    return np.stack([_model_for(model, h + 1).p_any(X) for h in range(horizon)], -1)


def last12(P: pn.Panel, level: str, t0: int, horizon: int, kind: str = "all") -> np.ndarray:
    """(12, U, horizon): cada miembro es uno de los 12 meses anteriores al origen, repetido."""
    y = P.deaths(level, kind)[:, max(t0 - 12, 0):t0]
    if y.shape[1] < 12:
        y = np.concatenate([np.zeros((y.shape[0], 12 - y.shape[1])), y], 1)
    return np.repeat(y.T[:, :, None], horizon, 2)


def zero(P: pn.Panel, level: str, horizon: int) -> np.ndarray:
    U = P.K if level == "country" else P.C
    return np.zeros((1, U, horizon))
