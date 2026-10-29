"""Entrenamiento por origen, pronóstico por celda (1-12 meses) y la combinación de los dos modelos.

Para cada origen ``T`` (último mes con datos) y horizonte ``h``:

* ``Logistic`` (directo, uno por h y por resultado) da P(evento en t+h) para cada tipo (``sb``, ``ns``,
  ``os``) y para cualquiera (``any``), y otro para "al menos un evento en los próximos ``H`` meses".
* ``Hawkes`` (uno por tipo, un paso) se SIMULA ``S`` veces desde los estados de T: cada trayectoria
  propaga los eventos simulados por la celda, sus anillos y el país. La marginal de cada mes se estima
  con Rao-Blackwell (media de P(Y >= 1 | lambda simulada)), sin el ruido del indicador.
* La combinación promedia los logits con peso ``weight_logistic`` (0,5: el reto de ViEWS 2023/24 muestra
  que los promedios simples de modelos distintos son difíciles de mejorar; el peso no se ajustó a la
  validación para no contaminarla).
* Las muertes estatales: Bernoulli(P combinada de evento sb) x ``Size`` (log-normal discretizada).

Nada de lo posterior a T entra: las filas de entrenamiento para el horizonte h son las de meses
m <= T - h, y su inclusión depende sólo de la historia (``features.build``).
"""
from __future__ import annotations

import numpy as np

from world4.conflict import features as F
from world4.conflict.grid import TYPES, Events, Grid
from world4.conflict.models import Hawkes, Logistic, Size, _sigmoid

OUTCOMES = TYPES + ("any",)
HORIZON = 12
WEIGHT_LOGISTIC = 0.5
MC_WINDOW_MIN_HITS = 20


def _logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def window_hit(ev: Events, outcome: str, months, cells, H: int) -> np.ndarray:
    out = np.zeros(len(months), bool)
    for h in range(1, H + 1):
        out |= ev.lookup("count", outcome, months + h, cells) > 0
    return out


class Model:
    def __init__(self, origin: int, horizon: int = HORIZON, weight_logistic: float = WEIGHT_LOGISTIC):
        self.origin, self.horizon, self.weight_logistic = int(origin), int(horizon), float(weight_logistic)
        self.logistic: dict[str, list[Logistic]] = {}
        self.window: dict[str, Logistic] = {}
        self.size: list[Size] = []
        self.hawkes: dict[str, Hawkes] = {}
        self.info: dict = {}

    def to_dict(self) -> dict:
        return {"origin": self.origin, "horizon": self.horizon, "weight_logistic": self.weight_logistic,
                "features": list(F.FEATURES), "hawkes_columns": list(F.HAWKES), "background": list(F.BACKGROUND),
                "logistic": {o: [m.to_dict() for m in ms] for o, ms in self.logistic.items()},
                "window": {o: m.to_dict() for o, m in self.window.items()},
                "size": [m.to_dict() for m in self.size], "hawkes": {v: m.to_dict() for v, m in self.hawkes.items()},
                "info": self.info}

    @classmethod
    def from_dict(cls, d: dict) -> "Model":
        if tuple(d["features"]) != F.FEATURES or tuple(d["hawkes_columns"]) != F.HAWKES:
            raise ValueError("saved conflict-grid model was fitted with other features: refit it")
        m = cls(d["origin"], d["horizon"], d["weight_logistic"])
        m.logistic = {o: [Logistic.from_dict(x) for x in xs] for o, xs in d["logistic"].items()}
        m.window = {o: Logistic.from_dict(x) for o, x in d["window"].items()}
        m.size = [Size.from_dict(x) for x in d["size"]]
        m.hawkes = {v: Hawkes.from_dict(x) for v, x in d["hawkes"].items()}
        m.info = d.get("info", {})
        return m

    def branching(self) -> dict:
        """Hijos esperados de un evento de tipo u, por tipo de hijo v (sumando celda, anillos con su número de
        celdas y país): la razón de ramificación del proceso. < 1 en total: estacionario."""
        out = {}
        for v, hk in self.hawkes.items():
            a = hk.alpha
            row = {}
            for ui, u in enumerate(TYPES):
                s = 0.0
                for ri, nring in enumerate((1, 8, 16)):
                    for hi in range(len(F.HALF_LIVES)):
                        s += a[(ui * 3 + ri) * 3 + hi] * nring
                s += a[27 + ui]
                row[u] = float(s)
            out[v] = row
        return out


def fit(ev: Events, grid: Grid, rows: dict, origin: int, *, horizon: int = HORIZON, outcomes=OUTCOMES,
        log=None) -> Model:
    log = log or (lambda *a: None)
    mo, ce, w, X, Z = rows["month"], rows["cell"], rows["weight"], rows["X"], rows["Z"]
    model = Model(origin, horizon)
    for o in outcomes:
        model.logistic[o] = []
    for h in range(1, horizon + 1):
        sel = np.nonzero(mo + h <= origin)[0]
        Xs, ws = X[sel], w[sel]
        for o in outcomes:
            y = ev.lookup("count", o, mo[sel] + h, ce[sel]) > 0
            model.logistic[o].append(Logistic().fit(Xs, y.astype(float), ws))
        cnt = ev.lookup("count", "sb", mo[sel] + h, ce[sel])
        pos = cnt > 0
        deaths = ev.lookup("deaths", "sb", mo[sel] + h, ce[sel])
        model.size.append(Size().fit(Xs[pos], deaths[pos], ws[pos]))
        log(f"  logistic + size h={h}: {len(sel)} rows")
    sel = np.nonzero(mo + horizon <= origin)[0]
    for o in outcomes:
        y = window_hit(ev, o, mo[sel], ce[sel], horizon)
        model.window[o] = Logistic().fit(X[sel], y.astype(float), w[sel])
    sel = np.nonzero(mo + 1 <= origin)[0]
    B = np.column_stack([np.ones(len(sel)), grid.static[ce[sel]]])
    for v in TYPES:
        y = ev.lookup("count", v, mo[sel] + 1, ce[sel])
        model.hawkes[v] = Hawkes().fit(B, Z[sel], y, w[sel])
        log(f"  hawkes {v}: theta={model.hawkes[v].theta:.3f} converged={model.hawkes[v].converged}")
    model.info = {"training_rows": int(len(mo)), "rows_months": [int(mo.min()), int(mo.max())]}
    return model


def _alpha_parts(alpha):
    """alpha (30,) -> {ring: {(u, hl): a}} y país {u: a}."""
    parts = {r: {} for r in ("own", "r1", "r2")}
    for ui, u in enumerate(TYPES):
        for ri, r in enumerate(("own", "r1", "r2")):
            for hi, hl in enumerate(F.HALF_LIVES):
                parts[r][(u, hl)] = alpha[(ui * 3 + ri) * 3 + hi]
    return parts, {u: alpha[27 + ui] for ui, u in enumerate(TYPES)}


def excitation(grid: Grid, E: dict, alpha, split: bool = False):
    """Suma de excitaciones (n,) o (n, S) para un tipo de destino, sin formar la matriz (n, S, 30)."""
    parts, cty = _alpha_parts(alpha)
    comb = {}
    for r in ("own", "r1", "r2"):
        acc = 0.0
        for (u, hl), a in parts[r].items():
            if a > 0:
                acc = acc + (a * F.BETA[hl]) * E[u][hl]
        comb[r] = acc
    own = comb["own"]
    ring = 0.0
    if not np.isscalar(comb["r1"]):
        ring = ring + grid.ring1 @ comb["r1"]
    if not np.isscalar(comb["r2"]):
        ring = ring + grid.ring2 @ comb["r2"]
    cm = 0.0
    for u, a in cty.items():
        if a > 0:
            cm = cm + (a * F.BETA[F.COUNTRY_HL]) * E[u][F.COUNTRY_HL]
    country = 0.0
    if not np.isscalar(cm):
        c = (grid.country_matrix @ cm)[grid.country]
        cells = grid.cells_per_country[grid.country]
        country = c / (cells if np.ndim(c) == 1 else cells[:, None])
    if split:
        z = lambda x: x if not np.isscalar(x) else np.zeros(grid.n)  # noqa: E731
        return {"own": z(own), "neighbours": z(ring), "country": z(country)}
    return own + ring + country


def simulate_hawkes(model: Model, grid: Grid, snap: dict, *, samples: int = 200, seed: int = 0) -> dict:
    """Trayectorias de Hawkes desde el origen. Devuelve, por tipo y 'any': p (H, n) marginal (Rao-Blackwell),
    mean (H, n) eventos esperados, y p_window (n,)."""
    rng = np.random.default_rng(seed)
    H, S, n = model.horizon, samples, grid.n
    B = np.column_stack([np.ones(n), grid.static])
    bg = {v: model.hawkes[v].background(B).astype(np.float32) for v in TYPES}
    E = {u: {hl: np.repeat(snap["E"][u][hl].astype(np.float32)[:, None], S, axis=1) for hl in F.HALF_LIVES}
         for u in TYPES}
    p = {o: np.zeros((H, n)) for o in OUTCOMES}
    mean = {v: np.zeros((H, n)) for v in TYPES}
    for h in range(H):
        ys, q_all = {}, 1.0
        for v in TYPES:
            hk = model.hawkes[v]
            lam = bg[v][:, None] + excitation(grid, E, hk.alpha)
            q = (hk.theta / (hk.theta + lam)) ** hk.theta
            p[v][h] = 1.0 - q.mean(1)
            mean[v][h] = lam.mean(1)
            q_all = q_all * q
            ys[v] = hk.sample(lam, rng).astype(np.float32)
        p["any"][h] = 1.0 - q_all.mean(1)
        anyy = sum(ys[v] for v in TYPES) > 0
        for u in TYPES:
            for hl in F.HALF_LIVES:
                E[u][hl] = E[u][hl] * np.float32(0.5 ** (1.0 / hl)) + ys[u]
        if h == 0:
            ever = {v: ys[v] > 0 for v in TYPES}
            ever["any"] = anyy
        else:
            for v in TYPES:
                ever[v] |= ys[v] > 0
            ever["any"] |= anyy
    out = {"p": p, "mean": mean, "p_window": {}}
    for o in OUTCOMES:
        mc = ever[o].mean(1)
        indep = 1.0 - np.prod(1.0 - p[o], axis=0)
        enough = ever[o].sum(1) >= MC_WINDOW_MIN_HITS
        out["p_window"][o] = np.where(enough, mc, indep)
    return out


def predict(model: Model, grid: Grid, snap: dict, *, samples: int = 200, seed: int = 0) -> dict:
    """Pronóstico de todas las celdas desde el origen del modelo. Claves: ``p`` (combinada), ``p_logistic``,
    ``p_hawkes`` (dict resultado -> (H, n)), ``p_window`` y sus componentes (dict -> (n,)), ``events`` (Hawkes,
    eventos esperados por tipo (H, n)), ``deaths`` (muertes estatales: mean, p50, p90 (H, n))."""
    X = snap["X"]
    H = model.horizon
    pl = {o: np.stack([m.predict(X) for m in model.logistic[o]]) for o in model.logistic}
    plw = {o: model.window[o].predict(X) for o in model.window}
    sim = simulate_hawkes(model, grid, snap, samples=samples, seed=seed)
    wl = model.weight_logistic
    comb = lambda a, b: _sigmoid(wl * _logit(a) + (1 - wl) * _logit(b))  # noqa: E731
    p = {o: comb(pl[o], sim["p"][o]) for o in pl}
    pw = {o: comb(plw[o], sim["p_window"][o]) for o in plw}
    rng = np.random.default_rng(seed + 1)
    dm, d50, d90 = (np.zeros((H, grid.n)) for _ in range(3))
    for h in range(H):
        D = death_samples(model, X, p["sb"][h], h + 1, rng, samples)
        dm[h] = D.mean(1)
        d50[h], d90[h] = np.percentile(D, [50, 90], axis=1)
    return {"origin": model.origin, "horizon": H, "p": p, "p_logistic": pl, "p_hawkes": sim["p"],
            "p_window": pw, "p_window_logistic": plw, "p_window_hawkes": sim["p_window"],
            "events": sim["mean"], "deaths": {"mean": dm, "p50": d50, "p90": d90}}


def death_samples(model: Model, X, p_sb, h: int, rng, S: int) -> np.ndarray:
    """(n, S) muertes estatales en t+h: Bernoulli(p) x tamaño log-normal discretizado."""
    occur = rng.random((len(p_sb), S)) < p_sb[:, None]
    out = np.zeros((len(p_sb), S))
    idx = np.nonzero(occur.any(1))[0]
    if len(idx):
        size = np.maximum(model.size[h - 1].sample(X[idx], rng, S), 1.0)
        out[idx] = np.where(occur[idx], size, 0.0)
    return out


def decompose(model: Model, grid: Grid, snap: dict) -> dict:
    """Intensidad de Hawkes del mes siguiente separada en fondo estructural, contagio de la propia celda,
    de las celdas vecinas (anillos 1-2) y del país, por tipo: (n,) cada parte, en eventos esperados."""
    B = np.column_stack([np.ones(grid.n), grid.static])
    out = {}
    for v in TYPES:
        hk = model.hawkes[v]
        parts = excitation(grid, snap["E"], hk.alpha, split=True)
        out[v] = {"background": hk.background(B), **parts}
    return out
