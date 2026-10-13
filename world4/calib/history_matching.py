"""*History matching* por oleadas (Vernon, Goldstein y Bower 2010; Andrianakis et al. 2015).

Para cada salida j (un resumen de la corrida comparable con un dato z_j):

    I_j(θ) = |z_j - E[f_j(θ)]| / sqrt(Var_em,j(θ) + Var_obs,j + Var_disc,j + Var_stoch,j)

y θ es implausible si la SEGUNDA mayor I_j supera 3 (el criterio de Vernon et al. con muchas salidas;
la regla de las 3 sigmas de Pukelsheim garantiza >= 95 % de masa dentro para cualquier unimodal).

Cada oleada:

1. siembra corridas dentro de la región todavía no descartada (NROY): hipercubo latino en la caja de la
   previa filtrado por las oleadas anteriores y, cuando la NROY se hace pequeña, perturbaciones de sus
   puntos (Andrianakis et al. 2015);
2. corre el simulador y ajusta un emulador GP por salida (``gp.Emulator``);
3. descarta las salidas que el emulador no reproduce (LOO: más del 10 % de |z| > 3) en esa oleada;
4. estima el volumen de la NROY con 1e5 puntos uniformes de la caja.

Se para cuando la NROY deja de encogerse (< 10 % entre oleadas) o cuando la varianza del emulador ya es
pequeña frente a las otras (mediana < 10 %): a partir de ahí el emulador no es lo que limita.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from world4.calib.gp import Emulator

CUTOFF = 3.0
KTH = 2                     # segunda mayor implausibilidad
LOO_BAD = 0.10              # una salida se deja fuera de la oleada si > 10 % de sus |z| LOO pasan de 3
VOLUME_POINTS = 100_000


@dataclass
class Problem:
    """Qué se calibra: ``simulate(theta (n, d)) -> (y (n, q), var_stoch (n, q) | None)``."""
    names: tuple
    box: np.ndarray                     # (d, 2)
    simulate: callable
    z: np.ndarray                       # (q,)
    var_obs: np.ndarray                 # (q,)
    var_disc: np.ndarray                # (q,)
    labels: tuple = ()
    meta: dict = field(default_factory=dict)

    @property
    def d(self):
        return len(self.names)


def lhs(n: int, d: int, rng) -> np.ndarray:
    """Hipercubo latino en [0, 1]^d."""
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for j in range(d):
        u[:, j] = u[rng.permutation(n), j]
    return u


def implausibility(mu, var_em, z, var_other):
    return np.abs(z[None] - mu) / np.sqrt(var_em + var_other[None])


def kth_max(I, k: int = KTH):
    if I.shape[1] < k:
        return I.max(1)
    return -np.partition(-I, k - 1, axis=1)[:, k - 1]


@dataclass
class Wave:
    emulator: Emulator
    active: np.ndarray                  # salidas usadas en esta oleada
    theta: np.ndarray
    y: np.ndarray
    var_stoch: np.ndarray | None
    nroy_fraction: float
    diagnostics: dict
    seconds: float


class HistoryMatch:
    def __init__(self, problem: Problem, seed: int = 0, cutoff: float = CUTOFF, kth: int = KTH, log=print):
        self.pb, self.cutoff, self.kth, self.log = problem, cutoff, kth, log
        self.rng = np.random.default_rng(seed)
        self.waves: list[Wave] = []
        self.var_other = problem.var_obs + problem.var_disc

    # ---------------------------------------------------------------- NROY
    def _from_unit(self, u):
        lo, hi = self.pb.box[:, 0], self.pb.box[:, 1]
        return lo + u * (hi - lo)

    def max_implausibility(self, theta, waves=None, chunk: int = 20_000) -> np.ndarray:
        """k-ésima mayor implausibilidad de cada θ en cada oleada (n, n_waves)."""
        waves = self.waves if waves is None else waves
        theta = np.atleast_2d(theta)
        out = np.zeros((len(theta), len(waves)))
        for i in range(0, len(theta), chunk):
            th = theta[i:i + chunk]
            for w, wv in enumerate(waves):
                mu, var = wv.emulator.predict(th)
                a = wv.active
                I = implausibility(mu, var, self.pb.z[a], self.var_other[a] + self._stoch_floor(wv)[a])
                out[i:i + chunk, w] = kth_max(I, self.kth)
        return out

    def _stoch_floor(self, wv):
        """Varianza estocástica típica de una corrida (mediana de la oleada): una corrida nueva la tiene."""
        if wv.var_stoch is None:
            return np.zeros(len(self.pb.z))
        return np.median(wv.var_stoch, 0)

    def nroy(self, theta) -> np.ndarray:
        if not self.waves:
            return np.ones(len(np.atleast_2d(theta)), bool)
        return np.all(self.max_implausibility(theta) <= self.cutoff, 1)

    def sample(self, n: int, pool: int = 50_000, max_rounds: int = 30) -> np.ndarray:
        """n puntos de la NROY actual (uniformes en la caja, y perturbaciones si la NROY es pequeña)."""
        d = self.pb.d
        kept = []
        for r in range(max_rounds):
            th = self._from_unit(lhs(pool, d, self.rng))
            if r > 0 and kept:
                seeds = np.concatenate(kept)
                cov = np.cov(seeds.T) if len(seeds) > d else np.diag(((self.pb.box[:, 1] - self.pb.box[:, 0]) / 20) ** 2)
                cov = np.atleast_2d(cov) * 0.25 + 1e-12 * np.eye(d)
                pick = seeds[self.rng.integers(len(seeds), size=pool)]
                th = np.concatenate([th[: pool // 2], (pick + self.rng.multivariate_normal(np.zeros(d), cov, pool))[: pool // 2]])
                inbox = np.all((th >= self.pb.box[:, 0]) & (th <= self.pb.box[:, 1]), 1)
                th = th[inbox]
            ok = self.nroy(th)
            if ok.any():
                kept.append(th[ok])
            if sum(len(k) for k in kept) >= 3 * n:
                break
        allk = np.concatenate(kept) if kept else np.empty((0, d))
        if len(allk) < n:
            raise RuntimeError(f"NROY too small to seed {n} runs ({len(allk)} points found): the model cannot "
                               "match the data within the declared uncertainties")
        return allk[self.rng.choice(len(allk), n, replace=False)]

    def volume(self, n: int = VOLUME_POINTS) -> float:
        th = self._from_unit(self.rng.random((n, self.pb.d)))
        return float(self.nroy(th).mean())

    # ---------------------------------------------------------------- oleadas
    def wave(self, n_runs: int) -> Wave:
        t = time.time()
        theta = self._from_unit(lhs(n_runs, self.pb.d, self.rng)) if not self.waves else self.sample(n_runs)
        y, vs = self.pb.simulate(theta)
        y = np.asarray(y, float)
        good = np.all(np.isfinite(y), 1)
        theta, y = theta[good], y[good]
        vs = None if vs is None else np.asarray(vs, float)[good]
        em = Emulator(self.pb.box, seed=len(self.waves)).fit(theta, y, vs)
        z = em.loo_z()
        bad = np.mean(np.abs(z) > 3, 0) > LOO_BAD
        active = np.where(~bad)[0]
        # implausibilidad de las corridas con el SIMULADOR (sin varianza de emulador)
        Isim = kth_max(implausibility(y, np.zeros_like(y) if vs is None else vs, self.pb.z, self.var_other), self.kth)
        wv = Wave(em, active, theta, y, vs, float("nan"), {}, 0.0)
        self.waves.append(wv)
        wv.nroy_fraction = self.volume()
        _, var_em = em.predict(theta)
        ratio = np.median(var_em[:, active] / self.var_other[active][None]) if len(active) else float("nan")
        wv.diagnostics = {
            **em.diagnostics(), "outputs_dropped": [self.pb.labels[j] if self.pb.labels else int(j) for j in np.where(bad)[0]],
            "runs": int(len(theta)), "runs_failed": int((~good).sum()),
            "runs_non_implausible_by_simulator": float(np.mean(Isim <= self.cutoff)),
            "median_var_em_over_var_other": float(ratio)}
        wv.seconds = time.time() - t
        self.log(f"  wave {len(self.waves)}: {len(theta)} runs, {len(active)}/{len(self.pb.z)} outputs, NROY "
                 f"{wv.nroy_fraction:.2e} of the prior box; runs non-implausible (simulator) "
                 f"{wv.diagnostics['runs_non_implausible_by_simulator']:.0%}; var_em/var_other {ratio:.2f}; "
                 f"LOO |z|>3 {wv.diagnostics['loo_abs_z_gt3_share']:.1%} ({wv.seconds:.0f} s)")
        return wv

    def run(self, n_runs: int, max_waves: int = 5, min_shrink: float = 0.10, var_ratio_stop: float = 0.10) -> dict:
        reason = f"{max_waves} waves"
        for k in range(max_waves):
            prev = self.waves[-1].nroy_fraction if self.waves else 1.0
            try:
                wv = self.wave(n_runs)
            except RuntimeError as e:
                reason = str(e)
                break
            if wv.nroy_fraction == 0.0:
                reason = "empty NROY: no parameter in the box matches the data within the declared uncertainties"
                break
            if k > 0 and wv.nroy_fraction > (1 - min_shrink) * prev:
                reason = f"NROY shrank < {min_shrink:.0%} in the last wave"
                break
            if wv.diagnostics["median_var_em_over_var_other"] < var_ratio_stop:
                reason = "emulator variance < 10 % of the other variances: the emulator no longer limits"
                break
        return self.report(reason)

    def report(self, reason: str = "") -> dict:
        return {"waves": [{"nroy_fraction": w.nroy_fraction, "outputs_active": int(len(w.active)), **w.diagnostics,
                           "seconds": w.seconds} for w in self.waves],
                "stop": reason, "cutoff": self.cutoff, "kth_max": self.kth, "parameters": list(self.pb.names),
                "box": self.pb.box.tolist(), "outputs": list(self.pb.labels)}

    def runs(self):
        """Todas las corridas de todas las oleadas (θ, y)."""
        return (np.concatenate([w.theta for w in self.waves]), np.concatenate([w.y for w in self.waves]))


# ------------------------------------------------------------------ la caja no debe decidir
EDGE_RULE = 0.05            # p10 o p90 del posterior a < 5 % del ancho de un borde
MAX_WIDEN = 2


def edge_knobs(names, theta, box, rule: float = EDGE_RULE) -> list:
    """Perillas cuyo posterior toca un borde de la caja (un borde inferior en 0 es físico y no cuenta)."""
    out = []
    for j, n in enumerate(names):
        p10, p90 = np.percentile(theta[:, j], [10, 90])
        w = box[j, 1] - box[j, 0]
        if (p10 - box[j, 0] < rule * w and box[j, 0] != 0.0) or box[j, 1] - p90 < rule * w:
            out.append(n)
    return out


def widen(names, box, which) -> np.ndarray:
    """Dobla el ancho de la caja de ``which`` alrededor de su centro (con un borde inferior físico en 0, sólo
    sube el superior)."""
    box = np.array(box, float)
    for j, n in enumerate(names):
        if n in which:
            lo, hi = box[j]
            w = hi - lo
            box[j] = (0.0, hi + w) if lo == 0.0 else (lo - w / 2, hi + w / 2)
    return box
