"""El MDP del aprendizaje por refuerzo: el L1 (Atlas + F1-F3) en lazo cerrado, 2019 -> 2030.

* **Agente:** un planificador que, en 2020, 2022, 2024, 2026 y 2028 (5 decisiones), observa el estado del
  mundo y fija el desvío del logit de los 6 canales en cada zona del sistema-mundo (periferia,
  semiperiferia, centro): acción continua de 18 dimensiones en [-0,5, 0,5] (la región de confianza del IRL).
  La acción se mantiene hasta la siguiente decisión. Los países siguen su regla revelada más el desvío.
* **Estado observado (36):** por zona y ponderado por población 2019, el log del PIB per cápita, el
  crecimiento del consumo per cápita del último año, el riesgo de conflicto, la deuda y la intensidad
  fósil del PIB; el tiempo, la renta de recursos mundial (psi) y la deuda mundial; y la media de las
  acciones pasadas (18), porque la puntuación final depende de la trayectoria y no sólo del estado.
* **Episodios en paralelo:** cada corrida del L1 tiene R miembros con su propio azar; cada miembro es un
  episodio independiente (el agente actúa sobre cada uno según su estado).
* **Recompensa:** terminal (2030), la misma puntuación compuesta de tres pilares del paso 3 (``s3_learning``),
  calculada por miembro frente al mismo miembro de la línea base (números aleatorios comunes); los términos
  de World3 salen del emulador (``w3_surrogate``). Más una recompensa intermedia pequeña (potencial):
  la variación del índice de desarrollo mundial frente a la base entre decisiones.
"""
from __future__ import annotations

import numpy as np

import common as c
import s2_strategies as s2

DECISIONS = (2020, 2022, 2024, 2026, 2028)
ACT_DIM, OBS_DIM = 18, 36
ZONES = ("periferia", "semiperiferia", "centro")
SHAPING = 0.5


class _Ctx:
    hook = None


def _patched_policy():
    from world4.core.l1_atlas import behavior

    class Hooked(behavior.BehaviorPolicy):
        def __call__(self, sim, t):
            if _Ctx.hook is not None:
                _Ctx.hook(sim, t, self.offsets)
            return super().__call__(sim, t)
    return behavior, Hooked


class Env:
    def __init__(self, R=48):
        from w3_surrogate import Surrogate
        import s3_learning as s3
        self.R = R
        self.W, self.t0 = c.live()
        self.zone = s3.zones()
        self.N = self.W.N
        self.pop0 = np.nan_to_num(np.asarray(c.load("strategies.json")["pop_2030"], float))
        self.sd = s3.scales()
        self.surr = Surrogate()
        self.th = s2.theta_matrix()
        self._bases = {}
        self._score = s3.score

    # ------------------------------------------------------------ simulación
    def _run(self, hook, seed, offsets=None):
        from world4.core import coupling
        behavior, Hooked = _patched_policy()
        orig = behavior.BehaviorPolicy
        offs = offsets or {ch: np.zeros((self.R, self.N)) for ch in c.CHANNELS}
        behavior.BehaviorPolicy = Hooked
        _Ctx.hook = hook
        try:
            return coupling.forecast(self.t0, R=self.R, seed=seed, world=self.W, horizon=c.HORIZON,
                                     f3=coupling.F3Config(), offsets=offs, raw=True)
        finally:
            behavior.BehaviorPolicy = orig
            _Ctx.hook = None

    def observe(self, sim, t):
        last = sim.last
        R, N = self.R, self.N
        ly = np.nan_to_num(last["ly"], nan=np.nan)
        cons = last["cons_pc"]
        prev = getattr(self, "_prev_cons", None)
        g = np.log(cons / prev) if prev is not None else np.zeros((R, N))
        self._prev_cons = cons
        haz = np.asarray(sim.hazard, float)
        debt = last["debt"]
        fint = np.zeros((R, N))
        if sim.en is not None and (t - 1) in sim.en.hist:
            Y = np.exp(ly) * np.maximum(self.pop0, 1e-6)[None]
            fint = sim.en.hist[t - 1]["gen_fossil"] / np.maximum(Y, 1e-9) * 1e3
        feats = []
        for z in range(3):
            w = np.where((self.zone == z) & np.isfinite(ly).all(0), self.pop0, 0.0)
            wz = w / w.sum()
            for x in (ly, g, haz, debt, fint):
                feats.append(np.nansum(np.nan_to_num(x) * wz[None], 1))
        psi = np.asarray(last.get("psi", np.zeros(R)), float).reshape(R, -1)[:, 0]
        Yw = np.exp(np.nan_to_num(ly)) * self.pop0[None]
        dw = np.nansum(np.nan_to_num(debt) * Yw, 1) / Yw.sum(1)
        feats += [np.full(R, (1950 + t - 2020) / 10.0), psi * 10, dw]
        return np.stack(feats, 1)                   # (R, 18)

    def apply(self, offsets, a):
        """a (R, 18) -> desvío por país (R, N) por canal, in place."""
        a = np.clip(a, -0.5, 0.5)
        for j, ch in enumerate(c.CHANNELS):
            offsets[ch][:] = a[:, self.zone * 6 + j]

    def rollout(self, policy, seed, keep_f=False):
        """``policy(obs (R, 18), step) -> acciones (R, 18)``. Devuelve obs (5, R, 18), act (5, R, 18),
        obs final (R, 18), recompensas (5, R), puntuación terminal (R,) y diagnósticos."""
        dec_t = {y - 1950: k for k, y in enumerate(DECISIONS)}
        O, A = [None] * len(DECISIONS), [None] * len(DECISIONS)
        self._prev_cons = None
        last = {}
        acc = np.zeros((self.R, ACT_DIM))

        def hook(sim, t, offsets):
            if not sim.last:
                return
            if t in dec_t or t == self.t0 + c.HORIZON:
                n = dec_t.get(t, len(DECISIONS))
                o = np.concatenate([self.observe(sim, t), acc / max(n, 1)], 1)
            if t in dec_t:
                k = dec_t[t]
                a = np.asarray(policy(o, k), float)
                O[k], A[k] = o, np.clip(a, -0.5, 0.5)
                acc[:] += A[k]
                self.apply(offsets, a)
            if t == self.t0 + c.HORIZON:
                last["obs"] = o

        f = self._run(hook, seed)
        rew, score, info = self.reward(f, seed)
        obs_T = last.get("obs", O[-1])
        out = {"obs": np.stack(O), "act": np.stack(A), "obs_T": obs_T, "rew": rew, "score": score, "info": info}
        if keep_f:
            out["f"] = f
        return out

    # ------------------------------------------------------------ recompensa
    def base(self, seed):
        if seed not in self._bases:
            fb = self._run(None, seed)
            mb, sb = c.country_metrics(fb), c.world_signals(fb)
            mb = {k: v for k, v in mb.items() if v is not None}
            self._bases[seed] = {"m": mb, "sig": sb, "id_path": self._id_path(fb)}   # sin la simulación: memoria
        return self._bases[seed]

    def _id_path(self, f):
        """Índice de desarrollo mundial ponderado (R, H+1) para la recompensa intermedia."""
        ID = s2.dev_index(np.exp(f["ly"]), f["hc"], f["hazard"])
        w = self.pop0[None, :, None] * np.isfinite(ID)
        return np.nansum(np.nan_to_num(ID) * w, 1) / w.sum(1)

    def member_metrics(self, f, b):
        """metric_vector de s3 por miembro (R, 8): Δ frente al mismo miembro de la base."""
        m, sig = c.country_metrics(f), c.world_signals(f)
        mb, sb = b["m"], b["sig"]
        pop = self.pop0
        ID, IDb = s2.dev_index(m["gdp_pc"], m["hc"], m["hazard"]), s2.dev_index(mb["gdp_pc"], mb["hc"], mb["hazard"])
        wsum = lambda x: np.nansum(np.nan_to_num(x) * pop[None], 1) / np.nansum(np.isfinite(x) * pop[None], 1)  # noqa: E731
        q1 = np.nan_to_num(mb["gdp_pc0"]) <= np.nanquantile(mb["gdp_pc0"][0], 0.2)
        q1m = lambda x: np.nansum(np.where(q1, np.nan_to_num(x), 0), 1) / q1.sum(1)  # noqa: E731
        from world4.core.l1_atlas import irl as airl
        wel = np.array(airl.HAND["bienestar"], float)
        dv = np.nan_to_num((m["val"] - mb["val"]) @ wel)
        welfare = (dv * pop[None]).sum(1) / pop.sum()
        fossil = sig["fossil_cum"] / sb["fossil_cum"] - 1
        # World3: señales de cada miembro frente a la base media -> emulador
        R = self.R
        maps = []
        for r in range(R):
            mp = c.world3_constants({k: v[r:r + 1] for k, v in sig.items()}, sb)
            maps.append([mp[k] for k in ("fioac2_factor", "resource_tech_rate", "pollution_tech_rate", "services_factor",
                                         "icor_factor", "family_factor")])
        maps = np.array(maps)
        zero = np.array([[1.0, 0, 0, 1.0, 1.0, 1.0]])
        y, y0 = self.surr(maps), self.surr(zero)[0]
        return np.stack([wsum(ID) - wsum(IDb), q1m(ID) - q1m(IDb), welfare, fossil,
                         y[:, 2] / y0[2] - 1, y[:, 0] - y0[0], y[:, 1] - y0[1], y[:, 3] - y0[3]], 1), maps

    def reward(self, f, seed):
        b = self.base(seed)
        mv, maps = self.member_metrics(f, b)
        score = np.array([self._score(v, self.sd)[0] for v in mv])
        # recompensa intermedia: ΔID mundial frente a la base entre decisiones (potencial; suma telescópica)
        gap = (self._id_path(f) - b["id_path"]) / self.sd[0]                    # (R, H+1)
        idx = [y - c.ORIGIN for y in DECISIONS] + [c.HORIZON]
        inter = SHAPING * np.diff(gap[:, idx], axis=1).T / 3                     # (5, R)
        rew = inter.copy()
        rew[-1] += score - SHAPING * (gap[:, idx[-1]] - gap[:, idx[0]]) / 3      # el total suma la puntuación
        return rew, score, {"metrics": mv, "maps": maps}
