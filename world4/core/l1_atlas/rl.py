"""Multi-agent reinforcement learning on the calibrated world-system model.

Every country is an agent. All agents share one policy network (parameter sharing),
but each one conditions on its own state, so heterogeneous strategies can emerge
(core vs periphery, stable vs unstable, cohesive vs fragmented...).

Decision: every 5 years each nation chooses how much to spend (budget between SIG_LO and SIG_HI
of GDP, the rest is consumed) and how to split it over the six channels [k, r, m, x, f, w]
(softmax of Gaussian logits). Spending above the historical path is financed partly by domestic
saving (less consumption) and partly by foreign borrowing up to a deficit ceiling (model.py).
Algorithm: PPO (clipped surrogate) + GAE, implemented with numpy/autograd.

Reward schemes (one policy is trained per scheme):
  bienestar : growth of log consumption per capita, minus a penalty for civil conflict
  poder     : growth of the country's log share of world GNI (relative power, zero-sum)
  elite     : growth of log income per member of the elite (captured state)
  irl       : the reward estimated from observed behaviour (src/irl.py, results/irl.json):
              theta . phi with phi = [welfare growth, power-share growth, elite-income growth,
              -conflict probability, -resource dependence, -adjustment cost]
"""
import json
import sys
import time
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad

from .model import CHANNELS, Params, Simulator, build_world
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
OUT = ROOT / "results"
SIGMA = 0.35                      # budget of the default mix
SIG_LO, SIG_HI = 0.10, 0.60       # range of the chosen budget (share of GDP)
STEP = 5
N_OBS = 22
N_CH = len(CHANNELS)
N_ACT = N_CH + 1                  # six channel logits + budget level
HIST_DEFAULT = np.array([0.20, 0.018, 0.075, 0.01, 0.01, 0.037]) / 0.35  # typical historical mix (model units)


# ----------------------------------------------------------------------------- network
def init_mlp(sizes, rng, out_scale=0.01):
    ps = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        s = out_scale if i == len(sizes) - 2 else np.sqrt(2.0 / a)
        ps.append([rng.standard_normal((a, b)) * s, np.zeros(b)])
    return ps


def mlp(ps, x):
    for W, b in ps[:-1]:
        x = anp.tanh(anp.dot(x, W) + b)
    W, b = ps[-1]
    return anp.dot(x, W) + b


class Adam:
    def __init__(self, lr=3e-4):
        self.lr, self.m, self.v, self.t = lr, None, None, 0

    def step(self, params, grads):
        flat_p, flat_g = _flatten(params), _flatten(grads)
        if self.m is None:
            self.m = [np.zeros_like(p) for p in flat_p]
            self.v = [np.zeros_like(p) for p in flat_p]
        self.t += 1
        gn = np.sqrt(sum((g ** 2).sum() for g in flat_g))
        clip = min(1.0, 0.5 / (gn + 1e-8))
        for i, (p, g) in enumerate(zip(flat_p, flat_g)):
            g = g * clip
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _flatten(tree):
    if isinstance(tree, (list, tuple, dict)):
        items = tree.values() if isinstance(tree, dict) else tree
        return [x for t in items for x in _flatten(t)]
    return [tree]


# ----------------------------------------------------------------------------- env helpers
def observe(sim, t):
    """Per-agent features (R, N, N_OBS)."""
    L = sim.w.pop
    last = sim.last
    R, N = sim.R, sim.w.N
    if not last:
        z = np.zeros((R, N))
        last = dict(ly=np.log(np.maximum(sim.w.Y[:, max(t, 0)], 1) / L[:, t])[None].repeat(R, 0),
                    core=z, psi=z + 1, rent_in=z, res_share=z, ims=z, res_out=z)
    ly = np.nan_to_num(last["ly"], nan=0.0)
    act = sim.active
    mean_ly = np.sum(ly * act, 1, keepdims=True) / np.maximum(act.sum(1, keepdims=True), 1)
    tprev = max(t - STEP, 0)
    popg = np.log(L[:, t] / L[:, tprev]) / max(t - tprev, 1)
    Y = np.exp(ly) * L[:, t]
    _, _, _, res_sd, _ = sim._extraction()
    feats = [
        ly - mean_ly,
        np.nan_to_num(last["core"]),
        sim.E,
        np.log(np.maximum(sim.ne, 1e-3)),
        np.log(np.maximum(np.nan_to_num(last["psi"], nan=1.0), 1e-3)),
        sim.asab,
        sim.conf,
        np.nan_to_num(last["rent_in"]) * 10,
        np.log(np.maximum(sim.K, 1e-6) / np.maximum(Y, 1e-6)) - 1.0,
        np.minimum(sim.claims / np.maximum(Y, 1e-6), 5),
        # natural resources: what is left, who extracts it, how dependent on imports
        1 - np.clip(sim.X / np.maximum(sim.U, 1e-12), 0, 1),
        1 - res_sd,
        np.nan_to_num(last["res_share"]) * 10,
        np.nan_to_num(last["ims"]),
        np.nan_to_num(last["res_out"]) * 10,
        np.broadcast_to(np.log(np.maximum(sim.price, 1e-6)), (R, N)),
        np.clip(sim.debt, 0, 3),
        sim.poly,
        np.broadcast_to(popg * 30, (R, N)),
        np.broadcast_to(sim.w.open_[:, t], (R, N)),
        np.full((R, N), (t - 35) / 35),
        np.ones((R, N)),
    ]
    return np.clip(np.stack(feats, -1), -10, 10).astype(np.float64)


def to_shares(z):
    """Channel shares from the first six logits."""
    z = z[..., :N_CH]
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def to_budget(z):
    """Budget (share of GDP) from the last logit."""
    return SIG_LO + (SIG_HI - SIG_LO) / (1 + np.exp(-z[..., N_CH]))


def rollout(sim, pol, logstd, rng, mode, deterministic=False, record=False, t0=0):
    """Play one episode (1950-2019) in R parallel worlds. Returns transitions."""
    w = sim.w
    sim.reset(t0)
    rollout.a_prev = None
    T = w.T
    steps = list(range(t0, T, STEP))
    R, N = sim.R, w.N
    buf = dict(obs=[], z=[], mask=[], rew=[])
    rec = {k: np.full((R, N, T - t0), np.nan) for k in sim.REC} if record else None
    if record:
        rec["price"] = np.full((R, T - t0), np.nan)
    shares = np.broadcast_to(HIST_DEFAULT, (R, N, N_CH)).copy()
    budget = np.full((R, N), SIGMA)
    for s, t_dec in enumerate(steps):
        obs = observe(sim, t_dec)
        mean = mlp(pol, obs.reshape(-1, N_OBS)).reshape(R, N, N_ACT)
        z = mean if deterministic else mean + np.exp(logstd) * rng.standard_normal(mean.shape)
        mask = sim.active.copy()
        shares = np.where(mask[..., None], to_shares(z), shares)
        budget = np.where(mask, to_budget(z), budget)
        start = _metrics(sim, mode) if mode != "irl" else None
        conf_years = np.zeros((R, N))
        r_irl = np.zeros((R, N))
        for t in range(t_dec, min(t_dec + STEP, T)):
            newly = ~mask & sim.active   # entrants inside the block use the default mix
            if newly.any():
                shares = np.where(newly[..., None], HIST_DEFAULT, shares)
                budget = np.where(newly, SIGMA, budget)
            acts = {c: budget * shares[..., j] for j, c in enumerate(CHANNELS)}
            prev = _irl_state(sim) if mode == "irl" else None
            out = sim.step(acts)
            conf_years += np.nan_to_num(out["conflict"])
            if mode == "irl":
                cur = _irl_state(sim)
                a_now = np.stack([acts[c] for c in CHANNELS], -1)
                adj = (np.sum((a_now - rollout.a_prev) ** 2, -1) * 100) if getattr(rollout, "a_prev", None) is not None and rollout.a_prev.shape == a_now.shape else 0.0
                rollout.a_prev = a_now
                if prev is not None:
                    f = [cur[0] - prev[0], cur[1] - prev[1], cur[2] - prev[2],
                         -np.nan_to_num(sim.hazard), -np.nan_to_num(out["ims"]), -adj]
                    r_irl += np.nan_to_num(sum(th * fk for th, fk in zip(IRL_THETA, f)))
            if record:
                for k in sim.REC:
                    rec[k][:, :, t - t0] = out[k]
                rec["price"][:, t - t0] = sim.last_price
        if mode == "irl":
            r = r_irl
        else:
            end = _metrics(sim, mode)
            r = (end - start) * 10.0
            if mode == "bienestar":
                r = r - 0.3 * conf_years
        buf["obs"].append(obs); buf["z"].append(z); buf["mask"].append(mask)
        buf["rew"].append(np.where(mask, np.nan_to_num(r), 0.0))
    for k in buf:
        buf[k] = np.stack(buf[k], 2)  # (R, N, S, ...)
    return buf, rec


IRL_THETA = None


def load_irl_theta():
    """theta in feature units (theta_std / sd) from results/irl.json."""
    global IRL_THETA
    e = json.load(open(OUT / "irl.json"))
    src = e.get("parsimonioso", e)          # parsimonious reward avoids collinear weights ...
    if sum(abs(v) > 0 for v in src["theta_std"].values()) < 2:
        src = e                              # ... unless it keeps only inertia (nothing to optimise)
    th = np.array(list(src["theta_std"].values())) / np.array(e["sd"])
    IRL_THETA = th / np.abs(th).max()
    return IRL_THETA


def _irl_state(sim):
    """(log consumption pc, log share of world GNI, log income per elite member) or None."""
    last = sim.last
    if not last:
        return None
    L = sim.w.pop[:, sim.t - 1]
    g = last["gni_pc"] * L
    share = g / np.nansum(g, 1, keepdims=True)
    elite = sim.E * last["gni_pc"] / np.maximum(sim.ne, 1e-3)
    lg = lambda x: np.log(np.maximum(x, 1e-12))
    return lg(last["cons_pc"]), lg(share), lg(elite)


def _metrics(sim, mode):
    """Per-agent scalar whose growth is rewarded; NaN where the agent has no history yet."""
    last = sim.last
    if not last:
        return np.full((sim.R, sim.w.N), np.nan)
    if mode == "bienestar":
        return np.log(last["cons_pc"])
    L = sim.w.pop[:, sim.t - 1]
    g = last["gni_pc"] * L
    if mode == "poder":
        return np.log(g / np.nansum(g, 1, keepdims=True))
    if mode == "elite":
        return np.log(sim.E * g / (np.maximum(sim.ne, 1e-3) * L))
    raise ValueError(mode)


def gae(rew, val, mask, gamma=0.95, lam=0.9):
    R, N, S = rew.shape
    adv = np.zeros_like(rew)
    last = np.zeros((R, N))
    for s in reversed(range(S)):
        nv = val[:, :, s + 1] if s + 1 < S else 0.0
        nm = mask[:, :, s + 1] if s + 1 < S else np.zeros((R, N), bool)
        delta = rew[:, :, s] + gamma * nv * nm - val[:, :, s]
        last = delta + gamma * lam * nm * last
        adv[:, :, s] = last
    return adv, adv + val


def train(mode, params, iters=250, R=12, seed=0, log=print):
    rng = np.random.default_rng(seed)
    w = build_world()
    sim = Simulator(w, params, R=R, seed=seed)
    pol = init_mlp([N_OBS, 64, 64, N_ACT], rng)
    # bias the initial policy toward the historical mix so exploration starts from plausible behaviour
    pol[-1][1] = np.append(np.log(HIST_DEFAULT), 0.0)     # budget logit 0 -> SIGMA
    vf = init_mlp([N_OBS, 64, 64, 1], rng, out_scale=0.1)
    logstd = np.full(N_ACT, -0.5)
    th = {"pol": pol, "logstd": logstd}
    opt_p, opt_v = Adam(3e-4), Adam(1e-3)
    hist = []

    def logp(th, obs, z):
        mu = mlp(th["pol"], obs)
        ls = th["logstd"]
        return anp.sum(-0.5 * ((z - mu) / anp.exp(ls)) ** 2 - ls, axis=1)

    def pol_loss(th, obs, z, adv, oldlp):
        ratio = anp.exp(logp(th, obs, z) - oldlp)
        s1, s2 = ratio * adv, anp.clip(ratio, 0.8, 1.2) * adv
        ent = anp.sum(th["logstd"])
        return -anp.mean(anp.minimum(s1, s2)) - 0.003 * ent

    def v_loss(vf, obs, ret):
        return anp.mean((mlp(vf, obs)[:, 0] - ret) ** 2)

    gp, gv = grad(pol_loss), grad(v_loss)
    t_start = time.time()
    for it in range(iters):
        buf, _ = rollout(sim, th["pol"], th["logstd"], rng, mode)
        m = buf["mask"]
        obs = buf["obs"]
        val = mlp(vf, obs.reshape(-1, N_OBS)).reshape(m.shape)
        adv, ret = gae(buf["rew"], val, m)
        sel = m.reshape(-1)
        O, Z = obs.reshape(-1, N_OBS)[sel], buf["z"].reshape(-1, N_ACT)[sel]
        A, RET = adv.reshape(-1)[sel], ret.reshape(-1)[sel]
        A = (A - A.mean()) / (A.std() + 1e-8)
        old = logp(th, O, Z)
        n = len(A)
        for ep in range(4):
            perm = rng.permutation(n)
            for b in range(0, n, 2048):
                ix = perm[b:b + 2048]
                g = gp(th, O[ix], Z[ix], A[ix], old[ix])
                opt_p.step(th, g)
                th["logstd"] = np.clip(th["logstd"], -2.5, 0.5)
                opt_v.step(vf, gv(vf, O[ix], RET[ix]))
        ep_ret = float(buf["rew"].sum(2)[m[:, :, 0] | m.any(2)].mean())
        hist.append(ep_ret)
        if it % 10 == 0 or it == iters - 1:
            out = mlp(th["pol"], O)
            log(f"[{mode}] it {it:3d} ret {ep_ret:+.3f} std {np.exp(th['logstd']).mean():.2f} "
                f"budget {to_budget(out).mean():.3f} shares {np.round(to_shares(out).mean(0), 3)} "
                f"({time.time() - t_start:.0f}s)")
    return th, vf, hist


def main():
    OUT.mkdir(exist_ok=True)
    cal = json.load(open(OUT / "calibration.json"))
    params = Params().with_vector(list(cal["params"]), list(cal["params"].values()))
    modes = sys.argv[1:] or ["bienestar", "poder", "elite", "irl"]
    for mode in modes:
        if mode == "irl":
            load_irl_theta()
        th, vf, hist = train(mode, params)
        np.save(OUT / f"policy_{mode}.npy",
                np.array({"pol": th["pol"], "logstd": th["logstd"], "curve": hist}, dtype=object),
                allow_pickle=True)


if __name__ == "__main__":
    main()
