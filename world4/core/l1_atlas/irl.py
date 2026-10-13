"""Inverse reinforcement learning: which reward makes the observed behaviour look optimal?

Estimator (Bajari, Benkard & Levin 2007, "Estimating Dynamic Models of Imperfect Competition"):
  1. The policy countries actually follow is estimated from data (behavioral cloning,
     src/behavior.py).
  2. Starting from the observed world in several years, the simulator is run forward with every
     country following that rule, and each country's discounted sums of reward features are
     recorded: W_i = sum_h gamma^h phi_{i,t0+h}.
  3. Random subsets of countries deviate from the rule (one channel pushed up or down) with
     everything else identical, including the random numbers. If behaviour is optimal for the
     reward r = theta . phi, then theta . (W_i(rule) - W_i(deviation)) >= 0 for every deviation.
  4. theta is the direction on the unit sphere that best satisfies those inequalities (smoothed
     hinge / logistic loss). The share of satisfied inequalities is a "rationality index".

Reward features phi (per country-year):
  bienestar  growth of log consumption per capita
  poder      growth of the log share of world GNI
  elite      growth of log income per member of the elite
  paz        minus the probability of civil conflict
  autonomia  minus dependence on resources the country does not control
  inercia    minus the squared change of the allocation (adjustment cost; observed budgets are
             very persistent, rho ~ 0.9, and without this term inertia would be read as taste)
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from .behavior import BehaviorModel, BehaviorPolicy
from .model import CHANNELS, Params, Simulator
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RES = ROOT / "results"
FEATS = ["bienestar", "poder", "elite", "paz", "autonomia", "inercia"]
HAND = {  # the hand-coded rewards used in rl.py, as directions in feature space
    "bienestar": [1, 0, 0, 0.3, 0, 0], "poder": [0, 1, 0, 0, 0, 0], "elite": [0, 0, 1, 0, 0, 0]}


def reward_features(o, W, t0):
    """(R, N, H) array per feature from a Simulator.run record starting at t0."""
    H = o["ly"].shape[2]
    L = W.pop[:, t0:t0 + H][None]
    gni = o["gni_pc"] * L
    share = gni / np.nansum(gni, 1, keepdims=True)
    elite = o["E"] * o["gni_pc"] / np.maximum(o["ne"], 1e-3)
    d = lambda x: np.diff(np.log(np.maximum(x, 1e-12)), axis=2)
    # expected conflict (hazard) instead of realised events: smooth under common random numbers
    acts = np.stack([o[c] for c in CHANNELS], -1)                     # (R, N, H, 6)
    adj = -np.nansum(np.diff(acts, axis=2) ** 2, -1) * 100
    phi = np.stack([d(o["cons_pc"]), d(share), d(elite), -np.nan_to_num(o["hazard"][:, :, 1:]),
                    -o["ims"][:, :, 1:], adj], -1)
    return phi  # (R, N, H-1, F)


def discounted(phi, gamma):
    h = phi.shape[2]
    g = gamma ** np.arange(h)
    return np.nansum(phi * g[None, None, :, None], axis=2)  # (R, N, F)


def simulate_values(W, params, bm, t0, H, offsets, R, seed, gamma):
    # start one year early so the first deviating year's adjustment cost is measured
    sim = Simulator(W, params, R=R, seed=seed)
    pol = BehaviorPolicy(bm, params, noise=1.0, offsets=offsets, seed=seed + 1, offset_from=t0)
    o = sim.run(t0 - 1, min(t0 + H + 1, W.T), policy=pol)
    return discounted(reward_features(o, W, t0 - 1), gamma).mean(0), sim.active[0]


def collect_deviations(W, LY, params, starts=(10, 20, 30, 40, 50), H=15, draws=200, R=8,
                       frac=0.15, delta=0.5, gamma=0.95, seed=0, log=print):
    rng = np.random.default_rng(seed)
    rows = []
    t_start = time.time()
    for t0 in starts:
        bm = BehaviorModel(cutoff=t0).fit(W, LY)          # rule estimated with data up to t0
        base, active = simulate_values(W, params, bm, t0, H, {}, R, 100 + t0, gamma)
        idx = np.where(active & np.all(np.isfinite(base), 1))[0]
        ly0 = LY[:, t0]
        q80, q50 = np.nanquantile(ly0, 0.8), np.nanquantile(ly0, 0.5)
        zone = np.where(ly0 >= q80, 2, np.where(ly0 >= q50, 1, 0))
        demo = W.extra.get("v2x_polyarchy", np.full((W.N, W.T), np.nan))[:, t0]
        for j in range(draws):
            dev = rng.choice(idx, max(1, int(frac * len(idx))), replace=False)
            ch = rng.integers(0, len(CHANNELS), len(dev))
            sg = rng.choice([-1.0, 1.0], len(dev))
            offsets = {}
            for c_i, c in enumerate(CHANNELS):
                sel = ch == c_i
                if sel.any():
                    a = np.zeros((R, W.N))
                    a[:, dev[sel]] = sg[sel] * delta
                    offsets[c] = a
            val, _ = simulate_values(W, params, bm, t0, H, offsets, R, 100 + t0, gamma)
            dW = base[dev] - val[dev]
            for k, i in enumerate(dev):
                if np.all(np.isfinite(dW[k])):
                    rows.append(dict(t0=int(t0), i=int(i), iso3=W.iso3[i], canal=CHANNELS[ch[k]],
                                     signo=float(sg[k]), zona=int(zone[i]),
                                     demo=float(demo[i]) if np.isfinite(demo[i]) else None,
                                     dW=dW[k].tolist()))
        log(f"  start {1950 + t0}: {len(rows)} deviations so far ({time.time() - t_start:.0f}s)")
    return rows


def fit_theta(D, kappa=4.0, starts=24, seed=0):
    """Unit-norm theta maximising the smoothed share of satisfied inequalities theta . dW >= 0.
    D is standardised (columns divided by their sd)."""
    rng = np.random.default_rng(seed)

    def loss(v):
        nv = np.linalg.norm(v) + 1e-12
        th = v / nv
        m = kappa * D @ th
        val = np.mean(np.logaddexp(0, -m))
        g_th = -kappa * (D * (1 / (1 + np.exp(m)))[:, None]).mean(0)   # d loss / d theta
        g_v = (g_th - th * (th @ g_th)) / nv                            # chain rule through v/|v|
        return val, g_v

    best = None
    for s in range(starts):
        v0 = rng.standard_normal(D.shape[1])
        r = minimize(loss, v0, jac=True, method="L-BFGS-B")
        if best is None or r.fun < best.fun:
            best = r
    th = best.x / np.linalg.norm(best.x)
    return th, float(np.mean(D @ th >= 0))


def estimate(rows, n_boot=100, seed=0):
    dW = np.array([r["dW"] for r in rows])
    sd = dW.std(0) + 1e-12
    D = dW / sd
    th, rat = fit_theta(D)
    # bootstrap over countries (clustered)
    rng = np.random.default_rng(seed)
    countries = np.array([r["i"] for r in rows])
    uc = np.unique(countries)
    boots = []
    for b in range(n_boot):
        pick = rng.choice(uc, len(uc), replace=True)
        sel = np.concatenate([np.where(countries == c)[0] for c in pick])
        tb, _ = fit_theta(D[sel], starts=4, seed=b)
        boots.append(tb)
    boots = np.array(boots)
    # rationality under alternative rewards
    alt = {k: float(np.mean(D @ (np.array(v) / sd * sd.mean()) >= 0)) for k, v in HAND.items()}
    rnd = np.random.default_rng(1).standard_normal((2000, D.shape[1]))
    rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
    alt["aleatoria_media"] = float(np.mean((D @ rnd.T) >= 0))
    # theta in original feature units, scaled so that the welfare weight is +-1 when possible
    th_units = th / sd
    return dict(theta_std=dict(zip(FEATS, th.round(4).tolist())),
                theta_std_ic90={f: [float(np.quantile(boots[:, k], 0.05)), float(np.quantile(boots[:, k], 0.95))]
                                for k, f in enumerate(FEATS)},
                theta_unidades=dict(zip(FEATS, (th_units / np.abs(th_units).max()).round(4).tolist())),
                racionalidad=rat, racionalidad_alternativas=alt, n=len(rows), sd=sd.tolist())


def forward_select(rows, min_gain=0.005):
    """Parsimonious reward: add features greedily while the rationality index improves.
    Needed because some features are nearly collinear across deviations (poder vs elite)."""
    dW = np.array([r["dW"] for r in rows])
    sd = dW.std(0) + 1e-12
    D = dW / sd
    chosen, best_rat, path = [], 0.5, []
    while len(chosen) < len(FEATS):
        cand = []
        for k in range(len(FEATS)):
            if k in chosen:
                continue
            cols = chosen + [k]
            th, rat = fit_theta(D[:, cols], starts=8)
            cand.append((rat, k, th))
        rat, k, th = max(cand, key=lambda c: c[0])
        if rat - best_rat < min_gain:
            break
        chosen.append(k)
        best_rat = rat
        path.append(dict(agrega=FEATS[k], racionalidad=float(rat)))
    th, rat = fit_theta(D[:, chosen], starts=16)
    full = np.zeros(len(FEATS))
    full[chosen] = th
    return dict(theta_std=dict(zip(FEATS, full.round(4).tolist())), racionalidad=float(rat), camino=path,
                correlacion=np.corrcoef(D.T).round(3).tolist())


def subgroup(rows, key, groups):
    out = {}
    for name, pred in groups.items():
        sub = [r for r in rows if pred(r.get(key))]
        if len(sub) > 150:
            dW = np.array([r["dW"] for r in sub])
            D = dW / (dW.std(0) + 1e-12)
            th, rat = fit_theta(D, starts=12)
            out[name] = dict(theta_std=dict(zip(FEATS, th.round(3).tolist())), racionalidad=rat, n=len(sub))
    return out


def main():
    from .calibrate import LY, W
    cal = json.load(open(RES / "calibration.json"))
    params = Params().with_vector(list(cal["params"]), list(cal["params"].values()))
    t = time.time()
    rows = collect_deviations(W, LY, params)
    print("deviations", len(rows), round(time.time() - t), "s")
    est = estimate(rows)
    est["por_zona"] = subgroup(rows, "zona", {"periferia": lambda z: z == 0, "semiperiferia": lambda z: z == 1,
                                               "centro": lambda z: z == 2})
    est["por_regimen"] = subgroup(rows, "demo", {"democracias": lambda d: d is not None and d >= 0.5,
                                                  "autocracias": lambda d: d is not None and d < 0.5})
    est["por_decada"] = subgroup(rows, "t0", {str(1950 + t0): (lambda t0_: (lambda x: x == t0_))(t0)
                                               for t0 in (10, 20, 30, 40, 50)})
    # which channels would countries most like to change? (share of profitable deviations under theta)
    th = np.array(list(est["theta_std"].values())) / np.array(est["sd"])
    prof = {}
    for c in CHANNELS:
        for s in (-1.0, 1.0):
            sub = [r for r in rows if r["canal"] == c and r["signo"] == s]
            if sub:
                g = np.array([r["dW"] for r in sub]) @ th
                prof[f"{c}{'+' if s > 0 else '-'}"] = float(np.mean(g < 0))
    est["desvios_rentables"] = prof
    est["parsimonioso"] = forward_select(rows)
    json.dump(est, open(RES / "irl.json", "w"), indent=1)
    json.dump(rows, open(RES / "irl_deviations.json", "w"))
    print(json.dumps({k: v for k, v in est.items() if k != "sd"}, indent=1))


if __name__ == "__main__":
    main()
