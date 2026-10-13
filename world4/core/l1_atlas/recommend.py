"""Incremental recommendations: reinforcement learning anchored to the observed decision rule.

Instead of searching the global optimum (which lands on corner solutions the model cannot
vouch for), each country starts from what it would do anyway -- the decision rule estimated
from data (src/behavior.py) -- and the policy only learns bounded adjustments to it:

    a_c = a_c^BC * exp(delta_c),   delta_c = DMAX * tanh(z_c),   z ~ N(mu_theta(state), sigma)

with a quadratic penalty LAM * sum_c delta_c^2 per year (a trust region around observed
behaviour). Spending more is financed by domestic saving and foreign borrowing (model.py), so a
higher budget costs consumption now or debt service later.

Training is done in forecast mode (as in the validation): from several origins (1980 ... 2019)
the state is re-anchored to data, exogenous inputs are frozen, and coups, regimes, crises,
nationalisations, climate and disasters are endogenous for the next H years. The reward is the
difference with the decision rule alone under common random numbers:

    bienestar : 10 * (log consumption pc - BC) - 0.3 * (conflict - BC) per year
    revelada  : the reward revealed by the inverse problem (src/irl.py) without its inertia term,
                10 * sum_f w_f (feature_f - BC) per year over welfare, power (share of world GNI),
                elite income, peace and resource autonomy; the anchor plays the role of inertia
plus, at the horizon, the change in net worth (capital minus debt, share of GDP) valued at par,
so that borrowing to consume before the horizon is not free.

Outputs results/recomendaciones.json: per country, the recommended change of each channel for
2020-2030 (percentage points of GDP) against the rule, the expected effect on consumption,
income, debt and conflict, the probability of gain, and robustness across parameter sets
estimated with different data cutoffs.
"""
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad

from .behavior import BehaviorModel, BehaviorPolicy
from .model import CHANNELS, FIN_CH, Params, Simulator, build_world
from .rl import Adam, N_OBS, gae, init_mlp, mlp, observe
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RES = ROOT / "results"
ORIGINS = [30, 40, 50, 60, 69]          # 1980, 1990, 2000, 2010, 2019
H = 20                                   # years per episode (2019 origin: 11, population ends in 2030)
STEP = 5                                 # decision every 5 years
NC = len(CHANNELS)
DMAX = 0.5                               # |delta| <= 0.5: at most x1.65 or x0.61 the rule
LAM = 0.5                                # trust-region penalty per year
MODES = ("bienestar", "revelada")


def load_params(path):
    d = json.load(open(path))["params"]
    base = asdict(Params())
    return Params(**{k: float(d.get(k, v)) for k, v in base.items()})


# ----------------------------------------------------------------------------- simulator snapshots
def _snap(sim):
    import copy
    return {k: copy.deepcopy(v) for k, v in sim.__dict__.items() if k not in ("w", "p", "rng")}


def _restore(sim, st):
    import copy
    for k, v in st.items():
        setattr(sim, k, copy.deepcopy(v))


class Origin:
    """World re-anchored at origin t0 (data up to t0), ready to run H forecast years."""

    def __init__(self, Wx, W, LY, params, t0, R, seed=0):
        self.t0, self.R = t0, R
        self.bm = BehaviorModel(cutoff=t0).fit(W, LY)
        sim = Simulator(Wx, params, R=R, seed=seed)
        sim.obs_until = t0
        sim.reset(0)
        sim.psi_freeze = t0             # reset() clears it; the world rent share is frozen at t0
        for t in range(0, t0 + 1):
            if t == t0:
                sim.assimilate(t)
            sim.step(sim.hist_actions(t))
        self.sim, self.state = sim, _snap(sim)
        self.params = params
        self.t1 = min(t0 + 1 + H, Wx.T)

    def episode(self, seed, policy=None, rng=None, deterministic=False, record=False):
        """Run t0+1 .. t1-1. policy(obs) -> z (R, N, NC) or None for the rule alone.
        Returns per-year arrays and, if a policy is given, the decision buffers."""
        sim = self.sim
        _restore(sim, self.state)
        sim.rng = np.random.default_rng(seed)
        bc = BehaviorPolicy(self.bm, self.params, noise=1.0, freeze=self.t0, seed=seed + 7)
        R, N = self.R, sim.w.N
        years = list(range(self.t0 + 1, self.t1))
        out = {k: np.full((R, N, len(years)), np.nan) for k in
               ("lc", "lshare", "lelite", "ims", "conf", "ly", "debt", "hazard", "reserves", "borrow", "squeeze")}
        out.update({f"a_{c}": np.full((R, N, len(years)), np.nan) for c in CHANNELS})
        out.update({f"bc_{c}": np.full((R, N, len(years)), np.nan) for c in CHANNELS})
        buf = dict(obs=[], z=[], mask=[], delta=[])
        delta = np.zeros((R, N, NC))
        for i, t in enumerate(years):
            if policy is not None and (t - years[0]) % STEP == 0:
                obs = observe(sim, t)
                mu = policy(obs.reshape(-1, N_OBS)).reshape(R, N, NC)
                z = mu if deterministic else mu + policy.std * rng.standard_normal(mu.shape)
                delta = DMAX * np.tanh(z)
                buf["obs"].append(obs); buf["z"].append(z); buf["mask"].append(sim.active.copy())
                buf["delta"].append(delta)
            base = bc(sim, t)
            acts = {c: base[c] * np.exp(delta[..., j]) for j, c in enumerate(CHANNELS)}
            rec = sim.step(acts)
            L = sim.w.pop[:, t]
            gni = np.nan_to_num(rec["gni_pc"]) * L
            out["lc"][..., i] = np.log(np.maximum(rec["cons_pc"], 1e-9))
            out["lshare"][..., i] = np.log(np.maximum(gni / np.maximum(gni.sum(1, keepdims=True), 1e-12), 1e-12))
            out["lelite"][..., i] = np.log(np.maximum(sim.E * np.nan_to_num(rec["gni_pc"]) / np.maximum(sim.ne, 1e-3), 1e-9))
            out["ims"][..., i] = rec["ims"]
            out["conf"][..., i] = rec["conflict"]
            out["ly"][..., i] = rec["ly"]
            out["debt"][..., i] = rec["debt"]
            out["hazard"][..., i] = sim.hazard
            out["reserves"][..., i] = rec["reserves"]
            out["borrow"][..., i] = rec["borrow"]
            out["squeeze"][..., i] = rec["squeeze"]
            for c in CHANNELS:
                out[f"a_{c}"][..., i] = rec[c]
                out[f"bc_{c}"][..., i] = np.where(sim.active, base[c], np.nan)
        out["networth"] = np.where(sim.active, sim.K / np.maximum(sim.G_prev, 1e-9) - sim.debt, np.nan)
        out["active"] = sim.active.copy()
        return out, buf


_W = None


def revealed_weights():
    """Revealed-reward weights in feature units (irl.json, full estimate), inertia dropped,
    scaled so that the largest weight is 1."""
    global _W
    if _W is None:
        e = json.load(open(RES / "irl.json"))
        th = np.array(list(e["theta_std"].values())) / np.array(e["sd"])
        th = th[:5]                                   # bienestar, poder, elite, paz, autonomia
        _W = th / np.abs(th).max()
    return _W


def block_rewards(mode, pol_out, bc_out, deltas):
    """Per-decision rewards (R, N, S) relative to the rule under common random numbers."""
    n = pol_out["lc"].shape[-1]
    if mode == "bienestar":
        u = 10 * (pol_out["lc"] - bc_out["lc"]) - 0.3 * (np.nan_to_num(pol_out["conf"]) - np.nan_to_num(bc_out["conf"]))
    else:
        w = revealed_weights()
        f = [pol_out["lc"] - bc_out["lc"], pol_out["lshare"] - bc_out["lshare"], pol_out["lelite"] - bc_out["lelite"],
             -(pol_out["hazard"] - bc_out["hazard"]), -(pol_out["ims"] - bc_out["ims"])]
        u = 10 * sum(wk * np.nan_to_num(fk) for wk, fk in zip(w, f))
    u = np.nan_to_num(u)
    S = len(deltas)
    r = np.zeros(u.shape[:2] + (S,))
    for s in range(S):
        a, b = s * STEP, min((s + 1) * STEP, n)
        r[..., s] = u[..., a:b].sum(-1) - LAM * (b - a) * np.sum(deltas[s] ** 2, -1)
    r[..., -1] += 10 * np.nan_to_num(pol_out["networth"] - bc_out["networth"])
    return r


# ----------------------------------------------------------------------------- training
class Policy:
    def __init__(self, th):
        self.th = th

    @property
    def std(self):
        return np.exp(self.th["logstd"])

    def __call__(self, x):
        return mlp(self.th["pol"], x)


def train(mode, origins, iters=150, seeds=6, seed=0, log=print):
    rng = np.random.default_rng(seed)
    pol = init_mlp([N_OBS, 64, 64, NC], rng, out_scale=0.01)
    vf = init_mlp([N_OBS, 64, 64, 1], rng, out_scale=0.1)
    th = {"pol": pol, "logstd": np.full(NC, -1.0)}
    P = Policy(th)
    opt_p, opt_v = Adam(3e-4), Adam(1e-3)
    # rule-only baselines under the same random numbers
    base = {(o.t0, s): o.episode(1000 + s)[0] for o in origins for s in range(seeds)}

    def logp(th, obs, z):
        mu = mlp(th["pol"], obs)
        ls = th["logstd"]
        return anp.sum(-0.5 * ((z - mu) / anp.exp(ls)) ** 2 - ls, axis=1)

    def pol_loss(th, obs, z, adv, oldlp):
        ratio = anp.exp(logp(th, obs, z) - oldlp)
        return -anp.mean(anp.minimum(ratio * adv, anp.clip(ratio, 0.8, 1.2) * adv)) - 0.003 * anp.sum(th["logstd"])

    def v_loss(vf, obs, ret):
        return anp.mean((mlp(vf, obs)[:, 0] - ret) ** 2)

    gp, gv = grad(pol_loss), grad(v_loss)
    hist, t_start = [], time.time()
    for it in range(iters):
        O, Z, A, RET, gains = [], [], [], [], []
        for o in origins:
            s = it % seeds
            po, buf = o.episode(1000 + s, policy=P, rng=rng)
            r = block_rewards(mode, po, base[(o.t0, s)], buf["delta"])
            obs, z, m = (np.stack(buf[k], 2) for k in ("obs", "z", "mask"))
            val = mlp(vf, obs.reshape(-1, N_OBS)).reshape(m.shape)
            adv, ret = gae(np.where(m, r, 0.0), val, m, gamma=0.95, lam=0.9)
            sel = m.reshape(-1)
            O.append(obs.reshape(-1, N_OBS)[sel]); Z.append(z.reshape(-1, NC)[sel])
            A.append(adv.reshape(-1)[sel]); RET.append(ret.reshape(-1)[sel])
            gains.append(float(r.sum(-1)[m[..., 0]].mean()))
        O, Z, A, RET = (np.concatenate(x) for x in (O, Z, A, RET))
        A = (A - A.mean()) / (A.std() + 1e-8)
        old = logp(th, O, Z)
        n = len(A)
        for ep in range(4):
            perm = rng.permutation(n)
            for b in range(0, n, 2048):
                ix = perm[b:b + 2048]
                opt_p.step(th, gp(th, O[ix], Z[ix], A[ix], old[ix]))
                th["logstd"] = np.clip(th["logstd"], -3.0, 0.0)
                opt_v.step(vf, gv(vf, O[ix], RET[ix]))
        hist.append(float(np.mean(gains)))
        if it % 10 == 0 or it == iters - 1:
            d = DMAX * np.tanh(mlp(th["pol"], O))
            log(f"[{mode}] it {it:3d} gain {hist[-1]:+.3f} std {P.std.mean():.2f} "
                f"delta {np.round(d.mean(0), 3)} |d| {np.abs(d).mean():.3f} ({time.time() - t_start:.0f}s)")
    return th, hist


# ----------------------------------------------------------------------------- evaluation
def evaluate(th, origin, seeds=range(2000, 2008)):
    """Deterministic policy vs rule at one origin, several random-number seeds."""
    P = Policy(th)
    rows = []
    for s in seeds:
        b, _ = origin.episode(s)
        p, _ = origin.episode(s, policy=P, deterministic=True)
        rows.append((p, b))
    keys = list(rows[0][0].keys())
    stack = lambda i, k: np.concatenate([r[i][k] for r in rows], 0)
    P_, B_ = {k: stack(0, k) for k in keys}, {k: stack(1, k) for k in keys}
    return P_, B_


def country_table(W, P_, B_, t_idx):
    """Per-country recommendation summary from pooled worlds (R*seeds, N, years)."""
    act = P_["active"].any(0)
    first = slice(0, STEP)
    out = {}
    chg = {c: np.nanmean(P_[f"a_{c}"][..., first] - B_[f"a_{c}"][..., first], (0, 2)) for c in CHANNELS}
    base = {c: np.nanmean(B_[f"a_{c}"][..., first], (0, 2)) for c in CHANNELS}
    end = -1
    dlc = P_["lc"][..., end] - B_["lc"][..., end]
    dly = P_["ly"][..., end] - B_["ly"][..., end]
    dde = P_["debt"][..., end] - B_["debt"][..., end]
    dhz = np.nanmean(P_["hazard"] - B_["hazard"], 2)
    dres = P_["reserves"][..., end] - B_["reserves"][..., end]
    avg_lc = np.nanmean(P_["lc"] - B_["lc"], 2)
    for i in np.where(act)[0]:
        out[W.iso3[i]] = dict(
            cambio_pp={c: round(float(chg[c][i]) * 100, 2) for c in CHANNELS},
            regla_pct={c: round(float(base[c][i]) * 100, 2) for c in CHANNELS},
            presupuesto_pp=round(float(sum(chg[c][i] for c in FIN_CH)) * 100, 2),
            d_consumo_10a=round(float(np.nanmean(dlc[:, i])), 4),
            d_consumo_medio=round(float(np.nanmean(avg_lc[:, i])), 4),
            d_pib_10a=round(float(np.nanmean(dly[:, i])), 4),
            d_deuda_10a=round(float(np.nanmean(dde[:, i])), 4),
            d_conflicto=round(float(np.nanmean(dhz[:, i])), 4),
            d_reservas=round(float(np.nanmean(dres[:, i])), 4),
            prob_mejora=round(float(np.nanmean(avg_lc[:, i] > 0)), 3),
        )
    return out


def main(iters=150, R=8):
    t0 = time.time()
    from .calibrate import LY, W
    params = load_params(RES / "calibration.json")
    W30 = build_world(horizon=2030)
    origins = [Origin(W30, W, LY, params, t, R=R, seed=t) for t in ORIGINS]
    print("origins ready", round(time.time() - t0), "s", flush=True)
    res = {"meta": dict(DMAX=DMAX, LAM=LAM, H=H, origins=[1950 + t for t in ORIGINS],
                        iteraciones=iters, R=R)}
    # alternative parameter sets for robustness (estimated with data up to each cutoff)
    alt = {"final": params}
    for c in (2000, 2010):
        f = RES / "rolling" / f"params_{c}.json"
        if f.exists():
            alt[f"corte_{c}"] = load_params(f)
    for mode in MODES:
        th, curve = train(mode, origins, iters=iters)
        np.save(RES / f"policy_anclado_{mode}.npy", np.array({"pol": th["pol"], "logstd": th["logstd"],
                                                             "curve": curve}, dtype=object), allow_pickle=True)
        tabs = {}
        for name, pr in alt.items():
            o19 = Origin(W30, W, LY, pr, 69, R=R, seed=69) if name != "final" else origins[-1]
            P_, B_ = evaluate(th, o19)
            tabs[name] = country_table(W, P_, B_, 69)
            if name == "final":
                glob = dict(
                    d_consumo_mundial=float(np.nanmean(np.nanmean(P_["lc"] - B_["lc"], 2))),
                    d_pib_mundial_10a=float(np.nanmean(P_["ly"][..., -1] - B_["ly"][..., -1])),
                    d_deuda_10a=float(np.nanmean(P_["debt"][..., -1] - B_["debt"][..., -1])),
                    d_conflicto=float(np.nanmean(P_["hazard"] - B_["hazard"])),
                    endeudamiento_extra_medio=float(np.nanmean(P_["borrow"])),
                    recorte_forzado_medio=float(np.nanmean(P_["squeeze"])),
                )
        # long run: the same policy from the 2000 origin over 20 years (2001-2020)
        P_, B_ = evaluate(th, origins[ORIGINS.index(50)], seeds=range(2000, 2004))
        lr = {k: np.nanmean(P_[k] - B_[k], (0, 1)).round(4).tolist() for k in ("lc", "ly", "debt")}
        lr["anios"] = list(range(2001, 2001 + len(lr["lc"])))
        # robustness: sign agreement of each channel change across parameter sets
        fin = tabs["final"]
        for iso, row in fin.items():
            agree = {}
            for c in CHANNELS + ["presupuesto"]:
                v0 = row["presupuesto_pp"] if c == "presupuesto" else row["cambio_pp"][c]
                sg = [np.sign(v0)]
                for name, tb in tabs.items():
                    if name == "final" or iso not in tb:
                        continue
                    v = tb[iso]["presupuesto_pp"] if c == "presupuesto" else tb[iso]["cambio_pp"][c]
                    sg.append(np.sign(v))
                agree[c] = round(float(np.mean(np.array(sg) == sg[0])), 2)
            row["acuerdo_signo"] = agree
            row["mejora_en_todos"] = bool(all(tb.get(iso, {}).get("d_consumo_medio", -1) > 0 for tb in tabs.values())) \
                if mode == "bienestar" else None
        res[mode] = dict(curva=curve, global_=glob, paises=fin, largo_plazo=lr,
                         alternativos={k: v for k, v in tabs.items() if k != "final"})
        print(mode, "done", round(time.time() - t0), "s", json.dumps(glob), flush=True)
    json.dump(res, open(RES / "recomendaciones.json", "w"))


if __name__ == "__main__":
    main(*(int(a) for a in sys.argv[1:]))
