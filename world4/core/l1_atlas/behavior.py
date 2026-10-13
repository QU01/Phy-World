"""Behavioral cloning: the decision rule countries actually followed, estimated from data.

This is the first half of the inverse problem. For every action channel with an observable
proxy (investment, imports, resource rents, R&D, outward FDI, tax / social spending ...) we fit a
partial-adjustment rule on the logit scale

    z_{i,t} = rho_c * z_{i,t-1} + beta_c . phi(s_{i,t-1}) + eps,   z = logit(action share of GDP)

where phi are state features the model also tracks (relative income, coreness, elite share,
conflict, resource dependence, openness, population growth, democracy, trend). Fitting only on
data up to a cutoff year gives an out-of-sample decision rule for forecasting; inside the
simulator the same rule is fed with the simulated state, so the agents behave "as countries
did" instead of following an assumed optimum.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import CHANNELS
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT

# channel -> candidate proxy columns in the panel (first available wins), shares of GDP.
PROXIES = {  # composites built in src/sources/acciones.py (see docs/research/acciones.md)
    "k": ["acc_k", "csh_i"],        # gross fixed capital formation
    "m": ["acc_m", "csh_m"],        # manufactured imports
    "x": ["res_rents"],             # resource rents (reinvested share x_hist is calibrated)
    "r": ["acc_r"],                 # R&D + public education
    "f": ["acc_f"],                 # outward FDI
    "w": ["acc_w"],                 # social spending / transfers / civil government consumption
}
FEATURES = ["ly_rel", "ly_rel2", "core", "elite", "conflict", "res_share", "open", "popg",
            "demo", "debt", "trend"]
LO, HI = 1e-4, 0.95


def logit(y):
    y = np.clip(y, LO, HI)
    return np.log(y / (1 - y))


def expit(z):
    return 1 / (1 + np.exp(-z))


def _sig(x):
    return 1 / (1 + np.exp(-x))


def data_features(W, LY, t):
    """State features phi at year index t from observed data (N, F)."""
    ly = LY[:, t]
    ok = np.isfinite(ly)
    mean = np.nanmean(ly)
    rel = ly - mean
    q80 = np.nanquantile(ly, 0.8)
    core = _sig((ly - q80) / 0.35)
    top10 = W.locf("top10_share_ptinc")[:, t]
    elite = np.clip(np.where(np.isfinite(top10), top10, 0.45 + 0.5 * ((1 - W.labsh[:, t]) - 0.47)), 0.05, 0.95)
    conflict = W.conflict[:, t]
    res = W.rr[:, t]
    opn = W.open_[:, t]
    popg = np.log(W.pop[:, t] / W.pop[:, max(t - 1, 0)]) if t > 0 else np.zeros(W.N)
    demo = _demo(W, t)
    debt = np.clip(W.locf("debt_gdp", 0.25)[:, t], 0, 3)
    trend = np.full(W.N, (t - 35) / 35)
    F = np.stack([rel, rel ** 2, core, elite, conflict, res, opn, popg * 30, demo, debt, trend], 1)
    F[~ok] = np.nan
    return F


def _demo(W, t):
    d = W.extra.get("vdem_polyarchy", W.extra.get("v2x_polyarchy"))
    if d is None:
        return np.full(W.N, 0.45)
    col = d[:, t].copy()
    # carry last observed value forward (never future)
    if np.isnan(col).any():
        past = d[:, :t + 1]
        idx = np.where(np.isfinite(past), np.arange(t + 1)[None], -1).max(1)
        col = np.where(idx >= 0, past[np.arange(W.N), np.maximum(idx, 0)], 0.45)
    return col


def sim_features(sim, t, freeze=None):
    """Same features from the simulated state (R, N, F); exogenous data frozen after `freeze`."""
    W, last = sim.w, sim.last
    R, N = sim.R, W.N
    ly = np.nan_to_num(last["ly"], nan=0.0)
    act = sim.active
    mean = np.sum(ly * act, 1, keepdims=True) / np.maximum(act.sum(1, keepdims=True), 1)
    rel = np.where(act, ly - mean, 0.0)
    te = t - 1 if freeze is None else min(t - 1, freeze)
    te = max(te, 0)
    popg = np.log(W.pop[:, t - 1] / W.pop[:, max(t - 2, 0)]) if t > 1 else np.zeros(N)
    F = np.stack([
        rel, rel ** 2, np.nan_to_num(last["core"]), sim.E, np.nan_to_num(last["conflict"]),
        np.nan_to_num(last["res_share"]), np.broadcast_to(W.open_[:, te], (R, N)),
        np.broadcast_to(popg * 30, (R, N)), sim.poly, np.clip(sim.debt, 0, 3),
        np.full((R, N), (t - 35) / 35)], -1)
    return F


@dataclass
class ChannelRule:
    column: str
    rho: float
    beta: np.ndarray
    sd: float
    n: int
    r2: float
    r2_persist: float
    fe: np.ndarray = None          # country fixed effects (N,), shrunk toward 0


@dataclass
class BehaviorModel:
    cutoff: int                       # last year index used for estimation
    rules: dict = field(default_factory=dict)
    ridge: float = 30.0               # chosen by rolling out-of-sample checks (1980-2010 origins)
    fe_shrink: float = 50.0           # country effect = sum(residuals) / (n_i + fe_shrink)
    drop: tuple = ("trend",)          # a time trend extrapolates badly out of sample

    # ------------------------------------------------------------------ estimation
    def fit(self, W, LY, t_min=1):
        feats = {t: data_features(W, LY, t - 1) for t in range(t_min, self.cutoff + 1)}
        for ch, cands in PROXIES.items():
            col = next((c for c in cands if c in W.extra and np.isfinite(W.extra[c]).sum() > 200), None)
            if col is None:
                continue
            Y = W.extra[col]
            rows, ys, cid = [], [], []
            for t in range(t_min, self.cutoff + 1):
                y, yl = Y[:, t], Y[:, t - 1]
                ok = np.isfinite(y) & np.isfinite(yl) & np.all(np.isfinite(feats[t]), 1) & (y > 0)
                if ok.any():
                    rows.append(np.column_stack([logit(yl[ok]), feats[t][ok]]))
                    ys.append(logit(y[ok]))
                    cid.append(np.where(ok)[0])
            if not rows:
                continue
            X, z, cid = np.vstack(rows), np.concatenate(ys), np.concatenate(cid)
            keep = np.array([True] + [f not in self.drop for f in FEATURES])
            X = X * keep[None]                       # dropped features get zero coefficients
            Xc = np.column_stack([X, np.ones(len(X))])
            pen = np.eye(Xc.shape[1]) * self.ridge
            pen[0, 0] = pen[-1, -1] = 0.0          # no shrinkage on persistence / intercept
            fe = np.zeros(W.N)
            for _ in range(3):  # alternate pooled coefficients and shrunk country effects
                coef = np.linalg.solve(Xc.T @ Xc + pen, Xc.T @ (z - fe[cid]))
                res = z - Xc @ coef
                fe = np.bincount(cid, res, W.N) / (np.bincount(cid, minlength=W.N) + self.fe_shrink)
            res = z - Xc @ coef - fe[cid]
            r2 = 1 - res.var() / z.var()
            r2p = 1 - (z - X[:, 0]).var() / z.var()
            self.rules[ch] = ChannelRule(col, float(coef[0]), coef[1:], float(res.std()), len(z),
                                         float(r2), float(r2p), fe)
        return self

    def evaluate(self, W, LY, t0, horizons=(1, 5, 10)):
        """Out-of-sample check of the rule with observed states: iterate the rule h years from
        t0 (lagged action simulated, states observed) and compare with the observed action."""
        out = {}
        for ch, rule in self.rules.items():
            Y = W.extra[rule.column]
            res = {}
            for h in horizons:
                if t0 + h >= W.T:
                    continue
                z = logit(Y[:, t0])
                for s in range(1, h + 1):
                    F = data_features(W, LY, t0 + s - 1)
                    z = rule.rho * z + np.nan_to_num(F) @ rule.beta[:-1] + rule.beta[-1] + rule.fe
                truth = Y[:, t0 + h]
                ok = np.isfinite(truth) & np.isfinite(z) & np.isfinite(Y[:, t0])
                if ok.sum() < 10:
                    continue
                e_m = expit(z[ok]) - truth[ok]
                e_p = Y[ok, t0] - truth[ok]
                res[h] = dict(n=int(ok.sum()), rmse_rule=float(np.sqrt(np.mean(e_m ** 2))),
                              rmse_persist=float(np.sqrt(np.mean(e_p ** 2))))
            out[ch] = res
        return out

    def summary(self):
        return {ch: dict(column=r.column, rho=round(r.rho, 3), sd=round(r.sd, 3), n=r.n,
                         r2=round(r.r2, 3), r2_persistencia=round(r.r2_persist, 3),
                         beta={f: round(float(b), 3) for f, b in zip(FEATURES + ["const"], r.beta)})
                for ch, r in self.rules.items()}


class BehaviorPolicy:
    """Stateful policy for Simulator.run: the estimated rule applied to the simulated state.
    offsets (R, N, channel) shift the logit of a channel (used for inverse-RL deviations)."""

    def __init__(self, model: BehaviorModel, params, noise=1.0, freeze=None, offsets=None, seed=0,
                 offset_from=0):
        self.m, self.p, self.noise, self.freeze = model, params, noise, freeze
        self.offsets = offsets or {}
        self.offset_from = offset_from
        self.rng = np.random.default_rng(seed)
        self.z = {}

    def _init_lags(self, sim, t, mask):
        W = sim.w
        for ch, rule in self.m.rules.items():
            Y = W.extra[rule.column]
            past = Y[:, :t + 1]
            idx = np.where(np.isfinite(past), np.arange(t + 1)[None], -1).max(1)
            last = np.where(idx >= 0, past[np.arange(W.N), np.maximum(idx, 0)], np.nan)
            fallback = np.nanmedian(last)
            v = logit(np.where(np.isfinite(last), last, fallback))
            if ch not in self.z:
                self.z[ch] = np.broadcast_to(v, (sim.R, W.N)).copy()
            else:
                self.z[ch] = np.where(mask, v[None], self.z[ch])

    def __call__(self, sim, t):
        W, p = sim.w, self.p
        if not self.z or not sim.last:
            self._init_lags(sim, max(t - 1, 0), np.ones((sim.R, W.N), bool))
            self.seen = sim.active.copy()
        new = sim.active & ~self.seen
        if new.any():
            self._init_lags(sim, max(t - 1, 0), new)
            self.seen |= new
        prox = {}
        if sim.last:
            F = sim_features(sim, t, self.freeze)
            for ch, rule in self.m.rules.items():
                eps = self.rng.standard_normal((sim.R, W.N)) * rule.sd * self.noise
                z = rule.rho * self.z[ch] + F @ rule.beta[:-1] + rule.beta[-1] + rule.fe[None] + eps
                self.z[ch] = np.clip(z, logit(LO), logit(HI))
        on = t >= self.offset_from
        for ch in self.m.rules:
            prox[ch] = expit(self.z[ch] + (self.offsets.get(ch, 0.0) if on else 0.0))
        core = sim.last.get("core", np.zeros((sim.R, W.N)))
        k = prox.get("k", np.broadcast_to(W.csh_i[:, t], (sim.R, W.N)))
        # same mapping from observed spending to model channels as Simulator.hist_actions
        acts = dict(
            k=k,
            m=p.m_scale * np.clip(prox["m"], 0, 0.8) if "m" in prox else p.m_hist * W.csh_m[:, t],
            x=p.x_hist * prox["x"] if "x" in prox else p.x_hist * W.rr[:, t],
            r=p.r_scale * prox["r"] if "r" in prox else p.r_hist * k / 0.22,
            f=p.f_scale * np.clip(prox["f"], 0, 0.3) if "f" in prox else p.f_hist * core,
            w=p.w_scale * prox["w"] if "w" in prox else np.full((sim.R, W.N), p.w_hist),
        )
        # deviations for channels without an estimated rule act multiplicatively
        for ch, off in (self.offsets.items() if on else []):
            if ch not in self.m.rules:
                acts[ch] = acts[ch] * np.exp(off)
        return {c: np.broadcast_to(acts[c], (sim.R, W.N)) for c in CHANNELS}


def fit_and_report(cutoff_year=1990):
    from .calibrate import LY, W
    bm = BehaviorModel(cutoff=cutoff_year - 1950).fit(W, LY)
    ev = bm.evaluate(W, LY, cutoff_year - 1950)
    return bm, ev


if __name__ == "__main__":
    bm, ev = fit_and_report(1990)
    print(json.dumps(bm.summary(), indent=1))
    print(json.dumps(ev, indent=1))
