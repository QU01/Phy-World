"""Rolling-origin validation at 1, 5 and 10 years, with baselines and theory ablations.

For every origin year t0 (1970 ... 2018):
  * parameters come from the most recent calibration that only used data up to a cutoff <= t0
    (cutoffs 1970, 1980, 1990, 2000, 2010; results/rolling/params_<year>.json);
  * the decision rule (behavioral cloning) is re-estimated with data up to t0;
  * the model runs from 1950 with observed actions up to t0, is re-anchored to the data at t0
    (GDP, capital, elite share, conflict, resource rents), and then forecasts t0+1 ... t0+10 with
    countries following the estimated decision rule and exogenous series frozen at t0
    (population and human capital remain the observed ones — they are the only future data used).
Baselines use only data up to t0. Ablations switch off one theory at a time with the same
parameters (no re-calibration), which shows what each mechanism adds to out-of-sample accuracy.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .behavior import BehaviorModel, BehaviorPolicy
from .calibrate import CONF, DEBT, LY, RR, W, auc, calibrate_cutoff
from .model import Params, Simulator
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RES = ROOT / "results"
ROLL = RES / "rolling"
DEMO = np.where(W.observed, (W.locf("regime_row", np.nan) >= 2).astype(float), np.nan)
DEMO[~np.isfinite(W.locf("regime_row", np.nan))] = np.nan
GMST = W.world.get("clima__gmst_preind", np.full(W.T, np.nan))
CUTOFFS = [1970, 1980, 1990, 2000, 2010]
HORIZONS = [1, 5, 10]

ABLATIONS = {
    "completo": {},
    "sin_sistema_mundo": {"tau": 0.0, "mu": 0.0},
    "sin_demografia_estructural": {"b1": 0.0, "beta_e": 0.0, "eps_r": 0.0, "bY": 0.0},
    "sin_frontera_metaetnica": {"r_a": 0.0, "d_a": 0.0, "b3": 0.0},
    "sin_politica_de_recursos": {"zeta": 0.0, "roy": 1.0, "phi_R": 0.5},
    "sin_deuda_crisis": {"cr0": -30.0, "prem_p": 0.0, "prem_d": 0.0, "ext_int": 0.0},
    "sin_instituciones": {"cp0": -30.0, "ad0": -30.0, "da0": -30.0},
    "sin_clima_ecologia": {"bhm1": 0.0, "bhm2": 0.0, "dis_p": 0.0, "gam_cl": 0.0, "xi_N": 0.0},
}


def params_for(year):
    c = max([c for c in CUTOFFS if c <= year])
    d = json.load(open(ROLL / f"params_{c}.json"))
    return Params().with_vector(list(d["params"]), list(d["params"].values())), c


def run_rolling_calibration(maxiter=30, popsize=8):
    """One calibration per cutoff using only data up to it (stage 1 + stage 2)."""
    from .calibrate import NAMES
    ROLL.mkdir(parents=True, exist_ok=True)
    prev = json.load(open(RES / "calibration.json"))["params"] if (RES / "calibration.json").exists() else {}
    for c in CUTOFFS:
        f = ROLL / f"params_{c}.json"
        if f.exists():
            continue
        t = time.time()
        x0 = [prev.get(n, getattr(Params(), n)) for n in NAMES]
        params, fit, free, st1 = calibrate_cutoff(c, maxiter=maxiter, popsize=popsize, x0=x0)
        json.dump({"cutoff": c, "params": params, "libres": free, "fit": fit, "etapa1": st1,
                   "seconds": time.time() - t}, open(f, "w"), indent=1)
        print(f"calibrated cutoff {c} in {time.time() - t:.0f}s: {fit}", flush=True)


def forecast(params, t0, R=32, seed=0, policy="bc", overrides=None):
    """Ensemble forecast from origin t0; returns dict of (R, N, H+1) arrays for years t0..t0+10."""
    if overrides:
        d = asdict(params)
        d.update(overrides)
        params = Params(**d)
    t1 = min(t0 + max(HORIZONS) + 1, W.T)
    sim = Simulator(W, params, R=R, seed=seed)
    if policy == "bc":
        bm = BehaviorModel(cutoff=t0).fit(W, LY)
        bc = BehaviorPolicy(bm, params, noise=1.0, freeze=t0, seed=seed + 7)
        pol = lambda s, t: s.hist_actions(t) if t <= t0 else bc(s, t)
        o = sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0)
    else:  # frozen: each country keeps its 10-year average actions (previous approach)
        o = sim.run(0, t1, freeze_from=t0, assimilate_at=t0)
    out = {k: o[k][:, :, t0:t1] for k in ("ly", "hazard", "res_share", "debt", "demo")}
    out["Tg"] = o["Tg"][:, t0:t1]
    return out


def crps_ensemble(ens, y):
    """CRPS of an ensemble (R, n) against observations (n,)."""
    e = np.sort(ens, 0)
    R = e.shape[0]
    term1 = np.mean(np.abs(e - y[None]), 0)
    i = np.arange(1, R + 1)[:, None]
    term2 = np.sum((2 * i - R - 1) * e, 0) / (R * R)
    return term1 - term2


# ----------------------------------------------------------------------------- baselines
def baselines(t0, h):
    """Point forecasts of log GDP pc at t0+h using data up to t0 only."""
    y0 = LY[:, t0]
    out = {"persistencia": y0}
    g10 = (LY[:, t0] - LY[:, max(t0 - 10, 0)]) / min(10, t0) if t0 > 0 else np.zeros(W.N)
    out["deriva_propia"] = y0 + h * np.where(np.isfinite(g10), g10, 0.0)
    gw = np.nanmean(np.diff(LY[:, :t0 + 1], axis=1))
    out["deriva_global"] = y0 + h * gw
    # AR(1) in annual growth, pooled
    g = np.diff(LY[:, :t0 + 1], axis=1)
    x, y = g[:, :-1].ravel(), g[:, 1:].ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    b = np.polyfit(x[ok], y[ok], 1) if ok.sum() > 50 else np.array([0.0, gw])
    gl = LY[:, t0] - LY[:, t0 - 1] if t0 > 0 else np.zeros(W.N)
    gl = np.where(np.isfinite(gl), gl, gw)
    yy, gg = y0.copy(), gl.copy()
    for _ in range(h):
        gg = b[1] + b[0] * gg
        yy = yy + gg
    out["ar1_crecimiento"] = yy
    # direct panel regression: h-year growth on level, past 5-year growth and openness
    X, Y = [], []
    for s in range(5, t0 - h + 1):
        f = np.column_stack([np.ones(W.N), LY[:, s], LY[:, s] - LY[:, s - 5], W.open_[:, s]])
        tgt = LY[:, s + h] - LY[:, s]
        ok = np.all(np.isfinite(f), 1) & np.isfinite(tgt)
        X.append(f[ok]); Y.append(tgt[ok])
    if X and sum(len(v) for v in Y) > 50:
        X, Y = np.vstack(X), np.concatenate(Y)
        coef = np.linalg.lstsq(X, Y, rcond=None)[0]
        f0 = np.column_stack([np.ones(W.N), y0, y0 - LY[:, max(t0 - 5, 0)], W.open_[:, t0]])
        out["panel_directo"] = y0 + np.nan_to_num(f0 @ coef, nan=h * gw)
    return out


def conflict_baselines(t0, h):
    c0 = np.nan_to_num(CONF[:, t0])
    freq = np.nanmean(CONF[:, max(0, t0 - 20):t0 + 1], 1)
    out = {"persistencia": np.clip(c0, 0.02, 0.98), "frecuencia_pasada": np.nan_to_num(freq, nan=0.1)}
    # logit on current conflict, past frequency and income, fitted with pairs (s, s+h), s+h <= t0
    X, Y = [], []
    for s in range(20, t0 - h + 1):
        f = np.column_stack([np.ones(W.N), np.nan_to_num(CONF[:, s]), np.nanmean(CONF[:, s - 20:s + 1], 1),
                             LY[:, s] - np.nanmean(LY[:, s])])
        tgt = CONF[:, s + h]
        ok = np.all(np.isfinite(f), 1) & np.isfinite(tgt)
        X.append(f[ok]); Y.append(tgt[ok])
    if X and sum(len(v) for v in Y) > 100:
        X, Y = np.vstack(X), np.concatenate(Y)
        beta = np.zeros(X.shape[1])
        for _ in range(50):  # Newton-Raphson logit
            p = 1 / (1 + np.exp(-X @ beta))
            Hs = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
            beta += np.linalg.solve(Hs, X.T @ (Y - p))
        f0 = np.column_stack([np.ones(W.N), c0, np.nan_to_num(freq, nan=0.1), np.nan_to_num(LY[:, t0] - np.nanmean(LY[:, t0]))])
        out["logit_historico"] = 1 / (1 + np.exp(-f0 @ beta))
    return out


# ----------------------------------------------------------------------------- main loop
def evaluate(origins=range(20, 69), R=32, variants=("completo",), include_frozen=True, log=print):
    recs = []  # one row per (variant, origin, horizon, country)
    t_start = time.time()
    for t0 in origins:
        params, cut = params_for(1950 + t0)
        runs = {v: forecast(params, t0, R=R, seed=t0, overrides=ABLATIONS[v]) for v in variants}
        if include_frozen:
            runs["politica_congelada"] = forecast(params, t0, R=R, seed=t0, policy="frozen")
        for h in HORIZONS:
            t = t0 + h
            if t >= W.T:
                continue
            truth, ok0 = LY[:, t], np.isfinite(LY[:, t]) & np.isfinite(LY[:, t0])
            bl = baselines(t0, h)
            cb = conflict_baselines(t0, h)
            for v, o in runs.items():
                ens = o["ly"][:, :, h]
                okv = ok0 & np.all(np.isfinite(ens), 0)
                recs.append(dict(
                    kind="gdp", method=v, t0=t0, h=h, iso=np.where(okv)[0],
                    pred=np.nanmean(ens, 0)[okv], truth=truth[okv], base=LY[okv, t0],
                    crps=crps_ensemble(ens[:, okv], truth[okv]),
                    cover=((truth[okv] >= np.percentile(ens[:, okv], 10, 0)) &
                           (truth[okv] <= np.percentile(ens[:, okv], 90, 0)))))
                c_true = CONF[:, t]
                hz = np.nanmean(o["hazard"][:, :, h], 0)
                okc = np.isfinite(c_true) & np.isfinite(hz)
                recs.append(dict(kind="conf", method=v, t0=t0, h=h, pred=hz[okc], truth=c_true[okc]))
                rs = np.nanmean(o["res_share"][:, :, h], 0)
                okr = np.isfinite(RR[:, t]) & np.isfinite(rs) & np.isfinite(RR[:, t0])
                recs.append(dict(kind="rr", method=v, t0=t0, h=h, pred=rs[okr], truth=RR[okr, t]))
                ds = np.nanmean(o["debt"][:, :, h], 0)
                okd = np.isfinite(DEBT[:, t]) & np.isfinite(ds) & np.isfinite(DEBT[:, t0]) & (DEBT[:, t] < 3)
                recs.append(dict(kind="debt", method=v, t0=t0, h=h, pred=ds[okd], truth=DEBT[okd, t]))
                dm = np.nanmean(o["demo"][:, :, h], 0)
                okm = np.isfinite(DEMO[:, t]) & np.isfinite(dm)
                recs.append(dict(kind="demo", method=v, t0=t0, h=h, pred=dm[okm], truth=DEMO[okm, t]))
                tg = GMST[t]
                if np.isfinite(tg):
                    recs.append(dict(kind="temp", method=v, t0=t0, h=h, pred=np.array([np.mean(o["Tg"][:, h])]),
                                     truth=np.array([tg])))
            for name, pr in bl.items():
                okb = ok0 & np.isfinite(pr)
                recs.append(dict(kind="gdp", method=name, t0=t0, h=h, iso=np.where(okb)[0], pred=pr[okb],
                                 truth=truth[okb], base=LY[okb, t0]))
            c_true = CONF[:, t]
            for name, pr in cb.items():
                okc = np.isfinite(c_true) & np.isfinite(pr)
                recs.append(dict(kind="conf", method=name, t0=t0, h=h, pred=pr[okc], truth=c_true[okc]))
            okr = np.isfinite(RR[:, t]) & np.isfinite(RR[:, t0])
            recs.append(dict(kind="rr", method="persistencia", t0=t0, h=h, pred=RR[okr, t0], truth=RR[okr, t]))
            okd = np.isfinite(DEBT[:, t]) & np.isfinite(DEBT[:, t0]) & (DEBT[:, t] < 3)
            recs.append(dict(kind="debt", method="persistencia", t0=t0, h=h, pred=DEBT[okd, t0], truth=DEBT[okd, t]))
            okm = np.isfinite(DEMO[:, t]) & np.isfinite(DEMO[:, t0])
            recs.append(dict(kind="demo", method="persistencia", t0=t0, h=h,
                             pred=np.clip(DEMO[okm, t0], 0.02, 0.98), truth=DEMO[okm, t]))
            if np.isfinite(GMST[t]):
                slope = np.polyfit(np.arange(10), GMST[t0 - 9:t0 + 1], 1)[0]
                recs.append(dict(kind="temp", method="persistencia", t0=t0, h=h, pred=np.array([GMST[t0]]),
                                 truth=np.array([GMST[t]])))
                recs.append(dict(kind="temp", method="tendencia_10a", t0=t0, h=h, pred=np.array([GMST[t0] + slope * h]),
                                 truth=np.array([GMST[t]])))
        if (t0 - list(origins)[0]) % 5 == 0:
            log(f"  origin {1950 + t0} (params cutoff {cut}) {time.time() - t_start:.0f}s")
    return recs


def summarise(recs):
    out = {}
    for kind in ("gdp", "conf", "rr", "debt", "demo", "temp"):
        out[kind] = {}
        for h in HORIZONS:
            rows = [r for r in recs if r["kind"] == kind and r["h"] == h]
            methods = sorted({r["method"] for r in rows})
            res = {}
            for m in methods:
                rr = [r for r in rows if r["method"] == m]
                pred = np.concatenate([r["pred"] for r in rr])
                truth = np.concatenate([r["truth"] for r in rr])
                if kind == "gdp":
                    base = np.concatenate([r["base"] for r in rr])
                    e = pred - truth
                    gp, gt = pred - base, truth - base
                    d = dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))),
                             corr_crecimiento=float(np.corrcoef(gp, gt)[0, 1]) if np.std(gp) > 0 else None)
                    if "crps" in rr[0]:
                        d["crps"] = float(np.mean(np.concatenate([r["crps"] for r in rr])))
                        d["cobertura_80"] = float(np.mean(np.concatenate([r["cover"] for r in rr])))
                    else:
                        d["crps"] = float(np.mean(np.abs(e)))  # point forecast: CRPS = MAE
                elif kind in ("conf", "demo"):
                    p = np.clip(pred, 1e-4, 1 - 1e-4)
                    d = dict(n=int(len(p)), brier=float(np.mean((p - truth) ** 2)), auc=float(auc(truth, p)),
                             logloss=float(-np.mean(truth * np.log(p) + (1 - truth) * np.log(1 - p))))
                else:
                    e = pred - truth
                    d = dict(n=int(len(e)), rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))))
                res[m] = d
            out[kind][str(h)] = res
    return out


def _eval_chunk(args):
    origins, variants, R = args
    return evaluate(origins=list(origins), variants=variants, R=R, log=lambda *a: None)


def main(variants=tuple(ABLATIONS), R=32, workers=4):
    run_rolling_calibration()
    t = time.time()
    origins = list(range(20, 69))
    if workers > 1:
        from multiprocessing import get_context
        chunks = [origins[k::workers] for k in range(workers)]
        with get_context("fork").Pool(workers) as pool:
            parts = pool.map(_eval_chunk, [(c, variants, R) for c in chunks])
        recs = [r for part in parts for r in part]
    else:
        recs = evaluate(variants=variants, R=R)
    summ = summarise(recs)
    summ["meta"] = dict(origins=[1970, 2018], cutoffs=CUTOFFS, horizons=HORIZONS, R=R,
                        seconds=time.time() - t)
    json.dump(summ, open(RES / "validation.json", "w"), indent=1)
    for kind in ("gdp", "conf", "rr", "debt", "demo", "temp"):
        for h in HORIZONS:
            print(kind, h)
            for m, d in sorted(summ[kind][str(h)].items(), key=lambda kv: kv[1].get("rmse", kv[1].get("brier", 0))):
                print("   ", m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()})


if __name__ == "__main__":
    main()
