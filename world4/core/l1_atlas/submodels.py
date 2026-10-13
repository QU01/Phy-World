"""Stage-1 estimation: reduced-form submodels fitted directly on data up to a cutoff year.

These hazards and reaction functions are observable in the data, so they are estimated by
maximum likelihood / least squares instead of being tuned through the simulator. Re-fitting them
with data <= cutoff keeps the rolling-origin validation free of look-ahead.

  coups        logit P(attempt) on log income, lagged growth, coup trap, democracy, Cold War
  regimes      logits P(A->D) and P(D->A) on log income, lagged growth, share of democracies
  crises       logit P(any financial crisis onset) on debt over threshold, US real rate,
               lagged growth, share of countries in crisis last year
  debt         OLS change of debt / GDP on lagged debt, growth, crisis (reaction function)
  nationalis.  logit P(resource nationalisation) on 3-year oil price shock, price level,
               polity, post-1985 regime (countries with resource rents > 3% of GDP)
  financing    Feldstein-Horioka style: within-country change of the current account on the
               change of the investment rate with year effects, dCA = (eta - 1) dI, so eta is the
               share of extra investment financed by domestic saving; the ceiling on the
               current-account deficit is its 90th percentile
"""
import numpy as np

from .model import Params


def logit_fit(X, y, ridge=1e-3, iters=60):
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
        H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - p) - ridge * beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    p = 1 / (1 + np.exp(-np.clip(X @ beta, -30, 30)))
    H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se, int(len(y)), float(y.mean())


def _stack(rows):
    X = np.vstack([r[0] for r in rows]) if rows else np.zeros((0, 1))
    y = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0)
    return X, y


def estimate(W, LY, cutoff, verbose=False):
    """Returns (dict of Params overrides, report dict) using data with year index <= cutoff."""
    x = W.extra
    N, T = W.N, W.T
    g = np.full((N, T), np.nan)
    g[:, 1:] = np.diff(LY, axis=1)
    ell = LY - np.log(10000.0)
    cw = W.world.get("geopolitica__cold_war", np.zeros(T))
    rstar = np.clip(np.nan_to_num(W.world.get("finanzas__us_real_strate", np.full(T, 0.01)), nan=0.01), -0.05, 0.1)
    demo = (W.locf("regime_row", np.nan) >= 2).astype(float)
    demo[~np.isfinite(W.locf("regime_row", np.nan))] = np.nan
    out, rep = {}, {}

    # ---- coups
    coup = x.get("coup_any")
    trap = W.locf("coups_past10", np.nan)
    rows = []
    for t in range(2, cutoff + 1):
        f = np.column_stack([np.ones(N), ell[:, t - 1], g[:, t - 1], (trap[:, t - 1] > 0), demo[:, t - 1],
                             np.full(N, cw[t] if np.isfinite(cw[t]) else 0)])
        ok = np.all(np.isfinite(f), 1) & np.isfinite(coup[:, t])
        rows.append((f[ok], coup[ok, t]))
    X, y = _stack(rows)
    if len(y) > 300 and y.sum() > 20:
        b, se, n, rate = logit_fit(X, y)
        out.update(dict(zip(["cp0", "cp_l", "cp_g", "cp_T", "cp_D", "cp_CW"], b)))
        rep["golpes"] = dict(coef=dict(zip(["const", "ell", "g", "trampa", "democracia", "guerra_fria"], b.round(3).tolist())),
                             se=se.round(3).tolist(), n=n, tasa=round(rate, 4))

    # ---- regime transitions
    WD = np.array([np.nanmean(demo[:, t]) for t in range(T)])
    for key, frm, to, names in (("ad", 0, 1, ["ad0", "ad_l", "ad_g", "ad_W"]), ("da", 1, 0, ["da0", "da_l", "da_g", "da_W"])):
        rows = []
        for t in range(2, cutoff + 1):
            f = np.column_stack([np.ones(N), ell[:, t - 1], g[:, t - 1], np.full(N, WD[t - 1])])
            ok = np.all(np.isfinite(f), 1) & (demo[:, t - 1] == frm) & np.isfinite(demo[:, t])
            rows.append((f[ok], (demo[ok, t] == to).astype(float)))
        X, y = _stack(rows)
        if len(y) > 300 and y.sum() > 10:
            b, se, n, rate = logit_fit(X, y, ridge=1e-2)
            out.update(dict(zip(names, b)))
            rep["transicion_" + key] = dict(coef=dict(zip(["const", "ell", "g", "W_D"], b.round(3).tolist())),
                                            se=se.round(3).tolist(), n=n, tasa=round(rate, 4))

    # ---- financial crises
    cr = x.get("crisis_any")
    debt = W.locf("debt_gdp", np.nan)
    q80 = np.array([np.nanquantile(LY[:, t], 0.8) if np.isfinite(LY[:, t]).any() else np.nan for t in range(T)])
    core = 1 / (1 + np.exp(-(LY - q80[None]) / 0.35))
    dstar = 0.4 + 0.8 * core
    share = np.array([np.nanmean(cr[:, t]) if np.isfinite(cr[:, t]).any() else 0 for t in range(T)])
    rows = []
    for t in range(2, cutoff + 1):
        f = np.column_stack([np.ones(N), debt[:, t - 1] / dstar[:, t - 1] - 1, np.full(N, rstar[t]), g[:, t - 1],
                             np.full(N, share[t - 1])])
        ok = np.all(np.isfinite(f), 1) & np.isfinite(cr[:, t])
        rows.append((f[ok], cr[ok, t]))
    X, y = _stack(rows)
    if len(y) > 300 and y.sum() > 20:
        b, se, n, rate = logit_fit(X, y, ridge=1e-2)
        out.update(dict(zip(["cr0", "cr_d", "cr_r", "cr_g", "cr_cont"], b)))
        rep["crisis"] = dict(coef=dict(zip(["const", "deuda_sobre_umbral", "tasa_real_eeuu", "g", "contagio"], b.round(3).tolist())),
                             se=se.round(3).tolist(), n=n, tasa=round(rate, 4))

    # ---- debt dynamics: d_t - d_{t-1} on lagged debt, growth, crisis onset and world-rate shocks
    rows = []
    rb = float(np.mean(rstar[:cutoff + 1]))
    for t in range(2, cutoff + 1):
        dd = debt[:, t] - debt[:, t - 1] - (rstar[t] - rb) * debt[:, t - 1]
        f = np.column_stack([np.ones(N), debt[:, t - 1], g[:, t], np.nan_to_num(cr[:, t])])
        raw_d = x.get("debt_gdp")
        ok = (np.all(np.isfinite(f), 1) & np.isfinite(raw_d[:, t]) & np.isfinite(raw_d[:, t - 1])
              & (np.abs(dd) < 0.3) & (debt[:, t - 1] < 3))
        rows.append((f[ok], dd[ok]))
    X, y = _stack(rows)
    if len(y) > 300:
        b = np.linalg.lstsq(X, y, rcond=None)[0]
        out.update(dict(fd0=b[0], fd_d=b[1], fd_g=b[2], fd_c=b[3], r_bar=rb))
        rep["deuda"] = dict(coef=dict(zip(["const", "deuda", "g", "crisis"], b.round(4).tolist())), n=int(len(y)))

    # ---- nationalisations
    ev = np.nan_to_num(x.get("nac_evento_petroleo", np.zeros((N, T)))) + np.nan_to_num(x.get("nac_evento_mineria", np.zeros((N, T))))
    oil = W.world.get("recursos__petroleo_precio_real_usd2019_bbl")
    pol = W.locf("polity2", np.nan)
    if pol is None or not np.isfinite(pol).any():
        pol = W.locf("geopolitica__polity2", np.nan)
    rows = []
    if oil is not None:
        lp = np.log(oil / oil[0])
        for t in range(3, cutoff + 1):
            f = np.column_stack([np.ones(N), np.full(N, lp[t] - lp[t - 3]), np.full(N, lp[t]), pol[:, t - 1],
                                 np.full(N, float(1950 + t >= 1985))])
            ok = np.all(np.isfinite(f), 1) & (W.rr[:, t] > 0.03)
            rows.append((f[ok], (ev[ok, t] > 0).astype(float)))
        X, y = _stack(rows)
        if len(y) > 300 and y.sum() >= 8:
            if X[:, 4].std() == 0:
                X = X[:, :4]
            b, se, n, rate = logit_fit(X, y, ridge=0.1)
            names = ["nat0", "nat_shock", "nat_level", "nat_dem", "nat_post85"][:X.shape[1]]
            out.update(dict(zip(names, b)))
            rep["nacionalizacion"] = dict(coef=dict(zip(names, b.round(3).tolist())), se=se.round(3).tolist(),
                                          n=n, tasa=round(rate, 4))
    # ---- financing: domestic saving vs foreign borrowing
    I, CA = x.get("acc_k"), x.get("ca_gdp")
    if I is not None and CA is not None:
        dI, dC = np.diff(I[:, :cutoff + 1], axis=1), np.diff(CA[:, :cutoff + 1], axis=1)
        a, b = [], []
        for t in range(dI.shape[1]):
            ok = np.isfinite(dI[:, t]) & np.isfinite(dC[:, t]) & (np.abs(dI[:, t]) < 0.15) & (np.abs(dC[:, t]) < 0.2)
            if ok.sum() > 5:
                a.append(dI[ok, t] - dI[ok, t].mean())
                b.append(dC[ok, t] - dC[ok, t].mean())
        if a:
            a, b = np.concatenate(a), np.concatenate(b)
            beta = float(a @ b / (a @ a))
            res = b - beta * a
            se = float(np.sqrt(np.sum((a * res) ** 2)) / (a @ a))          # robust (HC0)
            out["fh_eta"] = float(np.clip(1 + beta, 0.0, 1.0))
            deficit = -CA[:, :cutoff + 1]
            out["cm0"] = float(np.nanquantile(deficit[np.isfinite(deficit)], 0.9))
            rep["financiamiento"] = dict(coef=dict(dCA_dI=round(beta, 3), eta=round(1 + beta, 3),
                                                   techo_deficit_q90=round(out["cm0"], 3)),
                                         se=[round(se, 3)], n=int(len(a)))
    out = {k: float(v) for k, v in out.items() if np.isfinite(v)}
    if verbose:
        import json
        print(json.dumps(rep, indent=1))
    return out, rep


def params_with_stage1(base: Params, W, LY, cutoff):
    ov, rep = estimate(W, LY, cutoff)
    return base.with_vector(list(ov), list(ov.values())), rep


if __name__ == "__main__":
    from .calibrate import LY, W
    estimate(W, LY, 69, verbose=True)
