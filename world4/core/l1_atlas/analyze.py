"""Evaluate the calibrated model, the estimated decision rules, the revealed reward (IRL), the
rolling validation and the RL policies, and export everything the dashboard needs.

Output: results/dashboard_data.json and results/summary.json
"""
import json
from pathlib import Path

import numpy as np

from .behavior import BehaviorModel, BehaviorPolicy
from .calibrate import CONF, DEBT, LY, W
from .model import CHANNEL_NAMES, CHANNELS, Params, Simulator, build_world
from .rl import mlp, observe, rollout, to_budget, to_shares, load_irl_theta
from ._paths import ATLAS_ROOT

ROOT = ATLAS_ROOT
RES = ROOT / "results"
MODES = ["bienestar", "poder", "elite", "irl"]
NK = 5  # number of policy archetypes
VARS = ["ly", "core", "E", "ne", "psi", "asab", "conflict", "rent_in", "res_share", "reserves", "fshare",
        "ims", "debt", "crisis", "demo", "temp", "dclim", "natcap", "co2", "eco_ue"] + CHANNELS
H = 2030                 # dashboard timeline 1950-2030 (projection after 2019)
TT = H - 1950 + 1


def rnd(a, d=3):
    a = np.asarray(a, float)
    out = np.round(a, d).astype(object)
    out[~np.isfinite(a)] = None
    return out.tolist()


def pad(a):
    """(N, T) -> (N, TT) with NaN after the last column."""
    a = np.asarray(a, float)
    if a.shape[-1] >= TT:
        return a[..., :TT]
    out = np.full(a.shape[:-1] + (TT,), np.nan)
    out[..., :a.shape[-1]] = a
    return out


def gini_w(x, w):
    ok = np.isfinite(x) & np.isfinite(w)
    x, w = x[ok], w[ok]
    o = np.argsort(x)
    x, w = x[o], w[o]
    cw = np.cumsum(w) / w.sum()
    cx = np.cumsum(x * w) / (x * w).sum()
    return float(1 - np.sum((cx[1:] + cx[:-1]) * np.diff(cw)) - cx[0] * cw[0])


def zone_of(ly):
    q80 = np.nanquantile(ly, 0.8, axis=-2, keepdims=True)
    q50 = np.nanquantile(ly, 0.5, axis=-2, keepdims=True)
    z = np.where(ly >= q80, 2, np.where(ly >= q50, 1, 0)).astype(float)
    z[~np.isfinite(ly)] = np.nan
    return z


def outcomes(o, t19=69):
    """World indicators in 2019 from per-run arrays (R, N, T)."""
    m = {k: np.nanmean(o[k], 0) for k in ("ly", "E", "psi", "debt", "demo", "dclim", "natcap", "reserves", "fshare", "ims", "res_share")}
    ly = m["ly"]
    pop = W.pop
    z = zone_of(ly)
    ok = np.isfinite(z[:, 20]) & np.isfinite(z[:, t19])
    y19 = np.exp(ly[:, t19])
    zc, zp = z[:, t19] == 2, z[:, t19] == 0
    res = dict(
        gini_1970=gini_w(np.exp(ly[:, 20]), pop[:, 20]), gini_2019=gini_w(y19, pop[:, t19]),
        pib_pc_mundial_2019=float(np.nansum(y19 * pop[:, t19]) / np.nansum(pop[:, t19] * np.isfinite(y19))),
        pib_pc_mediano_2019=float(np.nanmedian(y19)),
        conflicto_1990_2019=float(np.nanmean(o["hazard"][:, :, 40:t19 + 1])),
        movilidad_ascendente=float(np.mean(z[ok, t19] > z[ok, 20])),
        movilidad_descendente=float(np.mean(z[ok, t19] < z[ok, 20])),
        elite_share_2019=float(np.nanmean(m["E"][:, t19])),
        psi_centro_2019=float(np.nanmedian(m["psi"][zc, t19])), psi_periferia_2019=float(np.nanmedian(m["psi"][zp, t19])),
        deuda_mediana_2019=float(np.nanmedian(m["debt"][:, t19])),
        democracias_2019=float(np.nanmean(m["demo"][:, t19])),
        temperatura_global_2019=float(np.nanmean(o["Tg"][:, t19])) if "Tg" in o else None,
        dano_climatico_mediano_2019=float(np.nanmedian(m["dclim"][:, t19])),
        capital_natural_mediano_2019=float(np.nanmedian(m["natcap"][:, t19])),
        precio_recursos_2019=float(np.nanmean(o["price"][:, t19] / o["price"][:, 20])),
        reservas_restantes_2019=float(np.nanmedian(m["reserves"][:, t19])),
        extraccion_extranjera_periferia_2019=float(np.nanmean(m["fshare"][zp, t19])),
        dependencia_importacion_centro_2019=float(np.nanmean(m["ims"][zc, t19])),
    )
    return res


def financing(o, t0=20, t1=69):
    """Spending, foreign borrowing above the historical path and forced saving (shares of GDP)."""
    tot = sum(o[c] for c in CHANNELS)[:, :, t0:t1 + 1]
    return dict(presupuesto_medio=float(np.nanmean(tot)),
                endeudamiento_extra_medio=float(np.nanmean(o["borrow"][:, :, t0:t1 + 1])),
                recorte_forzado_medio=float(np.nanmean(o["squeeze"][:, :, t0:t1 + 1])),
                costo_ajuste_medio=float(np.nanmean(o["adjcost"][:, :, t0:t1 + 1])))


def kmeans(X, k=4, seed=0, iters=100):
    rng = np.random.default_rng(seed)
    best = None
    for rep in range(10):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), 1)
            C2 = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(C2, C):
                break
            C = C2
        inertia = ((X - C[lab]) ** 2).sum()
        if best is None or inertia < best[0]:
            best = (inertia, lab, C)
    return best[1], best[2]


def load_params(f):
    d = json.load(open(f))["params"]
    return Params().with_vector(list(d), list(d.values()))


def forecast_run(Wx, params, t0, t1, R=48, seed=5):
    """Model with observed actions up to t0 and the estimated decision rule afterwards."""
    bm = BehaviorModel(cutoff=t0).fit(W, LY)          # rule estimated with data <= t0
    bc = BehaviorPolicy(bm, params, noise=1.0, freeze=t0, seed=seed + 7)
    pol = lambda s, t: s.hist_actions(t) if t <= t0 else bc(s, t)
    sim = Simulator(Wx, params, R=R, seed=seed)
    return sim.run(0, t1, policy=pol, freeze_from=t0, assimilate_at=t0), bm


def main():
    cal = json.load(open(RES / "calibration.json"))
    params = load_params(RES / "calibration.json")
    data = {"meta": {
        "years": list(range(1950, H + 1)), "iso3": W.iso3, "names": W.names, "isonum": W.isonum,
        "channels": CHANNELS, "channel_names": CHANNEL_NAMES, "last_data_year": 2019},
        "calibration": cal, "scenarios": {}, "mundo": {}}
    W30 = build_world(horizon=H)
    data["meta"]["pop"] = rnd(W30.pop, 3)
    summary = {}

    # ---- observed data
    top10 = W.extra.get("top10_share_ptinc")
    demo_obs = np.where(np.isfinite(W.locf("regime_row", np.nan)), (W.locf("regime_row", np.nan) >= 2).astype(float), np.nan)
    obs = {"ly": LY, "conflict": CONF, "E": np.where(W.observed, top10, np.nan),
           "core": zone_of(LY) / 2, "res_share": np.where(W.observed, W.rr_obs, np.nan),
           "debt": DEBT, "demo": np.where(W.observed, demo_obs, np.nan),
           "temp": np.where(W.observed, W.extra.get("tmp_level"), np.nan),
           "co2": np.where(W.observed, W.extra.get("co2_mt"), np.nan)}
    for c in CHANNELS:
        a = W.extra.get(f"a_{c}")
        if a is not None:
            obs[c] = np.where(W.observed, a, np.nan)
    data["scenarios"]["datos"] = {k: rnd(pad(v)) for k, v in obs.items() if v is not None}
    z = zone_of(LY)
    ok = np.isfinite(z[:, 20]) & np.isfinite(z[:, 69])
    y19 = np.exp(LY[:, 69])
    summary["datos"] = dict(
        gini_1970=gini_w(np.exp(LY[:, 20]), W.pop[:, 20]), gini_2019=gini_w(y19, W.pop[:, 69]),
        pib_pc_mundial_2019=float(np.nansum(y19 * W.pop[:, 69]) / np.nansum(W.pop[:, 69] * np.isfinite(y19))),
        pib_pc_mediano_2019=float(np.nanmedian(y19)), conflicto_1990_2019=float(np.nanmean(CONF[:, 40:])),
        movilidad_ascendente=float(np.mean(z[ok, 69] > z[ok, 20])), movilidad_descendente=float(np.mean(z[ok, 69] < z[ok, 20])),
        elite_share_2019=float(np.nanmean(obs["E"][:, 69])), deuda_mediana_2019=float(np.nanmedian(DEBT[:, 69])),
        democracias_2019=float(np.nanmean(obs["demo"][:, 69])),
        temperatura_global_2019=float(W.world.get("clima__gmst_preind", np.full(70, np.nan))[69]))

    # ---- model replaying history with observed actions (in-sample)
    sim = Simulator(W, params, R=48, seed=21)
    o = sim.run(0, W.T)
    mean = {k: np.nanmean(o[k], 0) for k in VARS}
    mean["conflict"] = np.nanmean(o["hazard"], 0)
    data["scenarios"]["modelo_hist"] = {k: rnd(pad(mean[k])) for k in VARS}
    summary["modelo_hist"] = outcomes(o)
    summary["modelo_hist"].update(financing(o))
    tot_h = sum(o[c] for c in CHANNELS)
    zh = zone_of(o["ly"])
    summary["modelo_hist"]["asignacion_por_zona"] = {
        zn: {c: float(np.nanmean((o[c] / tot_h)[zh == zi])) for c in CHANNELS}
        for zi, zn in enumerate(["periferia", "semiperiferia", "centro"])}
    data["mundo"]["precio_modelo_hist"] = rnd(pad(np.nanmean(o["price"], 0)[None])[0])

    # ---- out-of-sample forecast from 1990 (parameters and decision rule estimated with data <= 1990)
    p90 = load_params(RES / "rolling" / "params_1990.json")
    of, _ = forecast_run(W, p90, 40, W.T)
    sc = {}
    for k in ("ly", "hazard", "res_share", "debt", "demo", "temp", "dclim", "co2"):
        a = np.nanmean(of[k], 0)
        a[:, :40] = np.nan
        sc["conflict" if k == "hazard" else k] = rnd(pad(a))
    for q, name in ((10, "ly_p10"), (90, "ly_p90")):
        a = np.nanpercentile(of["ly"], q, 0)
        a[:, :40] = np.nan
        sc[name] = rnd(pad(a))
    data["scenarios"]["pronostico"] = sc
    data["mundo"]["tg_pronostico_1990"] = rnd(pad(np.where(np.arange(W.T) >= 40, np.nanmean(of["Tg"], 0), np.nan)[None])[0])
    data["mundo"]["co2_pronostico_1990"] = rnd(pad(np.where(np.arange(W.T) >= 40, np.nansum(np.nanmean(of["co2"], 0), 0), np.nan)[None])[0])

    # ---- projection 2020-2030 (all data up to 2019)
    op, bm_all = forecast_run(W30, params, 69, W30.T)
    sp = {}
    for k in ("ly", "hazard", "res_share", "debt", "demo", "temp", "dclim", "co2", "psi", "E", "natcap", "reserves", "crisis") + tuple(CHANNELS):
        a = np.nanmean(op[k], 0)
        a[:, :69] = np.nan
        sp["conflict" if k == "hazard" else k] = rnd(pad(a))
    for q, name in ((10, "ly_p10"), (90, "ly_p90")):
        a = np.nanpercentile(op["ly"], q, 0)
        a[:, :69] = np.nan
        sp[name] = rnd(pad(a))
    data["scenarios"]["proyeccion"] = sp
    tgp = np.nanmean(op["Tg"], 0)
    data["mundo"]["tg_proyeccion"] = rnd(np.where(np.arange(TT) >= 69, tgp, np.nan))
    L30 = W30.pop
    y = np.exp(op["ly"])
    wgdp = np.nansum(y * L30[None], 1) / np.nansum(np.where(np.isfinite(y), L30[None], 0), 1)
    summary["proyeccion_2030"] = dict(
        crecimiento_pib_pc_mundial_anual=float(np.nanmean(np.log(wgdp[:, 80] / wgdp[:, 69])) / 11),
        temperatura_global_2030=float(np.nanmean(op["Tg"][:, 80])),
        prob_conflicto_media_2030=float(np.nanmean(op["hazard"][:, :, 80])),
        democracias_2030=float(np.nanmean(op["demo"][:, :, 80])),
        deuda_mediana_2030=float(np.nanmedian(np.nanmean(op["debt"][:, :, 80], 0))),
        crisis_anuales_2020_2030=float(np.nanmean(op["crisis"][:, :, 70:81])))

    # ---- observed world series
    gm = W.world.get("clima__gmst_preind")
    data["mundo"]["tg_obs"] = rnd(pad(gm[None])[0]) if gm is not None else None
    co2w = W.world.get("clima__co2_mt")
    data["mundo"]["co2_obs"] = rnd(pad(co2w[None])[0]) if co2w is not None else None

    # ---- decision rules (behavioral cloning) and their out-of-sample checks
    ev = {}
    for t0 in (30, 40, 50, 60):
        e = BehaviorModel(cutoff=t0).fit(W, LY).evaluate(W, LY, t0, horizons=(1, 5, 9))
        for ch, d in e.items():
            for h, r in d.items():
                ev.setdefault(ch, {}).setdefault(str(h), []).append(r["rmse_rule"] / r["rmse_persist"])
    data["conducta"] = {"reglas": bm_all.summary(),
                        "error_relativo": {ch: {h: float(np.mean(v)) for h, v in d.items()} for ch, d in ev.items()}}
    if (RES / "irl.json").exists():
        data["irl"] = json.load(open(RES / "irl.json"))
    if (RES / "validation.json").exists():
        data["validacion"] = json.load(open(RES / "validation.json"))

    show = ["USA", "CHN", "KOR", "BRA", "IND", "NGA", "DEU", "MEX", "ARG", "ZAF", "RUS", "EGY"]
    data["showcase"] = [c for c in show if c in W.iso3]

    # ---- RL policies
    rl_summary, archetype_rows, resp = {}, [], {}
    for mode in MODES:
        f = RES / f"policy_{mode}.npy"
        if not f.exists():
            continue
        if mode == "irl":
            load_irl_theta()
        th = np.load(f, allow_pickle=True).item()
        rng = np.random.default_rng(5)
        simr = Simulator(W, params, R=32, seed=33)
        _, rec = rollout(simr, th["pol"], th["logstd"], rng, mode, deterministic=True, record=True)
        rec["hazard"] = rec["conflict"]
        rec["Tg"] = np.tile(np.nanmean(o["Tg"], 0), (rec["ly"].shape[0], 1))
        m = {k: np.nanmean(rec[k], 0) for k in VARS}
        tot = sum(m[c] for c in CHANNELS)
        for c in CHANNELS:
            m[c] = m[c] / tot
        data["scenarios"][f"rl_{mode}"] = {k: rnd(pad(m[k])) for k in VARS}
        out = outcomes(rec)
        out.update(financing(rec))
        z = zone_of(rec["ly"])
        tot_r = sum(rec[c] for c in CHANNELS)
        out["asignacion_por_zona"] = {zn: {c: float(np.nanmean((rec[c] / tot_r)[z == zi])) for c in CHANNELS}
                                      for zi, zn in enumerate(["periferia", "semiperiferia", "centro"])}
        out["asignacion_por_decada"] = {str(d0): {c: float(np.nanmean(m[c][:, d0 - 1950:min(d0 - 1940, W.T)])) for c in CHANNELS}
                                        for d0 in range(1950, 2020, 10)}
        out["curva_entrenamiento"] = [float(x) for x in th["curve"]]
        data["mundo"][f"precio_rl_{mode}"] = rnd(pad(np.nanmean(rec["price"], 0)[None])[0])
        rl_summary[mode] = out
        avg = np.stack([np.nanmean(m[c], 1) for c in CHANNELS], 1)
        for i in range(W.N):
            if np.all(np.isfinite(avg[i])):
                archetype_rows.append((mode, i, avg[i]))
        simr2 = Simulator(W, params, R=4, seed=3)
        rollout(simr2, th["pol"], th["logstd"], rng, mode, deterministic=True)
        med = np.median(observe(simr2, W.T - 1)[simr2.active], 0)
        feats = {"ly_rel": (0, np.linspace(-2.5, 2.5, 21)), "reservas": (10, np.linspace(0.02, 1.0, 21)),
                 "extranjera": (11, np.linspace(0.0, 0.9, 21)), "log_psi": (4, np.linspace(-2, 3, 21)),
                 "deuda": (16, np.linspace(0.0, 1.5, 21)), "democracia": (17, np.linspace(0.05, 0.9, 21))}
        resp[mode] = {}
        for fname, (j, grid) in feats.items():
            X = np.repeat(med[None], len(grid), 0)
            X[:, j] = grid
            if fname == "ly_rel":
                X[:, 1] = 1 / (1 + np.exp(-(grid - 1.2) / 0.35))
            zz = mlp(th["pol"], X)
            sh = to_shares(zz)
            resp[mode][fname] = {"grid": grid.round(3).tolist(), **{c: sh[:, k].round(4).tolist() for k, c in enumerate(CHANNELS)},
                                 "presupuesto": to_budget(zz).round(4).tolist()}

    # observed allocation by zone (data) for comparison with the learned policies
    shares = np.stack([W.extra[f"a_{c}"] for c in CHANNELS], -1) if all(f"a_{c}" in W.extra for c in CHANNELS) else None
    if shares is not None:
        z = zone_of(LY)
        summary["datos_asignacion_por_zona"] = {zn: {c: float(np.nanmean(shares[..., k][z == zi])) for k, c in enumerate(CHANNELS)}
                                               for zi, zn in enumerate(["periferia", "semiperiferia", "centro"])}

    if archetype_rows:
        X = np.array([r[2] for r in archetype_rows])
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lab, C = kmeans(Xs, k=NK)
        cent = np.array([X[lab == j].mean(0) for j in range(NK)])
        names = {"k": "Acumulador", "r": "Desarrollista tecnológico", "m": "Importador que aprende",
                 "x": "Extractivista", "f": "Buscador de recursos fuera", "w": "Redistributivo"}
        ratio = cent / X.mean(0)
        ratio[:, X.mean(0) < 0.02] = -np.inf
        top, used = {}, set()
        for j, ci in sorted(np.ndindex(ratio.shape), key=lambda jc: -ratio[jc]):
            if j not in top and ci not in used and np.isfinite(ratio[j, ci]):
                top[j] = CHANNELS[ci]
                used.add(ci)
        arche = []
        for j in range(NK):
            members = [(archetype_rows[i][0], W.iso3[archetype_rows[i][1]]) for i in np.where(lab == j)[0]]
            arche.append({"id": j, "nombre": names[top.get(j, "k")], "canal_dominante": top.get(j, "k"),
                          "centroide": {c: float(v) for c, v in zip(CHANNELS, cent[j])}, "n": int((lab == j).sum()),
                          "por_modo": {md: int(sum(1 for a, _ in members if a == md)) for md in MODES},
                          "ejemplos": members[:400]})
        country_arch = {}
        for (md, i, _), l in zip(archetype_rows, lab):
            country_arch.setdefault(md, {})[W.iso3[i]] = int(l)
        data["arquetipos"] = arche
        data["arquetipo_pais"] = country_arch
    data["respuesta"] = resp
    if (RES / "recomendaciones.json").exists():
        data["recomendaciones"] = json.load(open(RES / "recomendaciones.json"))
    summary["rl"] = rl_summary
    data["resumen"] = summary
    json.dump(data, open(RES / "dashboard_data.json", "w"), separators=(",", ":"))
    json.dump({"resumen": summary, "arquetipos": data.get("arquetipos"), "respuesta": resp,
               "conducta": data["conducta"]}, open(RES / "summary.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "rl"}, indent=1)[:6000])
    for md, s in rl_summary.items():
        print(md, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                   if k not in ("asignacion_por_decada", "curva_entrenamiento")})


if __name__ == "__main__":
    main()
