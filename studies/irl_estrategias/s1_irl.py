"""Paso 1: la política REAL de los países por aprendizaje por refuerzo inverso (IRL, Bajari-Benkard-Levin).

Repite el estimador del Atlas (``world4.core.l1_atlas.irl``) con seis orígenes (1960-2005) y lo amplía:

* desvíos aleatorios (15 % de los países por sorteo, un canal ±0,5 logit), como el Atlas;
* desvíos DIRIGIDOS a 20 países foco (todos a la vez, canal y signo al azar), para estimar una recompensa
  revelada POR PAÍS;
* θ global, contemporáneo (orígenes 2000-2005), por zona del sistema-mundo, régimen y década; θ por país
  con bootstrap; índice de racionalidad frente a recompensas alternativas; desvíos rentables por canal
  (bajo θ revelado y bajo la recompensa de bienestar).
"""
import json
import time

import numpy as np

import common as c
from world4.core.l1_atlas import behavior, calibrate, irl
from world4.core.l1_atlas.model import CHANNELS, Params

STARTS = (10, 20, 30, 40, 50, 55)
H, R, FRAC, DELTA = 15, 8, 0.15, 0.5
DRAWS_RANDOM, DRAWS_FOCUS = 200, 150


def collect(W, LY, params, seed=0):
    rng = np.random.default_rng(seed)
    iso = list(W.iso3)
    focus = np.array([iso.index(x) for x in c.FOCUS])
    rows, quality = [], {}
    t_start = time.time()
    for t0 in STARTS:
        bm = behavior.BehaviorModel(cutoff=t0).fit(W, LY)
        quality[1950 + t0] = {ch: {"r2": r["r2"], "r2_persistencia": r["r2_persistencia"], "rho": r["rho"]}
                              for ch, r in bm.summary().items()}
        base, active = irl.simulate_values(W, params, bm, t0, H, {}, R, 100 + t0, c.GAMMA)
        idx = np.where(active & np.all(np.isfinite(base), 1))[0]
        fidx = np.intersect1d(focus, idx)
        ly0 = LY[:, t0]
        q80, q50 = np.nanquantile(ly0, 0.8), np.nanquantile(ly0, 0.5)
        zone = np.where(ly0 >= q80, 2, np.where(ly0 >= q50, 1, 0))
        demo = W.extra.get("v2x_polyarchy", np.full((W.N, W.T), np.nan))[:, t0]
        for kind, draws in (("aleatorio", DRAWS_RANDOM), ("foco", DRAWS_FOCUS)):
            for j in range(draws):
                dev = (rng.choice(idx, max(1, int(FRAC * len(idx))), replace=False) if kind == "aleatorio" else fidx)
                ch = rng.integers(0, len(CHANNELS), len(dev))
                sg = rng.choice([-1.0, 1.0], len(dev))
                offsets = {}
                for c_i, chn in enumerate(CHANNELS):
                    sel = ch == c_i
                    if sel.any():
                        a = np.zeros((R, W.N))
                        a[:, dev[sel]] = sg[sel] * DELTA
                        offsets[chn] = a
                val, _ = irl.simulate_values(W, params, bm, t0, H, offsets, R, 100 + t0, c.GAMMA)
                dW = base[dev] - val[dev]
                for k, i in enumerate(dev):
                    if np.all(np.isfinite(dW[k])):
                        rows.append(dict(t0=int(t0), i=int(i), iso3=W.iso3[i], canal=CHANNELS[ch[k]], signo=float(sg[k]),
                                         zona=int(zone[i]), demo=float(demo[i]) if np.isfinite(demo[i]) else None,
                                         tipo=kind, dW=dW[k].tolist()))
        print(f"  origen {1950 + t0}: {len(rows)} desvíos ({time.time() - t_start:.0f}s)", flush=True)
    return rows, quality


def theta_units(th_std, sd):
    th = np.asarray(th_std) / np.asarray(sd)
    return th / np.abs(th).max()


def profitable(rows, th_units):
    out = {}
    for chn in CHANNELS:
        for s in (-1.0, 1.0):
            sub = [r for r in rows if r["canal"] == chn and r["signo"] == s]
            if sub:
                g = np.array([r["dW"] for r in sub]) @ th_units
                out[f"{chn}{'+' if s > 0 else '-'}"] = float(np.mean(g < 0))   # dW = regla - desvío
    return out


def country(rows, sd, n_boot=60, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    wel = np.array(irl.HAND["bienestar"], float)
    for iso in c.FOCUS:
        sub = [r for r in rows if r["iso3"] == iso]
        if len(sub) < 100:
            continue
        D = np.array([r["dW"] for r in sub]) / sd
        th, rat = irl.fit_theta(D, starts=12)
        boots = []
        for b in range(n_boot):
            pick = rng.integers(0, len(D), len(D))
            tb, _ = irl.fit_theta(D[pick], starts=3, seed=b)
            boots.append(tb)
        boots = np.array(boots)
        thu = theta_units(th, sd)
        rat_w = float(np.mean(D @ (wel / sd * sd.mean()) >= 0))
        out[iso] = {"theta_std": dict(zip(c.FEATS, th.round(4).tolist())),
                    "theta_std_ic90": {f: [float(np.quantile(boots[:, k], 0.05)), float(np.quantile(boots[:, k], 0.95))]
                                       for k, f in enumerate(c.FEATS)},
                    "theta_unidades": dict(zip(c.FEATS, thu.round(4).tolist())), "racionalidad": rat,
                    "racionalidad_bienestar": rat_w, "n": len(sub),
                    "desvios_rentables_revelada": profitable(sub, thu),
                    "desvios_rentables_bienestar": profitable(sub, wel)}
        print(f"  {iso}: racionalidad {rat:.2f} (bienestar {rat_w:.2f}), n={len(sub)}", flush=True)
    return out


def main():
    W, LY = calibrate.W, calibrate.LY
    cal = json.load(open(irl.RES / "calibration.json"))
    params = Params().with_vector(list(cal["params"]), list(cal["params"].values()))
    t = time.time()
    rows, quality = collect(W, LY, params)
    print("desvíos", len(rows), round(time.time() - t), "s", flush=True)
    rnd = [r for r in rows if r["tipo"] == "aleatorio"]
    est = irl.estimate(rnd)                                       # el estimador del Atlas, sobre los desvíos aleatorios
    sd = np.array(est["sd"])
    thu = theta_units([est["theta_std"][f] for f in c.FEATS], sd)
    est["desvios_rentables_revelada"] = profitable(rnd, thu)
    est["desvios_rentables_bienestar"] = profitable(rnd, np.array(irl.HAND["bienestar"], float))
    est["parsimonioso"] = irl.forward_select(rnd)
    out = {"global": est, "calidad_regla": quality, "n_total": len(rows),
           "por_zona": irl.subgroup(rnd, "zona", {"periferia": lambda z: z == 0, "semiperiferia": lambda z: z == 1,
                                                  "centro": lambda z: z == 2}),
           "por_regimen": irl.subgroup(rnd, "demo", {"democracias": lambda d: d is not None and d >= 0.5,
                                                    "autocracias": lambda d: d is not None and d < 0.5}),
           "por_decada": irl.subgroup(rnd, "t0", {str(1950 + t0): (lambda a: (lambda x: x == a))(t0) for t0 in STARTS}),
           "contemporaneo": irl.subgroup(rnd, "t0", {"2000-2005": lambda x: x >= 50})["2000-2005"]}
    D = np.array([r["dW"] for r in rnd]) / sd
    out["atlas_original"] = {k: v for k, v in json.load(open(irl.RES / "irl.json")).items()
                             if k in ("theta_std", "theta_unidades", "racionalidad", "racionalidad_alternativas")}
    for k in ("por_zona", "por_regimen", "por_decada"):
        for g, v in out[k].items():
            v["theta_unidades"] = dict(zip(c.FEATS, theta_units([v["theta_std"][f] for f in c.FEATS], sd).round(4).tolist()))
    out["paises"] = country(rows, sd)
    out["setup"] = {"origenes": [1950 + s for s in STARTS], "horizonte": H, "miembros": R, "delta_logit": DELTA,
                    "fraccion": FRAC, "sorteos_aleatorios": DRAWS_RANDOM, "sorteos_foco": DRAWS_FOCUS,
                    "gamma": c.GAMMA, "segundos": round(time.time() - t)}
    c.dump("irl.json", out)
    g = out["global"]
    print(json.dumps({"theta_unidades": g["theta_unidades"], "racionalidad": g["racionalidad"],
                      "alternativas": g["racionalidad_alternativas"]}, indent=1))


if __name__ == "__main__":
    main()
