"""Paso 3: aprendizaje de estrategias (método de entropía cruzada, CEM) sobre el L1 x L0.

A) Planificador mundial: un vector de 6 desvíos de canal (logit, caja ±0,5 = la región de confianza del
   IRL) POR ZONA del sistema-mundo (periferia, semiperiferia, centro: cuantiles de ingreso 2019) y tres
   perillas mundiales (eficiencia, difusión no fósil, fecundidad, con su coste declarado en k): 21
   dimensiones. Objetivo compuesto con tres pilares de igual peso, cada métrica estandarizada por su
   dispersión entre las estrategias del paso 2:
     desarrollo = (ΔID mundial + ΔID del quintil más pobre + Δvalor de bienestar) / 3
     ambiente   = -(Δenergía fósil acumulada + Δpico de contaminación de World3) / 2
     colapso    = (-ΔP(colapso <= 2070) - ΔP(colapso severo) + ΔHWI 2100) / 3
B) Aprendizaje multiagente: los 20 países foco aprenden A LA VEZ su propio vector de 6 desvíos (CEM
   independiente por país, todos desvían juntos, cada uno puntuado por su propio resultado:
   ΔID nacional + Δvalor de bienestar, estandarizados). Es una mejor respuesta aprendida, no un equilibrio
   demostrado.
Validación fuera de muestra: semillas 101 y 202, R = 32.
"""
import time

import numpy as np

import common as c
import s2_strategies as s2

R_SEARCH, SEED = 12, 0
LO, HI = -0.5, 0.5
KNOBS = {"intensity": (-0.02, 0.0), "nonfossil": (0.0, 0.08), "tfr": (-0.015, 0.0)}
METRICS = ("dev", "dev_q1", "welfare", "fossil", "ppolx", "p2070", "psevere", "hwi")


def zones():
    W, t0 = c.live()
    ly = np.log(np.asarray(c.load("strategies.json")["gdp_pc_2019"]))
    q80, q50 = np.nanquantile(ly, 0.8), np.nanquantile(ly, 0.5)
    return np.where(ly >= q80, 2, np.where(ly >= q50, 1, 0))


def scales():
    d = c.load("strategies.json")["strategies"]
    b = d["base"]
    rows = []
    for v in d.values():
        rows.append(metric_vector(v["world"], v["w3"], v["global"], b))
    sd = np.array(rows).std(0)
    return np.where(sd > 1e-9, sd, 1.0)


def metric_vector(world, w3, glob, base):
    bw, b3 = base["world"], base["w3"]
    pop = np.asarray(c.load("strategies.json")["pop_2030"]) if "_pop" not in _CACHE else _CACHE["_pop"]
    _CACHE["_pop"] = pop
    wv = np.asarray(glob["welfare_value"], float)
    ok = np.isfinite(wv)
    return np.array([world["dev_index"] - bw["dev_index"],
                     world["dev_index_poorest_quintile"] - bw["dev_index_poorest_quintile"],
                     np.sum(wv[ok] * pop[ok]) / np.sum(pop[ok]),
                     world["fossil_cum_twh"] / bw["fossil_cum_twh"] - 1, w3["ppolx_peak"] / b3["ppolx_peak"] - 1,
                     w3["p_collapse_2070"] - b3["p_collapse_2070"], w3["p_severe"] - b3["p_severe"],
                     w3["hwi_2100"] - b3["hwi_2100"]])


_CACHE = {}


def score(mv, sd):
    z = mv / sd
    dev = (z[0] + z[1] + z[2]) / 3
    env = -(z[3] + z[4]) / 2
    col = (-z[5] - z[6] + z[7]) / 3
    return (dev + env + col) / 3, {"desarrollo": dev, "ambiente": env, "colapso": col}


def decode(x, zone):
    offs = {ch: np.array([x[zone_i * 6 + j] for zone_i in range(3)])[zone] for j, ch in enumerate(c.CHANNELS)}
    calib = {k: float(x[18 + i]) for i, k in enumerate(KNOBS)}
    offs["k"] = offs["k"] + c.green_cost(calib)
    return offs, calib


def evaluate(x, zone, ens, base, R, seed):
    offs, calib = decode(x, zone)
    f = c.l1_run(offsets=offs, calib=calib, R=R, seed=seed)
    m, sig = c.country_metrics(f), c.world_signals(f)
    mp = c.world3_constants(sig, base["sig"])
    w3 = c.world3_metrics(mp, ens)
    w3.pop("_collapse")
    glob = {k: v for k, v in s2.per_country(m, base["m"], base["th"]).items()}
    world = s2.world_summary(f, m, sig)
    return metric_vector(world, w3, glob, base["entry"]), world, w3, mp, glob


def make_base(R, seed, ens, th):
    fb = c.l1_run(R=R, seed=seed)
    mb, sb = c.country_metrics(fb), c.world_signals(fb)
    w3 = c.world3_metrics(c.world3_constants(sb, sb), ens)
    w3.pop("_collapse")
    return {"m": mb, "sig": sb, "th": th, "entry": {"world": s2.world_summary(fb, mb, sb), "w3": w3}}


def cem_world(ens, th, sd, zone, iters=8, pop=24, elite=6, seed=0):
    rng = np.random.default_rng(seed)
    lo = np.r_[np.full(18, LO), [v[0] for v in KNOBS.values()]]
    hi = np.r_[np.full(18, HI), [v[1] for v in KNOBS.values()]]
    mu, sig = (lo + hi) / 2 * 0, (hi - lo) / 4
    mu[18:] = (lo[18:] + hi[18:]) / 2
    base = make_base(R_SEARCH, SEED, ens, th)
    hist, best = [], None
    for it in range(iters):
        X = np.clip(mu + sig * rng.standard_normal((pop, len(mu))), lo, hi)
        if it == 0:
            X[0] = np.r_[np.zeros(18), [0, 0, 0]]
        S = []
        for x in X:
            mv, *_ = evaluate(x, zone, ens, base, R_SEARCH, SEED)
            s, pil = score(mv, sd)
            S.append(s)
            hist.append({"iter": it, "x": x.tolist(), "score": s, "pilares": pil, "metrics": mv.tolist()})
            if best is None or s > best[0]:
                best = (s, x.copy())
        S = np.array(S)
        el = X[np.argsort(S)[-elite:]]
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sig = np.maximum(0.7 * el.std(0) + 0.3 * sig, (hi - lo) * 0.02)
        print(f"  CEM mundial iter {it}: mejor {S.max():+.3f} media élite {np.sort(S)[-elite:].mean():+.3f}", flush=True)
    return best, mu, hist


def cem_countries(th, iters=10, pop=24, elite=6, seed=1):
    """Cada país foco aprende su vector (6,) con su propio objetivo; todos desvían a la vez."""
    W, _ = c.live()
    iso = list(W.iso3)
    idx = np.array([iso.index(x) for x in c.FOCUS])
    rng = np.random.default_rng(seed)
    nF = len(idx)
    mu, sg = np.zeros((nF, 6)), np.full((nF, 6), 0.25)
    fb = c.l1_run(R=R_SEARCH, seed=SEED)
    mb = c.country_metrics(fb)
    d = c.load("strategies.json")["strategies"]
    uni = np.array([[np.nanstd(np.asarray(v["unilateral"][k], float)) for k in ("dev", "welfare_value")]
                    for v in d.values() if "unilateral" in v]).mean(0)
    hist = []
    best = {x: (-np.inf, None, None) for x in c.FOCUS}
    for it in range(iters):
        X = np.clip(mu[None] + sg[None] * rng.standard_normal((pop, nF, 6)), LO, HI)
        if it == 0:
            X[0] = 0
        S = np.zeros((pop, nF))
        for p in range(pop):
            offs = {}
            for j, ch in enumerate(c.CHANNELS):
                a = np.zeros(W.N)
                a[idx] = X[p, :, j]
                offs[ch] = a
            f = c.l1_run(offsets=offs, R=R_SEARCH, seed=SEED)
            pc = s2.per_country(c.country_metrics(f), mb, th)
            S[p] = pc["dev"][idx] / uni[0] + pc["welfare_value"][idx] / uni[1]
            for q, x in enumerate(c.FOCUS):
                if S[p, q] > best[x][0]:
                    best[x] = (float(S[p, q]), X[p, q].tolist(), {k: float(pc[k][idx[q]]) for k in pc})
        order = np.argsort(S, 0)[-elite:]                   # (elite, nF)
        el = np.take_along_axis(X, order[:, :, None], 0)    # (elite, nF, 6)
        mu = 0.7 * el.mean(0) + 0.3 * mu
        sg = np.maximum(0.7 * el.std(0) + 0.3 * sg, 0.02)
        hist.append({"iter": it, "mean_best": float(S.max(0).mean()), "mean": float(S.mean())})
        print(f"  CEM países iter {it}: media del mejor por país {S.max(0).mean():+.3f}", flush=True)
    return {x: {"score": best[x][0], "offsets": dict(zip(c.CHANNELS, best[x][1])), "efectos": best[x][2],
                "mu": dict(zip(c.CHANNELS, mu[q].round(3).tolist()))} for q, x in enumerate(c.FOCUS)}, hist


def main():
    t0 = time.time()
    th = s2.theta_matrix()
    ens = c.world3_ensemble(400)
    sd = scales()
    zone = zones()
    (bs, bx), mu, hist = cem_world(ens, th, sd, zone)
    # validación: el mejor y la media final, semillas nuevas, R = 32
    val = {}
    for name, x in (("aprendida_mejor", bx), ("aprendida_media", np.clip(mu, None, None))):
        reps = []
        for seed in (101, 202):
            base = make_base(32, seed, ens, th)
            mv, world, w3, mp, glob = evaluate(x, zone, ens, base, 32, seed)
            s, pil = score(mv, sd)
            reps.append({"seed": seed, "score": s, "pilares": pil, "metrics": dict(zip(METRICS, mv.tolist())),
                         "world": world, "w3": w3, "w3_mapping": mp,
                         "global": {k: np.asarray(v).tolist() for k, v in glob.items()},
                         "base_world": base["entry"]["world"], "base_w3": base["entry"]["w3"]})
        offs, calib = decode(x, zone)
        val[name] = {"x": x.tolist(), "por_zona": {zn: dict(zip(c.CHANNELS, x[i * 6:(i + 1) * 6].round(3).tolist()))
                                                   for i, zn in enumerate(("periferia", "semiperiferia", "centro"))},
                     "perillas": calib, "validacion": reps}
        print(f"{name}: puntuación fuera de muestra {np.mean([r['score'] for r in reps]):+.3f}", flush=True)
    # las estrategias fijas del paso 2, puntuadas igual (en la muestra del paso 2)
    d = c.load("strategies.json")["strategies"]
    fixed = {}
    for k, v in d.items():
        mv = metric_vector(v["world"], v["w3"], v["global"], d["base"])
        s, pil = score(mv, sd)
        fixed[k] = {"score": s, "pilares": pil, "metrics": dict(zip(METRICS, mv.tolist()))}
    countries, chist = cem_countries(th)
    c.dump("learning.json", {"world": val, "history": hist, "fixed_scores": fixed, "scales": dict(zip(METRICS, sd.tolist())),
                             "countries": countries, "countries_history": chist, "zones": zone.tolist(),
                             "setup": {"R_search": R_SEARCH, "box": [LO, HI], "knobs": KNOBS, "segundos": round(time.time() - t0)}})


if __name__ == "__main__":
    main()
