"""W3 y W5 con el L1 acoplado: validación de origen móvil 1970-2018 a 1, 5 y 10 años.

Mismo protocolo que ``validate.evaluate`` del Atlas (49 orígenes, calibración de corte <= origen,
regla de conducta estimada hasta el origen, R = 32, semilla = origen) con la población y el capital
humano en tres modos (``coupling.forecast``):

    observed     el Atlas: población y hc OBSERVADOS después del origen (fuga D1/D2). Es la
                 referencia de W3 y, a la vez, el mutante del test de fuga.
    trend        extrapolación de 5 años (el estimador de F0.5)
    endogenous   los sectores de F1

más ``observed`` con otra semilla, que mide cuánto se mueve una métrica sólo por el azar del
ensemble (la banda de W5).

Las métricas son las del Atlas (CRPS del log PIB pc, Brier de conflicto y democracia, RMSE de
rentas y deuda, error de GMST), con el AUC de ``world4.validate.metrics`` (sin el defecto de
empates). Se añade la población: CRPS del log de la población de cada país.
"""
from __future__ import annotations

import json
import time
from multiprocessing import get_context

import numpy as np

ORIGINS = tuple(range(20, 69))                 # 1970..2018
HORIZONS = (1, 5, 10)
MODES = ("observed", "trend", "endogenous", "observed_seed2", "f2", "f2_persist", "f3", "f4", "f5")
SEED2 = 10_000
KINDS = ("gdp", "pop", "conf", "rr", "debt", "demo", "temp", "psi", "pe", "gmst", "co2", "cereal", "cropland", "kcal")
LOWER_IS_BETTER = {"gdp": "crps", "pop": "crps", "conf": "brier", "rr": "rmse", "debt": "rmse", "demo": "brier",
                   "temp": "mae", "psi": "crps", "pe": "crps", "gmst": "crps", "co2": "crps", "cereal": "crps",
                   "cropland": "crps", "kcal": "crps"}
CRPS_KINDS = ("gdp", "pop", "psi", "pe", "gmst", "co2", "cereal", "cropland", "kcal")
FOOD_KINDS = {"cereal": ("food_prod", "cereal_prod"), "cropland": ("food_cropland", "cropland"), "kcal": ("food_kcal", "kcal")}


def _acc():
    return {"sum": 0.0, "n": 0}


def one_origin(t0: int, R: int = 32, modes=MODES) -> dict:
    """Sumas por (modo, variable, horizonte) para un origen, y las de las referencias."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, validate
    from world4.validate.metrics import crps_ensemble
    W, LY = calibrate.W, calibrate.LY
    CONF, RR, DEBT, DEMO, GMST = calibrate.CONF, calibrate.RR, calibrate.DEBT, validate.DEMO, validate.GMST
    LP = np.log(W.pop)
    PSI = np.where(np.arange(W.T) >= 20, W.psi_t, np.nan)       # rentas del WDI desde 1970
    from world4.sectors import energy as _en
    _pe = _en.panel(W)["pe"]
    LPE = np.log(np.where(_pe > 0, _pe, np.nan))
    from world4.data import climate as _cd
    yrs = 1950 + np.arange(W.T)
    GOBS = _cd.gmst_observed(yrs)                              # HadCRUT5 respecto de 1850-1900
    LCO2 = np.log(_cd.co2_ffi_observed(yrs))                   # CO2 fósil e industrial (CMIP7)
    from world4.data import food as _fd
    _fa = _fd.aligned(W.iso3, W.isonum, W.years)
    LFOOD = {k: np.log(np.where(_fa[src] > 0, _fa[src], np.nan)) for k, (_, src) in FOOD_KINDS.items()}
    out = {"t0": t0, "modes": {}, "baselines": {}}
    for mode in modes:
        pm = {"observed_seed2": "observed", "f2": "endogenous", "f2_persist": "endogenous",
              "f3": "endogenous", "f4": "endogenous", "f5": "endogenous"}.get(mode, mode)
        seed = t0 + (SEED2 if mode == "observed_seed2" else 0)
        f2 = {"f2": coupling.F2Config(), "f2_persist": coupling.F2Config(persist_psi=True),
              "f3": coupling.F2Config(), "f4": coupling.F2Config(), "f5": coupling.F2Config()}.get(mode)
        f3 = coupling.F3Config() if mode in ("f3", "f4", "f5") else None
        f4 = coupling.F4Config() if mode in ("f4", "f5") else None
        f5 = coupling.F5Config() if mode == "f5" else None
        f = coupling.forecast(t0, R=R, seed=seed, pop_mode=pm, f2=f2, f3=f3, f4=f4, f5=f5)
        res = {}
        for h in HORIZONS:
            t = t0 + h
            if t >= W.T:
                continue
            r = {}
            ens = f["ly"][:, :, h]
            ok = np.isfinite(LY[:, t]) & np.isfinite(LY[:, t0]) & np.all(np.isfinite(ens), 0)
            c = crps_ensemble(ens[:, ok], LY[ok, t])
            lo, hi = np.percentile(ens[:, ok], 10, 0), np.percentile(ens[:, ok], 90, 0)
            r["gdp"] = {"crps": float(c.sum()), "cover": float(((LY[ok, t] >= lo) & (LY[ok, t] <= hi)).sum()),
                        "n": int(ok.sum())}
            pe = np.log(np.maximum(f["pop"][:, :, h], 1e-12))
            okp = np.isfinite(LP[:, t]) & np.all(np.isfinite(pe), 0)
            r["pop"] = {"crps": float(crps_ensemble(pe[:, okp], LP[okp, t]).sum()), "n": int(okp.sum()),
                        "cover": float(((LP[okp, t] >= np.percentile(pe[:, okp], 10, 0))
                                        & (LP[okp, t] <= np.percentile(pe[:, okp], 90, 0))).sum())}
            hz = np.nanmean(f["hazard"][:, :, h], 0)
            okc = np.isfinite(CONF[:, t]) & np.isfinite(hz)
            r["conf"] = {"brier": float(((np.clip(hz[okc], 1e-4, 1 - 1e-4) - CONF[okc, t]) ** 2).sum()),
                         "n": int(okc.sum()), "pred": hz[okc].tolist(), "truth": CONF[okc, t].tolist()}
            rs = np.nanmean(f["res_share"][:, :, h], 0)
            okr = np.isfinite(RR[:, t]) & np.isfinite(rs) & np.isfinite(RR[:, t0])
            r["rr"] = {"sse": float(((rs[okr] - RR[okr, t]) ** 2).sum()), "n": int(okr.sum())}
            ds = np.nanmean(f["debt"][:, :, h], 0)
            okd = np.isfinite(DEBT[:, t]) & np.isfinite(ds) & np.isfinite(DEBT[:, t0]) & (DEBT[:, t] < 3)
            r["debt"] = {"sse": float(((ds[okd] - DEBT[okd, t]) ** 2).sum()), "n": int(okd.sum())}
            dm = np.nanmean(f["demo"][:, :, h], 0)
            okm = np.isfinite(DEMO[:, t]) & np.isfinite(dm)
            r["demo"] = {"brier": float(((np.clip(dm[okm], 1e-4, 1 - 1e-4) - DEMO[okm, t]) ** 2).sum()),
                         "n": int(okm.sum())}
            if np.isfinite(GMST[t]):
                r["temp"] = {"abs": float(abs(np.mean(f["Tg"][:, h]) - GMST[t])), "n": 1}
            if "energy_PE" in f and t0 + h < LPE.shape[1]:
                ee = np.log(np.maximum(f["energy_PE"][:, :, h], 1e-9))
                oke = np.isfinite(LPE[:, t0 + h]) & np.all(np.isfinite(ee), 0) & np.isfinite(LPE[:, t0])
                r["pe"] = {"crps": float(crps_ensemble(ee[:, oke], LPE[oke, t0 + h]).sum()), "n": int(oke.sum()),
                           "cover": float(((LPE[oke, t0 + h] >= np.percentile(ee[:, oke], 10, 0))
                                           & (LPE[oke, t0 + h] <= np.percentile(ee[:, oke], 90, 0))).sum())}
            tg = f["Tg"][:, h]
            if np.isfinite(GOBS[t]) and np.all(np.isfinite(tg)):
                r["gmst"] = {"crps": float(crps_ensemble(tg[:, None], GOBS[t:t + 1])[0]), "n": 1,
                             "cover": float(np.percentile(tg, 10) <= GOBS[t] <= np.percentile(tg, 90))}
            if "climate_co2_ffi" in f and np.isfinite(LCO2[t]):
                lc = np.log(np.maximum(f["climate_co2_ffi"][:, h], 1e-9))
                r["co2"] = {"crps": float(crps_ensemble(lc[:, None], LCO2[t:t + 1])[0]), "n": 1,
                            "cover": float(np.percentile(lc, 10) <= LCO2[t] <= np.percentile(lc, 90))}
            for kind, (key, _) in FOOD_KINDS.items():
                if key in f:
                    tr = LFOOD[kind]
                    ee = np.log(np.maximum(f[key][:, :, h], 1e-9))
                    okf = np.isfinite(tr[:, t]) & np.isfinite(tr[:, t0]) & np.all(np.isfinite(ee), 0)
                    if okf.any():
                        r[kind] = {"crps": float(crps_ensemble(ee[:, okf], tr[okf, t]).sum()), "n": int(okf.sum()),
                                   "cover": float(((tr[okf, t] >= np.percentile(ee[:, okf], 10, 0))
                                                   & (tr[okf, t] <= np.percentile(ee[:, okf], 90, 0))).sum())}
            ps = f["psi"][:, h]
            if np.isfinite(PSI[t]) and np.all(np.isfinite(ps)):
                r["psi"] = {"crps": float(crps_ensemble(ps[:, None], PSI[t:t + 1])[0]), "n": 1,
                            "cover": float(np.percentile(ps, 10) <= PSI[t] <= np.percentile(ps, 90))}
            res[str(h)] = r
        out["modes"][mode] = res
    for h in HORIZONS:
        t = t0 + h
        if t >= W.T:
            continue
        b = {}
        ok0 = np.isfinite(LY[:, t]) & np.isfinite(LY[:, t0])
        for name, pr in validate.baselines(t0, h).items():
            okb = ok0 & np.isfinite(pr)
            b[f"gdp/{name}"] = (float(np.abs(pr[okb] - LY[okb, t]).sum()), int(okb.sum()))
        g5 = (LP[:, t0] - LP[:, max(t0 - 5, 0)]) / min(5, t0)
        for name, pr in (("persistencia", LP[:, t0]), ("tendencia_5a", LP[:, t0] + h * g5)):
            okp = np.isfinite(LP[:, t]) & np.isfinite(pr)
            b[f"pop/{name}"] = (float(np.abs(pr[okp] - LP[okp, t]).sum()), int(okp.sum()))
        for name, pr in validate.conflict_baselines(t0, h).items():
            okc = np.isfinite(CONF[:, t]) & np.isfinite(pr)
            b[f"conf/{name}"] = (float(((np.clip(pr[okc], 1e-4, 1 - 1e-4) - CONF[okc, t]) ** 2).sum()), int(okc.sum()))
        okr = np.isfinite(RR[:, t]) & np.isfinite(RR[:, t0])
        b["rr/persistencia"] = (float(((RR[okr, t0] - RR[okr, t]) ** 2).sum()), int(okr.sum()))
        okd = np.isfinite(DEBT[:, t]) & np.isfinite(DEBT[:, t0]) & (DEBT[:, t] < 3)
        b["debt/persistencia"] = (float(((DEBT[okd, t0] - DEBT[okd, t]) ** 2).sum()), int(okd.sum()))
        okm = np.isfinite(DEMO[:, t]) & np.isfinite(DEMO[:, t0])
        b["demo/persistencia"] = (float(((np.clip(DEMO[okm, t0], 0.02, 0.98) - DEMO[okm, t]) ** 2).sum()),
                                  int(okm.sum()))
        g5e = (LPE[:, t0] - LPE[:, max(t0 - 5, 0)]) / 5
        for name, pr in (("persistencia", LPE[:, t0]), ("tendencia_5a", LPE[:, t0] + h * g5e)):
            oke = np.isfinite(LPE[:, t]) & np.isfinite(pr)
            if oke.any():
                b[f"pe/{name}"] = (float(np.abs(pr[oke] - LPE[oke, t]).sum()), int(oke.sum()))
        if np.isfinite(PSI[t]) and np.isfinite(PSI[t0]):
            b["psi/persistencia"] = (float(abs(PSI[t0] - PSI[t])), 1)
            b["psi/media_10a"] = (float(abs(np.nanmean(PSI[max(20, t0 - 9):t0 + 1]) - PSI[t])), 1)
        if np.isfinite(GMST[t]):
            slope = np.polyfit(np.arange(10), GMST[t0 - 9:t0 + 1], 1)[0]
            b["temp/persistencia"] = (float(abs(GMST[t0] - GMST[t])), 1)
            b["temp/tendencia_10a"] = (float(abs(GMST[t0] + slope * h - GMST[t])), 1)
        if np.isfinite(GOBS[t]):
            slope = np.polyfit(np.arange(10), GOBS[t0 - 9:t0 + 1], 1)[0]
            b["gmst/persistencia"] = (float(abs(GOBS[t0] - GOBS[t])), 1)
            b["gmst/tendencia_10a"] = (float(abs(GOBS[t0] + slope * h - GOBS[t])), 1)
            b["gmst/media_10a"] = (float(abs(np.mean(GOBS[t0 - 9:t0 + 1]) - GOBS[t])), 1)
        for kind, tr in LFOOD.items():
            g5f = (tr[:, t0] - tr[:, t0 - 5]) / 5
            for name, pr in (("persistencia", tr[:, t0]), ("tendencia_5a", tr[:, t0] + h * g5f)):
                okf = np.isfinite(tr[:, t]) & np.isfinite(pr)
                if okf.any():
                    b[f"{kind}/{name}"] = (float(np.abs(pr[okf] - tr[okf, t]).sum()), int(okf.sum()))
        if np.isfinite(LCO2[t]):
            g5c = (LCO2[t0] - LCO2[t0 - 5]) / 5
            b["co2/persistencia"] = (float(abs(LCO2[t0] - LCO2[t])), 1)
            b["co2/tendencia_5a"] = (float(abs(LCO2[t0] + h * g5c - LCO2[t])), 1)
        out["baselines"][str(h)] = b
    return out


def _metric(kind, cell):
    if kind in CRPS_KINDS:
        return cell["crps"], cell["n"]
    if kind in ("conf", "demo"):
        return cell["brier"], cell["n"]
    if kind in ("rr", "debt"):
        return cell["sse"], cell["n"]
    return cell["abs"], cell["n"]


def _finish(kind, s, n):
    if n == 0:
        return float("nan")
    return float(np.sqrt(s / n)) if kind in ("rr", "debt") else float(s / n)


def summarise(parts: list, n_boot: int = 1000, seed: int = 0) -> dict:
    """Métricas agregadas, cociente frente al Atlas con IC95 por remuestreo de orígenes, y referencias."""
    from world4.validate.metrics import auc
    parts = sorted(parts, key=lambda p: p["t0"])
    modes = list(parts[0]["modes"])
    rng = np.random.default_rng(seed)
    out = {"modes": {}, "ratio_vs_observed": {}, "baselines": {}, "best_baseline": {}}
    for kind in KINDS:
        for h in map(str, HORIZONS):
            cells = {m: [(p["t0"],) + _metric(kind, p["modes"][m][h][kind]) for p in parts
                         if h in p["modes"][m] and kind in p["modes"][m][h]] for m in modes}
            for m in modes:
                s = sum(c[1] for c in cells[m])
                n = sum(c[2] for c in cells[m])
                d = {"value": _finish(kind, s, n), "n": n, "metric": LOWER_IS_BETTER[kind]}
                if kind in CRPS_KINDS:
                    d["coverage80"] = float(sum(p["modes"][m][h][kind]["cover"] for p in parts
                                            if h in p["modes"][m] and kind in p["modes"][m][h])
                                            / max(n, 1))
                if kind == "conf":
                    pr = np.concatenate([p["modes"][m][h]["conf"]["pred"] for p in parts if h in p["modes"][m]])
                    tr = np.concatenate([p["modes"][m][h]["conf"]["truth"] for p in parts if h in p["modes"][m]])
                    d["auc"] = auc(tr, pr)
                out["modes"].setdefault(m, {}).setdefault(kind, {})[h] = d
            base = np.array([[c[1], c[2]] for c in cells.get("observed", [])])
            for m in modes:
                if m == "observed":
                    continue
                arr = np.array([[c[1], c[2]] for c in cells[m]])
                if len(arr) != len(base) or len(arr) == 0:
                    continue
                b0 = _finish(kind, base[:, 0].sum(), base[:, 1].sum())
                if not b0 > 0:                       # la población observada no tiene error: no hay cociente
                    continue
                r0 = _finish(kind, arr[:, 0].sum(), arr[:, 1].sum()) / b0
                bs = []
                for _ in range(n_boot):
                    k = rng.integers(0, len(arr), len(arr))
                    a = _finish(kind, arr[k, 0].sum(), arr[k, 1].sum())
                    b = _finish(kind, base[k, 0].sum(), base[k, 1].sum())
                    bs.append(a / b if b > 0 else np.nan)
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                out["ratio_vs_observed"].setdefault(m, {}).setdefault(kind, {})[h] = \
                    {"ratio": float(r0), "ci95": [float(lo), float(hi)]}
            bl = {}
            for p in parts:
                for key, (s, n) in p["baselines"].get(h, {}).items():
                    if key.startswith(kind + "/"):
                        a = bl.setdefault(key.split("/", 1)[1], [0.0, 0])
                        a[0] += s
                        a[1] += n
            fin = {k: (float(v[0] / v[1]) if kind in CRPS_KINDS + ("conf", "demo", "temp")
                       else float(np.sqrt(v[0] / v[1]))) for k, v in bl.items() if v[1]}
            out["baselines"].setdefault(kind, {})[h] = fin
            if fin:
                best = min(fin, key=fin.get)
                out["best_baseline"].setdefault(kind, {})[h] = {"name": best, "value": fin[best]}
    return out


def _chunk(args):
    origins, R, modes = args
    return [one_origin(t0, R=R, modes=modes) for t0 in origins]


def run(R: int = 32, workers: int = 6, origins=ORIGINS, modes=MODES, log=print) -> dict:
    t = time.time()
    chunks = [list(origins)[k::workers] for k in range(workers)]
    with get_context("spawn").Pool(workers) as pool:
        parts = [p for ch in pool.map(_chunk, [(c, R, modes) for c in chunks if c]) for p in ch]
    summ = summarise(parts)
    summ["meta"] = {"origins": [1950 + min(origins), 1950 + max(origins)], "R": R, "modes": list(modes),
                    "seconds": time.time() - t}
    log(f"W3 rolling: {len(parts)} origins x {len(modes)} modes in {time.time() - t:.0f} s")
    return summ


MODEL_MODE = "f5"                 # el modelo vigente: F1 población + F2 recursos + F3 energía + F4 clima + F5 alimentos
# W5 por capa: (modo nuevo, modo sin la capa, variables que la capa NO debe mover)
LAYERS = {"F1": ("endogenous", "observed", ("conf", "rr", "debt", "demo", "temp")),
          "F2": ("f2", "endogenous", ("conf", "debt", "demo", "temp", "pop")),
          "F3": ("f3", "f2", ("conf", "debt", "demo", "temp", "pop")),
          "F4": ("f4", "f3", ("debt", "demo", "pop", "pe", "psi")),
          "F5": ("f5", "f4", ("debt", "demo", "pe", "psi"))}


def verdict_w3(s: dict, tol: float = 0.02, mode: str = MODEL_MODE) -> dict:
    """W3: el CRPS del PIB pc del modelo vigente no peor que el del Atlas en más de ``tol`` a 1, 5 y
    10 años; las rentas de recursos a 10 años no peor (salida de F2); >= 4 de 6 variables baten a su
    mejor referencia a 10 años; y el test de fuga de F1: con la población observada (el mutante) el
    PIB pc debería puntuar MEJOR que con la endógena."""
    g = s["ratio_vs_observed"][mode]["gdp"]
    worst = max(v["ratio"] for v in g.values())
    rr10 = s["ratio_vs_observed"][mode]["rr"]["10"]["ratio"]
    beat = {}
    for kind in ("gdp", "conf", "rr", "debt", "demo", "temp"):
        mv = s["modes"][mode][kind]["10"]["value"]
        bb = s["best_baseline"].get(kind, {}).get("10")
        if bb is not None and mv == mv:
            beat[kind] = bool(mv < bb["value"])
    g1 = s["ratio_vs_observed"]["endogenous"]["gdp"]
    return {"mode": mode, "gdp_crps_ratio": {h: v["ratio"] for h, v in g.items()}, "worst_ratio": worst,
            "ok_2pct": worst <= 1 + tol, "rr_rmse_ratio_10y": rr10, "rr_ok": rr10 <= 1 + tol,
            "leak_detected": bool(any(v["ratio"] > 1 for v in g1.values())),
            "beats_best_baseline_10y": beat, "n_beaten_10y": int(sum(beat.values()))}


def verdict_w5(s: dict, tol: float = 0.02) -> dict:
    """W5: cada capa nueva no mueve las variables que no le tocan fuera de su banda. Banda =
    max(``tol``, lo que se mueve la métrica sólo por cambiar la semilla del ensemble)."""
    rows = {}
    for layer, (new, base, kinds) in LAYERS.items():
        if new not in s["modes"] or base not in s["modes"]:
            continue
        for kind in kinds:
            for h in map(str, HORIZONS):
                a_, b_ = s["modes"][new][kind][h]["value"], s["modes"][base][kind][h]["value"]
                if not (a_ == a_ and b_ == b_ and b_ > 0):
                    continue
                r = a_ / b_
                seed = s["ratio_vs_observed"]["observed_seed2"].get(kind, {}).get(h)
                noise = abs(seed["ratio"] - 1) if seed else 0.0
                band = max(tol, noise)
                rows[f"{layer}:{kind}/{h}"] = {"ratio": r, "band": band, "inside": bool(abs(r - 1) <= band)}
    return {"cells": rows, "outside": [k for k, v in rows.items() if not v["inside"]]}


if __name__ == "__main__":
    import sys
    from pathlib import Path
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs/w3.json")
    s = run()
    s["w3"], s["w5"] = verdict_w3(s), verdict_w5(s)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(s, indent=1), encoding="utf-8")
    print(json.dumps({"w3": s["w3"], "w5_outside": s["w5"]["outside"]}, indent=1))
