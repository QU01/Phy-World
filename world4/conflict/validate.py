"""Validación fuera de muestra con origen rodante: para cada origen T (diciembre de cada año) se entrena con
datos <= T, se pronostican los 12 meses siguientes en las 64 818 celdas y se puntúa contra lo observado.

La verdad de 2020-2025 es UCDP GED 26.1 (final); la de 2026 (origen 2025-12, enero-septiembre) son los
eventos CANDIDATOS de UCDP: el único tramo realmente posterior a todo lo que existía cuando se fijaron los
datos del módulo. Defecto declarado: los predictores de los orígenes anteriores usan el GED final (revisado),
no lo que se había publicado en T (en T sólo existían los candidatos).

Subconjuntos: ``all`` (mundo), ``ame`` (África y Oriente Medio, Gleditsch-Ward 400-699: el dominio de ViEWS,
para comparar órdenes de magnitud con sus tablas) y ``onset`` (celdas sin ningún evento en ellas ni en sus 8
vecinas en los 24 meses previos: el "problema difícil" de Mueller y Rauh 2022).
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

from world4.conflict import benchmarks as BM
from world4.conflict import features as F
from world4.conflict import forecast as FC
from world4.conflict import metrics as M
from world4.conflict.grid import Events, Grid
from world4.data import conflict_grid as cdata

ORIGINS = ("2019-12", "2020-12", "2021-12", "2022-12", "2023-12", "2024-12", "2025-12")
MODELS = ("ensemble", "logistic", "hawkes")


def _subsets(ev: Events, grid: Grid, T: int) -> dict:
    hist = (ev.window("count", "any", T - 23, T + 1) > 0).sum(0).astype(float)
    quiet = (hist + grid.ring1 @ hist) == 0
    return {"all": np.ones(grid.n, bool), "ame": grid.africa_middle_east(), "onset": quiet}


def _mean(rows: list, keys=("brier", "log_loss", "auroc", "aupr", "crps")) -> dict:
    out = {}
    for k in keys:
        v = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        if v:
            out[k] = float(np.mean(v))
    for k in ("events", "expected", "n"):
        if rows and k in rows[0]:
            out[k] = float(np.sum([r[k] for r in rows]))
    return out


def run(origins=ORIGINS, *, horizon: int = FC.HORIZON, samples: int = 200, seed: int = 0,
        p_near: float = 0.25, p_far: float = 0.004, log=print) -> dict:
    grid = Grid()
    ev = Events(grid)
    Ts = [cdata.month_index(o) for o in origins]
    last = ev.n_months - 1
    log(f"sweep to {cdata.month_label(max(Ts))} ({len(Ts)} origins)")
    rows = F.build(ev, grid, until=max(Ts), snapshots=Ts, p_near=p_near, p_far=p_far, seed=seed, log=log)
    per = []                    # una fila por (origen, h, resultado, subconjunto, modelo)
    calib = {}                  # (modelo, resultado, h-grupo) -> intervalos sumados
    windows = []
    models_out = {}
    for T in Ts:
        log(f"origin {cdata.month_label(T)}: fit")
        model = FC.fit(ev, grid, rows, T, horizon=horizon, log=log)
        snap = rows["snapshots"][T]
        P = FC.predict(model, grid, snap, samples=samples, seed=seed)
        models_out[cdata.month_label(T)] = {"branching": model.branching(),
                                            "theta": {v: float(model.hawkes[v].theta) for v in model.hawkes}}
        subs = _subsets(ev, grid, T)
        truth_src = "candidate" if T + 1 > ev.last_final_month else "final"
        bdeath = BM.death_samples(ev, grid, T, samples=samples, seed=seed)
        rng = np.random.default_rng(seed + 7)
        for o in FC.OUTCOMES:
            bprob = BM.probabilities(ev, grid, T, o)
            for h in range(1, min(horizon, last - T) + 1):
                y = ev.window("count", o, T + h, T + h + 1)[0] > 0
                cands = {"ensemble": P["p"][o][h - 1], "logistic": P["p_logistic"][o][h - 1],
                         "hawkes": P["p_hawkes"][o][h - 1], **{f"bm_{k}": v for k, v in bprob.items()}}
                for sname, s in subs.items():
                    for mname, p in cands.items():
                        r = M.scores(p[s], y[s])
                        per.append({"origin": cdata.month_label(T), "h": h, "outcome": o, "subset": sname,
                                    "model": mname, "truth": truth_src, **r})
                for mname in MODELS:
                    key = f"{mname}|{o}|{'h1-3' if h <= 3 else 'h4-12'}"
                    c = M.calibration(cands[mname], y)
                    calib[key] = c if key not in calib else [[a + b for a, b in zip(x, z)] for x, z in zip(calib[key], c)]
                if o == "sb":
                    d = ev.window("deaths", "sb", T + h, T + h + 1)[0]
                    D = FC.death_samples(model, snap["X"], P["p"]["sb"][h - 1], h, rng, samples)
                    dcands = {"ensemble": D, **{f"bm_{k}": v for k, v in bdeath.items()}}
                    for sname, s in subs.items():
                        for mname, smp in dcands.items():
                            per.append({"origin": cdata.month_label(T), "h": h, "outcome": "sb_deaths",
                                        "subset": sname, "model": mname, "truth": truth_src,
                                        "crps": float(M.crps_samples(smp[s], d[s]).mean()),
                                        "observed_deaths": float(d[s].sum()), "n": int(s.sum())})
            if T + horizon <= last:
                y = ev.window("count", o, T + 1, T + horizon + 1).sum(0) > 0
                wc = {"ensemble": P["p_window"][o], "logistic": P["p_window_logistic"][o],
                      "hawkes": P["p_window_hawkes"][o], "bm_neighbors12": 1 - (1 - bprob["neighbors12"]) ** 12,
                      "bm_cell12": 1 - (1 - bprob["cell12"]) ** 12, "bm_persistence": bprob["persistence"]}
                for sname, s in subs.items():
                    for mname, p in wc.items():
                        windows.append({"origin": cdata.month_label(T), "outcome": o, "subset": sname,
                                        "model": mname, **M.scores(p[s], y[s])})
        log(f"origin {cdata.month_label(T)}: scored ({truth_src} truth)")
    return summarize(per, windows, calib, models_out, origins=origins, horizon=horizon, samples=samples,
                     versions=ev.versions)


def summarize(per, windows, calib, models_out, **meta) -> dict:
    summary = {}
    for o in FC.OUTCOMES + ("sb_deaths",):
        for s in ("all", "ame", "onset"):
            names = sorted({r["model"] for r in per if r["outcome"] == o and r["subset"] == s})
            for mname in names:
                sel = [r for r in per if r["outcome"] == o and r["subset"] == s and r["model"] == mname]
                summary.setdefault(o, {}).setdefault(s, {})[mname] = {
                    "all_h": _mean(sel), "h1": _mean([r for r in sel if r["h"] == 1]),
                    "h4-12": _mean([r for r in sel if r["h"] >= 4]),
                    "2026_candidate": _mean([r for r in sel if r["truth"] == "candidate"])}
    wsum = {}
    for o in FC.OUTCOMES:
        for s in ("all", "ame", "onset"):
            for mname in sorted({r["model"] for r in windows}):
                sel = [r for r in windows if r["outcome"] == o and r["subset"] == s and r["model"] == mname]
                if sel:
                    wsum.setdefault(o, {}).setdefault(s, {})[mname] = _mean(sel)
    return {"meta": {**{k: (list(v) if isinstance(v, tuple) else v) for k, v in meta.items()},
                     "run_date": date.today().isoformat(),
                     "metrics_note": "means over origin-months (Brier, log-loss, CRPS are pooled means; AUROC and AUPR "
                                     "are averaged per origin-month)"},
            "summary": summary, "window12": wsum, "calibration": calib, "hawkes": models_out, "per_origin_month": per}


def save(result: dict, path) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
