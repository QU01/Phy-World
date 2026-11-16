"""W13: el L2 contra su vara, con origen móvil y sin mirar el futuro.

Cada origen es un mes ``t0``; todo se estima con eventos anteriores y se puntúa contra el GED FINAL de los
meses ``t0 .. t0 + H - 1`` (el candidato de 2026 nunca es verdad de prueba). Se puntúan las muertes mensuales
(``best``, los tres tipos de violencia juntos) por país y por celda en los pasos ``STEPS``:

``crps_log``  CRPS del ensemble sobre ln(1 + muertes) (la escala en la que un mes de 10 y uno de 10 000 pesan
              parecido; es la métrica que decide);
``crps``      CRPS en muertes (la domina la guerra más grande);
``brier1``    Brier de «al menos una muerte»;
``brier25``   Brier de «25 muertes o más» (sólo países).

**Dos periodos, declarados antes de correr.** Los orígenes ``DEV`` (2005-2014) sirvieron para elegir la
forma del simulador; los ``TEST`` (2015-2024) se corrieron una vez con esa forma fijada y son los únicos
que cuentan para el veredicto. La diferencia con cada referencia lleva un IC95 por remuestreo de países
enteros (las celdas se agrupan por su país): las preguntas de un país no son independientes.

Fuga declarada ``D-L2-1``: los rasgos del origen usan el GED final, que en su día sólo existía como
candidato o no existía.
"""
from __future__ import annotations

import json
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from world4.validate import metrics

STEPS = {"country": (1, 3, 6, 12, 24, 36), "cell": (1, 3, 6, 12)}
METRICS = {"country": ("crps_log", "crps", "brier1", "brier25"), "cell": ("crps_log", "crps", "brier1")}
REFERENCES = ("zero", "last12", "hurdle")
MEMBERS = 64
STRIDE = 6                                         # meses entre orígenes
DEV_YEARS, TEST_YEARS = (2005, 2014), (2015, 2024)


def origins(P, years) -> list[int]:
    """Orígenes de enero y julio cuyo primer mes de pronóstico cae dentro del GED final."""
    out = [P.month(y, m) for y in range(years[0], years[1] + 1) for m in range(1, 13, STRIDE)]
    return [t for t in out if t <= P.last_final_month]


def _score(ens: np.ndarray, y: np.ndarray, level: str, steps) -> dict:
    """``ens`` (M, U, H), ``y`` (U, H) -> métrica -> (U, len(steps)) con NaN donde el paso no tiene verdad."""
    U = y.shape[0]
    out = {m: np.full((U, len(steps)), np.nan) for m in METRICS[level]}
    for j, s in enumerate(steps):
        if s > y.shape[1] or s > ens.shape[2]:
            continue
        e, o = ens[:, :, s - 1], y[:, s - 1]
        out["crps_log"][:, j] = metrics.crps_ensemble(np.log1p(e), np.log1p(o))
        out["crps"][:, j] = metrics.crps_ensemble(e, o)
        out["brier1"][:, j] = (np.mean(e >= 1, 0) - (o >= 1)) ** 2
        if "brier25" in out:
            out["brier25"][:, j] = (np.mean(e >= 25, 0) - (o >= 25)) ** 2
    return out


def one_origin(args) -> dict:
    """Todas las variantes desde un origen. ``args`` = (región, t0, miembros, semilla, opciones)."""
    region, t0, members, seed, opt = args
    from world4.core.l2 import agents, baseline, forecast, panel
    P = panel.load(region)
    Q = P.truncated(t0)                              # nada de t0 en adelante puede entrar en el ajuste
    H = {lv: max(s for s in STEPS[lv]) for lv in STEPS}
    truth_hi = P.last_final_month + 1
    res = {"t0": int(t0), "models": {}}
    y = {lv: P.deaths(lv)[:, t0:min(t0 + H[lv], truth_hi)] for lv in STEPS}
    ens = {}
    run = forecast.run(Q, t0, H["country"], members=members, seed=seed, l1=opt.get("l1", False),
                       foot=opt.get("foot"), births=opt.get("births", True), displacement=False,
                       **opt.get("model", {}))
    ens["agents"] = {"country": run.deaths("country"), "cell": run.deaths("cell")[:, :, :H["cell"]]}
    res["foot"] = run.model.foot
    for lv in STEPS:
        m = baseline.fit(Q, lv, t0)
        ens.setdefault("hurdle", {})[lv] = baseline.forecast(Q, m, t0, H[lv], members, seed)
        ens.setdefault("last12", {})[lv] = baseline.last12(Q, lv, t0, H[lv])
        ens.setdefault("zero", {})[lv] = baseline.zero(Q, lv, H[lv])
    for name, by_level in ens.items():
        res["models"][name] = {lv: _score(by_level[lv], y[lv], lv, STEPS[lv]) for lv in STEPS}
    res["observed"] = {lv: [float(np.sum(y[lv][:, :s])) if s <= y[lv].shape[1] else None for s in STEPS[lv]]
                       for lv in STEPS}
    return res


def _cluster_ci(diff: np.ndarray, cluster: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """IC95 de la media de ``diff`` (n,) remuestreando grupos enteros."""
    ok = np.isfinite(diff)
    diff, cluster = diff[ok], cluster[ok]
    ug = np.unique(cluster)
    if len(ug) < 2:
        return [float("nan"), float("nan")]
    sums = np.array([diff[cluster == g].sum() for g in ug])
    cnts = np.array([(cluster == g).sum() for g in ug])
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(ug), (n_boot, len(ug)))
    means = sums[pick].sum(1) / np.maximum(cnts[pick].sum(1), 1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def summarise(P, parts: list, model: str = "agents") -> dict:
    """Media por nivel, paso y métrica de cada variante, y ``model`` menos cada referencia con su IC."""
    out = {"origins": [P.label(p["t0"]) for p in parts], "n_origins": len(parts), "levels": {}}
    cluster = {"country": np.arange(P.K), "cell": P.cell_country}
    for lv in STEPS:
        L = {}
        for j, s in enumerate(STEPS[lv]):
            row = {}
            for metric in METRICS[lv]:
                stack = {name: np.stack([p["models"][name][lv][metric][:, j] for p in parts])
                         for name in parts[0]["models"]}                       # (origins, U)
                n = int(np.isfinite(stack[model]).sum())
                cell = {"n": n, "mean": {k: (float(np.nanmean(v)) if n else None) for k, v in stack.items()}}
                if n:
                    cl = np.broadcast_to(cluster[lv][None], stack[model].shape).reshape(-1)
                    for ref in stack:
                        if ref == model:
                            continue
                        d = (stack[model] - stack[ref]).reshape(-1)
                        cell[f"minus_{ref}"] = {"mean": float(np.nanmean(d)), "ci95": _cluster_ci(d, cl)}
                row[metric] = cell
            L[str(s)] = row
        out["levels"][lv] = L
    return out


def verdict(summary: dict, metric: str = "crps_log") -> dict:
    """Por nivel y paso: ¿el simulador es mejor que TODAS las referencias con el IC entero bajo cero?"""
    out = {}
    for lv, L in summary["levels"].items():
        for s, row in L.items():
            cell = row[metric]
            if not cell["n"]:
                continue
            refs = {r: cell.get(f"minus_{r}") for r in REFERENCES if f"minus_{r}" in cell}
            beats = [r for r, d in refs.items() if d["ci95"][1] < 0]
            loses = [r for r, d in refs.items() if d["ci95"][0] > 0]
            best_ref = min(refs, key=lambda r: cell["mean"][r])
            out[f"{lv}/{s}"] = {"beats": beats, "loses_to": loses, "best_reference": best_ref,
                                "skill_vs_best": 1.0 - cell["mean"]["agents"] / max(cell["mean"][best_ref], 1e-12),
                                "forecast_grade": len(beats) == len(refs)}
    return out


def run(region: str = "mideast", period: str = "test", members: int = MEMBERS, workers: int = 4, opt=None,
        log=print) -> dict:
    from world4.core.l2 import panel
    P = panel.load(region)
    opt = dict(opt or {})
    ts = origins(P, DEV_YEARS if period == "dev" else TEST_YEARS)
    t = time.time()
    jobs = [(region, t0, members, 1000 + t0, opt) for t0 in ts]
    if workers > 1:
        with get_context("spawn").Pool(workers) as pool:
            parts = pool.map(one_origin, jobs, chunksize=1)
    else:
        parts = [one_origin(j) for j in jobs]
    s = summarise(P, parts)
    s.update({"region": region, "period": period, "members": members, "options": opt, "seconds": time.time() - t,
              "unit": "deaths per month (UCDP best estimate, all three types of violence)",
              "verdict": verdict(s), "exploratory": True,
              "leaks_declared": ["D-L2-1: features at the origin use the final GED, which at that date existed "
                                 "only as candidate events or not at all"]})
    for lv in STEPS:
        for st in STEPS[lv]:
            c = s["levels"][lv][str(st)]["crps_log"]
            if c["n"]:
                log(f"  {lv:7s} step {st:2d} n={c['n']:6d} crps_log " + " ".join(
                    f"{k} {v:.4f}" for k, v in c["mean"].items()))
    return s


def write(summary: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- desplazamiento
DISP_ORIGIN_YEARS = (2012, 2015, 2018, 2021, 2023)
DISP_YEARS_AHEAD = (1, 2, 3)


def _disp_origin(args) -> dict:
    """Stocks de fin de año a 1, 2 y 3 años desde enero de ``year``: el modelo movido por las muertes
    OBSERVADAS (mide sólo la respuesta), por las SIMULADAS (el pronóstico entero) y la persistencia."""
    region, year, members, seed = args
    from world4.core.l2 import displacement as dm
    from world4.core.l2 import forecast, panel
    P = panel.load(region)
    t0 = P.month(year, 1)
    Q = P.truncated(t0)
    th = dm.fit(Q, t0, maxiter=200)
    H = 12 * max(DISP_YEARS_AHEAD)
    site = P.site_deaths()                                  # la verdad, con los sitios del panel completo
    Hd = min(H, P.T - t0)
    truth = np.stack([dm.cell_deaths(P, site[:, t0 + h], P.pop_at(t0 + h, "cell")) for h in range(Hd)], -1)
    driven = dm.simulate(Q, th, t0, by_cell=truth[None])
    full = forecast.run(Q, t0, H, members=members, seed=seed, disp_params=th).displaced
    j0 = year - 1 - int(P.years[0])
    last = {"ref": np.nan_to_num(P.ref_stock[:, :, j0].sum(1)), "idp": np.nan_to_num(P.idp_stock[:, j0])}
    out = {"year": int(year), "params": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in th.items()
                                         if k != "x"}, "rows": []}
    for a in DISP_YEARS_AHEAD:
        j, h = j0 + a, 12 * a - 1
        if j >= len(P.years):
            continue
        obs = {"ref": P.ref_stock[:, :, j].sum(1), "idp": P.idp_stock[:, j]}
        for what in ("ref", "idp"):
            for k in range(P.K):
                o = float(obs[what][k])
                if not np.isfinite(o) or (what == "idp" and o <= 0 and last[what][k] <= 0):
                    continue                                   # sin registro de desplazados internos
                ens = full[what][:, k, h]
                out["rows"].append({"years_ahead": a, "what": what, "country": P.iso3[k], "observed": o,
                                    "persistence": float(last[what][k]),
                                    "driven": float(driven[what][0, k, h]) if h < Hd else None,
                                    "p10": float(np.percentile(ens, 10)), "p50": float(np.median(ens)),
                                    "p90": float(np.percentile(ens, 90))})
    return out


def displacement_run(region: str = "mideast", members: int = 48, workers: int = 4, log=print) -> dict:
    """Error de los stocks de desplazados fuera de muestra, en asinh(personas / 10 000) (≈ error logarítmico
    para stocks grandes), contra la persistencia del último stock publicado."""
    from world4.core.l2 import displacement as dm
    t = time.time()
    jobs = [(region, y, members, 500 + y) for y in DISP_ORIGIN_YEARS]
    if workers > 1:
        with get_context("spawn").Pool(min(workers, len(jobs))) as pool:
            parts = pool.map(_disp_origin, jobs, chunksize=1)
    else:
        parts = [_disp_origin(j) for j in jobs]
    f = lambda v: np.arcsinh(np.asarray(v, float) / dm.SCALE)  # noqa: E731
    rows = [r for p in parts for r in p["rows"]]
    table = {}
    for a in DISP_YEARS_AHEAD:
        for what in ("ref", "idp"):
            sel = [r for r in rows if r["years_ahead"] == a and r["what"] == what and r["driven"] is not None]
            if not sel:
                continue
            o = f([r["observed"] for r in sel])
            cell = {"n": len(sel)}
            for m, key in (("persistence", "persistence"), ("driven_by_observed_deaths", "driven"),
                           ("full_forecast_p50", "p50")):
                cell[m] = float(np.mean(np.abs(f([r[key] for r in sel]) - o)))
            cell["coverage_p10_p90"] = float(np.mean([r["p10"] <= r["observed"] <= r["p90"] for r in sel]))
            table[f"{what}/{a}"] = cell
            log(f"  {what} +{a}y n={cell['n']:3d} |error| persistence {cell['persistence']:.3f} driven "
                f"{cell['driven_by_observed_deaths']:.3f} forecast {cell['full_forecast_p50']:.3f}; p10-p90 covers "
                f"{cell['coverage_p10_p90']:.2f}")
    return {"region": region, "origins": list(DISP_ORIGIN_YEARS), "members": members, "exploratory": True,
            "unit": "mean absolute error of asinh(persons / 10 000) of the end-of-year stock; coverage is a share",
            "table": table, "rows": rows, "params": {str(p["year"]): p["params"] for p in parts},
            "seconds": time.time() - t,
            "note": "ref = refugees and asylum seekers by origin (UNHCR); idp = internally displaced, the larger of "
                    "UNHCR and IDMC; stocks of year y - 1 are taken as known in January of y"}
