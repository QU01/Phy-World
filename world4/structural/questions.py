"""Preguntas históricas autoevaluables desde el panel (W9a) y su contraparte numérica (W8).

Una pregunta histórica se "hace" el 1 de julio del año Y sobre la ventana de años Y+1..Y+h
(h = 1..5) para un país. Todo lo que la responde se sabía en esa fecha:

* el L1 pronostica desde el origen Y-3 (PIB de la PWT publicado con 3 años de retraso);
* la tasa base y la referencia condicionada sólo ven años publicados (``known_until``);
* el recalibrador de la pregunta sólo se entrena con preguntas cuyo desenlace ya estaba publicado.

Las preguntas se agrupan por k2 = (fin de ventana - origen) = 3 + h, que es lo que el modo
vivo comparte con ellas (desde el origen 2019, una pregunta de 2026 a 2027 tiene k2 = 8).
"""
from __future__ import annotations

import json
import time
from datetime import date

import numpy as np

from world4.structural import engine, observed, recal, store

H_RANGE = (1, 2, 3, 4, 5)
ORIGIN_LAG = 3
FIRST_TEST_YEAR = 1990          # antes sólo entrenan recalibradores
MIN_MEMBERS = 8


def records_path():
    return store.path().with_name(f"records_v{store.STORE_VERSION}.npz")


def generate(log=print) -> dict:
    """Todas las preguntas históricas con su probabilidad cruda y su desenlace."""
    t = time.time()
    P = observed.panel()
    N = len(P["iso3"])
    ev_rows, num_rows = [], []
    for Y in range(1970 + ORIGIN_LAG, engine.L1_LAST_DATA_YEAR):
        o = Y - ORIGIN_LAG
        ens = store.get(o)
        for h in H_RANGE:
            y1, y2 = Y + 1, Y + h
            if y2 > engine.L1_LAST_DATA_YEAR:
                continue
            k1, k2 = y1 - o, y2 - o
            for ei, ev in enumerate(engine.EVENTS):
                for i in range(N):
                    m = engine.event_members(ens, ev, i, k1, k2)
                    if np.isfinite(m).sum() < MIN_MEMBERS:
                        continue
                    y = observed.event_outcome(ev, i, y1, y2)
                    if not np.isfinite(y):
                        continue
                    ev_rows.append((ei, i, Y, h, k2, y2, float(np.nanmean(m)), y))
            for vi, var in enumerate(engine.NUMERIC):
                for i in range(N):
                    m = engine.numeric_members(ens, var, i, k2)
                    if np.isfinite(m).sum() < MIN_MEMBERS:
                        continue
                    y = observed.numeric_outcome(var, i, y2)
                    if not np.isfinite(y):
                        continue
                    pit = float(recal.ens_cdf(m, y)[0])
                    num_rows.append((vi, i, Y, h, k2, y2, pit, y, *np.nanpercentile(m, [10, 50, 90])))
    ev = np.array(ev_rows, float).reshape(-1, 8)
    nm = np.array(num_rows, float).reshape(-1, 11)
    out = {"events": ev, "numeric": nm,
           "event_names": list(engine.EVENTS), "numeric_names": list(engine.NUMERIC)}
    np.savez_compressed(records_path(), events=ev, numeric=nm,
                        names=np.frombuffer(json.dumps({"events": out["event_names"],
                                                        "numeric": out["numeric_names"]}).encode(), np.uint8))
    log(f"records: {len(ev)} event questions, {len(nm)} numeric, {time.time() - t:.0f} s -> {records_path()}")
    return out


_REC = None


def records() -> dict:
    global _REC
    if _REC is None:
        p = records_path()
        if not p.is_file():
            raise store.StoreMissing(f"no question records at {p}: python world4_cli.py --build-store")
        z = np.load(p)
        names = json.loads(bytes(z["names"]).decode())
        _REC = {"events": z["events"], "numeric": z["numeric"],
                "event_names": names["events"], "numeric_names": names["numeric"]}
    return _REC


def event_calibrator(event: str, k2: int, as_of: date) -> recal.EventCalibrator:
    R = records()
    ei = R["event_names"].index(event)
    until = observed.known_until(engine.EVENTS[event]["source"], as_of)
    E = R["events"]
    m = (E[:, 0] == ei) & (E[:, 4] == k2) & (E[:, 5] <= until)
    return recal.EventCalibrator(E[m, 6], E[m, 7])


# ------------------------------------------------------------------ F7: recalibradores numéricos
# Candidatos declarados (los dos remedios que dejó F0.5: ventana móvil y seguir el sesgo):
#   all       todos los PIT publicados de ese horizonte (el recalibrador de F0.5)
#   w5        sólo los desenlaces de los últimos 5 años, mismo horizonte
#   w5pool    los de los últimos 5 años de todos los horizontes de la variable
#   bias3w10  residuo estandarizado corregido por el sesgo de los últimos 3 años (``recal.ZCalibrator``),
#             con los desenlaces de los últimos 10 años
# En cada fecha se ELIGE uno: el de menor distancia KS sobre los desenlaces de los SELECT_YEARS años ya
# publicados, transformados cada uno con el recalibrador que existía en SU fecha (selección anidada: el
# periodo de prueba de W8 no decide el método).
CANDIDATES = ("all", "w5", "w5pool", "bias3w10")
SELECT_YEARS = 8
BIAS_YEARS, BIAS_TRAIN_YEARS = 3, 10


def _numeric_rows(variable: str):
    R = records()
    vi = R["numeric_names"].index(variable)
    X = R["numeric"]
    return X[X[:, 0] == vi]


def _z(rows):
    s = np.maximum((rows[:, 10] - rows[:, 8]) / recal.Z_SCALE, 1e-6)
    return (rows[:, 7] - rows[:, 9]) / s


_BIAS: dict = {}


def recent_bias(variable: str, until: int) -> float:
    key = (variable, until)
    if key not in _BIAS:
        A = _numeric_rows(variable)
        r = A[(A[:, 5] <= until) & (A[:, 5] > until - BIAS_YEARS)]
        _BIAS[key] = float(np.median(_z(r))) if len(r) >= recal.MIN_PLATT else 0.0
    return _BIAS[key]


_CAND: dict = {}


def candidate_calibrator(variable: str, k2: int, as_of: date, cand: str):
    until = observed.known_until(engine.NUMERIC[variable]["source"], as_of)
    key = (variable, k2, until, cand)
    if key in _CAND:
        return _CAND[key]
    A = _numeric_rows(variable)
    same = A[A[:, 4] == k2]
    if cand == "all":
        cal = recal.NumericCalibrator(same[same[:, 5] <= until, 6])
    elif cand == "w5":
        cal = recal.NumericCalibrator(same[(same[:, 5] <= until) & (same[:, 5] > until - 5), 6])
    elif cand == "w5pool":
        cal = recal.NumericCalibrator(A[(A[:, 5] <= until) & (A[:, 5] > until - 5), 6])
    elif cand == "bias3w10":
        tr = same[(same[:, 5] <= until) & (same[:, 5] > until - BIAS_TRAIN_YEARS)]
        zc = [_z(tr[tr[:, 2] == Yq]) - recent_bias(variable, observed.known_until(engine.NUMERIC[variable]["source"],
                                                                                   date(int(Yq), 7, 1)))
              for Yq in np.unique(tr[:, 2])]
        cal = recal.ZCalibrator(np.concatenate(zc) if zc else [], recent_bias(variable, until))
    else:
        raise KeyError(cand)
    _CAND[key] = cal
    return cal


def transformed_pit(cal, rows) -> np.ndarray:
    """PIT recalibrado de filas de ``records`` con el recalibrador ``cal``."""
    if isinstance(cal, recal.ZCalibrator):
        return np.asarray(cal.cdf_z(_z(rows)), float)
    return np.asarray(cal.C(rows[:, 6]), float)


_PIT: dict = {}


def _candidate_pits(variable: str, k2: int, cand: str):
    """(filas, PIT recalibrado) de todas las preguntas de ese horizonte, cada una con el candidato de su fecha."""
    key = (variable, k2, cand)
    if key not in _PIT:
        rows = _numeric_rows(variable)
        rows = rows[rows[:, 4] == k2]
        u = np.full(len(rows), np.nan)
        for Y in np.unique(rows[:, 2]):
            m = rows[:, 2] == Y
            u[m] = transformed_pit(candidate_calibrator(variable, k2, date(int(Y), 7, 1), cand), rows[m])
        _PIT[key] = (rows, u)
    return _PIT[key]


def select_candidate(variable: str, k2: int, as_of: date) -> tuple[str, dict]:
    until = observed.known_until(engine.NUMERIC[variable]["source"], as_of)
    scores = {}
    for c in CANDIDATES:
        rows, u = _candidate_pits(variable, k2, c)
        m = (rows[:, 5] <= until) & (rows[:, 5] > until - SELECT_YEARS) & np.isfinite(u)
        if m.sum() >= recal.MIN_PLATT:
            scores[c] = recal.ks_distance(u[m])
    if not scores:
        return "all", scores
    return min(scores, key=lambda c: (scores[c], CANDIDATES.index(c))), scores


def numeric_calibrator(variable: str, k2: int, as_of: date):
    """El recalibrador de F7 para (variable, horizonte) con lo publicado en ``as_of``."""
    cand, scores = select_candidate(variable, k2, as_of)
    cal = candidate_calibrator(variable, k2, as_of, cand)
    cal.selected, cal.selection_ks = cand, scores
    return cal


def numeric_calibrator_f05(variable: str, k2: int, as_of: date) -> recal.NumericCalibrator:
    """El de F0.5 (todos los PIT publicados): se conserva para la ablación de W8."""
    return candidate_calibrator(variable, k2, as_of, "all")


# ------------------------------------------------------------------ evaluación W9a / W8
def _cluster_ci(diff: np.ndarray, groups: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    means = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug))
        sel = np.concatenate([idx[g] for g in pick])
        means.append(float(diff[sel].mean()))
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def evaluate_w9a(log=print) -> dict:
    R = records()
    E = R["events"]
    test = E[E[:, 2] >= FIRST_TEST_YEAR]
    out = {"first_test_year": FIRST_TEST_YEAR, "origin_lag": ORIGIN_LAG, "events": {}, "notes": [
        "questions asked on 1 July of year Y about years Y+1..Y+h, origin Y-3, everything known at that date",
        "clusters for the CI are countries (questions of one country are not independent)",
        "ViEWS and PITF are NOT reported yet: no access in F0.5 (W9a cannot be GREEN without them)"]}
    cache_mk, cache_cal = {}, {}
    all_rows = []
    for ei, ev in enumerate(R["event_names"]):
        rows = test[test[:, 0] == ei]
        p_w4, p_raw, p_base, p_mk, y, grp = [], [], [], [], [], []
        for (_, i, Y, h, k2, y2, praw, yy) in rows:
            i, Y, h, k2, y2 = int(i), int(Y), int(h), int(k2), int(y2)
            as_of = date(Y, 7, 1)
            key = (ev, k2, Y)
            if key not in cache_cal:
                cache_cal[key] = event_calibrator(ev, k2, as_of)
            br = observed.base_rate(ev, i, as_of, Y + 1, y2)
            if br["status"] != "ok":
                continue
            z = observed.zones(observed.known_until("pwt", as_of))[i]
            st_key = (ev, Y, h, z, _state(ev, i, as_of))
            if st_key not in cache_mk:
                cache_mk[st_key] = observed.markov_rate(ev, i, as_of, Y + 1, y2)
            p_w4.append(float(cache_cal[key](praw))); p_raw.append(praw)
            p_base.append(br["p_window"]); p_mk.append(cache_mk[st_key]); y.append(yy); grp.append(i)
        arr = {k: np.asarray(v, float) for k, v in
               dict(world4=p_w4, atlas_raw=p_raw, base_rate=p_base, markov=p_mk).items()}
        yv, gv = np.asarray(y, float), np.asarray(grp)
        ok = np.isfinite(arr["markov"])
        brier = {k: float(np.mean((v - yv) ** 2)) for k, v in arr.items() if k != "markov"}
        brier["markov"] = float(np.mean((arr["markov"][ok] - yv[ok]) ** 2))
        d_base = (arr["world4"] - yv) ** 2 - (arr["base_rate"] - yv) ** 2
        d_mk = ((arr["world4"] - yv) ** 2 - (arr["markov"] - yv) ** 2)[ok]
        out["events"][ev] = {
            "n": int(len(yv)), "base_rate_of_event": float(yv.mean()), "brier": brier,
            "delta_vs_base_rate": float(d_base.mean()), "delta_vs_base_rate_ci95": _cluster_ci(d_base, gv),
            "delta_vs_markov": float(d_mk.mean()), "delta_vs_markov_ci95": _cluster_ci(d_mk, gv[ok]),
            "what": engine.EVENTS[ev]["what"],
        }
        all_rows.append((d_base, gv + 1000 * ei))
        log(f"  {ev:10s} n={len(yv):6d} Brier world4 {brier['world4']:.4f} raw {brier['atlas_raw']:.4f} "
            f"base {brier['base_rate']:.4f} markov {brier['markov']:.4f}")
    d = np.concatenate([a for a, _ in all_rows]); g = np.concatenate([b for _, b in all_rows])
    out["pooled_delta_vs_base_rate"] = float(d.mean())
    out["pooled_delta_vs_base_rate_ci95"] = _cluster_ci(d, g)
    return out


def _state(ev, i, as_of):
    P = observed.panel()[ev][i]
    until = observed.known_until(engine.EVENTS[ev]["source"], as_of)
    if ev == "democracy":
        return float(P[until - 1950])
    seg = P[max(0, until - 1950 - 2): until - 1950 + 1]
    return float(np.any(seg > 0)) if np.all(np.isfinite(seg)) else np.nan


def clustered_null(u, year, n_sim: int = 400, seed: int = 0) -> dict:
    """Distancia KS que daría un pronosticador PERFECTAMENTE calibrado con la misma correlación dentro de
    cada año (choques comunes a todos los países) y el mismo número de preguntas por año: z = √ρ a_año +
    √(1-ρ) e, PIT = Φ(z). Es informativa: el listón de W8 (D <= 0,05) no cambia."""
    from scipy.stats import norm
    u = np.clip(np.asarray(u, float), 1e-4, 1 - 1e-4)
    z = norm.ppf(u)
    yrs, inv = np.unique(year, return_inverse=True)
    n_y = np.bincount(inv)
    means = np.bincount(inv, weights=z) / n_y
    within = np.sum((z - means[inv]) ** 2) / (len(z) - len(yrs))
    between = max(float(np.var(means, ddof=1) - within * np.mean(1.0 / n_y)), 0.0)
    rho = between / (between + within)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_sim):
        a = rng.standard_normal(len(yrs))[inv]
        d.append(recal.ks_distance(norm.cdf(np.sqrt(rho) * a + np.sqrt(1 - rho) * rng.standard_normal(len(z)))))
    return {"within_year_correlation": float(rho), "ks_d_null_p50_year_clustered": float(np.median(d)),
            "ks_d_null_p95_year_clustered": float(np.percentile(d, 95))}


def evaluate_w8(log=print) -> dict:
    """Cobertura p10-p90 y distancia KS del PIT fuera de muestra (preguntas de 1990 en adelante): crudo, el
    recalibrador de F0.5 y el de F7 (selección anidada). Se informan también los candidatos sueltos: esa
    tabla SÍ mira el periodo de prueba (es exploración, no decide)."""
    out = {"first_test_year": FIRST_TEST_YEAR, "method": "F7 nested selection among " + ", ".join(CANDIDATES)
           + f" by KS distance over the last {SELECT_YEARS} published years", "variables": {}}
    R = records()
    for var in R["numeric_names"]:
        out["variables"][var] = {}
        A = _numeric_rows(var)
        for k2 in sorted({int(v) for v in A[:, 4]}):
            rows = A[(A[:, 4] == k2) & (A[:, 2] >= FIRST_TEST_YEAR)]
            if not len(rows):
                continue
            raw = rows[:, 6]
            f05, f7, chosen = np.empty(len(rows)), np.empty(len(rows)), {}
            for Y in np.unique(rows[:, 2]):
                m = rows[:, 2] == Y
                as_of = date(int(Y), 7, 1)
                f05[m] = transformed_pit(numeric_calibrator_f05(var, k2, as_of), rows[m])
                cal = numeric_calibrator(var, k2, as_of)
                chosen[int(Y)] = cal.selected
                f7[m] = transformed_pit(cal, rows[m])
            cov = lambda u: float(np.mean((u >= 0.1) & (u <= 0.9)))  # noqa: E731
            res = {"n": int(len(rows)),
                   "coverage80_raw": cov(raw), "coverage80_f05": cov(f05), "coverage80_recal": cov(f7),
                   "ks_d_raw": recal.ks_distance(raw), "ks_d_f05": recal.ks_distance(f05),
                   "ks_d_recal": recal.ks_distance(f7),
                   "ks_p_recal": recal.ks_uniform_p(f7), "selected_by_year": chosen,
                   "candidates_on_test_period_ks_d": {}}
            res.update(clustered_null(f7, rows[:, 2]))
            for c in CANDIDATES:
                rr, u = _candidate_pits(var, k2, c)
                mm = rr[:, 2] >= FIRST_TEST_YEAR
                res["candidates_on_test_period_ks_d"][c] = recal.ks_distance(u[mm])
            out["variables"][var][str(k2)] = res
            log(f"  {var:14s} k2={k2} n={res['n']:5d} cov80 raw {res['coverage80_raw']:.2f} F0.5 "
                f"{res['coverage80_f05']:.2f} F7 {res['coverage80_recal']:.2f}; KS D raw {res['ks_d_raw']:.3f} F0.5 "
                f"{res['ks_d_f05']:.3f} F7 {res['ks_d_recal']:.3f} (null p95 if clustered {res['ks_d_null_p95_year_clustered']:.3f})")
    return out
