"""F8: búsqueda robusta de políticas (patrón MORDM: Kasprzyk et al. 2013) y sus vulnerabilidades.

* **Plugin de phycore** (``World4Plugin``): θ = las palancas de ``levers``, evaluadas en el L0 regional
  sobre un ensemble FIJO de estados del mundo (``narratives.sow``: posterior de F7 + incertidumbre
  profunda + eficacia de las palancas). Con números aleatorios comunes, dos políticas se comparan en los
  mismos mundos.
* **Objetivos** (todos se minimizan), robustos por percentil sobre el ensemble:
    f0 = -p10 del bienestar: media descontada (2 %/año) del log del consumo per cápita, ponderada por población;
    f1 = p90 de la GMST máxima 2020-2100;
    f2 = -p10 del log del consumo per cápita de la región más pobre en 2100.
* **Restricciones exactas** (g <= 0), límites como en MORDM [M]:
    g0 alimento: p10 del mínimo regional (en el periodo) del cereal per cápita relativo a 2019 >= 0,9;
    g1 tierra:   p90 de las tierras de cultivo de 2100 / 2019 <= 1,10;
    g2 deuda:    p90 de la deuda/PIB mundial de 2100 <= 1,5.
* **Búsqueda:** ``phycore.Designer`` con NSGA-II a fidelidad 0 (el L0 es la verdad: sin sustituto) y el
  frente VERIFICADO (sólo evaluaciones reales).
* **Arrepentimiento** (Savage) sobre el frente y la línea base, en cada estado del mundo.
* **PRIM** (Friedman y Fisher 1999), implementado aquí: en qué región del espacio incierto falla cada
  política (GMST máxima > 3 °C o bienestar peor que no hacer nada). EMA Workbench (BSD-3, verificada en
  su LICENSE el 2026-10-13) es la referencia; va como extra opcional ``world4[dmdu]``, no como dependencia.

Todo resultado es un modo de comportamiento a 2100 (``exploratory``), nunca un pronóstico.
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from world4.scenarios import levers, narratives

START, END = 2019, 2100
DISCOUNT = 0.02
FOOD_FLOOR, LAND_CAP, DEBT_CAP = 0.9, 1.10, 1.5
FAIL_GMST = 3.0
OBJECTIVES = ("neg_welfare_p10", "peak_gmst_p90", "neg_poorest_p10")
CONSTRAINTS = ("food_floor", "land_cap", "debt_cap")
RECORD = ("P", "Y", "C", "Tg", "cpc", "lnC", "debt", "nfs", "E")
UNITS = {
    "neg_welfare_p10": "minus the p10 of welfare (discounted mean log consumption per person, log 2017 USD)",
    "peak_gmst_p90": "degC above 1850-1900 (p90 of the 2020-2100 maximum)",
    "neg_poorest_p10": "minus the p10 of log consumption per person of the poorest region in 2100 (log 2017 USD)",
    "food_floor": "g = 0.9 - p10 of the lowest regional cereal per person relative to 2019 (fraction; <= 0 is inside)",
    "land_cap": "g = p90 of cropland 2100 / 2019 - 1.10 (ratio; <= 0 is inside)",
    "debt_cap": "g = p90 of world public debt / GDP in 2100 - 1.5 (ratio to GDP; <= 0 is inside)",
    "welfare": "discounted mean log consumption per person (log 2017 USD)", "peak_gmst": "degC",
    "poorest": "log consumption per person of the poorest region in 2100 (log 2017 USD)",
    "food_ratio_min": "lowest regional cereal per person relative to 2019 (fraction)",
    "land_ratio": "cropland 2100 / 2019", "debt_world": "world public debt / GDP in 2100", "gmst_2050": "degC",
    "gmst_2100": "degC", "nonfossil_2050": "non-fossil share of primary energy in 2050 (fraction)",
}


@lru_cache(maxsize=4)
def _base(start: int = START):
    import warnings
    warnings.simplefilter("ignore")
    from world4.data import atlas
    atlas.require()                         # sin datos del Atlas: un error con su remedio, no un traceback
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    return model.calibrate(W, min(start, 2019)), model.initial(W, start)


def run(theta, mult: dict, start: int = START, end: int = END, record=RECORD) -> dict:
    """Una política θ (d,) sobre los estados del mundo ``mult`` ({k: (n,)}): series del L0 (años, n, 10)."""
    from world4.core.l0r import model
    p, ini = _base(start)
    n = len(next(iter(mult.values())))
    pol = {k: np.full(n, float(v[0])) for k, v in levers.to_policy(theta).items()}
    with np.errstate(all="ignore"):
        return model.L0R(p, ini, M=n, mult=mult, policy=pol).run(end - start, record=record)


def metrics(r: dict) -> dict:
    """Por estado del mundo (n,)."""
    H = r["P"].shape[0]
    beta = (1 + DISCOUNT) ** -np.arange(H)
    P = r["P"]
    lc = np.log(r["C"] * 1e6 / np.maximum(P * 1e6, 1e-9))          # USD por persona (Y en millones, P en millones)
    w = np.sum(beta[:, None] * np.sum(P * lc, 2) / np.sum(P, 2), 0) / beta.sum()
    cpc0 = r["cpc"][0]
    food = np.min(np.min(r["cpc"] / np.maximum(cpc0[None], 1e-9), 0), 1)
    land = np.exp(r["lnC"][-1]).sum(1) / np.exp(r["lnC"][0]).sum(1)
    Y = r["Y"][-1]
    debt = np.sum(r["debt"][-1] * Y, 1) / np.sum(Y, 1)
    return {"welfare": w, "peak_gmst": r["Tg"][1:, :, 0].max(0), "poorest": lc[-1].min(1),
            "food_ratio_min": food, "land_ratio": land, "debt_world": debt,
            "gmst_2050": r["Tg"][min(31, H - 1), :, 0], "gmst_2100": r["Tg"][-1, :, 0],
            "nonfossil_2050": np.sum(r["nfs"][min(31, H - 1)] * r["E"][min(31, H - 1)], 1) / np.sum(r["E"][min(31, H - 1)], 1)}


def summarise(m: dict) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([-np.percentile(m["welfare"], 10), np.percentile(m["peak_gmst"], 90),
                  -np.percentile(m["poorest"], 10)])
    g = np.array([FOOD_FLOOR - np.percentile(m["food_ratio_min"], 10), np.percentile(m["land_ratio"], 90) - LAND_CAP,
                  np.percentile(m["debt_world"], 90) - DEBT_CAP])
    return f, g


class World4Plugin:
    """El contrato de ``phycore.plugin.PhyPlugin`` para la búsqueda de políticas sobre el L0."""
    plugin_id = "world4"
    integer_dims = ()
    objective_names = OBJECTIVES

    def __init__(self, members: int = 100, narrative: str = narratives.DEFAULT, seed: int = 0, report_only=()):
        from world4 import MODEL_REVISION
        self.model_revision = MODEL_REVISION
        self.bounds = levers.BOX.copy()
        self.members, self.narrative, self.seed = members, narrative, seed
        self.mult = narratives.sow(members, narrative, seed)
        # restricciones que se informan pero no restringen (variante declarada de la búsqueda)
        self.active = np.array([c not in report_only for c in CONSTRAINTS])

    def spec_vec(self, spec) -> np.ndarray:
        return np.zeros(1)

    def evaluate(self, theta, spec=None, fidelity: int = 0) -> dict:
        from phycore.plugin import repair
        th = repair(self, theta)
        m = metrics(run(th, self.mult))
        f, g = summarise(m)
        return {"theta": th, "fidelity_used": "L0", "f": f, "g": g,
                "m": {k: v.astype(np.float32) for k, v in m.items()}}

    def objectives(self, rec, spec=None):
        return rec["f"]

    def objective_bands(self, rec, spec=None):
        from world4.calib.l0r import disc_sd
        h = END - START
        # la discrepancia declarada del L0 a 2100 (la misma para todas las políticas: no cambia el orden)
        return np.array([0.0, float(disc_sd("gmst", h)), float(disc_sd("gdp", h))])

    def constraints(self, rec, spec=None):
        return np.where(self.active, rec["g"], np.minimum(rec["g"], 0.0))

    def constraint_names(self, spec=None):
        return CONSTRAINTS

    def embedding(self, rec):
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        return (np.asarray(rec["theta"]) - lo) / (hi - lo)

    def residual(self, rec, spec=None):
        return None

    def apply_residual(self, rec, spec=None, res=None):
        out = dict(rec)
        out["corrected_by_surrogate"] = True
        return out


def evaluate_policy(theta, *, members: int = 200, narrative: str = narratives.DEFAULT, seed: int = 0,
                    vulnerability: bool = True) -> dict:
    """Una política sin buscar (``--theta``): objetivos, restricciones, comparación con no hacer nada y,
    si se pide, sus vulnerabilidades (PRIM)."""
    th = np.asarray(theta, float)
    mult = narratives.sow(members, narrative, seed)
    m = metrics(run(th, mult))
    m0 = metrics(run(np.zeros(len(levers.NAMES)), mult))
    f, g = summarise(m)
    f0, g0 = summarise(m0)
    out = {"theta": dict(zip(levers.NAMES, th.tolist())), "narrative": narrative, "members": members,
           "objectives": dict(zip(OBJECTIVES, f.tolist())), "constraints": dict(zip(CONSTRAINTS, g.tolist())),
           "feasible": bool(np.all(g <= 0)), "baseline_objectives": dict(zip(OBJECTIVES, f0.tolist())),
           "baseline_constraints": dict(zip(CONSTRAINTS, g0.tolist())),
           "percentiles": {k: {q: float(np.percentile(v, q)) for q in (10, 50, 90)} for k, v in m.items()},
           "better_than_baseline_share": {
               "welfare": float(np.mean(m["welfare"] > m0["welfare"])),
               "peak_gmst": float(np.mean(m["peak_gmst"] < m0["peak_gmst"]))},
           "exploratory": True, "fidelity": "L0", "units": UNITS}
    if vulnerability:
        out["vulnerability"] = vulnerabilities(th, members=max(members, 1000), narrative=narrative, seed=seed + 1)
    return out


# ------------------------------------------------------------------ arrepentimiento
def regret(thetas, members: int = 200, narrative: str = narratives.DEFAULT, seed: int = 0) -> dict:
    """Arrepentimiento de Savage del bienestar y de la GMST máxima: en cada mundo, la distancia a la mejor
    política del conjunto (que incluye no hacer nada)."""
    mult = narratives.sow(members, narrative, seed)
    T = [np.zeros(len(levers.NAMES))] + [np.asarray(t, float) for t in thetas]
    M = [metrics(run(t, mult)) for t in T]
    W = np.array([m["welfare"] for m in M])
    G = np.array([m["peak_gmst"] for m in M])
    rw, rg = W.max(0)[None] - W, G - G.min(0)[None]
    return {"policies": [dict(zip(levers.NAMES, t.tolist())) for t in T],
            "welfare_regret_p90": np.percentile(rw, 90, 1).tolist(), "gmst_regret_p90": np.percentile(rg, 90, 1).tolist(),
            "welfare_regret_max": rw.max(1).tolist(), "gmst_regret_max": rg.max(1).tolist(),
            "note": "index 0 is no policy; regret is against the best of this set in each world"}


# ------------------------------------------------------------------ PRIM
def prim(X: np.ndarray, y: np.ndarray, names, alpha: float = 0.05, min_support: float = 0.05,
         min_coverage: float = 0.5) -> dict:
    """Peeling de PRIM: en cada paso quita la fracción ``alpha`` de los puntos por el lado (dimensión,
    inferior o superior) que más sube la densidad de fallos dentro de la caja. Se elige la caja más densa
    con cobertura >= ``min_coverage`` [declarado]."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, d = X.shape
    total = y.sum()
    if total == 0:
        return {"failures": 0, "boxes": [], "chosen": None, "base_rate": 0.0}
    lo, hi = X.min(0).copy(), X.max(0).copy()
    inside = np.ones(n, bool)
    traj = []

    def rec(box_lo, box_hi, ins):
        return {"coverage": float(y[ins].sum() / total), "density": float(y[ins].mean()),
                "support": float(ins.mean()), "lo": box_lo.tolist(), "hi": box_hi.tolist()}
    traj.append(rec(lo, hi, inside))
    while inside.mean() > min_support:
        best = None
        for j in range(d):
            xj = X[inside, j]
            for side in (0, 1):
                cut = np.quantile(xj, alpha if side == 0 else 1 - alpha)
                keep = inside & ((X[:, j] >= cut) if side == 0 else (X[:, j] <= cut))
                if keep.sum() == inside.sum() or keep.sum() == 0:
                    continue
                dens = y[keep].mean()
                if best is None or dens > best[0]:
                    best = (dens, j, side, cut, keep)
        if best is None:
            break
        _, j, side, cut, keep = best
        if side == 0:
            lo[j] = cut
        else:
            hi[j] = cut
        inside = keep
        traj.append(rec(lo, hi, inside))
    ok = [b for b in traj if b["coverage"] >= min_coverage]
    chosen = max(ok, key=lambda b: (b["density"], b["coverage"])) if ok else None
    full_lo, full_hi = X.min(0), X.max(0)
    if chosen is not None:
        restricted = []
        for j, nm in enumerate(names):
            span = full_hi[j] - full_lo[j]
            cut = 1 - (chosen["hi"][j] - chosen["lo"][j]) / span if span > 0 else 0.0
            if cut > 0.05:
                restricted.append({"name": nm, "lo": chosen["lo"][j], "hi": chosen["hi"][j],
                                   "range": [float(full_lo[j]), float(full_hi[j])], "cut": float(cut)})
        restricted.sort(key=lambda r: -r["cut"])
        chosen = {**chosen, "restricted": restricted}
    return {"failures": int(total), "base_rate": float(y.mean()), "boxes": len(traj), "chosen": chosen,
            "trajectory": [{k: b[k] for k in ("coverage", "density", "support")} for b in traj]}


def vulnerabilities(theta, members: int = 1000, narrative: str = narratives.DEFAULT, seed: int = 1) -> dict:
    """En qué mundos falla la política θ: GMST máxima > FAIL_GMST, o bienestar peor que sin política."""
    mult = narratives.sow(members, narrative, seed)
    m = metrics(run(np.asarray(theta, float), mult))
    m0 = metrics(run(np.zeros(len(levers.NAMES)), mult))
    names = sorted(mult)
    X = np.column_stack([np.asarray(mult[k], float) for k in names])
    out = {}
    for label, y in (("peak_gmst_above_3C", m["peak_gmst"] > FAIL_GMST),
                     ("welfare_below_no_policy", m["welfare"] < m0["welfare"])):
        out[label] = prim(X, y.astype(float), names)
    out["members"] = members
    return out


# ------------------------------------------------------------------ la búsqueda
def design(members: int = 100, narrative: str = narratives.DEFAULT, seed: int = 0, n_init: int = 60, rounds: int = 3,
           pop: int = 40, generations: int = 20, report_only=(), log=print) -> dict:
    """``report_only``: restricciones que se informan pero no restringen (p. ej. ("land_cap",) cuando ninguna
    política la cumple y el frente degenera en el punto de mínima violación)."""
    from phycore.engine.designer import Designer
    t = time.time()
    plug = World4Plugin(members, narrative, seed, report_only=report_only)
    res = Designer(plug, spec=None, fidelity=0, seed=20261013 + seed,
                   log=lambda tag, msg: log(f"  [{tag}] {msg}")).run(n_init=n_init, rounds=rounds, pop=pop,
                                                                     generations=generations)
    front = res.front
    feas = [i for i in front if np.all(np.asarray(i.rec["g"]) <= 0)]
    F = np.array([i.f for i in front]) if front else np.zeros((0, 3))
    rows = [{"theta": dict(zip(levers.NAMES, np.asarray(i.theta, float).tolist())),
             "objectives": dict(zip(OBJECTIVES, np.asarray(i.f, float).tolist())),
             "constraints": dict(zip(CONSTRAINTS, np.asarray(i.rec["g"], float).tolist())),
             "feasible": bool(np.all(np.asarray(i.rec["g"]) <= 0)), "feasible_active": bool(i.feasible)}
            for i in front]
    base = plug.evaluate(np.zeros(len(levers.NAMES)))
    allg = np.array([i.rec["g"] for i in res.population])
    diag = diagnose(allg)
    out = {"narrative": narrative, "members": members, "evaluations": res.evaluations, "seconds": time.time() - t,
           "cheap_evaluations": res.n_cheap, "report_only": list(report_only),
           "best_reachable_constraints": dict(zip(CONSTRAINTS, allg.min(0).tolist())), "diagnosis": diag,
           "front": rows, "front_size": len(front), "feasible_on_front": len(feas),
           "baseline": {"objectives": dict(zip(OBJECTIVES, base["f"].tolist())),
                        "constraints": dict(zip(CONSTRAINTS, base["g"].tolist()))},
           "levers": levers.describe(), "objectives_minimised": list(OBJECTIVES), "constraints": list(CONSTRAINTS),
           "units": UNITS,
           "exploratory": True, "fidelity": "L0"}
    if len(front):
        # tres representantes: mejor bienestar, mejor clima, compromiso (más cerca del ideal normalizado)
        lo, hi = F.min(0), F.max(0)
        Z = (F - lo) / np.where(hi > lo, hi - lo, 1.0)
        pool = [k for k, i in enumerate(front) if i.feasible] or list(range(len(front)))
        reps = {"best_welfare": min(pool, key=lambda k: F[k, 0]), "best_climate": min(pool, key=lambda k: F[k, 1]),
                "compromise": min(pool, key=lambda k: float(np.linalg.norm(Z[k])))}
        out["representatives"] = {}
        for name, k in reps.items():
            th = np.asarray(front[k].theta, float)
            out["representatives"][name] = {"index": int(k), "theta": rows[k]["theta"],
                                            "vulnerability": vulnerabilities(th, narrative=narrative, seed=seed + 1)}
        out["regret"] = regret([np.asarray(front[k].theta, float) for k in reps.values()], members=members,
                               narrative=narrative, seed=seed + 2)
        out["regret"]["labels"] = ["no_policy"] + list(reps)
    log(f"  front: {len(front)} verified policies ({len(feas)} feasible) from {res.evaluations} L0 ensemble "
        f"evaluations in {out['seconds']:.0f} s")
    return out


# ------------------------------------------------------------------ por qué no hay política factible
ADVICE = {
    "food_floor": "no lever keeps every region above 90 % of its 2019 cereal per person: raise agri_rd, or accept "
                  "imports (the floor is regional self-supply, not food security)",
    "land_cap": "in the L0 cropland only responds to population: only the fertility lever moves it; there is no "
                "land-sparing lever (yields do not spare land in the data before 2019)",
    "debt_cap": "world debt above the cap: raise consolidation",
}


def diagnose(G=None, *, narrative: str = narratives.DEFAULT, members: int = 50, n: int = 64, seed: int = 0) -> dict:
    """WALLS (ninguna política de la población cumple la restricción: hay que cambiar la especificación, no buscar
    más), TIGHT (la mayoría falla, alguna la cumple) y THE LAST BLOCKER (la restricción que, sola, bloquea
    políticas que cumplirían todas las demás). ``G`` (n, 3) de una población ya evaluada; si no, un hipercubo
    latino de ``n`` políticas en la caja de las palancas."""
    if G is None:
        from world4.calib.history_matching import lhs
        rng = np.random.default_rng(seed)
        th = levers.BOX[:, 0] + lhs(n, len(levers.NAMES), rng) * (levers.BOX[:, 1] - levers.BOX[:, 0])
        plug = World4Plugin(members, narrative, seed)
        G = np.array([plug.evaluate(t)["g"] for t in th])
    G = np.asarray(G, float)
    frac, gmin = (G > 0).mean(0), G.min(0)
    walls = [j for j in range(G.shape[1]) if frac[j] >= 0.999 and gmin[j] > 0]
    tight = [j for j in range(G.shape[1]) if j not in walls and frac[j] >= 0.5]
    last = []
    for j in range(G.shape[1]):
        others = np.delete(G, j, axis=1)
        ok = others.max(1) <= 0
        if ok.any() and np.any(G[ok, j] > 0):
            last.append({"name": CONSTRAINTS[j], "n": int(np.sum(ok & (G[:, j] > 0))), "g_closest": float(G[ok, j].min())})
    last.sort(key=lambda r: -r["n"])
    return {"population": int(len(G)), "feasible": int(np.sum(G.max(1) <= 0)),
            "walls": [{"name": CONSTRAINTS[j], "violated_share": float(frac[j]), "g_best": float(gmin[j]),
                       "advice": ADVICE[CONSTRAINTS[j]]} for j in walls],
            "tight": [{"name": CONSTRAINTS[j], "violated_share": float(frac[j]), "g_best": float(gmin[j])} for j in tight],
            "last_blocker": last}
