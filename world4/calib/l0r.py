"""Calibración bayesiana del L0 regional (F7).

El L0 se estima ecuación a ecuación con MCO (``core.l0r.model.calibrate``): cada ecuación ajusta bien el
año siguiente, pero nada garantiza que la corrida LIBRE de 27 años reproduzca la historia. F7 calibra
la corrida libre:

1. **Perillas** (``PARAMS``): multiplicadores y desplazamientos aditivos sobre las ecuaciones de MCO, con
   su caja de previa uniforme [S]. Las que la historia 1992-2019 no puede identificar (``EXPLORE``:
   recurso último, daño, EROI, sustitución, ...) no se calibran: viajan con su previa y se dice así.
2. **Datos** (los del candado P2, ``validate.p2``): agregados mundiales de población, PIB, energía primaria,
   cereales, tierras de cultivo y GMST en años de control de la ventana, y población, PIB y energía de
   las 10 regiones al final de la ventana.
3. **Varianzas declaradas ANTES de medir** [S]:
   * del dato (``OBS_SD``): error de medida y, en cereales y GMST, la variabilidad interanual que el L0
     no tiene;
   * de la discrepancia (``disc_sd``): la mitad de la banda de P2 (declarada en F6 antes de medir P2)
     a 27 años, creciendo con la raíz del horizonte. La banda de P2 queda como ~2σ del error estructural.
4. *History matching* por oleadas (``history_matching``) con el emulador GP, y después el posterior
   por ABC-SMC con el simulador directo (``abc``), restringido a la región no descartada.
5. Discrepancia AR1 por variable mundial (``discrepancy``) con los pares de la ventana, a la mediana del
   posterior.

``calibrate(2019)`` da el posterior de producción (``models/posterior.npz``). ``out_of_sample(2005)``
repite todo con datos <= 2005 y comprueba la banda 2006-2019: es la prueba de que el método no se
limita a ajustar.
"""
from __future__ import annotations

import time

import numpy as np

from world4.calib import abc, discrepancy, history_matching as hm
from world4.core.l0r import data as rdata
from world4.core.l0r import model, regions

START = 1992                     # el primer año con las 10 regiones (P2)
H_REF = 27                       # años de la ventana de P2 (1992-2019): la discrepancia declarada es a ese horizonte
WORLD_VARS = ("pop", "gdp", "energy", "cereal", "cropland", "gmst")
REGION_VARS = ("pop", "gdp", "energy")
CHECK_STEP = 7                   # años de control: START+6, +13, +20 y el corte
# [S] desviación típica del dato (relativa; °C para la GMST)
OBS_SD = {"pop": 0.01, "gdp": 0.03, "energy": 0.03, "cereal": 0.04, "cropland": 0.02, "gmst": 0.10}
# [S] discrepancia a H_REF años: la mitad de las bandas de P2 (declaradas en F6 antes de medir)
P2_BANDS = {"pop": 0.05, "gdp": 0.25, "energy": 0.25, "cereal": 0.25, "cropland": 0.10, "gmst": 0.25}
MARKS = {"pop": "[S]", "gdp": "[S]", "energy": "[S]", "cereal": "[S]", "cropland": "[M]", "gmst": "[S]"}

# perilla -> (mín, máx, qué es). Caja uniforme [S]: lo bastante ancha para que el borde no decida
PARAMS = {
    "s": (0.8, 1.2, "investment share x (PWT csh_i of the last 20 years)"),
    "cbr": (0.9, 1.1, "crude birth rate x"),
    "cdr": (0.85, 1.15, "crude death rate x"),
    "d_cbr": (-0.01, 0.01, "+ d ln(birth rate) per year"),
    "lam_A": (0.0, 2.0, "TFP convergence speed x (0 is a physical bound)"),
    "d_gA": (-0.01, 0.01, "+ TFP growth per year"),
    "d_int": (-0.01, 0.01, "+ d ln(energy intensity) per year"),
    "d_z": (-0.05, 0.05, "+ d logit(non-fossil share) per year"),
    "d_crop": (-0.005, 0.005, "+ d ln(cropland) per year"),
    "d_yield": (-0.01, 0.01, "+ d ln(cereal yield) per year"),
    "tcre": (0.7, 1.3, "effective TCRE x (F4 ensemble spread)"),
}
# no identificables con 1992-2019: viajan con su previa (exploratorio)
EXPLORE = {
    "resource": (0.5, 2.0, "ultimate fossil resource x"),
    "damage": (0.0, 2.0, "BHM damage x"),
    "eroi": (0.7, 1.3, "fossil EROI x"),
    "subst": (0.0, 2.0, "substitution under physical scarcity x"),
    "deficit": (0.5, 1.5, "public deficit x"),
    "yield_T": (0.5, 1.5, "yield sensitivity to temperature x"),
}


def disc_sd(var: str, h) -> np.ndarray:
    return P2_BANDS[var] / 2 * np.sqrt(np.asarray(h, float) / H_REF)


def truth(W, start: int, end: int) -> dict:
    """Series del dato como las compara P2: niveles sobre el conjunto fijo de países de 1992 (10, años) y
    GMST mundial (años,)."""
    R = rdata.regional(W)
    L, I, tb = R["levels"], R["logindex"], R["base"]
    sl = slice(start - 1950, end - 1950 + 1)
    lev = lambda name: np.exp(np.log(np.maximum(L[name][:, tb], 1))[:, None] + I[name])[:, sl]  # noqa: E731
    out = {"pop": L["pop"][:, sl], "gdp": lev("Y"), "energy": L["pe"][:, sl], "cereal": lev("cereal"),
           "cropland": lev("cropland")}
    out["gmst"] = rdata.world(W)["gmst"][sl]
    return out


def series(r: dict, lnC: np.ndarray) -> dict:
    """De una corrida del L0 (años, M, 10) a las mismas series que ``truth`` (M, 10, años); GMST (M, años)."""
    P = r["P"]
    return {"pop": P.transpose(1, 2, 0), "gdp": (np.exp(r["ly"]) * P).transpose(1, 2, 0),
            "energy": r["E"].transpose(1, 2, 0), "cereal": (r["cpc"] * P * 1e3).transpose(1, 2, 0),
            "cropland": np.exp(lnC).transpose(1, 2, 0), "gmst": r["Tg"][:, :, 0].T}


RECORD = ("P", "ly", "E", "cpc", "Tg", "lnC")


def simulate(p, ini, names, theta, years: int, fixed: dict | None = None) -> dict:
    theta = np.atleast_2d(theta)
    M = len(theta)
    mult = {n: theta[:, j] for j, n in enumerate(names)}
    for k, v in (fixed or {}).items():
        mult[k] = np.broadcast_to(np.asarray(v, float), (M,))
    with np.errstate(all="ignore"):
        r = model.L0R(p, ini, M=M, mult=mult).run(years, record=RECORD)
    return series(r, r["lnC"])


class Setup:
    """El problema de calibración del L0 con datos <= ``cutoff`` en la ventana ``start``..``cutoff``."""

    def __init__(self, W, cutoff: int, start: int = START, box=None):
        self.W, self.cutoff, self.start = W, cutoff, start
        self.p = model.calibrate(W, cutoff)
        self.ini = model.initial(W, start)
        self.H = cutoff - start
        self.names = tuple(PARAMS)
        self.box = np.array([PARAMS[n][:2] for n in self.names], float) if box is None else np.asarray(box, float)
        self.obs = truth(W, start, cutoff)
        self.checks = sorted({y for y in range(start + CHECK_STEP - 1, cutoff, CHECK_STEP)} | {cutoff})
        labels, z, vo, vd = [], [], [], []
        for v in WORLD_VARS:
            for y in self.checks:
                k = y - self.start
                if v == "gmst":
                    z.append(self.obs[v][k]); vo.append(OBS_SD[v] ** 2)
                else:
                    z.append(np.log(np.nansum(self.obs[v][:, k]))); vo.append(OBS_SD[v] ** 2)
                vd.append(float(disc_sd(v, k)) ** 2)
                labels.append(f"world {v} {y}")
        for v in REGION_VARS:
            for r_ in range(rdata.NR):
                z.append(np.log(self.obs[v][r_, self.H])); vo.append(OBS_SD[v] ** 2)
                vd.append(float(disc_sd(v, self.H)) ** 2)
                labels.append(f"{v} {regions.CODES[r_]} {cutoff}")
        self.labels, self.z = tuple(labels), np.array(z)
        self.var_obs, self.var_disc = np.array(vo), np.array(vd)

    def summaries(self, s: dict) -> np.ndarray:
        cols = []
        for v in WORLD_VARS:
            for y in self.checks:
                k = y - self.start
                cols.append(s[v][:, k] if v == "gmst" else np.log(np.sum(s[v][:, :, k], 1)))
        for v in REGION_VARS:
            for r_ in range(rdata.NR):
                cols.append(np.log(s[v][:, r_, self.H]))
        return np.column_stack(cols)

    def simulate(self, theta, years: int | None = None) -> dict:
        return simulate(self.p, self.ini, self.names, theta, self.H if years is None else years)

    def problem(self) -> hm.Problem:
        return hm.Problem(self.names, self.box, lambda th: (self.summaries(self.simulate(th)), None), self.z,
                          self.var_obs, self.var_disc, self.labels,
                          meta={"cutoff": self.cutoff, "start": self.start})


def _growth(s: dict, v: str) -> np.ndarray:
    """Crecimiento mundial desde el inicio (M, años): log del cociente, o °C para la GMST."""
    if v == "gmst":
        return s[v] - s[v][:, :1]
    w = np.sum(s[v], 1)
    return np.log(w / w[:, :1])


def _obs_growth(obs: dict, v: str) -> np.ndarray:
    if v == "gmst":
        return obs[v] - obs[v][0]
    w = np.nansum(obs[v], 0)
    return np.log(w / w[0])


def calibrate(cutoff: int = 2019, start: int = START, log=print, **kw) -> dict:
    """*History matching* + ABC-SMC + discrepancia del L0 con datos <= ``cutoff``. Si el posterior de una
    perilla toca el borde de su caja, la caja se dobla y se recalibra (``hm.widen``, hasta ``hm.MAX_WIDEN``
    veces; regla declarada): el historial de cajas queda en ``box_history``."""
    box, hist = None, []
    for k in range(hm.MAX_WIDEN + 1):
        out = _calibrate_once(cutoff, start, box=box, log=log, **kw)
        hist.append({"box": out.get("box"), "edge": out.get("knobs_at_the_box_edge", [])})
        if not out.get("ok") or not out["knobs_at_the_box_edge"] or k == hm.MAX_WIDEN:
            break
        log(f"  posterior at the box edge for {out['knobs_at_the_box_edge']}: box doubled, recalibrating")
        box = hm.widen(out["names"], out["box"], out["knobs_at_the_box_edge"])
    out["box_history"] = hist
    return out


def _calibrate_once(cutoff: int = 2019, start: int = START, n_runs: int = 150, max_waves: int = 5, particles: int = 1000,
                    seed: int = 0, box=None, log=print) -> dict:
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l1_atlas import calibrate as acal
    t = time.time()
    S = Setup(acal.W, cutoff, start, box=box)
    log(f"L0 calibration, data {start}-{cutoff}: {len(S.names)} parameters, {len(S.z)} outputs")
    H = hm.HistoryMatch(S.problem(), seed=seed, log=log)
    rep = H.run(n_runs, max_waves=max_waves)
    if not H.waves or H.waves[-1].nroy_fraction == 0.0:
        return {"cutoff": cutoff, "start": start, "history_matching": rep, "ok": False,
                "seconds": time.time() - t}
    box = S.box
    sigma = np.sqrt(S.var_obs + S.var_disc)

    def log_prior(th):
        inbox = np.all((th >= box[:, 0]) & (th <= box[:, 1]), 1)
        out = np.full(len(th), -np.inf)
        if inbox.any():
            out[np.where(inbox)[0][H.nroy(th[inbox])]] = 0.0
        return out

    sim_sum = lambda th: S.summaries(S.simulate(th))  # noqa: E731
    post = abc.abc_smc(lambda n, rng: H.sample(n), log_prior, sim_sum, S.z, sigma, n=particles, seed=seed + 1,
                       log=log)
    theta = post["theta"]
    # comprobación del emulador: implausibilidad con el SIMULADOR de las partículas del posterior
    y = sim_sum(theta)
    Isim = hm.kth_max(hm.implausibility(y, np.zeros_like(y), S.z, S.var_obs + S.var_disc))
    # discrepancia AR1 por variable mundial, a la mediana del posterior
    s = S.simulate(theta)
    med = {v: np.median(_growth(s, v), 0) for v in WORLD_VARS}
    h = np.arange(S.H + 1)
    disc, disc_rep = {}, {}
    for v in WORLD_VARS:
        og = _obs_growth(S.obs, v)
        try:
            gp = discrepancy.fit(h, med[v], og, MARKS[v], seed=seed)
            disc[v] = {"h": h, "model": med[v], "obs": og, "mark": MARKS[v]}
            disc_rep[v] = discrepancy.report(gp, P2_BANDS[v])
        except Exception as e:                       # noqa: BLE001  (se informa, no se esconde)
            disc_rep[v] = {"verdict": "BLOCKED", "reason": f"{type(e).__name__}: {e}"}
    # cobertura en la ventana (en muestra: sólo comprueba el montaje)
    ins = {}
    for v in WORLD_VARS:
        g = _growth(s, v)
        lo, hi = np.percentile(g, [10, 90], 0)
        og = _obs_growth(S.obs, v)
        ins[v] = float(np.mean((og[1:] >= lo[1:]) & (og[1:] <= hi[1:])))
    out = {"cutoff": cutoff, "start": start, "names": list(S.names), "box": box.tolist(),
           "history_matching": rep, "abc": {k: post[k] for k in ("generations", "simulations", "eps_final", "converged")},
           "theta": theta, "posterior_summary": _summary(S.names, theta, box),
           "posterior_runs_simulator_non_implausible": float(np.mean(Isim <= hm.CUTOFF)),
           "discrepancy": disc, "discrepancy_report": disc_rep, "in_sample_inside_p10_p90_parametric_only": ins,
           "knobs_at_the_box_edge": hm.edge_knobs(S.names, theta, box),
           "obs_sd": OBS_SD, "p2_bands": P2_BANDS, "marks": MARKS, "explore": EXPLORE, "ok": True,
           "seconds": time.time() - t}
    log(f"L0 posterior ({cutoff}): " + "; ".join(f"{n} {v['p50']:.4g} [{v['p10']:.4g}, {v['p90']:.4g}] "
                                                   f"({v['width_vs_prior']:.0%} of prior)"
                                                   for n, v in out["posterior_summary"].items()))
    return out


def _summary(names, theta, box) -> dict:
    out = {}
    for j, n in enumerate(names):
        q = np.percentile(theta[:, j], [10, 50, 90])
        out[n] = {"p10": float(q[0]), "p50": float(q[1]), "p90": float(q[2]),
                  "width_vs_prior": float((np.percentile(theta[:, j], 90) - np.percentile(theta[:, j], 10))
                                          / (0.8 * (box[j, 1] - box[j, 0])))}
    return out


def _wlevel(s: dict, v: str) -> np.ndarray:
    """Nivel mundial (M, años): log de la suma, o °C para la GMST."""
    return s[v] if v == "gmst" else np.log(np.sum(s[v], 1))


def _obs_level(obs: dict, v: str) -> np.ndarray:
    return obs[v] if v == "gmst" else np.log(np.nansum(obs[v], 0))


def predictive(post: dict, ini, p, years: int, members: int = 250, draws: int = 4, seed: int = 0) -> dict:
    """Banda del DATO mundial desde ``ini`` (niveles; log salvo la GMST):

    * ``parametric``: sólo el posterior de las perillas;
    * ``reality``: + la discrepancia DECLARADA (``disc_sd``, la misma que entra en la verosimilitud del
      history matching y del ABC) + el error del dato (``OBS_SD``). Es el modelo de predicción coherente
      con el de calibración (Vernon et al. 2010; Brynjarsdóttir y O'Hagan 2014: la discrepancia ajustada
      con los residuos de la ventana subestima la de fuera, porque la calibración ya los absorbió).

    La discrepancia AR1 (``discrepancy``) se informa como diagnóstico del sesgo de escala (ρ), no entra
    en la banda [cambio declarado tras la comprobación de desarrollo con corte 2005]."""
    rng = np.random.default_rng(seed)
    theta = post["theta"]
    pick = theta[rng.choice(len(theta), min(members, len(theta)), replace=False)]
    s = simulate(p, ini, tuple(post["names"]), pick, years)
    h = np.arange(years + 1)
    out = {"_series": s}
    for v in WORLD_VARS:
        g = _wlevel(s, v)
        sd = np.sqrt(disc_sd(v, h) ** 2 + OBS_SD[v] ** 2)
        noise = sd[None, None] * rng.standard_normal((draws,) + g.shape)
        out[v] = {"parametric": g, "reality": (g[None] + noise).reshape(-1, g.shape[1])}
    return out


def out_of_sample(cutoff: int = 2005, end: int = 2019, log=print, **kw) -> dict:
    """Calibra con datos <= ``cutoff`` (ventana 1992-``cutoff``) y comprueba la banda de ``cutoff`` a
    ``end`` desde el dato de ``cutoff`` (el L0 arranca del estado observado, como el L1 en el origen).
    Con corte 2005 es la comprobación de DESARROLLO del método (se miró para fijarlo)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l1_atlas import calibrate as acal
    W = acal.W
    post = calibrate(cutoff, log=log, **kw)
    if not post["ok"]:
        return {"cutoff": cutoff, "end": end, "ok": False, "calibration": _jsonable(post)}
    p = model.calibrate(W, cutoff)
    ini = model.initial(W, cutoff)
    pr = predictive(post, ini, p, end - cutoff)
    obs = truth(W, cutoff, end)
    base = simulate(p, ini, (), np.zeros((1, 0)), end - cutoff)       # MCO sin calibrar (F6)
    res = {}
    for v in WORLD_VARS:
        ol = _obs_level(obs, v)
        row = {"obs": ol.tolist(), "unit": "degC" if v == "gmst" else "log of the world total"}
        for kind in ("parametric", "reality"):
            q10, q50, q90 = np.percentile(pr[v][kind], [10, 50, 90], 0)
            ins = (ol >= q10) & (ol <= q90)
            row[kind] = {"p10": q10.tolist(), "p50": q50.tolist(), "p90": q90.tolist(),
                         "inside_share": float(ins[1:].mean()), "err_p50_end": float(q50[-1] - ol[-1])}
        row["ols_f6_err_end"] = float(_wlevel(base, v)[0, -1] - ol[-1])
        res[v] = row
    out = {"cutoff": cutoff, "end": end, "variables": res, "calibration": _jsonable(post),
           "criterion": ">= 80 % of years inside p10-p90 (W4 threshold), world aggregates"}
    out["ok"] = all(r["reality"]["inside_share"] >= 0.80 for r in res.values())
    log(f"L0 out of sample {cutoff}->{end}: " + "; ".join(
        f"{v} inside {r['reality']['inside_share']:.0%} (parametric only {r['parametric']['inside_share']:.0%}; "
        f"p50 err end {r['reality']['err_p50_end']:+.3f}, F6 OLS {r['ols_f6_err_end']:+.3f})" for v, r in res.items()))
    return out


def _jsonable(post: dict) -> dict:
    """Lo que va al JSON de ``runs/`` (sin las partículas ni los pares, que van al .npz)."""
    return {k: v for k, v in post.items() if k not in ("theta", "discrepancy")}
