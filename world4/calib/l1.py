"""Calibración bayesiana del L1 (F7) y el hindcast calibrado de W4.

El L1 ya lleva incertidumbre paramétrica de cada ecuación sectorial (coeficientes por miembro con su
covarianza robusta) y choques AR(1) comunes y propios. Lo que no lleva es el SESGO de su corrida libre:
una ecuación ajustada con datos <= origen sigue con su tendencia media aunque la historia ya la esté
doblando (la fecundidad de los noventa, la intensidad energética, el ahorro de tierra). F7 lo calibra
con lo único que se sabía en el origen O:

1. **Pseudo-hindcasts** desde O-15, O-10 y O-5 hasta O (``WINDOWS``), cada uno con TODO estimado con
   datos <= su propio origen, como el hindcast real.
2. **Perillas** (``coupling.L1_KNOBS``): un desplazamiento del intercepto de las ecuaciones de fecundidad,
   esperanza de vida, intensidad energética, cuotas no fósiles, rendimiento y tierras de cultivo, el
   mismo en las tres ventanas (un sesgo persistente), con caja de previa uniforme [S] (``BOX``).
3. **Resúmenes:** log de los agregados mundiales (población, PIB, energía primaria, CO2 fósil, cereales,
   tierras) y GMST al final de cada ventana y a mitad de la de 15 años; varianza del dato ``OBS_SD`` y de
   la discrepancia declarada ANTES de medir (la de ``calib.l0r``: la mitad de las bandas de P2 a 27
   años, con la raíz del horizonte).
4. *History matching* por oleadas con el emulador GP (cada corrida de diseño: 3 ventanas × ``R_DESIGN``
   miembros, en procesos aparte) y posterior por ABC-SMC SOBRE EL EMULADOR (el L1 cuesta segundos por
   corrida: el ABC directo exigiría ~10^5).
5. **Discrepancia AR1** por variable con los pares (h, crecimiento del modelo, crecimiento observado) de
   las tres ventanas, al posterior: DIAGNÓSTICO del sesgo de escala (ρ), no entra en la banda.
6. **Hindcast** O -> 2023 con un θ del posterior por miembro, la discrepancia declarada y el error del dato
   (el método que fijó la comprobación de desarrollo del L0, ``calib.l0r.predictive``): la banda que se
   compara con W4. Ningún dato posterior a O entra en 1-5.

Fuga declarada heredada de F4: el posterior de FaIR se restringió con la GMST hasta 2023.
"""
from __future__ import annotations

import time
from multiprocessing import get_context

import numpy as np

from world4.calib import abc, discrepancy, history_matching as hm
from world4.calib.l0r import OBS_SD as _OBS0, P2_BANDS, H_REF

END = 2023
VARS = ("pop", "gdp", "energy", "co2", "cereal", "cropland", "gmst")
OBS_SD = {**_OBS0, "co2": 0.05}
DISC_BANDS = {**P2_BANDS, "co2": P2_BANDS["energy"]}
MARKS = {"pop": "[S]", "gdp": "[S]", "energy": "[S]", "co2": "[S]", "cereal": "[S]", "cropland": "[M]", "gmst": "[S]"}
UNITS = {"pop": "million persons (180 L1 countries)", "gdp": "2017 USD PPP, countries with GDP at the origin",
         "energy": "TWh primary energy (substitution), countries with energy data at the origin",
         "co2": "GtCO2/year fossil (CMIP7 history)", "cereal": "t cereals (FAOSTAT, simulated countries)",
         "cropland": "1000 ha cropland (FAOSTAT, simulated countries)", "gmst": "degC above 1850-1900 (HadCRUT5)"}
# Caja de partida [S]. Regla declarada antes de ver el hindcast (``hm.edge_knobs``/``hm.widen``): si el p10 o
# el p90 de una perilla queda a < 5 % del ancho de un borde, la caja se dobla y se recalibra. En la primera
# calibración con origen 1990 la intensidad y el rendimiento tocaban el borde de ±0,015: la regla los dobla.
BOX = {"tfr": (-0.02, 0.02), "e0": (-0.2, 0.2), "intensity": (-0.015, 0.015), "nonfossil": (-0.10, 0.10),
       "yield": (-0.015, 0.015), "cropland": (-0.008, 0.008)}
WINDOW_LAGS = (15, 10, 5)
R_DESIGN = 8
WORKERS = 8


def disc_sd(var, h):
    return DISC_BANDS[var] / 2 * np.sqrt(np.asarray(h, float) / H_REF)


# ------------------------------------------------------------------ una corrida y sus agregados
_W23 = None


def world23():
    global _W23
    if _W23 is None:
        from world4.core.l1_atlas import model
        _W23 = model.build_world(horizon=END)
    return _W23


def run_window(origin: int, end: int, R: int, seed: int, calib: dict | None) -> dict:
    """Corrida del L1 F1-F5 desde ``origin`` hasta ``end`` y sus agregados mundiales con su verdad:
    {var: (model (R, H+1), truth (H+1,))}."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.data import climate as cdata
    from world4.data import energy as edata
    from world4.data import food as fdata
    W = world23()
    t0, H = origin - 1950, end - origin
    f = coupling.forecast(t0, R=R, seed=seed, pop_mode="endogenous", world=W, horizon=H, f5=coupling.F5Config(),
                          calib=calib)
    years = np.arange(origin, end + 1)
    out = {}
    out["pop"] = (np.nansum(f["pop"], 1), np.nansum(W.pop[:, t0:t0 + H + 1], 0))
    # PIB: países con PIB en el origen; verdad sólo donde hay dato (hasta 2019)
    okY = np.isfinite(W.Y[:, t0]) & W.observed[:, t0]
    gdp_m = np.nansum(np.where(okY[None, :, None], np.exp(f["ly"]) * f["pop"], 0), 1)
    Yobs = W.Y[:, t0:t0 + H + 1]
    full = W.observed[:, t0:t0 + H + 1][okY].all(0)
    out["gdp"] = (gdp_m, np.where(full, np.nansum(np.where(okY[:, None], Yobs, 0), 0), np.nan))
    a = edata.aligned(W.iso3, W.isonum, years)
    oke = np.isfinite(a["pe"][:, 0])
    out["energy"] = (np.nansum(np.where(oke[None, :, None], f["energy_PE"], 0), 1),
                     np.nansum(np.where(oke[:, None], a["pe"], 0), 0))
    out["co2"] = (f["climate_co2_ffi"], cdata.co2_ffi_observed(years))
    okf = f["sim"].fd.ok
    fa = fdata.aligned(W.iso3, W.isonum, years)

    def tot(key):
        return np.array([np.nansum(np.where(okf & np.isfinite(fa[key][:, j]), fa[key][:, j], 0))
                         if np.isfinite(fa[key][okf, j]).sum() > 0.8 * okf.sum() else np.nan for j in range(len(years))])
    out["cereal"] = (f["foodw_cereal_prod"], tot("cereal_prod"))
    out["cropland"] = (f["foodw_cropland"], tot("cropland"))
    out["gmst"] = (f["Tg"], cdata.gmst_observed(years))
    return out


def _task(args):
    origin, end, R, seed, calib = args
    r = run_window(origin, end, R, seed, calib)
    return {k: (np.asarray(m, float), np.asarray(t, float)) for k, (m, t) in r.items()}


def _level(v, x):
    return x if v == "gmst" else np.log(np.maximum(x, 1e-300))


# ------------------------------------------------------------------ el problema de calibración
class Setup:
    def __init__(self, origin: int, workers: int = WORKERS, box=None):
        self.origin = origin
        self.windows = tuple(origin - L for L in WINDOW_LAGS)
        self.names = tuple(BOX)
        self.box = np.array([BOX[n] for n in self.names], float) if box is None else np.asarray(box, float)
        self.workers = workers
        # resúmenes: (ventana, año) por variable
        self.points = []
        for w in self.windows:
            self.points.append((w, self.origin))
            if self.origin - w >= 15:
                self.points.append((w, w + 7))
        self._truth = None

    def _run(self, jobs):
        if self.workers <= 1:
            return [_task(j) for j in jobs]
        with get_context("spawn").Pool(self.workers) as pool:
            return pool.map(_task, jobs)

    def run_design(self, theta, R: int = R_DESIGN, seed: int = 0):
        """Para cada θ y ventana: agregados (lista por θ de {ventana: {var: (model, truth)}})."""
        jobs = [(w, self.origin, R, seed + 1000 * (j % 7) + w, {n: float(t[k]) for k, n in enumerate(self.names)})
                for j, t in enumerate(theta) for w in self.windows]
        res = self._run(jobs)
        out, i = [], 0
        for _ in theta:
            d = {}
            for w in self.windows:
                d[w] = res[i]
                i += 1
            out.append(d)
        if self._truth is None and out:
            self._truth = {w: {v: out[0][w][v][1] for v in VARS} for w in self.windows}
        return out

    def summaries(self, runs):
        """y (n, q) = nivel (log) de la mediana del ensemble; var_stoch (n, q) de esa mediana."""
        Y, V = [], []
        for d in runs:
            y, vv = [], []
            for v in VARS:
                for (w, yr) in self.points:
                    m = _level(v, d[w][v][0][:, yr - w])
                    y.append(np.median(m))
                    vv.append(np.pi / 2 * np.var(m, ddof=1) / len(m))
            Y.append(y)
            V.append(vv)
        return np.array(Y), np.array(V)

    def targets(self):
        z, vo, vd, labels = [], [], [], []
        for v in VARS:
            for (w, yr) in self.points:
                t = self._truth[w][v][yr - w]
                z.append(_level(v, t))
                vo.append(OBS_SD[v] ** 2)
                vd.append(float(disc_sd(v, yr - w)) ** 2)
                labels.append(f"{v} {w}->{yr}")
        return np.array(z), np.array(vo), np.array(vd), tuple(labels)


def calibrate(origin: int = 1990, log=print, **kw) -> dict:
    """Calibración del L1 para el origen ``origin``, con la regla de la caja (``hm.widen``) como en el L0."""
    box, hist = None, []
    for k in range(hm.MAX_WIDEN + 1):
        out = _calibrate_once(origin, box=box, log=log, **kw)
        hist.append({"box": out.get("box"), "edge": out.get("knobs_at_the_box_edge", [])})
        if not out.get("ok") or not out["knobs_at_the_box_edge"] or k == hm.MAX_WIDEN:
            break
        log(f"  posterior at the box edge for {out['knobs_at_the_box_edge']}: box doubled, recalibrating")
        box = hm.widen(out["names"], out["box"], out["knobs_at_the_box_edge"])
    out["box_history"] = hist
    return out


def _calibrate_once(origin: int = 1990, n_runs: int = 48, max_waves: int = 4, particles: int = 1000, seed: int = 0,
                    workers: int = WORKERS, box=None, log=print) -> dict:
    t = time.time()
    S = Setup(origin, workers, box=box)
    log(f"L1 calibration for origin {origin}: pseudo-hindcasts from {S.windows} to {origin}, "
        f"{len(S.names)} knobs, {len(S.points) * len(VARS)} outputs")
    # la verdad (no depende de θ): una corrida barata
    S.run_design(np.zeros((1, len(S.names))), R=2)
    z, vo, vd, labels = S.targets()
    ok_out = np.isfinite(z)

    def simulate(theta):
        y, vs = S.summaries(S.run_design(theta))
        return y[:, ok_out], vs[:, ok_out]
    pb = hm.Problem(S.names, S.box, simulate, z[ok_out], vo[ok_out], vd[ok_out],
                    tuple(l for l, k in zip(labels, ok_out) if k), meta={"origin": origin})
    H = hm.HistoryMatch(pb, seed=seed, log=log)
    rep = H.run(n_runs, max_waves=max_waves)
    out = {"origin": origin, "windows": list(S.windows), "names": list(S.names), "box": S.box.tolist(),
           "history_matching": rep, "outputs": list(pb.labels), "z": pb.z.tolist(), "obs_sd": OBS_SD,
           "disc_bands": DISC_BANDS, "marks": MARKS}
    if not H.waves or H.waves[-1].nroy_fraction == 0.0:
        out.update(ok=False, seconds=time.time() - t)
        return out
    last = H.waves[-1]
    floor = H._stoch_floor(last)
    sigma = np.sqrt(pb.var_obs + pb.var_disc)
    rng_e = np.random.default_rng(seed + 5)

    def em_sim(th):
        mu, var = last.emulator.predict(th)
        return mu + np.sqrt(var + floor[None]) * rng_e.standard_normal(mu.shape)

    def log_prior(th):
        inbox = np.all((th >= S.box[:, 0]) & (th <= S.box[:, 1]), 1)
        lp = np.full(len(th), -np.inf)
        if inbox.any():
            lp[np.where(inbox)[0][H.nroy(th[inbox])]] = 0.0
        return lp
    post = abc.abc_smc(lambda n, rng: H.sample(n), log_prior, em_sim, pb.z, sigma, n=particles, seed=seed + 1, log=log)
    theta = post["theta"]
    out["theta"] = theta
    out["abc"] = {k: post[k] for k in ("generations", "simulations", "eps_final", "converged")}
    out["abc"]["simulator"] = "emulator (last wave) + its variance + the ensemble's stochastic variance"
    out["posterior_summary"] = {n: {"p10": float(np.percentile(theta[:, j], 10)), "p50": float(np.median(theta[:, j])),
                                    "p90": float(np.percentile(theta[:, j], 90)),
                                    "width_vs_prior": float(np.subtract(*np.percentile(theta[:, j], [90, 10]))
                                                            / (0.8 * (S.box[j, 1] - S.box[j, 0])))}
                                for j, n in enumerate(S.names)}
    # comprobación con el simulador y pares de la discrepancia: R miembros, cada uno con su θ del posterior
    rng = np.random.default_rng(seed + 9)
    Rp = 32
    pick = theta[rng.choice(len(theta), Rp, replace=False)]
    calib = {n: pick[:, k] for k, n in enumerate(S.names)}
    runs = S._run([(w, origin, Rp, seed + 77 + w, calib) for w in S.windows])
    pairs = {v: {"h": [], "model": [], "obs": []} for v in VARS}
    check = []
    for w, r in zip(S.windows, runs):
        for v in VARS:
            m, tr = r[v]
            g = _level(v, m) - _level(v, tr[0])
            og = _level(v, tr) - _level(v, tr[0])
            med = np.median(g, 0)
            for hh in range(1, len(og)):
                if np.isfinite(og[hh]):
                    pairs[v]["h"].append(hh); pairs[v]["model"].append(med[hh]); pairs[v]["obs"].append(og[hh])
            q10, q90 = np.percentile(g, [10, 90], 0)
            check.append(float(np.mean(((og >= q10) & (og <= q90))[1:][np.isfinite(og[1:])])))
    out["pseudo_hindcast_inside_share_at_posterior"] = float(np.mean(check))
    disc_rep = {}
    for v in VARS:
        try:
            gp = discrepancy.fit(pairs[v]["h"], pairs[v]["model"], pairs[v]["obs"], MARKS[v], seed=seed)
            disc_rep[v] = discrepancy.report(gp, DISC_BANDS[v])
        except Exception as e:                       # noqa: BLE001
            disc_rep[v] = {"verdict": "BLOCKED", "reason": f"{type(e).__name__}: {e}"}
    out["discrepancy"] = {v: {k: np.asarray(x, float) for k, x in pairs[v].items()} | {"mark": MARKS[v]} for v in VARS}
    out["discrepancy_report"] = disc_rep
    edge = hm.edge_knobs(S.names, theta, S.box)
    out["knobs_at_the_box_edge"] = edge
    out["ok"] = True
    out["seconds"] = time.time() - t
    log(f"L1 posterior (origin {origin}): " + "; ".join(f"{n} {v['p50']:+.4f} [{v['p10']:+.4f}, {v['p90']:+.4f}]"
                                                         for n, v in out["posterior_summary"].items()))
    return out


# ------------------------------------------------------------------ W4 calibrado
def hindcast(post: dict, end: int = END, R: int = 64, seed: int = 0, draws: int = 4, log=print) -> dict:
    """Hindcast O -> ``end`` con el L1 F1-F5: sin calibrar, con el posterior (θ por miembro) y con
    posterior + discrepancia. Banda p10-p90; criterio de W4 >= 80 % de los años dentro."""
    origin = post["origin"]
    rng = np.random.default_rng(seed)
    theta = post["theta"]
    pick = theta[rng.choice(len(theta), R, replace=len(theta) < R)]
    calib = {n: pick[:, k] for k, n in enumerate(post["names"])}
    raw, cal = (_task((origin, end, R, seed + origin, c)) for c in (None, calib))
    years = np.arange(origin, end + 1)
    out = {"origin": origin, "end": end, "R": R, "years": years.tolist(), "variables": {},
           "leak": "FaIR posterior constrained with observed GMST up to 2023 (F4); nothing else after the origin"}
    for v in VARS:
        tr = raw[v][1]
        og = _level(v, tr) - _level(v, tr[0])
        row = {"unit": UNITS[v], "obs": [float(x) if np.isfinite(x) else None for x in tr]}
        d = post["discrepancy"][v]
        gp = None
        try:
            gp = discrepancy.fit(d["h"], d["model"], d["obs"], d["mark"], seed=0)
        except Exception as e:                       # noqa: BLE001
            row["discrepancy_error"] = f"{type(e).__name__}: {e}"
        kinds = {"uncalibrated": _level(v, raw[v][0]) - _level(v, tr[0]),
                 "posterior": _level(v, cal[v][0]) - _level(v, tr[0])}
        # el método congelado con el desarrollo del L0 (calib.l0r.predictive): discrepancia DECLARADA + dato
        sd = np.sqrt(disc_sd(v, np.arange(len(years))) ** 2 + OBS_SD[v] ** 2)
        g = kinds["posterior"]
        kinds["posterior_declared"] = (g[None] + sd[None, None] * rng.standard_normal((draws,) + g.shape)).reshape(-1, g.shape[1])
        if gp is not None:                           # informativo: la AR1 ajustada en las ventanas
            kinds["posterior_ar1"] = discrepancy.sample(gp, np.arange(len(years)), g, rng, draws)
        good = np.isfinite(og)
        good[0] = False
        for k, e in kinds.items():
            q10, q50, q90 = np.percentile(e, [10, 50, 90], 0)
            ins = (og >= q10) & (og <= q90)
            back = (lambda x: x + tr[0]) if v == "gmst" else (lambda x: np.exp(x) * tr[0])
            last = int(np.where(good)[0][-1])
            row[k] = {"p10": back(q10).tolist(), "p50": back(q50).tolist(), "p90": back(q90).tolist(),
                      "inside_share": float(ins[good].mean()), "n_years": int(good.sum()),
                      "err_p50_end": float(q50[last] - og[last]), "end_year": int(years[last])}
        out["variables"][v] = row
    best = "posterior_declared"
    out["ok"] = all(r.get(best, r["posterior"])["inside_share"] >= 0.80 for r in out["variables"].values())
    log(f"W4 calibrated hindcast {origin}->{end}: " + "; ".join(
        f"{v} {r['uncalibrated']['inside_share']:.0%} -> {r['posterior']['inside_share']:.0%} -> "
        f"{r.get(best, {}).get('inside_share', float('nan')):.0%}" for v, r in out["variables"].items()))
    return out


def jsonable(post: dict) -> dict:
    return {k: v for k, v in post.items() if k not in ("theta", "discrepancy")}
