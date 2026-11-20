"""Bandas de incertidumbre del L0 vigente (fase 5): el conjunto de modelos que la historia no descarta.

Una sola trayectoria dice poco cuando lo que decide el siglo no se pudo estimar. Aquí el modelo se corre muchas
veces, y cada miembro del conjunto difiere en tres cosas:

1. **Lo que la historia fija a medias** (las cinco perillas de ``technology``) y **lo que no fija** (los ejes de
   ``AXES``: los dos de la curva de coste de los recursos, suelos de eficiencia y de contaminación, techo del
   rendimiento, daño climático). Se muestrean de sus rangos declarados y se quedan los miembros cuyo ajuste a la historia, de
   1971 al corte, no es peor que ``TOLERANCE`` veces (el doble) el mejor ajuste en NINGUNO de los cinco
   indicadores. Es descarte, no ponderación: un miembro o reproduce la historia o no. Como en diez dimensiones
   casi ningún punto de la caja inicial pasa (6 de 6.000 con 1,5 veces: medido), se muestrea por oleadas: cada
   una, uniforme en la caja que envuelve a los que pasaron la anterior con un listón más laxo (``WAVES``).
2. **Los coeficientes estimados**: cada miembro lleva una versión de las ecuaciones de fecundidad y esperanza de
   vida (países remuestreados) y otra de las rectas de carbono y clima (años remuestreados por bloques).
3. **El error del modelo**: aunque todos los parámetros fueran ciertos, el modelo se equivoca. Ese error se mide
   en la partición ANTERIOR (el modelo estimado con datos hasta ``previous`` y comparado con los años hasta el
   corte) como una deriva por año de horizonte, y se suma a cada miembro. No se retroalimenta al modelo.

Las reglas de ``equations.Rules`` no se muestrean: son decisiones.

Rangos de los ejes [S]. Los recursos no renovables ya no son un stock con un factor (V5: de 0,7 a 3 veces el de
World3) sino la curva de coste de ``resources`` (V6), con dos ejes que la historia no puede fijar porque solo
actúan después del último dato: ``substitution`` (de 0, los combustibles mantienen su cuota, a 1, se extrae
siempre lo más barato) y ``resource_scale`` (lo que queda de cada grado, de la mitad al doble de lo evaluado).
El resto son cajas alrededor de los valores fijados en V3 y V4.

El conjunto aceptado viaja en ``world4/models/l0v_posterior.npz`` (parámetros del modelo, sin datos de terceros).
Se reconstruye con ``python -m world4.core.l0v.ensemble``.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

import numpy as np

KNOBS = {"r_yield": (0.004, 0.045), "r_resource": (0.02, 0.09), "r_pollution": (0.01, 0.07), "kappa": (-0.8, 0.6),
         "land_restraint": (0.0, 0.6)}
AXES = {"substitution": (0.0, 1.0), "resource_scale": (0.5, 2.0), "resource_floor": (0.05, 0.35), "pollution_floor": (0.05, 0.35),
        "yield_cap": (1.6, 3.0), "yield_damage": (0.02, 0.08)}
NAMES = tuple(KNOBS) + tuple(AXES)
LOG_UNIFORM = ("resource_scale",)
TOLERANCE = 2.0
WAVES = (6.0, 3.5, 2.5, TOLERANCE)                 # listón de cada oleada, en veces el mejor ajuste
MARGIN = 0.05                                      # holgura de la caja de una oleada, en fracción de su anchura
KAPPA_LEVELS = 15
PER_LEVEL = 400
LAST_WAVE = 5                                      # la última oleada, tantas veces mayor: es la que se guarda
CUTOFFS = (2000, 2023)
PREVIOUS = {2000: 1990, 2023: 2000}               # la partición con la que se mide el error del modelo
SCORED = ("population", "life_expectancy", "fertility", "birth_rate", "death_rate", "output", "food_pc", "land_yield",
          "arable_land", "resource_use", "co2", "gmst")
ADDITIVE = ("gmst",)                               # error en sus unidades; el resto, relativo
EXTRA = ("iopc", "sopc", "fpc", "nrfr", "fcaor", "ppolx", "emissions")   # series sin dato con el que medir su error
SUBSISTENCE = 230.0                                # ``sfpc`` de World3, kg de equivalente vegetal por persona y año


def _path():
    from world4 import home
    return home.PACKAGE_MODELS / "l0v_posterior.npz"


# ------------------------------------------------------------------ muestreo y descarte
def sample(per_level: int = PER_LEVEL, levels: int = KAPPA_LEVELS, seed: int = 0, box: dict | None = None) -> np.ndarray:
    """Hipercubo latino en ``box`` (por defecto, los rangos de ``KNOBS`` y ``AXES``):
    ``(levels * per_level, len(NAMES))``. ``kappa`` va en niveles porque es una tabla de World3 y todos los miembros
    de una corrida vectorizada la comparten."""
    from scipy.stats import qmc
    box = box or {**KNOBS, **AXES}
    out = []
    for k, kappa in enumerate(np.linspace(*box["kappa"], levels)):
        u = qmc.LatinHypercube(d=len(NAMES), seed=seed + k).random(per_level)
        th = np.empty_like(u)
        for j, name in enumerate(NAMES):
            lo, hi = box[name]
            th[:, j] = np.exp(np.log(lo) + u[:, j] * (np.log(hi) - np.log(lo))) if name in LOG_UNIFORM \
                else lo + u[:, j] * (hi - lo)
        th[:, NAMES.index("kappa")] = kappa
        out.append(th)
    return np.concatenate(out)


def tech_of(theta, base=None):
    """El ``Tech`` de uno o varios miembros (``theta``: ``(len(NAMES),)`` o ``(n, len(NAMES))``, misma ``kappa``)."""
    from world4.core.l0v import technology
    th = np.asarray(theta, float)
    col = lambda name: th[..., NAMES.index(name)]  # noqa: E731
    kappa = float(np.unique(col("kappa"))[0]) if th.ndim > 1 else float(col("kappa"))
    base = base or technology.Tech(**technology.ESTIMATED)
    return replace(base, r_yield=col("r_yield"), r_resource=col("r_resource"), r_pollution=col("r_pollution"),
                   kappa=kappa, land_restraint=col("land_restraint"), yield_cap=col("yield_cap"),
                   resource_floor=col("resource_floor"), pollution_floor=col("pollution_floor"))


def resources_of(theta, cutoff: int | None = None):
    """Los ``Resources`` de un miembro, con la mezcla de combustibles observada hasta ``cutoff``."""
    from world4.core.l0v import resources
    th = np.asarray(theta, float)
    return resources.Resources(substitution=float(th[NAMES.index("substitution")]),
                               scale=float(th[NAMES.index("resource_scale")]), cutoff=cutoff)


def history_errors(theta, cutoff: int, first: int = 1971) -> np.ndarray:
    """Error cuadrático medio en logaritmos de los cinco indicadores de ``technology.FIT``, de ``first`` a
    ``cutoff``, de cada miembro: ``(n, 5)``. La población es la observada. Una corrida vectorizada por ``kappa``.
    Los recursos son la curva central: sus dos ejes solo actúan después del último dato."""
    from world4.core.l0v import model, technology
    theta = np.asarray(theta, float)
    inputs = model.world_inputs()
    oy, obs = technology._observed()
    names = [f[0] for f in technology.FIT]
    out = np.empty((len(theta), len(names)))
    kcol = theta[:, NAMES.index("kappa")]
    for kappa in np.unique(kcol):
        m = np.where(kcol == kappa)[0]
        r, _ = model.run(technology=tech_of(theta[m]), n=len(m), fertility="observed", mortality="observed",
                         t1=float(cutoff) + 1.0, inputs=inputs, resources="estimated", cutoff=cutoff,
                         record=("ly", "nr", "ppgr", "io", "so", "al", "pop"))
        t = np.asarray(r.time, float)
        ok = np.isclose(t, np.round(t))
        years = np.round(t[ok]).astype(int)
        nr = r["nr"][ok]
        use = np.empty_like(nr)
        use[1:-1] = (nr[:-2] - nr[2:]) / 2.0
        use[0], use[-1] = nr[0] - nr[1], nr[-2] - nr[-1]
        series = {"land_yield": r["ly"][ok], "resource_use": use, "pollution_generation": r["ppgr"][ok],
                  "output": r["io"][ok] + r["so"][ok], "arable_land": r["al"][ok] / 1e6}
        i70 = int(np.where(years == 1970)[0][0])
        for j, name in enumerate(names):
            mod = series[name] if name in technology.LEVELS else series[name] / series[name][i70]
            sel = (oy >= first) & (oy <= cutoff) & np.isfinite(obs[name])
            at = np.searchsorted(years, oy[sel])
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.log(mod[at]) - np.log(obs[name][sel])[:, None]
            out[m, j] = np.sqrt(np.nanmean(d ** 2, 0))
    return np.where(np.isfinite(out), out, np.inf)


def accept(errors: np.ndarray, cutoff: int, tolerance: float = TOLERANCE) -> np.ndarray:
    """Los miembros que la historia no descarta: en cada indicador, error <= ``tolerance`` x el del mejor ajuste
    guardado para ese corte."""
    from world4.core.l0v import technology
    best = technology.stored()[technology.BLOCKS[True]][str(cutoff)]["rms_pct"]
    limit = np.array([best[f[0]] for f in technology.FIT]) / 100.0 * tolerance
    return np.all(errors <= limit, axis=1)


def _shrink(theta, prior: dict) -> dict:
    """La caja que envuelve a ``theta``, con holgura y sin salirse de ``prior``."""
    box = {}
    for j, name in enumerate(NAMES):
        lo, hi = float(theta[:, j].min()), float(theta[:, j].max())
        pad = MARGIN * (prior[name][1] - prior[name][0])
        box[name] = (max(prior[name][0], lo - pad), min(prior[name][1], hi + pad))
    return box


def build(cutoffs=CUTOFFS, seed: int = 0, log=print) -> dict:
    """Muestrea por oleadas, corre la historia, descarta y guarda el conjunto aceptado de cada corte."""
    prior = {**KNOBS, **AXES}
    out = {"names": np.array(NAMES)}
    for c in cutoffs:
        box, total = prior, 0
        for w, tol in enumerate(WAVES):
            theta = sample(PER_LEVEL * (LAST_WAVE if w == len(WAVES) - 1 else 1), seed=seed + 1000 * w, box=box)
            err = history_errors(theta, c)
            ok = accept(err, c, tol)
            total += len(theta)
            if log:
                log(f"cutoff {c}, wave {w + 1} (within {tol} x the best fit): {int(ok.sum())} of {len(theta)}")
            if not ok.any():
                raise RuntimeError(f"cutoff {c}: wave {w + 1} ruled out every member; widen WAVES")
            box = _shrink(theta[ok], prior)
        out[f"theta_{c}"], out[f"errors_{c}"] = theta[ok], err[ok]
        out[f"sampled_{c}"] = np.array([total])
    np.savez_compressed(_path(), **out)
    posterior.cache_clear()
    return out


@lru_cache(maxsize=4)
def posterior(cutoff: int = 2023) -> dict:
    """El conjunto aceptado para un corte: ``{"names", "theta" (m, d), "errors" (m, 5), "sampled"}``."""
    z = np.load(_path())
    if f"theta_{cutoff}" not in z.files:
        raise ValueError(f"no stored ensemble for cutoff {cutoff}; available {CUTOFFS} (ensemble.build)")
    return {"names": tuple(str(n) for n in z["names"]), "theta": z[f"theta_{cutoff}"], "errors": z[f"errors_{cutoff}"],
            "sampled": int(z[f"sampled_{cutoff}"][0])}


# ------------------------------------------------------------------ el error del modelo
def _member_series(run, carbon_par, origin):
    """Las series de una corrida que entran en las bandas, en años enteros."""
    from world4.core.l0v import carbon
    from world4.validate import world3
    m = world3.model_series(run)
    d = carbon.diagnose(run, carbon_par, origin)
    t = np.asarray(run.time, float)
    ok = np.isclose(t, np.round(t))
    out = {k: m[k] for k in SCORED if k in m}
    out["co2"] = np.interp(m["years"], d["years"], d["co2_ppm"]) - world3.PREINDUSTRIAL_PPM
    out["gmst"] = np.interp(m["years"], d["years"], d["gmst"])
    out["emissions"] = np.interp(m["years"], d["years"], d["emissions"])
    for k in ("iopc", "sopc", "fpc", "nrfr", "fcaor", "ppolx"):
        out[k] = np.asarray(run[k], float)[ok, 0]
    return m["years"], out


def _observed_series():
    from world4.data import world
    from world4.validate import world3
    o = world3.observed()
    w = world.load()
    return {**o, "gmst": w["gmst"]}


def discrepancy(cutoff: int = 2023, groups: str = "regions") -> dict:
    """Deriva del modelo por año de horizonte, medida en la partición anterior: el modelo central estimado con
    datos <= ``PREVIOUS[cutoff]`` frente a lo observado hasta ``cutoff``. Por indicador, la pendiente que mejor
    explica el error como proporcional al horizonte, ``raíz(Σ error² / Σ horizonte²)``: relativa (logaritmo) o, en
    ``ADDITIVE``, en sus unidades. Pesa los horizontes largos; dividir cada error por su horizonte daba al ruido de
    un año suelto el peso de una deriva (la temperatura salía a 0,027 °C por año: 2 °C de banda en 2100)."""
    from world4.core.l0v import carbon, model
    from world4.validate import world3
    prev = PREVIOUS[cutoff]
    r, _ = model.run(groups=groups, origin=prev, cutoff=prev, technology="estimated", climate="estimated",
                     resources="estimated", t1=float(cutoff))
    years, mod = _member_series(r, carbon.params(prev), prev)
    obs = _observed_series()
    out = {"previous": prev, "cutoff": cutoff, "per_year": {}}
    for ind in world3.INDICATORS + (world3.Indicator("gmst", "global temperature", "gmst", False),):
        k = ind.key
        oy, ov = obs["years"], obs[k]
        sel = (oy > prev) & (oy <= cutoff) & np.isfinite(ov) & np.isin(oy, years)
        if not sel.any():
            continue
        m, o = np.interp(oy, years, mod[k]), ov
        if ind.index:
            i70 = int(np.where(oy == world3.BASE_YEAR)[0][0])
            m, o = m / np.interp(world3.BASE_YEAR, years, mod[k]), o / o[i70]
        h = (oy[sel] - prev).astype(float)
        d = (m[sel] - o[sel]) if k in ADDITIVE else np.log(m[sel] / o[sel])
        good = np.isfinite(d)
        if good.any():
            out["per_year"][k] = float(np.sqrt((d[good] ** 2).sum() / (h[good] ** 2).sum()))
    return out


# ------------------------------------------------------------------ proyección y bandas
def project(origin: int = 2023, cutoff: int | None = None, members: int = 60, seed: int = 0, groups: str = "regions",
            t1: float = 2100.0) -> dict:
    """Corre ``members`` miembros del conjunto aceptado desde ``origin`` (``cutoff``: por defecto, ``origin``).
    Devuelve ``{"years", "theta" (m, d), "series": {nombre: (m, años)}}``, sin el error del modelo."""
    from world4.core.l0v import carbon, equations, model
    cutoff = int(origin if cutoff is None else cutoff)
    post = posterior(cutoff)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(post["theta"]), size=min(members, len(post["theta"])), replace=False)
    eq_draws = equations.bootstrap(cutoff, len(pick), seed + 1)
    c_draws = carbon.bootstrap(cutoff, len(pick), seed + 2)
    years, series = None, {}
    for j, i in enumerate(pick):
        th = post["theta"][i]
        cpar = replace(c_draws[j], yield_damage=float(th[NAMES.index("yield_damage")]))
        r, _ = model.run(groups=groups, origin=origin, cutoff=cutoff, par=eq_draws[j], technology=tech_of(th),
                         climate=cpar, resources=resources_of(th, cutoff), t1=t1)
        years, s = _member_series(r, cpar, origin)
        for k, v in s.items():
            series.setdefault(k, []).append(v)
    return {"years": years, "origin": origin, "cutoff": cutoff, "theta": post["theta"][pick],
            "series": {k: np.stack(v) for k, v in series.items()}}


def with_model_error(proj: dict, disc: dict, seed: int = 0) -> dict:
    """Las series de ``project`` con la deriva del modelo sumada: a cada miembro e indicador, un sorteo normal por
    la deriva por año y por los años desde el origen. Las series de ``EXTRA`` no tienen dato y quedan igual. La
    fecundidad no baja de su suelo ni la esperanza de vida pasa de su techo (reglas de ``equations``)."""
    from world4.core.l0v import equations
    limits = {"fertility": (equations.F_LO, None), "life_expectancy": (None, equations.E_HI)}
    rng = np.random.default_rng(seed + 7)
    h = np.maximum(proj["years"] - proj["origin"], 0).astype(float)
    out = {}
    for k, v in proj["series"].items():
        s = disc["per_year"].get(k)
        if s is None:
            out[k] = v
            continue
        z = rng.standard_normal(len(v))[:, None]
        out[k] = v + z * s * h if k in ADDITIVE else v * np.exp(z * s * h)
        if k in limits:
            out[k] = np.clip(out[k], *limits[k])
    return out


def bands(proj: dict, disc: dict | None = None, quantiles=(10, 50, 90), seed: int = 0) -> dict:
    """Percentiles por año de cada serie: ``{"years", nombre: {"p10": ..., "p50": ..., "p90": ...}}``."""
    series = proj["series"] if disc is None else with_model_error(proj, disc, seed)
    out = {"years": proj["years"]}
    for k, v in series.items():
        q = np.percentile(v, quantiles, axis=0)
        out[k] = {f"p{p}": q[i] for i, p in enumerate(quantiles)}
    return out


def coverage(origin: int = 2000, members: int = 60, seed: int = 0, last: int = 2023, groups: str = "regions") -> dict:
    """La prueba de las bandas: el conjunto no descartado con datos <= ``origin`` y su error del modelo medido en
    la partición anterior, proyectado hasta ``last``. Por indicador, la fracción de los años con dato de
    ``origin + 1`` a ``last`` que caen dentro de p10-p90 (debería rondar 0,8), con y sin el error del modelo."""
    from world4.validate import world3
    proj = project(origin, origin, members, seed, groups, t1=float(last))
    disc = discrepancy(origin, groups)
    obs = _observed_series()
    out = {"origin": origin, "last": last, "members": int(len(proj["theta"])), "indicators": {}}
    index = {i.key: i.index for i in world3.INDICATORS}
    for label, b in (("with_model_error", bands(proj, disc, seed=seed)), ("parameters_only", bands(proj))):
        for k in SCORED:
            if k not in b:
                continue
            oy, ov = obs["years"], obs[k].astype(float)
            sel = (oy > origin) & (oy <= last) & np.isfinite(ov)
            if not sel.any():
                continue
            lo, hi = np.interp(oy[sel], b["years"], b[k]["p10"]), np.interp(oy[sel], b["years"], b[k]["p90"])
            o = ov[sel]
            if index.get(k):                                               # índice 1970 = 1: mismo anclaje que el modelo
                o = o / ov[oy == world3.BASE_YEAR][0]
                base = np.median(np.interp(world3.BASE_YEAR, proj["years"], np.median(proj["series"][k], 0)))
                lo, hi = lo / base, hi / base
            out["indicators"].setdefault(k, {"n": int(sel.sum())})[label] = float(np.mean((o >= lo) & (o <= hi)))
    return out


def summary(proj: dict, disc: dict | None = None, seed: int = 0) -> dict:
    """Los números del conjunto: máximo industrial, de población, de alimentos por persona y de esperanza de vida,
    y varios indicadores en 2050 y 2100, como p10/p50/p90; y qué ejes mueven más el año del máximo industrial y
    los alimentos por persona de 2100 (correlación de rangos). ``life_expectancy_falls`` es la parte de los
    miembros cuya esperanza de vida de 2100 queda más de dos años por debajo de su máximo."""
    from scipy.stats import spearmanr
    years, s = proj["years"], (proj["series"] if disc is None else with_model_error(proj, disc, seed))
    q = lambda v: [float(x) for x in np.percentile(v, (10, 50, 90))]  # noqa: E731
    at = lambda k, y: s[k][:, int(np.where(years == y)[0][0])]  # noqa: E731
    after = years >= proj["origin"]
    ipeak = years[after][np.argmax(proj["series"]["iopc"][:, after], axis=1)]
    ppeak = years[after][np.argmax(s["population"][:, after], axis=1)]
    raw = proj["series"]
    fpeak = years[after][np.argmax(raw["fpc"][:, after], axis=1)]
    lpeak = years[after][np.argmax(raw["life_expectancy"][:, after], axis=1)]
    out = {"members": int(len(proj["theta"])), "origin": proj["origin"],
           "industrial_peak_year": q(ipeak), "industrial_peak_after_2100": float(np.mean(ipeak >= years[-1])),
           "population_peak_year": q(ppeak), "population_peak": q(s["population"][:, after].max(1)),
           "food_peak_year": q(fpeak), "life_expectancy_peak_year": q(lpeak),
           "life_expectancy_falls": float(np.mean(raw["life_expectancy"][:, -1]
                                                  < raw["life_expectancy"][:, after].max(1) - 2.0)),
           "at": {}, "drivers_of_industrial_peak": {}, "drivers_of_food_2100": {}}
    for y in (2050, 2100):
        out["at"][y] = {k: q(at(k, y)) for k in ("population", "life_expectancy", "fertility", "food_pc", "co2", "gmst",
                                                 "nrfr", "fcaor", "ppolx")}
        out["at"][y]["iopc_over_origin"] = q(at("iopc", y) / at("iopc", proj["origin"]))
        out["at"][y]["food_below_subsistence"] = float(np.mean(at("fpc", y) < SUBSISTENCE))
    for j, name in enumerate(NAMES):
        col = proj["theta"][:, j]
        if np.ptp(col) > 0 and np.ptp(ipeak) > 0:
            out["drivers_of_industrial_peak"][name] = float(spearmanr(col, ipeak)[0])
        if np.ptp(col) > 0:
            out["drivers_of_food_2100"][name] = float(spearmanr(col, raw["fpc"][:, -1])[0])
    return out


if __name__ == "__main__":
    build()


def report(origin: int = 2023, members: int = 60, seed: int = 0, test_origin: int = 2000,
           series=("population", "life_expectancy", "fertility", "output", "food_pc", "co2", "gmst", "iopc", "fcaor", "ppolx")) -> dict:
    """Todo lo que se enseña de las bandas: el resumen de la proyección desde ``origin``, sus bandas por año, el
    error del modelo usado y la prueba de cobertura desde ``test_origin``."""
    proj = project(origin, origin, members, seed)
    disc = discrepancy(origin)
    b = bands(proj, disc, seed=seed)
    post = posterior(origin)
    keep = proj["years"] >= origin
    return {"origin": origin, "members": int(len(proj["theta"])),
            "ensemble": {"kept": int(len(post["theta"])), "sampled": post["sampled"], "tolerance": TOLERANCE,
                         "ranges": {n: {"prior": list({**KNOBS, **AXES}[n]),
                                        "kept": [float(x) for x in np.percentile(post["theta"][:, j], (5, 50, 95))]}
                                    for j, n in enumerate(NAMES)}},
            "model_error_per_year": disc,
            "summary": summary(proj, disc, seed),
            "summary_parameters_only": summary(proj),
            "bands": {"years": [int(y) for y in proj["years"][keep]],
                      **{k: {p: [float(x) for x in v[keep]] for p, v in b[k].items()} for k in series}},
            "coverage": coverage(test_origin, members, seed)}
