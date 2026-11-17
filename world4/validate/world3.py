"""Marcador del L0: cuánto se desvía una corrida de World3 de las series mundiales observadas.

Cada indicador empareja una salida del modelo con una serie de ``world4.data.world``:

* **directos** (mismas unidades): población, esperanza de vida, fecundidad, natalidad, mortalidad;
* **por índice** (1970 = 1), porque World3 mide en unidades propias y el dato real es un APODERADO [declarado]:
  producción (industria + servicios) frente al PIB; alimento por persona frente a las kilocalorías por persona en
  equivalente vegetal (las de origen animal cuentan por 7, como en World3);
  rendimiento de la tierra frente al de los cereales; tierra cultivable frente a las tierras de cultivo de HYDE
  (ésta en hectáreas, sin índice); uso de recursos (caída anual del stock ``nr``) frente al
  CO2 de combustibles fósiles; y el CO2 atmosférico sobre 280 ppm que implican las emisiones del modelo
  (``world4.core.l0v.carbon``) frente al de Mauna Loa, en ppm. La contaminación persistente de World3 ya no se
  compara con nada: no tiene equivalente medible.

El error de un indicador es la media de ``|modelo / real - 1|`` en los años de la ventana que tienen dato
(por defecto 2000-2023: hasta 2000 el escenario 1 sigue bien a la población y es después cuando se separa).
Es una medida de ajuste al pasado, no una prueba fuera de muestra: World3 no se calibró con estos datos, pero
tampoco se le oculta nada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BASE_YEAR = 1970
WINDOW = (2000, 2023)
PREINDUSTRIAL_PPM = 280.0


@dataclass(frozen=True)
class Indicator:
    key: str
    label: str
    obs: str                 # serie de world4.data.world
    index: bool              # True: se comparan índices con BASE_YEAR = 1
    proxy: str = ""          # qué apoderado es el dato real, si lo es


INDICATORS = (
    Indicator("population", "population", "pop", False),
    Indicator("life_expectancy", "life expectancy", "le", False),
    Indicator("fertility", "total fertility", "tfr", False),
    Indicator("birth_rate", "crude birth rate", "cbr", False),
    Indicator("death_rate", "crude death rate", "cdr", False),
    Indicator("output", "output (industry + services)", "gdp", True, "world GDP (Maddison)"),
    Indicator("food_pc", "food per person", "food_veq", True, "vegetal-equivalent kcal per person (FAO)"),
    Indicator("land_yield", "land yield", "cereal_yield", True, "cereal yield (World Bank)"),
    Indicator("arable_land", "arable land", "cropland", False, "cropland (HYDE)"),
    Indicator("resource_use", "resource use", "fossil_co2", True, "fossil CO2 emissions"),
    Indicator("co2", "atmospheric CO2 above 280 ppm", "co2_ppm", False, "Mauna Loa; model: emissions from resource use"),
)
HEADLINE = ("population", "life_expectancy", "output", "food_pc", "resource_use", "co2")
ECONOMY_KEYS = ("output", "food_pc", "land_yield", "arable_land", "resource_use", "co2")
# variables de l0.run que el marcador necesita
NEEDS = ("pop", "le", "tf", "cbr", "cdr", "io", "so", "fpc", "ly", "al", "nr")


def _co2_excess(run, years, obs, member):
    """El CO2 sobre 280 ppm que implica la corrida; NaN si no hay series de carbono con que calcularlo."""
    from world4.core.l0v import carbon
    try:
        if obs is None:
            from world4.data import world
            obs = world.load()
        par = carbon.estimate(getattr(run, "cutoff", None), obs)
        d = carbon.diagnose(run, par, getattr(run, "origin", None), obs, member)
    except (KeyError, RuntimeError):
        return np.full(len(years), np.nan)
    return np.interp(years, d["years"], d["co2_ppm"]) - PREINDUSTRIAL_PPM


def model_series(run, member: int = 0, obs: dict | None = None) -> dict:
    """Salidas de una corrida (``World3Run``) en años enteros: ``{"years": ..., indicador: valores}``."""
    t = np.asarray(run.time, float)
    ok = np.isclose(t, np.round(t))
    years = np.round(t[ok]).astype(int)
    g = lambda k: np.asarray(run[k], float)[ok, member]  # noqa: E731
    nr = g("nr")
    use = np.full(len(years), np.nan)
    use[1:-1] = (nr[:-2] - nr[2:]) / (years[2:] - years[:-2])          # diferencia centrada
    use[0] = (nr[0] - nr[1]) / (years[1] - years[0])
    return {"years": years, "population": g("pop") / 1e6, "life_expectancy": g("le"), "fertility": g("tf"),
            "birth_rate": g("cbr"), "death_rate": g("cdr"), "output": g("io") + g("so"), "food_pc": g("fpc"),
            "land_yield": g("ly"), "arable_land": g("al") / 1e6, "resource_use": use,
            "co2": _co2_excess(run, years, obs, member)}


def observed(obs: dict | None = None) -> dict:
    """Las series reales en la forma del marcador (el CO2 atmosférico como exceso sobre 280 ppm)."""
    if obs is None:
        from world4.data import world
        obs = world.load()
    out = {"years": np.asarray(obs["years"], int)}
    for ind in INDICATORS:
        v = np.asarray(obs[ind.obs], float)
        out[ind.key] = v - PREINDUSTRIAL_PPM if ind.obs == "co2_ppm" else v
    return out


def _at(years, values, y):
    i = np.where(years == y)[0]
    return float(values[i[0]]) if len(i) else float("nan")


def paired(run, obs: dict | None = None, member: int = 0) -> dict:
    """Por indicador: años con dato real y, en ellos, modelo y real (como índice si el indicador lo es)."""
    m, o = model_series(run, member, obs), observed(obs)
    out = {}
    for ind in INDICATORS:
        yo, vo = o["years"], o[ind.key]
        ok = np.isfinite(vo) & np.isin(yo, m["years"])
        years = yo[ok]
        real = vo[ok]
        mod = np.array([_at(m["years"], m[ind.key], y) for y in years])
        if ind.index:
            real = real / _at(yo, vo, BASE_YEAR)
            mod = mod / _at(m["years"], m[ind.key], BASE_YEAR)
        good = np.isfinite(mod) & np.isfinite(real)
        out[ind.key] = {"years": years[good], "model": mod[good], "observed": real[good]}
    return out


def score(run, obs: dict | None = None, window=WINDOW, member: int = 0) -> dict:
    """Error por indicador en la ventana: ``mae_pct``, número de años, y modelo y real en el último año con dato."""
    out = {}
    for key, p in paired(run, obs, member).items():
        w = (p["years"] >= window[0]) & (p["years"] <= window[1])
        if not w.any():
            out[key] = {"mae_pct": float("nan"), "n": 0}
            continue
        y, mod, real = p["years"][w], p["model"][w], p["observed"][w]
        out[key] = {"mae_pct": float(100 * np.mean(np.abs(mod / real - 1))), "n": int(w.sum()),
                    "bias_pct": float(100 * np.mean(mod / real - 1)),
                    "last_year": int(y[-1]), "model_last": float(mod[-1]), "observed_last": float(real[-1])}
    return out


def table(runs: dict, obs: dict | None = None, window=WINDOW) -> dict:
    """Varias corridas a la vez: ``{"window", "indicators": [...], "runs": {nombre: score}}``."""
    return {"window": list(window), "base_year": BASE_YEAR,
            "indicators": [{"key": i.key, "label": i.label, "index": i.index, "proxy": i.proxy} for i in INDICATORS],
            "runs": {name: score(r, obs, window) for name, r in runs.items()}}


# ------------------------------------------------------------------ variantes del L0 y su marcador
VARIANTS = {
    "original": None,                                                   # World3-03 tal cual (cuatro cajas)
    "cohorts": {},                                                      # cohortes; fecundidad y e0 de World3
    "cohorts_obs_fertility": {"fertility": "observed"},                 # diagnóstico: fecundidad total real
    "cohorts_obs_mortality": {"mortality": "observed"},                 # diagnóstico: esperanza de vida real
    "cohorts_obs_both": {"fertility": "observed", "mortality": "observed"},      # el motor de cohortes solo
    # V2: 10 regiones con ecuaciones estimadas; historia observada hasta 2000 y, desde ahí, el modelo con datos <= 2000
    "groups": {"groups": "regions", "origin": 2000, "cutoff": 2000},
    # V3: lo mismo con la tecnología estimada (rendimientos, recursos, contaminación, coste de capital, tierra)
    "full": {"groups": "regions", "origin": 2000, "cutoff": 2000, "technology": "estimated", "climate": "estimated"},
}
HINDCAST_KEYS = ("population", "life_expectancy", "fertility", "birth_rate", "death_rate")


def run_variant(name: str, scenario: int = 1, **kw):
    """Una corrida del L0 por nombre de variante; devuelve el ``World3Run``."""
    from world4.core import l0
    if name not in VARIANTS:
        raise ValueError(f"variant must be one of {sorted(VARIANTS)}, got {name!r}")
    if VARIANTS[name] is None:
        return l0.run(scenario=scenario, **kw)
    from world4.core.l0v import model
    return model.run(scenario=scenario, **VARIANTS[name], **kw)[0]


def scoreboard(scenario: int = 1, variants=None, window=WINDOW) -> dict:
    """El marcador de varias variantes del L0 (por defecto, las que los datos instalados permiten)."""
    from world4.data import wpp
    if variants is None:
        variants = tuple(VARIANTS) if wpp.available() else ("original",)
    out = table({v: run_variant(v, scenario) for v in variants}, window=window)
    out["scenario"] = scenario
    return out


def cohort_check(start: int = 1950) -> dict:
    """El motor de cohortes SOLO: con la fecundidad total y la esperanza de vida reales, ¿reproduce la población
    mundial por edad de WPP? Compara a 31 de diciembre de cada año. Errores en % de la población real."""
    from world4.core.l0v import model
    from world4.core.l0v.cohorts import GROUPS
    inp = model.world_inputs()
    _, c = model.run(start=start, fertility="observed", mortality="observed", t1=float(inp.years[-1]) + 1.0,
                     inputs=inp)
    sel = np.isin(c.years, inp.years)
    years = c.years[sel]
    mod = c.popM[sel, 0] + c.popF[sel, 0]
    i = np.searchsorted(inp.years, years)
    real = inp.popM[i] + inp.popF[i]
    err = mod.sum(1) / real.sum(1) - 1
    groups = {g: mod[:, a:b].sum(1) / real[:, a:b].sum(1) - 1 for g, (a, b) in GROUPS.items()}
    return {"years": [int(years[0]), int(years[-1])],
            "total_max_abs_pct": float(100 * np.abs(err).max()), "total_mean_abs_pct": float(100 * np.abs(err).mean()),
            "total_last_pct": float(100 * err[-1]),
            "group_max_abs_pct": {g: float(100 * np.abs(e).max()) for g, e in groups.items()}}


def hindcast(origins=(1980, 1990, 2000, 2010), groups: str = "regions", economy: str = "model",
             scenario: int = 1, last: int = WINDOW[1]) -> dict:
    """Prueba fuera de muestra de la demografía por grupos, con origen móvil: para cada ``origin`` las ecuaciones
    se estiman con datos <= ``origin``, la historia hasta ``origin`` es la observada y el modelo corre solo hasta
    ``last``. Error medio de ``origin + 1`` a ``last`` de la versión estimada y de la que CONGELA la fecundidad y la
    esperanza de vida de cada grupo en su valor del origen. ``economy="observed"`` aísla la demografía de los
    errores de la economía de World3."""
    from world4.core.l0v import equations, model
    out = {"groups": groups, "economy": economy, "last": last, "origins": {}}
    for origin in origins:
        res = {}
        for name, par in (("estimated", None), ("frozen", {**equations.ZERO, "floor": equations.F_LO})):
            r, _ = model.run(scenario=scenario, groups=groups, origin=origin, cutoff=origin, par=par, economy=economy,
                             t1=float(last))
            sc = score(r, window=(origin + 1, last))
            res[name] = {k: {"mae_pct": sc[k]["mae_pct"], "model_last": sc[k]["model_last"],
                             "observed_last": sc[k]["observed_last"]} for k in HINDCAST_KEYS}
        out["origins"][int(origin)] = res
    return out


def group_check(groups: str = "regions", start: int = 1950) -> dict:
    """El motor por grupos SOLO: cohortes de cada grupo con su fecundidad, su esperanza de vida y su migración
    observadas. Error de la población mundial y de la de cada grupo frente a WPP, a 31 de diciembre (%)."""
    from world4.core.l0v import groups as _groups
    from world4.core.l0v import model
    gi = _groups.inputs(groups)
    _, g = model.run(groups=gi, start=start, origin=int(gi.years[-1]), t1=float(gi.years[-1]) + 1.0)
    sel = np.isin(g.years, gi.years)
    i = np.searchsorted(gi.years, g.years[sel])
    mod = g.popM[sel].sum(2) + g.popF[sel].sum(2)                       # (años, G)
    real = gi.popM[i].sum(2) + gi.popF[i].sum(2)
    world = mod.sum(1) / real.sum(1) - 1
    return {"groups": list(gi.names), "years": [int(g.years[sel][0]), int(g.years[sel][-1])],
            "world_mean_abs_pct": float(100 * np.abs(world).mean()), "world_max_abs_pct": float(100 * np.abs(world).max()),
            "group_last_pct": {n: float(100 * (mod[-1, k] / real[-1, k] - 1)) for k, n in enumerate(gi.names)}}


def technology_hindcast(origins=(1990, 2000, 2010), groups: str = "regions", scenario: int = 1,
                        last: int = WINDOW[1]) -> dict:
    """Prueba fuera de muestra del modelo completo: demografía por grupos y tecnología, las dos estimadas con datos
    <= ``origin`` (los cortes guardados en ``l0v_technology.json``). Error medio de ``origin + 1`` a ``last`` de
    todos los indicadores, con la tecnología estimada y con la de World3 (apagada en el escenario 1)."""
    from world4.core.l0v import model
    out = {"groups": groups, "last": last, "origins": {}}
    for origin in origins:
        res = {}
        from world4.core.l0v import carbon
        for name, tech in (("technology", "estimated"), ("world3", None)):
            r, _ = model.run(scenario=scenario, groups=groups, origin=origin, cutoff=origin, technology=tech,
                             climate=tech, t1=float(last))
            sc = score(r, window=(origin + 1, last))
            res[name] = {k: {"mae_pct": v["mae_pct"], "model_last": v.get("model_last"),
                             "observed_last": v.get("observed_last")} for k, v in sc.items()}
            res[name]["climate"] = carbon.check(r, carbon.params(origin), origin, window=(origin + 1, last))
        out["origins"][int(origin)] = res
    return out
