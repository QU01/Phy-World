"""Superficie pública de World4: la que usan por igual la CLI, el MCP, el addon y Prime Radiant.

F0 la sirve con lo que hay: el L0 (World3-03) y el L1 (Atlas portado, sin cambios); F1 le acopla
la población y el capital humano endógenos (``world4.core.coupling``). Cada
resultado lleva sus límites DECLARADOS en ``caveats`` y ``exploratory``; ninguna cifra sale sin
unidad, y ninguna banda se presenta como calibrada (no lo está hasta F0.5, criterio W8).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date

import numpy as np

from world4 import MODEL_REVISION

# Horizonte con validación fuera de muestra del Atlas (49 orígenes, 1, 5 y 10 años)
VALIDATED_HORIZON_YEARS = 10
L1_FIRST_ORIGIN, L1_LAST_YEAR = 1970, 2019

# Defectos conocidos del L1, por identificador del informe de F0 / hoja de ruta §2. D1 y D2 se
# cierran en F1: la población y el capital humano después del origen salen de los sectores
# endógenos (world4.sectors), y lo que esos sectores no tienen se declara aquí.
L1_CAVEATS = (
    "D7: p10-p90 bands too narrow (cover 60-70 % instead of 80 %); not recalibrated here (the structural "
    "estimator recalibrates, W8)",
    "vintages: the Atlas uses year-t0 data at origin t0, actually published 1-3 years later",
    "F1 population: age structure and age rates at the origin are WPP 2024 (revised history; revision leak not "
    "measured); with F5 the calorie shortfall enters e0 but is not identified (coefficient 0 by its sign before ~2010); no disaster term in mortality; the free hindcast 1990-2023 overshoots "
    "world population by up to ~2 % (fertility falls faster than the equation fitted to <= 1990 says)",
    "F1 human capital: schooling profile by cohort at the origin is assumed linear [S]",
    "F3 energy: primary-energy mix by source from estimated diffusion; EROI and learning rates are literature "
    "values [S, not verified]; wind+solar learning is not identifiable from data before ~2010; no electricity, "
    "storage or vehicles; the 1990 hindcast of the mix is worse than keeping the origin's mix",
    "F4 climate: FaIR v2.2.4 with the calibrated, constrained ensemble 1.6.0 (one configuration per member); the "
    "posterior was constrained with observed GMST up to 2023, so climate parameters are not out of sample in "
    "hindcasts; non-CO2 emissions follow declared rules (energy-linked species with fossil CO2; with F5, N2O/NH3 follow "
    "cropland N inputs and 35 % of CH4 food supply; halogens with their trend) [S]; land-use CO2 is persistence plus, with F5 "
    "and origins >= 1999, the modelled forest loss; no future volcanoes; "
    "tipping elements off",
    "F5 food: cereals, cropland, pasture, forest, mineral N, irrigation and food supply per country from equations "
    "estimated <= as_of (FAOSTAT); the 1990 hindcast extrapolates the Green Revolution (cereals +41 %, cropland "
    "+14 % by 2023) and mineral N converges to the origin's frontier by assumption [S]; the food -> mortality term "
    "is estimated but not identified (0 by its sign before ~2010); food does not enter the Atlas's PSI; footprint "
    "and biocapacity are computed from the model's flows (no MRIO)",
    "F2 resources: psi_t comes from 4 world markets (oil, gas, coal, metals); ultimate resource = reserves known at "
    "the origin x a reserve-growth factor in [1, 3] [S]; stationary real prices [S]; no critical metals, no metal "
    "depletion, no substitutes until F3; psi bands cover ~60 % instead of 80 %",
)

@dataclass
class Observable:
    variable: str
    unit: str
    fidelity: str
    regions: str
    validated_horizon_years: int | None
    resolution_source: str
    note: str = ""


# variable pública -> (salida del Atlas, unidad, transformación, fuente con la que se resolvería)
_L1_VARS = {
    "gdp_pc": ("ly", "2017 USD at constant national prices per person (PWT rgdpna/pop)", np.exp,
               "PWT 10.01 rgdpna/pop"),
    "log_gdp_pc": ("ly", "ln(2017 USD at constant national prices per person)", None, "PWT 10.01 rgdpna/pop"),
    "conflict_hazard": ("hazard", "annual probability of civil conflict (>= 25 deaths)", None, "UCDP/PRIO ACD"),
    "debt_gdp": ("debt", "public debt / GDP (fraction)", None, "Global Macro Database / FMI"),
    "resource_rents_gdp": ("res_share", "resource rents / GDP (fraction)", None, "WDI NY.GDP.TOTL.RT.ZS"),
    "democracy": ("demo", "probability of a democratic regime (V-Dem RoW >= 2)", None, "V-Dem RoW"),
    "gmst": ("Tg", "degC above 1850-1900 (HadCRUT5 basis)", None, "HadCRUT5"),
    "co2_emissions_world": ("climate_co2_ffi", "GtCO2/year, fossil and industrial", None, "Global Carbon Budget / CMIP7"),
    "co2_concentration": ("climate_co2_conc", "ppm CO2", None, "NOAA GML / CMIP7"),
    "cereal_production": ("food_prod", "t of cereals per year", None, "FAOSTAT QCL (Cereals, primary)"),
    "cropland": ("food_cropland", "1000 ha of cropland", None, "FAOSTAT RL"),
    "food_supply": ("food_kcal", "kcal per person per day", None, "FAOSTAT FBS"),
    "resource_rent_share_world": ("psi", "world resource rents / world GDP (fraction)", None,
                                  "WDI NY.GDP.TOTL.RT.ZS, GDP-weighted"),
    "oil_price": ("price_oil", "real USD 2019 per barrel", None, "EI / World Bank CMO"),
    "primary_energy": ("energy_PE", "TWh of primary energy (substitution method)", None,
                       "Energy Institute Statistical Review (OWID)"),
    "population": ("pop", "million persons (PWT pop)", None,
                   "PWT 10.01 pop / UN WPP"),
    "human_capital": ("hc", "PWT human-capital index", None,
                      "PWT 10.01 hc"),
}


WORLD_KEYS = ("Tg", "psi", "price_oil", "climate_co2_ffi", "climate_co2_conc")


def catalog() -> list[dict]:
    rows = [asdict(Observable(name, unit, "L1", "180 countries (ISO3)" if v not in WORLD_KEYS else "world",
                              VALIDATED_HORIZON_YEARS, src)) for name, (v, unit, _, src) in _L1_VARS.items()]
    for name, unit in (("pop", "persons"), ("f", "vegetable-equivalent kg/year"), ("io", "1968 USD/year"),
                       ("nr", "resource units"), ("ppolx", "index, 1970 = 1"), ("hwi", "index 0-1"),
                       ("hef", "index, 1 = Earth capacity")):
        rows.append(asdict(Observable(name, unit, "L0", "world (World3-03)", None, "none",
                                      "behaviour mode, not a forecast")))
    return rows


@dataclass
class Distribution:
    variable: str
    unit: str
    region: str
    as_of: int
    target: int
    p10: float
    p50: float
    p90: float
    mean: float
    members: int
    fidelity: str
    calibrated: bool = False
    exploratory: bool = False
    caveats: list = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)


class QueryError(ValueError):
    pass


def run_l0(scenario: int = 1, **kw):
    from world4.core import l0
    return l0.run(scenario=scenario, **kw)


def _year(x) -> int:
    return x.year if isinstance(x, date) else int(x)


def query(variable: str, region: str, as_of, target, *, members: int = 32, seed: int | None = None) -> Distribution:
    """Distribución del L1 para ``variable`` en ``region`` (ISO3, o 'WLD' para gmst) en ``target``,
    pronosticada desde ``as_of`` con la calibración de corte <= as_of y la regla de conducta
    estimada hasta as_of (el camino de ``validate.forecast`` del Atlas) y, desde F1, con la población
    y el capital humano endógenos después del origen (F1) y la psi_t de los mercados de recursos (F2)
    (``coupling.forecast``)."""
    if variable not in _L1_VARS:
        raise QueryError(f"unknown variable: {variable}. See catalog().")
    t_asof, t_tgt = _year(as_of), _year(target)
    if not (L1_FIRST_ORIGIN <= t_asof < L1_LAST_YEAR):
        raise QueryError(f"as_of {t_asof} outside [{L1_FIRST_ORIGIN}, {L1_LAST_YEAR - 1}] in F0: the ported L1 "
                         "only has a world up to 2019; projecting beyond arrives with F0.5")
    h = t_tgt - t_asof
    if not (1 <= h <= VALIDATED_HORIZON_YEARS) or t_tgt > L1_LAST_YEAR:
        raise QueryError(f"target {t_tgt}: the F0 L1 forecasts 1-{VALIDATED_HORIZON_YEARS} years ahead, not past "
                         f"{L1_LAST_YEAR}")
    import warnings

    from world4.data import atlas
    atlas.require()
    warnings.simplefilter("ignore")
    from world4.data import wpp
    if not wpp.available():
        raise QueryError("the F1 population sector needs the WPP 2024 age data: python world4_cli.py --install-wpp")
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, validate
    W = calibrate.W
    out_key, unit, transform, src = _L1_VARS[variable]
    t0 = t_asof - 1950
    _, cut = validate.params_for(t_asof)
    from world4.data import energy as edata
    if not edata.available():
        raise QueryError("the F3 energy sector needs the energy data: python world4_cli.py --install-energy")
    from world4.data import climate as cdata
    with_f4 = cdata.available()
    if not with_f4 and out_key.startswith("climate_"):
        raise QueryError(f"{variable} needs the F4 climate data: python world4_cli.py --install-climate")
    from world4.data import food as fdata
    with_f5 = with_f4 and fdata.available()
    if not with_f5 and out_key.startswith("food_"):
        raise QueryError(f"{variable} needs the F5 food data: python world4_cli.py --install-food")
    f = coupling.forecast(t0, R=members, seed=t0 if seed is None else seed, pop_mode="endogenous",
                          f2=coupling.F2Config(), f3=coupling.F3Config(),
                          f4=coupling.F4Config() if with_f4 else None, f5=coupling.F5Config() if with_f5 else None)
    if out_key in WORLD_KEYS:
        if region not in ("WLD", "world"):
            raise QueryError(f"{variable} is a world variable: region='WLD'")
        ens = f[out_key][:, h]
    else:
        iso = list(W.iso3)
        if region not in iso:
            raise QueryError(f"region {region!r} is not in the L1 (180 ISO3)")
        ens = f[out_key][:, iso.index(region), h]
    if transform is not None:
        ens = transform(ens)
    ens = ens[np.isfinite(ens)]
    if ens.size == 0:
        raise QueryError(f"the L1 produced no finite values for {variable} in {region} {t_tgt}")
    p10, p50, p90 = (float(np.percentile(ens, q)) for q in (10, 50, 90))
    caveats = list(L1_CAVEATS)
    if with_f4 and not with_f5:
        caveats.append("F5 food data not installed: footprint and biocapacity frozen at the origin (Atlas D4) "
                       "(python world4_cli.py --install-food)")
    if not with_f4:
        caveats.append("F4 climate data not installed: GMST follows the Atlas TCRE (python world4_cli.py "
                       "--install-climate)")
    return Distribution(variable, unit, region, t_asof, t_tgt, p10, p50, p90, float(ens.mean()), int(ens.size),
                        "L1", calibrated=False, exploratory=False, caveats=caveats,
                        provenance={"model_revision": MODEL_REVISION, "atlas_commit": atlas.ATLAS_COMMIT,
                                    "params_cutoff": cut, "seed": t0 if seed is None else seed,
                                    "resolution_source": src, "population": "endogenous (F1)", "psi": "world markets (F2)", "energy": "endogenous mix with EROI (F3)",
                                    "climate": ("FaIR v2.2.4, calibration 1.6.0 (F4)" if with_f4 else "Atlas TCRE"),
                                    "food": ("endogenous food, land, N and water (F5)" if with_f5 else "none"),
                                    "wpp_edition": "2024-07-11"})


def run_l0r(start: int = 2019, years: int = 81, *, posterior: bool = False, explore: bool = False,
            members: int = 200, seed: int = 0, **mult) -> dict:
    """L0 regional de F6 (10 regiones, 112 stocks) desde ``start`` con datos <= ``start``; ``mult`` son las
    palancas declaradas (``world4.validate.w6.LEVERS``) como multiplicadores. Es un modo de comportamiento,
    no un pronóstico (exploratory).

    Con ``posterior=True`` (F7): ``members`` corridas con las perillas del posterior de ``models/posterior.npz``
    (corte 2019) y bandas p10/p50/p90 mundiales que incluyen la discrepancia declarada y el error del dato;
    con ``explore=True`` las perillas que la historia no identifica (recurso, daño, EROI, sustitución...)
    se sacan de su caja de previa en lugar de quedar en su valor de F6."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l0r import model, regions
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    p, ini = model.calibrate(W, min(start, 2019)), model.initial(W, start)
    if not posterior:
        m = model.L0R(p, ini, mult={k: [float(v)] for k, v in mult.items()})
        r = m.run(years)
        P = r["P"][:, 0]
        return {"years": r["years"].tolist(), "regions": list(regions.CODES),
                "population_millions": P.tolist(), "gdp_pc_2017usd": (r["Y"][:, 0] / P).tolist(),
                "gmst_degC": r["Tg"][:, 0, 0].tolist(), "fossil_depletion": r["x"][:, 0, 0].tolist(),
                "units": {"population_millions": "million persons", "gdp_pc_2017usd": "2017 USD PPP per person",
                          "gmst_degC": "degC above 1850-1900", "fossil_depletion": "fraction of the ultimate resource"},
                "band": "none: one deterministic run with the F6 OLS parameters; posterior=True gives p10-p90 bands",
                "exploratory": True, "fidelity": "L0"}
    import numpy as np
    from world4.calib import l0r as cl0
    from world4.calib import posterior as cpost
    post = cpost.load()["l0r"]
    rng = np.random.default_rng(seed)
    kn = cpost.draw_l0r(post, members, rng, explore=explore)
    for k, v in mult.items():
        kn[k] = np.full(members, float(v)) * (kn[k] if k in kn else 1.0)
    with np.errstate(all="ignore"):
        r = model.L0R(p, ini, M=members, mult=kn).run(years, record=cl0.RECORD + ("Y",))
    s = cl0.series(r, r["lnC"])
    h = np.arange(years + 1)
    band = {}
    for v, unit in (("pop", "million persons"), ("gdp", "2017 USD PPP"), ("energy", "TWh primary energy"),
                    ("cereal", "t cereals"), ("cropland", "1000 ha"), ("gmst", "degC above 1850-1900")):
        g = cl0._wlevel(s, v)
        sd = np.sqrt(cl0.disc_sd(v, h) ** 2 + cl0.OBS_SD[v] ** 2)
        e = (g[None] + sd[None, None] * rng.standard_normal((4,) + g.shape)).reshape(-1, len(h))
        q = np.percentile(e, [10, 50, 90], 0)
        if v != "gmst":
            q = np.exp(q)
        band[v] = {"unit": unit, "p10": q[0].tolist(), "p50": q[1].tolist(), "p90": q[2].tolist()}
    return {"years": r["years"].tolist(), "world": band, "members": members, "fidelity": "L0", "exploratory": True,
            "posterior_cutoff": post["cutoff"], "not_identified": sorted(post["explore"]),
            "not_identified_drawn_from_prior": bool(explore),
            "band": "p10-p90 of posterior x declared discrepancy x data error; beyond 2019 a behaviour mode, not a forecast"}


def evaluate_policy(theta, *, members: int = 200, narrative: str = "ssp2", seed: int = 0) -> dict:
    """F8: una política (vector θ de ``world4.scenarios.levers`` o dict {palanca: valor}) sobre el ensemble
    del L0 a 2100: objetivos robustos, límites, comparación con no hacer nada y vulnerabilidades (PRIM).
    Exploratorio."""
    import numpy as np
    from world4.scenarios import levers, robust
    if isinstance(theta, dict):
        th = np.zeros(len(levers.NAMES))
        for k, v in theta.items():
            th[levers.NAMES.index(k)] = float(v)
    else:
        th = np.asarray(theta, float)
    return robust.evaluate_policy(th, members=members, narrative=narrative, seed=seed)


def design(*, narrative: str = "ssp2", members: int = 100, report_only=(), seed: int = 0, **kw) -> dict:
    """F8: búsqueda robusta de políticas (``phycore.Designer``, NSGA-II sobre el L0): el frente verificado,
    tres representantes con sus vulnerabilidades y el arrepentimiento. Exploratorio."""
    from world4.scenarios import robust
    return robust.design(members=members, narrative=narrative, report_only=report_only, seed=seed,
                         log=lambda *a: None, **kw)


def loop_dominance(start: int = 2019, years: int = 81, top: int = 3, step: int = 10) -> list:
    """Lazos dominantes del L0 regional (vista mundial; puntuación de Loops that Matter)."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l0r import loops, model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    m = model.L0R(model.calibrate(W, min(start, 2019)), model.initial(W, start))
    out = m.run(years, keep_states=True)
    return loops.dominance_table(loops.analyse(*loops.l0_world_view(m, out)), top=top, step=step)
