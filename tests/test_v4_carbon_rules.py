"""V4 (World3 vigente, fase 4): carbono y clima, daño al rendimiento y reglas de la demografía ante la escasez."""
import numpy as np
import pytest

from world4.core import l0
from world4.core.l0v import carbon as C
from world4.core.l0v import equations as eq
from world4.core.l0v import technology as T
from world4.validate import world3

INC, FOOD, SERV = np.array([300.0, 900.0]), np.array([380.0, 520.0]), np.array([200.0, 600.0])
PAR = {"floor": eq.F_LO, "e0": {"income": 0.08, "food": 0.6, "time": 0.018}, "tfr": {"e0": -0.05, "time": -0.02}}


def _sector(rules=eq.Rules()):
    s = eq.EstimatedSector(2, PAR, l0.CONSTANTS, l0.TABLES, 0.5, rules)
    s.anchor(2000.0, INC, FOOD, SERV, 3.0, np.array([55.0, 76.0]), np.array([5.0, 1.7]))
    return s


def _run(sector, years, services):
    out = None
    for k in range(1, 2 * years + 1):
        out = sector.step(2000.0 + 0.5 * k, INC, FOOD, services, 3.0)
    return out


# ------------------------------------------------------------------ sin datos: las reglas
def test_rules_do_nothing_while_services_and_life_expectancy_rise():
    a, b = _run(_sector(), 30, 1.5 * SERV), _run(_sector(eq.NO_RULES), 30, 1.5 * SERV)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_progress_slows_with_services_below_the_origin():
    full, half = _sector(), _sector()
    _run(full, 20, SERV)
    _run(half, 20, 0.5 * SERV)
    assert np.allclose(full.clock, 20.0) and np.allclose(half.clock, 10.0)
    off = _sector(eq.Rules(progress_needs_services=False))
    _run(off, 20, 0.5 * SERV)
    assert np.allclose(off.clock, 20.0)


def test_health_table_bites_only_below_the_origin_and_with_its_delay():
    only_health = eq.Rules(health_table=True, progress_needs_services=False, fertility_ratchet=False)
    base = _run(_sector(eq.NO_RULES), 40, 0.25 * SERV)[0]
    soon = _run(_sector(only_health), 2, 0.25 * SERV)[0] / _run(_sector(eq.NO_RULES), 2, 0.25 * SERV)[0]
    late = _run(_sector(only_health), 40, 0.25 * SERV)[0] / base
    assert np.all(late < soon) and np.all(soon > 0.97) and np.all(late < 0.9)     # el retardo de 20 años de World3
    assert np.allclose(_run(_sector(only_health), 40, 3 * SERV)[0], _run(_sector(eq.NO_RULES), 40, 3 * SERV)[0])


def test_fertility_ratchet_keeps_the_maximum_life_expectancy():
    s, free = _sector(), _sector(eq.NO_RULES)
    for sec in (s, free):
        for k in range(1, 21):
            sec.step(2000.0 + 0.5 * k, INC, FOOD, SERV, 3.0)
    before = s.step(2010.5, INC, FOOD, SERV, 3.0)
    crisis = s.step(2011.0, INC, 0.4 * FOOD, SERV, 3.0)
    free.step(2010.5, INC, FOOD, SERV, 3.0)
    crisis_free = free.step(2011.0, INC, 0.4 * FOOD, SERV, 3.0)
    assert np.all(crisis[0] < before[0]) and np.all(crisis[1] <= before[1] + 1e-12)
    assert np.all(crisis_free[1] > before[1])


# ------------------------------------------------------------------ sin datos: carbono
def _synthetic_obs():
    years = np.arange(1900, 2026)
    fi = 2000.0 * np.exp(0.025 * (years - 1900))                      # Mt/año
    luc = np.full(len(years), 4000.0)
    cum_fi, cum_luc = 40000.0 + np.cumsum(fi), 160000.0 + np.cumsum(luc)
    cum = (cum_fi + cum_luc) / 1e3
    ppm = np.where(years >= 1959, 270.0 + 0.44 * cum / C.PPM_GT, np.nan)
    return {"years": years, "co2_fi": fi, "luc_co2": luc, "cum_co2_fi": cum_fi, "cum_luc_co2": cum_luc, "co2_ppm": ppm,
            "gmst": -0.4 + 0.5 * cum / 1000.0}


def test_carbon_estimate_recovers_a_known_world():
    p = C.estimate(2000, _synthetic_obs())
    assert p.cutoff == 2000 and p.airborne == pytest.approx(0.44) and p.c0 == pytest.approx(270.0)
    assert p.tcre == pytest.approx(0.5) and p.t0 == pytest.approx(-0.4)
    assert p.damage(0.0) == 1.0 and p.damage(2.0) == pytest.approx(0.9) and p.damage(-1.0) == 1.0
    assert p.damage(100.0) == p.damage_floor


def test_diagnose_follows_resource_use_and_respects_the_origin():
    obs = _synthetic_obs()
    r = l0.run()
    par = C.estimate(None, obs)
    free = C.diagnose(r, par, None, obs)
    held = C.diagnose(r, par, 2000, obs)
    i = lambda y: int(np.where(free["years"] == y)[0][0])  # noqa: E731
    o = C.observed(obs)
    assert held["cumulative"][i(2000)] == pytest.approx(o["cum"][o["years"] == 2000][0])
    assert np.allclose(held["emissions"][: i(2000) + 1], (o["fi"] + o["luc"])[: i(2000) + 1])
    assert held["emissions"][i(2030)] == pytest.approx(free["emissions"][i(2030)])           # después, las del modelo
    assert free["emissions"][i(1970)] == pytest.approx(o["fi"][70] + o["luc"][70])           # ancladas en 1970
    assert np.all(np.diff(free["cumulative"]) > 0) and np.all(np.diff(free["gmst"]) > 0)
    twice = C.diagnose(l0.run(constants={"nri": 2e12}), par, None, obs)
    assert twice["co2_ppm"][i(2100)] > free["co2_ppm"][i(2100)]       # más recursos, más carbono


def test_climate_damage_lowers_yields_only_after_the_origin():
    obs = _synthetic_obs()
    par = C.estimate(None, obs)
    tech = T.Tech(r_yield=0.015, r_resource=0.04, kappa=0.0)

    def go(climate, damage=0.05):
        hook = T.TechHook(tech, 0.5, climate=None if climate is None else C.Carbon(**{**par.__dict__, "yield_damage": damage}),
                          origin=2000, obs=obs)
        r = l0.run(hook=hook, tables=tech.tables(l0.TABLES), constants=tech.constants(l0.CONSTANTS))
        hook.patch(r)
        return r

    plain, hot, hotter = go(None), go(par), go(par, 0.2)
    k = lambda y: int(round((y - 1900) / 0.5))  # noqa: E731
    assert np.array_equal(hot["ly"][: k(2000) + 1], plain["ly"][: k(2000) + 1])
    assert hotter["ly"][k(2040), 0] < hot["ly"][k(2040), 0] < plain["ly"][k(2040), 0]
    w = hot.series["warming"][:, 0]
    assert np.all(w[: k(2000) + 1] == 0) and np.all(np.diff(w[k(2000):]) >= 0) and 0.3 < w[-1] < 4.0
    assert "warming" not in plain.series


# ------------------------------------------------------------------ con datos (--install-world)
@pytest.fixture(scope="module")
def ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    assert world.verify_raw()["ok"]
    return world.load()


def test_carbon_relations_hold_out_of_sample_with_real_emissions(ready):
    o = C.observed()
    for cut in (1990, 2000, 2010):
        p = C.estimate(cut)
        assert 0.40 < p.airborne < 0.46 and 0.40 < p.tcre < 0.56, cut
        sel = (o["years"] > cut) & (o["years"] <= 2023)
        assert np.abs(p.ppm(o["cum"][sel]) - o["ppm"][sel]).mean() < 1.5, cut         # ppm
        assert np.abs(p.gmst(o["cum"][sel]) - o["gmst"][sel]).mean() < 0.15, cut      # °C
    assert C.params(2000) is C.params(2000) and C.params().cutoff == 2024


def test_scoreboard_compares_atmospheric_co2(ready):
    sb = world3.scoreboard(variants=("original", "groups", "full"))
    co2 = {k: v["co2"] for k, v in sb["runs"].items()}
    assert co2["original"]["mae_pct"] == pytest.approx(32.7, abs=0.2) and co2["original"]["bias_pct"] > 0
    assert co2["full"]["mae_pct"] < 5 < co2["groups"]["mae_pct"]
    assert "pollution" not in sb["runs"]["full"] and "co2" in world3.HEADLINE


def test_full_model_climate_out_of_sample(ready):
    h = world3.technology_hindcast(origins=(2000,))["origins"][2000]
    tech, w3 = h["technology"]["climate"], h["world3"]["climate"]
    assert tech["co2_ppm_mae"] < 5 and tech["gmst_mae"] < 0.2
    assert tech["co2_ppm_mae"] < w3["co2_ppm_mae"]
    assert tech["gmst_last"][1] < tech["gmst_last"][2]                 # medido: se queda corto en 2023


def test_rules_decide_the_scarcity_run_and_not_the_tests_with_data(ready):
    from world4.core.l0v import model
    at = lambda r, k, y: float(r[k][list(r.time).index(float(y)), 0])  # noqa: E731
    kw = dict(groups="regions", origin=2000, cutoff=2000, technology="estimated", t1=2023.0)
    with_rules = world3.score(model.run(**kw)[0], window=(2001, 2023))
    without = world3.score(model.run(rules=eq.NO_RULES, **kw)[0], window=(2001, 2023))
    for k in ("population", "life_expectancy", "birth_rate"):
        assert with_rules[k]["mae_pct"] == pytest.approx(without[k]["mae_pct"], abs=0.3), k
    scarce = model.run(groups="regions", origin=2023)[0]                          # sin tecnología: hambruna
    loose = model.run(groups="regions", origin=2023, rules=eq.NO_RULES)[0]
    assert at(loose, "tf", 2100) > 3.5 and at(scarce, "tf", 2100) < 2.6
    assert at(scarce, "pop", 2100) < 0.6 * at(loose, "pop", 2100)


def test_projection_with_everything_on(ready):
    from world4.core.l0v import model
    at = lambda r, k, y: float(r[k][list(r.time).index(float(y)), 0])  # noqa: E731
    r, g = model.run(groups="regions", origin=2023, technology="estimated", climate="estimated")
    assert r.origin == 2023 and r.cutoff is None
    d = C.diagnose(r, C.params(), 2023)
    i = int(np.where(d["years"] == 2100)[0][0])
    assert 480 < d["co2_ppm"][i] < 650 and 1.5 < d["gmst"][i] < 3.2
    assert r.series["warming"][-1, 0] == pytest.approx(d["gmst"][i] - d["gmst"][int(np.where(d["years"] == 2023)[0][0])], abs=0.15)
    peak = int(np.argmax(r["iopc"][:, 0]))
    assert 2035 < r.time[peak] < 2060 and at(r, "fpc", 2100) > l0.CONSTANTS["sfpc"]
    assert at(r, "pop", 2100) < r["pop"].max() and 1.0 < at(r, "tf", 2100) < 1.6
    with pytest.raises(ValueError):
        model.run(groups="regions", origin=2023, climate="estimated", t1=2030.0)
