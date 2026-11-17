"""V3 (World3 vigente, fase 2): tecnología estimada con datos y el modelo completo fuera de muestra."""
from dataclasses import replace

import numpy as np
import pytest

from world4.core import l0
from world4.core.l0v import technology as T
from world4.validate import world3


# ------------------------------------------------------------------ sin datos
def test_technology_curves_start_at_one_and_stay_between_their_bounds():
    t = T.Tech(r_yield=0.02, r_resource=0.05, r_pollution=0.03)
    assert t.paths(1960.0) == (1.0, 1.0, 1.0) and t.paths(t.t0) == (1.0, 1.0, 1.0)
    a, b, far = t.paths(2000.0), t.paths(2050.0), t.paths(4000.0)
    assert 1 < a[0] < b[0] < t.yield_cap and t.resource_floor < b[1] < a[1] < 1 and t.pollution_floor < b[2] < a[2] < 1
    assert far == pytest.approx((t.yield_cap, t.resource_floor, t.pollution_floor))
    frozen = replace(t, freeze_at=2023.0)
    assert frozen.paths(2060.0) == t.paths(2023.0) and frozen.paths(2010.0) == t.paths(2010.0)


def test_capital_cost_tables_scale_with_kappa():
    base = l0.TABLES
    same = T.Tech(kappa=1.0).tables(base)
    for name in ("icormrrct", "icormlyt", "icormpt"):
        assert np.allclose(same[name][3], base[name][3])
    free = T.Tech(kappa=0.0).tables(base)
    assert np.allclose(free["icormrrct"][3], 3.0) and np.allclose(free["icormlyt"][3], 1.0)
    cheaper = T.Tech(kappa=-0.5).tables(base)
    assert max(cheaper["icormlyt"][3]) == 1.0 and min(cheaper["icormlyt"][3]) == pytest.approx(0.75)


def test_no_technology_is_world3_and_each_knob_moves_what_it_should():
    base = l0.run(t1=2030.0)

    def go(tech):
        return l0.run(t1=2030.0, hook=T.TechHook(tech), tables=tech.tables(l0.TABLES),
                      constants=tech.constants(l0.CONSTANTS))

    off = go(T.Tech())
    assert all(np.allclose(off[k], base[k], rtol=1e-12) for k in l0.RECORD)
    y = go(T.Tech(r_yield=0.02, kappa=0.0))
    assert y["ly"][-1, 0] > 1.2 * base["ly"][-1, 0] and np.array_equal(y["ly"][:150], base["ly"][:150])
    r = go(T.Tech(r_resource=0.05, kappa=0.0))
    assert r["nrfr"][-1, 0] > base["nrfr"][-1, 0] + 0.1
    p = go(T.Tech(r_pollution=0.05, kappa=0.0))
    assert p["ppolx"][-1, 0] < 0.6 * base["ppolx"][-1, 0]
    held = go(T.Tech(land_restraint=0.1))
    grow = lambda run: run["al"][-1, 0] - run["al"][150, 0]  # noqa: E731
    assert 0 < grow(held) < 0.9 * grow(base) and np.array_equal(held["al"][:151], base["al"][:151])
    costly, cheap = go(T.Tech(r_yield=0.02, kappa=1.0)), go(T.Tech(r_yield=0.02, kappa=0.0))
    assert cheap["io"][210, 0] > costly["io"][210, 0]          # 2005; después quien crece más agota antes los recursos


def test_hooks_compose():
    a = lambda t, st, x: {"flow": {"al": 0.5}, "set": {"lyt": 1.2}}  # noqa: E731
    b = lambda t, st, x: {"flow": {"al": 0.5, "pal": 2.0}, "scale": {"ic": 1.1}, "set": {"p1": 3.0}}  # noqa: E731
    out = T.compose(a, None, b)(0.0, {}, {})
    assert out == {"flow": {"al": 0.25, "pal": 2.0}, "scale": {"ic": 1.1}, "set": {"lyt": 1.2, "p1": 3.0}}
    assert T.compose(None, a) is a


def test_stored_technology_is_complete_and_stable():
    st = T.stored()
    assert set(st["by_cutoff"]) == {"1990", "2000", "2010", "2023"} and st["fixed"]["erosion_control"] is True
    latest, old = T.params(), T.params(2000)
    assert latest == T.params(2023) and latest.delay == 1.0 and latest.erosion_control
    for t in (latest, old):
        assert 0.01 < t.r_yield < 0.03 and 0.04 < t.r_resource < 0.06 and 0.02 < t.r_pollution < 0.04
        assert -0.5 < t.kappa < 0.2 and 0.0 < t.land_restraint < 0.5      # la tecnología no encarece el capital
    with pytest.raises(ValueError):
        T.params(1985)


# ------------------------------------------------------------------ con datos (--install-world)
@pytest.fixture(scope="module")
def ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    assert world.verify_raw()["ok"]
    return world.load()


def test_new_series_cropland_and_vegetal_equivalent_food(ready):
    y = ready["years"]
    crop = ready["cropland"]
    assert crop[y == 1990][0] == pytest.approx(1517, abs=2) and crop[y == 2016][0] == pytest.approx(1593, abs=2)
    veq, kcal = ready["food_veq"], ready["kcal"]
    animal_share = (veq - kcal) / 6 / kcal
    assert 0.12 < animal_share[y == 1961][0] < 0.14 and 0.15 < animal_share[y == 2013][0] < 0.17
    assert veq[y == 2013][0] / veq[y == 1970][0] > kcal[y == 2013][0] / kcal[y == 1970][0]


def test_stored_parameters_are_a_minimum_of_the_fit(ready):
    stored = T.stored()["by_cutoff"]["2000"]
    tech = T.params(2000)
    best = T.loss(tech, 2000)
    assert best == pytest.approx(stored["loss"], rel=1e-3)
    for knob, step in (("r_yield", 0.004), ("r_resource", 0.008), ("r_pollution", 0.006), ("kappa", 0.2),
                       ("land_restraint", 0.08)):
        for sign in (-1, 1):
            assert T.loss(replace(tech, **{knob: getattr(tech, knob) + sign * step}), 2000) > best, (knob, sign)
    parts = T.loss(tech, 2000, detail=True)
    assert max(parts.values()) < 0.06 ** 2 and T.loss(T.Tech(), 2000) > 20 * best


def test_world3_erosion_with_real_yields_loses_the_land_that_is_still_there(ready):
    from world4.core.l0v import model
    tech = T.params()
    at = lambda r, k, y: float(r[k][list(r.time).index(float(y)), 0]) / 1e6  # noqa: E731
    w3 = model.run(groups="regions", origin=2023, t1=2050.0, technology=replace(tech, erosion_control=False, land_restraint=1.0))[0]
    hyde = float(ready["cropland"][ready["years"] == 2016][0])
    assert at(w3, "al", 2016) < 0.8 * hyde and at(w3, "al", 2050) < 200
    loose = model.run(groups="regions", origin=2023, t1=2016.0, technology=replace(tech, land_restraint=1.0))[0]
    assert at(loose, "al", 2016) > 1.2 * hyde
    est = model.run(groups="regions", origin=2023, t1=2016.0, technology=tech)[0]
    assert at(est, "al", 2016) == pytest.approx(hyde, rel=0.05)


def test_full_model_scoreboard_meets_the_phase_2_targets(ready):
    sb = world3.scoreboard(variants=("original", "groups", "full"))
    full, v2 = sb["runs"]["full"], sb["runs"]["groups"]
    assert full["output"]["mae_pct"] < 10 and full["food_pc"]["mae_pct"] < 3 and full["resource_use"]["mae_pct"] < 20
    assert full["land_yield"]["mae_pct"] < 6 and full["arable_land"]["mae_pct"] < 6 and full["population"]["mae_pct"] < 1.5
    for k in world3.ECONOMY_KEYS:
        assert full[k]["mae_pct"] < 0.5 * v2[k]["mae_pct"], k
    assert full["death_rate"]["mae_pct"] < v2["death_rate"]["mae_pct"]
    assert full["birth_rate"]["mae_pct"] > v2["birth_rate"]["mae_pct"]         # medido: la natalidad empeora


def test_technology_hindcast_from_each_stored_cutoff(ready):
    h = world3.technology_hindcast()
    assert set(h["origins"]) == {1990, 2000, 2010}
    for origin, res in h["origins"].items():
        tech, w3 = res["technology"], res["world3"]
        assert tech["output"]["mae_pct"] < 8 and tech["output"]["mae_pct"] < 0.3 * w3["output"]["mae_pct"], origin
        assert tech["resource_use"]["mae_pct"] < 15 and tech["population"]["mae_pct"] < 2.5, origin
    assert h["origins"][1990]["technology"]["land_yield"]["mae_pct"] > 15      # medido: desde 1990 el rendimiento no


def test_with_technology_the_limit_moves_from_food_to_resources(ready):
    from world4.core.l0v import model
    at = lambda r, k, y: float(r[k][list(r.time).index(float(y)), 0])  # noqa: E731
    v2 = model.run(groups="regions", origin=2023)[0]
    v3 = model.run(groups="regions", origin=2023, technology="estimated")[0]
    sfpc = l0.CONSTANTS["sfpc"]
    assert at(v2, "fpc", 2050) < sfpc < at(v3, "fpc", 2100)                    # ya no hay hambruna
    assert at(v2, "le", 2050) < 50 < 75 < at(v3, "le", 2050)
    peak = int(np.argmax(v3["iopc"][:, 0]))
    assert 2035 < v3.time[peak] < 2060 and at(v3, "iopc", 2100) < 0.25 * v3["iopc"][peak, 0]
    assert at(v3, "nrfr", 2100) < 0.2 and at(v3, "fcaor", 2100) > 0.5          # se agotan los recursos


def test_api_and_cli_show_the_full_model(ready, capsys):
    import json
    from world4 import api, cli
    r, g = api.run_l0v(groups="regions", origin=2010, cutoff=2010, technology="estimated", t1=2023.0)
    assert g.names[5] == "SSA" and world3.score(r, window=(2011, 2023))["output"]["mae_pct"] < 5
    assert cli.main(["--world3-score", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "full" in out["runs"] and set(out["technology_hindcast"]["origins"]) == {"1990", "2000", "2010"}
