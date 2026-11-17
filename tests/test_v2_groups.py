"""V2 (World3 vigente): demografía por grupos de países, ecuaciones estimadas y prueba con origen móvil."""
import numpy as np
import pytest

from world4.core import l0
from world4.core.l0v import equations as eq
from world4.core.l0v.cohorts import A, Cohorts
from world4.core.l0v.popsector import PopSector
from world4.validate import world3


# ------------------------------------------------------------------ sin datos
@pytest.mark.parametrize("scenario", [1, 3])
def test_world3_population_equations_outside_the_loop_are_the_same(scenario):
    sc_c, sc_t = l0.SCENARIOS[scenario]
    ps = PopSector(1, {**l0.CONSTANTS, **sc_c}, {**l0.TABLES, **sc_t}, 0.5)
    worst = [0.0]

    def hook(t, st, a):
        le, tf = ps.step(t, a["iopc"], a["sopc"], a["fpc"], a["ppolx"], a["pop"])
        worst[0] = max(worst[0], abs(le[0] / a["le"][0] - 1), abs(tf[0] / a["tf"][0] - 1))

    l0.run(scenario=scenario, hook=hook)
    assert worst[0] < 1e-12


def test_world3_tables_give_a_poor_group_an_absurd_life_expectancy():
    """El resultado negativo de V2: las tablas de World3, con un quinto del ingreso mundial de 2000, dan menos de 45 años."""
    ps = PopSector(2, l0.CONSTANTS, l0.TABLES, 0.5)
    r = l0.run(t1=2000.0)
    g = lambda k: float(r[k][-1, 0])  # noqa: E731
    rel = np.array([0.2, 1.0])
    for _ in range(200):                                              # hasta el equilibrio de los retardos
        le, _tf = ps.step(2000.0, g("iopc") * rel, g("sopc") * rel, g("fpc") * np.array([0.85, 1.0]), g("ppolx"), g("pop"))
    assert le[0] < 45 < le[1]


def _sector(par=None, G=2):
    par = par or {"floor": eq.F_LO, "e0": {"income": 0.08, "food": 0.6, "time": 0.018}, "tfr": {"e0": -0.05, "time": -0.02}}
    s = eq.EstimatedSector(G, par, l0.CONSTANTS, l0.TABLES)
    s.anchor(2000.0, np.array([300.0, 900.0]), np.array([380.0, 520.0]), 3.0, np.array([55.0, 76.0]), np.array([5.0, 1.7]))
    return s


def test_estimated_equations_start_from_the_observed_and_respond_the_right_way():
    s = _sector()
    e0, tfr = s.step(2000.0, np.array([300.0, 900.0]), np.array([380.0, 520.0]), 3.0)
    assert np.allclose(e0, [55.0, 76.0]) and np.allclose(tfr, [5.0, 1.7])
    richer, _ = s.step(2000.0, np.array([600.0, 1800.0]), np.array([380.0, 520.0]), 3.0)
    fed, tfr_fed = s.step(2000.0, np.array([300.0, 900.0]), np.array([480.0, 620.0]), 3.0)
    later, tfr_later = s.step(2020.0, np.array([300.0, 900.0]), np.array([380.0, 520.0]), 3.0)
    assert np.all(richer > e0) and np.all(fed > e0) and np.all(later > e0)
    assert np.all(tfr_fed < tfr) and np.all(tfr_later < tfr)          # más esperanza de vida, menos hijos
    assert np.all(later < eq.E_HI) and np.all(tfr_later > eq.F_LO)
    dirty, _ = s.step(2000.0, np.array([300.0, 900.0]), np.array([380.0, 520.0]), 40.0)
    assert np.all(dirty < e0)                                         # la contaminación sigue con la tabla de World3


def test_below_subsistence_world3_food_table_takes_over():
    s = _sector()
    sfpc = l0.CONSTANTS["sfpc"]
    scarce, tfr_scarce = s.step(2000.0, np.array([300.0, 900.0]), np.array([0.5 * sfpc, 520.0]), 3.0)
    only_estimated = eq.e0_of(eq.e0_z(55.0) + 0.6 * np.log(0.5 * sfpc / 380.0))
    assert scarce[0] == pytest.approx(only_estimated * np.interp(0.5, *l0._grid("lmf", l0.TABLES["lmf"])))
    assert scarce[0] < 0.7 * only_estimated and scarce[1] == pytest.approx(76.0)
    assert tfr_scarce[0] > 5.0                                        # con menos esperanza de vida sube la fecundidad


def test_zero_coefficients_freeze_the_rates():
    s = _sector({**eq.ZERO, "floor": eq.F_LO})
    e0, tfr = s.step(2050.0, np.array([3000.0, 90.0]), np.array([600.0, 300.0]), 3.0)
    assert np.allclose(e0, [55.0, 76.0]) and np.allclose(tfr, [5.0, 1.7])


def test_logit_transforms_are_inverses_and_bounded():
    e = np.array([25.0, 50.0, 80.0, 91.0])
    assert np.allclose(eq.e0_of(eq.e0_z(e)), e)
    f = np.array([1.0, 2.1, 6.5])
    assert np.allclose(eq.tfr_of(eq.tfr_z(f)), f)
    assert eq.tfr_of(-50.0) == pytest.approx(eq.F_LO) and eq.e0_of(50.0) == pytest.approx(eq.E_HI)


def test_cohorts_take_group_patterns_sex_ratios_and_migrants():
    c = Cohorts(np.full((2, A), 1e5), np.full((2, A), 1e5), dt=0.5, n=2)
    pattern = np.zeros((2, A))
    pattern[0, 20:30] = 0.1
    pattern[1, 30:40] = 0.1
    births, _ = c.step(np.full((2, 2, A), 0.01), np.array([2.0, 4.0]), pattern, np.array([1.05, 1.10]))
    assert births[1] == pytest.approx(2 * births[0], rel=1e-9)
    assert c.N[1, 0, 0] / c.N[1, 1, 0] == pytest.approx(1.10)
    profile = np.zeros(A)
    profile[25] = 1.0
    before = c.by_age().copy()
    moved = c.migrate(np.array([1e4, -1e9]), profile)
    after = c.by_age()
    assert moved[0] == pytest.approx(1e4) and after[0, :, 25].sum() == pytest.approx(before[0, :, 25].sum() + 1e4)
    assert after[1, :, 25].sum() == 0 and moved[1] == pytest.approx(-before[1, :, 25].sum())      # no baja de cero
    assert np.array_equal(after[:, :, :25], before[:, :, :25])


# ------------------------------------------------------------------ con datos (--install-world)
@pytest.fixture(scope="module")
def ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    assert world.verify_raw()["ok"]
    return world.load()


def test_country_panel_and_partitions(ready):
    from world4.core.l0v import groups
    from world4.data import world
    c = world.load_countries()
    assert c["tfr"].shape == c["gdp"].shape == (236, 74)
    assert np.isfinite(c["gdp"][:, 50]).sum() == 164 and (c["iso3"] != "").all()
    for kind, n in (("world", 1), ("income", 4), ("regions", 10)):
        names, member = groups.partition(kind)
        assert len(names) == n and member.min() == 0 and member.max() == n - 1 and len(member) == 236
    with pytest.raises(ValueError):
        groups.partition("continents")


def test_the_one_group_world_is_the_world_of_wpp(ready):
    from world4.core.l0v import groups
    gi = groups.inputs("world")
    i = np.searchsorted(ready["years"], gi.years)
    assert np.allclose(gi.tfr[:, 0], ready["tfr"][i], rtol=0.01)
    assert np.abs(gi.e0[:, 0] - ready["le"][i]).max() < 1.0 and np.abs(gi.e0[:, 0] - ready["le"][i]).mean() < 0.3
    assert np.allclose(gi.births[:, 0] / 1e6, ready["births"][i], rtol=1e-4)
    assert np.abs(gi.migration).max() < 1e4                           # el mundo no tiene migración neta
    assert np.allclose(gi.rho, 1.0) and np.allclose(gi.phi, 1.0)


def test_groups_add_up_and_differ(ready):
    from world4.core.l0v import groups
    gi, one = groups.inputs("regions"), groups.inputs("world")
    assert np.allclose(gi.popM.sum(1), one.popM[:, 0]) and np.allclose(gi.births.sum(1), one.births[:, 0])
    assert np.abs(gi.migration.sum(1)).max() < 1e4
    ssa, eur = gi.names.index("SSA"), gi.names.index("EUR")
    assert gi.tfr[-1, ssa] > 2 * gi.tfr[-1, eur] and gi.e0[-1, eur] - gi.e0[-1, ssa] > 15
    assert gi.rho[50, eur] > 5 * gi.rho[50, ssa]
    assert np.allclose(gi.pattern.sum(2), 1.0)


def test_one_group_with_world3_equations_is_the_cohort_model_of_v1(ready):
    from world4.core.l0v import model
    a = model.run(t1=2023.0)[0]
    b = model.run(t1=2023.0, groups="world", equations="world3")[0]
    assert np.abs(b["pop"][:, 0] / a["pop"][:, 0] - 1).max() < 0.005
    assert np.abs(b["le"][-1, 0] - a["le"][-1, 0]) < 0.5


def test_group_engine_alone_reproduces_each_region(ready):
    c = world3.group_check("regions")
    assert c["world_mean_abs_pct"] < 1.5 and c["world_max_abs_pct"] < 2.5, c
    assert max(abs(v) for v in c["group_last_pct"].values()) < 6.0, c


def test_world3_tables_by_region_fail_and_the_estimated_ones_do_not(ready):
    from world4.core.l0v import model
    bad = world3.score(model.run(groups="regions", equations="world3", t1=2023.0)[0])
    good = world3.score(model.run(groups="regions", origin=2000, cutoff=2000, t1=2023.0)[0])
    assert bad["life_expectancy"]["mae_pct"] > 10 and bad["fertility"]["mae_pct"] > 25
    assert good["life_expectancy"]["mae_pct"] < 3.5 and good["population"]["mae_pct"] < 2.0


def test_estimated_coefficients_have_the_expected_signs_and_are_stable(ready):
    old, new = eq.estimate(1990), eq.estimate(None)
    assert old["cutoff"] == 1990 and new["cutoff"] == 2023
    for p in (old, new):
        assert p["e0"]["income"] > 0 and p["e0"]["food"] > 0 and p["e0"]["time"] > 0
        assert p["tfr"]["e0"] < 0 and p["tfr"]["time"] < 0
    assert new["e0"]["time"] == pytest.approx(old["e0"]["time"], rel=0.25)
    assert eq.params(2000) is eq.params(2000)


def test_rolling_origin_hindcast_beats_freezing_the_rates(ready):
    h = world3.hindcast(origins=(1990, 2010))
    for origin, res in h["origins"].items():
        est, frozen = res["estimated"], res["frozen"]
        for k in ("birth_rate", "death_rate", "life_expectancy"):
            assert est[k]["mae_pct"] < frozen[k]["mae_pct"], (origin, k)
        assert est["population"]["mae_pct"] < 2.5, origin
    assert h["origins"][2010]["estimated"]["population"]["mae_pct"] < 1.0
    assert h["origins"][1990]["estimated"]["population"]["mae_pct"] < h["origins"][1990]["frozen"]["population"]["mae_pct"]


def test_the_observed_economy_isolates_demography(ready):
    from world4.core.l0v import model
    kw = dict(groups="regions", origin=2000, cutoff=2000, t1=2023.0)
    own = world3.score(model.run(**kw)[0], window=(2001, 2023))
    real = world3.score(model.run(economy="observed", **kw)[0], window=(2001, 2023))
    assert real["life_expectancy"]["mae_pct"] < 1.0 < own["life_expectancy"]["mae_pct"]
    assert real["death_rate"]["mae_pct"] < own["death_rate"]["mae_pct"]
    with pytest.raises(ValueError):
        model.run(groups="regions", economy="guess", t1=1960.0)
    with pytest.raises(ValueError):
        model.run(groups="regions", n=2, t1=1960.0)


def test_selection_table_shows_why_income_is_not_in_the_fertility_equation(ready):
    s = eq.selection("regions", cutoffs=(2000,), floors=(eq.F_LO,))
    t = s["tfr"][(2000, eq.F_LO)]
    assert t["e0+time"] < t["trend"] < t["frozen"]
    assert t["income+e0+time"] > t["e0+time"] and t["income"] > t["trend"]
    e = s["e0"][2000]
    assert e["income+food+time"] < e["income+time"] < e["frozen"] and e["income+food+time"] < 1.0


def test_cli_reports_the_hindcast(ready, capsys):
    import json
    from world4 import cli
    assert cli.main(["--world3-score", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert "groups" in out["runs"] and set(out["hindcast"]["origins"]) == {"1980", "1990", "2000", "2010"}
    assert out["hindcast"]["origins"]["2000"]["estimated"]["population"]["mae_pct"] < 2.0
