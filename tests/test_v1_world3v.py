"""V1 (World3 vigente, fases 0 y 1): el L0 frente al mundo observado (marcador) y la población por cohortes (``world4.core.l0v``)."""
import numpy as np
import pytest

from world4.core import l0
from world4.core.l0v import lifetable
from world4.core.l0v.cohorts import A, GROUPS, Cohorts
from world4.validate import world3


# ------------------------------------------------------------------ sin datos: gancho, cohortes, tablas de vida
def test_hook_set_replaces_a_stock_and_default_run_is_unchanged():
    base = l0.run(t1=1910.0)
    same = l0.run(t1=1910.0, hook=lambda t, st, a: None)
    assert all(np.array_equal(base[k], same[k]) for k in l0.RECORD)
    fixed = l0.run(t1=1910.0, hook=lambda t, st, a: {"set": {"nr": 5e11}})
    assert fixed["nr"][0, 0] == l0.CONSTANTS["nri"]            # el estado inicial no lo toca
    assert np.all(fixed["nr"][1:, 0] == 5e11)


def _gompertz(e_shift=0.0):
    a = np.arange(A, dtype=float)
    return 2e-4 * np.exp(0.085 * a - e_shift) + 0.03 * np.exp(-a) + 5e-4


def test_cohorts_conserve_people():
    c = Cohorts(np.full(A, 1e6), np.full(A, 1e6), dt=0.5)
    before = c.total()
    pattern = np.zeros(A)
    pattern[15:50] = 1 / 35
    births, deaths = c.step(np.stack([_gompertz(), _gompertz(0.3)]), 2.5, pattern)
    assert births > 0 and deaths > 0
    assert np.allclose(c.total(), before + births - deaths, rtol=1e-12)
    assert np.allclose(sum(c.groups().values()), c.total(), rtol=1e-12)
    assert list(GROUPS) == ["p1", "p2", "p3", "p4"]


def test_a_cohort_ages_as_a_wave_and_survives_like_its_life_table():
    pop = np.zeros(A)
    pop[20] = 1e6
    mx = _gompertz()
    for dt in (0.5, 1.0):
        c = Cohorts(pop, np.zeros(A), dt=dt)
        for _ in range(round(10 / dt)):
            c.step(np.stack([mx, mx]), 0.0, np.zeros(A))
        by = c.by_age()[0, 0]
        assert by[30] > 0 and np.all(by[:30] == 0) and np.all(by[31:] == 0)      # nadie se adelanta ni se rezaga
        expect = 1e6 * np.exp(-(0.5 * mx[20] + mx[21:30].sum() + 0.5 * mx[30]))
        assert by[30] == pytest.approx(expect, rel=2e-3), dt


def test_infants_are_exposed_one_year_not_more():
    """Con ``dt = 0,5`` y sin la corrección de las edades que cada casilla recorre, la edad 0 se expone 1,25 años."""
    mx = np.zeros(A)
    mx[0] = 0.15
    c = Cohorts(np.zeros(A), np.full(A, 1e6), dt=0.5)
    pattern = np.zeros(A)
    pattern[20:40] = 1 / 20
    for _ in range(8):
        births, deaths = c.step(np.stack([mx, mx]), 2.0, pattern, srb=1.0)
    per_birth = deaths / births
    assert per_birth == pytest.approx(1 - np.exp(-0.15), rel=0.02)
    c.w = np.zeros_like(c.w)                                                     # mutante: sin corrección
    for _ in range(8):
        births, deaths = c.step(np.stack([mx, mx]), 2.0, pattern, srb=1.0)
    assert deaths / births > 1.15 * per_birth


def test_cohorts_reject_a_step_that_does_not_divide_a_year():
    with pytest.raises(ValueError):
        Cohorts(np.ones(A), np.ones(A), dt=0.3)


def test_life_table_family_returns_the_life_expectancy_asked():
    shifts = np.linspace(-1.5, 2.5, 120)
    groups = np.r_[0, 1, np.arange(5, 100, 5), 100]
    mxM = np.array([_gompertz(s)[groups] for s in shifts])
    mxF = np.array([_gompertz(s + 0.3)[groups] for s in shifts])
    fam = lifetable.fit(mxM, mxF)
    assert np.all(np.diff(fam.e0) > 0)
    for target in (45.0, 60.0, 72.5):
        m = fam.mx(target)
        assert lifetable.e0_both(m[0], m[1]) == pytest.approx(target, abs=0.05)
    two = fam.mx(np.array([50.0, 70.0]))
    assert two.shape == (2, 2, A) and np.all(two[0] > two[1] - 1e-12)


def test_scoreboard_is_zero_against_itself_and_sees_a_mutant():
    r = l0.run()
    m = world3.model_series(r)
    obs = {"years": m["years"], "pop": m["population"], "le": m["life_expectancy"], "tfr": m["fertility"],
           "cbr": m["birth_rate"], "cdr": m["death_rate"], "gdp": m["output"], "food_veq": m["food_pc"],
           "cereal_yield": m["land_yield"], "cropland": m["arable_land"], "fossil_co2": m["resource_use"],
           "co2_ppm": np.full(len(m["years"]), 400.0)}
    s = world3.score(r, obs)
    assert set(s) == {i.key for i in world3.INDICATORS}
    assert s.pop("co2")["n"] == 0                      # sin series de carbono no hay CO2 que comparar
    assert max(v["mae_pct"] for v in s.values()) < 1e-9
    s2 = world3.score(l0.run(constants={"nri": 2e12}), obs, window=(2030, 2060))
    assert s2["output"]["mae_pct"] > 5
    assert set(world3.NEEDS) <= set(l0.RECORD)


def test_unknown_variant_and_modes_are_refused():
    with pytest.raises(ValueError):
        world3.run_variant("nope")
    from world4.core.l0v import model
    with pytest.raises(ValueError):
        model.CohortCoupler(1, fertility="guess", inputs=object())


# ------------------------------------------------------------------ con datos (--install-world)
@pytest.fixture(scope="module")
def world_ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    v = world.verify_raw()
    assert v["ok"], v
    return world.load()


def test_world_series_cover_what_the_scoreboard_reads(world_ready):
    from world4.data import vintages, world
    for ind in world3.INDICATORS:
        y, v = world.series(ind.obs)
        assert len(y) >= 10 and np.all(np.isfinite(v)), ind.key
        assert world3.BASE_YEAR in y, ind.key
    assert set(world.SOURCE_OF) == set(world.SERIES) == set(world.UNITS)
    assert set(world.SOURCE_OF.values()) <= set(vintages.sources())
    assert world_ready["pop"][2023 - world.YEAR0] == pytest.approx(8091.7, abs=1.0)


def test_scenario_1_score_is_the_measured_one(world_ready):
    s = world3.score(l0.run())
    expect = {"population": 4.7, "life_expectancy": 8.4, "output": 26.7, "food_pc": 14.1, "arable_land": 17.5,
              "resource_use": 106.4, "co2": 32.7}
    for k, v in expect.items():
        assert s[k]["mae_pct"] == pytest.approx(v, abs=0.1), k
    assert s["population"]["bias_pct"] < 0 < s["resource_use"]["bias_pct"]


def test_world_family_gives_the_world_its_own_mortality(world_ready):
    from world4.sectors.population import single_age
    fam = lifetable.wpp_world_family()
    w = world_ready
    le = dict(zip(w["years"].tolist(), w["le"].tolist()))
    for k in (0, 6, 13):
        p = int(w["mx_periods"][k])
        official = np.mean([le[y] for y in range(p - 2, p + 3)])
        m = fam.mx(official)
        assert np.allclose(m[0], single_age(w["mxM"][k]), rtol=1e-6)
        assert np.allclose(m[1], single_age(w["mxF"][k]), rtol=1e-6)
    base = lifetable.wpp_family()
    child = fam.mx(72.0)[:, 1:5].mean() / base.mx(72.0)[:, 1:5].mean()
    old = fam.mx(72.0)[:, 75:95].mean() / base.mx(72.0)[:, 75:95].mean()
    assert child > 1.5 and old < 0.9          # el mundo no es un país medio: más muertes de niños, menos de mayores


def test_cohort_engine_alone_reproduces_world_population(world_ready):
    c = world3.cohort_check()
    assert c["years"] == [1950, 2023]
    assert c["total_max_abs_pct"] < 1.5 and c["total_mean_abs_pct"] < 1.0, c
    assert max(c["group_max_abs_pct"].values()) < 3.0, c


def test_coupled_run_starts_from_the_observed_population_and_patches_the_rates(world_ready):
    from world4.core.l0v import model
    inp = model.world_inputs()
    r, c = model.run(t1=1960.0, inputs=inp)
    t = list(r.time)
    assert r["pop"][t.index(1950.5), 0] == pytest.approx(inp.popM[0].sum() + inp.popF[0].sum(), rel=1e-9)
    assert c.time[0] == 1950.5 and c.years[0] == 1950
    i = t.index(1955.0)
    j = int(np.where(c.time == 1955.0)[0][0])
    assert r["cbr"][i, 0] == pytest.approx(1000 * c["births"][j, 0] / r["pop"][i, 0])
    assert np.array_equal(r["pop"][: t.index(1950.0) + 1], l0.run(t1=1960.0)["pop"][: t.index(1950.0) + 1])
    two = model.run(t1=1960.0, n=2, constants={"dcfsn": np.array([3.8, 4.5])}, inputs=inp)[0]
    assert np.allclose(two["pop"][:, 0], r["pop"][:, 0]) and two["pop"][-1, 1] > two["pop"][-1, 0]


def test_cohorts_unmask_world3_fertility(world_ready):
    """Lo medido en V1: con cohortes la población empeora (se destapa el error de fecundidad que las cuatro
    cajas compensaban) y casi todo el error desaparece si se impone la fecundidad real."""
    sb = world3.scoreboard()
    pop = {k: v["population"]["mae_pct"] for k, v in sb["runs"].items()}
    assert pop["cohorts"] > pop["original"]
    assert pop["cohorts_obs_fertility"] < 2.0 and pop["cohorts_obs_both"] < 1.5
    assert sb["runs"]["cohorts"]["population"]["bias_pct"] > 0 > sb["runs"]["original"]["population"]["bias_pct"]
    assert sb["runs"]["cohorts_obs_both"]["birth_rate"]["mae_pct"] < 1.0


def test_refitting_fertility_fails_out_of_sample(world_ready):
    from world4.core.l0v import fit, model
    inp = model.world_inputs()
    base = fit.errors(None, 2000, inp)
    refit = fit.errors((1.245, 0.292, 0.391), 2000, inp)        # el ajuste de fit_fertility(2000), fijado aquí
    assert refit["in_sample_pct"] < base["in_sample_pct"]
    assert refit["out_of_sample_pct"] > 2 * base["out_of_sample_pct"]
    assert fit.sfsn_table(fit.X0)[3] == pytest.approx(l0.TABLES["sfsn"][3], abs=2e-3)


def test_api_and_cli(world_ready, capsys):
    import json
    from world4 import api, cli
    sb = api.world3_score(variants=("original",))
    assert list(sb["runs"]) == ["original"] and sb["scenario"] == 1
    r, c = api.run_l0v(t1=1960.0)
    assert r["pop"].shape[1] == 1 and c.popM.shape[2] == A
    assert cli.main(["--world3-score", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cohort_engine_check"]["total_mean_abs_pct"] < 1.0
    assert cli.main(["--world3-score", "--scenario", "9"]) == 2
