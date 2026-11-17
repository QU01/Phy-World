"""V5 (World3 vigente, fase 5): el conjunto que la historia no descarta, el error del modelo y las bandas."""
import numpy as np
import pytest

from world4.core.l0v import ensemble as E
from world4.core.l0v import technology as T


# ------------------------------------------------------------------ sin datos
def test_sample_fills_the_declared_ranges():
    th = E.sample(per_level=50, levels=5, seed=3)
    assert th.shape == (250, len(E.NAMES)) and E.NAMES[:5] == tuple(E.KNOBS)
    box = {**E.KNOBS, **E.AXES}
    for j, name in enumerate(E.NAMES):
        assert box[name][0] <= th[:, j].min() and th[:, j].max() <= box[name][1], name
    assert len(np.unique(th[:, E.NAMES.index("kappa")])) == 5                       # una tabla por corrida vectorizada
    nri = th[:, E.NAMES.index("nri_factor")]
    assert np.median(nri) == pytest.approx(np.sqrt(0.7 * 3.0), rel=0.1)             # uniforme en el logaritmo
    small = E.sample(per_level=20, levels=2, box={**box, "r_yield": (0.01, 0.02), "kappa": (-0.1, 0.1)})
    assert small[:, 0].min() >= 0.01 and small[:, 0].max() <= 0.02
    assert np.array_equal(E.sample(10, 2, seed=1), E.sample(10, 2, seed=1))


def test_members_become_technologies_and_resources():
    th = E.sample(per_level=4, levels=1, seed=0)
    one, many = E.tech_of(th[0]), E.tech_of(th)
    assert isinstance(one, T.Tech) and one.erosion_control and one.delay == 1.0
    assert float(one.r_yield) == th[0, 0] and many.r_yield.shape == (4,) and isinstance(many.kappa, float)
    assert E.nri_of(th).shape == (4,) and float(E.nri_of(th[0])) == pytest.approx(1e12 * th[0, E.NAMES.index("nri_factor")])
    assert np.all(many.paths(2050.0)[0] <= many.yield_cap)


def test_acceptance_needs_every_indicator_within_the_tolerance():
    best = np.array([T.stored()["by_cutoff"]["2023"]["rms_pct"][f[0]] for f in T.FIT]) / 100.0
    errors = np.array([best, 1.9 * best, 2.1 * best, best * np.array([1, 1, 1, 1, 2.5])])
    assert E.accept(errors, 2023).tolist() == [True, True, False, False]
    assert E.accept(errors, 2023, tolerance=3.0).tolist() == [True, True, True, True]


def test_shrunk_box_wraps_the_survivors_inside_the_prior():
    prior = {**E.KNOBS, **E.AXES}
    th = E.sample(per_level=30, levels=3, seed=0)
    keep = th[th[:, 0] < 0.02]
    box = E._shrink(keep, prior)
    assert box["r_yield"][1] < prior["r_yield"][1] and box["r_yield"][0] == prior["r_yield"][0]
    assert all(prior[n][0] <= box[n][0] < box[n][1] <= prior[n][1] for n in E.NAMES)


def test_stored_ensemble_is_inside_the_ranges_and_fits_history():
    for cutoff in E.CUTOFFS:
        post = E.posterior(cutoff)
        assert post["names"] == E.NAMES and len(post["theta"]) >= 100 and post["sampled"] >= 20000
        assert E.accept(post["errors"], cutoff).all()
        box = {**E.KNOBS, **E.AXES}
        for j, name in enumerate(E.NAMES):
            assert box[name][0] <= post["theta"][:, j].min() and post["theta"][:, j].max() <= box[name][1], name
    nri = E.posterior(2023)["theta"][:, E.NAMES.index("nri_factor")]
    assert np.percentile(nri, 95) / np.percentile(nri, 5) > 2.5                     # la historia apenas fija los recursos
    kappa = E.posterior(2023)["theta"][:, E.NAMES.index("kappa")]
    assert kappa.max() < 0.3                                                        # y sí descarta el recargo de World3
    with pytest.raises(ValueError):
        E.posterior.__wrapped__(1985)


def _toy_projection():
    years = np.arange(2020, 2031)
    base = np.tile(np.linspace(100.0, 110.0, len(years)), (200, 1))
    return {"years": years, "origin": 2023, "cutoff": 2023, "theta": np.zeros((200, len(E.NAMES))),
            "series": {"population": base.copy(), "gmst": base / 100.0, "fertility": base / 80.0, "iopc": base.copy()}}


def test_model_error_grows_with_the_horizon_and_respects_the_rules():
    proj = _toy_projection()
    disc = {"per_year": {"population": 0.01, "gmst": 0.02, "fertility": 0.05}}
    out = E.with_model_error(proj, disc, seed=1)
    hist = proj["years"] <= 2023
    assert np.array_equal(out["population"][:, hist], proj["series"]["population"][:, hist])   # la historia no se toca
    assert np.array_equal(out["iopc"], proj["series"]["iopc"])                                 # sin dato, sin error
    spread = np.std(np.log(out["population"] / proj["series"]["population"]), axis=0)
    assert spread[-1] == pytest.approx(0.01 * 7, rel=0.15) and np.all(np.diff(spread[3:]) > 0)
    add = out["gmst"] - proj["series"]["gmst"]
    assert np.std(add[:, -1]) == pytest.approx(0.02 * 7, rel=0.15)                             # aditivo, en grados
    from world4.core.l0v import equations
    assert out["fertility"].min() >= equations.F_LO
    assert np.array_equal(E.with_model_error(proj, disc, seed=1)["population"], out["population"])
    b = E.bands(proj, disc, seed=1)
    assert np.all(b["population"]["p10"] <= b["population"]["p50"]) and np.all(b["population"]["p50"] <= b["population"]["p90"])
    assert np.allclose(E.bands(proj)["population"]["p10"], E.bands(proj)["population"]["p90"])  # sin error: todos iguales


# ------------------------------------------------------------------ con datos (--install-world)
@pytest.fixture(scope="module")
def ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    assert world.verify_raw()["ok"]
    return world.load()


def test_vectorised_history_errors_match_the_single_run_fit(ready):
    from dataclasses import replace
    post = E.posterior(2000)
    th = post["theta"][:3].copy()
    th[:, E.NAMES.index("kappa")] = th[0, E.NAMES.index("kappa")]
    err = E.history_errors(th, 2000)
    assert err.shape == (3, len(T.FIT))
    from world4.core.l0v import model
    for k in range(3):
        r, _ = model.run(technology=E.tech_of(th[k]), fertility="observed", mortality="observed", t1=2001.0,
                         constants={"nri": float(E.nri_of(th[k]))})
        y, mod = T._indices(r)
        oy, obs = T._observed()
        for j, f in enumerate(T.FIT):
            sel = (oy >= 1971) & (oy <= 2000) & np.isfinite(obs[f[0]])
            d = np.log(np.interp(oy[sel], y, mod[f[0]])) - np.log(obs[f[0]][sel])
            assert err[k, j] == pytest.approx(np.sqrt(np.mean(d ** 2)), rel=1e-6), (k, f[0])


def test_history_rules_out_too_few_resources_and_keeps_the_best_fit(ready):
    best = T.params(2023)
    th = np.array([[best.r_yield, best.r_resource, best.r_pollution, best.kappa, best.land_restraint, 1.0,
                    best.resource_floor, best.pollution_floor, best.yield_cap, 0.05]] * 3)
    th[1, E.NAMES.index("nri_factor")] = 0.4
    th[2, E.NAMES.index("nri_factor")] = 3.0
    ok = E.accept(E.history_errors(th, 2023), 2023)
    assert ok.tolist() == [True, False, True]            # con muchos más recursos la historia es la misma


def test_model_error_is_measured_on_the_previous_split(ready):
    d = E.discrepancy(2023)
    assert d["previous"] == 2000 and set(d["per_year"]) == set(E.SCORED)
    assert 0.0003 < d["per_year"]["population"] < 0.003 and 0.005 < d["per_year"]["gmst"] < 0.02
    assert d["per_year"]["resource_use"] > d["per_year"]["population"]
    assert E.discrepancy(2000)["previous"] == 1990


def test_bands_from_2000_cover_what_happened(ready):
    cov = E.coverage(2000, members=40, seed=0)
    ind = cov["indicators"]
    assert cov["members"] == 40 and set(ind) == set(E.SCORED)
    assert min(v["with_model_error"] for k, v in ind.items() if k != "gmst") >= 0.75
    assert 0.5 <= ind["gmst"]["with_model_error"] < 0.9                      # medido: la de temperatura se queda corta
    assert ind["population"]["parameters_only"] < 0.3                        # sin el error del modelo, sí
    assert ind["population"]["with_model_error"] > ind["population"]["parameters_only"]
    assert np.mean([v["with_model_error"] for v in ind.values()]) > 0.9      # medido: son anchas de más


def test_projection_summary_and_what_drives_it(ready):
    proj = E.project(2023, members=30, seed=0)
    assert proj["series"]["population"].shape == (30, 201) and proj["theta"].shape == (30, len(E.NAMES))
    i23 = int(np.where(proj["years"] == 2023)[0][0])
    assert np.ptp(proj["series"]["population"][:, i23]) < 1.0                 # la historia es la misma en todos
    s = E.summary(proj, E.discrepancy(2023))
    lo, mid, hi = s["industrial_peak_year"]
    assert 2030 <= lo < mid <= hi <= 2100 and hi - lo > 30                    # el máximo industrial no está fijado
    assert s["drivers_of_industrial_peak"]["nri_factor"] > 0.7                # y lo mueven los recursos iniciales
    assert 2045 < s["population_peak_year"][1] < 2075 and 8800 < s["population_peak"][1] < 10200
    p10, p50, p90 = s["at"][2100]["population"]
    assert 6000 < p10 < p50 < p90 < 9500 and s["at"][2100]["gmst"][2] > s["at"][2050]["gmst"][2]


def test_api_and_cli_bands(ready, capsys):
    import json
    from world4 import cli
    assert cli.main(["--world3-bands", "--members", "12", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["members"] == 12 and out["ensemble"]["tolerance"] == E.TOLERANCE
    assert out["bands"]["years"][0] == 2023 and len(out["bands"]["population"]["p50"]) == 78
    assert set(out["coverage"]["indicators"]) == set(E.SCORED) and "nri_factor" in out["ensemble"]["ranges"]
