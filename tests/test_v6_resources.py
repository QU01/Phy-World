"""V6 (World3 vigente): los recursos no renovables acotados con datos, como curva de coste de extracción."""
import numpy as np
import pytest

from world4.core.l0v import resources as R


@pytest.fixture(scope="module")
def ready():
    from world4.data import world, wpp
    if not (world.available() and wpp.available()):
        pytest.skip("series mundiales o WPP por edad sin instalar: python world4_cli.py --install-world")
    assert world.verify_raw()["ok"]
    return world.load()


def test_supply_curves_by_fuel_and_grade(ready):
    g = R.grades()
    assert set(g) == set(R.FUELS)
    for fuel, d in g.items():
        assert np.all(np.diff(d["cost"]) > 0) and np.all(d["gt"] > 0), fuel            # grados de coste creciente
        assert d["gt"] == pytest.approx(d["ej"] * d["kgc_per_gj"] * R.C_TO_CO2 / 1000.0)
    assert g["coal"]["kgc_per_gj"] > g["crude oil"]["kgc_per_gj"] > g["natural gas"]["kgc_per_gj"]
    assert g["coal"]["gt"].sum() == pytest.approx(26742, rel=1e-3)
    assert g["crude oil"]["gt"].sum() == pytest.approx(8066, rel=1e-3)
    assert g["natural gas"]["gt"].sum() == pytest.approx(12780, rel=1e-3)
    assert R.fuel_cost("coal", 0.0) == g["coal"]["cost"][0] and R.fuel_cost("coal", 1e9) == g["coal"]["cost"][-1]


def test_proved_reserves_plus_emitted_are_the_stock_of_world3(ready):
    c = R.reserves_check()
    assert c["year"] == 2015
    assert sum(c["proved_reserves_gt"].values()) == pytest.approx(2857, rel=0.01)
    assert c["reserves_plus_emitted_gt"] == pytest.approx(c["world3_stock_gt"], rel=0.03)   # medido: 4.310 y 4.296
    assert c["curve_total_gt"] > 10 * c["world3_stock_gt"]                               # y la curva es once veces más
    for fuel in R.FUELS:                                # lo probado está en el grado actual de la curva o en el siguiente
        assert c["cost_at_reserves"][fuel] <= 2.0 * c["cost_now"][fuel], fuel
        assert c["proved_reserves_gt"][fuel] + c["emitted_gt"][fuel] < c["curve_by_fuel_gt"][fuel]


def test_cost_rises_along_the_curve_and_substitution_delays_it(ready):
    h = R.history()
    q70 = float(h["total"][h["years"] == 1970][0])
    assert R.capital_share(q70) == pytest.approx(R.BASE_SHARE)                    # 1970 es el 5 % de World3
    assert R.capital_share(0.5 * q70) == pytest.approx(R.BASE_SHARE)              # y antes no era más barato
    q23 = float(h["total"][h["years"] == 2023][0])
    assert 0.07 < R.capital_share(q23) < 0.09                                     # medido: 8,1 % en 2023
    q = np.linspace(0.0, R.total_gt(), 3000)
    for s in (0.0, 0.5, 1.0):
        assert np.all(np.diff(R.cost(q, s)) >= -1e-12), s
    mid = q[(q > R._last()) & (q < 20000)]
    assert np.all(R.cost(mid, 1.0) <= R.cost(mid, 0.0) + 1e-12)                   # lo más barato primero cuesta menos
    assert np.any(R.cost(mid, 1.0) < R.cost(mid, 0.0) - 0.1)
    assert R.cost(q23, 0.0) == R.cost(q23, 1.0)                                   # en la historia no hay elección
    assert R.capital_share(R.total_gt(), 1.0) == R.CAP
    assert sum(h["share"].values()) == pytest.approx(1.0) and sum(h["energy_weight"].values()) == pytest.approx(1.0)
    assert h["energy_weight"]["coal"] < h["share"]["coal"]                        # el carbón pesa menos en energía


def test_scale_stretches_only_what_is_left(ready):
    last, full = R._last(), R.total_gt()
    assert R.total_gt(2.0) == pytest.approx(last + 2.0 * (full - last))
    q = np.linspace(0.0, full, 2000)
    assert np.all(R.cost(q, 0.5, 2.0) <= R.cost(q, 0.5, 1.0) + 1e-12)
    assert np.all(R.cost(q, 0.5, 0.5) >= R.cost(q, 0.5, 1.0) - 1e-12)
    hist = q <= last
    assert np.array_equal(R.cost(q[hist], 0.5, 0.5), R.cost(q[hist], 0.5, 2.0))


def test_a_cutoff_hides_the_later_fuel_mix(ready):
    h00, h = R.history(2000), R.history()
    assert h00["years"][-1] == 2000 and h["years"][-1] >= 2023
    assert h00["share"]["crude oil"] > h["share"]["crude oil"]                   # en los noventa mandaba el petróleo
    q70 = float(h["total"][h["years"] == 1970][0])
    assert R.capital_share(q70, cutoff=2000) == pytest.approx(R.BASE_SHARE)
    assert R.Resources(cutoff=2000).tables() != R.Resources().tables()
    with pytest.raises(ValueError):
        R.history(1975)


def test_the_model_stock_is_aligned_with_the_observed_emissions(ready):
    from world4.core import l0
    from world4.core.l0v import model
    res = R.Resources()
    a = R._anchor()
    assert a["gt_per_unit"] * l0.CONSTANTS["nri"] == pytest.approx(4296, rel=0.01)
    t = res.tables()["fcaor1"]
    assert t[:3] == (0.0, 1.0, 1.0 / R.TABLE_POINTS) and len(t[3]) == R.TABLE_POINTS + 1
    assert t[3][-1] == R.BASE_SHARE and np.all(np.diff(t[3]) <= 1e-12) and t[3][0] > 0.6
    assert res.constants()["nri"] > 10 * l0.CONSTANTS["nri"]
    r, _ = model.run(technology="estimated", resources="estimated", fertility="observed", mortality="observed",
                     t1=2024.0)
    time = np.asarray(r.time, float)
    at = lambda k, y: float(np.asarray(r[k], float)[np.isclose(time, y), 0][0])  # noqa: E731
    h = R.history()
    for year, tol in ((1970, 1e-6), (2000, 0.08), (2023, 0.05)):                # el acumulado del modelo es el real
        assert float(res.emitted(at("nrfr", year))) == pytest.approx(float(h["total"][h["years"] == year][0]), rel=tol)
    assert at("fcaor", 1970) == pytest.approx(0.05) and 0.07 < at("fcaor", 2023) < 0.09


def test_without_resources_nothing_changes(ready):
    from world4.core.l0v import model, technology
    kw = dict(groups="regions", origin=2010, cutoff=2010, technology="estimated", t1=2030.0)
    a, b = model.run(**kw)[0], model.run(resources=None, **kw)[0]
    assert np.array_equal(a["pop"], b["pop"]) and np.array_equal(a["io"], b["io"])
    assert technology.params(2010) != technology.params(2010, resources=True)
    c = model.run(resources="estimated", **kw)[0]
    assert not np.array_equal(a["io"], c["io"])
    assert np.asarray(c["fcaor"])[-1, 0] < np.asarray(a["fcaor"])[-1, 0] + 0.1


def test_history_fits_as_well_with_the_curve_and_cannot_tell_them_apart(ready):
    from world4.core.l0v import technology as T
    st = T.stored()
    for cutoff in ("1990", "2000", "2010", "2023"):
        stock, curve = st["by_cutoff"][cutoff], st["by_cutoff_curve"][cutoff]
        assert curve["loss"] <= 1.02 * stock["loss"], cutoff                     # medido: algo mejor en los cuatro
        assert curve["kappa"] < stock["kappa"]                                   # el coste de capital lo compensa
        for k in ("r_yield", "r_resource", "r_pollution"):
            assert curve[k] == pytest.approx(stock[k], rel=0.05), (cutoff, k)    # y lo demás no se mueve
    assert T.loss(T.params(2023, resources=True), 2023, resources="estimated") == pytest.approx(
        st["by_cutoff_curve"]["2023"]["loss"], rel=1e-6)


def test_the_limit_moves_from_resources_to_pollution(ready):
    from world4.core.l0v import model
    kw = dict(groups="regions", origin=2023, technology="estimated", climate="estimated")
    out = {}
    for name, extra in (("stock", {}), ("curve", {"resources": "estimated"})):
        r, _ = model.run(**kw, **extra)
        t = np.asarray(r.time, float)
        ok = np.isclose(t, np.round(t))
        years = np.round(t[ok]).astype(int)
        out[name] = {k: np.asarray(r[k], float)[ok, 0] for k in ("iopc", "fcaor", "ppolx", "fpc", "le", "pop")}
        out[name]["years"] = years
    y = out["stock"]["years"]
    i23, after = int(np.where(y == 2023)[0][0]), y >= 2023
    stock, curve = out["stock"], out["curve"]
    assert y[after][np.argmax(stock["iopc"][after])] < 2060 and stock["fcaor"][-1] > 0.5      # World3: se agotan
    assert y[after][np.argmax(curve["iopc"][after])] == 2100 and curve["fcaor"][-1] < 0.25    # con la curva, no
    assert curve["iopc"][-1] > 4 * curve["iopc"][i23]
    assert curve["ppolx"][-1] > 5 * stock["ppolx"][-1] and curve["ppolx"][-1] > 20            # y manda la contaminación
    assert curve["fpc"][-1] < 0.6 * curve["fpc"][i23]
    assert curve["le"][-1] < curve["le"][after].max() - 5
    assert abs(y[after][np.argmax(curve["pop"][after])] - y[after][np.argmax(stock["pop"][after])]) <= 3


def test_scoreboard_has_the_bounded_variant(ready):
    from world4.validate import world3
    sb = world3.scoreboard(variants=("full", "bounded"))
    full, bounded = sb["runs"]["full"], sb["runs"]["bounded"]
    for k in world3.HEADLINE:
        assert abs(bounded[k]["mae_pct"] - full[k]["mae_pct"]) < 3.0, k         # medido: la historia sale casi igual
    assert bounded["population"]["mae_pct"] < 2.0


def test_hindcast_with_the_curve(ready):
    from world4.validate import world3
    out = world3.technology_hindcast(origins=(2000,), resources="estimated")
    base = world3.technology_hindcast(origins=(2000,))
    a, b = out["origins"][2000], base["origins"][2000]
    assert a["world3"]["output"]["mae_pct"] == b["world3"]["output"]["mae_pct"]        # la de World3 no lleva curva
    assert a["technology"]["output"]["mae_pct"] != b["technology"]["output"]["mae_pct"]
    assert a["technology"]["output"]["mae_pct"] < a["world3"]["output"]["mae_pct"]
