"""F8: palancas, narrativas, búsqueda robusta (plugin de phycore), PRIM, arrepentimiento y conducta."""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------ sin datos
def test_levers_have_box_unit_and_sign_and_reject_bad_input():
    from world4.core.l0r import model
    from world4.scenarios import levers
    assert levers.NAMES == model.POLICY
    th = levers.parse("carbon_price=100,renewables=0.01")
    assert th[0] == 100 and th[1] == 0.01 and th[2:].sum() == 0
    with pytest.raises(ValueError):
        levers.parse("carbon=5")
    with pytest.raises(ValueError):
        levers.parse("carbon_price=500")                     # fuera de la caja
    assert all(r["unit"] and r["declared_sign"] in (-1, 1) and r["mark"] == "[M]" for r in levers.describe())


def test_prim_finds_a_planted_box():
    from world4.scenarios import robust
    rng = np.random.default_rng(0)
    X = rng.random((3000, 4))
    y = ((X[:, 1] > 0.7) & (X[:, 3] < 0.4)).astype(float)
    y = np.where(rng.random(3000) < 0.03, 1 - y, y)          # ruido
    r = robust.prim(X, y, ["a", "b", "c", "d"])
    ch = r["chosen"]
    names = [b["name"] for b in ch["restricted"][:2]]
    assert set(names) == {"b", "d"}
    b = {x["name"]: x for x in ch["restricted"]}
    assert abs(b["b"]["lo"] - 0.7) < 0.08 and abs(b["d"]["hi"] - 0.4) < 0.08
    assert ch["density"] > 0.8 and ch["coverage"] >= 0.5


def test_narratives_are_parameter_sets():
    from world4.scenarios import narratives
    m = {"cbr": np.ones(3), "d_int": np.zeros(3), "lam_A": np.ones(3)}
    a = narratives.apply(m, "ssp1")
    assert np.allclose(a["cbr"], 0.92) and np.allclose(a["d_int"], -0.003) and np.allclose(a["lam_A"], 1.5)
    assert np.allclose(narratives.apply(m, "ssp2")["cbr"], 1.0)
    with pytest.raises(ValueError):
        narratives.apply(m, "ssp9")


def test_advised_conduct_follows_the_profitable_deviations():
    from world4.scenarios import conduct
    irl = {"desvios_rentables": {"k-": 0.2, "k+": 0.1, "r+": 0.9, "r-": 0.1, "m-": 0.7, "m+": 0.2,
                                 "x+": 0.4, "x-": 0.3, "f-": 0.5, "f+": 0.4, "w+": 0.6, "w-": 0.4}}
    assert conduct.advised_offsets(irl) == {"r": 0.5, "m": -0.5, "w": 0.5}


# ------------------------------------------------------------------ con datos
@pytest.fixture(scope="module")
def ready(atlas_ready):
    from world4.calib import posterior
    from world4.data import climate, energy, food, wpp
    if not (food.available() and climate.available() and energy.available() and wpp.available()):
        pytest.skip("faltan datos de F3-F5")
    if not posterior.PATH.is_file():
        pytest.skip("sin models/posterior.npz: python world4_cli.py --calibrate")


def test_zero_policy_is_bit_identical(ready):
    from world4.core.l0r import model
    from world4.scenarios import robust
    p, ini = robust._base()
    a = model.L0R(p, ini).run(30)
    b = model.L0R(p, ini, policy={k: 0.0 for k in model.POLICY}).run(30)
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_policy_levers_are_monotone_and_the_mutant_is_caught(ready):
    from world4.validate import f8
    r = f8.run(log=lambda *a: None)
    assert r["n_monotone"] == r["n"] and r["mutant_caught"]


def test_plugin_satisfies_the_phycore_contract(ready):
    from phycore.plugin import validate_plugin
    from world4.scenarios import robust
    plug = robust.World4Plugin(members=20)
    assert validate_plugin(plug, None) == []
    rec = plug.evaluate(np.zeros(len(plug.bounds)))
    assert rec["fidelity_used"] == "L0" and len(plug.objectives(rec)) == 3 and len(plug.constraints(rec)) == 3


def test_evaluate_policy_and_regret(ready):
    from world4 import api
    from world4.scenarios import robust
    r = api.evaluate_policy({"carbon_price": 150, "efficiency": 0.01}, members=40)
    assert r["exploratory"] is True
    assert r["objectives"]["peak_gmst_p90"] < r["baseline_objectives"]["peak_gmst_p90"]
    assert set(r["vulnerability"]) >= {"peak_gmst_above_3C", "welfare_below_no_policy"}
    g = robust.regret([np.r_[150.0, 0, 0.01, 0, 0, 0, 0]], members=40)
    assert g["gmst_regret_p90"][1] <= g["gmst_regret_p90"][0]        # la política: menos arrepentimiento climático


def test_offsets_empty_leave_the_l1_unchanged(ready):
    from world4.core import coupling
    kw = dict(R=2, seed=0, horizon=2, f5=coupling.F5Config())
    a = coupling.forecast(1985 - 1950, **kw)
    b = coupling.forecast(1985 - 1950, offsets={}, **kw)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a if k != "sim")
    c = coupling.forecast(1985 - 1950, offsets={"w": 0.5}, raw=True, **kw)
    assert "cons_pc" in c["raw"] and not np.array_equal(c["debt"], a["debt"], equal_nan=True)
