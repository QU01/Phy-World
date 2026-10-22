"""Módulo social acoplado a World3 (``world4.core.social``): contabilidad, números comunes y la vuelta."""
import numpy as np
import pytest

from world4.core import l0, social

N = 8


@pytest.fixture(scope="module")
def params():
    try:
        return social.load_params()
    except FileNotFoundError:
        pytest.skip("social_params.json no generado (reports/acople/calibrar.py)")


def test_hook_none_and_noop_hook_leave_world3_unchanged():
    a = l0.run()
    b = l0.run(hook=lambda t, st, aux: None)
    c = l0.run(hook=lambda t, st, aux: {"flow": {"ic": 1.0}, "scale": {"sc": 1.0}})
    for k in ("pop", "io", "le", "ppolx"):
        assert np.array_equal(a[k], b[k]) and np.array_equal(a[k], c[k])


def test_hook_scale_and_flow_act():
    a = l0.run(t1=2050)
    b = l0.run(t1=2050, hook=lambda t, st, aux: {"scale": {"ic": 0.999}} if t >= 2020 else None)
    i = int(np.searchsorted(a.time, 2030))
    assert b["ic"][i, 0] < a["ic"][i, 0]
    assert np.array_equal(a["ic"][: int(np.searchsorted(a.time, 2020)) + 1], b["ic"][: int(np.searchsorted(a.time, 2020)) + 1])


def test_one_way_coupling_leaves_world3_identical(params):
    a = l0.run(n=N)
    b, soc = social.run_coupled(n=N, feedback=False, seed=3)
    for k in ("pop", "io", "le", "ppolx", "nr"):
        assert np.array_equal(a[k], b[k])
    assert soc.years[0] == 2020 and soc.years[-1] == 2100
    assert soc["conflict_onset"].shape == (81, N)


def test_common_random_numbers_and_determinism(params):
    _, s1 = social.run_coupled(n=N, feedback=False, seed=5, t1=2060)
    _, s2 = social.run_coupled(n=N, feedback=False, seed=5, t1=2060)
    _, s3 = social.run_coupled(n=N, feedback=True, seed=5, t1=2060)
    assert np.array_equal(s1["coup"], s2["coup"])
    # con la vuelta World3 cambia poco: los eventos de los primeros años son los mismos
    assert np.array_equal(s1["coup"][:2], s3["coup"][:2])


def test_feedback_destroys_capital_by_the_social_loss(params):
    a, sa = social.run_coupled(n=N, feedback=False, capacity=False, seed=7, t1=2060)
    b, sb = social.run_coupled(n=N, feedback=True, capacity=False, seed=7, t1=2060)
    i = int(np.searchsorted(a.time, 2021))
    loss2020 = sb["loss"][0]
    assert np.all(loss2020 >= 0) and np.all(loss2020 < 0.01)
    assert np.allclose(b["ic"][i] / a["ic"][i], 1 - loss2020, rtol=1e-3)
    assert np.all(b["io"][-1] <= a["io"][-1] * 1.05)


def test_world_gdp_accounting(params):
    """Con vuelta, el crecimiento mundial que reciben los países es el de World3 más la pérdida social del año
    anterior (que World3 ya absorbió en su capital): no se cuenta dos veces."""
    soc = social.Society(params, 4, seed=1)
    soc.ypc_prev = np.full(4, 1000.0)
    prev = None
    for k, y in enumerate(range(2020, 2030)):
        loss, cap, rec, _, _ = soc.step(y, np.full(4, 1000.0 * 1.01 ** (k + 1)))
        assert np.all((cap > 0.9) & (cap <= 1.0)) and np.all(loss < 0.02)
        expect = np.log(1.01) - (np.log(1 - prev) if prev is not None else 0.0)
        assert np.allclose(rec["growth_world"], expect, atol=1e-9)
        prev = loss


def test_hazards_are_probabilities_and_respond_to_income(params):
    soc = social.Society(params, 2, seed=0)
    f = soc._features(np.full(soc.N, 0.2))
    rich, poor = dict(f), dict(f)
    rich["lny"] = np.full_like(f["lny"], np.log(40000))
    poor["lny"] = np.full_like(f["lny"], np.log(1500))
    for k in ("conflict_onset", "coup", "dem_breakdown"):
        pr, pp = soc.hazard(k, rich), soc.hazard(k, poor)
        assert np.all((pr >= 0) & (pr <= 1)) and pp.mean() > pr.mean()
