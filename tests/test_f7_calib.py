"""F7: emulador GP, history matching, ABC-SMC, discrepancia, perillas del L0/L1, recalibración de W8."""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------ métodos (sin datos)
def test_gp_gradient_is_analytic_and_loo_is_closed_form():
    from scipy.optimize import check_grad
    from world4.calib import gp
    rng = np.random.default_rng(0)
    X = rng.random((30, 3))
    y = X @ [1.0, 2.0, -1.0] + 0.3 * np.sin(4 * X[:, 0])
    ys = (y - y.mean()) / y.std()
    g = gp.GP()
    D2s = (X[:, None] - X[None]) ** 2
    nv = np.full(30, 0.01)
    p0 = np.array([-0.5, 0.1, 0.3, -0.4, -2.0])
    err = check_grad(lambda p: g._nll_grad(p, X, ys, nv, D2s)[0], lambda p: g._nll_grad(p, X, ys, nv, D2s)[1], p0)
    assert err < 1e-4
    fit = gp.GP().fit(X, y)
    Xs = rng.random((200, 3))
    mu, var = fit.predict(Xs)
    yt = Xs @ [1.0, 2.0, -1.0] + 0.3 * np.sin(4 * Xs[:, 0])
    assert np.sqrt(np.mean((mu - yt) ** 2)) < 0.05
    assert np.mean(np.abs(fit.loo()) > 3) < 0.1


def _toy():
    from world4.calib import history_matching as hm
    truth = np.array([0.3, -0.5])
    f = lambda th: np.column_stack([th[:, 0] + th[:, 1] ** 2, th[:, 0] * th[:, 1], np.exp(th[:, 0]) - th[:, 1]])  # noqa: E731
    z = f(truth[None])[0]
    v = np.full(3, 4e-4)
    pb = hm.Problem(("a", "b"), np.array([[-1, 1], [-1, 1.0]]), lambda th: (f(th), None), z, v, v, ("1", "2", "3"))
    return truth, f, z, v, pb


def test_history_matching_keeps_the_truth_and_abc_matches_the_exact_posterior():
    from world4.calib import abc, history_matching as hm
    truth, f, z, v, pb = _toy()
    H = hm.HistoryMatch(pb, seed=1, kth=1, log=lambda *a: None)
    H.run(30, max_waves=4)
    assert H.nroy(truth[None])[0]
    assert H.waves[-1].nroy_fraction < 0.1
    lp = lambda th: np.where(np.all((th >= -1) & (th <= 1), 1) & H.nroy(th), 0.0, -np.inf)  # noqa: E731
    r = abc.abc_smc(lambda n, rng: H.sample(n), lp, f, z, np.sqrt(2 * v), n=500, log=lambda *a: None)
    assert r["converged"]
    g = np.linspace(-1, 1, 801)
    A, B = np.meshgrid(g, g)
    T = np.column_stack([A.ravel(), B.ravel()])
    ll = -0.5 * np.sum((f(T) - z) ** 2 / (2 * v), 1)
    w = np.exp(ll - ll.max())
    w /= w.sum()
    m = w @ T
    sd = np.sqrt(w @ (T - m) ** 2)
    assert np.all(np.abs(r["theta"].mean(0) - m) < 0.3 * sd)
    assert np.all(np.abs(r["theta"].std(0) / sd - 1) < 0.2)


def test_discrepancy_needs_a_mark_with_a_prior():
    from phycore.plugin import ContractError
    from world4.calib import discrepancy
    h = np.arange(20)
    with pytest.raises(ContractError):
        discrepancy.fit(h, 0.01 * h, 0.012 * h, mark="[?]")
    gp = discrepancy.fit(h, 0.01 * h, 0.012 * h + 0.001 * np.sin(h), mark="[S]")
    assert gp.rho > 1.05                        # el modelo crece de menos: ρ > 1


def test_recent_bias_recalibrator_follows_a_drifting_bias():
    from world4.structural import recal
    rng = np.random.default_rng(0)
    members = rng.standard_normal(64)
    z = rng.standard_normal(3000) + 0.8          # el dato cae 0,8 σ por encima del ensemble
    cal = recal.ZCalibrator(z - 0.8, bias=0.8)
    p50, s = recal.ens_scale(members)
    y = p50 + s * (rng.standard_normal(2000) + 0.8)
    u = cal.cdf(members, y)
    assert recal.ks_distance(u) < 0.05
    assert recal.ks_distance(recal.ens_cdf(members, y)) > 0.2      # sin recalibrar está lejos
    q = cal.quantile(members, [0.1, 0.5, 0.9])
    assert q[0] < q[1] < q[2] and abs(q[1] - (p50 + 0.8 * s)) < 0.2 * s


def test_posterior_roundtrip(tmp_path):
    from world4.calib import posterior
    post = {"theta": np.random.default_rng(0).random((50, 2)), "names": ["a", "b"], "box": [[0, 1], [0, 1]],
            "cutoff": 2019, "explore": {"resource": (0.5, 2.0, "x")},
            "discrepancy": {"pop": {"h": np.arange(5), "model": np.zeros(5), "obs": np.ones(5), "mark": "[S]"}}}
    p = posterior.save(post, path=tmp_path / "p.npz")
    back = posterior.load(p)["l0r"]
    assert np.array_equal(back["theta"], post["theta"]) and back["discrepancy"]["pop"]["mark"] == "[S]"
    kn = posterior.draw_l0r(back, 7, np.random.default_rng(1), explore=True)
    assert set(kn) == {"a", "b", "resource"} and np.all((kn["resource"] >= 0.5) & (kn["resource"] <= 2.0))


# ------------------------------------------------------------------ perillas sobre el modelo (con datos)
@pytest.fixture(scope="module")
def sectors_ready(atlas_ready):
    from world4.data import climate, energy, food, wpp
    if not (food.available() and climate.available() and energy.available() and wpp.available()):
        pytest.skip("faltan datos de F3-F5")


def test_l0r_knobs_at_their_default_change_nothing(sectors_ready):
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    p, ini = model.calibrate(calibrate.W, 2019), model.initial(calibrate.W, 1992)
    a = model.L0R(p, ini).run(27)
    zero = {k: [0.0] for k in model.SHIFTS} | {"cdr": [1.0]}
    b = model.L0R(p, ini, mult=zero).run(27)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    c = model.L0R(p, ini, mult={"d_gA": [0.005]}).run(27)
    assert c["Y"][-1].sum() > a["Y"][-1].sum() * 1.1
    with pytest.raises(KeyError):
        model.L0R(p, ini, mult={"not_a_knob": [1.0]})


def test_l1_knobs_at_zero_are_bit_identical_and_move_their_sector(sectors_ready):
    from world4.core import coupling
    o = 1985 - 1950
    kw = dict(R=2, seed=0, horizon=3, f5=coupling.F5Config())
    a = coupling.forecast(o, **kw)
    b = coupling.forecast(o, calib={k: 0.0 for k in coupling.L1_KNOBS}, **kw)
    assert all(np.array_equal(a[k], b[k], equal_nan=True) for k in a if k != "sim")
    c = coupling.forecast(o, calib={"intensity": -0.02}, **kw)
    assert np.nansum(c["energy_PE"][:, :, -1]) < 0.97 * np.nansum(a["energy_PE"][:, :, -1])
    with pytest.raises(KeyError):
        coupling.forecast(o, calib={"psi": 0.1}, **kw)


def test_w8_mutant_without_recalibration_is_caught():
    """El mutante de W8 («quitar la recalibración»): el PIT crudo no cumple ni la cobertura ni la forma."""
    from world4.structural import store
    try:
        from world4.structural import questions
        questions.records()
    except store.StoreMissing:
        pytest.skip("sin registros de preguntas: python world4_cli.py --build-store")
    from world4.structural import recal
    X = questions.records()["numeric"]
    raw = X[X[:, 2] >= questions.FIRST_TEST_YEAR, 6]
    cov = np.mean((raw >= 0.1) & (raw <= 0.9))
    assert not (0.75 <= cov <= 0.85 and recal.ks_distance(raw) <= 0.05)


def test_the_box_does_not_decide():
    from world4.calib import history_matching as hm
    rng = np.random.default_rng(0)
    box = np.array([[-1.0, 1.0], [0.0, 2.0], [-1.0, 1.0]])
    th = np.column_stack([rng.uniform(0.9, 1.0, 500), rng.uniform(0.0, 0.3, 500), rng.normal(0, 0.2, 500)])
    edge = hm.edge_knobs(("a", "b", "c"), th, box)
    assert edge == ["a"]                       # b toca el 0, que es un borde físico; c está dentro
    wide = hm.widen(("a", "b", "c"), box, ["a", "b"])
    assert np.allclose(wide, [[-2.0, 2.0], [0.0, 4.0], [-1.0, 1.0]])


def test_run_l0r_with_the_shipped_posterior_has_bands_and_is_exploratory(sectors_ready):
    from world4 import api
    from world4.calib import posterior
    if not posterior.PATH.is_file():
        pytest.skip("sin models/posterior.npz: python world4_cli.py --calibrate")
    r = api.run_l0r(posterior=True, members=40, years=30)
    assert r["exploratory"] is True and r["posterior_cutoff"] == 2019
    for v in ("pop", "gdp", "gmst"):
        b = r["world"][v]
        assert all(lo <= m <= hi for lo, m, hi in zip(b["p10"], b["p50"], b["p90"])) and b["unit"]
    assert r["world"]["pop"]["p90"][-1] > r["world"]["pop"]["p10"][-1]
