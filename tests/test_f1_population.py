"""F1: población y capital humano endógenos (cohortes, acoplamiento al L1, fugas y contabilidad)."""
import dataclasses

import numpy as np
import pytest

from world4.sectors import _panel, population

pytestmark = pytest.mark.filterwarnings("ignore")
T0 = 40                                    # origen 1990


@pytest.fixture(scope="session")
def wpp_ready(atlas_ready):
    from world4.data import wpp
    if not wpp.available():
        pytest.skip("datos WPP 2024 por edad sin instalar: python world4_cli.py --install-wpp")
    return True


# ------------------------------------------------------------------ piezas sin datos
def test_life_expectancy_of_constant_hazard():
    m = np.full(population.A, 0.02)
    e0 = population.life_expectancy(m)
    assert abs(e0 - 1 / 0.02) / 50 < 0.02                         # exponencial: e0 = 1/m (con a0, a = 0.5)


def test_pwt_schooling_map_round_trips():
    s = np.linspace(0, 15, 61)
    assert np.allclose(population.phi_inv(population.phi_hc(s)), s)
    assert np.all(np.diff(population.phi_hc(s)) > 0)


def test_sign_restriction_drops_wrong_signed_term():
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal(n)
    y = 0.5 * x + 0.1 * rng.standard_normal(n)
    X = np.column_stack([np.ones(n), x])
    eq = _panel.fit("t", y, X, np.arange(n) % 20, ("const", "x"), 2000, signs={"x": -1})
    assert eq.dropped == ("x",) and eq.beta[1] == 0.0
    eq = _panel.fit("t", y, X, np.arange(n) % 20, ("const", "x"), 2000, signs={"x": +1})
    assert eq.dropped == () and abs(eq.beta[1] - 0.5) < 0.05


def test_ar1_shock_has_its_declared_scale():
    rng = np.random.default_rng(1)
    s = _panel.Shock(0.3, 0.8, (1, 20000), rng)
    xs = [s()[0].copy() for _ in range(5)]
    assert abs(np.std(xs[-1]) - 0.3) < 0.01
    assert abs(np.corrcoef(xs[-2], xs[-1])[0, 1] - 0.8) < 0.02


# ------------------------------------------------------------------ con el L1
def _forecast(**kw):
    from world4.core import coupling
    return coupling.forecast(T0, R=4, seed=T0, **kw)


def test_coupled_simulator_without_sectors_is_the_atlas(wpp_ready):
    """P3: con los sectores apagados, la subclase es bit a bit el Simulator del Atlas."""
    from world4.core import coupling
    from world4.core.l1_atlas import behavior, calibrate, model, validate
    W = calibrate.W
    params, _ = validate.params_for(1990)
    bm = behavior.BehaviorModel(cutoff=T0).fit(W, calibrate.LY)
    outs = []
    for cls, kw in ((model.Simulator, {}), (coupling.make_simulator(model), {"f1": None})):
        bc = behavior.BehaviorPolicy(bm, params, noise=1.0, freeze=T0, seed=T0 + 7)
        sim = cls(W, params, R=4, seed=T0, **kw)
        outs.append(sim.run(0, T0 + 6, policy=lambda s, t: s.hist_actions(t) if t <= T0 else bc(s, t),
                            freeze_from=T0, assimilate_at=T0))
    for k in ("ly", "hazard", "debt", "demo", "conflict"):
        assert np.array_equal(outs[0][k], outs[1][k], equal_nan=True), k


def test_endogenous_forecast_never_reads_future_population_or_hc(wpp_ready):
    """El test de fuga de D1/D2: alterar pop y hc DESPUÉS del origen no cambia nada."""
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    pop, hc = W.pop.copy(), W.hc.copy()
    pop[:, T0 + 1:] *= 3.0
    hc[:, T0 + 1:] *= 0.5
    Wm = dataclasses.replace(W, pop=pop, hc=hc, _locf=W._locf)
    a = _forecast(pop_mode="endogenous")
    b = _forecast(pop_mode="endogenous", world=Wm)
    for k in ("ly", "pop", "hc", "debt"):
        assert np.array_equal(a[k], b[k], equal_nan=True), k
    c = _forecast(pop_mode="observed", world=Wm)                     # el mutante sí lo ve
    assert not np.allclose(a["ly"][:, :, -1], c["ly"][:, :, -1], equal_nan=True)


def test_fits_use_only_data_up_to_the_origin(wpp_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    extra = {k: v.copy() for k, v in W.extra.items()}
    for k in ("tfr", "life_expectancy", "net_migration_rate", "urban_share"):
        extra[k][:, T0 + 1:] = np.nan_to_num(extra[k][:, T0 + 1:]) * 1.7 + 0.3
    hc = W.hc.copy()
    hc[:, T0 + 1:] *= 1.5
    Wm = dataclasses.replace(W, extra=extra, hc=hc, _locf={})
    a, b = coupling.fits(W, calibrate.LY, T0), coupling.fits(Wm, calibrate.LY, T0)
    for k in a:
        assert np.array_equal(a[k].beta, b[k].beta), k
        assert a[k].cutoff_year == 1990


def test_population_starts_at_the_l1_population_and_moves_after(wpp_ready):
    from world4.core.l1_atlas import calibrate
    f = _forecast(pop_mode="endogenous")
    P = f["pop"]
    assert np.allclose(P[:, :, 0], calibrate.W.pop[None, :, T0], equal_nan=True)
    assert np.all(np.isfinite(P[:, :, -1]))
    assert np.std(np.nansum(P[:, :, -1], 1)) > 0                   # los miembros difieren
    assert not np.allclose(P[0, :, -1], calibrate.W.pop[:, T0 + 10])


def test_cohort_accounting_closes(wpp_ready):
    """Contabilidad por paso: P_t = supervivientes + nacimientos + migración neta (suma mundial 0)."""
    from world4.core import coupling
    f = _forecast(pop_mode="endogenous")
    ps = f["sim"].pop_s
    P0 = ps.P.copy()
    tot0 = P0.sum((2, 3))
    R, N = tot0.shape
    z = np.zeros((R, N))
    ps.advance(10_000, z + 9.0, z, z, z, z, z + 8.0)
    P1 = ps.P
    surv = np.exp(-ps._mx(ps.kappa))                                 # las tasas del paso
    act = ps.ok[None] & (tot0 > 0)
    survivors = (P0[..., :-1] * surv[..., :-1]).sum((2, 3)) + (P0[..., -1] * surv[..., -1]).sum(-1)
    b = ps.bmale[None]
    newborn = ps.last_births * (b * np.sqrt(surv[:, :, 0, 0]) + (1 - b) * np.sqrt(surv[:, :, 1, 0]))
    migr = ps.s["mig"] * tot0
    expected = survivors + newborn + migr
    clipped = np.any(P1 == 0, (2, 3))                                 # un grupo vaciado por la emigración
    ok = act & ~clipped
    assert np.max(np.abs(P1.sum((2, 3))[ok] / expected[ok] - 1)) < 1e-9
    assert abs(np.sum(np.where(act, migr, 0.0), 1)).max() < 1e-9 * tot0.sum()   # migración neta mundial = 0
    assert np.all(ps.last_births[act] > 0) and np.all(P1 >= 0)
    assert coupling.F1Config().population


def test_observed_drivers_reproduce_population_within_two_percent(wpp_ready):
    """Con TFR, e0 y migración observadas, las cohortes siguen a la población de la PWT."""
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    f = coupling.forecast(T0, R=1, pop_mode="endogenous", f1=coupling.F1Config(noise=False, drivers="observed"))
    wp = np.nansum(f["pop"][0, :, -1])
    assert abs(wp / np.nansum(calibrate.W.pop[:, T0 + 10]) - 1) < 0.02


def test_hc_is_continuous_and_responds_to_education_effort(wpp_ready):
    from world4.core.l1_atlas import calibrate
    from world4.sectors import human_capital
    f = _forecast(pop_mode="endogenous")
    assert np.allclose(f["hc"][:, :, 0], calibrate.W.hc[None, :, T0], equal_nan=True)
    hc_s = f["sim"].hc_s
    assert hc_s.eq.beta[3] >= 0                                    # signo declarado del canal r
    hs = []
    for r in (0.0, 0.2):
        h = human_capital.HumanCapital(calibrate.W, calibrate.LY, T0, hc_s.eq, f["sim"].pop_s, 4, 0, noise=False)
        z = np.zeros((4, calibrate.W.N))
        for t in range(1, 30):
            out = h.advance(T0 + t, z + 9.0, z + r)
        hs.append(np.nanmean(out))
    assert hs[1] >= hs[0]


def test_youth_bulge_comes_from_the_cohorts(wpp_ready):
    f = _forecast(pop_mode="endogenous")
    sim = f["sim"]
    t = T0 + 5
    y = sim.ex("youth_bulge_15_24_adult", t, 0.28)
    assert y.shape == (4, sim.w.N) and np.all((y > 0) & (y < 0.6))
