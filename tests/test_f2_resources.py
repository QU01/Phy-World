"""F2: escasez global endógena (mercados por clase, psi_t como salida, normalización, W6)."""
import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
T0 = 40                                    # origen 1990


def _f2(**kw):
    from world4.core import coupling
    return coupling.F2Config(**kw)


def test_psi_equal_to_the_atlas_reproduces_the_atlas(atlas_ready):
    """La normalización de la Cobb-Douglas es exactamente 0 cuando psi coincide con la del Atlas."""
    from world4.core import coupling
    from world4.core.l1_atlas import behavior, calibrate, model, validate
    W = calibrate.W
    params, _ = validate.params_for(1990)
    bm = behavior.BehaviorModel(cutoff=T0).fit(W, calibrate.LY)
    outs = []
    for forced in (False, True):
        sim = coupling.make_simulator(model)(W, params, R=2, seed=T0, data=(W, calibrate.LY), f2=_f2(noise=False))
        if forced:
            orig = sim._setup_markets

            def setup(t, sim=sim, orig=orig):
                orig(t)

                def step(tt, gY):
                    sim.mk.hist[tt] = np.full((sim.R, 1), float(model.Simulator._psi(sim, tt)))
                    return sim.mk.hist[tt]
                sim.mk.step = step
            sim._setup_markets = setup
        else:
            sim.f2 = None
        bc = behavior.BehaviorPolicy(bm, params, noise=0.0, freeze=T0, seed=7)
        outs.append(sim.run(0, T0 + 5, policy=lambda s, t: s.hist_actions(t) if t <= T0 else bc(s, t),
                            freeze_from=T0, assimilate_at=T0))
    for k in ("ly", "res_share", "debt"):
        assert np.allclose(outs[0][k], outs[1][k], equal_nan=True, rtol=0, atol=1e-9), k


def test_markets_never_read_the_future(atlas_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    world = dict(W.world)
    for k in list(world):
        if k.startswith("recursos__"):
            world[k] = world[k].copy()
            world[k][T0 + 1:] = world[k][T0 + 1:] * 2.5
    Wm = dataclasses.replace(W, world=world, _locf=W._locf)
    from world4.sectors import resources
    a, b = resources.fit(W, T0), resources.fit(Wm, T0)
    for key in a:
        assert np.array_equal(a[key].beta, b[key].beta), key
    f = coupling.forecast(T0, R=4, pop_mode="observed", f2=_f2())
    g = coupling.forecast(T0, R=4, pop_mode="observed", f2=_f2(), world=Wm)
    assert np.array_equal(f["psi"], g["psi"])


def test_psi_starts_at_the_data_and_is_an_output(atlas_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    f = coupling.forecast(T0, R=8, pop_mode="observed", f2=_f2())
    assert np.allclose(f["psi"][:, 0], calibrate.W.psi_t[T0])
    assert np.all(np.isfinite(f["psi"])) and np.all(f["psi"] > 0) and np.all(f["psi"] < 1)
    assert np.std(f["psi"][:, -1]) > 0
    for c in ("oil", "gas", "coal", "metals"):
        assert np.all(f[f"price_{c}"][:, -1] > 0)


def test_market_accounting(atlas_ready):
    from world4.core.l1_atlas import calibrate
    from world4.core.coupling import market_fits
    from world4.sectors import resources
    mk = resources.Markets(calibrate.W, 69, market_fits(calibrate.W, 69), 4, 0, u_scale=0.05)
    for k in range(1, 60):
        X0 = {c: mk.st[c]["X"].copy() for c in ("oil", "gas")}
        U = {c: mk.st[c]["U"] for c in ("oil", "gas")}
        mk.step(69 + k, np.full(4, 0.03))
        for c in ("oil", "gas"):
            Q = np.exp(mk.st[c]["lnQ"])
            assert np.allclose(mk.st[c]["X"] - X0[c], Q)
            assert np.all(Q <= resources.R_MAX_RATE * (U[c] - X0[c]) * (1 + 1e-9) + 1e-6)
            assert np.all(mk.st[c]["X"] <= U[c] * (1 + 1e-9))


def test_w6_more_resource_peaks_later_and_no_limit_no_peak(atlas_ready):
    """Relación metamórfica de W6: más recurso último -> pico más tarde; sin límite -> sin pico."""
    from world4.core.l1_atlas import calibrate
    from world4.sectors import resources
    peaks = []
    for u in (0.5, 1.0, 1.5):
        r = resources.standalone(calibrate.W, 69, 150, g_world=0.02, u_scale=u)
        peaks.append(r["years"][int(np.argmax(r["Q"]["oil"]))])
    assert peaks[0] < peaks[1] < peaks[2], peaks
    r = resources.standalone(calibrate.W, 69, 150, g_world=0.02, u_scale=1e6)
    assert r["years"][int(np.argmax(r["Q"]["oil"]))] == r["years"][-1]
