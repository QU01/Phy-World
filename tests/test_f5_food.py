"""F5: alimentos, suelo, nitrógeno y agua (fugas, cuentas cerradas, acoplamiento, relaciones)."""
import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
T0 = 55                                    # origen 2005


@pytest.fixture(scope="session")
def food_ready(atlas_ready):
    from world4.data import climate, energy, food, wpp
    if not (food.available() and climate.available() and energy.available() and wpp.available()):
        pytest.skip("datos de alimentos, clima, energía o WPP sin instalar: python world4_cli.py --install-food")
    return True


@pytest.fixture(scope="session")
def f5_run(food_ready):
    from world4.core import coupling
    return coupling.forecast(T0, R=4, horizon=8, f5=coupling.F5Config())


def test_food_never_reads_the_future(food_ready, monkeypatch):
    """Datos de FAOSTAT posteriores al origen x1.7: las ecuaciones de F5 (que reconstruyen el panel) no cambian."""
    from world4.core.l1_atlas import calibrate
    from world4.data import food as fdata
    from world4.sectors import food
    W, LY = calibrate.W, calibrate.LY
    base = food.fit(W, LY, T0)
    orig = fdata.aligned

    def altered(iso3, isonum, years):
        out = orig(iso3, isonum, years)
        k = np.asarray(years) > 1950 + T0
        for key in list(out):
            if key != "world":
                out[key] = out[key].copy()
                out[key][:, k] *= 1.7
        return out

    monkeypatch.setattr(fdata, "aligned", altered)
    Wm = dataclasses.replace(W, _locf=W._locf)
    alt = food.fit(Wm, LY, T0)
    assert not np.allclose(food.panel(Wm, LY)["lC"][:, T0 + 3], food.panel(W, LY)["lC"][:, T0 + 3], equal_nan=True)
    for k in base:
        if k != "meta":
            assert np.array_equal(base[k].beta, alt[k].beta), k


def test_land_and_nitrogen_close(f5_run):
    """W7: cultivo + pastos + bosque + resto = superficie de tierra; entradas de N = salidas + balance."""
    fd = f5_run["sim"].fd
    L = fd.land()
    ok = np.isfinite(L["land"]) & fd.ok[None]
    tot = L["cropland"] + L["pasture"] + L["forest"] + L["other"]
    assert np.allclose(tot[ok], L["land"][ok], rtol=1e-12)
    nb = fd.n_budget()
    good = np.isfinite(nb["input"])
    assert np.allclose(nb["input"][good], (nb["output"] + nb["balance"])[good], rtol=1e-12)
    assert np.allclose(sum(nb["inputs"].values())[good], nb["input"][good], rtol=1e-12)


def test_food_starts_at_the_data(f5_run):
    from world4.core.l1_atlas import calibrate
    from world4.data import food as fdata
    a = fdata.aligned(calibrate.W.iso3, calibrate.W.isonum, calibrate.W.years)
    fd = f5_run["sim"].fd
    ok = fd.ok
    assert np.allclose(f5_run["food_prod"][0, ok, 0], a["cereal_prod"][ok, T0], rtol=1e-9)
    assert np.allclose(f5_run["food_cropland"][0, ok, 0], a["cropland"][ok, T0], rtol=1e-9)
    assert np.std(f5_run["foodw_cereal_prod"][:, -1]) > 0


def test_f5_needs_f4(food_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, model, validate
    params, _ = validate.params_for(2005)
    with pytest.raises(ValueError):
        coupling.make_simulator(model)(calibrate.W, params, R=2, f2=coupling.F2Config(), f3=coupling.F3Config(),
                                       f5=coupling.F5Config())


def test_footprint_replaces_the_frozen_atlas_series(f5_run):
    """D4: después del origen la huella y la biocapacidad que ve el Atlas son las del sector (no el dato
    congelado del origen)."""
    sim = f5_run["sim"]
    t = sim.origin + 3
    ef = sim.ex("ef_cons_pc", t, 2.5)
    assert ef.shape == (4, sim.w.N)
    frozen = sim.w.locf("ef_cons_pc", 2.5)[:, sim.origin]
    ok = np.isfinite(ef[0]) & np.isfinite(frozen) & sim.fd.ef_on
    assert not np.allclose(ef[:, ok], frozen[None, ok])


def test_heat_lowers_yields_in_hot_countries(food_ready):
    """Relación de F5 con coeficientes estimados (<= 2019): +2 °C de anomalía local baja el rendimiento de
    cereales de los países cálidos."""
    from world4.core.l1_atlas import calibrate
    from world4.core.coupling import food_fits
    from world4.sectors import food
    W, LY = calibrate.W, calibrate.LY
    eqs = {k: v for k, v in food_fits(W, LY, 69).items() if k != "_e0_food"}
    out = []
    for dT in (0.0, 2.0):
        fd = food.Food(W, LY, 69, eqs, 2, 0, noise=False)
        R, N = 2, W.N
        ly = np.broadcast_to(np.nan_to_num(LY[:, 69]), (R, N))
        pop = np.broadcast_to(W.pop[:, 69], (R, N))
        tloc = fd.tbar[None] + fd.tdev_prev + dT
        fd.step(70, ly, np.zeros((R, N)), tloc, pop)
        out.append(fd.lY.copy())
    hot = (fd.tbar > 24) & fd.ok
    assert np.all(out[1][:, hot] < out[0][:, hot])


def test_agricultural_emissions_follow_food(f5_run):
    cl = f5_run["sim"].cl
    assert cl.agri is not None and "nin_ratio" in cl.agri
    assert np.all(np.isfinite(f5_run["climate_gmst"]))


def test_food_query(food_ready):
    from world4 import api
    d = api.query("cereal_production", "IND", 2005, 2010, members=4)
    assert d.unit.startswith("t") and d.p50 > 1e8
    assert d.provenance["food"].startswith("endogenous")
