"""F3: energía con EROI (mezcla, cuentas cerradas, fugas, aprendizaje, materiales, acoplamiento)."""
import dataclasses

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
T0 = 55                                    # origen 2005


@pytest.fixture(scope="session")
def energy_ready(atlas_ready):
    from world4.data import energy, wpp
    if not (energy.available() and wpp.available()):
        pytest.skip("datos de energía o WPP sin instalar: python world4_cli.py --install-energy --install-wpp")
    return True


def _f3(**kw):
    from world4.core import coupling
    return coupling.forecast(T0, R=kw.pop("R", 4), f2=coupling.F2Config(), f3=coupling.F3Config(), **kw)


def test_energy_accounts_close_and_mix_sums(energy_ready):
    from world4.sectors import energy
    f = _f3()
    en = f["sim"].en
    a = en.accounts()
    assert np.max(np.abs(a["gross"] - a["own_use"] - a["net"])) <= 1e-9 * np.max(a["gross"])
    total = sum(f[f"energy_gen_{s}"] for s in energy.SOURCES)
    ok = np.isfinite(f["energy_PE"])
    assert np.allclose(total[ok], f["energy_PE"][ok], rtol=1e-12)
    for s in energy.SOURCES:
        g = f[f"energy_gen_{s}"]
        assert np.all(g[np.isfinite(g)] >= -1e-9), s
    assert np.all(a["net"] < a["gross"] + 1e-12) and np.all(a["net"][en.ok[None].repeat(4, 0)] > 0)


def test_energy_starts_at_the_data(energy_ready):
    from world4.core.l1_atlas import calibrate
    from world4.sectors import energy
    p = energy.panel(calibrate.W)
    f = _f3()
    pe0 = p["pe"][:, T0]
    ok = np.isfinite(pe0)
    assert np.allclose(f["energy_PE"][0, ok, 0], pe0[ok])
    assert np.std(np.nansum(f["energy_PE"][:, :, -1], 1)) > 0


def test_energy_never_reads_the_future(energy_ready, monkeypatch):
    """Datos de energía (todas las fuentes y la energía total) posteriores al origen x1.9: los ajustes de
    F3, que reconstruyen todo el panel, no cambian."""
    from world4.core.l1_atlas import calibrate
    from world4.data import energy as edata
    from world4.sectors import energy
    W = calibrate.W
    base = energy.fit(W, T0)
    orig = edata.aligned

    def altered(iso3, isonum, years):
        out = orig(iso3, isonum, years)
        k = np.asarray(years) > 1950 + T0
        for key in list(out):
            if key == "pe" or key.startswith("src_"):
                out[key] = out[key].copy()
                out[key][:, k] *= 1.9
        return out

    monkeypatch.setattr(edata, "aligned", altered)
    Wm = dataclasses.replace(W, _locf=W._locf)               # otro objeto: el panel se reconstruye
    alt = energy.fit(Wm, T0)
    assert not np.allclose(energy.panel(Wm)["pe"][:, T0 + 2], energy.panel(W)["pe"][:, T0 + 2], equal_nan=True)
    for k in base:
        if k != "meta":
            assert np.array_equal(base[k].beta, alt[k].beta), k


def test_f3_needs_f2(energy_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, model, validate
    params, _ = validate.params_for(2005)
    with pytest.raises(ValueError):
        coupling.make_simulator(model)(calibrate.W, params, R=2, f3=coupling.F3Config())


def test_dearer_fossil_fuels_bring_more_wind_and_solar(energy_ready):
    """Relación metamórfica de F3: con un precio fósil más alto, la eólica + solar no es menor (aprendizaje
    y brecha de coste, coeficiente identificado con datos <= 2019)."""
    from world4.core.l1_atlas import calibrate
    from world4.core.coupling import energy_fits
    from world4.sectors import energy
    W = calibrate.W
    eqs = energy_fits(W, 69)
    assert eqs["vre"].beta[3] > 0
    out = []
    for dp in (0.0, 0.7):
        en = energy.Energy(W, 69, eqs, 2, 0, noise=False)
        for k in range(1, 16):
            en.step(69 + k, np.full((2, W.N), 0.02), np.full((2, 1), dp))
        out.append(float(np.sum(en.generation()["vre"][0]) / np.sum(en.PE[0])))
    assert out[1] > out[0], out


def test_learning_lowers_vre_cost_and_raises_its_eroi(energy_ready):
    from world4.core.l1_atlas import calibrate
    from world4.core.coupling import energy_fits
    from world4.sectors import energy
    en = energy.Energy(calibrate.W, 69, energy_fits(calibrate.W, 69), 2, 0, noise=False)
    e0 = en.eroi()["vre"].copy()
    en.cum = en.cum * 4
    assert np.all(en.eroi()["vre"] > e0)
    sh_before = en.sh["vre"].copy()
    en.sh["vre"] = sh_before                      # misma cuota: sólo el aprendizaje
    assert np.all(en.tfp_shift()[:, en.ok] >= -1e-12)


def test_materials_follow_new_wind_and_solar(energy_ready):
    from world4.sectors import energy
    f = _f3(R=4, horizon=8)
    cu, ni = f["material_copper"], f["material_nickel"]
    ok = np.isfinite(cu)
    assert np.all(cu[ok] >= 0) and np.allclose(ni[ok] / np.maximum(cu[ok], 1e-12) * (cu[ok] > 0),
                                               (cu[ok] > 0) * energy.MATERIALS_T_PER_MW["nickel"]
                                               / energy.MATERIALS_T_PER_MW["copper"])


def test_physical_shortage_lowers_output(energy_ready):
    from world4.core.l1_atlas import calibrate
    from world4.core.coupling import energy_fits
    from world4.sectors import energy
    en = energy.Energy(calibrate.W, 69, energy_fits(calibrate.W, 69), 2, 0, noise=False)
    d0 = en.shortage_shift(np.zeros((2, 1)))
    d1 = en.shortage_shift(np.full((2, 1), 0.2))
    assert np.all(np.asarray(d0) == 0) and np.all(d1[:, en.ok] < 0)
    assert np.all(en.shortage_shift(np.full((2, 1), 0.4))[:, en.ok] <= d1[:, en.ok])


def test_primary_energy_query(energy_ready):
    from world4 import api
    d = api.query("primary_energy", "DEU", 2005, 2010, members=4)
    assert d.unit.startswith("TWh") and 1000 < d.p50 < 6000
    assert d.provenance["energy"].startswith("endogenous")
    _ = dataclasses
