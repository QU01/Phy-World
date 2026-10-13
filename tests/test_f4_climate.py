"""F4: clima y carbono con FaIR (paridad del paso a paso, montaje de la calibración, fugas, cuentas de
carbono, acoplamiento con el L1 y relaciones metamórficas)."""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
T0 = 55                                    # origen 2005


@pytest.fixture(scope="session")
def climate_ready(atlas_ready):
    from world4.data import climate, energy, wpp
    if not (climate.available() and energy.available() and wpp.available()):
        pytest.skip("datos de clima, energía o WPP sin instalar: python world4_cli.py --install-climate")
    return True


def _ids(n=6):
    from world4.data import climate as cd
    return list(cd.load()["params"].index[:n])


def test_stepper_reproduces_fair_run(climate_ready):
    """El bucle partido (``_fair_step.Stepper``) da exactamente lo mismo que ``FAIR.run`` con las
    mismas emisiones (historia CMIP7 1750-2023, variabilidad interna incluida)."""
    from world4.sectors import climate
    from world4.sectors._fair_step import Stepper
    a = climate.build_fair(_ids(), 2024)
    a.run(progress=False)
    b = climate.build_fair(_ids(), 2024)
    s = Stepper(b)
    s.run_to(len(b.timebounds) - 1)
    assert np.array_equal(a.temperature.loc[dict(layer=0)].data, s.temperature())
    assert np.array_equal(a.concentration.data, s.conc, equal_nan=True)


def test_override_matches_fair(climate_ready):
    """``climate._override`` (vectorizado) deja los mismos parámetros que ``FAIR.override_defaults``."""
    from world4.data import climate as cd
    from world4.sectors import climate
    a = climate.build_fair(_ids(), 1760)
    b = climate.build_fair(_ids(), 1760)
    b.override_defaults(str(cd.raw("calibrated_constrained_parameters.csv")))
    for v in a.climate_configs:
        assert np.array_equal(a.climate_configs[v].data, b.climate_configs[v].data, equal_nan=True), v
    for v in a.species_configs:
        assert np.array_equal(a.species_configs[v].data, b.species_configs[v].data, equal_nan=True), v


def test_setup_matches_the_calibration(climate_ready):
    """El montaje reproduce el calentamiento 2004-2023 que publica la calibración para cada
    configuración (``warming_baselines.csv``) a 1e-4 °C."""
    import pandas as pd
    from world4.data import climate as cd
    from world4.sectors import climate
    ids = _ids(20)
    f = climate.build_fair(ids, 2024)
    f.run(progress=False)
    T = f.temperature.loc[dict(layer=0, scenario="historical")].data
    yrs = f.timebounds[:-1]
    ann = 0.5 * (T[1:] + T[:-1])
    ann = ann - ann[(yrs >= 1850) & (yrs <= 1900)].mean(0)
    ours = ann[(yrs >= 2004) & (yrs <= 2023)].mean(0)
    ref = pd.read_csv(cd.raw("warming_baselines.csv"), index_col=0).iloc[:, 0].reindex(ids).to_numpy()
    assert np.max(np.abs(ours - ref)) < 1e-4


def test_climate_never_reads_the_future(climate_ready, monkeypatch):
    """Emisiones y GMST posteriores al origen alteradas: las reglas de emisión, la persistencia y el
    estado de FaIR en el origen no cambian."""
    from world4.core.l1_atlas import calibrate
    from world4.data import climate as cd
    from world4.sectors import climate
    base = climate.Climate(calibrate.W, T0, 4, 1, end_t=70, noise=False)
    d = cd.load()
    alt = {k: v for k, v in d.items()}
    y = d["emis_years"]
    alt["emis"] = {s: (u, np.where(y > 1950 + T0, v * 1.7, v)) for s, (u, v) in d["emis"].items()}
    alt["gmst"] = np.where(d["gmst_years"] > 1950 + T0, d["gmst"] + 0.5, d["gmst"])
    monkeypatch.setattr(cd, "load", lambda: alt)
    other = climate.Climate(calibrate.W, T0, 4, 1, end_t=70, noise=False)
    assert base.g == other.g and base.afolu_mean == other.afolu_mean and base.rho == other.rho
    assert np.array_equal(base.slow, other.slow) and np.array_equal(base.fast, other.fast)
    assert np.array_equal(base.hist[T0]["gmst"], other.hist[T0]["gmst"])


def test_carbon_mass_closes(climate_ready):
    """W7 (C): emitido acumulado = atmósfera + sumideros; la carga atmosférica es la suma de las cajas
    y equivale al exceso de concentración, a 1e-9 relativo."""
    from world4.core import coupling
    f = coupling.forecast(T0, R=4, horizon=8, f4=coupling.F4Config())
    a = f["sim"].cl.carbon_accounts()
    assert np.allclose(a["cumulative"], a["airborne"] + a["uptake"], rtol=1e-12)
    assert np.allclose(a["boxes"], a["airborne"], rtol=1e-9)
    assert np.allclose(a["conc_excess"], a["airborne_as_conc"], rtol=1e-9)


def test_f4_needs_f3(climate_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate, model, validate
    params, _ = validate.params_for(2005)
    with pytest.raises(ValueError):
        coupling.make_simulator(model)(calibrate.W, params, R=2, f2=coupling.F2Config(), f4=coupling.F4Config())


def test_coupled_climate_starts_at_the_data(climate_ready):
    """En el origen la GMST es la observada y las emisiones son las CMIP7; después la GMST del L1 es la
    de FaIR (el Atlas ya no suma la TCRE)."""
    from world4.core import coupling
    from world4.data import climate as cd
    f = coupling.forecast(T0, R=4, horizon=6, f4=coupling.F4Config())
    assert np.allclose(f["climate_gmst"][:, 0], cd.gmst_observed([1950 + T0])[0])
    assert np.allclose(f["climate_co2_ffi"][:, 0], cd.co2_ffi_observed([1950 + T0])[0])
    assert np.allclose(f["Tg"][:, 1:], f["climate_gmst"][:, 1:])
    assert np.all(np.isfinite(f["climate_co2_conc"]))


def test_more_fossil_more_warming(climate_ready):
    """Relación metamórfica de F4: con más consumo fósil hay más CO2 siempre, pero más calentamiento
    sólo a largo plazo. A 10 años la mediana se ENFRÍA (los aerosoles del fósil enmascaran); a 30 años
    calienta en la mediana y a 60 años en todos los miembros."""
    from world4.core.l1_atlas import calibrate
    from world4.sectors import climate
    R = 8
    out = {}
    for F in (0.5, 1.0, 1.5):
        cl = climate.Climate(calibrate.W, 69, R, 3, end_t=69 + 62, noise=False)
        for k in range(1, 61):
            g = cl.step(69 + k, {c: np.full(R, F) for c in climate.FUELS}, np.full(R, F), np.full(R, cl.P0))
            if k in (10, 30, 60):
                out[(F, k)] = (g.copy(), cl.hist[69 + k]["co2_conc"])
    assert np.median(out[(1.5, 10)][0]) < np.median(out[(0.5, 10)][0])           # enmascaramiento
    assert np.median(out[(1.5, 30)][0]) > np.median(out[(1.0, 30)][0]) > np.median(out[(0.5, 30)][0])
    assert np.all(out[(1.5, 60)][0] > out[(1.0, 60)][0]) and np.all(out[(1.0, 60)][0] > out[(0.5, 60)][0])
    assert all(np.all(out[(1.5, k)][1] > out[(1.0, k)][1]) for k in (10, 30, 60))


def test_tipping_is_off_by_default_and_only_adds_carbon(climate_ready):
    from world4.core import coupling
    from world4.core.l1_atlas import calibrate
    from world4.sectors import climate
    assert coupling.F4Config().tipping is False
    cl = climate.Climate(calibrate.W, 69, 4, 5, end_t=101, noise=False, tipping=True)
    cl.thr = {k: np.zeros(4) for k in climate.TIPPING}        # todos cruzados desde el primer año
    base = climate.Climate(calibrate.W, 69, 4, 5, end_t=101, noise=False)
    for k in range(1, 26):
        Q = {c: np.ones(4) for c in climate.FUELS}
        a = cl.step(69 + k, Q, np.ones(4), np.full(4, cl.P0))
        b = base.step(69 + k, Q, np.ones(4), np.full(4, base.P0))
    assert np.all(cl.hist[94]["co2_afolu"] > base.hist[94]["co2_afolu"]) and np.all(a > b)


def test_tcre_inside_ar6_likely_range(climate_ready):
    from world4.sectors import climate
    r = climate.tcre(_ids(60))
    lo, hi = climate.AR6_TCRE_LIKELY
    assert lo <= r["p"][50] <= hi
    assert r["p"][17] >= lo * 0.9 and r["p"][83] <= hi * 1.1


def test_gmst_query(climate_ready):
    from world4 import api
    d = api.query("gmst", "WLD", 2005, 2010, members=4)
    assert d.unit.startswith("degC") and 0.6 < d.p50 < 1.6
    assert d.provenance["climate"].startswith("FaIR")
