"""F6: L0 regional (10 regiones), dominancia de lazos, P2, W6 y W10."""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="session")
def l0r(atlas_ready):
    from world4.data import climate, energy, food, wpp
    if not (food.available() and climate.available() and energy.available() and wpp.available()):
        pytest.skip("faltan datos de F3-F5")
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    p = model.calibrate(calibrate.W, 2019)
    return p, model.initial(calibrate.W, 2019)


def test_every_country_has_a_region(atlas_ready):
    from world4.core.l0r import regions
    from world4.core.l1_atlas import calibrate
    r = regions.region_index(calibrate.W.iso3, calibrate.W.isonum)
    assert np.all(r >= 0) and len(set(r.tolist())) == 10


def test_l0r_starts_at_the_data_and_has_about_80_stocks(l0r):
    from world4.core.l0r import model
    p, ini = l0r
    m = model.L0R(p, ini)
    st = m.x0()
    assert sum(v.size for v in st.values()) == 10 * len(model.REG_STOCKS) + len(model.WORLD_STOCKS)
    r = m.run(1, record=("P",))
    assert np.allclose(r["P"][0, 0], ini.reg["P"])


def test_l0r_never_reads_the_future(atlas_ready, monkeypatch):
    """Datos regionales posteriores al corte x1.5: la calibración con corte 2005 no cambia."""
    from world4.core.l0r import data as rdata
    from world4.core.l0r import model
    from world4.core.l1_atlas import calibrate
    W = calibrate.W
    base = model.calibrate(W, 2005)
    real = rdata.regional(W)
    alt = {"levels": {k: v.copy() for k, v in real["levels"].items()},
           "chain": {k: v.copy() for k, v in real["chain"].items()},
           "logindex": {k: v.copy() for k, v in real["logindex"].items()},
           **{k: real[k] for k in ("members", "ridx", "base")}}
    for grp in ("levels", "chain"):
        for k in alt[grp]:
            alt[grp][k][:, 2006 - 1950:] *= 1.5
    for k in alt["logindex"]:
        alt["logindex"][k][:, 2006 - 1950:] += 0.4
    monkeypatch.setattr(rdata, "regional", lambda W, base_year=rdata.BASE_DEFAULT: alt)
    other = model.calibrate(W, 2005)
    for name in ("g_A", "lam_A", "b_i", "tcre_eff"):
        assert getattr(base, name) == getattr(other, name), name
    assert np.array_equal(base.u_A, other.u_A) and base.c_coef[1] == other.c_coef[1]


def test_loop_score_switches_at_the_logistic_inflection():
    """Validación de la puntuación de lazos: en la logística, el lazo dominante pasa de nacimientos (+) a
    muertes (-) en el paso donde P cruza K/2."""
    from world4.core.l0r import loops
    r, K = 0.1, 1000.0
    P = [10.0]
    for _ in range(150):
        P.append(P[-1] + r * P[-1] - r * P[-1] ** 2 / K)
    ev = lambda sts, pv: {"births": r * np.array([s["P"] for s in sts]),  # noqa: E731
                          "deaths": r * np.array([s["P"] for s in sts]) ** 2 / K}
    res = loops.analyse(["P"], {"births": ("P", 1), "deaths": ("P", -1)}, ev, [{"P": x} for x in P],
                        [None] * len(P), np.arange(len(P)))
    d = res["dominant"]
    sw = [t for t in range(1, len(d)) if d[t] != d[t - 1]]
    assert len(sw) == 1 and P[sw[0]] < K / 2 * 1.1 and P[sw[0] - 1] < K / 2 < P[sw[0] + 1]
    assert d[0].endswith("[births] -> P") and res["relative"][0][res["loops"].index(d[0])] > 0
    assert d[-1].endswith("[deaths] -> P")


def test_l0r_loops_are_found(l0r):
    from world4.core.l0r import loops, model
    p, ini = l0r
    m = model.L0R(p, ini)
    out = m.run(20, keep_states=True)
    res = loops.analyse(*loops.l0_world_view(m, out))
    names = res["loops"]
    assert "P -[births] -> P" in names and "K -[investment] -> K" in names and "K -[depreciation] -> K" in names
    assert np.all(np.abs(res["relative"]).sum(1) <= 1 + 1e-9)


def test_w6_relations_and_mutant(l0r):
    from world4.validate import w6
    good = w6.relations_l0r()
    assert good["ok"], {k: v for k, v in good.items() if isinstance(v, dict) and not v.get("ok", True)}
    bad = w6.relations_l0r({"damage_sign": -1.0})
    assert not bad["ok"]


def test_w10_l0_under_10_ms(l0r):
    import time
    from world4.core.l0r import model
    p, ini = l0r
    m = model.L0R(p, ini)
    m.run(81)
    ts = []
    for _ in range(20):
        t = time.perf_counter()
        m.run(81)
        ts.append(time.perf_counter() - t)
    assert np.median(ts) < 0.020          # holgura x2 frente al listón de W10 (10 ms) para máquinas cargadas
