"""F12: la capa de eventos (L2). Todo corre sobre el mundo sintético, sin descargar nada; las pruebas con el
panel real de Oriente Medio se saltan si no está instalado (``world4_cli.py --install-l2``)."""
import json

import numpy as np
import pytest

from world4.core.l2 import agents, baseline, displacement, forecast, regions, stats, synthetic
from world4.core.l2 import panel as pn

T0 = 108


@pytest.fixture(scope="module")
def world():
    return synthetic.world(seed=3)


@pytest.fixture(scope="module")
def P(world):
    return pn.from_arrays(world)


@pytest.fixture(scope="module")
def fc(P):
    return forecast.run(P, T0, 12, members=24, seed=5, disp_params=displacement.fit(P, T0, maxiter=60))


# ------------------------------------------------------------------ panel y contabilidad
def test_three_levels_add_up(P, world):
    """Sitios -> celdas -> países: nada se pierde salvo lo que UCDP no localiza, que sólo cuenta en el país."""
    total = float(world["ev_best"].sum())
    assert P.site_deaths().sum() == pytest.approx(total)
    assert P.deaths("country").sum() == pytest.approx(total)
    unlocated = float(world["ev_best"][world["ev_cell"] < 0].sum())
    assert unlocated > 0
    assert P.deaths("cell").sum() == pytest.approx(total - unlocated)
    by_type = sum(P.deaths("country", k).sum() for k in pn.TYPES)
    assert by_type == pytest.approx(total)


def test_event_across_months_is_spread_not_duplicated(P, world):
    span = world["ev_t1"] > world["ev_t0"]
    assert span.any()
    i = int(np.where(span)[0][0])
    d = int(world["ev_dyad"][i])
    assert P.dyad_series("deaths")[d].sum() == pytest.approx(world["ev_best"][world["ev_dyad"] == d].sum())
    assert P.dyad_series("events")[d].sum() == pytest.approx((world["ev_dyad"] == d).sum())


def test_border_events_keep_their_country(P, world):
    """Un evento en una celda de otro país cuenta en SU país: para eso existen los sitios."""
    cc = world["cell_country"][np.maximum(world["ev_cell"], 0)]
    other = (world["ev_cell"] >= 0) & (cc != world["ev_country"])
    assert other.any()
    for k in range(P.K):
        assert P.deaths("country")[k].sum() == pytest.approx(world["ev_best"][world["ev_country"] == k].sum())


def test_lagged_features_do_not_read_the_present():
    rng = np.random.default_rng(0)
    x = rng.poisson(1.0, (3, 40)).astype(float)
    y = x.copy()
    y[:, 25:] += 100.0                                      # cambiar el mes 25 y los siguientes...
    for k in pn.WINDOWS:
        assert np.array_equal(pn.lagged_mean(x, k)[:, :26], pn.lagged_mean(y, k)[:, :26])   # ...no toca la columna 25
    assert np.array_equal(pn.months_since(x)[:, :26], pn.months_since(y)[:, :26])
    assert pn.lagged_mean(x, 3)[:, 10] == pytest.approx(x[:, 7:10].mean(1))
    assert pn.lagged_mean(x, 3).shape == (3, 41)            # la columna T describe el mes siguiente al último dato


def test_truncated_panel_has_no_future(P):
    Q = P.truncated(T0)
    assert Q.T == T0 and Q.ev["t0"].max() < T0 and Q.ev["t1"].max() < T0
    assert Q.site_deaths().sum() == pytest.approx(P.ev["best"][P.ev["t1"] < T0].sum())
    assert Q.site_deaths().sum() <= P.site_deaths()[:, :T0].sum() + 1e-9      # lo que cruza T0 no está
    y = P.ym(T0)[0]
    assert np.all(np.isnan(Q.ref_stock[..., Q.years >= y])) and np.all(np.isfinite(Q.ref_stock[..., Q.years < y]))


# ------------------------------------------------------------------ estadística
def test_logit_and_poisson_recover_known_coefficients():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30000, 3)) * [1, 10, 100] + [0, 5, -50]             # escalas muy distintas
    b = np.array([-1.0, 0.8, -0.05, 0.01])
    y = (rng.random(len(X)) < stats.sigmoid(stats.predict(b, X))).astype(float)
    assert np.allclose(stats.logit_fit(X, y), b, atol=0.08)
    n = rng.poisson(np.exp(stats.predict(b / 2, X)))
    assert np.allclose(stats.poisson_fit(X, n), b / 2, atol=0.05)
    bo, res = stats.ols_fit(X, stats.predict(b, X) + rng.normal(0, 0.1, len(X)))
    assert np.allclose(bo, b, atol=0.01) and abs(res.mean()) < 1e-6


# ------------------------------------------------------------------ simulador de agentes
def test_spread_conserves_mass(P):
    W = np.random.default_rng(0).random((4, P.S))
    for diffuse in (0.0, 0.1, 0.3):
        out = agents.spread(W, P.site_nbr, diffuse)
        assert np.allclose(out.sum(1), W.sum(1))
    assert agents.spread(W, P.site_nbr, 0.0) is W
    one = np.zeros((1, P.S))
    c = int(np.where((P.site_nbr >= 0).all(1))[0][0])                         # un sitio interior
    one[0, c] = 1.0
    out = agents.spread(one, P.site_nbr, 0.3)[0]
    assert out[c] == pytest.approx(0.7) and np.allclose(out[P.site_nbr[c]], 0.3 / 8)


def test_simulation_accounting(fc, P):
    r = fc.run
    assert r.site_deaths.min() >= 0 and np.allclose(r.site_deaths, np.round(r.site_deaths))
    assert r.country_type.sum() == pytest.approx(r.site_deaths.sum())          # por tipo y por sitio, lo mismo
    assert np.allclose(fc.deaths("country").sum(1), r.site_deaths.sum(1))
    assert np.all(r.site_civ <= r.site_deaths + 1e-6)
    active = r.dyad_events > 0
    assert np.all(r.dyad_events[active] >= 1) and r.dyad_events.max() <= fc.model.n_cap
    assert r.births.sum((1, 2)).max() <= agents.BIRTH_SLOTS
    assert fc.deaths("cell").shape == (24, P.C, 12)


def test_same_seed_same_worlds(P, fc):
    th = fc.displacement_params
    again = forecast.run(P, T0, 12, members=24, seed=5, disp_params=th)
    assert np.array_equal(again.run.site_deaths, fc.run.site_deaths)
    other = forecast.run(P, T0, 12, members=24, seed=6, disp_params=th)
    assert not np.array_equal(other.run.site_deaths, fc.run.site_deaths)


def test_forecast_never_reads_the_future(world, P, fc):
    """El pronóstico desde T0 no cambia si se borra o se altera todo lo ocurrido desde T0 (prueba de fuga)."""
    th = fc.displacement_params
    cut = forecast.run(P.truncated(T0), T0, 12, members=24, seed=5, disp_params=th)
    assert np.array_equal(cut.run.site_deaths, fc.run.site_deaths)
    w2 = {k: v.copy() for k, v in world.items()}
    late = w2["ev_t0"] >= T0
    w2["ev_best"][late] *= 50.0                                               # otro futuro
    w2["ev_cell"][late] = 0
    w2["ref_stock"][..., -2:] *= 9.0
    alt = forecast.run(pn.from_arrays(w2), T0, 12, members=24, seed=5, disp_params=th)
    assert np.array_equal(alt.run.site_deaths, fc.run.site_deaths)
    for lv in ("country", "cell"):
        a = baseline.forecast(P, baseline.fit(P, lv, T0), T0, 6, 20, 1)
        b = baseline.forecast(pn.from_arrays(w2), baseline.fit(pn.from_arrays(w2), lv, T0), T0, 6, 20, 1)
        assert np.array_equal(a, b)
        assert np.array_equal(baseline.last12(P, lv, T0, 6), baseline.last12(pn.from_arrays(w2), lv, T0, 6))


def test_leak_test_catches_a_leaky_model(world, P):
    """Mutante: un modelo ajustado un mes más tarde de lo debido SÍ ve el mes T0, y la prueba de fuga lo nota."""
    w2 = {k: v.copy() for k, v in world.items()}
    w2["ev_best"][w2["ev_t0"] >= T0] *= 50.0
    P2 = pn.from_arrays(w2)
    honest = [agents.fit(X, T0, foot={"decay": 0.9, "diffuse": 0.1, "floor": 0.02, "mix": 0.5}) for X in (P, P2)]
    leaky = [agents.fit(X, T0 + 1, foot={"decay": 0.9, "diffuse": 0.1, "floor": 0.02, "mix": 0.5}) for X in (P, P2)]
    assert np.array_equal(honest[0].pools, honest[1].pools)
    assert not np.array_equal(leaky[0].pools, leaky[1].pools)


def test_quiet_dyads_die_and_no_births_when_switched_off(P):
    M = agents.fit(P, T0, foot={"decay": 0.9, "diffuse": 0.1, "floor": 0.02, "mix": 0.5})
    r = agents.simulate(P, M, 6, members=8, seed=1, births=False)
    assert r.births.sum() == 0 and np.all(r.dyad_events[:, M.D0:] == 0)
    dead = agents.AgentModel(**{**M.__dict__, "b_act": np.r_[-40.0, np.zeros(len(agents.FEATURES))]})
    r0 = agents.simulate(P, dead, 6, members=8, seed=1, births=False)
    assert r0.site_deaths.sum() == 0                                           # nadie se activa: no hay muertes


def test_more_persistence_means_more_violence(P):
    """Relación metamórfica: subir la constante de activación no puede bajar la violencia (mismos sorteos)."""
    M = agents.fit(P, T0, foot={"decay": 0.9, "diffuse": 0.1, "floor": 0.02, "mix": 0.5})
    hot = agents.AgentModel(**{**M.__dict__, "b_act": M.b_act + np.r_[1.5, np.zeros(len(agents.FEATURES))]})
    a = agents.simulate(P, M, 12, members=32, seed=2, births=False).dyad_events
    b = agents.simulate(P, hot, 12, members=32, seed=2, births=False).dyad_events
    assert (b > 0).sum() > (a > 0).sum()


def test_footprint_parameters_come_from_the_grid(P):
    M = agents.fit(P, T0)
    f = M.foot
    assert f["fitted"] and all(f[k] in agents.FOOT_GRID[k] for k in agents.FOOT_GRID) and np.isfinite(f["loglik"])
    assert np.allclose(M.prior.sum(1), 1.0) and M.W0.shape == (M.D0, P.S)


def test_interstate_dyad_is_a_dyad_of_two_governments(P):
    F = agents.history_features(P, T0)["F"]
    j = agents.FEATURES.index("interstate")
    both_gov = (P.actor_kind[P.dyad_a] == 0) & (P.actor_kind[P.dyad_b] == 0)
    assert both_gov.sum() == 1 and np.array_equal(F[:, 0, j] == 1, both_gov)
    civ = P.actor_kind[P.dyad_b] == 2                                          # los civiles no «deciden»
    assert np.all(F[civ][:, :, agents.FEATURES.index("side_b")] == 0)


# ------------------------------------------------------------------ desplazamiento
def test_people_are_conserved_and_flee_only_from_violence(P):
    th = displacement.unpack(displacement.START)
    G = displacement.geometry(P)
    pop = P.pop_at(T0, "cell")
    st = displacement.State(G, pop[None].copy(), np.zeros((1, P.K)), np.zeros((1, P.K)), np.zeros((1, P.C)),
                            np.zeros((1, P.K, 12)))
    total = st.home.sum()
    fl = st.step(th, np.zeros((1, P.C)), pop, P.pop_at(T0))
    assert fl["new_ref"].sum() == 0 and fl["new_idp"].sum() == 0               # sin muertes no huye nadie
    d = np.zeros((1, P.C))
    d[0, 5] = 500.0
    for _ in range(6):
        fl = st.step(th, d, pop, P.pop_at(T0))
        assert st.home.sum() + st.idp.sum() + st.ref.sum() == pytest.approx(total, rel=1e-9)
    assert st.idp.sum() + st.ref.sum() > 0 and st.home.min() >= 0
    near = [5] + [int(c) for c in P.nbr[5] if c >= 0]
    far = np.setdiff1d(np.arange(P.C), near)
    assert np.all(st.home[0, far] >= pop[far] - 1e-6)                           # lejos de la violencia nadie se va


def test_more_deaths_more_displaced(P, fc):
    th = fc.displacement_params
    base = displacement.simulate(P, th, T0, fc.run.site_deaths)
    more = displacement.simulate(P, th, T0, fc.run.site_deaths * 5)
    tot = lambda d: (d["new_ref"] + d["new_idp"]).sum()  # noqa: E731
    assert tot(more) > tot(base) > 0
    assert base["ref"].shape == (24, P.K, 12) and base["ref"].min() >= 0


def test_displacement_fit_improves_on_its_start_and_ignores_the_future(P, world):
    th = displacement.fit(P, T0, maxiter=60)
    assert th["fitted"] and th["loss"] <= th["loss_start"]
    assert all(m in displacement.MULT_GRID for m in th["mult"])
    w2 = {k: v.copy() for k, v in world.items()}
    w2["ref_stock"][..., P.years >= P.ym(T0)[0]] *= 7.0
    assert displacement.fit(pn.from_arrays(w2), T0, maxiter=60)["loss"] == pytest.approx(th["loss"])


# ------------------------------------------------------------------ vara y W13
def test_hurdle_beats_zero_on_a_persistent_world(P):
    from world4.validate import metrics
    m = baseline.fit(P, "country", T0)
    ens = baseline.forecast(P, m, T0, 12, 60, 0)
    y = P.deaths("country")[:, T0:T0 + 12]
    crps = lambda e: metrics.crps_ensemble(np.log1p(e).reshape(e.shape[0], -1), np.log1p(y).ravel()).mean()  # noqa: E731
    assert crps(ens) < crps(baseline.zero(P, "country", 12))
    assert baseline.last12(P, "cell", T0, 5).shape == (12, P.C, 5)
    assert set(m["models"]) <= set(baseline.STEPS["country"])
    pa = baseline.p_any(P, m, T0, 3)
    assert pa.shape == (P.K, 3) and np.all((pa >= 0) & (pa <= 1))


def test_w13_runs_and_summarises(P, monkeypatch):
    from world4.validate import l2 as v
    monkeypatch.setitem(pn._PANELS, "synthetic", P)
    parts = [v.one_origin(("synthetic", t0, 16, 7, {})) for t0 in (84, 90)]
    s = v.summarise(P, parts)
    c = s["levels"]["country"]["1"]["crps_log"]
    assert set(c["mean"]) == {"agents", "hurdle", "last12", "zero"} and c["n"] == 2 * P.K
    assert c["mean"]["agents"] < c["mean"]["zero"]
    assert s["levels"]["country"]["36"]["crps_log"]["n"] > 0                    # 90 + 36 <= 132: hay verdad
    ver = v.verdict(s)
    assert "country/1" in ver and isinstance(ver["country/1"]["forecast_grade"], bool)
    lo, hi = v._cluster_ci(np.r_[np.full(50, -1.0), np.full(50, -0.5)], np.repeat(np.arange(10), 10))
    assert lo <= -0.75 <= hi < 0


def test_bridge_to_the_structural_event(fc, P):
    y = P.ym(T0)[0]
    m = forecast.conflict_members(fc, y)
    assert m.shape == (24, P.K) and set(np.unique(m)) <= {0.0, 1.0}
    assert np.all(np.isnan(forecast.conflict_members(fc, y + 3)))               # fuera del horizonte
    sb = fc.run.country_type[:, :, :, 0].sum(2)
    assert np.array_equal(m, (sb >= forecast.CONFLICT_DEATHS).astype(float))


# ------------------------------------------------------------------ API, honestidad y datos
def test_api_result_passes_the_honesty_audit(P, monkeypatch):
    from world4 import api, honesty
    monkeypatch.setitem(pn._PANELS, "synthetic", P)
    r = api.l2_forecast("synthetic", 12, as_of=P.label(T0), members=16)
    js = json.loads(json.dumps(r))
    assert honesty.audit(js) == [] and honesty.exploratory_ok(js, True) == []
    assert honesty.offending_words(json.dumps(js)) == []
    assert js["first_month"] == P.label(T0) and js["as_of"] == P.label(T0 - 1)
    d = js["by_country"]["AAA"]["deaths"]
    assert len(d["p50"]) == 12 and all(a <= b <= c for a, b, c in zip(d["p10"], d["p50"], d["p90"]))
    assert len(js["hotspots"]["cells"]) == 20 and "displacement" in js
    mut = json.loads(json.dumps(r))
    mut.pop("unit")                                                             # mutante: sin unidad
    assert honesty.audit(mut)
    with pytest.raises(api.QueryError):
        api.l2_forecast("synthetic", 99)
    with pytest.raises(api.QueryError):
        api.l2_forecast("synthetic", 6, as_of="1989-02")


def test_no_forbidden_word_in_the_l2_code():
    from world4 import honesty
    assert not {k: v for k, v in honesty.code_scan().items() if "l2" in k or "conflict" in k}


def test_region_and_manifest_are_consistent():
    from world4.data import conflict, vintages
    r = regions.get("mideast")
    assert len(r.iso3) == len(r.gw) == len(r.m49) == 15 and len(set(r.gw)) == 15
    assert r.index("syr") == r.iso3.index("SYR")
    with pytest.raises(KeyError):
        regions.get("atlantis")
    man = conflict.manifest()["files"]
    assert set(man) == set(conflict.URLS)
    assert all(len(v["sha256"]) == 64 for v in man.values())
    assert all(u.startswith("https://") for u in conflict.URLS.values())
    assert conflict.UNHCR_COMMIT in conflict.URLS["unhcr_population.rda"]       # fijado a un commit, no a HEAD
    src = vintages.sources()
    for sid in ("ucdp_ged", "prio_grid", "unhcr_refugees"):
        assert sid in src and src[sid]["mark"] in ("[V]", "[S]", "[?]")
    assert src["prio_grid"]["redistributable"] is not True                      # sin licencia explícita


def test_atlas_checksum_ignores_line_endings_only(tmp_path):
    """El manifiesto del Atlas se hizo en Windows (CRLF): en Linux los mismos datos tienen LF."""
    import hashlib
    from world4.data import atlas
    crlf = b"a,b\r\n1,2\r\n"
    expected = hashlib.sha256(crlf).hexdigest()
    p = tmp_path / "x.csv"
    p.write_bytes(b"a,b\n1,2\n")
    assert atlas.matches(p, expected)
    p.write_bytes(b"a,b\n1,3\n")
    assert not atlas.matches(p, expected)                                       # otro valor: no pasa
    q = tmp_path / "x.gz"
    q.write_bytes(b"a,b\n1,2\n")
    assert not atlas.matches(q, expected)                                       # un binario se compara byte a byte


# ------------------------------------------------------------------ con el panel real, si está instalado
@pytest.fixture(scope="module")
def mideast():
    from world4.data import conflict
    if not conflict.available("mideast"):
        pytest.skip("datos del L2 sin instalar: python world4_cli.py --install-l2")
    return pn.load("mideast")


def test_mideast_panel_matches_ucdp_totals(mideast):
    from world4.data import conflict
    assert conflict.verify_raw()["ok"]
    P = mideast
    assert P.K == 15 and P.label(0) == "1989-01" and P.label(P.last_final_month) == "2025-12"
    syr = P.deaths("country")[P.iso3.index("SYR")]
    assert syr[P.month(2014, 1):P.month(2015, 1)].sum() == pytest.approx(78709, abs=1)   # GED 26.1, Siria 2014
    assert P.deaths("country").sum() == pytest.approx(P.ev["best"].sum())
    assert (P.ev["cell"] < 0).mean() < 0.05                                     # casi todo está localizado
    assert abs(P.pop_at(P.month(2020), "cell").sum() / P.pop_at(P.month(2020)).sum() - 1) < 0.02


def test_mideast_forecast_never_reads_the_future(mideast):
    P = mideast
    t0 = P.month(2020, 1)
    foot = {"decay": 0.8, "diffuse": 0.1, "floor": 0.02, "mix": 0.5}
    a = forecast.run(P, t0, 3, members=8, seed=1, foot=foot, displacement=False)
    b = forecast.run(P.truncated(t0), t0, 3, members=8, seed=1, foot=foot, displacement=False)
    assert np.array_equal(a.run.site_deaths, b.run.site_deaths)
