"""F11: predicción de conflicto por celda (``world4.conflict``). Casi todo corre sobre un mundo sintético pequeño
(sin datos instalados); la última prueba usa UCDP GED + PRIO-GRID si están instalados."""
import json

import numpy as np
import pytest

from world4 import honesty
from world4.conflict import benchmarks as BM
from world4.conflict import features as F
from world4.conflict import forecast as FC
from world4.conflict import metrics as M
from world4.conflict import service
from world4.conflict.grid import Events, Grid
from world4.conflict.models import Hawkes, Logistic, Size
from world4.data import conflict_grid as cdata


def synthetic(n_rows=24, n_cols=24, months=180, seed=0):
    """Un bloque de celdas terrestres en dos países y un proceso de Hawkes simulado con un foco de conflicto."""
    rng = np.random.default_rng(seed)
    rows, cols = np.meshgrid(np.arange(200, 200 + n_rows), np.arange(400, 400 + n_cols), indexing="ij")
    row, col = rows.ravel(), cols.ravel()
    gid = (row - 1) * 720 + col
    o = np.argsort(gid)
    gid, row, col = gid[o], row[o], col[o]
    lon, lat = cdata.rowcol_to_lonlat(row, col)
    n = len(gid)
    pg = {f"pg_{k}": rng.random(n) for k in cdata.PRIOGRID_VARS}
    pg["pg_gwno"] = np.where(col < 400 + n_cols // 2, 500.0, 501.0)
    for k in ("petroleum_s", "diamsec_s", "goldplacer_s", "gem_s"):
        pg[f"pg_{k}"] = np.where(rng.random(n) < 0.1, 1.0, np.nan)
    hot = (np.abs(row - 210) <= 2) & (np.abs(col - 405) <= 2)
    base = np.where(hot, 0.08, 0.002)
    E = np.zeros(n)
    ev_m, ev_c, ev_t = [], [], []
    for t in range(months):
        lam = base + 0.5 * 0.5 * E
        y = rng.poisson(lam)
        E = 0.5 * E + y
        for c in np.nonzero(y)[0]:
            for _ in range(y[c]):
                ev_m.append(t), ev_c.append(c), ev_t.append(rng.integers(0, 3))
    k = len(ev_m)
    return {"gid": gid, "row": row, "col": col, "lon": lon, "lat": lat, **pg,
            "ev_month": np.array(ev_m, np.int32), "ev_cell": np.array(ev_c, np.int32),
            "ev_type": np.array(ev_t, np.int8), "ev_deaths": rng.poisson(3.0, k).astype(np.float32),
            "ev_country": np.where(col[ev_c] < 400 + n_cols // 2, 500, 501).astype(np.int32),
            "ev_candidate": np.zeros(k, bool), "ev_where_prec": np.ones(k, np.int8),
            "first_year": np.int32(1989), "n_months": np.int32(months), "last_final_month": np.int32(months - 1),
            "versions": np.array(["synthetic", "synthetic"])}


@pytest.fixture(scope="module")
def world():
    d = synthetic()
    g = Grid(d)
    return d, g, Events(g, d)


def test_metrics_against_brute_force():
    rng = np.random.default_rng(1)
    p, y = rng.random(400), rng.random(400) < 0.2
    pairs = np.mean([(a > b) + 0.5 * (a == b) for a in p[y] for b in p[~y]])
    assert M.auroc(p, y) == pytest.approx(pairs)
    # AUPR de un clasificador perfecto es 1; con ranking al revés es ~ la tasa base
    assert M.average_precision(y.astype(float), y) == pytest.approx(1.0)
    X, obs = rng.poisson(2.0, (50, 30)).astype(float), rng.poisson(2.0, 50).astype(float)
    brute = np.abs(X - obs[:, None]).mean(1) - 0.5 * np.abs(X[:, :, None] - X[:, None, :]).mean((1, 2))
    assert np.allclose(M.crps_samples(X, obs), brute)
    assert np.allclose(M.crps_samples(np.zeros((5, 1)), np.arange(5.0)), np.arange(5.0))   # CRPS del cero = y


def test_logistic_recovers_weighted_coefficients():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((40000, 2))
    y = rng.random(40000) < 1 / (1 + np.exp(-(-3 + 1.5 * X[:, 0] - 0.7 * X[:, 1])))
    keep = y | (rng.random(40000) < 0.25)                        # submuestreo de ceros con peso 1 / 0,25
    w = np.where(y, 1.0, 4.0)[keep]
    m = Logistic(l2=1e-6).fit(X[keep], y[keep].astype(float), w)
    b = m.coef[1:] / m.sd
    assert b == pytest.approx([1.5, -0.7], abs=0.12)
    assert m.predict(X).mean() == pytest.approx(y.mean(), rel=0.1)


def test_hawkes_em_recovers_branching():
    rng = np.random.default_rng(3)
    n, T = 3000, 60
    beta = F.BETA[1]
    E, rows_Z, rows_y = np.zeros(n), [], []
    for t in range(T):
        lam = 0.02 + 0.6 * beta * E
        y = rng.poisson(lam)
        rows_Z.append(beta * E), rows_y.append(y)
        E = (1 - beta) * E + y
    Z = np.concatenate(rows_Z)[:, None]
    y = np.concatenate(rows_y).astype(float)
    B = np.ones((len(y), 1))
    hk = Hawkes().fit(B, Z, y, np.ones(len(y)))
    assert hk.alpha[0] == pytest.approx(0.6, abs=0.08)
    assert hk.background(B[:1])[0] == pytest.approx(0.02, rel=0.15)
    assert hk.theta > 20                                        # datos de Poisson: casi sin sobredispersión


def test_grid_rings_and_country(world):
    _, g, _ = world
    deg = np.asarray(g.ring1.sum(1)).ravel()
    assert deg.max() == 8 and deg.min() == 3                     # esquinas del bloque: 3 vecinas
    assert np.asarray(g.ring2.sum(1)).ravel().max() == 16
    assert (g.ring1 != g.ring1.T).nnz == 0
    assert g.n_countries == 2 and g.cells_per_country.sum() == g.n


def test_features_never_look_ahead(world):
    d, g, ev = world
    t = 100
    a = F.Sweep(ev, g)
    a.run_to(t)
    d2 = dict(d)
    late = d["ev_month"] > t
    d2["ev_cell"] = np.where(late, (d["ev_cell"] + 37) % g.n, d["ev_cell"])   # el futuro, cambiado
    b = F.Sweep(Events(g, d2), g)
    b.run_to(t)
    assert np.array_equal(a.features(), b.features())
    assert np.array_equal(a.hawkes(), b.hawkes())


def test_end_to_end_beats_persistence_and_payload_is_honest(world):
    d, g, ev = world
    T = 150
    rows = F.build(ev, g, until=T, snapshots=[T], p_near=1.0, p_far=1.0)
    assert set(np.unique(rows["weight"])) == {1.0}
    model = FC.fit(ev, g, rows, T, horizon=6)
    model2 = FC.Model.from_dict(json.loads(json.dumps(model.to_dict())))
    assert np.allclose(model2.logistic["any"][0].coef, model.logistic["any"][0].coef)
    P = FC.predict(model, g, rows["snapshots"][T], samples=60)
    for o in FC.OUTCOMES:
        assert P["p"][o].shape == (6, g.n) and np.all((P["p"][o] >= 0) & (P["p"][o] <= 1))
        assert np.all(P["p_window"][o] >= P["p"][o].max(0) - 0.05)
    # el foco simulado tiene más riesgo que el resto, y el modelo no es peor que la persistencia en Brier
    hot = (np.abs(g.row - 210) <= 2) & (np.abs(g.col - 405) <= 2)
    assert P["p_window"]["any"][hot].mean() > 5 * P["p_window"]["any"][~hot].mean()
    y = ev.window("count", "any", T + 1, T + 2)[0] > 0
    pers = BM.probabilities(ev, g, T, "any")["persistence"]
    assert M.brier(P["p"]["any"][0], y) <= M.brier(pers, y)
    assert all(v >= 0 for row in model.branching().values() for v in row.values())
    P["decomposition"] = FC.decompose(model, g, rows["snapshots"][T])
    P["_grid"], P["_ev"], P["_model"], P["_snap"] = g, ev, model, rows["snapshots"][T]
    out = json.loads(json.dumps(service.payload(P, min_p=0.0)))
    assert out["schema"] == service.SCHEMA and out["summary"]["cells_returned"] == g.n
    assert len(out["cells"]["p_month_any"][0]) == 6
    shares = sum(np.array(out["cells"][f"driver_{k}"]) for k in ("background", "own", "neighbours", "country"))
    assert np.allclose(shares, 1.0, atol=0.01)
    assert honesty.audit(out) == [] and honesty.exploratory_ok(out, False) == []
    assert honesty.offending_words(json.dumps(out)) == []


def test_benchmarks_neighbours12_matches_definition(world):
    _, g, ev = world
    T = 120
    b = BM.probabilities(ev, g, T, "sb")
    hits = (ev.window("count", "sb", T - 11, T + 1) > 0)
    i = int(np.argmax(b["neighbors12"]))
    nb = g.ring1[i].indices
    assert b["neighbors12"][i] == pytest.approx((hits[:, i].sum() + hits[:, nb].sum()) / 108)
    ds = BM.death_samples(ev, g, T, samples=10)
    assert ds["neighbors12"].shape == (g.n, 108) and ds["cell12"].shape == (g.n, 12)


def test_installed_data_matches_manifest():
    if not cdata.available():
        pytest.skip("conflict-grid data not installed: python world4_cli.py --install-conflict-grid")
    v = cdata.verify_raw()
    assert v["ok"], v
    d = cdata.load()
    assert len(d["gid"]) == 64818 and d["gid"].min() == 49182 and d["gid"].max() == 249344
    assert int(d["n_months"]) > int(d["last_final_month"])         # hay meses candidatos después del GED final
    lon, lat = cdata.rowcol_to_lonlat(*cdata.gid_to_rowcol(np.array([1, 720 * 360])))
    assert lon.tolist() == [-179.75, 179.75] and lat.tolist() == [-89.75, 89.75]
