"""F0.5: estimador estructural (origen y población honestos, recalibración, estimador, W8/W9a)."""
from datetime import date

import numpy as np
import pytest

from world4.structural import engine, recal

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------------------ origen y vintages
def test_origin_is_three_years_behind_and_capped():
    assert engine.origin_for(date(2005, 7, 1)) == 2002
    assert engine.origin_for(date(2005, 6, 30)) == 2001
    assert engine.origin_for(date(2026, 10, 11)) == 2019


def test_honest_world_keeps_history_and_extrapolates(atlas_ready):
    from world4.core.l1_atlas.calibrate import W
    Wh = engine.honest_world(W, 40)
    assert np.array_equal(Wh.pop[:, :41], W.pop[:, :41], equal_nan=True)
    assert not np.allclose(Wh.pop[:, 50], W.pop[:, 50], equal_nan=True)     # el futuro ya no es el observado
    assert W.pop is not Wh.pop                                             # el mundo global no se toca


# ------------------------------------------------------------------ recalibración
def test_isotonic_is_monotone_and_bounded():
    rng = np.random.default_rng(0)
    p = rng.random(2000)
    y = (rng.random(2000) < p ** 2).astype(float)
    cal = recal.EventCalibrator(p, y)
    assert cal.kind == "isotonic"
    out = cal(np.linspace(0, 1, 101))
    assert np.all(np.diff(out) >= -1e-12) and out.min() > 0 and out.max() < 1
    assert abs(cal(0.5) - 0.25) < 0.08


def test_calibrator_ladder():
    assert recal.EventCalibrator([0.1] * 10, [0] * 10).kind == "identity"
    rng = np.random.default_rng(1)
    p = rng.random(60)
    assert recal.EventCalibrator(p, (rng.random(60) < p).astype(float)).kind == "platt"


def test_numeric_recalibration_fixes_underdispersion():
    rng = np.random.default_rng(2)
    members = rng.normal(0, 1, 32)
    obs = rng.normal(0, 3, 3000)                       # el ensemble es 3 veces más estrecho que la verdad
    pits = recal.ens_cdf(members, obs)
    raw_cov = np.mean((pits >= 0.1) & (pits <= 0.9))
    cal = recal.NumericCalibrator(pits[:1500])
    new = cal.C(pits[1500:])
    assert raw_cov < 0.6 and 0.75 <= np.mean((new >= 0.1) & (new <= 0.9)) <= 0.85
    assert recal.ks_distance(new) <= 0.05
    assert recal.ks_distance(pits[1500:]) > 0.2          # el crudo está lejos


def test_ks_detects_a_skewed_pit():
    rng = np.random.default_rng(3)
    assert recal.ks_uniform_p(rng.random(2000)) > 0.05
    assert recal.ks_uniform_p(rng.random(2000) ** 2) < 1e-6


def test_bins_sum_to_one():
    cal = recal.NumericCalibrator([])
    p = recal.bin_probabilities(cal, np.linspace(-1, 1, 32), [-0.5, 0, 0.5])
    assert len(p) == 4 and abs(sum(p) - 1) < 1e-12 and all(x >= 0 for x in p)


# ------------------------------------------------------------------ estimador (necesita el almacén)
@pytest.fixture(scope="module")
def store_ready(atlas_ready):
    from world4.structural import store
    if not store.path().is_file():
        pytest.skip("almacén estructural sin construir: python world4_cli.py --build-store")
    return store


def _req(**kw):
    base = {"variable": "conflict", "region": "COL", "as_of": "2026-10-11", "closes_at": "2027-12-31",
            "kind": "binary"}
    return {**base, **kw}


def test_estimator_live_binary(store_ready):
    from world4.structural import estimator
    out = estimator.estimate(_req())
    assert out["status"] == "ok" and out["origin"] == 2019 and out["live"]
    assert 0 < out["p"] < 1 and out["base_rate_data"]["p_window"] > 0
    assert any("2019" in c for c in out["caveats"])


@pytest.mark.parametrize("req,status", [
    (_req(variable="gdp"), "unbound"), (_req(region="XXX"), "unbound"),
    (_req(closes_at="2031-12-31"), "out_of_domain"), (_req(as_of="2023-01-01"), "asof_violation"),
    (_req(variable="gdp_pc_growth"), "unbound"),                 # numérica pedida como binaria
])
def test_estimator_refuses_with_a_name(store_ready, req, status):
    from world4.structural import estimator
    assert estimator.estimate(req)["status"] == status


def test_estimator_numeric(store_ready):
    from world4.structural import estimator
    out = estimator.estimate(_req(variable="gdp_pc_growth", region="MEX", kind="numeric", edges=[0, 2, 4]))
    assert out["status"] == "ok" and set(out["distribution"]) == {"b0", "b1", "b2", "b3"}
    assert abs(sum(out["distribution"].values()) - 1) < 1e-9
    assert out["quantiles"]["p10"] < out["quantiles"]["p50"] < out["quantiles"]["p90"]


def test_historical_estimate_uses_historical_origin(store_ready):
    from world4.structural import estimator
    out = estimator.estimate(_req(as_of="2005-07-01", closes_at="2008-12-31"))
    assert out["status"] == "ok" and out["origin"] == 2002 and not out["live"] and out["k"] == [4, 6]


def test_recalibrator_never_sees_unpublished_outcomes(store_ready):
    from world4.structural import questions
    early = questions.event_calibrator("conflict", 6, date(1995, 7, 1))
    late = questions.event_calibrator("conflict", 6, date(2010, 7, 1))
    assert early.n < late.n
    R = questions.records()["events"]
    # todos los pares de entrenamiento de 1995 acaban, como tarde, en 1994 (UCDP, lag 1)
    m = (R[:, 0] == 0) & (R[:, 4] == 6) & (R[:, 5] <= 1994)
    assert early.n == int(m.sum())


def test_w9a_and_w8_runs_are_scored(store_ready):
    from world4 import campaign
    if not (campaign.RUNS / "w9a.json").is_file():
        pytest.skip("sin runs/w9a.json")
    scores = {s.criterion.cid: s for s in campaign.build().score_all()}
    assert scores["W9a"].verdict in ("RED", "AMBER", "GREEN") and scores["W9a"].measured
    assert scores["W8"].verdict in ("RED", "GREEN") and scores["W8"].measured
