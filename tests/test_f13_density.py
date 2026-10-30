"""F13: factores de los eventos (``world4.conflict.factors``) y sistema de densidad (``world4.conflict.density``)."""
import json

import numpy as np
import pandas as pd
import pytest

from world4.conflict.density import Prep
from world4.conflict.factors import NB


def test_ppml_matches_total_and_recovers_dispersion():
    rng = np.random.default_rng(0)
    n = 6000
    X = rng.standard_normal((n, 2))
    off = rng.normal(0, 1, n)
    mu = np.exp(-0.5 + 0.8 * X[:, 0] - 0.4 * X[:, 1] + off)
    alpha = np.exp(0.7 - 0.2 * np.log(mu))                      # dispersión que depende de la media
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu)).astype(float)
    m = NB(l2=0.0).fit(X, y, off)
    assert m.mu(X, off).sum() == pytest.approx(y.sum(), rel=1e-6)          # PPML cuadra el total
    assert m.b[1:] == pytest.approx([0.8, -0.4], abs=0.08)
    assert m.disp == pytest.approx([0.7, -0.2], abs=0.15)
    p = m.p_any(X, off)
    assert np.all((p >= 0) & (p <= 1))
    rps = m.rps(X, y, off)
    assert np.all((rps >= 0) & (rps <= 4))


def test_prep_round_trip_and_missing_flags():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0, 4.0, np.nan], "b": [2.0, 2.0, 2.0, 2.0, 2.0]})
    p = Prep().fit(df)
    Z = p.transform(df)
    assert "a__na" in Z.columns and "b" not in Z.columns                   # constante: fuera
    q = Prep.from_dict(json.loads(json.dumps(p.to_dict())))
    assert np.allclose(q.transform(df).to_numpy(), Z.to_numpy())


@pytest.fixture(scope="module")
def panel():
    from world4.data import atlas, conflict_grid
    if not (atlas.verify()["ok"] and conflict_grid.available()):
        pytest.skip("needs the Atlas and the conflict-grid data")
    from world4.conflict import country_panel as cp
    return cp, cp.build()


def test_panel_totals_match_ged_and_targets_shift(panel):
    cp, df = panel
    from world4.data import conflict_grid as cdata
    d = cdata.load()
    years = 1989 + d["ev_month"] // 12
    for y in (1995, 2014, 2025):
        assert df[df.year == y].ev_all.sum() == (years == y).sum()
    t = cp.with_targets(df, 2)
    row = t[(t.iso3 == "COL") & (t.year == 2010)].iloc[0]
    assert row.y == df[(df.iso3 == "COL") & (df.year == 2012)].ev_all.iloc[0]
    assert set(cp.ALL) <= set(df.columns) and df.atlas_frozen.max() == 2025 - 2019


def test_density_system_predicts_saves_and_downscales(panel):
    from world4.conflict import density as D
    s = D.DensitySystem(horizons=(1, 2)).fit(train_until=2015)
    pr = s.predict(2015, 1)
    assert len(pr) == len(panel[1][panel[1].year == 2015])
    assert np.all((pr.p_any >= 0) & (pr.p_any <= 1)) and np.all(pr.p10 <= pr.p90)
    s2 = D.DensitySystem.from_dict(json.loads(json.dumps(s.to_dict())))
    assert np.allclose(s2.predict(2015, 2).events_expected, s.predict(2015, 2).events_expected)
    cells = D.cell_density(pr)
    tot = cells.groupby("iso3").events_expected.sum()
    exp = pr.set_index("iso3").events_expected
    common = tot.index.intersection(exp.index)
    assert np.allclose(tot[common], exp[common], rtol=1e-6)
