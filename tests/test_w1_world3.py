"""W1: el L0 reproduce World3-03 (salida de Vensim) — y los mutantes que debe detectar."""
import numpy as np
import pandas as pd
import pytest

from world4.core import l0
from world4.campaign import VENSIM, W1_TOL

VARS = ("f", "io", "nr", "ppolx", "pop")
# Candado de regresión, mucho más fino que el listón de W1: Vensim imprime 6 cifras (~1e-6) y
# el L0 llega a 7e-6. El 1 % de W1 no detectaría la conmutación >= en 1940 (0,4 %); éste sí.
GUARD = 1e-4


def _err(run):
    ref = pd.read_csv(VENSIM, comment="#")
    assert len(ref) == len(run.time) == 401
    assert np.allclose(ref["time"], run.time)
    return {k: float(np.max(np.abs(run[k][:, 0] - ref[k]) / np.abs(ref[k]))) for k in VARS}


@pytest.fixture(scope="module")
def base():
    return l0.run()


def test_w1_within_one_percent(base):
    e = _err(base)
    assert max(e.values()) <= W1_TOL, e


def test_regression_guard(base):
    e = _err(base)
    assert max(e.values()) <= GUARD, e


def test_mutant_lmhs_switch_caught_by_guard_not_by_w1():
    e = _err(l0.run(constants={"lmhs_year": 1939.5}))   # t > 1939.5 == t >= 1940
    assert max(e.values()) > GUARD
    assert max(e.values()) < W1_TOL          # documentado: el 1 % de W1 es demasiado laxo para esto


def test_mutant_table_caught_by_w1():
    # la tabla de tamaño de familia de 1972 en lugar de la de 2003 (lo que pydynamo comenta)
    e = _err(l0.run(tables={"sfsn": (0, 800, 200, [1.25, 1, 0.9, 0.8, 0.75])}))
    assert max(e.values()) > W1_TOL, e


def test_mutant_constant_caught_by_w1():
    e = _err(l0.run(constants={"nri": 1.1e12}))
    assert max(e.values()) > W1_TOL, e


def test_regions_do_not_mix(base):
    r = l0.run(n=3, constants={"nri": np.array([1e12, 2e12, 5e11])})
    for k in ("pop", "io", "ppolx", "nr", "hwi"):
        assert np.array_equal(r[k][:, 0], base[k][:, 0]), k
    assert not np.allclose(r["io"][:, 1], r["io"][:, 0])


@pytest.mark.parametrize("sc", sorted(l0.SCENARIOS))
def test_scenarios_finite(sc):
    r = l0.run(scenario=sc)
    for k in l0.RECORD:
        assert np.all(np.isfinite(r[k])), (sc, k)


def test_table_grid_checked():
    with pytest.raises(ValueError):
        l0.run(tables={"sfsn": (0, 800, 200, [1.25, 0.94, 0.715, 0.59])})


def test_stocks_nonnegative(base):
    for k in l0.STOCKS:
        assert np.all(base[k] >= 0), k
