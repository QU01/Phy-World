"""Servidor de la interfaz (``world4.server``): rutas, errores con remedio y JSON sin NaN."""
import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from world4 import server as S


def test_status_and_l0_routes():
    code, st = S.route("GET", "/api/status", {}, None)
    assert code == 200 and set(st["capabilities"]) >= {"conflict", "social", "l0", "l1", "l0r", "policy"}
    assert st["capabilities"]["l0"]["ok"] is True
    code, r = S.route("GET", "/api/l0", {"scenario": ["2"]}, None)
    assert code == 200 and r["years"][0] == 1900 and r["years"][-1] == 2100
    assert len(r["series"]["pop"]) == len(r["years"]) and r["exploratory"] is True
    assert set(r["units"]) >= set(r["series"])


def test_bad_requests_and_unknown_route():
    assert S.route("GET", "/api/nope", {}, None)[0] == 404
    with pytest.raises(ValueError):
        S.route("GET", "/api/l1", {}, None)
    with pytest.raises(ValueError):
        S.route("GET", "/api/l0", {"scenario": ["99"]}, None)


def test_clean_replaces_non_finite():
    out = S._clean({"a": [1.0, float("nan"), float("inf")], "b": {"c": -math.inf}})
    assert out == {"a": [1.0, None, None], "b": {"c": None}}
    json.dumps(out, allow_nan=False)


def test_http_round_trip_with_remedy_on_missing_data():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), S.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/api/levers") as r:
            levers = json.loads(r.read())
        assert [x["name"] for x in levers["levers"]][0] == "carbon_price"
        from world4.data import food
        if not food.available():
            with pytest.raises(urllib.error.HTTPError) as e:
                urllib.request.urlopen(base + "/api/l0r")
            assert e.value.code == 503
            body = json.loads(e.value.read())
            assert body["remedy"] == "python world4_cli.py --install-food"
    finally:
        httpd.shutdown()


def test_query_path_matches_query(atlas_ready):
    from world4 import api
    from world4.data import energy, wpp
    if not (wpp.available() and energy.available()):
        pytest.skip("L1 needs WPP and energy data")
    a = api.query("gdp_pc", "MEX", 2005, 2008, members=8, seed=3)
    b = api.query_path("gdp_pc", "MEX", 2005, 5, members=8, seed=3)
    k = b["years"].index(2008)
    assert (b["p10"][k], b["p50"][k], b["p90"][k]) == (a.p10, a.p50, a.p90)
    assert b["exploratory"] is False and "band" in b
