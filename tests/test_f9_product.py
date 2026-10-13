"""F9: producto independiente (hogar, CLI, MCP, informe, honestidad W11, dashboard)."""
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, str(ROOT / "world4_cli.py")]


def cli(*args, env=None, stdin=None):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, encoding="utf-8", cwd=ROOT,
                          env={**os.environ, **(env or {})}, input=stdin)


# ------------------------------------------------------------------ hogar y paquete
def test_home_is_the_checkout_and_w4_home_overrides(tmp_path):
    from world4 import home
    assert home.SOURCE_CHECKOUT and home.home() == ROOT
    r = subprocess.run([sys.executable, "-c", "from world4 import home; print(home.home(), home.runs())"],
                       capture_output=True, text=True, cwd=ROOT, env={**os.environ, "W4_HOME": str(tmp_path)})
    assert str(tmp_path) in r.stdout


def test_the_package_carries_its_posterior_and_references_but_no_third_party_data():
    from world4 import home
    assert (home.PACKAGE_MODELS / "posterior.npz").is_file()
    assert (home.REFERENCE / "atlas_9e96f6b.json").is_file()
    assert not (home.REFERENCE / "world3_03_vensim.csv").exists()            # W12: dato de terceros, fuera
    py = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'world4 = "world4.cli:main"' in py and "models/*.npz" in py


# ------------------------------------------------------------------ honestidad (W11)
def test_honesty_rules_and_their_mutants():
    from world4 import honesty
    good = {"exploratory": True, "world": {"gmst": {"unit": "degC", "p10": [1, 2], "p50": [1.5, 2.5], "p90": [2, 3]}}}
    assert honesty.audit(good) == [] and honesty.exploratory_ok(good, True) == []
    no_unit = {"exploratory": True, "x": {"p10": 1.0, "p50": 2.0, "p90": 3.0}}
    assert any("without unit" in v for v in honesty.audit(no_unit))
    no_band = {"exploratory": True, "unit": "degC", "series": [1.0, 2.0, 3.0]}
    assert any("without band" in v for v in honesty.audit(no_band))
    assert honesty.audit({**no_band, "band": "none: one deterministic run"}) == []
    assert any("p50 without" in v for v in honesty.audit({"unit": "K", "p50": 1.0}))
    named = {"units": {"a": "m"}, "a": 1.0, "b": 2.0}
    assert honesty.audit(named) == ["$.b: number without unit"]               # un dict de unidades sólo cubre lo que nombra
    assert honesty.exploratory_ok({}, True) and honesty.exploratory_ok({"exploratory": False}, True)
    assert honesty.offending_words('x = "the model is validated"') and not honesty.offending_words("validated_horizon_years")
    assert honesty.code_scan() == {}


# ------------------------------------------------------------------ MCP
def _rpc(*msgs):
    from world4.mcp import server
    out = io.StringIO()
    server.serve(io.StringIO("\n".join(json.dumps(m) for m in msgs) + "\n"), out)
    return [json.loads(x) for x in out.getvalue().splitlines()]


def test_mcp_protocol_without_data():
    r = _rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
             {"jsonrpc": "2.0", "method": "notifications/initialized"},
             {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
             {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "w4_levers", "arguments": {}}},
             {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "w4_levers", "arguments": {"x": 1}}},
             {"jsonrpc": "2.0", "id": 5, "method": "nope"})
    assert [m["id"] for m in r] == [1, 2, 3, 4, 5]                           # la notificación no responde
    assert r[0]["result"]["protocolVersion"] == "2025-06-18" and "tools" in r[0]["result"]["capabilities"]
    names = {t["name"] for t in r[1]["result"]["tools"]}
    assert {"w4_query", "w4_evaluate_policy", "w4_l0r_bands", "w4_catalog"} <= names
    assert all(t["inputSchema"]["type"] == "object" for t in r[1]["result"]["tools"])
    assert r[2]["result"]["isError"] is False and r[2]["result"]["structuredContent"]["exploratory"] is True
    assert r[3]["result"]["isError"] is True
    assert r[4]["error"]["code"] == -32601


# ------------------------------------------------------------------ CLI
def test_help_numbers_and_fidelity():
    r = cli("--help-numbers", "--json")
    assert r.returncode == 0 and "--theta" in json.loads(r.stdout)["modes"]
    assert cli("--query", "gdp_pc", "--fidelity", "L0", "--no-color").returncode == 2
    assert cli("--theta", "carbon_price=10", "--fidelity", "L1", "--no-color").returncode == 2


def test_bare_call_without_a_terminal_is_info():
    r = cli("--no-color", stdin="")
    assert r.returncode == 0 and "wizard needs a terminal" in r.stdout


def test_wizard_asks_only_for_its_mode(monkeypatch, capsys):
    from world4 import cli as wcli
    answers = iter(["5"])
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    a = wcli.build_parser().parse_args(["--no-color"])
    assert wcli.run_wizard(a) == 0
    assert "What do you want to compute" in capsys.readouterr().out


@pytest.fixture(scope="module")
def ready(atlas_ready):
    from world4.calib import posterior
    from world4.data import climate, energy, food, wpp
    if not (food.available() and climate.available() and energy.available() and wpp.available()):
        pytest.skip("faltan datos de F3-F5")
    if not posterior.PATH.is_file():
        pytest.skip("sin posterior")


def test_infeasible_policy_gets_walls_and_exit_3(ready):
    r = cli("--theta", "carbon_price=150", "--sow", "40", "--no-color")
    assert r.returncode == 3, r.stdout[-500:]
    assert "WALLS" in r.stdout and "land_cap" in r.stdout and "THE LAST BLOCKER" in r.stdout


def test_report_is_self_contained_with_units_and_bands(ready, tmp_path):
    from world4 import report
    d = report.collect(None, members=30, campaign=False, log=lambda *a: None)
    page = report.build(d)
    assert "<script" not in page and "<link" not in page and "http://" not in page
    assert page.count("<svg") == 6 and page.count("<polygon") >= 6 and page.count('class="u">[') == 6
    assert "Exploratory." in page and "SHA-256" in page


def test_w11_is_measured_live(ready):
    from world4 import honesty
    r = honesty.measure(log=lambda *a: None, with_report=False)
    assert r["ok"], r


def test_dashboard_geometry_is_pinned(ready):
    from world4 import dashboard
    from world4.data import atlas
    p = dashboard.geo_path()
    if not p.is_file():
        pytest.skip("geometry not installed: python world4_cli.py --dashboard")
    man = json.loads(dashboard.MANIFEST.read_text(encoding="utf-8"))
    assert atlas.sha256(p) == man["sha256"]
    assert "globe.gl@2.46.2" in dashboard.GLOBE_GL
