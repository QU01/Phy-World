"""API pública, CLI (códigos de salida, ASCII) y honestidad (W11 parcial)."""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from world4 import api

ROOT = Path(__file__).resolve().parents[1]
CLI = [sys.executable, str(ROOT / "world4_cli.py")]


def cli(*args):
    return subprocess.run(CLI + list(args), capture_output=True, text=True, encoding="utf-8", cwd=ROOT)


# ------------------------------------------------------------------ API
def test_catalog_has_units_everywhere():
    rows = api.catalog()
    assert rows and all(r["unit"] for r in rows)
    assert {r["fidelity"] for r in rows} == {"L0", "L1"}


@pytest.mark.parametrize("args", [("gdp_pc", "MEX", 1960, 1965), ("gdp_pc", "MEX", 2010, 2025),
                                  ("gdp_pc", "MEX", 2000, 2000), ("nope", "MEX", 2000, 2005),
                                  ("gmst", "MEX", 2000, 2005), ("gdp_pc", "XXX", 2000, 2005)])
def test_query_refuses_out_of_domain(args, atlas_ready):
    with pytest.raises(api.QueryError):
        api.query(*args, members=4)


def test_query_declares_its_limits(atlas_ready):
    d = api.query("gdp_pc", "MEX", 2005, 2010, members=8)
    assert d.p10 <= d.p50 <= d.p90 and d.unit
    assert d.calibrated is False
    assert not any(c.startswith("D1") for c in d.caveats)                  # F1 cierra D1
    assert any(c.startswith("D7") for c in d.caveats) and any(c.startswith("F1 population") for c in d.caveats)
    assert d.provenance["params_cutoff"] == 2000 and d.provenance["population"] == "endogenous (F1)"
    assert not any(c.startswith("D3") for c in d.caveats) and any(c.startswith("F2") for c in d.caveats)


def test_query_world_rent_share(atlas_ready):
    d = api.query("resource_rent_share_world", "WLD", 2000, 2005, members=8)
    assert 0 < d.p10 <= d.p50 <= d.p90 < 1 and d.provenance["psi"] == "world markets (F2)"
    with pytest.raises(api.QueryError):
        api.query("resource_rent_share_world", "MEX", 2000, 2005, members=4)


def test_query_population_is_endogenous(atlas_ready):
    from world4.core.l1_atlas import calibrate
    d = api.query("population", "NGA", 2000, 2010, members=8)
    obs = calibrate.W.pop[calibrate.W.iso3.index("NGA"), 60]
    assert d.p10 < d.p90 and d.unit.startswith("million")
    assert abs(d.p50 / obs - 1) < 0.15


# ------------------------------------------------------------------ CLI
def test_info_is_ascii_without_colour():
    r = cli("--info", "--no-color")
    assert r.returncode == 0, r.stderr
    assert all(ord(c) < 128 for c in r.stdout), [c for c in r.stdout if ord(c) >= 128][:5]
    assert "emdat" in r.stdout and "no-redistribution" in r.stdout
    assert "W12" in r.stdout


def test_l0_json():
    r = cli("--l0", "--json")
    assert r.returncode == 0, r.stderr
    d = json.loads(r.stdout)
    assert d["years"][0] == 1900 and abs(d["pop"][0] - 1.6e9) < 1


def test_spec_error_exit_2():
    assert cli("--query", "gdp_pc", "--no-color").returncode == 2
    assert cli("--l0", "--scenario", "9", "--no-color").returncode == 2


def test_strict_exit_4_on_degradation(atlas_ready):
    r = cli("--query", "gdp_pc", "--region", "MEX", "--as-of", "2005", "--target", "2010",
            "--members", "4", "--strict", "--no-color")
    assert r.returncode == 4, (r.returncode, r.stdout[-400:], r.stderr[-400:])


def test_campaign_json(atlas_ready):
    r = cli("--campaign", "--json")
    assert r.returncode == 0, r.stderr
    v = {s["cid"]: s["verdict"] for s in json.loads(r.stdout)["scores"]}
    assert v["W1"] == "GREEN"
    assert v["W2"] in ("GREEN", "AMBER")
    assert v["W12"] in ("GREEN", "AMBER")
    assert v["W9b"] == "BLOCKED"
    assert v["W11"] in ("GREEN", "RED")                                       # F9: medido en vivo
    assert v["W10"] in ("GREEN", "RED", "BLOCKED")                           # F6: medido si hay runs/w10.json
    assert v["W7"] in ("AMBER", "RED", "BLOCKED")                             # F3-F5: cuentas medidas
    assert v["W6"] in ("GREEN", "AMBER", "RED")                              # F6: W6 completo
    assert all(v[c] in ("GREEN", "AMBER", "RED", "BLOCKED") for c in ("W3", "W4", "W5"))   # F1: medidos si hay runs/
    assert v["W8"] != "GREEN" or v["W9a"] != "GREEN"           # F0.5: medidos (o BLOQUEADOS sin almacén)


# ------------------------------------------------------------------ honestidad
# 'validated' sólo puede aparecer como nombre del horizonte MEDIDO fuera de muestra, como la
# propia regla que lo prohíbe, o en el código del Atlas portado (que no se toca, W2).
ALLOWED = (re.compile(r"validated_horizon_years"), re.compile(r"VALIDATED_HORIZON_YEARS"),
           re.compile(r"'validated' forbidden"), re.compile(r"forbids 'validated'"),
           re.compile(r"validated <= "))
WORD = re.compile(r"\bvalidated\b", re.I)


def offending(text: str) -> list:
    out = []
    for ln in text.splitlines():
        stripped = ln
        for a in ALLOWED:
            stripped = a.sub("", stripped)
        if WORD.search(stripped):
            out.append(ln.strip())
    return out


def test_detector_catches_a_planted_word():
    assert offending('print("model validated against data")')
    assert not offending('ui.kv(x, f"validated <= {n} y")')


def test_no_validated_claims_in_world4_code():
    # honesty.py contiene la palabra porque ES la regla (W11, F9)
    files = [p for p in (ROOT / "world4").rglob("*.py") if "l1_atlas" not in p.parts and p.name != "honesty.py"]
    files += [ROOT / "world4_cli.py"]
    bad = {str(p.relative_to(ROOT)): offending(p.read_text(encoding="utf-8")) for p in files}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, bad
