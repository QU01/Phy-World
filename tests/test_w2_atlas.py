"""W2: el L1 portado es el Atlas 9e96f6b bit a bit — y los mutantes que debe detectar."""
import json
import shutil

import pytest

from world4.campaign import GOLDEN, full_validation_diff
from world4.validate import atlas_parity as ap


@pytest.fixture(scope="module")
def port(atlas_ready):
    from world4.core.l1_atlas import calibrate, model, validate
    return model, calibrate, validate


@pytest.fixture(scope="module")
def golden():
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def test_golden_comes_from_the_pinned_commit(golden):
    assert golden["atlas_commit"] == "9e96f6b9f470ca990213cd977762da6f440b283b"
    assert len(golden["arrays"]) >= 80


def test_probe_bit_exact(port, golden):
    res = ap.compare(golden, ap.fingerprints(ap.probe(*port)))
    assert not res["failed"] and not res["missing"], res
    if res["mode"] != "sha256":
        pytest.xfail(f"otra plataforma ({res['mode']}): la huella exacta sólo vale en {golden['platform']}")


def test_mutant_alpha_breaks_parity(port, golden, monkeypatch):
    model = port[0]
    monkeypatch.setattr(model, "ALPHA", model.ALPHA * (1 + 1e-9))
    res = ap.compare(golden, ap.fingerprints(ap.probe(*port)))
    assert res["failed"], "un cambio de 1e-9 en ALPHA no rompió la paridad: la sonda no mide nada"


def test_atlas_files_verified(atlas_ready):
    assert atlas_ready["ok"]


def test_tampered_file_detected(atlas_ready, tmp_path):
    from world4.data import atlas
    base = atlas.root()
    rel = "results/calibration.json"
    (tmp_path / "results").mkdir()
    for r in atlas.manifest()["files"]:
        dst = tmp_path / r
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / r, dst)
    assert atlas.verify(tmp_path)["ok"]
    p = tmp_path / rel
    p.write_bytes(p.read_bytes().replace(b"0.0", b"0.1", 1))
    v = atlas.verify(tmp_path)
    assert not v["ok"] and v["mismatched"] == [rel]


def test_full_validation_matches_original_and_stored(atlas_ready):
    full = full_validation_diff()
    if full is None:
        pytest.skip("validación completa sin correr: python scripts/run_atlas_validation.py runs/validation_port.json")
    if full["same_platform"] is not None:
        assert full["same_platform"]["n_diff"] == 0, full
    assert full["stored"]["max_rel_non_auc"] <= 1e-12, full
