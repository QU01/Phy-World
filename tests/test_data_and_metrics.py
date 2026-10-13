"""W12 (registro de fuentes, vintages, manifiesto) y la métrica AUC con empates."""
from datetime import date
from pathlib import Path

import numpy as np

from world4.data import atlas, vintages
from world4.validate import metrics

ROOT = Path(__file__).resolve().parents[1]


def test_every_source_registered_completely():
    for sid, s in vintages.sources().items():
        assert s["mark"] in ("[V]", "[S]", "[?]"), sid
        assert isinstance(s["lag_years"], int) and s["lag_years"] >= 0, sid
        assert s.get("used_in"), sid
        if s["mark"] == "[?]":
            assert s.get("redistributable") is None, f"{sid}: licencia desconocida no puede ser redistribuible"


def test_registry_is_ascii():
    assert all(ord(c) < 128 for c in vintages.SOURCES_FILE.read_text(encoding="utf-8"))


def test_emdat_marked_not_redistributable():
    assert vintages.sources()["emdat"]["redistributable"] is False


def test_no_third_party_data_inside_package():
    vendored = [p for p in (ROOT / "world4").rglob("*") if p.suffix in (".csv", ".gz", ".rda", ".parquet")]
    assert not vendored, vendored


def test_known_time_and_projection_rule():
    assert vintages.known_time("pwt", 2010) == date(2013, 7, 1)
    assert vintages.known_time("wpp", 2050) == date(2024, 7, 11)       # proyección: fecha de la edición
    assert vintages.known_time("wpp", 2000) == date(2001, 7, 1)
    assert not vintages.usable("pwt", 2019, date(2020, 1, 1))
    assert vintages.latest_year("wdi", date(2010, 12, 31)) == 2009


def test_leakage_report_quantifies_the_atlas_lead():
    rep = vintages.leakage_report()
    assert rep["pwt"]["max_years_ahead"] == 3
    assert rep["emdat"]["max_years_ahead"] == 0


def test_manifest_shape():
    man = atlas.manifest()
    assert man["commit"] == atlas.ATLAS_COMMIT
    assert "data/processed/panel_full.csv" in man["files"]
    assert man["files"]["data/processed/panel_full.csv"]["generated"] is True
    assert all(len(m["sha256"]) == 64 for m in man["files"].values())


def test_auc_tie_invariant_but_atlas_auc_is_not(atlas_ready):
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.15).astype(float)
    s = np.clip(y + (rng.random(400) < 0.1), 0.02, 0.98)              # puntuación casi binaria
    from world4.core.l1_atlas.calibrate import auc as atlas_auc        # el del Atlas, sin tocar
    ours, theirs = set(), set()
    for k in range(20):
        p = rng.permutation(400)
        ours.add(round(metrics.auc(y[p], s[p]), 12))
        theirs.add(round(float(atlas_auc(y[p], s[p])), 12))
    assert len(ours) == 1
    assert len(theirs) > 1, "el AUC del Atlas no dependió del orden: revisar el diagnóstico del informe de F0"


def test_auc_matches_definition():
    y = np.array([0, 0, 1, 1]); s = np.array([0.1, 0.4, 0.35, 0.8])
    assert metrics.auc(y, s) == 0.75
