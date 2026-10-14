"""F10: el shim de Studio (``world4.studio_compat``) y los datos del explorador (``world4.explorer``).

Lo que Studio da por hecho de este paquete, probado aquí para que no dependa de Studio:

* el contrato del módulo de física (Fidelity, FAMILIES, VAR_NAMES, cajas, ``denormalize``, ``evaluate``);
* ``run_studio`` escribe los entregables del explorador en el ``outdir`` y en NINGÚN otro sitio;
* los tres JSON pasan la auditoría de honestidad W11 (``honesty.audit``) y son coherentes entre sí
  (índices de flujos dentro de ``countries``, aristas del grafo entre nodos que existen);
* el drenaje bilateral reproduce el TOTAL por país que registra el L1 (la regla repartida, no inventada).
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from world4 import explorer, honesty, studio_compat as sc
from world4.scenarios import levers


def _data_ok():
    from world4.data import atlas, climate, energy, food, wpp
    try:
        atlas.require()
    except Exception:
        return False
    return all(m.available() for m in (climate, energy, food, wpp))


needs_data = pytest.mark.skipif(not _data_ok(), reason="World4 data not installed (world4_cli.py --install-*)")


def test_contrato_del_modulo_de_fisica():
    assert [f.name for f in sc.Fidelity] == ["L0", "L1", "L2"]
    assert sc.FAMILIES == ("scenario", "design")
    for fam in sc.FAMILIES:
        assert sc.VAR_NAMES[fam] == list(levers.NAMES) and sc.NDIM[fam] == 7
        assert sc.BOUNDS_LO[fam] == levers.BOX[:, 0].tolist() and sc.BOUNDS_HI[fam] == levers.BOX[:, 1].tolist()
    assert np.allclose(sc.denormalize(np.zeros(7)), levers.BOX[:, 0])
    assert np.allclose(sc.denormalize(np.ones(7) * 2), levers.BOX[:, 1])          # se recorta a la caja
    assert set(sc.UNITS) == set(levers.NAMES)


@needs_data
def test_evaluate_registro_plano_y_signo():
    rec = sc.evaluate(np.zeros(7), sc.Fidelity.L0, "scenario", members=40)
    assert rec["source"] == "world4(L0)" and rec["exploratory"] is True
    assert all(isinstance(v, (int, float, str, bool)) for v in rec.values())
    rec1 = sc.evaluate([150, 0, 0, 0, 0, 0, 0], "L1", "scenario", members=40)
    assert "L1 has no levers" in rec1["source"]                                   # declarado, no inventado
    assert rec1["peak_gmst_p90"] < rec["peak_gmst_p90"]
    with pytest.raises(ValueError):
        sc.evaluate([500, 0, 0, 0, 0, 0, 0], "L0", "scenario")


@needs_data
def test_run_studio_L0_solo_escribe_en_outdir(tmp_path):
    out = tmp_path / "run"
    before = {p for p in tmp_path.rglob("*")}
    r = sc.run_studio({"family": "scenario", "fidelity": "L0", "carbon_price": 50, "members": 30}, out,
                      log=lambda *a: None)
    after = {p for p in tmp_path.rglob("*")} - before
    assert all(out in p.parents or p == out for p in after)
    assert set(r["deliverables"]) >= {"policy", "world", "geo", "graph", "report"}
    assert r["record"]["source"] == "world4(L0)" and r["fidelity"] == "L0"
    world = json.loads((out / "world.json").read_text(encoding="utf-8"))
    assert world["l1"] is None and world["l0"]["countries"] and world["theta"]["carbon_price"] == 50
    for name in ("world.json", "graph.json", "policy.json"):
        assert honesty.audit(json.loads((out / name).read_text(encoding="utf-8"))) == [], name


@needs_data
def test_run_studio_L2_se_degrada_a_L1_con_aviso(tmp_path):
    lines = []
    r = sc.run_studio({"family": "scenario", "fidelity": "L2", "members": 30, "l1_members": 8}, tmp_path / "r",
                      log=lines.append)
    assert r["fidelity"] == "L1" and any("L2 is not separate" in ln for ln in lines)


@needs_data
def test_payload_L1_coherente_y_drenaje_repartido():
    w = explorer.world_payload(members=30, R=8, log=lambda *a: None)
    l1 = w["l1"]
    n, T = len(l1["countries"]), len(l1["years"])
    assert l1["origin"] == 2019 and T == 11 and n >= 170
    assert honesty.audit(json.loads(json.dumps(w))) == []
    for name, s in l1["series"].items():
        grid = s.get("p50", s.get("mean"))
        assert len(grid) == n and all(len(row) == T for row in grid), name
        assert s["unit"], name
    share = np.array(l1["channels"]["share"], dtype=float)
    assert share.shape == (n, T, 6) and np.nanmax(share) < 1.0
    for kind in ("trade", "drain"):
        for rows in l1["flows"][kind]:
            assert all(0 <= a < n and 0 <= b < n and a != b and v > 0 for a, b, v in rows)
    part = l1["flows"]["partners"]
    assert len(part["drain_out"]) == n and all(len(x) == T for x in part["drain_out"])
    # el drenaje que sale de cada país suma (casi) su pérdida registrada x PIB: la regla del L1, repartida
    gdp = np.array(l1["series"]["gdp_pc"]["p50"], float) * np.array(l1["series"]["population"]["p50"], float)
    lost = np.array(l1["series"]["drain"]["p50"], float) * gdp
    i = int(np.nanargmax(lost[:, 5]))
    top3 = sum(v for _, v in part["drain_out"][i][5])
    assert 0 < top3 <= lost[i, 5] * 1.05


@needs_data
def test_grafo_y_lazos():
    w = explorer.world_payload(members=30, with_l1=False, log=lambda *a: None)
    lp = explorer.loops_payload()
    g = explorer.graph_payload(w, loops=lp)
    ids = {nd["id"] for nd in g["nodes"]}
    assert len(ids) == len(g["nodes"])
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])
    assert {nd["id"] for nd in g["nodes"] if nd["status"] == "skipped"} >= {"agents", "flows"}
    assert set(lp["stocks"]) >= {ln["source"] for ln in lp["links"]} | {ln["target"] for ln in lp["links"]}
    assert honesty.audit(json.loads(json.dumps(g))) == []


def test_geo_con_centroides():
    try:
        g = explorer.geo_payload()
    except Exception as exc:                                 # sin red ni geometría instalada
        pytest.skip(f"geometry not installed: {exc}")
    assert len(g["features"]) > 150
    for f in g["features"]:
        lon, lat = f["properties"]["centroid"]
        assert -180 <= lon <= 180 and -90 <= lat <= 90
    fr = next(f for f in g["features"] if f["properties"]["name"] == "France")
    assert -5 < fr["properties"]["centroid"][0] < 10 and 40 < fr["properties"]["centroid"][1] < 52
