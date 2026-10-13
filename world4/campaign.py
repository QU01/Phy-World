"""Campaña de aceptación W1-W12 de World4 (hoja de ruta §6), sobre ``phycore.vvuq.Campaign``.

En F0 se MIDEN W1, W2 y W12; F0.5 añade W8 y W9a; F1 añade W3, W5 y la población de W4; F2-F4 añaden
psi, energía, CO2 y GMST a W4, sus relaciones a W6 y sus cuentas (energía, masa de C) a W7; F5 añade
cereales y superficie cultivada a W4, el calor y el rendimiento a W6 y el suelo y el N a W7; F7 juzga W4
con el hindcast calibrado y W8 con la recalibración de selección anidada. El resto
queda BLOQUEADO y dice qué fase lo desbloquea: un criterio sin dato detrás es BLOQUEADO, nunca verde.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phycore.vvuq.campaign import AMBER, BLOCKED, GREEN, RED, Campaign, Criterion, Score

from world4 import home

ROOT = home.home()
# la salida de Vensim es un dato de terceros (WorldDynamics.jl, MIT): vive en tests/ del repo, no en el paquete (W12)
VENSIM = home.home() / "tests" / "reference" / "world3_03_vensim.csv"
GOLDEN = home.REFERENCE / "atlas_9e96f6b.json"
FULL_VALIDATION = ROOT / "runs" / "validation_port.json"
W1_TOL = 0.01

CRITERIA = [
    Criterion("W1", "Parity L0 <-> World3-03 (P1)",
              "the 5 variables with Vensim output (f, io, nr, ppolx, pop) within 1 % over 1900-2100",
              "Vensim World3-03 output, scenario 1, via WorldDynamics.jl test/world3_03.jl", mark="[V]"),
    Criterion("W2", "Parity L1 <-> Atlas 9e96f6b (P3)",
              "identical SHA-256 fingerprints of the probe (history, 3 forecasts from 1990, baselines, loss); "
              "full validation = the Atlas validation.json to <= 1e-12 except AUCs of tied scores",
              "Atlas del Sistema-Mundo, commit 9e96f6b", mark="[V]"),
    Criterion("W3", "Rolling origin at 1/5/10 years", "GDP-pc CRPS not worse than the Atlas by > 2 %; resource "
              "rents at 10 years not worse (F2); beat the best reference in >= 4 of 6 variables at 10 years; mutant "
              "'observed population' must score better (leak test)",
              "the Atlas re-run on this machine with the same protocol (runs/w3.json)", mark="[V]"),
    Criterion("W4", "Free hindcast of world aggregates 1990-2023", ">= 80 % of years inside the p10-p90 band "
              "(world resource-rent share psi, F2: >= 70 % from 1990)",
              "world series (WPP, PWT, WDI rents, EI, GCP, FAOSTAT, GMST)", mark="[S]"),
    Criterion("W5", "Zero recalibration", "adding a sector moves no other outside its band", "W3/W4"),
    Criterion("W6", "Behaviour modes", "declared metamorphic relations", "World3-03 + structure"),
    Criterion("W7", "Conservation and SFC accounts", "balances < 1e-9; employment <= working-age population",
              "accounting"),
    Criterion("W8", "Band calibration", "p10-p90 coverage 75-85 %; PIT uniform to KS distance D <= 0.05 "
              "(changed in F0.5 from 'KS p > 0.05': the recalibrator is estimated from finite samples, so a "
              "one-sample p-value is invalid at n ~ 5000; the change does not flip the verdict)",
              "rolling-origin validation"),
    Criterion("W9a", "Structural estimator, historical", "Brier < empirical base rate; ViEWS/PITF reported",
              "questions generated from the panel"),
    Criterion("W9b", "Live contribution to Prime Radiant d1",
              "Brier(d1 with) < Brier(d1 without), CI excludes 0, N >= 30", "Prime Radiant ledger"),
    Criterion("W10", "Performance", "L0 < 10 ms per run; L1 64 runs 2019->2100 < 60 s", "reference machine"),
    Criterion("W11", "Honesty", "no number without unit and band; 'validated' forbidden; exploratory flag correct",
              "honesty tests"),
    Criterion("W12", "Data: licence, SHA-256 and known_time", "every source registered with licence and mark; every "
              "data file with SHA-256; nothing non-redistributable inside the repo",
              "sources.json + atlas/wpp/energy/climate/food manifests",
              mark="[V]"),
]

BLOCKS = {
    "W9b": ("F0.5 + months of real resolutions",),
}


def _w1(crit) -> Score:
    import pandas as pd

    from world4.core import l0
    if not VENSIM.is_file():
        return Score(crit, BLOCKED, cause="the Vensim reference (third-party, MIT) is not in this install: it lives in "
                                          "the repo's tests/reference/world3_03_vensim.csv",
                     waits_on=("a checkout of the repo, or W4_HOME pointing at one",))
    ref = pd.read_csv(VENSIM, comment="#")
    r = l0.run()
    errs = {k: float(np.max(np.abs(r[k][:, 0] - ref[k].to_numpy()) / np.abs(ref[k].to_numpy())))
            for k in ("f", "io", "nr", "ppolx", "pop")}
    worst = max(errs, key=errs.get)
    verdict = GREEN if errs[worst] <= W1_TOL else RED
    measured = f"P1: max error {errs[worst]:.2e} ({worst}); " + ", ".join(f"{k} {v:.1e}" for k, v in errs.items())
    detail = {"p1": errs}
    cause = ""
    from world4.data import climate as cdata, food as fdata
    if verdict == GREEN and cdata.available() and fdata.available():   # F6: candado P2 del L0 regional
        import warnings
        warnings.simplefilter("ignore")
        from world4.validate import p2
        r = p2.run(log=lambda *a: None)
        detail["p2"] = r
        v = r["variables"]
        measured += ("; P2 (L0 regional free 1992-2019 vs the L1 aggregate): world inside every year for "
                     + ", ".join(k for k in ("pop", "gdp", "energy", "cereal", "cropland") if v[k]["world_inside_all_years"])
                     + "; region-year cells inside " + ", ".join(f"{k} {v[k]['cells_inside']:.0%}"
                                                              for k in ("pop", "gdp", "energy", "cereal", "cropland"))
                     + f", GMST {v['gmst']['inside_share']:.0%}")
        if not r["ok"]:                       # P2 es otro candado (hoja de ruta §3.2): se muestra, no decide W1
            bad = [k for k in ("pop", "gdp", "energy", "cereal", "cropland")
                   if not (v[k]["world_inside_all_years"] and v[k]["cells_inside"] >= p2.MIN_CELLS)]
            measured += (f"; P2 FAILS its declared threshold (80 % of region-year cells) in {', '.join(bad)}: crude "
                         "birth and death rates without age structure drift 7-9 % in some regions (W1's threshold is P1)")
    return Score(crit, verdict, measured=measured, detail=detail, cause=cause)


def _w2(crit) -> Score:
    from world4.data import atlas
    v = atlas.verify()
    if not v["ok"]:
        return Score(crit, BLOCKED, cause="Atlas data not installed or not matching the manifest",
                     waits_on=("python world4_cli.py --install-atlas",))
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l1_atlas import calibrate, model, validate
    from world4.validate import atlas_parity as ap
    cur = ap.fingerprints(ap.probe(model, calibrate, validate))
    res = ap.compare(json.loads(GOLDEN.read_text(encoding="utf-8")), cur)
    measured = f"probe: {res['n'] - len(res['failed']) - len(res['missing'])}/{res['n']} fingerprints equal ({res['mode']})"
    detail = {"probe": res}
    verdict = GREEN if (not res["failed"] and not res["missing"]) else RED
    if res["mode"] != "sha256" and verdict == GREEN:
        verdict = AMBER
        measured += "; other platform: statistics only"
    full = full_validation_diff()
    if full is None:
        measured += "; full validation not run (scripts/run_atlas_validation.py)"
        if verdict == GREEN:
            verdict = AMBER
    else:
        detail["full"] = full
        same = full["same_platform"]
        if same is not None:
            measured += (f"; full validation vs original on this platform: {same['n'] - same['n_diff']}/{same['n']} "
                         "values identical")
            if same["n_diff"]:
                verdict = RED
        else:
            measured += "; no original run for this platform (exact comparison skipped)"
            if verdict == GREEN:
                verdict = AMBER
        st = full["stored"]
        measured += (f"; vs the Atlas validation.json: max rel {st['max_rel_non_auc']:.1e} outside AUC "
                     f"(AUC max abs {st['max_abs_auc']:.2f}, not reproducible across machines)")
        if st["max_rel_non_auc"] > 1e-12:
            verdict = RED
    return Score(crit, verdict, measured=measured, detail=detail,
                 cause="" if verdict == GREEN else "see detail")


# Los AUC del Atlas dependen del orden de desempate de argsort siempre que la puntuación tenga
# empates: referencias casi binarias Y probabilidades de democracia (promedio de 32 regímenes
# binarios). No son reproducibles entre máquinas; ver docs/F0-informe.md §3 y validate/metrics.py.
ORIGINAL_RUN = home.REFERENCE / "validation_original_9e96f6b.json"


def _values(d: dict):
    for kind, hs in d.items():
        if kind.startswith(("meta", "_")):
            continue
        for h, methods in hs.items():
            for m, vals in methods.items():
                for k, v in vals.items():
                    yield f"{kind}/{h}/{m}/{k}", v


def full_validation_diff(path: Path = FULL_VALIDATION) -> dict | None:
    """Compara la validación completa del port con (a) la corrida del ORIGINAL en esta misma
    plataforma, valor a valor y exacta, y (b) el validation.json del Atlas, de otra máquina."""
    from world4.data import atlas
    from world4.validate.atlas_parity import platform_id
    if not path.is_file():
        return None
    port = dict(_values(json.loads(path.read_text(encoding="utf-8"))))
    same = None
    if ORIGINAL_RUN.is_file():
        orig = json.loads(ORIGINAL_RUN.read_text(encoding="utf-8"))
        if orig.get("_provenance", {}).get("platform") == platform_id():
            ov = dict(_values(orig))
            same = {"n": len(ov), "n_diff": sum(1 for k, v in ov.items() if port.get(k) != v)}
    stored = dict(_values(json.loads((atlas.root() / "results" / "validation.json").read_text(encoding="utf-8"))))
    worst, worst_key, auc_abs = 0.0, "", 0.0
    for k, a in stored.items():
        b = port[k]
        if a == b:
            continue
        if k.endswith("/auc"):
            auc_abs = max(auc_abs, abs(a - b))
            continue
        rel = abs(a - b) / max(abs(a), 1e-300)
        if rel > worst:
            worst, worst_key = rel, k
    return {"same_platform": same,
            "stored": {"n": len(stored), "max_rel_non_auc": worst, "worst": worst_key, "max_abs_auc": auc_abs}}


def _w12(crit) -> Score:
    from world4.data import atlas, vintages
    problems = []
    for sid, s in vintages.sources().items():
        for key in ("name", "mark", "lag_years"):
            if key not in s:
                problems.append(f"{sid}: missing {key}")
        if s.get("mark") not in ("[V]", "[S]", "[?]"):
            problems.append(f"{sid}: invalid mark {s.get('mark')}")
    man = atlas.manifest()
    if not all(len(m["sha256"]) == 64 for m in man["files"].values()):
        problems.append("Atlas manifest with an invalid hash")
    from world4.data import wpp
    wman = wpp.manifest()
    if not all(len(m["sha256"]) == 64 for m in wman["files"].values()):
        problems.append("WPP manifest with an invalid hash")
    from world4.data import energy as edata
    eman = edata.manifest()
    if not all(len(m["sha256"]) == 64 for m in eman["files"].values()):
        problems.append("energy manifest with an invalid hash")
    from world4.data import climate as cdata
    cman = cdata.manifest()
    if not all(len(m["sha256"]) == 64 for m in cman["files"].values()):
        problems.append("climate manifest with an invalid hash")
    from world4.data import food as fdata
    fman = fdata.manifest()
    if not all(len(m["sha256"]) == 64 for m in fman["files"].values()):
        problems.append("food manifest with an invalid hash")
    vendored = [p for p in (ROOT / "world4").rglob("*.csv")] + [p for p in (ROOT / "world4").rglob("*.csv.gz")]
    if vendored:
        problems.append(f"data inside the package: {[p.name for p in vendored]}")
    unknown = sum(1 for s in vintages.sources().values() if s["mark"] == "[?]")
    restricted = [sid for sid, s in vintages.sources().items() if s.get("redistributable") is False]
    if problems:
        return Score(crit, RED, measured="; ".join(problems))
    return Score(crit, AMBER if unknown else GREEN,
                 measured=(f"{len(vintages.sources())} sources registered, {unknown} with licence [?]; "
                           f"not redistributable: {', '.join(restricted) or 'none'}; "
                           f"{len(man['files'])} Atlas + {len(wman['files'])} WPP + {len(eman['files'])} energy + "
                           f"{len(cman['files'])} climate + {len(fman['files'])} food files with SHA-256; "
                           "0 third-party data in the package"),
                 cause=(f"{unknown} unverified licences: not blocking because World4 redistributes nothing, "
                        "but those data may not be used outside the local install") if unknown else "")


RUNS = ROOT / "runs"


def _w8(crit) -> Score:
    f = RUNS / "w8.json"
    if not f.is_file():
        return Score(crit, BLOCKED, cause="no W8 run", waits_on=("python world4_cli.py --calibrate",))
    d = json.loads(f.read_text(encoding="utf-8"))
    cells = [(v, k, r) for v, ks in d["variables"].items() for k, r in ks.items()]
    cov = [r["coverage80_recal"] for *_, r in cells]
    ks_d = [r["ks_d_recal"] for *_, r in cells]
    cov_ok = all(0.75 <= c <= 0.85 for c in cov)
    pit_ok = all(x <= 0.05 for x in ks_d)
    f7 = "ks_d_f05" in cells[0][2]
    measured = (f"{len(cells)} cells (variable x k2): recalibrated p10-p90 coverage {min(cov):.2f}-{max(cov):.2f} "
                f"(raw {min(r['coverage80_raw'] for *_, r in cells):.2f}-{max(r['coverage80_raw'] for *_, r in cells):.2f}); "
                f"KS distance of the recalibrated PIT {min(ks_d):.3f}-{max(ks_d):.3f} "
                f"(raw {min(r['ks_d_raw'] for *_, r in cells):.3f}-{max(r['ks_d_raw'] for *_, r in cells):.3f})")
    if f7:
        n_ok = sum(x <= 0.05 for x in ks_d)
        null = [r["ks_d_null_p95_year_clustered"] for *_, r in cells]
        beyond = [f"{v}/k2={k}" for v, k, r in cells if r["ks_d_recal"] > r["ks_d_null_p95_year_clustered"]]
        measured += (f"; F7 nested selection: {n_ok}/{len(cells)} cells with D <= 0.05 (F0.5 recalibrator "
                     f"{min(r['ks_d_f05'] for *_, r in cells):.3f}-{max(r['ks_d_f05'] for *_, r in cells):.3f}); "
                     f"informative: a perfectly calibrated forecaster with the measured within-year correlation "
                     f"gives D p95 {min(null):.3f}-{max(null):.3f}; beyond it: {', '.join(beyond) or 'none'}")
    if cov_ok and pit_ok:
        return Score(crit, GREEN, measured=measured, detail=d)
    cause = []
    if not cov_ok:
        cause.append("coverage outside 75-85 %")
    if not pit_ok:
        cause.append("PIT not uniform out of sample to D <= 0.05: the model's bias drifts between decades; F7's "
                     "recent-bias and moving-window recalibrators (chosen by nested selection) cut the distance "
                     "but not to the threshold in every cell" if f7 else
                     "PIT not uniform out of sample: the model's bias drifts between decades, so a "
                     "recalibrator fitted on earlier decades shifts later PITs (in-sample it is uniform)")
    return Score(crit, RED, measured=measured, cause="; ".join(cause), detail=d)


def _w9a(crit) -> Score:
    f = RUNS / "w9a.json"
    if not f.is_file():
        return Score(crit, BLOCKED, cause="no W9a run", waits_on=("python world4_cli.py --build-store",))
    d = json.loads(f.read_text(encoding="utf-8"))
    per = ", ".join(f"{e} {v['delta_vs_base_rate']:+.4f} [{v['delta_vs_base_rate_ci95'][0]:+.4f}, "
                    f"{v['delta_vs_base_rate_ci95'][1]:+.4f}]" for e, v in d["events"].items())
    ci = d["pooled_delta_vs_base_rate_ci95"]
    measured = (f"Brier(World4) - Brier(base rate), pooled {d['pooled_delta_vs_base_rate']:+.4f} "
                f"[{ci[0]:+.4f}, {ci[1]:+.4f}] over {sum(v['n'] for v in d['events'].values())} questions; {per}")
    if ci[1] < 0:
        return Score(crit, AMBER, measured=measured, detail=d,
                     cause="beats the base rate, but ViEWS/PITF are not reported yet (required by W9a)")
    worse = [e for e, v in d["events"].items() if v["delta_vs_base_rate_ci95"][0] > 0]
    return Score(crit, RED, measured=measured, detail=d,
                 cause=(f"pooled CI includes 0; worse than the base rate on {', '.join(worse) or 'none'} "
                        "(democracy: the honest origin is 3 years behind the last published regime); "
                        "ViEWS/PITF not reported"))


def _rolling() -> dict | None:
    for name in ("w3.json", "w3_f1.json"):
        f = RUNS / name
        if f.is_file():
            return json.loads(f.read_text(encoding="utf-8"))
    return None


VALIDATE = ("python world4_cli.py --validate",)


def _w3(crit) -> Score:
    d = _rolling()
    if d is None:
        return Score(crit, BLOCKED, cause="no rolling-origin run", waits_on=VALIDATE)
    v = d["w3"]
    r = v["gdp_crps_ratio"]
    mode = v.get("mode", "endogenous")
    label = {"f4": "F1-F4", "f3": "F1+F2+F3", "f2": "F1+F2", "endogenous": "F1"}.get(mode, mode)
    pop = d["modes"]["endogenous"]["pop"]["10"]["value"]
    pop_tr = d["modes"]["trend"]["pop"]["10"]["value"]
    measured = (f"model {label}: GDP-pc CRPS / Atlas {r['1']:.3f} / {r['5']:.3f} / {r['10']:.3f} at 1/5/10 y; beats the "
                f"best reference at 10 y in {v['n_beaten_10y']}/6 "
                f"({', '.join(k for k, b in v['beats_best_baseline_10y'].items() if b)}); log-population CRPS at 10 y "
                f"{pop:.4f} (F0.5 trend {pop_tr:.4f})")
    if "rr_rmse_ratio_10y" in v:
        measured += f"; resource-rent RMSE at 10 y / Atlas {v['rr_rmse_ratio_10y']:.3f}"
        if "psi" in d["modes"].get(mode, {}):
            ps, pa = d["modes"][mode]["psi"]["10"]["value"], d["modes"]["observed"]["psi"]["10"]["value"]
            measured += f"; world rent share psi CRPS at 10 y {ps:.4f} (Atlas, psi frozen: {pa:.4f})"
    if not v["ok_2pct"] or v["n_beaten_10y"] < 4 or not v.get("rr_ok", True):
        return Score(crit, RED, measured=measured, detail=v,
                     cause="GDP-pc CRPS or resource rents degraded > 2 %, or < 4/6 variables beat their reference")
    if not v["leak_detected"]:
        return Score(crit, AMBER, measured=measured, detail=v,
                     cause=("the declared mutant (observed population after the origin) does not score better on "
                            "GDP-pc CRPS, so this criterion cannot see that leak; the leak is guarded instead by an "
                            "invariance test (future pop/hc altered -> forecast bit-identical)"))
    return Score(crit, GREEN, measured=measured, detail=v)


def _w4(crit) -> Score:
    parts, red, detail = [], [], {}
    f = RUNS / "w4_pop_hindcast.json"
    if f.is_file():
        d = json.loads(f.read_text(encoding="utf-8"))
        detail["population"] = d
        parts.append(f"world population {d['origin']}->{d['end']} inside p10-p90 in {d['inside_share']:.0%} of years "
                     f"(p50 in {d['end']} {d['rel_err_p50_end']:+.1%}; F0.5 trend {d['rel_err_trend_end']:+.1%})")
        if not d["ok"]:
            red.append("population: the fertility equation fitted on <= 1990 misses the faster decline of the 1990s "
                       "(with observed TFR the overshoot disappears)")
    for origin in (1990, 2005):
        f = RUNS / f"w4_energy_hindcast_{origin}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            detail[f"energy_{origin}"] = d
            w = d["world_pe"]
            parts.append(f"world primary energy {origin}->{d['end']} inside in {w['inside_share']:.0%} of years "
                         f"(p50 in {d['end']} {w['rel_err_p50_end']:+.1%}); regional mix (6 regions x 5 sources) "
                         f"inside in {d['mix_inside_share']:.0%} of cells, MAE {d['mix_mae_p50']:.4f} vs persistence "
                         f"{d['mix_mae_persistence']:.4f}")
            if not d["ok"]:
                red.append(f"world primary energy from {origin} inside the band in < 80 % of years")
    for origin in (1980, 1990):
        f = RUNS / f"w4_psi_hindcast_{origin}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            detail[f"psi_{origin}"] = d
            parts.append(f"world rent share psi {origin}->{d['end']} inside in {d['inside_share']:.0%} of years "
                         f"(MAE p50 {d['mae_p50']:.4f} vs Atlas 10-year mean {d['mae_atlas']:.4f})")
    psi = detail.get("psi_1990")
    if psi is not None and psi["inside_share"] < 0.70:
        red.append("psi below its F2 threshold (70 % of years from 1990)")
    for origin in (1990, 2005):
        f = RUNS / f"w4_climate_hindcast_{origin}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            detail[f"climate_{origin}"] = d
            g, c = d["gmst"], d["co2_ffi"]
            parts.append(f"GMST {origin}->{d['end']} inside in {g['inside_share']:.0%} of years (MAE p50 {g['mae_p50']:.3f} "
                         f"degC; Atlas TCRE {g['mae_atlas_tcre']:.3f}; 10-year trend {g['mae_trend_10y']:.3f}); fossil CO2 "
                         f"inside in {c['inside_share']:.0%} (p50 in {d['end']} {c['rel_err_p50_end']:+.0%})")
            if origin == 1990:
                if g["inside_share"] < 0.80:
                    red.append("GMST from 1990 inside the band in < 80 % of years (F3's fossil energy overshoots from "
                               "1990, and more fossil means more aerosol cooling in the short run)")
                if c["inside_share"] < 0.80:
                    red.append("fossil CO2 from 1990 inside the band in < 80 % of years")
    for origin in (1990, 2005):
        f = RUNS / f"w4_food_hindcast_{origin}.json"
        if f.is_file():
            d = json.loads(f.read_text(encoding="utf-8"))
            detail[f"food_{origin}"] = d
            c, cl = d["cereal_prod"], d["cropland"]
            parts.append(f"cereals {origin}->{d['end']} inside in {c['inside_share']:.0%} of years (p50 in {d['end']} "
                         f"{c['rel_err_p50_end']:+.0%}); cropland inside in {cl['inside_share']:.0%} "
                         f"({cl['rel_err_p50_end']:+.0%}); {d['countries']} countries")
            if origin == 1990:
                if c["inside_share"] < 0.80:
                    red.append("cereals from 1990 inside the band in < 80 % of years")
                if cl["inside_share"] < 0.80:
                    red.append("cropland from 1990 inside the band in < 80 % of years (the 1961-1990 expansion is "
                               "extrapolated; the post-1990 land sparing is not in the data before the origin)")
    f = RUNS / "f4_fair_observed.json"
    if f.is_file():
        d = json.loads(f.read_text(encoding="utf-8"))
        parts.append(f"FaIR with observed emissions: GMST 1950-2023 inside in {d['inside_share']:.0%} (in sample)")
    if not parts:
        return Score(crit, BLOCKED, cause="no hindcast run", waits_on=VALIDATE)
    measured = "uncalibrated (F1-F6): " + "; ".join(parts)
    f = RUNS / "f7_l0_oos_2005.json"
    if f.is_file():
        d = json.loads(f.read_text(encoding="utf-8"))
        detail["f7_l0_oos_2005"] = d
        if "variables" in d:
            measured += ("; F7 L0 (development check, data <= 2005, band 2006-2019): " + ", ".join(
                f"{v} {r['reality']['inside_share']:.0%}" for v, r in d["variables"].items()))
    f = RUNS / "f7_l1_1990.json"
    if f.is_file():
        d = json.loads(f.read_text(encoding="utf-8"))
        detail["f7_l1_1990"] = d
        h = d.get("hindcast")
        if h is None:
            return Score(crit, RED, measured=measured, detail=detail,
                         cause="F7 calibration of the L1 for origin 1990 found no non-implausible parameters: "
                               + d["calibration"]["history_matching"].get("stop", ""))
        rows = h["variables"]
        key = lambda r: r.get("posterior_declared", r["posterior"])  # noqa: E731
        measured += (f"; F7 calibrated L1 hindcast 1990->{h['end']} (posterior + declared discrepancy + data error, data <= 1990 only), "
                     "years inside p10-p90 uncalibrated -> posterior -> posterior + declared discrepancy: " + ", ".join(
                         f"{v} {r['uncalibrated']['inside_share']:.0%} -> {r['posterior']['inside_share']:.0%} -> "
                         f"{key(r)['inside_share']:.0%}" for v, r in rows.items()))
        bad = [v for v, r in rows.items() if key(r)["inside_share"] < 0.80]
        if bad:
            return Score(crit, RED, measured=measured, detail=detail,
                         cause=f"calibrated (F7) hindcast inside the band in < 80 % of years for {', '.join(bad)}")
        return Score(crit, GREEN, measured=measured, detail=detail)
    if red:
        return Score(crit, RED, measured=measured, detail=detail, cause="; ".join(red) + "; F7 calibration not run")
    return Score(crit, AMBER, measured=measured, detail=detail,
                 cause="all aggregates measured and inside; band calibration is W8")


def _w5(crit) -> Score:
    d = _rolling()
    if d is None:
        return Score(crit, BLOCKED, cause="no rolling-origin run", waits_on=VALIDATE)
    d = d["w5"]
    cells = d["cells"]
    worst = max(cells, key=lambda k: abs(cells[k]["ratio"] - 1))
    layers = sorted({k.split(":")[0] for k in cells}) or ["F1"]
    measured = (f"layers {', '.join(layers)} added with no recalibration of the L1: "
                f"{len(cells) - len(d['outside'])}/{len(cells)} metric cells of the variables each layer must not "
                f"move inside their band (max(2 %, seed noise)); largest move {worst} ratio {cells[worst]['ratio']:.3f}")
    if d["outside"]:
        return Score(crit, RED, measured=measured, detail=d, cause=f"outside the band: {', '.join(d['outside'])}")
    return Score(crit, GREEN, measured=measured, detail=d)


def _w6(crit) -> Score:
    """Relación metamórfica de F2 (más recurso último -> pico más tarde; sin límite -> sin pico) con los
    mercados solos. El resto de W6 (modos de World3, signos de cada palanca) espera a F6."""
    from world4.data import atlas
    if not atlas.verify()["ok"]:
        return Score(crit, BLOCKED, cause="Atlas data not installed", waits_on=("python world4_cli.py --install-atlas",))
    import warnings
    warnings.simplefilter("ignore")
    from world4.core.l1_atlas import calibrate
    from world4.sectors import resources
    peaks = {}
    for u in (0.5, 1.0, 1.5, 1e6):
        r = resources.standalone(calibrate.W, 69, 150, g_world=0.02, u_scale=u)
        peaks[u] = {c: r["years"][int(np.argmax(r["Q"][c]))] for c in ("oil", "gas")}
    later = all(peaks[0.5][c] < peaks[1.0][c] < peaks[1.5][c] for c in ("oil", "gas"))
    nolimit = all(peaks[1e6][c] == 2169 for c in ("oil", "gas"))
    measured = (f"F2: oil peak {peaks[0.5]['oil']} / {peaks[1.0]['oil']} / {peaks[1.5]['oil']} "
                f"with ultimate resource x0.5 / x1 / x1.5, gas {peaks[0.5]['gas']} / {peaks[1.0]['gas']} / "
                f"{peaks[1.5]['gas']}; no limit -> no peak before 2169: {'yes' if nolimit else 'NO'}")
    ok = later and nolimit
    from world4.data import climate as cdata
    if cdata.available():                     # F4: más fósil -> más calentamiento a largo plazo (no a 10 años)
        from world4.sectors import climate
        R, T = 8, {}
        for F in (0.5, 1.0, 1.5):
            cl = climate.Climate(calibrate.W, 69, R, 3, end_t=69 + 62, noise=False)
            for k in range(1, 61):
                g = cl.step(69 + k, {c: np.full(R, F) for c in climate.FUELS}, np.full(R, F), np.full(R, cl.P0))
                if k in (10, 60):
                    T[(F, k)] = g.copy()
        warm60 = bool(np.all(T[(1.5, 60)] > T[(1.0, 60)]) and np.all(T[(1.0, 60)] > T[(0.5, 60)]))
        mask10 = bool(np.median(T[(1.5, 10)]) < np.median(T[(0.5, 10)]))
        measured += (f"; F4: fossil x0.5 / x1 / x1.5 from 2019 -> GMST 2079 median {np.median(T[(0.5, 60)]):.2f} / "
                     f"{np.median(T[(1.0, 60)]):.2f} / {np.median(T[(1.5, 60)]):.2f} degC, monotone in every member: "
                     f"{'yes' if warm60 else 'NO'}; at 10 years more fossil is COOLER in the median (aerosol masking): "
                     f"{'yes' if mask10 else 'no'}")
        ok = ok and warm60
    from world4.data import food as fdata
    if fdata.available() and cdata.available():   # F5: el calor baja el rendimiento en los países cálidos
        from world4.core.coupling import food_fits
        from world4.sectors import food
        W, LY = calibrate.W, calibrate.LY
        eqs = {k: v for k, v in food_fits(W, LY, 69).items() if k != "_e0_food"}
        lyld = []
        for dT in (0.0, 2.0):
            fd = food.Food(W, LY, 69, eqs, 2, 0, noise=False)
            R, N = 2, W.N
            fd.step(70, np.broadcast_to(np.nan_to_num(LY[:, 69]), (R, N)), np.zeros((R, N)),
                    fd.tbar[None] + fd.tdev_prev + dT, np.broadcast_to(W.pop[:, 69], (R, N)))
            lyld.append(fd.lY[0].copy())
        hot, mild = (fd.tbar > 24) & fd.ok, (fd.tbar < 12) & fd.ok
        dh, dm = float(np.mean(lyld[1][hot] - lyld[0][hot])), float(np.mean(lyld[1][mild] - lyld[0][mild]))
        measured += (f"; F5: +2 degC local anomaly -> cereal yield {100 * (np.exp(dh) - 1):+.1f} % in hot countries "
                     f"(mean T > 24 degC), {100 * (np.exp(dm) - 1):+.1f} % in cool ones (< 12 degC)")
        ok = ok and dh < 0
    from world4.validate import w6
    r = w6.run(log=lambda *a: None)
    l0r = r["l0r"]
    mono = l0r["d_levers_monotone"]
    measured += (f"; L0 (F6): more resource -> later fossil peak {l0r['a_more_resources_later_peak']['peak_year']}; "
                 f"no limits -> max GDP-pc drop {l0r['b_no_limits_no_collapse']['max_drop_from_peak']:.1%}; low resource "
                 f"-> overshoot and collapse (drop {l0r['c_low_resources_overshoot_collapse']['max_drop_from_peak']:.0%}); "
                 f"{sum(v['ok'] for v in mono.values())}/{len(mono)} levers monotone with the declared sign; "
                 f"pinned stocks in 2100: {r['pinned']['pinned'] or 'none'}; mutant (climate-damage loop inverted) "
                 f"{'caught' if r['mutant_caught'] else 'NOT caught'}; the dominant loop changes at "
                 f"{r['dominance']['share_matched']:.0%} of the mode changes ({r['dominance']['mode_change_years']}); "
                 f"World3-03 (informative): more resources -> later peak "
                 f"{'yes' if r['world3']['a_more_resources_later_peak']['ok'] else 'NO'}, low resources -> collapse "
                 f"{'yes' if r['world3']['c_low_resources_overshoot_collapse']['ok'] else 'NO'}, no limits -> no collapse "
                 f"{'yes' if r['world3']['b_no_limits_no_collapse']['ok'] else 'NO (other limits in its structure)'}")
    ok = ok and r["ok"]
    # F8: las palancas de política, medidas en vivo (unos segundos)
    from world4.validate import f8
    pol = f8.run(log=lambda *a: None)
    r = {**r, "f8_policy_levers": pol}
    measured += (f"; F8 policy levers: {pol['n_monotone']}/{pol['n']} monotone with the declared sign, mutant "
                 f"(renewables effect inverted) {'caught' if pol['mutant_caught'] else 'NOT caught'}")
    ok = ok and pol["n_monotone"] == pol["n"] and pol["mutant_caught"]
    if not ok:
        return Score(crit, RED, measured=measured, detail=r, cause="a declared relation fails")
    return Score(crit, GREEN, measured=measured, detail=r)


def _w7(crit) -> Score:
    """Cuentas cerradas de F3 (energía: bruta = autoconsumo + neta; la mezcla suma la energía primaria)
    en un pronóstico corto, más la de las cohortes de F1 (pruebas). Faltan masas (C, N), SFC y empleo."""
    from world4.data import atlas, energy as edata, wpp
    if not (atlas.verify()["ok"] and wpp.available() and edata.available()):
        return Score(crit, BLOCKED, cause="Atlas, WPP or energy data not installed",
                     waits_on=("python world4_cli.py --install-atlas --install-wpp --install-energy",))
    import warnings
    warnings.simplefilter("ignore")
    from world4.core import coupling
    from world4.sectors import energy
    f = coupling.forecast(60, R=4, horizon=5, f2=coupling.F2Config(), f3=coupling.F3Config())
    en = f["sim"].en
    a = en.accounts()
    gross = np.maximum(a["gross"], 1e-12)
    bal = float(np.max(np.abs(a["gross"] - a["own_use"] - a["net"]) / gross))
    mix = float(np.max(np.abs(sum(a["gen"][s] for s in energy.SOURCES) - en.PE) / np.maximum(en.PE, 1e-12)))
    neg = float(min(np.min(a["gen"][s]) for s in energy.SOURCES))
    measured = (f"energy (F3): |gross - own use - net| / gross max {bal:.1e}; |sum of sources - primary energy| / "
                f"primary energy max {mix:.1e}; min generation {neg:.2e} TWh; cohorts (F1): closed to 1e-9 in tests")
    cbal = 0.0
    from world4.data import climate as cdata
    if cdata.available():
        fc = coupling.forecast(60, R=4, horizon=5, f4=coupling.F4Config())
        c = fc["sim"].cl.carbon_accounts()
        cbal = float(max(np.max(np.abs(c["cumulative"] - c["airborne"] - c["uptake"]) / np.abs(c["cumulative"])),
                         np.max(np.abs(c["boxes"] - c["airborne"]) / np.abs(c["airborne"])),
                         np.max(np.abs(c["conc_excess"] - c["airborne_as_conc"]) / np.abs(c["conc_excess"]))))
        measured += f"; carbon (F4): emitted = airborne + sinks, boxes = airborne, ppm = airborne x ppm/Gt, max {cbal:.1e}"
    from world4.data import food as fdata
    nbal = lbal = 0.0
    if cdata.available() and fdata.available():
        ff = coupling.forecast(60, R=4, horizon=5, f5=coupling.F5Config())
        fd = ff["sim"].fd
        L = fd.land()
        okl = np.isfinite(L["land"]) & fd.ok[None]
        lbal = float(np.max(np.abs(L["cropland"] + L["pasture"] + L["forest"] + L["other"] - L["land"])[okl]
                            / L["land"][okl]))
        nb = fd.n_budget()
        g = np.isfinite(nb["input"]) & (nb["input"] > 0)
        nbal = float(np.max(np.abs(nb["input"] - nb["output"] - nb["balance"])[g] / nb["input"][g]))
        measured += f"; land (F5): crop + pasture + forest + other = land, max {lbal:.1e}; nitrogen (F5): inputs = outputs + balance, max {nbal:.1e}"
        cbal = max(cbal, lbal, nbal)
    if bal > 1e-9 or mix > 1e-9 or neg < -1e-9 or cbal > 1e-9:
        return Score(crit, RED, measured=measured, cause="an energy or carbon balance does not close")
    return Score(crit, AMBER, measured=measured,
                 cause=("stock-flow-consistent debt and assets, and employment <= working-age population, need a "
                        "labour market and financial accounts the L1 does not have"))


def _w10(crit) -> Score:
    f = RUNS / "w10.json"
    if not f.is_file():
        return Score(crit, BLOCKED, cause="no W10 run", waits_on=("python -m world4.validate.w10",))
    d = json.loads(f.read_text(encoding="utf-8"))
    measured = (f"on {d['cpu']}: L0 regional {d['l0r_ms_median']:.2f} ms per run 2019-2100 (median of 50; p90 "
                f"{d['l0r_ms_p90']:.2f}); World3-03 {d['world3_ms_single']:.0f} ms single, "
                f"{d['world3_ms_per_member_n1000']:.2f} ms per member in a batch of 1000")
    if "l1_64_2019_2100_s" in d:
        measured += (f"; L1 F1-F5 64 members 2019-2100 {d.get('l1_64_parallel_8x8_s', float('nan')):.0f} s in 8 processes "
                     f"x 8 members ({d['l1_64_2019_2100_s']:.0f} s in one process)")
    ok = d["l0_ok"] and d.get("l1_ok", False)
    if ok:
        return Score(crit, GREEN, measured=measured, detail=d)
    cause = []
    if not d["l0_ok"]:
        cause.append("L0 above 10 ms")
    if not d.get("l1_ok", False):
        cause.append("the L1 with F1-F5 (FaIR, cohorts, markets) takes more than 60 s for 64 members to 2100 even in "
                     "8 processes")
    return Score(crit, RED, measured=measured, detail=d, cause="; ".join(cause))


def _w11(crit) -> Score:
    from world4.data import atlas
    if not atlas.verify()["ok"]:
        return Score(crit, BLOCKED, cause="Atlas data not installed", waits_on=("python world4_cli.py --install-atlas",))
    from world4 import honesty
    r = honesty.measure(log=lambda *a: None)
    n = sum(len(v) for v in r["violations"].values())
    measured = (f"{len(r['paths'])} public paths (API, every MCP tool, the HTML report): {n} numbers without unit or "
                f"band or wrong 'exploratory'; report {'clean' if not r['report'] else r['report']}; "
                f"forbidden word in the code: {'none' if not r['code_words'] else list(r['code_words'])}; mutants caught "
                f"{sum(r['mutants'].values())}/{len(r['mutants'])} ({', '.join(r['mutants'])})")
    if r["ok"]:
        return Score(crit, GREEN, measured=measured, detail=r)
    return Score(crit, RED, measured=measured, detail=r, cause="a public path breaks a honesty rule, or a detector "
                                                             "misses its mutant")


def build() -> Campaign:
    camp = Campaign("Phy-World", CRITERIA)
    camp.measure("W1")(_w1)
    camp.measure("W2")(_w2)
    camp.measure("W12")(_w12)
    camp.measure("W8")(_w8)
    camp.measure("W9a")(_w9a)
    camp.measure("W3")(_w3)
    camp.measure("W4")(_w4)
    camp.measure("W5")(_w5)
    camp.measure("W6")(_w6)
    camp.measure("W7")(_w7)
    camp.measure("W10")(_w10)
    camp.measure("W11")(_w11)
    for cid, waits in BLOCKS.items():
        camp.block(cid, waits)
    return camp
