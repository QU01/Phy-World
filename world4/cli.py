"""QUASAR Phy-World - command line (F0-F9).

Family pattern (Phy-HX / Phy-CT): flat flags, no sub-commands; cli_style transplanted; every
degradation announced and recorded; exit codes 0 ok, 2 spec error, 3 infeasible, 4 --strict with
degradations. F0 offers: --info, --campaign, --install-atlas, --l0 and --query; F0.5 adds
--build-store and --structural; F1 adds --install-wpp and --validate (W3/W4/W5 runs); F3 adds --install-energy; F4 adds --install-climate; F5 adds --install-food; F11 adds --install-conflict-grid, --conflict-grid and --validate-conflict-grid; F13 adds --event-density and --study-factors. The wizard (no arguments)
arrives in F9: with no mode, a terminal gets the wizard and a pipe gets --info. F9 adds --help-numbers,
--fidelity, --report, --dashboard and --mcp, and WALLS / TIGHT / THE LAST BLOCKER when a policy is infeasible.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from world4 import cli_style as ui
from world4 import home

ROOT = home.home()

EXIT_OK, EXIT_SPEC, EXIT_INFEASIBLE, EXIT_STRICT = 0, 2, 3, 4


class SpecError(ValueError):
    pass


def _data_errors() -> tuple:
    from world4.calib.posterior import PosteriorMissing
    from world4.data.atlas import AtlasDataError
    from world4.structural.store import StoreMissing
    out = [AtlasDataError, PosteriorMissing, StoreMissing, FileNotFoundError]
    for mod, name in (("world4.data.wpp", "WPPDataError"), ("world4.data.energy", "EnergyDataError"),
                      ("world4.data.climate", "ClimateDataError"), ("world4.data.food", "FoodDataError"),
                      ("world4.data.conflict_grid", "ConflictGridDataError")):
        try:
            out.append(getattr(__import__(mod, fromlist=[name]), name))
        except (ImportError, AttributeError):
            pass
    return tuple(out)


DATA_ERRORS = _data_errors()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="world4_cli.py", description=__doc__.splitlines()[0])
    m = p.add_argument_group("what to do")
    m.add_argument("--info", action="store_true", help="environment, data sources, licences, campaign")
    m.add_argument("--campaign", action="store_true", help="measure the acceptance criteria W1-W12")
    m.add_argument("--install-atlas", action="store_true",
                   help="install the Atlas data (copied from its repo at the pinned commit, SHA-256 checked)")
    m.add_argument("--l0", action="store_true", help="run the L0 (World3-03) and print the main stocks")
    m.add_argument("--query", metavar="VARIABLE", help="L1 distribution of VARIABLE (see --info)")
    m.add_argument("--build-store", action="store_true",
                   help="F0.5: L1 ensembles for every origin + historical questions + W8/W9a evaluation")
    m.add_argument("--install-wpp", action="store_true",
                   help="F1: download UN WPP 2024 by age and sex (SHA-256 checked) for the population sector")
    m.add_argument("--install-energy", action="store_true",
                   help="F3: download OWID energy-data (pinned commit, SHA-256 checked) and the UN regions")
    m.add_argument("--install-climate", action="store_true",
                   help="F4: download the calibrated FaIR ensemble 1.6.0 (Zenodo, CC BY 4.0) and HadCRUT5 (SHA-256 "
                        "checked); needs pip install fair==2.2.4")
    m.add_argument("--install-food", action="store_true",
                   help="F5: download FAOSTAT (crops, land use, fertilisers, N budget, emissions, food balances) and "
                        "WDI water withdrawals (SHA-256 checked)")
    m.add_argument("--install-conflict-grid", action="store_true",
                   help="F11: UCDP GED 26.1 + UCDP Candidate events and PRIO-GRID 2.0 cells (SHA-256 checked)")
    m.add_argument("--conflict-grid", metavar="JSON", nargs="?", const="",
                   help="F11: forecast of organized violence per PRIO-GRID cell, next 12 months (JSON payload to "
                        "runs/conflict_grid/forecast_<origin>.json or JSON)")
    m.add_argument("--validate-conflict-grid", metavar="JSON", nargs="?", const="",
                   help="F11: rolling-origin validation 2019-2025 against the ViEWS benchmarks (~1 h)")
    m.add_argument("--event-density", metavar="JSON", nargs="?", const="",
                   help="F13: events per 10 000 km² per country in --origin + --horizon from the Atlas regressions")
    m.add_argument("--study-factors", action="store_true",
                   help="F13: which Atlas and World3 factors predict events (out of sample) and the backtest (~15 min)")
    m.add_argument("--validate", "--validate-f1", dest="validate", action="store_true",
                   help="F1-F5: rolling-origin W3/W5 (Atlas, F1 ... F1-F5) + free hindcasts of population, psi, "
                        "the energy mix, CO2, GMST, cereals and cropland + AR6 carbon-budget check")
    m.add_argument("--calibrate", action="store_true",
                   help="F7: history matching + ABC-SMC of the L0 (2019, and 2005 out of sample) and the L1 (origin "
                        "1990 for W4, live 2019), W8 recalibration; writes models/posterior.npz and runs/f7_*.json")
    m.add_argument("--theta", metavar="LEVERS",
                   help="F8: evaluate one policy without searching, e.g. 'carbon_price=100,renewables=0.01' "
                        "(levers: see --levers); robust objectives, limits and vulnerabilities (PRIM)")
    m.add_argument("--levers", action="store_true", help="F8: the policy levers with box and unit")
    m.add_argument("--design", action="store_true",
                   help="F8: robust policy search (NSGA-II over the L0 ensemble), PRIM, regret, conduct; "
                        "writes runs/f8_*.json (~10 min)")
    m.add_argument("--help-numbers", action="store_true", help="F9: the numbers each mode needs, and nothing else")
    m.add_argument("--report", metavar="HTML", nargs="?", const="",
                   help="F9: self-contained HTML report (bands, loops, limits, provenance, campaign); with --theta it "
                        "includes the policy and its vulnerabilities")
    m.add_argument("--dashboard", metavar="DIR", nargs="?", const="",
                   help="F9: the Atlas globe with World4 layers (one HTML file, globe.gl from its CDN)")
    m.add_argument("--serve", metavar="PORT", nargs="?", const=8765, type=int,
                   help="F12: the React interface and its JSON API on http://127.0.0.1:PORT (default 8765)")
    m.add_argument("--mcp", action="store_true", help="F9: run the MCP server on stdio (JSON-RPC 2.0)")
    m.add_argument("--structural", metavar="REQUEST_JSON",
                   help="F0.5: answer one Prime Radiant request (a JSON file, or - for stdin) as JSON")
    n = p.add_argument_group("the numbers")
    n.add_argument("--scenario", type=int, default=1, help="L0 scenario of the 2004 book (1-5)")
    n.add_argument("--fidelity", default=None, choices=("L0", "L1", "L2"),
                   help="F9: L1 for --query (default), L0 for --theta/--report (default); L2 degrades with notice")
    n.add_argument("--no-campaign", action="store_true", help="F9: --report without measuring the campaign (faster)")
    n.add_argument("--narrative", default="ssp2", help="F8: SSP-like narrative of the ensemble (ssp1-ssp5)")
    n.add_argument("--sow", type=int, default=200, help="F8: states of the world in the ensemble for --theta")
    n.add_argument("--region", default=None, help="ISO3 code for --query (WLD for gmst)")
    n.add_argument("--as-of", type=int, default=None, help="origin year for --query (1970-2018)")
    n.add_argument("--target", type=int, default=None, help="target year for --query (1-10 years ahead)")
    n.add_argument("--members", type=int, default=32, help="ensemble members for --query")
    n.add_argument("--origin", default=None,
                   help="forecast origin: AAAA-MM for --conflict-grid (last data month), AAAA for --event-density (2025)")
    n.add_argument("--horizon", type=int, default=1, help="F13: years ahead for --event-density (1-5)")
    n.add_argument("--min-p", type=float, default=0.01,
                   help="F11: --conflict-grid keeps cells with P(any event in 12 months) >= this (0 = all cells)")
    n.add_argument("--atlas-src", default=None, help="local Atlas clone for --install-atlas (else git clone)")
    o = p.add_argument_group("output")
    o.add_argument("--json", action="store_true", help="print the result as JSON and nothing else")
    o.add_argument("--strict", action="store_true", help="exit 4 if anything was degraded")
    o.add_argument("--no-color", action="store_true")
    return p


# ------------------------------------------------------------------ modes
def run_info(a) -> int:
    from world4 import MODEL_REVISION, __version__, api
    from world4.data import atlas, vintages
    print(ui.banner())
    print()
    ui.title("QUASAR Phy-World - environment")
    ui.kv("version", __version__, "", f"MODEL_REVISION {MODEL_REVISION}")
    ui.kv("L0", "World3-03 (NumPy)", "", "equations from pydynamo (public domain), checked against Vensim")
    ui.kv("L1", "Atlas del Sistema-Mundo", "", f"ported unchanged from {atlas.ATLAS_COMMIT[:7]}")
    v = atlas.verify()
    ui.kv("Atlas data", ui.status(v["ok"], "INSTALLED", "MISSING"), "",
          v["root"] if v["ok"] else "python world4_cli.py --install-atlas [--atlas-src <clone>]")
    if not v["ok"]:
        ui.degraded("L1", "Atlas data not installed: --query and W2 are unavailable")

    ui.section("Phases and what each gives you")
    ui.kv("F0 - foundations", "implemented", "", "L0 + ported L1 + data registry + this CLI")
    ui.kv("F0.5 - Prime Radiant", "implemented", "", "structural estimator (--build-store, --structural)")
    from world4.data import wpp
    ui.kv("F1 - population", "implemented" if wpp.available() else ui.c("WPP DATA MISSING", "warn"), "",
          "endogenous cohorts + human capital in --query" + ("" if wpp.available() else
                                                            ": python world4_cli.py --install-wpp"))
    ui.kv("F2 - resources", "implemented", "", "psi_t from 4 world markets (oil, gas, coal, metals) in --query")
    from world4.data import energy as _ed
    ui.kv("F3 - energy", "implemented" if _ed.available() else ui.c("ENERGY DATA MISSING", "warn"), "",
          "mix, EROI, net energy, transition materials" + ("" if _ed.available() else ": --install-energy"))
    from world4.data import climate as _cd
    ui.kv("F4 - climate", "implemented" if _cd.available() else ui.c("CLIMATE DATA MISSING", "warn"), "",
          "FaIR calibrated ensemble, endogenous emissions" + ("" if _cd.available() else ": --install-climate"))
    from world4.data import food as _fdd
    ui.kv("F5 - food", "implemented" if _fdd.available() else ui.c("FOOD DATA MISSING", "warn"), "",
          "cereals, land, N, water; footprint and biocapacity computed" + ("" if _fdd.available() else ": --install-food"))
    for ph, what in (
                     ("F6-F8", "regional L0, Bayesian calibration, robust policies"),
                     ("F9-F12", "product, Studio addon, full Prime Radiant, hardening")):
        ui.kv(ph, ui.c("NOT BUILT", "warn"), "", what)

    ui.section("Observables (api.catalog)")
    for o in api.catalog():
        hz = f"validated <= {o['validated_horizon_years']} y" if o["validated_horizon_years"] else o["note"]
        ui.kv(f"{o['fidelity']} {o['variable']}", o["unit"], "", hz)

    ui.section("Data sources and licences (W12)")
    for r in vintages.license_table():
        tag = ui.c(r["mark"], "ok" if r["mark"] == "[V]" else ("warn" if r["mark"] == "[S]" else "err"))
        red = r["redistributable"]
        red = ui.c("no-redistribution", "err") if red == "NO" else ("redistributable" if red == "yes" else "licence ?")
        ui.kv(r["id"], f"{r['license']} {tag}", "", f"{red}; lag {r['lag_years']} y; {r['used_in']}")
    print()
    print("  " + ui.dim("[V] verified in this session, [S] assumed, [?] unknown. World4 vendors no third-party "
                        "data."))
    ui.section("Acceptance campaign (not measured here: --campaign measures it)")
    from world4 import campaign
    for crit in campaign.CRITERIA:
        waits = campaign.BLOCKS.get(crit.cid)
        ui.kv(crit.cid, crit.what, "", ("waits on " + waits[0]) if waits else "measured by --campaign")
    return EXIT_OK


def run_campaign(a) -> int:
    from world4 import campaign
    camp = campaign.build()
    scores = camp.score_all()
    gate = camp.gate(scores)
    if a.json:
        print(json.dumps({"gate": gate, "scores": [{"cid": s.criterion.cid, "verdict": s.verdict,
                                                     "measured": s.measured, "cause": s.cause,
                                                     "waits_on": list(s.waits_on)} for s in scores]},
                         indent=1))
        return EXIT_OK
    print(ui.banner())
    print()
    print(camp.report(scores))
    print()
    ui.kv("release gate", ui.status(gate["passed"], "PASSES", "BLOCKED"), "",
          f"{len(gate['blocking'])} criteria red or blocked (expected before v1.0)")
    return EXIT_OK


def run_install(a) -> int:
    from world4.data import atlas
    if not a.json:
        print(ui.banner())
        ui.step("installing the Atlas data" + (f" from {a.atlas_src}" if a.atlas_src else " (git clone)"))
    v = atlas.install(a.atlas_src)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("files", str(len(atlas.manifest()["files"])), "", "SHA-256 checked")
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["root"])
    if not v["ok"]:
        ui.err(f"missing {v['missing'][:3]} mismatched {v['mismatched'][:3]}")
        return EXIT_SPEC
    return EXIT_OK


def run_l0(a) -> int:
    from world4.core import l0
    if a.scenario not in l0.SCENARIOS:
        raise SpecError(f"--scenario must be one of {sorted(l0.SCENARIOS)}")
    r = l0.run(scenario=a.scenario)
    years = (1900, 1950, 2000, 2025, 2050, 2075, 2100)
    idx = [int(round((y - 1900) / 0.5)) for y in years]
    keys = (("pop", "population", 1e9, "billion"), ("f", "food", 1e12, "1e12 veg-eq kg/y"),
            ("io", "industrial output", 1e12, "1e12 1968 USD/y"), ("nrfr", "resources left", 1, "fraction"),
            ("ppolx", "persistent pollution", 1, "index 1970 = 1"), ("hwi", "human welfare index", 1, "0-1"))
    if a.json:
        print(json.dumps({"scenario": a.scenario, "years": years,
                          **{k: [float(r[k][i, 0]) for i in idx] for k, *_ in keys}}, indent=1))
        return EXIT_OK
    print(ui.banner())
    print()
    ui.title(f"L0 - World3-03, scenario {a.scenario}")
    print("  " + ui.dim("a behaviour mode of the 2004 model, not a forecast"))
    print("  " + "".join(f"{y:>9d}" for y in years).rjust(24 + 9 * len(years)))
    for k, label, scale, unit in keys:
        print(f"  {label:<22s}" + "".join(f"{r[k][i, 0] / scale:9.3g}" for i in idx) + "  " + ui.dim(unit))
    if a.scenario != 1:
        print()
        print("  " + ui.warn_text("only scenario 1 is anchored to Vensim output (W1); this one is unanchored"))
    return EXIT_OK


def run_query(a) -> int:
    from world4 import api
    missing = [f for f, v in (("--region", a.region), ("--as-of", a.as_of), ("--target", a.target)) if v is None]
    if missing:
        raise SpecError(f"--query needs {', '.join(missing)}")
    try:
        d = api.query(a.query, a.region, a.as_of, a.target, members=a.members)
    except api.QueryError as exc:
        raise SpecError(str(exc)) from None
    if a.json:
        print(d.to_json())
        return EXIT_OK
    print(ui.banner())
    print()
    print(ui.panel(f"{d.variable} | {d.region} | {d.as_of} -> {d.target}", [
        ("p10 / p50 / p90", f"{d.p10:.4g} / {d.p50:.4g} / {d.p90:.4g}"),
        ("unit", d.unit), ("members", str(d.members)), ("fidelity", d.fidelity),
        ("calibrated band", ui.c("NO (W8, F0.5)", "warn")),
        ("population", d.provenance.get("population", "observed")),
        ("resource rents", d.provenance.get("psi", "Atlas (frozen)")),
        ("energy", d.provenance.get("energy", "none")),
        ("climate", d.provenance.get("climate", "Atlas TCRE")),
        ("food", d.provenance.get("food", "none")),
        ("params cutoff", str(d.provenance["params_cutoff"])),
        ("resolves against", d.provenance["resolution_source"]),
    ]))
    print()
    for cav in d.caveats:
        print("  " + ui.warn_text(cav))
    ui.degraded("band", "p10-p90 not calibrated (D7); do not read it as an 80 % interval")
    return EXIT_OK


RUNS = home.runs()


def run_install_wpp(a) -> int:
    from world4.data import wpp
    log = (lambda *x: None) if a.json else (lambda m: ui.step(m))
    if not a.json:
        print(ui.banner())
        ui.step("UN WPP 2024 by single age and sex (CC BY 3.0 IGO)")
    v = wpp.install(log=log)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["npz"])
    return EXIT_OK if v["ok"] else EXIT_SPEC


def run_install_energy(a) -> int:
    from world4.data import energy
    log = (lambda *x: None) if a.json else (lambda m: ui.step(m))
    if not a.json:
        print(ui.banner())
        ui.step(f"OWID energy-data at {energy.COMMIT[:7]} + UN regions")
    v = energy.install(log=log)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["npz"])
    return EXIT_OK if v["ok"] else EXIT_SPEC


def run_install_climate(a) -> int:
    from world4.data import climate
    log = (lambda *x: None) if a.json else (lambda m: ui.step(m))
    if not a.json:
        print(ui.banner())
        ui.step("FaIR calibration data 1.6.0 (Zenodo 18828694, CC BY 4.0) + HadCRUT5 5.1.0.0 (OGL v3)")
    v = climate.install(log=log)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["root"])
    return EXIT_OK if v["ok"] else EXIT_SPEC


def run_install_food(a) -> int:
    from world4.data import food
    log = (lambda *x: None) if a.json else (lambda m: ui.step(m))
    if not a.json:
        print(ui.banner())
        ui.step("FAOSTAT bulk files (CC BY 4.0 + FAO terms) + WDI water withdrawals (CC BY 4.0)")
    v = food.install(log=log)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["npz"])
    return EXIT_OK if v["ok"] else EXIT_SPEC


def run_install_conflict_grid(a) -> int:
    from world4.data import conflict_grid
    log = (lambda *x: None) if a.json else (lambda m: ui.step(m))
    if not a.json:
        print(ui.banner())
        ui.step("UCDP GED 26.1 + UCDP Candidate 2026 (CC BY 4.0) and PRIO-GRID 2.0 cells and covariates")
    v = conflict_grid.install(log=log)
    if a.json:
        print(json.dumps(v, indent=1))
    else:
        ui.kv("result", ui.status(v["ok"], "OK", "MISMATCH"), "", v["npz"])
    return EXIT_OK if v["ok"] else EXIT_SPEC


def run_conflict_grid(a) -> int:
    from world4 import api
    from world4.data import conflict_grid as cdata
    if not cdata.available():
        raise cdata.ConflictGridDataError("conflict-grid data not installed: python world4_cli.py --install-conflict-grid")
    if not a.json:
        print(ui.banner())
        ui.step("forecast of organized violence per PRIO-GRID cell (logistic hurdle + spatio-temporal Hawkes)")
    try:
        out = api.conflict_grid_forecast(a.origin, min_p=a.min_p)
    except api.QueryError as exc:
        raise SpecError(str(exc)) from exc
    path = Path(a.conflict_grid) if a.conflict_grid else home.runs() / "conflict_grid" / f"forecast_{out['origin']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if a.json:
        print(json.dumps({k: out[k] for k in ("schema", "origin", "months", "summary")} | {"file": str(path)}, indent=1))
        return EXIT_OK
    c = out["cells"]
    ui.kv("origin", out["origin"], "", f"{out['months'][0]} .. {out['months'][-1]}")
    ui.kv("cells", str(out["summary"]["cells_returned"]), "", f"with P(any event) >= {a.min_p}")
    e = out["summary"]["expected_events_window"]
    ui.kv("events expected", f"sb {e['sb']:.0f}  ns {e['ns']:.0f}  os {e['os']:.0f}", "events", "whole world")
    print()
    print("  " + ui.dim("highest-risk cells: P(any event in 12 months), driver shares of next month's intensity"))
    for i in range(min(10, len(c["gid"]))):
        print(f"    gid {c['gid'][i]:>6}  ({c['lat'][i]:6.2f}, {c['lon'][i]:7.2f})  GW {c['country_gw'][i]:>3}  "
              f"P {c['p_window_any'][i]:.2f}  sb-deaths {c['deaths_sb_expected'][i]:7.1f}  "
              f"background {c['driver_background'][i]:.2f} own {c['driver_own'][i]:.2f} "
              f"neighbours {c['driver_neighbours'][i]:.2f} country {c['driver_country'][i]:.2f}")
    print()
    ui.kv("written", str(path))
    return EXIT_OK


def run_validate_conflict_grid(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.conflict import validate
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("F11 rolling-origin validation: origins " + ", ".join(validate.ORIGINS))
    r = validate.run(log=log)
    path = Path(a.validate_conflict_grid) if a.validate_conflict_grid else \
        home.runs() / "conflict_grid" / "validacion.json"
    validate.save(r, path)
    if a.json:
        print(json.dumps(r["summary"], indent=1))
    else:
        for o in ("sb", "any"):
            for m, v in r["summary"][o]["all"].items():
                x = v["all_h"]
                ui.kv(f"{o} {m}", f"Brier {x['brier']:.5f}  AUPR {x['aupr']:.3f}  AUROC {x['auroc']:.3f}")
        ui.kv("written", str(path))
    return EXIT_OK


def run_event_density(a) -> int:
    from world4 import api
    if not a.json:
        print(ui.banner())
        ui.step("F13: event density per country from the Atlas regressions")
    try:
        out = api.event_density(int(a.origin) if a.origin else None, a.horizon, cells=bool(a.event_density))
    except (api.QueryError, ValueError) as exc:
        raise SpecError(str(exc)) from exc
    if a.event_density:
        Path(a.event_density).parent.mkdir(parents=True, exist_ok=True)
        Path(a.event_density).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    if a.json:
        print(json.dumps({k: v for k, v in out.items() if k not in ("cells",)}, ensure_ascii=False, default=str))
        return EXIT_OK
    ui.kv("origin -> target", f"{out['origin']} -> {out['target']}", "", f"model trained with targets <= {out['train_until']}")
    ui.kv("world", f"{out['world_events_expected']:,.0f}", "events expected")
    print()
    print("  " + ui.dim("country   expected  density/10k km2  P(any)    p10      p50      p90   observed"))
    for r in out["countries"][:15]:
        obs = "" if r["observed"] is None else f"{r['observed']:.0f}"
        print(f"  {r['iso3']:<7} {r['events_expected']:>9.0f} {r['density']:>16.2f} {r['p_any']:>7.2f} {r['p10']:>8.0f} "
              f"{r['p50']:>8.0f} {r['p90']:>8.0f} {obs:>9}")
    if a.event_density:
        ui.kv("written", a.event_density)
    return EXIT_OK


def run_study_factors(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.conflict import density, factors
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("F13: factors of organized-violence events (Atlas + World3), out of sample")
    res = factors.run(horizons=(1, 3, 5), log=log)
    factors.save(res, density.STUDY)
    bt = density.backtest(log=log)
    density.STUDY.with_name("backtest.json").write_text(json.dumps(bt, ensure_ascii=False, default=float), encoding="utf-8")
    if a.json:
        print(json.dumps(bt["summary"], default=float))
    else:
        ui.kv("written", str(density.STUDY.parent))
    return EXIT_OK


def run_validate(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.validate import hindcast, rolling
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("W3/W5: 49 origins x (Atlas, F0.5 trend, F1, F1+F2, F1+F2+F3, F1-F4, F1-F5, ablation, Atlas with "
                "another seed), R = 32")
    s = rolling.run(log=log)
    s["w3"], s["w5"] = rolling.verdict_w3(s), rolling.verdict_w5(s)
    RUNS.mkdir(exist_ok=True)
    (RUNS / "w3.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    if not a.json:
        ui.step("W4: free hindcasts (population 1990 -> 2023; energy mix 1990 and 2005 -> 2023; psi 1980 and 1990 -> 2019)")
    h = hindcast.run(log=log)
    (RUNS / "w4_pop_hindcast.json").write_text(json.dumps(h, indent=1), encoding="utf-8")
    en = {}
    for origin in hindcast.ENERGY_ORIGINS:
        en[origin] = hindcast.run_energy(origin, log=log)
        (RUNS / f"w4_energy_hindcast_{origin}.json").write_text(json.dumps(en[origin], indent=1), encoding="utf-8")
    psi = {}
    for origin in hindcast.PSI_ORIGINS:
        psi[origin] = hindcast.run_psi(origin, log=log)
        (RUNS / f"w4_psi_hindcast_{origin}.json").write_text(json.dumps(psi[origin], indent=1), encoding="utf-8")
    cl = {}
    from world4.data import climate as cdata
    if cdata.available():
        if not a.json:
            ui.step("W4/F4: CO2 and GMST 1990 and 2005 -> 2023; FaIR with observed emissions; TCRE and AR6 budget")
        for origin in hindcast.CLIMATE_ORIGINS:
            cl[origin] = hindcast.run_climate(origin, log=log)
            (RUNS / f"w4_climate_hindcast_{origin}.json").write_text(json.dumps(cl[origin], indent=1), encoding="utf-8")
        (RUNS / "f4_fair_observed.json").write_text(json.dumps(hindcast.run_climate_observed(log=log), indent=1),
                                                    encoding="utf-8")
        (RUNS / "f4_carbon_budget.json").write_text(json.dumps(hindcast.run_carbon_budget(log=log), indent=1),
                                                    encoding="utf-8")
    from world4.data import food as fdata
    if cdata.available() and fdata.available():
        if not a.json:
            ui.step("W4/F5: cereals, cropland, food supply and N balance 1990 and 2005 -> 2023")
        for origin in hindcast.FOOD_ORIGINS:
            (RUNS / f"w4_food_hindcast_{origin}.json").write_text(json.dumps(hindcast.run_food(origin, log=log), indent=1),
                                                                  encoding="utf-8")
    if a.json:
        print(json.dumps({"w3": s["w3"], "w5": s["w5"],
                          "w4_population": {k: h[k] for k in ("inside_share", "rel_err_p50_end", "rel_err_trend_end", "ok")},
                          "w4_psi": {o: {k: r[k] for k in ("inside_share", "mae_p50", "mae_atlas", "ok")}
                                     for o, r in psi.items()}}, indent=1))
    else:
        ui.ok(f"wrote runs/w3.json and runs/w4_*.json; --campaign scores them")
    return EXIT_OK


def run_build_store(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.structural import questions, store
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("L1 ensembles for origins 1970-2018 and the live origin 2019 (spawn workers)")
    store.build(log=log)
    questions.generate(log=log)
    RUNS.mkdir(exist_ok=True)
    w9a = questions.evaluate_w9a(log=log)
    (RUNS / "w9a.json").write_text(json.dumps(w9a, indent=1), encoding="utf-8")
    w8 = questions.evaluate_w8(log=log)
    (RUNS / "w8.json").write_text(json.dumps(w8, indent=1), encoding="utf-8")
    if a.json:
        print(json.dumps({"w9a": w9a, "w8": w8}, indent=1))
    else:
        ui.ok(f"wrote {RUNS / 'w9a.json'} and {RUNS / 'w8.json'}")
    return EXIT_OK


def run_calibrate(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.calib import l0r, l1, posterior
    from world4.structural import questions
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    dump = lambda name, obj: (RUNS / name).write_text(json.dumps(obj, indent=1, default=_np_default),  # noqa: E731
                                                     encoding="utf-8")
    RUNS.mkdir(exist_ok=True)
    if not a.json:
        print(ui.banner())
        ui.step("F7/L0: history matching + ABC-SMC with data <= 2005, band 2006-2019 (development check)")
    dump("f7_l0_oos_2005.json", l0r.out_of_sample(2005, 2019, log=log))
    if not a.json:
        ui.step("F7/L0: production posterior with data 1992-2019")
    p0 = l0r.calibrate(2019, log=log)
    dump("f7_l0_2019.json", l0r._jsonable(p0))
    res = {}
    for origin in (1990, 2019):
        if not a.json:
            ui.step(f"F7/L1: pseudo-hindcasts before {origin}, history matching with a GP emulator, ABC-SMC"
                    + (", calibrated hindcast -> 2023 (W4)" if origin < 2019 else " (live origin)"))
        p1 = l1.calibrate(origin, log=log)
        out = {"calibration": l1.jsonable(p1)}
        if p1["ok"] and origin < 2019:
            out["hindcast"] = l1.hindcast(p1, log=log)
        dump(f"f7_l1_{origin}.json", out)
        res[origin] = p1
    path = posterior.save(p0, res[2019] if res[2019].get("ok") else None)
    if not a.json:
        ui.step("W8: recalibration with nested selection (F7)")
    w8 = questions.evaluate_w8(log=log)
    dump("w8.json", w8)
    if a.json:
        print(json.dumps({"posterior": str(path)}))
    else:
        ui.ok(f"wrote {path} and runs/f7_*.json, runs/w8.json; --campaign scores them")
    return EXIT_OK


def _np_default(o):
    return o.tolist() if hasattr(o, "tolist") else str(o)


def run_levers(a) -> int:
    from world4.scenarios import levers, narratives
    rows = levers.describe()
    if a.json:
        print(json.dumps({"levers": rows, "narratives": {k: v["what"] for k, v in narratives.NARRATIVES.items()}},
                         indent=1))
        return EXIT_OK
    print(ui.banner())
    print()
    ui.title("F8 policy levers (theta)")
    for r in rows:
        print(f"  {r['name']:<14s} [{r['box'][0]:g}, {r['box'][1]:g}] {r['unit']:<30s} {ui.dim(r['what'])}")
        way = " up" if r["declared_sign"] > 0 else " down"
        print(f"  {'':<14s} {ui.dim('declared: ' + r['target'] + way + ' ' + r['mark'])}")
    print()
    ui.title("narratives (parameter sets, not trajectories)")
    for k, v in narratives.NARRATIVES.items():
        print(f"  {k:<6s} {v['what']}")
    return EXIT_OK


def run_theta(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.scenarios import levers, robust
    try:
        th = levers.parse(a.theta)
    except ValueError as exc:
        raise SpecError(str(exc)) from exc
    r = robust.evaluate_policy(th, members=a.sow, narrative=a.narrative)
    walls = [k for k, v in r["constraints"].items() if v > 0]
    if a.json:
        print(json.dumps(r, indent=1, default=_np_default))
        return EXIT_INFEASIBLE if walls else EXIT_OK
    print(ui.banner())
    print()
    ui.title(f"F8 policy, narrative {a.narrative}, {r['members']} states of the world (L0, to 2100)")
    print("  " + ui.dim("a behaviour mode under deep uncertainty, not a forecast"))
    for k, v in r["theta"].items():
        if v:
            print(f"  {k:<14s} {v:g}")
    print()
    for k, v in r["objectives"].items():
        print(f"  {k:<18s} {v:+9.3f}   no policy {r['baseline_objectives'][k]:+9.3f}")
    print(f"  better than no policy: welfare in {r['better_than_baseline_share']['welfare']:.0%} of worlds, "
          f"peak GMST in {r['better_than_baseline_share']['peak_gmst']:.0%}")
    print()
    for k, v in r["constraints"].items():
        tag = ui.warn_text("WALL") if v > 0 else "ok  "
        print(f"  {tag} {k:<12s} g = {v:+.3f}   (no policy {r['baseline_constraints'][k]:+.3f})")
    for label, v in r["vulnerability"].items():
        if not isinstance(v, dict) or not v.get("chosen"):
            continue
        ch = v["chosen"]
        print()
        print(f"  fails ({label}) in {v['base_rate']:.0%} of {r['vulnerability']['members']} worlds; PRIM box covers "
              f"{ch['coverage']:.0%} of failures at density {ch['density']:.0%}:")
        for b in ch["restricted"][:4]:
            print(f"    {b['name']:<12s} in [{b['lo']:.3g}, {b['hi']:.3g}] of [{b['range'][0]:.3g}, {b['range'][1]:.3g}]")
    if walls:
        from world4.scenarios import robust as rb
        print_diagnosis(rb.diagnose(narrative=a.narrative), "policy")
        return EXIT_INFEASIBLE
    return EXIT_OK


def print_diagnosis(d: dict, what: str = "policy") -> None:
    """WALLS / TIGHT / THE LAST BLOCKER sobre TODA la población (la lección de Phy-CT y Phy-Prop)."""
    ui.section("Why the limits are not met")
    ui.kv("population", f"{d['population']} {what} evaluations", "",
          f"{d['feasible']} feasible; the diagnosis is over all of them, not the one asked for")
    if d["walls"]:
        print()
        print("  " + ui.c("WALLS", "err", bold=True) + ui.dim("  -- not one policy in the lever box satisfies these. "
                                                             "More search will not help; the SPEC has to change."))
        for w in d["walls"]:
            ui.kv(w["name"], f"violated by {100 * w['violated_share']:.0f} %", "",
                  f"best achievable g = {w['g_best']:+.3f} (needs <= 0)")
    if d["tight"]:
        print()
        print("  " + ui.c("TIGHT", "warn", bold=True) + ui.dim("  -- most policies miss these, but some clear them."))
        for w in d["tight"]:
            ui.kv(w["name"], f"violated by {100 * w['violated_share']:.0f} %", "", f"best g = {w['g_best']:+.3f}")
    if d["last_blocker"]:
        print()
        print("  " + ui.c("THE LAST BLOCKER", "warn", bold=True) + ui.dim("  -- policies that would be feasible if this "
                                                                         "ONE constraint were dropped."))
        for b in d["last_blocker"][:3]:
            ui.kv(b["name"], f"{b['n']} policies blocked by it alone", "", f"closest g = {b['g_closest']:+.3f}")
    for w in d["walls"] + d["last_blocker"][:1]:
        adv = w.get("advice")
        if adv is None:
            from world4.scenarios import robust as rb
            adv = rb.ADVICE.get(w["name"])
        if adv:
            print()
            ui.warn(f"{ui.bold(w['name'])}: {adv}")
            break


def run_design(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4.scenarios import conduct, narratives, robust
    from world4.validate import f8
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    dump = lambda name, obj: (RUNS / name).write_text(json.dumps(obj, indent=1, default=_np_default),  # noqa: E731
                                                     encoding="utf-8")
    RUNS.mkdir(exist_ok=True)
    if not a.json:
        print(ui.banner())
        ui.step("F8: lever monotonicity and its mutant")
    dump("f8_levers.json", f8.run(log=log))
    if not a.json:
        ui.step(f"F8: robust search, narrative {a.narrative}, declared limits (food, land, debt)")
    res = robust.design(narrative=a.narrative, log=log)
    dump(f"f8_front_{a.narrative}.json", res)
    if not a.json and res["feasible_on_front"] == 0:
        print_diagnosis(res["diagnosis"], "policy")
    for nar in narratives.NARRATIVES:
        if not a.json:
            ui.step(f"F8: robust search, narrative {nar}, land reported instead of constraining")
        dump(f"f8_front_{nar}_land_reported.json", robust.design(narrative=nar, report_only=("land_cap",), rounds=4,
                                                                 log=log))
    if not a.json:
        ui.step("F8: conduct 2019->2030, 'would' (cloned rule) vs 'advised' (IRL-profitable deviations), L1 F1-F5")
    dump("f8_conduct.json", conduct.compare(log=log))
    if not a.json:
        ui.ok("wrote runs/f8_*.json")
    return EXIT_OK


MODE_NUMBERS = {
    "--query": [("--query VARIABLE", "an L1 variable of --info (gdp_pc, population, gmst...)"),
                ("--region ISO3", "a country, or WLD for world variables"),
                ("--as-of YEAR", "origin year, 1970-2018 (data <= origin only)"),
                ("--target YEAR", "1-10 years after --as-of (the measured horizon)")],
    "--theta": [("--theta 'lever=value,...'", "any subset of the levers below; the rest stay at 0"),
                ("--narrative ssp1..ssp5", "the world the policy lands in (default ssp2)"),
                ("--sow N", "states of the world in the ensemble (default 200)")],
    "--l0": [("--scenario 1-5", "World3-03 scenario of the 2004 book")],
    "--report": [("--report [FILE.html]", "optional; add --theta for a policy, --no-campaign to skip the campaign")],
    "--dashboard": [("--dashboard [DIR]", "optional output directory")],
}


def run_help_numbers(a) -> int:
    from world4.scenarios import levers
    if a.json:
        print(json.dumps({"modes": {k: [n for n, _ in v] for k, v in MODE_NUMBERS.items()},
                          "levers": levers.describe()}, indent=1))
        return EXIT_OK
    print(ui.banner())
    print()
    ui.title("The numbers, per mode")
    print("  You do not write a scenario. You write these, and nothing else.\n")
    for mode, rows in MODE_NUMBERS.items():
        print(ui.panel(f"{mode}  -  {len(rows)} numbers", [(f"{i}.", ui.bold(n) + "  " + ui.dim(w))
                                                           for i, (n, w) in enumerate(rows, 1)]))
        print()
    ui.section("The levers of --theta (box, unit; effects are [M])")
    for r in levers.describe():
        ui.kv(r["name"], f"[{r['box'][0]:g}, {r['box'][1]:g}]", r["unit"], r["what"])
    return EXIT_OK


def _fidelity(a, default: str, allowed: tuple) -> str:
    f = a.fidelity or default
    if f not in allowed:
        if f == "L2":
            ui.degraded("fidelity L2", f"L2's emulators (FaIR, crops) are already coupled inside the L1 since F4/F5; "
                                       f"running {allowed[-1]}")
            return allowed[-1]
        raise SpecError(f"--fidelity {f} is not available for this mode (allowed: {', '.join(allowed)})")
    return f


def run_report(a) -> int:
    import warnings
    warnings.simplefilter("ignore")
    from world4 import report
    from world4.scenarios import levers
    _fidelity(a, "L0", ("L0",))
    th = None
    if a.theta:
        try:
            th = levers.parse(a.theta)
        except ValueError as exc:
            raise SpecError(str(exc)) from exc
    out = a.report or str(RUNS / "report.html")
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("F9: HTML report" + (" with the policy" if th is not None else ""))
    if a.no_campaign:
        ui.degraded("report campaign", "--no-campaign: the acceptance campaign is not in this report")
    r = report.write(out, th, narrative=a.narrative, members=a.sow, campaign=not a.no_campaign, log=log)
    if a.json:
        print(json.dumps(r))
    else:
        ui.ok(f"wrote {r['html']} ({r['bytes'] / 1e3:.0f} kB, self-contained) and its data next to it")
    return EXIT_OK


def run_dashboard(a) -> int:
    from world4 import dashboard
    out = a.dashboard or str(RUNS / "dashboard")
    log = (lambda *x: None) if a.json else (lambda m: print("  " + m, flush=True))
    if not a.json:
        print(ui.banner())
        ui.step("F9: the Atlas globe with World4 layers")
    r = dashboard.build(out, log=log)
    if a.json:
        print(json.dumps(r))
    else:
        ui.ok(f"wrote {r['html']} ({r['countries']} countries, {r['bytes'] / 1e3:.0f} kB); open it in a browser "
              "(the globe library comes from its CDN; offline it shows the table)")
    return EXIT_OK


def run_wizard(a) -> int:
    """The guided path: asks what to compute, then only for the numbers of that mode."""
    print(ui.banner())
    print()
    ui.section("What do you want to compute?")

    def ask(prompt, default, cast=str):
        try:
            s_ = input("  " + ui.c(prompt, "key") + ui.dim(f" [{default}]") + ": ").strip()
        except EOFError:
            print()
            return default
        if not s_:
            return default
        try:
            return cast(s_)
        except ValueError:
            ui.warn(f"not a {cast.__name__}, using {default}")
            return default

    menu = [("1", "a country variable 1-10 years ahead (L1, --query)"),
            ("2", "a policy over the L0 ensemble to 2100 (--theta, exploratory)"),
            ("3", "the HTML report (--report)"), ("4", "the acceptance campaign (--campaign)"),
            ("5", "environment, data and licences (--info)")]
    for k, t in menu:
        print(f"  {ui.bold(k)}  {t}")
    print()
    pick = ask("choice", "5")
    print()
    if pick == "1":
        a.query = ask("variable", "gdp_pc")
        a.region = ask("region (ISO3, or WLD)", "MEX")
        a.as_of = ask("as of (origin year, 1970-2018)", 2008, int)
        a.target = ask("target year (1-10 years later)", a.as_of + 5, int)
        return run_query(a)
    if pick == "2":
        from world4.scenarios import levers
        print("  levers: " + ", ".join(f"{lv.name} [{lv.lo:g}-{lv.hi:g} {lv.unit}]" for lv in levers.LEVERS))
        a.theta = ask("policy (lever=value,...)", "carbon_price=100,efficiency=0.01")
        a.narrative = ask("narrative (ssp1-ssp5)", "ssp2")
        return run_theta(a)
    if pick == "3":
        if ask("include a policy? (y/n)", "n").lower().startswith("y"):
            a.theta = ask("policy (lever=value,...)", "carbon_price=100,efficiency=0.01")
        return run_report(a)
    if pick == "4":
        return run_campaign(a)
    return run_info(a)


def run_structural(a) -> int:
    """JSON in, JSON out, nothing else on stdout: this is what Studio calls."""
    import warnings
    warnings.simplefilter("ignore")
    from world4.structural import estimator
    raw = sys.stdin.read() if a.structural == "-" else Path(a.structural).read_text(encoding="utf-8")
    try:
        req = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(json.dumps({"status": "unbound", "message": f"bad JSON: {exc}"}))
        return EXIT_SPEC
    out = estimator.estimate(req)
    print(json.dumps(out))
    return EXIT_OK if out["status"] == "ok" else EXIT_INFEASIBLE


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    ui.init(no_color=a.no_color or a.json)
    ui.reset_degradations()
    try:
        if a.structural:
            return run_structural(a)
        if a.mcp:
            from world4.mcp import server
            return server.serve()
        if a.serve is not None:
            from world4 import server as ui_server
            ui_server.serve(a.serve)
            return EXIT_OK
        if a.install_atlas:
            code = run_install(a)
        elif a.install_wpp:
            code = run_install_wpp(a)
        elif a.install_energy:
            code = run_install_energy(a)
        elif a.install_climate:
            code = run_install_climate(a)
        elif a.install_food:
            code = run_install_food(a)
        elif a.install_conflict_grid:
            code = run_install_conflict_grid(a)
        elif a.conflict_grid is not None:
            code = run_conflict_grid(a)
        elif a.validate_conflict_grid is not None:
            code = run_validate_conflict_grid(a)
        elif a.event_density is not None:
            code = run_event_density(a)
        elif a.study_factors:
            code = run_study_factors(a)
        elif a.validate:
            code = run_validate(a)
        elif a.build_store:
            code = run_build_store(a)
        elif a.calibrate:
            code = run_calibrate(a)
        elif a.help_numbers:
            code = run_help_numbers(a)
        elif a.report is not None:
            code = run_report(a)
        elif a.dashboard is not None:
            code = run_dashboard(a)
        elif a.levers:
            code = run_levers(a)
        elif a.theta is not None:
            _fidelity(a, "L0", ("L0",))
            code = run_theta(a)
        elif a.design:
            code = run_design(a)
        elif a.campaign:
            code = run_campaign(a)
        elif a.l0:
            code = run_l0(a)
        elif a.query:
            _fidelity(a, "L1", ("L1",))
            code = run_query(a)
        elif not a.info and not a.json and sys.stdin.isatty():
            code = run_wizard(a)
        else:
            code = run_info(a)
            if not a.info and not a.json:
                print()
                print("  " + ui.dim("no mode given and no terminal to ask in: this is --info (the wizard needs a "
                                    "terminal; --help-numbers lists what each mode needs)"))
    except SpecError as exc:
        ui.err(str(exc))
        return EXIT_SPEC
    except DATA_ERRORS as exc:
        ui.err(f"missing data or model file: {exc}")
        print("  " + ui.dim("install it first: --install-atlas, --install-wpp, --install-energy, --install-climate, "
                            "--install-food, --install-conflict-grid (the message above names which); W4_HOME says where it goes"))
        return EXIT_SPEC
    if a.strict and ui.strict_verdict():
        if not a.json:
            ui.err(f"--strict: {len(ui.strict_verdict())} degradation(s)")
        return EXIT_STRICT
    return code


if __name__ == "__main__":
    sys.exit(main())
