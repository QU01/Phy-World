# QUASAR Phy-World (World4)

An integrated world-system model: a World3-03 biophysical core (L0) coupled, in later phases, to a
180-country agent model of the world system (L1: core/periphery, structural-demographic
pressure, debt, institutions, conflict). Phase **F0** delivers the foundations: both layers run,
each one pinned to the model it comes from, and every data source is registered with its licence.
Phase **F1** makes population (single-age, sex-specific cohorts) and human capital (schooling by
cohort) endogenous after the forecast origin, coupled to the L1 without touching the ported Atlas.
Phase **F2** makes global scarcity endogenous: the world resource-rent share `psi_t`, which the Atlas
read from the data, is now an OUTPUT of four world markets (oil, gas, coal, metals) with depletion.
Phase **F3** adds energy with EROI: the primary-energy mix of every country (fossil, nuclear, hydro,
wind+solar, other), net energy, learning, and the copper/nickel demand of the transition.
Phase **F4** replaces the Atlas's instantaneous TCRE with the FaIR v2.2.4 climate model and its calibrated,
constrained ensemble (one configuration per member): CO2 and the other emissions come from the
model's energy, markets and population, and the L1 damages (BHM, disasters) see FaIR's temperature.
Phase **F5** adds food, land, nitrogen and water per country (FAOSTAT): cereal yields respond to
fertiliser and to the L1's local temperature, land is conserved, the cropland N balance closes,
agricultural and forest emissions feed FaIR, and the ecological footprint and biocapacity that the
Atlas read as data are now computed from the model's own flows.
Phase **F6** adds the regional L0 (10 IAM-style regions, 112 stocks, the L1's structure aggregated and
estimated on the same data; under 10 ms per run to 2100) and loop-dominance analysis (an open
implementation of the *Loops that Matter* score): `api.run_l0r()`, `api.loop_dominance()`.
Phase **F7** adds Bayesian calibration: wave history matching with a NumPy GP emulator, ABC-SMC with the
declared model error as the final kernel (Wilkinson 2013), AR1 discrepancy diagnostics
(`phycore.uq.DiscrepancyGP`), for the L0 (data 1992-2019) and for the L1 (pseudo-hindcasts before the
origin), shipped as `models/posterior.npz`; and the W8 recalibration by nested selection.
`api.run_l0r(posterior=True)` gives bands; `world4_cli.py --calibrate` rebuilds everything.
Phase **F8** adds policy levers (carbon price, renewables, efficiency, agricultural R&D, aid, fertility,
fiscal consolidation) with boxes and units, SSP-like narratives as parameter sets, a robust search with
`phycore.Designer` over an ensemble of states of the world (percentile objectives, planetary-style limits as
constraints, regret) and PRIM for the conditions in which each policy fails; and the Atlas's inverse problem
as an alternative conduct ("what countries would do" vs "what would suit them"). `api.evaluate_policy()`,
`api.design()`; `world4_cli.py --theta "carbon_price=100,renewables=0.01"` evaluates one policy.
Phase **F9** makes it a standalone product: `pip install` gives a `world4` command; a wizard when the command
line says nothing, `--help-numbers`, `--fidelity`, WALLS / TIGHT / THE LAST BLOCKER when the limits cannot be met;
a self-contained HTML report (`--report`), the Atlas globe with World4 layers (`--dashboard`), an MCP server
(`--mcp`), and honesty measured as a criterion (W11).
Phase **F11** adds conflict and event prediction on discrete map cells: every PRIO-GRID cell (0.5°, ~55 km,
64 818 land cells) gets, for each of the next 12 months, the probability of state-based, non-state and one-sided
violence (UCDP GED 26.1 + UCDP Candidate 2026), the expected events and state-based deaths, and the share of its
risk that is structural background or contagion from its own past, its neighbours and its country. Two models are
averaged: a direct logistic hurdle per horizon (as in ViEWS) and a discrete-time spatio-temporal Hawkes process
fitted by EM and simulated forward; both are backtested against the ViEWS 2023/24 benchmarks
([docs/F11-informe.md](docs/F11-informe.md)). `api.conflict_grid_forecast()` returns a column-oriented JSON
payload meant for the coming React interface.
Phase **F12** adds that interface: `world4_cli.py --serve` serves a React app (`ui/`, built with `npm run build`) and
a JSON API over the whole system: the conflict map per cell, World3 <-> society, the ten World3 scenarios, the L1
fan per country, policies and regions, and the data and licences ([ui/README.md](ui/README.md)). The science is in
[docs/Quasar_World4_Science.md](docs/Quasar_World4_Science.md).

| Layer | What it is in F0 | Pinned against | Result |
|---|---|---|---|
| L0 | World3-03 re-implemented in NumPy, vectorised over regions | Vensim World3-03 output (5 variables, 1900-2100) | max error 6.9e-6 |
| L1 | Atlas del Sistema-Mundo, ported unchanged | the Atlas at commit 9e96f6b | 81/81 probe fingerprints and 780/780 validation values bit-identical |
| F1 | population cohorts + human capital coupled to the L1 | the Atlas re-run with observed population (49 origins) | GDP-pc CRPS ratio 0.998 / 0.993 / 0.985 at 1/5/10 years; population CRPS at 10 years -33 % vs a 5-year trend |

It is **not** a forecaster of the long-run future. Results within 10 years come from the
Atlas's out-of-sample-tested L1, with its known defects declared in every answer; L0 runs are
behaviour modes of the 2004 model.

## Install

```bash
pip install .                                                # from a checkout (the repo is not published yet)
world4 --info                                                # the CLI; data go to $W4_HOME (default ~/.world4)
world4 --install-atlas --install-wpp --install-energy --install-climate --install-food
world4 --install-conflict-grid                               # F11: UCDP GED + Candidate, PRIO-GRID (no Atlas needed)
```

From a checkout everything lives next to the code (`data_cache/`, `runs/`), as below.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e .[test]
.venv/Scripts/python world4_cli.py --install-atlas            # clones the Atlas, checks SHA-256
.venv/Scripts/python world4_cli.py --install-atlas --atlas-src <local clone at 9e96f6b>
.venv/Scripts/python -m pip install -e .[wpp]
.venv/Scripts/python world4_cli.py --install-wpp              # UN WPP 2024 by age and sex (F1), SHA-256 checked
.venv/Scripts/python world4_cli.py --install-energy           # OWID energy-data at a pinned commit (F3)
.venv/Scripts/python -m pip install -e .[climate]             # FaIR 2.2.4 (F4)
.venv/Scripts/python world4_cli.py --install-climate          # FaIR calibration 1.6.0 + HadCRUT5, SHA-256 checked (F4)
.venv/Scripts/python world4_cli.py --install-food             # FAOSTAT + WDI water, SHA-256 checked (F5)
```

The Atlas data are not shipped here: several derive from sources that cannot be redistributed
(EM-DAT) or whose licence is unverified. See `world4/data/sources.json` and `NOTICE`.

## Use

```bash
python world4_cli.py --info                 # environment, observables, sources and licences
python world4_cli.py --campaign             # acceptance criteria W1-W12
python world4_cli.py --l0 --scenario 2      # World3-03 behaviour mode
python world4_cli.py --query gdp_pc --region MEX --as-of 2005 --target 2010
python world4_cli.py --query population --region NGA --as-of 2005 --target 2015
python world4_cli.py --query resource_rent_share_world --region WLD --as-of 2005 --target 2012
python world4_cli.py --query primary_energy --region DEU --as-of 2005 --target 2012
python world4_cli.py --query gmst --region WLD --as-of 2005 --target 2015
python world4_cli.py --query co2_emissions_world --region WLD --as-of 2005 --target 2015
python world4_cli.py --query cereal_production --region IND --as-of 2005 --target 2015
python world4_cli.py                        # no mode: the wizard (in a terminal)
python world4_cli.py --help-numbers         # what each mode needs, and nothing else
python world4_cli.py --report runs/report.html --theta "carbon_price=150,efficiency=0.01"   # self-contained HTML
python world4_cli.py --dashboard            # the Atlas globe with World4 layers (runs/dashboard/index.html)
python world4_cli.py --mcp                  # MCP server on stdio (tools: w4_query, w4_evaluate_policy, ...)
python world4_cli.py --levers               # F8: policy levers and narratives
python world4_cli.py --theta "carbon_price=150,efficiency=0.01" --narrative ssp2   # one policy, robust objectives, PRIM
python world4_cli.py --design               # F8: robust search, PRIM, regret, conduct (~10 min)
python world4_cli.py --calibrate            # F7: posterior (models/posterior.npz), calibrated W4 hindcast, W8 (~1 h)
python world4_cli.py --conflict-grid        # F11: violence forecast per PRIO-GRID cell, next 12 months (JSON payload)
python world4_cli.py --conflict-grid out.json --origin 2025-12 --min-p 0   # any origin, all cells
python world4_cli.py --validate-conflict-grid   # F11: rolling origins 2019-2025 vs the ViEWS benchmarks
python world4_cli.py --serve                # F12: the React interface on http://127.0.0.1:8765/ (cd ui && npm install && npm run build first)
python world4_cli.py --validate             # W3/W5 rolling origin + hindcasts (population, psi, energy, CO2, GMST, cereals, cropland) + AR6 check (~50 min)
```

Exit codes: 0 ok, 2 spec error, 3 infeasible (for `--theta`: a declared limit is violated), 4 `--strict` with degradations.

```python
from world4 import api
api.query("conflict_hazard", "COL", 2000, 2005)   # Distribution with p10/p50/p90, unit, caveats
api.run_l0(scenario=1)
api.conflict_grid_forecast(min_p=0.05)            # F11: cells, months, P(event), deaths, drivers, countries
```

## Tests

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/run_atlas_validation.py runs/validation_port.json   # full W2 (~8 min)
```

## Licence

Apache 2.0. Third-party material is listed in `NOTICE`.
