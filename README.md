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
(`--mcp`), and honesty measured as a criterion (W11). The science is in
[docs/Quasar_World4_Science.md](docs/Quasar_World4_Science.md).
Phase **V1** starts a *current World3*: the L0 is scored against observed world series (population, life
expectancy, GDP, food, fossil CO2, atmospheric CO2; `--world3-score`) and its population can run as cohorts by
single age and sex instead of four boxes (`api.run_l0v()`). Measured: the cohort engine reproduces world
population 1950-2023 within 1 % given the real fertility and life expectancy, and it shows that World3's own
fertility equation, not its age structure, is what remains wrong ([docs/V1-informe.md](docs/V1-informe.md)).
Phase **V2** splits the world into fixed groups of countries (10 regions by default, or 4 income groups), each with
its own cohorts, and replaces World3's fertility and life-expectancy tables with two equations estimated on a
236-country panel. History up to an origin year is data; the model runs from there. Out of sample (data up to
the origin only), world population to 2023 is within 2.4 % from 1980, 1.8 % from 1990, 1.2 % from 2000 and 0.3 %
from 2010. The behaviour to 2100 is not usable yet: it rests on extrapolations the data do not cover
([docs/V2-informe.md](docs/V2-informe.md)).
Phase **V3** estimates World3's technology with data instead of leaving it off: yield, resource use and pollution
per unit of activity follow curves fitted to cereal yields and fossil CO2, the capital cost of technology is fitted
to GDP (it comes out as zero or negative, not World3's surcharge), and arable land follows HYDE cropland. With data
up to 2000 only, the 2000-2023 error falls from 26.7 % to 6.6 % in output, from 14.1 % to 2.2 % in food per person
and from 106 % to 13.6 % in resource use. In the projection the famine of scenario 1 disappears and the limit
becomes non-renewable resources; when it bites depends on assumptions history does not pin down
([docs/V3-informe.md](docs/V3-informe.md)).
Phase **V4** adds carbon and climate (emissions from the model's resource use, atmospheric CO2 and global
temperature from two lines fitted to Mauna Loa and GISTEMP, a declared 5 % yield loss per degree of warming) and
six explicit rules for what the data do not cover, such as life expectancy when services fall or fertility after a
crisis. Out of sample from 2000, CO2 is within 3.4 ppm and temperature within 0.13 degC to 2023; the rules change
no test against data and decide the scarcity scenarios ([docs/V4-informe.md](docs/V4-informe.md)).
Phase **V5** replaces the single trajectory with the ensemble of models history does not rule out: ten uncertain
parameters sampled from declared ranges and kept only if they fit 1971-2023, bootstrapped coefficients, and a model
error measured on the previous split. The industrial peak falls in 2037 / 2071 / 2100 or later (p10 / p50 / p90) and
is decided by the initial non-renewable resources, which history barely pins down; the population peak (2055-2066,
9.2-10.0 billion) is robust. From 2000 the bands cover what happened in 11 of 12 indicators but are too wide
([docs/V5-informe.md](docs/V5-informe.md)).

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
.venv/Scripts/python -m pip install -e .[world]                # rdata + pycountry (V1, V2)
.venv/Scripts/python world4_cli.py --install-world            # observed world series and country panel for the L0 + WPP by age (V1, V2)
```

The Atlas data are not shipped here: several derive from sources that cannot be redistributed
(EM-DAT) or whose licence is unverified. See `world4/data/sources.json` and `NOTICE`.

## Use

```bash
python world4_cli.py --info                 # environment, observables, sources and licences
python world4_cli.py --campaign             # acceptance criteria W1-W12
python world4_cli.py --l0 --scenario 2      # World3-03 behaviour mode
python world4_cli.py --world3-score         # V1-V4: World3-03, cohorts, regions, technology and climate against the observed world; out-of-sample tests
python world4_cli.py --world3-bands --members 100   # V5: uncertainty bands to 2100 and their coverage test (~70 s)
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
python world4_cli.py --validate             # W3/W5 rolling origin + hindcasts (population, psi, energy, CO2, GMST, cereals, cropland) + AR6 check (~50 min)
```

Exit codes: 0 ok, 2 spec error, 3 infeasible (for `--theta`: a declared limit is violated), 4 `--strict` with degradations.

```python
from world4 import api
api.query("conflict_hazard", "COL", 2000, 2005)   # Distribution with p10/p50/p90, unit, caveats
api.run_l0(scenario=1)
w3, cohorts = api.run_l0v()                       # World3-03 with population cohorts by age and sex (V1)
w3, regions = api.run_l0v(groups="regions", origin=2023)   # 10 regions, estimated equations, history to 2023 (V2)
w3, regions = api.run_l0v(groups="regions", origin=2023, technology="estimated", climate="estimated")   # the whole current World3 (V3, V4)
api.world3_score()                                # error of each L0 variant against the observed world
api.world3_bands(members=60)                      # p10/p50/p90 to 2100 from the ensemble history does not rule out (V5)
```

## Tests

```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python scripts/run_atlas_validation.py runs/validation_port.json   # full W2 (~8 min)
```

## Licence

Apache 2.0. Third-party material is listed in `NOTICE`.
