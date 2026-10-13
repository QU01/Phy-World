---
tags: [quasar, phy-world, world4, science, world3, system-dynamics, agent-based, calibration, history-matching, dmdu]
created: 2026-10-13
related: [[Quasar_PhyProp_Science]], [[Quasar_PhyHX_Science]]
status: Technical Document v1.0 (F0-F9)
author: Quasar Solutions Lab
---

# QUASAR Phy-World (World4) — Scientific Foundations
## The structure, the data and the evidence behind an integrated world-system model, and nothing else

> This document is the technical companion to [README.md](../README.md). The README says *what* the tool
> does; this one gives the *why*: the equations of each layer, how each was estimated, how it was judged,
> and what failed. The phase reports in `docs/F*-informe.md` (Spanish) carry the full measurements.
>
> **Admission rule.** A number appears here only if (a) it was measured by this repository (the command
> that reproduces it is named), or (b) its source was opened and read. Every assumption carries its mark:
> `[V]` verified in the primary source or measured here · `[S]` secondary or scaled from a sibling ·
> `[M]` a modelled assumption, chosen, with its uncertainty declared · `[?]` not verified, never an anchor.
> Section 12 lists what was deliberately left out.
>
> **What World4 is not.** It is not a forecaster of the long-run future. Up to 10 years it gives
> distributions that were measured out of sample; beyond that it gives behaviour modes, labelled
> `exploratory`, which are never published as forecasts.

---

## Table of Contents

1. [Layers and conventions](#1)
2. [L0a: World3-03, re-implemented](#2)
3. [L0b: the regional L0](#3)
4. [L1: the Atlas and its endogenous sectors](#4)
5. [Why: loop dominance](#5)
6. [Calibration: history matching and ABC-SMC](#6)
7. [Bands delivered to others: the W8 recalibration](#7)
8. [Policies under deep uncertainty](#8)
9. [Conduct: what countries would do and what would suit them](#9)
10. [Honesty and data as tests](#10)
11. [The acceptance campaign](#11)
12. [What this document does not contain, and why](#12)
13. [How to check all of this](#13)

---

<a name="1"></a>
## 1. Layers and conventions

| Layer | Content | Cost (measured, Ryzen 7 5700X) | Use |
|---|---|---|---|
| L0a | World3-03 in NumPy (`core/l0.py`) | 134 ms per run; 0.32 ms per member in a batch of 1000 | Parity anchor (W1), informative behaviour modes |
| L0b | Regional L0: 10 IAM-style regions, 112 stocks (`core/l0r`) | 9.3 ms per run 2019-2100; 0.09 ms per member in a batch | Behaviour modes, calibration, policy search, loop dominance |
| L1 | The Atlas del Sistema-Mundo (180 countries) with endogenous population, markets, energy, climate and food (`core/l1_atlas`, `core/coupling.py`, `sectors/`) | 64 members 2019-2100 in 36 s on 8 processes | Country distributions 1-10 years ahead |

FaIR and the crop sector, which the roadmap placed in a separate L2, are coupled inside the L1 since F4/F5;
`--fidelity L2` therefore degrades to L1 with a notice.

Conventions enforced in code:
- **Time.** Annual steps in L1 and the regional L0; Euler with dt = 0.5 in World3-03.
- **What was known on `as_of`.** Every data source has a `known_time` (`data/vintages.py`): its publication lag. A forecast from an origin uses parameters estimated with data up to that origin only.
- **Seeds.** Ensembles use common random numbers, so two policies or two model variants are compared in the same worlds.

<a name="2"></a>
## 2. L0a: World3-03, re-implemented

**Equations.** Transcribed from pydynamo, which is public domain, but given **Vensim semantics**:
- at each instant, all auxiliaries are computed from that instant's stocks;
- SMOOTH, SMOOTH3 and DELAY3 are explicit stocks.

**Reference.** The Vensim output of World3-03, scenario 1, as published by WorldDynamics.jl (MIT) in `test/world3_03.jl`. It is copied with its SHA-256 into `tests/reference/` and is not part of the package.

**Result (W1) [V].** The maximum relative error over food, industrial output, non-renewable resources, the persistent pollution index and population, 1900-2100, is **6.9e-6**: the rounding of Vensim's six printed digits. The threshold is 1 %.

**Findings that changed the plan:**
- The `lmhs` table switches at `Time > 1940`, not `>=`. With `>=` the error rises to 0.4 %.
- A regression lock at 1e-4 exists because the 1 % threshold cannot see that mutant.
- pydynamo's own simulation departs from Vensim by up to 12 %, so it is not a numerical reference.

<a name="3"></a>
## 3. L0b: the regional L0

**Regions.** NAM, LAM, EUR, REF, MEA, SSA, SAS, CHN, SEA, OPA, built from the UN M49 subregions with OECD exceptions. They resemble the IAM R10 regions but are not identical.

**Stocks.** 11 per region, plus cumulative fossil extraction and cumulative CO₂. Each flow is the aggregated form of an L1 equation:

| Stock | Flow equation (estimated on the regional panel, data ≤ cutoff) |
|---|---|
| Population P | births − deaths + migration, from crude rates. Mortality depends on income (clamped to the observed range) and on the birth rate, as a proxy of age structure [S] |
| ln crude birth rate | converges to a level that depends on income |
| Capital K | ΔK = s·Y − δK |
| ln TFP | catch-up to the leading region plus an own effect (20-year mean residual shrunk by n/(n+5)) |
| Climate damage D | BHM response to regional temperature, with persistence ρ_D |
| ln energy intensity | autonomous improvement plus an income elasticity |
| logit non-fossil share | diffusion plus substitution under PHYSICAL fossil scarcity [S] (not identifiable in history) |
| ln cropland, ln yield, ln N per ha | the F5 forms, with F5's temperature coefficients |
| Debt / GDP | deficit minus dilution by growth |

**Production.** ln Y = ln A + D + α ln K + (1−α) ln P + E. E is the energy term: the real cost of energy with a fossil EROI that falls with depletion, plus a CES shortage when extraction capacity (10 % of the remaining resource per year) cannot meet demand.

**Climate.** GMST = T₀ + TCRE_eff · ΔCO₂ cumulative, with the effective TCRE estimated on 1970-cutoff.

**P2 parity with the L1 aggregate, 1992-2019** (bands declared before measuring):
- The world aggregate stays inside every year for population, GDP, energy, cereals and cropland.
- Region-year cells inside: population 79 % (the threshold is 80 %), GDP 90 %, energy 90 %, cereals 83 %, cropland 92 %, GMST 100 %.
- **Population fails by one cell:** crude rates without age structure drift 7-9 % in MEA, CHN and OPA.

<a name="4"></a>
## 4. L1: the Atlas and its endogenous sectors

**The port.** The Atlas (commit `9e96f6b`) is ported **without behaviour changes** (W2):
- 81/81 SHA-256 fingerprints of a probe and 780/780 validation values are bit-identical to the original on this machine;
- a 1e-9 change in `ALPHA` breaks parity (a mutant in the tests).

**Atlas defects found and kept on purpose, so the port stays bit-identical:**
- **AUCs with tied scores** are not reproducible across CPUs, because `argsort` breaks ties differently. World4's own metric uses average ranks.
- **`panel_full.csv` missing** makes the Atlas fall back silently to `panel.csv`. World4 generates the file and requires its SHA-256 before any L1 run.

The sectors are coupled through `CoupledSimulator` without touching the port. Each is estimated by pooled OLS with country-clustered covariance, declared sign restrictions and own effects shrunk towards zero (`sectors/_panel.py`). Each member draws its own coefficients and AR(1) common and idiosyncratic shocks.

| Phase | What becomes endogenous | Where the evidence is |
|---|---|---|
| F1 | Population by single age and sex (WPP 2024), human capital by cohort | `docs/F1-informe.md` |
| F2 | The world resource-rent share ψ_t, from four markets (oil, gas, coal, metals) with depletion | `docs/F2-informe.md` |
| F3 | Primary-energy mix by country, EROI, net energy, copper and nickel demand | `docs/F3-informe.md` |
| F4 | FaIR v2.2.4 with its calibrated, constrained ensemble 1.6.0 instead of the Atlas's TCRE | `docs/F4-informe.md` |
| F5 | Cereals, land, N and water by country (FAOSTAT); footprint and biocapacity computed from flows | `docs/F5-informe.md` |

**Zero recalibration (W5).** Adding F1-F5 moved none of the 72 metric cells that each layer must not move outside its band. The Atlas's calibration was never refitted.

**A finding about the Atlas.** In the Atlas, ψ is the exponent of an unnormalised Cobb-Douglas: +0.01 of ψ raised every country's GDP per person by 5 %. F2 normalises the production function; with the Atlas's ψ the result is identical.

<a name="5"></a>
## 5. Why: loop dominance

`core/l0r/loops.py` is an open implementation of the *Loops that Matter* score (Schoenberg, Davidsen and Eberlein 2020). That method otherwise exists only inside STELLA. The formula was written from the paper [S].

**Validation.** On the logistic model, the dominant loop switches from births to deaths in the step where P crosses K/2 (test).

**World view of the regional L0.** 13 stock types and 136 elementary cycles of up to 4 stocks:
- **around 2020:** the fertility loops dominate;
- **2040-2060:** the climate-damage loop through GDP, intensity, emissions and CO₂ dominates;
- **2080-2100:** fossil substitution under scarcity dominates.
- The dominant loop changes in the years the behaviour mode changes (2069-2071).

**Limitations, declared:**
- Flows in differences link stocks through changes, so part of the change goes unattributed.
- The polarity of first-order loops can flip when the net flow changes sign.
- There is no comparison yet with PPM or LEEA.

<a name="6"></a>
## 6. Calibration: history matching and ABC-SMC

**Emulator** (`calib/gp.py`). One GP per output:
- Matérn 5/2 kernel with ARD and a linear mean integrated out (universal kriging);
- REML with an analytic gradient (checked against finite differences to 3e-6);
- leave-one-out errors in closed form (Dubrule 1983).

**History matching** (`calib/history_matching.py`, after Vernon, Goldstein and Bower 2010; Andrianakis et al. 2015). The implausibility of output j is

  I_j(θ) = |z_j − E[f_j(θ)]| / √(Var_em,j + Var_obs,j + Var_disc,j + Var_stoch,j)

and θ is ruled out when the second-largest I_j exceeds 3.

**Posterior** (`calib/abc.py`). ABC-SMC with a Gaussian kernel. ε decreases adaptively, keeping half of the effective sample size at each step, until ε = 1, where the kernel *is* the declared error model. The result is then the exact posterior under that error (Wilkinson 2013).

On a toy problem with a known posterior (a grid), the method reproduces:
- the mean to within 0.3 sd (0.346/−0.435 against 0.349/−0.431);
- the sd to within 20 %.

**Declared variances [S].**
- **Data error:** population 1 %, GDP 3 %, energy 3 %, CO₂ 5 %, cereals 4 %, cropland 2 %, GMST 0.10 °C.
- **Discrepancy:** half of the P2 bands at 27 years, growing with √h.

**The box must not decide.** If the posterior's p10 or p90 sits within 5 % of the width of a box edge, the box is doubled and the calibration repeated, at most twice. This rule was fixed before any hindcast was seen.

**Results.**

- **Regional L0, development check** (data ≤ 2005, band 2006-2019): population, GDP, energy, cereals and cropland are inside 100 % of the years, GMST 64 %.
  - The prediction band (posterior + declared discrepancy + data error) is **conservative**.
  - The calibrated median is no better than OLS: **the contribution is the band, not the centre.**
  - The first version, which used the AR1 discrepancy fitted on in-window residuals, covered only 0-43 %. Those residuals had already been absorbed by the calibration (Brynjarsdóttir and O'Hagan 2014). The method was changed after this check, which is declared.

- **Regional L0, production posterior** (1992-2019; shipped in `world4/models/posterior.npz`):
  - It asks for savings ×1.32, mortality ×1.20 and TFP growth −0.9 %/year. These are **effective** values that compensate for the missing cohorts (the P2 failure), not physical parameters.
  - Savings, TFP convergence and TFP growth form a ridge. Savings and convergence still touch their boxes after two doublings.

- **L1** (pseudo-hindcasts from O−15, O−10 and O−5 to O; six intercept shifts):
  - **The clean test** calibrates only on data ≤ 1990 and hindcasts 1990-2023 (W4).
  - It keeps GDP, cereals and GMST inside (100 / 100 / 91 % of years), but energy and CO₂ fall to 30 %.
  - The bias learnt on 1975-1990 (the post-oil-shock fall in energy intensity) did not persist. **Bias is not stationary**, and 15 years of pseudo-history cannot separate a persistent bias from an epoch's.

<a name="7"></a>
## 7. Bands delivered to others: the W8 recalibration

**Numeric variables** (GDP growth, debt) are recalibrated by quantile recalibration (Kuleshov et al. 2018), over a Student-t kernel mixture with 3 d.o.f.: a Gaussian kernel saturated PITs at 0 and 1.

F7 adds a **recent-bias** recalibrator: standardised residuals are corrected by the median of the last 3 published years. One of four candidates is then chosen at each date by **nested selection**, using the KS distance on the PITs of the last 8 published years, each computed with the recalibrator that existed at its own date.

**Results (W8):**
- Coverage of the p10-p90 band is 0.78-0.82 in all 10 cells.
- The KS distance falls from 0.06-0.20 (F0.5) to 0.03-0.11; only 4 of 10 cells reach D ≤ 0.05.

**A methodological finding.** PITs of one year are correlated across countries, through common shocks. A perfectly calibrated forecaster with the measured within-year correlation gives D of 0.06-0.10 (p95). It is reported; **the threshold was not changed.**

<a name="8"></a>
## 8. Policies under deep uncertainty

**Levers** (`scenarios/levers.py`, effects [M]). Each lever pays:
- spending comes out of investment;
- the carbon price removes 0.3 × revenue from GDP;
- consolidation costs public investment.

| Lever | Box | Effect per year at full ramp (10 years) |
|---|---|---|
| carbon price | 0-200 USD/tCO₂ | non-fossil logit +0.05·ln(1+τ/100); intensity −0.01·ln(1+τ/100) on the fossil part |
| renewables | 0-2 % of GDP | non-fossil logit +0.04 per 1 % |
| efficiency | 0-2 % of GDP | intensity −0.5 % per 1 % |
| agricultural R&D | 0-0.5 % of GDP | yield +0.3 % per 0.1 % |
| aid | 0-1 % of donor GDP | NAM, EUR, OPA → SSA, SAS as capital (half arrives) |
| fertility | 0-1 % of GDP in SSA, SAS, MEA | crude birth rate −1 % per 1 % |
| consolidation | 0-1 of the deficit | deficit cut; 0.5 of it lost from investment |

**Ensemble of states of the world:**
- the F7 posterior;
- the knobs history cannot identify (ultimate resource, damage, EROI, substitution, deficit, heat sensitivity), drawn uniformly from their boxes (deep uncertainty);
- the effectiveness of each lever group, U(0.5, 1.5) [M];
- **narratives:** SSP-like sets of parameters, after the qualitative reading of O'Neill et al. 2017, translated into numbers as [M].

**Robust search.** `phycore.Designer` runs NSGA-II at fidelity 0, so the L0 itself is the truth, over 100 worlds per evaluation, with common random numbers. The pattern is MORDM.
- **Objectives:** p10 of welfare (discounted mean log consumption per person), p90 of the 2020-2100 maximum GMST, and p10 of the poorest region's consumption in 2100.
- **Constraints [M]:** regional cereal per person ≥ 90 % of 2019 (p10), cropland 2100 ≤ 1.10 × 2019 (p90), world debt ≤ 1.5 × GDP (p90).
- **Regret:** Savage regret per world.
- **PRIM:** implemented here (Friedman and Fisher 1999). It finds where a policy fails: peak GMST above 3 °C, or welfare below doing nothing.

**Findings:**
- **The land limit is a WALL.** No policy in the lever box meets it, in any narrative: in the L0, cropland only responds to population.
- **Treating land as reported** gives fronts of 46-75 policies. The p90 of peak GMST falls from 6.0 to 3.4 °C (SSP2), and welfare's p10 is better than with no policy across the whole front.
- **Climate-heavy policies** are worse than doing nothing in 39-60 % of the worlds, almost always those with low climate damage.
- **Most levers end at the edge of their box:** the declared political limit is what binds.
- **The carbon price alone barely moves 2100 GMST.** The L0 exhausts its fossil resource before 2100, so emissions are delayed, not avoided. Policies lower the peak through clean diffusion, efficiency and population.

<a name="9"></a>
## 9. Conduct: what countries would do and what would suit them

The Atlas's inverse problem (Bajari, Benkard and Levin 2007 estimator, `irl.py`) measured which ±0.5-logit deviations from the cloned decision rule would have raised a country's welfare **with every other country following the rule**:
- more technology (92 % of cases), more own extraction (87 %) and more redistribution (61 %);
- less importing from the core (68 %) and less foreign resource investment (56 %).

**Measured here** (L1 F1-F5, 2019-2030), applying those deviations to **all** countries at once:
- consumption per person −0.2 %;
- welfare value +0.0017, better in 66 % of members;
- revealed value (power and inertia weigh; welfare weighs negatively) +0.045, better in 100 % of members.

**The deviations are a partial-equilibrium result.** The Atlas's anchored PPO (`recommend.py`) is kept as the optional extra `world4[rl]` and was not trained.

<a name="10"></a>
## 10. Honesty and data as tests

**W11** (`honesty.py`). Three rules, each with its detector and its mutant, applied to every public path (the API, every MCP tool and the HTML report):

1. **No number without a unit and a band.**
   - A `unit` string covers everything below it; a `units` dict covers only the keys it names.
   - Every p50 has a p10 and a p90.
   - A series without a band carries a note saying why.
2. **The word "validated" is forbidden** outside the name of the measured horizon.
3. **`exploratory` is correct on every path:** true for the L0 and F8, false for the L1 inside its measured 1-10-year horizon.

Mutants (removing the label, a unit or a band; planting the word) are all caught. The auditor's first full run found two real gaps, units missing from the policy objectives and from the threshold-crossing years of the report, and both were fixed at their source.

**W12.**
- Every data source is registered with its licence and mark.
- Every data file carries a SHA-256 in a manifest (Atlas, WPP, energy, climate, food, geometry).
- No third-party data travel inside the package. The Vensim reference stays in the repo's `tests/`; installed without it, W1 is BLOCKED, never green.
- Licences opened in a session are `[V]`; for example, EMA Workbench is BSD-3, read from its LICENSE on 2026-10-13.

<a name="11"></a>
## 11. The acceptance campaign

`world4_cli.py --campaign` scores W1-W12 with `phycore.vvuq.Campaign`. A criterion with no data behind it is BLOCKED, never green. As of F9, from a checkout with all the data:

| Verdict | Criteria |
|---|---|
| GREEN | W1 (P1; P2 shown, fails by one cell), W2, W5, W6, W10, W11 |
| AMBER | W3, W7, W12 |
| RED | **W4** (calibrated hindcast: population, energy, CO₂, cropland), **W8** (PIT shape), **W9a** (democracy; ViEWS/PITF not reported) |
| BLOCKED | W9b (needs months of resolved questions) |

**The release gate stays BLOCKED while there is a red,** and the causes are named.

<a name="12"></a>
## 12. What this document does not contain, and why

- **Numbers from the 2026-10-10 research reports.** They were produced by agents with several unverified sources. None enters the model without its primary source being opened.
- **Nebel et al. (2023)'s recalibrated World3 parameters:** the paper's table was never opened, so World3-03 is not calibrated here.
- **Kotz, Levermann and Wenz (2024):** retracted (Nature, 3 December 2025); it is not used.
- **Richardson et al. (2023) planetary boundaries:** non-commercial licence. F8's limits are relative and declared as [M] instead.
- **The size of each policy effect:** chosen, not estimated. Its uncertainty is in the ensemble; its form is not.

<a name="13"></a>
## 13. How to check all of this

```bash
python world4_cli.py --campaign                  # W1-W12, measured live where it can be
python -m pytest -q                              # unit, parity, mutant and honesty tests
python world4_cli.py --validate                  # W3/W4/W5 runs (~50 min)
python world4_cli.py --calibrate                 # F7 posterior, calibrated W4 hindcast, W8 (~40 min)
python world4_cli.py --design                    # F8 fronts, PRIM, regret, conduct (~10 min)
python world4_cli.py --report runs/report.html   # this evidence in one self-contained HTML file
python -m world4.validate.w10                    # performance on this machine
```
