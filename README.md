# Econometrics Game 2026

Team research project · second place

Two rounds of applied causal analysis: heterogeneous effects of performance-based financial aid in the preliminary round, followed by the relationship between OxyContin reformulation and fertility in the final round. The repository brings together Stata estimation scripts, Python data preparation and analysis, and a saved aggregate results summary.

This was collaborative competition work. The code and analysis are presented as team project materials, without assigning sole authorship.

## What connects the two rounds?

Both rounds ask how an intervention changes behaviour, but the available sources of identification are different. The preliminary round starts with randomised programme offers and asks whether the response varies with students' starting points. The final round starts with a nationwide policy shock and asks whether places with different prior exposure subsequently diverged. The project illustrates how an empirical strategy has to follow the assignment mechanism and the available data.

### Financial incentives and heterogeneous responses

An average programme effect can hide differences between students. Financial incentives might change GPA differently for students with different baseline achievement, which matters for understanding whom a programme reaches and how it could be targeted. The preliminary analysis studies these differences within the STAR financial-incentive programme.

The code distinguishes the effect of being offered a programme from the effect of participating in it. An intent-to-treat regression measures the former. When take-up is incomplete, an instrumental-variable analysis uses the randomised offer to isolate variation in participation induced by that offer. Interactions and subgroup linear combinations then examine how these estimates differ across baseline achievement groups. Interpreting those participation effects requires the additional assumptions of the instrumental-variable design; randomisation of an offer does not make actual participation random.

### OxyContin reformulation and fertility

The final round asks whether a change in the opioid environment was followed by changes in fertility, and whether the pattern differed across age groups. The motivating question extends beyond a direct drug-market outcome: a health and substance-use shock may also be connected with decisions and circumstances around family formation. Fertility counts alone cannot establish which mechanisms are responsible.

Because the 2010 reformulation was national, the design does not use an entirely untreated set of states. Instead, it compares changes in states with greater and smaller baseline exposure. The continuous-exposure difference-in-differences coefficient asks whether fertility changed more with each additional unit of pre-reformulation exposure. State effects absorb stable differences between states, while year effects absorb common national changes. Event studies examine the timing of that relationship; sensitivity specifications ask how dependent it is on controls, weighting, exposure definitions and assumptions about trends.

The data work is central to that comparison. Birth counts need population denominators before they can be compared across states or ages, and the available inputs require proxy denominators. The repository therefore documents how panels and outcomes are constructed as well as how regressions are fitted.

## Research questions and methods

| Round | Question | Implementation |
| --- | --- | --- |
| Preliminary | Does participation in the STAR financial-incentive programme affect GPA differently across baseline achievement groups? | Stata first stages, intent-to-treat regressions, interacted 2SLS, subgroup linear combinations and heterogeneity tests. |
| Final | Did fertility change more after the 2010 OxyContin reformulation in states with greater baseline exposure? | Python state/year fixed effects, continuous-exposure DiD, weighted specifications, event studies, age heterogeneity and sensitivity analyses; a separate Stata HonestDiD script. |

The preliminary design uses randomised programme offers as instruments for participation. The final design combines state exposure and control variables with natality counts; its fertility denominators are population proxies, including approximations for maternal age groups.

## Code map

```text
preliminary/
  regression_pipeline.do
  README.md
final/
  honestdid_standalone.do
  opioid_fertility_oxycontin/
    build_master_from_authors.py
    build_state_opioid_dataset.py
    run_master_analysis.py
    build_paper_bundle.py
    build_motivation_dashboard.py
docs/
  data-and-workflow.md
results/
  reported-findings.md
```

The Python workflow constructs the main state-year and state-year-age panels, estimates models, and creates tables and figures. The paper-bundle script adds alternative exposures, placebo timing, leave-one-state-out checks, exposure-permutation diagnostics and a LaTeX report. The dashboard script creates an HTML view of the state-level data.

## Final-round workflow

Use Python 3.10 or later. Install the dependencies from the repository root, then run the scripts **from `final/`**. They resolve their data and output paths relative to that working directory.

```bash
python -m pip install -r requirements.txt
cd final
python opioid_fertility_oxycontin/build_master_from_authors.py
python opioid_fertility_oxycontin/run_master_analysis.py
```

Before running, supply the author data and natality files described in the [data and workflow guide](docs/data-and-workflow.md). For the additional proxy-exposure tables and paper bundle:

```bash
# Still working from final/
python opioid_fertility_oxycontin/build_state_opioid_dataset.py
python opioid_fertility_oxycontin/build_paper_bundle.py
python opioid_fertility_oxycontin/build_motivation_dashboard.py
```

The proxy builder additionally requires the saved SAMHSA HTML table. Compiling the generated `.tex` report requires a LaTeX installation. The HTML dashboard loads D3, TopoJSON and map geometry from external CDNs when viewed.

For Stata sensitivity analysis, set the working directory to the generated processed-data directory so the existing script can locate its input:

```stata
* Starting from the repository root:
cd "final/opioid_fertility_oxycontin/data/processed"
do "../../../honestdid_standalone.do"
```

The Stata script contains package-installation commands for `honestdid`, `reghdfe` and `ftools`. It runs both unweighted and population-weighted event-study sensitivity specifications and writes figures into its working directory.

The [preliminary-round guide](preliminary/README.md) documents its input columns and Stata entry point.

## Reading the results

[Reported findings](results/reported-findings.md) preserves the saved master-analysis summary and identifies its scope. The broad full-control specifications in that report do not establish a precise aggregate fertility effect; selected age-group estimates differ. The paper-bundle models and standalone Stata specification use different choices and should be read separately.

The final-round outcome is a proxy fertility rate. Mechanism outcomes were unavailable in the saved merged sample. The annual Python post indicator begins in 2011, while the standalone Stata event study includes 2010 as event time zero and uses 2009 as its reference. These details matter when comparing specifications.

Input datasets, student records, generated analysis directories and submission PDFs are not distributed in this repository. Obtain source data under their applicable access and reuse terms.
