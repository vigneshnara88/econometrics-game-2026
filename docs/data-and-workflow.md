# Data and workflow

## Directory convention

The Python scripts use `ROOT = Path("opioid_fertility_oxycontin")`. Run them from `final/`, not from the repository root or from inside the Python directory. The author-data input resolves to the project-local `data/raw/oxycontindata_final.dta` beneath that root.

Create this private input structure before running:

```text
final/opioid_fertility_oxycontin/data/raw/
  oxycontindata_final.dta
  wonder/
    natality_state_year_births_2007_2016.csv
    natality_state_year_age_births_2007_2016.csv
  samhsa/
    pain_reliever_state_report.html
```

The scripts read local files; they do not download the datasets.

## Inputs

| Input | Role and expected structure |
| --- | --- |
| `oxycontindata_final.dta` | Author-curated state-year treatment and control data associated with the OxyContin reformulation literature. The loader uses `state`, `stfips`, `year`, baseline exposure fields such as `initial_oxy`, `initial_pr`, `oxyabuse2008`, `prabuse2008`, population fields and available demographic, economic and policy controls. Preserve the supplied names and definitions. |
| State-year natality CSV | CDC WONDER export with `State`, `State Code`, `Year`, `Births`, and optionally `Notes`. Non-empty notes rows are excluded by the loaders. |
| State-year-age natality CSV | Same fields plus `Age of Mother 9`; the proxy-dataset builder also expects `Age of Mother 9 Code`. Preserve source age labels. |
| `pain_reliever_state_report.html` | Saved SAMHSA state report containing the NSDUH prescription-pain-reliever-use table. The builder selects the fifth HTML table and expects its original nine-column structure. An arbitrary replacement page is not interchangeable. |

The author-data overlap sets the main panel to 2007–2013. The natality exports and separate proxy panel cover 2007–2016. Available controls vary with the supplied author file; the scripts retain or select columns according to their existing logic.

The SAMHSA measure is a broader pain-reliever misuse proxy, not the same treatment as baseline OxyContin misuse. Its 2009–2010 period overlaps the reformulation year; interpret it as an alternative exposure measure with that timing limitation.

## Pipeline stages

| Script | Reads | Produces |
| --- | --- | --- |
| `build_master_from_authors.py` | Author Stata file and both natality exports | Main state-year and state-year-age CSVs, derived controls, fertility proxies and a dataset brief under `data/processed/`. |
| `run_master_analysis.py` | Both main processed panels | Model tables, event studies, plots and an analysis summary under `analysis_outputs_master/`. |
| `build_state_opioid_dataset.py` | SAMHSA HTML and both natality exports | State exposure-proxy table and extended birth panels under `data/processed/`. |
| `build_paper_bundle.py` | Main processed panels plus the exposure-proxy table | Additional specifications, diagnostics, figures, LaTeX tables and report source under `paper_results_bundle/`. |
| `build_motivation_dashboard.py` | Main state-year processed panel | `paper_results_bundle/motivation_dashboard.html`. Run after the paper builder has created the output directory. |
| `honestdid_standalone.do` | `master_state_year_2007_2013.csv` in Stata's working directory | Event-study estimates, sensitivity output and two PNG figures in that working directory. |

The Python analysis scripts can create or overwrite their generated outputs. The saved aggregate summary under `results/` is separate from those generated directories.

## Interpretation details

The reformulation design uses a national shock with varying pre-period state exposure. It is not a comparison between entirely untreated and treated states. Causal interpretation depends on the continuous-exposure parallel-trends assumptions and the adequacy of the model specification.

The fertility proxy multiplies the female population by broad state age shares to approximate the population of women aged 15–44. Age-specific denominators also allocate broad age groups into narrower maternal-age bins. These quantities should not be presented as directly observed age-specific female population counts.

The master-analysis script, paper-bundle script and standalone Stata script are separate analytical specifications. In particular, the paper-bundle analysis excludes 2010 for its baseline sample, and the Stata event study includes a 2010 interaction. Package availability and numerical output from one script do not establish agreement with another.

The existing report-writing code includes interpretive prose around computed tables. Treat the generated report as analytical output requiring substantive review when changing inputs or specifications.

## Data handling

Raw inputs and generated person-level or aggregate extracts belong in ignored data/output directories. Neither the competition student dataset nor the author Stata dataset is bundled. The included results document contains aggregate coefficients and method notes only.
