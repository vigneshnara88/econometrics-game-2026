# Preliminary round: financial incentives and academic achievement

The STAR setting combines randomised offers with voluntary take-up. The analysis examines whether the effect of participation differs across three baseline academic-achievement groups, using offers and offer-by-group interactions as instruments.

`regression_pipeline.do` contains the existing estimation sequence: descriptive summaries, offer balance, first stages, reduced-form GPA models, interacted 2SLS for year-one and year-two GPA, subgroup linear combinations and tests of differences between groups.

## Input and execution

The script expects the prepared dataset to be loaded in Stata. It does not import, clean or construct the analysis variables. Place the private prepared CSV at `preliminary/data/final_clean_dataset.csv` and, from the repository root, use:

```stata
import delimited "preliminary/data/final_clean_dataset.csv", clear
do "preliminary/regression_pipeline.do"
```

The required prepared fields are:

| Fields | Role |
| --- | --- |
| `gpa0`, `gpa_year1`, `gpa_year2` | Baseline and follow-up academic outcomes. |
| `d`, `z` | Participation and randomised offer indicators. |
| `hsgroup`, `g2`, `g3` | Baseline group and indicators for groups two and three. |
| `z_g2`, `z_g3`, `d_g2`, `d_g3` | Offer and participation interacted with the group indicators. |
| `female`, `english`, `numcourses_nov1`, `lm_never`, `lm_rarely`, `mom1`, `mom2`, `dad1`, `dad2` | Prepared covariates used in the model control list. |
| `sex`, `mtongue`, `lastmin`, `mom_edn`, `dad_edn` | Additional fields referenced in the opening summaries. |

Use the preparation definitions associated with the competition analysis. In particular, the group boundaries and category reference levels cannot be recovered simply from the column names. Do not substitute newly chosen percentile cutoffs without documenting the resulting change in the sample and estimand.

## Specification and scope

The first-stage and reduced-form commands include the baseline group indicators. The supplied 2SLS commands include the control list and the three endogenous participation terms but do not separately list `g2` and `g3` as exogenous regressors. This differs from the group-intercept specification written in the team report; the script is retained as supplied, so its estimates should be described using the actual commands.

The input was prepared with outcome and data-quality exclusions, including removal of zero year-one GPA observations. Those exclusions can affect interpretation when missing or zero outcomes relate to treatment or withdrawal. Random assignment of an offer alone does not resolve outcome selection.

Interpreting IV coefficients as complier effects additionally requires relevance, exclusion and monotonicity. Subgroup linear combinations describe effects under that interacted specification; they are not effects for every student in a baseline group.

This folder contains estimation code only. Student-level records and submission documents are not included.
