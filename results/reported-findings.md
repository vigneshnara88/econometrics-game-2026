# Reported findings: master analysis

Source: the project's saved `analysis_outputs_master/analysis_summary.md`. The numerical results and analysis notes below preserve that report's contents. They describe the master-analysis specification, not every specification in the separate paper-bundle or Stata scripts.

The report's closing note about HonestDiD refers to that output pack. A separate sensitivity-analysis script is included at `final/honestdid_standalone.do`; no corresponding sensitivity results are included here.

## Saved analysis summary

Framework:
- Continuous-exposure DiD following Alpert, Powell, and Pacula (2018), using pre-reformulation OxyContin misuse intensity interacted with a post-2010 indicator.
- State and year fixed effects in all main specifications.
- Main outcome: `fertility_rate_proxy_per_1000_women_15_44`.
- Age analysis outcome: `age_specific_fertility_rate_proxy_per_1000`.
- Age-specific fertility rates use proxy female age denominators derived from the authors' state-year age shares.

Citations:
- Alpert, Abby, David Powell, and Rosalie Liccardo Pacula. 2018. "Supply-Side Drug Policy in the Presence of Substitutes: Evidence from the Introduction of Abuse-Deterrent Opioids." *AEJ: Economic Policy* 10(4): 1-35. DOI: 10.1257/pol.20170082
- Caudillo, Mónica L., and Andrés Villarreal. 2021. "The Opioid Epidemic and Nonmarital Childbearing in the United States, 2000-2016." *Demography* 58(1): 345-378. DOI: 10.1215/00703370-8937348
- Powell, David, and Rosalie Liccardo Pacula. 2021. "The Evolving Consequences of OxyContin Reformulation on Drug Overdoses." *American Journal of Health Economics* 7(1): 41-67. DOI: 10.1086/711723
- Rambachan, Ashesh, and Jonathan Roth. 2023. "A More Credible Approach to Parallel Trends." *Review of Economic Studies* 90(5): 2555-2591.

What was estimated:
- Unweighted and population-weighted DiD regressions.
- A full-control specification with demographic, macro, policy, and opioid-environment controls.
- A state-specific linear trend robustness specification.
- Event-study plots with a joint pre-trend test.
- Mechanism regressions on heroin and opioid overdose outcomes.

Best main results from the full state-year specifications:
- `initial_pr_post` | spec `full` | weighted `0` | coef `-0.2642` | se `0.1739` | p `0.1286`
- `initial_oxy_post` | spec `full` | weighted `0` | coef `1.3650` | se `1.1770` | p `0.2462`
- `oxyabuse2008_post` | spec `full` | weighted `0` | coef `0.6701` | se `0.6827` | p `0.3263`
- `prabuse2008_post` | spec `full` | weighted `0` | coef `-0.1298` | se `0.1746` | p `0.4575`

Most informative age-share results:
- `15-19 years` with `initial_oxy_post` | coef `6.688279` | se `3.096802` | p `0.0308`
- `15-19 years` with `oxyabuse2008_post` | coef `3.791385` | se `1.853965` | p `0.0409`
- `35-44 years` with `prabuse2008_post` | coef `0.078786` | se `0.041181` | p `0.0557`
- `20-24 years` with `initial_pr_post` | coef `-1.401633` | se `0.880241` | p `0.1113`
- `35-44 years` with `initial_pr_post` | coef `0.059099` | se `0.038088` | p `0.1208`
- `35-44 years` with `oxyabuse2008_post` | coef `0.235727` | se `0.163170` | p `0.1486`
- `15-19 years` with `initial_pr_post` | coef `-0.717898` | se `0.546750` | p `0.1892`
- `25-29 years` with `initial_pr_post` | coef `-0.632753` | se `0.501842` | p `0.2074`

Mechanism / first-stage style results:
- Mechanism outcomes were not populated in the merged 2007-2013 sample, so this first-stage block still needs an external overdose merge or a refreshed author file.

Modern metrics note:
- This setting is naturally a continuous-treatment DiD, not synthetic control, because every state is exposed to the same national reformulation shock but with different pre-period intensity.
- The event-study plus pre-trend test is implemented here.
- A Rambachan-Roth / HonestDiD sensitivity analysis would be a strong appendix extension, but it is not implemented in this output pack.
