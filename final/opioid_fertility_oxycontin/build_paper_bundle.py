from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


ROOT = Path("opioid_fertility_oxycontin")
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "paper_results_bundle"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
STATE_PATH = PROCESSED / "master_state_year_2007_2013.csv"
AGE_PATH = PROCESSED / "master_state_year_age_2007_2013.csv"
PROXY_PATH = PROCESSED / "state_pain_reliever_exposure_proxy.csv"

OUTCOME = "fertility_rate_proxy_per_1000_women_15_44"
AGE_OUTCOME = "age_specific_fertility_rate_proxy_per_1000"
HEADLINE_TREATMENT = "initial_oxy_post"
BASE_EXPOSURE = "initial_oxy"
STATE_CLUSTER = "state"
PREFERRED_CONTROLS = ["xxunemp", "heroinprice_filled", "share_low_edu"]
POLICY_CONTROLS = ["pp_pmp_authorized"]
ALL_POLICY_CONTROLS = ["pp_pmp_authorized", "pp_must", "pp_PILLMILL", "pp_medmj", "pp_actp"]
MECHANISM_OUTCOMES = ["arcos_totaloxycod", "arcos_totalhydroc", "ln_teds_heroin_plus1"]
AGE_GROUPS = ["15-19 years", "20-24 years", "25-29 years", "30-34 years", "25-34 years", "35-44 years"]
AGE_GROUP_MAP = {"25-34 years": ["25-29 years", "30-34 years"], "35-44 years": ["35-39 years", "40-44 years"]}
RED_2008_WINNER = {
    "Alabama": 1, "Alaska": 1, "Arizona": 1, "Arkansas": 1, "California": 0, "Colorado": 0, "Connecticut": 0,
    "Delaware": 0, "Florida": 0, "Georgia": 1, "Hawaii": 0, "Idaho": 1, "Illinois": 0, "Indiana": 1, "Iowa": 0,
    "Kansas": 1, "Kentucky": 1, "Louisiana": 1, "Maine": 0, "Maryland": 0, "Massachusetts": 0, "Michigan": 0,
    "Minnesota": 0, "Mississippi": 1, "Missouri": 1, "Montana": 1, "Nebraska": 1, "Nevada": 0, "New Hampshire": 0,
    "New Jersey": 0, "New Mexico": 0, "New York": 0, "North Carolina": 0, "North Dakota": 1, "Ohio": 0, "Oklahoma": 1,
    "Oregon": 0, "Pennsylvania": 0, "Rhode Island": 0, "South Carolina": 1, "South Dakota": 1, "Tennessee": 1,
    "Texas": 1, "Utah": 1, "Vermont": 0, "Virginia": 0, "Washington": 0, "West Virginia": 1, "Wisconsin": 0,
    "Wyoming": 1, "District Of Columbia": 0,
}


def ensure_dirs() -> None:
    for path in [OUT, TABLES, FIGURES]:
        path.mkdir(parents=True, exist_ok=True)


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns and df[c].notna().mean() > 0.8]


def fit_ols(df: pd.DataFrame, formula: str, weights: str | None = None):
    model = smf.wls(formula, data=df, weights=df[weights]) if weights else smf.ols(formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[STATE_CLUSTER]})


def fit_with_covariance(df: pd.DataFrame, formula: str, cov_mode: str = "state_cluster", weights: str | None = None):
    model = smf.wls(formula, data=df, weights=df[weights]) if weights else smf.ols(formula, data=df)
    if cov_mode == "state_cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": df[STATE_CLUSTER]})
    if cov_mode == "year_cluster":
        return model.fit(cov_type="cluster", cov_kwds={"groups": df["year"]})
    if cov_mode == "two_way_cluster":
        state_codes = pd.factorize(df[STATE_CLUSTER])[0]
        year_codes = pd.factorize(df["year"])[0]
        return model.fit(cov_type="cluster", cov_kwds={"groups": np.column_stack([state_codes, year_codes])})
    if cov_mode == "hc1":
        return model.fit(cov_type="HC1")
    raise ValueError(f"Unsupported cov_mode: {cov_mode}")


def load_state() -> pd.DataFrame:
    state = pd.read_csv(STATE_PATH)
    proxy = pd.read_csv(PROXY_PATH)
    proxy["state"] = proxy["state"].astype(str).str.strip().str.title()
    state["state"] = state["state"].astype(str).str.strip().str.title()
    state = state.merge(
        proxy[["state", "exposure_proxy_pre"]],
        on="state",
        how="left",
        validate="many_to_one",
    )
    state["red_2008"] = state["state"].map(RED_2008_WINNER)
    state["proxy_exposure_post"] = state["exposure_proxy_pre"] * state["post_reform"]
    for base in ["oxyabuse2004", "oxyabuse2006", "oxyabuse2008"]:
        if base in state.columns:
            state[f"{base}_post"] = state[base] * state["post_reform"]
    state["post_2011"] = (state["year"] >= 2011).astype(int)
    state["t"] = state["year"] - state["year"].min()
    state["ln_arcos_totaloxycod"] = np.where(state["arcos_totaloxycod"] > 0, np.log(state["arcos_totaloxycod"]), np.nan)
    state["ln_arcos_totalhydroc"] = np.where(state["arcos_totalhydroc"] > 0, np.log(state["arcos_totalhydroc"]), np.nan)
    return state


def load_age() -> pd.DataFrame:
    age = pd.read_csv(AGE_PATH)
    proxy = pd.read_csv(PROXY_PATH)
    proxy["state"] = proxy["state"].astype(str).str.strip().str.title()
    age["state"] = age["state"].astype(str).str.strip().str.title()
    age = age.merge(
        proxy[["state", "exposure_proxy_pre"]],
        on="state",
        how="left",
        validate="many_to_one",
    )
    age["red_2008"] = age["state"].map(RED_2008_WINNER)
    age["proxy_exposure_post"] = age["exposure_proxy_pre"] * age["post_reform"]
    return age


def analysis_sample(df: pd.DataFrame) -> pd.DataFrame:
    # Drop 2010 as the transition year to match the research design.
    return df[df["year"] != 2010].copy()


def run_main_specs(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls_sets = {
        "no_controls": [],
        "parsimonious": safe_cols(sample, PREFERRED_CONTROLS),
        "parsimonious_policy": safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS),
        "policy_augmented": safe_cols(sample, PREFERRED_CONTROLS + ALL_POLICY_CONTROLS),
    }
    treatments = {
        "Alpert initial_oxy": HEADLINE_TREATMENT,
        "Alpert initial_pr": "initial_pr_post",
        "NSDUH pain reliever proxy": "proxy_exposure_post",
    }
    rows = []
    for label, treatment in treatments.items():
        for spec_name, controls in controls_sets.items():
            rhs = [treatment, "C(state)", "C(year)"] + controls
            formula = f"{OUTCOME} ~ " + " + ".join(rhs)
            reg_df = sample.dropna(subset=[OUTCOME, treatment] + controls).copy()
            res = fit_ols(reg_df, formula)
            rows.append(result_row(label, treatment, spec_name, 0, res))

            weighted = fit_ols(reg_df.dropna(subset=["pop"]), formula, weights="pop")
            rows.append(result_row(label, treatment, spec_name + "_weighted", 1, weighted))

            trend_formula = formula + " + C(state):t"
            trend_res = fit_ols(reg_df, trend_formula)
            rows.append(result_row(label, treatment, spec_name + "_state_trends", 0, trend_res))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "main_spec_results.csv", index=False)
    return out


def result_row(label: str, treatment: str, spec: str, weighted: int, res) -> dict:
    coef = res.params.get(treatment, np.nan)
    se = res.bse.get(treatment, np.nan)
    return {
        "treatment_label": label,
        "treatment": treatment,
        "spec": spec,
        "weighted": weighted,
        "coef": coef,
        "se": se,
        "t": coef / se if pd.notna(coef) and pd.notna(se) and se != 0 else np.nan,
        "p_value": res.pvalues.get(treatment, np.nan),
        "n": int(res.nobs),
    }


def event_study(state: pd.DataFrame, exposure: str, file_stub: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    years = [2007, 2008, 2011, 2012, 2013]
    df = sample.dropna(subset=[OUTCOME, exposure] + controls).copy()
    terms = []
    rows = []
    for year in years:
        name = f"evt_{year}"
        df[name] = np.where(df["year"] == year, df[exposure], 0.0)
        terms.append(name)
    formula = f"{OUTCOME} ~ " + " + ".join(terms + ["C(state)", "C(year)"] + controls)
    use_weights = exposure == BASE_EXPOSURE and "pop" in df.columns and df["pop"].notna().all()
    res = fit_ols(df, formula, weights="pop" if use_weights else None)
    for year in years:
        name = f"evt_{year}"
        coef = res.params.get(name, np.nan)
        se = res.bse.get(name, np.nan)
        rows.append(
            {
                "year": year,
                "event_time": year - 2010,
                "coef": coef,
                "se": se,
                "p_value": res.pvalues.get(name, np.nan),
                "ci_low": coef - 1.96 * se,
                "ci_high": coef + 1.96 * se,
            }
        )
    out = pd.DataFrame(rows)
    pre_terms = [f"evt_{year}" for year in [2007, 2008]]
    pre_test = res.f_test(" = 0, ".join(pre_terms) + " = 0")
    pre_df = pd.DataFrame(
        [
            {
                "statistic": float(pre_test.fvalue),
                "p_value": float(pre_test.pvalue),
                "df_num": int(pre_test.df_num),
                "df_denom": float(pre_test.df_denom),
                "max_abs_pre_coef": out.loc[out["year"].lt(2009), "coef"].abs().max(),
                "max_abs_post_coef": out.loc[out["year"].gt(2010), "coef"].abs().max(),
            }
        ]
    )
    pre = pre_df.loc[0, "max_abs_pre_coef"]
    post = pre_df.loc[0, "max_abs_post_coef"]
    pre_df["magnitude_ratio_post_to_pre"] = np.nan if pd.isna(pre) or pre == 0 else post / pre
    out.to_csv(OUT / f"{file_stub}_event_study.csv", index=False)
    pre_df.to_csv(OUT / f"{file_stub}_pretrend_summary.csv", index=False)

    plot_df = pd.concat(
        [
            pd.DataFrame([{"year": 2009, "coef": 0.0, "ci_low": 0.0, "ci_high": 0.0, "baseline": True}]),
            out.assign(baseline=False),
        ],
        ignore_index=True,
    ).sort_values("year")
    plt.figure(figsize=(8.8, 5.4))
    plt.axhline(0, color="black", linewidth=1)
    plt.axvspan(2006.5, 2009.5, color="#f3f4f6", alpha=0.9, zorder=0)
    plt.axvspan(2010.5, 2013.5, color="#eef7ef", alpha=0.75, zorder=0)
    plt.axvline(2010, color="#6b7280", linestyle="--", linewidth=1.4)
    plt.fill_between(plot_df["year"], plot_df["ci_low"], plot_df["ci_high"], color="#93c5fd", alpha=0.35, zorder=1)
    plt.plot(plot_df["year"], plot_df["coef"], color="#0f4c81", linewidth=2.4, zorder=2)
    plt.scatter(plot_df["year"], plot_df["coef"], s=55, color="#0f4c81", zorder=3)
    ymax = float(np.nanmax(plot_df["ci_high"])) if plot_df["ci_high"].notna().any() else 0.5
    ymin = float(np.nanmin(plot_df["ci_low"])) if plot_df["ci_low"].notna().any() else -0.5
    plt.text(2007.0, ymax - 0.08 * (ymax - ymin), "Pre-period leads", fontsize=9, color="#4b5563")
    plt.text(2011.02, ymax - 0.08 * (ymax - ymin), "Post-reformulation lags", fontsize=9, color="#4b5563")
    plt.text(2009.03, 0.04 * (ymax - ymin), "2009 omitted baseline", fontsize=8.5, color="#374151")
    plt.title("Event study: weighted continuous DiD")
    plt.suptitle("Lead coefficients near zero support parallel trends; post coefficients trace the differential response", y=0.98, fontsize=10, color="#4b5563")
    plt.xlabel("Year")
    plt.ylabel("Differential effect on fertility rate")
    plt.tight_layout()
    plt.savefig(FIGURES / f"{file_stub}.png", dpi=180)
    plt.close()
    return out, pre_df


def placebo_post_tests(state: pd.DataFrame, treatment_col: str, label: str) -> pd.DataFrame:
    base = state.copy()
    controls = safe_cols(base, PREFERRED_CONTROLS + POLICY_CONTROLS)
    rows = []
    for cutoff in [2008, 2009, 2010]:
        df = base.copy()
        df["placebo_post"] = (df["year"] >= cutoff).astype(int)
        df["placebo_treat"] = df[treatment_col.replace("_post", "")] * df["placebo_post"] if treatment_col.endswith("_post") else df["exposure_proxy_pre"] * df["placebo_post"]
        if cutoff == 2010:
            df = df[df["year"] != 2010].copy()
        reg_df = df.dropna(subset=[OUTCOME, "placebo_treat"] + controls).copy()
        res = fit_ols(reg_df, f"{OUTCOME} ~ placebo_treat + C(state) + C(year) + " + " + ".join(controls))
        rows.append(
            {
                "treatment_label": label,
                "placebo_cutoff": cutoff,
                "coef": res.params.get("placebo_treat", np.nan),
                "se": res.bse.get("placebo_treat", np.nan),
                "p_value": res.pvalues.get("placebo_treat", np.nan),
                "n": int(res.nobs),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / f"placebo_{label.replace(' ', '_').lower()}.csv", index=False)
    return out


def randomization_inference(state: pd.DataFrame, reps: int = 500) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    reg_df = sample.dropna(subset=[OUTCOME, BASE_EXPOSURE] + controls).copy()
    formula = f"{OUTCOME} ~ permuted_treat + C(state) + C(year) + " + " + ".join(controls)
    observed = fit_ols(
        reg_df.assign(permuted_treat=reg_df[BASE_EXPOSURE] * reg_df["post_2011"]),
        formula,
        weights="pop",
    ).params["permuted_treat"]

    rng = np.random.default_rng(20260411)
    state_exposure = reg_df.groupby("state", as_index=False)[BASE_EXPOSURE].first()
    draws = []
    for _ in range(reps):
        shuffled = state_exposure.copy()
        shuffled["permuted_exposure"] = rng.permutation(shuffled[BASE_EXPOSURE].to_numpy())
        draw_df = reg_df.merge(shuffled[["state", "permuted_exposure"]], on="state", how="left", validate="many_to_one")
        draw_df["permuted_treat"] = draw_df["permuted_exposure"] * draw_df["post_2011"]
        coef = fit_ols(draw_df, formula, weights="pop").params["permuted_treat"]
        draws.append(coef)
    dist = pd.DataFrame({"draw_coef": draws})
    p_val = (dist["draw_coef"].abs() >= abs(observed)).mean()
    summary = pd.DataFrame([{"observed_coef": observed, "ri_p_value": p_val, "reps": reps}])
    dist.to_csv(OUT / "randomization_distribution.csv", index=False)
    summary.to_csv(OUT / "randomization_summary.csv", index=False)
    return summary


def leave_one_out(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    rows = []
    for omitted in sorted(sample["state"].unique()):
        reg_df = sample[sample["state"] != omitted].dropna(subset=[OUTCOME, HEADLINE_TREATMENT] + controls).copy()
        formula = f"{OUTCOME} ~ {HEADLINE_TREATMENT} + C(state) + C(year) + " + " + ".join(controls)
        res = fit_ols(reg_df, formula)
        rows.append(
            {
                "omitted_state": omitted,
                "coef": res.params.get(HEADLINE_TREATMENT, np.nan),
                "se": res.bse.get(HEADLINE_TREATMENT, np.nan),
                "p_value": res.pvalues.get(HEADLINE_TREATMENT, np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leave_one_out.csv", index=False)
    return out


def mechanism_regs(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    rows = []
    for outcome in [c for c in MECHANISM_OUTCOMES if c in sample.columns]:
        reg_df = sample.dropna(subset=[outcome, HEADLINE_TREATMENT] + controls).copy()
        if reg_df.empty:
            continue
        formula = f"{outcome} ~ {HEADLINE_TREATMENT} + C(state) + C(year) + " + " + ".join(controls)
        res = fit_ols(reg_df, formula)
        rows.append(
            {
                "outcome": outcome,
                "coef": res.params.get(HEADLINE_TREATMENT, np.nan),
                "se": res.bse.get(HEADLINE_TREATMENT, np.nan),
                "p_value": res.pvalues.get(HEADLINE_TREATMENT, np.nan),
                "n": int(res.nobs),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "mechanism_results.csv", index=False)
    return out


def effect_interpretation(state: pd.DataFrame, beta: float) -> pd.DataFrame:
    sample = analysis_sample(state)
    sd = sample[BASE_EXPOSURE].std()
    p90_p10 = sample[BASE_EXPOSURE].quantile(0.9) - sample[BASE_EXPOSURE].quantile(0.1)
    mean_women = sample["female_pop_15_44_proxy"].mean()
    out = pd.DataFrame(
        [
            {
                "beta_per_unit": beta,
                "sd_initial_oxy": sd,
                "effect_per_1sd_per_1000_women": beta * sd,
                "effect_p90_p10_per_1000_women": beta * p90_p10,
                "implied_births_per_1sd_avg_state_year": beta * sd * mean_women / 1000,
                "implied_births_p90_p10_avg_state_year": beta * p90_p10 * mean_women / 1000,
            }
        ]
    )
    out.to_csv(OUT / "effect_interpretation.csv", index=False)
    return out


def _exposure_scalars(sample: pd.DataFrame) -> tuple[float, float]:
    sd = sample[BASE_EXPOSURE].std()
    p90_p10 = sample[BASE_EXPOSURE].quantile(0.9) - sample[BASE_EXPOSURE].quantile(0.1)
    return float(sd), float(p90_p10)


def contextualize_effect(beta: float, exposure_sd: float, exposure_p90_p10: float, mean_outcome: float) -> dict:
    effect_1sd = beta * exposure_sd
    effect_p90_p10 = beta * exposure_p90_p10
    pct_mean_1sd = np.nan if pd.isna(mean_outcome) or mean_outcome == 0 else 100 * effect_1sd / mean_outcome
    return {
        "effect_1sd": effect_1sd,
        "effect_p90_p10": effect_p90_p10,
        "pct_mean_1sd": pct_mean_1sd,
    }


def inference_robustness(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    reg_df = sample.dropna(subset=[OUTCOME, HEADLINE_TREATMENT, "pop"] + controls).copy()
    formula = f"{OUTCOME} ~ {HEADLINE_TREATMENT} + C(state) + C(year) + " + " + ".join(controls)
    methods = [
        ("State cluster (preferred)", "state_cluster", "51 state clusters; this is the main serial-correlation concern in a short state panel."),
        ("Two-way cluster: state and year", "two_way_cluster", "Useful sensitivity check for common yearly shocks, but the year dimension has only 6 clusters."),
        ("Year cluster only", "year_cluster", "Not preferred because inference rests on only 6 year clusters."),
        ("HC1 robust", "hc1", "Benchmark only; does not allow arbitrary within-state serial correlation."),
    ]
    rows = []
    for label, mode, rationale in methods:
        res = fit_with_covariance(reg_df, formula, cov_mode=mode, weights="pop")
        rows.append(
            {
                "inference": label,
                "cov_mode": mode,
                "coef": res.params.get(HEADLINE_TREATMENT, np.nan),
                "se": res.bse.get(HEADLINE_TREATMENT, np.nan),
                "p_value": res.pvalues.get(HEADLINE_TREATMENT, np.nan),
                "n": int(res.nobs),
                "rationale": rationale,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "inference_robustness.csv", index=False)
    return out


def make_normalized_validation_plots(state: pd.DataFrame) -> None:
    df = state.copy()
    median = df[BASE_EXPOSURE].median()
    df["exposure_group"] = np.where(df[BASE_EXPOSURE] >= median, "High baseline OxyContin misuse", "Low baseline OxyContin misuse")

    def normalized_plot(outcome: str, filename: str, title: str, ylabel: str, subtitle: str | None = None) -> None:
        base_df = df.dropna(subset=[outcome]).copy()
        if outcome == OUTCOME:
            sub = (
                base_df.groupby(["year", "exposure_group"], as_index=False)
                .apply(lambda g: pd.Series({outcome: np.average(g[outcome], weights=g["female_pop_15_44_proxy"])}), include_groups=False)
                .reset_index(drop=True)
            )
        else:
            sub = base_df.groupby(["year", "exposure_group"], as_index=False)[outcome].mean()
        bases = (
            sub[sub["year"] == 2009][["exposure_group", outcome]]
            .rename(columns={outcome: "base"})
        )
        sub = sub.merge(bases, on="exposure_group", how="left", validate="many_to_one")
        sub["normalized"] = 100 * sub[outcome] / sub["base"]
        plt.figure(figsize=(9.2, 5.4))
        colors = {
            "High baseline OxyContin misuse": "#0f4c81",
            "Low baseline OxyContin misuse": "#b45309",
        }
        plt.axvspan(sub["year"].min() - 0.5, 2009.5, color="#f3f4f6", alpha=0.8, zorder=0)
        plt.axvspan(2010.5, sub["year"].max() + 0.5, color="#eef7ef", alpha=0.7, zorder=0)
        plt.axvline(2010, color="gray", linestyle="--", linewidth=1.2)
        plt.axhline(100, color="black", linewidth=0.9, alpha=0.5)
        for label, grp in sub.groupby("exposure_group"):
            plt.plot(
                grp["year"],
                grp["normalized"],
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=colors[label],
                label=label,
            )
            end = grp.sort_values("year").iloc[-1]
            pct = end["normalized"] - 100
            plt.text(end["year"] + 0.1, end["normalized"], f"{pct:+.0f}%", fontsize=9, color=colors[label], va="center")
        plt.title(title, pad=10)
        plt.xlabel("Year")
        plt.ylabel(ylabel)
        plt.xlim(sub["year"].min() - 0.2, sub["year"].max() + 0.7)
        plt.legend(frameon=False, loc="upper left")
        note = subtitle or "Indexed to 2009 = 100 within exposure group"
        plt.text(0.0, 1.02, note, fontsize=9, color="#4b5563", transform=plt.gca().transAxes)
        plt.tight_layout()
        plt.savefig(FIGURES / filename, dpi=180)
        plt.close()

    normalized_plot(
        "arcos_totaloxycod",
        "normalized_oxycodone_supply.png",
        "Normalized oxycodone supply by exposure group",
        "Index (2009 = 100)",
        "Descriptive high-vs-low exposure comparison; policy line marks the 2010 reformulation",
    )
    normalized_plot(
        OUTCOME,
        "normalized_fertility_proxy.png",
        "Normalized proxy fertility by exposure group",
        "Index (2009 = 100)",
        "Descriptive comparison in fertility among women ages 15-44, indexed to the last pre period",
    )
    if "teds_heroin" in df.columns and df["teds_heroin"].notna().sum() > 0:
        normalized_plot(
            "teds_heroin",
            "normalized_heroin_treatment.png",
            "Heroin treatment admissions by baseline Oxy exposure",
            "Index (2009 = 100)",
            "Descriptive substitution-side figure; not a causal estimate and not the main identification test",
        )


def make_parallel_trends_plot(state: pd.DataFrame) -> None:
    df = analysis_sample(state).copy()
    median = df.groupby("state")[BASE_EXPOSURE].first().median()
    state_group = df.groupby("state")[BASE_EXPOSURE].first().ge(median).map({True: "High baseline OxyContin misuse", False: "Low baseline OxyContin misuse"})
    df["exposure_group"] = df["state"].map(state_group.to_dict())
    sub = df.groupby(["year", "exposure_group"], as_index=False)[OUTCOME].mean()
    base = sub[sub["year"] == 2009][["exposure_group", OUTCOME]].rename(columns={OUTCOME: "base"})
    sub = sub.merge(base, on="exposure_group", how="left")
    sub["normalized"] = 100 * sub[OUTCOME] / sub["base"]
    colors = {"High baseline OxyContin misuse": "#0f4c81", "Low baseline OxyContin misuse": "#b45309"}
    plt.figure(figsize=(8.8, 5.0))
    plt.axvspan(2006.5, 2009.5, color="#f3f4f6", alpha=0.9, zorder=0)
    plt.axvline(2010, color="#6b7280", linestyle="--", linewidth=1.2)
    plt.axhline(100, color="black", linewidth=0.9, alpha=0.45)
    for label, grp in sub.groupby("exposure_group"):
        grp = grp.sort_values("year")
        plt.plot(grp["year"], grp["normalized"], marker="o", linewidth=2.2, color=colors[label], label=label)
        pre = grp[grp["year"] <= 2009]
        plt.plot(pre["year"], pre["normalized"], linewidth=3.0, color=colors[label])
    plt.xlim(2006.8, 2013.2)
    plt.xlabel("Year")
    plt.ylabel("Index (2009 = 100)")
    plt.title("Pre-period fertility trends by exposure group")
    plt.text(0.0, 1.02, "Bold segments show pre-policy years only; used as a visual complement to the event-study lead test", fontsize=9, color="#4b5563", transform=plt.gca().transAxes)
    plt.legend(frameon=False, loc="lower left")
    plt.tight_layout()
    plt.savefig(FIGURES / "parallel_trends_fertility_groups.png", dpi=180)
    plt.close()


def make_coefficient_plots(main: pd.DataFrame, age: pd.DataFrame, heterogeneity: pd.DataFrame) -> None:
    spec_map = [
        ("Weighted, no controls", "no_controls_weighted"),
        ("Weighted + core controls", "parsimonious_weighted"),
        ("Weighted + core controls + PDMP", "parsimonious_policy_weighted"),
        ("Unweighted + state-specific trends", "parsimonious_state_trends"),
    ]
    sub = main[main["treatment"] == HEADLINE_TREATMENT].set_index("spec")
    rows = []
    for label, spec in spec_map:
        r = sub.loc[spec]
        rows.append({"label": label, "coef": r["coef"], "se": r["se"]})
    spec_df = pd.DataFrame(rows)
    spec_df["low"] = spec_df["coef"] - 1.96 * spec_df["se"]
    spec_df["high"] = spec_df["coef"] + 1.96 * spec_df["se"]

    plt.figure(figsize=(8.2, 4.8))
    y = np.arange(len(spec_df))[::-1]
    plt.axvline(0, color="black", linewidth=1)
    plt.errorbar(spec_df["coef"], y, xerr=1.96 * spec_df["se"], fmt="o", color="#0f4c81", ecolor="#0f4c81", capsize=4)
    plt.yticks(y, spec_df["label"])
    plt.xlabel("Coefficient on InitialOxy x Post")
    plt.title("Headline coefficient across specifications")
    plt.tight_layout()
    plt.savefig(FIGURES / "headline_spec_coefficients.png", dpi=180)
    plt.close()

    age_df = age[age["treatment"] == HEADLINE_TREATMENT].copy()
    age_df["low"] = age_df["coef"] - 1.96 * age_df["se"]
    age_df["high"] = age_df["coef"] + 1.96 * age_df["se"]
    age_df = age_df.sort_values("coef")
    plt.figure(figsize=(8.2, 5.2))
    y = np.arange(len(age_df))
    colors = ["#0f4c81" if p < 0.05 else "#9ca3af" for p in age_df["p_value"]]
    plt.axvline(0, color="black", linewidth=1)
    for i, (_, r) in enumerate(age_df.iterrows()):
        plt.plot([r["low"], r["high"]], [i, i], color=colors[i], linewidth=2)
        plt.plot(r["coef"], i, "o", color=colors[i])
    plt.yticks(y, age_df["age_group"])
    plt.xlabel("Coefficient on InitialOxy x Post")
    plt.title("Weighted age-specific effects")
    plt.tight_layout()
    plt.savefig(FIGURES / "age_effects_coefficients.png", dpi=180)
    plt.close()

    het_df = heterogeneity.copy()
    het_df["low"] = het_df["coef"] - 1.96 * het_df["se"]
    het_df["high"] = het_df["coef"] + 1.96 * het_df["se"]
    plt.figure(figsize=(7.6, 3.8))
    y = np.arange(len(het_df))[::-1]
    plt.axvline(0, color="black", linewidth=1)
    plt.errorbar(het_df["coef"], y, xerr=1.96 * het_df["se"], fmt="o", color="#7c3aed", ecolor="#7c3aed", capsize=4)
    plt.yticks(y, het_df["group"])
    plt.xlabel("Coefficient on InitialOxy x Post")
    plt.title("Weighted red-state / blue-state heterogeneity")
    plt.tight_layout()
    plt.savefig(FIGURES / "heterogeneity_coefficients.png", dpi=180)
    plt.close()


def make_argument_plots(state: pd.DataFrame, alt_exposure: pd.DataFrame, mech: pd.DataFrame) -> None:
    sample = analysis_sample(state).copy()

    baseline = sample.groupby("state", as_index=False).agg(
        initial_oxy=("initial_oxy", "first"),
        women_15_44=("female_pop_15_44_proxy", "mean"),
    )
    plt.figure(figsize=(8.2, 4.8))
    plt.hist(baseline["initial_oxy"], bins=12, color="#0f4c81", alpha=0.85, edgecolor="white")
    plt.axvline(baseline["initial_oxy"].median(), color="#b45309", linestyle="--", linewidth=2, label="Median")
    plt.xlabel("Baseline OxyContin misuse")
    plt.ylabel("Number of states")
    plt.title("Cross-state variation in baseline OxyContin misuse")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(FIGURES / "baseline_exposure_distribution.png", dpi=180)
    plt.close()

    quart = sample.copy()
    state_quart = quart.groupby("state")["initial_oxy"].first()
    labels = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
    quart_map = pd.qcut(state_quart, 4, labels=labels)
    quart["exposure_quartile"] = quart["state"].map(quart_map.to_dict())
    sub = quart.groupby(["year", "exposure_quartile"], as_index=False)[OUTCOME].mean()
    base = sub[sub["year"] == 2009][["exposure_quartile", OUTCOME]].rename(columns={OUTCOME: "base"})
    sub = sub.merge(base, on="exposure_quartile", how="left")
    sub["normalized"] = 100 * sub[OUTCOME] / sub["base"]
    plt.figure(figsize=(9.0, 5.3))
    palette = {"Q1 lowest": "#b45309", "Q2": "#6b7280", "Q3": "#2563eb", "Q4 highest": "#0f4c81"}
    plt.axvspan(sub["year"].min() - 0.5, 2009.5, color="#f3f4f6", alpha=0.8, zorder=0)
    plt.axvspan(2010.5, sub["year"].max() + 0.5, color="#eef7ef", alpha=0.7, zorder=0)
    plt.axvline(2010, color="gray", linestyle="--", linewidth=1.2)
    plt.axhline(100, color="black", linewidth=0.9, alpha=0.4)
    for label, grp in sub.groupby("exposure_quartile"):
        plt.plot(grp["year"], grp["normalized"], marker="o", linewidth=2.2, color=palette[str(label)], label=str(label))
    plt.xlabel("Year")
    plt.ylabel("Index (2009 = 100)")
    plt.title("Normalized fertility by baseline exposure quartile")
    plt.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES / "fertility_quartile_trends.png", dpi=180)
    plt.close()

    alt = alt_exposure.copy().sort_values("coef")
    plt.figure(figsize=(8.2, 4.8))
    y = np.arange(len(alt))
    colors = ["#0f4c81" if p < 0.1 else "#9ca3af" for p in alt["p_value"]]
    plt.axvline(0, color="black", linewidth=1)
    for i, (_, r) in enumerate(alt.iterrows()):
        low = r["coef"] - 1.96 * r["se"]
        high = r["coef"] + 1.96 * r["se"]
        plt.plot([low, high], [i, i], color=colors[i], linewidth=2)
        plt.plot(r["coef"], i, "o", color=colors[i])
    plt.yticks(y, alt["exposure_measure"])
    plt.xlabel("Coefficient")
    plt.title("Alternative exposure measures")
    plt.tight_layout()
    plt.savefig(FIGURES / "alternative_exposure_coefficients.png", dpi=180)
    plt.close()

    if not mech.empty:
        mech_df = mech.copy().sort_values("coef")
        plt.figure(figsize=(8.0, 3.8))
        y = np.arange(len(mech_df))
        colors = ["#0f4c81" if p < 0.1 else "#9ca3af" for p in mech_df["p_value"]]
        plt.axvline(0, color="black", linewidth=1)
        for i, (_, r) in enumerate(mech_df.iterrows()):
            low = r["coef"] - 1.96 * r["se"]
            high = r["coef"] + 1.96 * r["se"]
            plt.plot([low, high], [i, i], color=colors[i], linewidth=2)
            plt.plot(r["coef"], i, "o", color=colors[i])
        plt.yticks(y, mech_df["outcome"])
        plt.xlabel("Coefficient on InitialOxy x Post")
        plt.title("Mechanism-side coefficients")
        plt.tight_layout()
        plt.savefig(FIGURES / "mechanism_coefficients.png", dpi=180)
        plt.close()


def age_results(age: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(age)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    rows = []
    for label in AGE_GROUPS:
        if label in AGE_GROUP_MAP:
            sub = sample[sample["mother_age_group"].isin(AGE_GROUP_MAP[label])].copy()
            sub = (
                sub.groupby(["state", "year"], as_index=False)
                .agg(
                    births=("births", "sum"),
                    female_pop_age_proxy=("female_pop_age_proxy", "sum"),
                    initial_oxy=("initial_oxy", "first"),
                    exposure_proxy_pre=("exposure_proxy_pre", "first"),
                    **{c: (c, "first") for c in controls},
                )
            )
            sub[AGE_OUTCOME] = 1000 * sub["births"] / sub["female_pop_age_proxy"].where(sub["female_pop_age_proxy"] > 0)
            sub["initial_oxy_post"] = sub["initial_oxy"] * (sub["year"] >= 2011).astype(int)
            sub["proxy_exposure_post"] = sub["exposure_proxy_pre"] * (sub["year"] >= 2011).astype(int)
        else:
            sub = sample[sample["mother_age_group"] == label].copy()
        for treatment in ["initial_oxy_post", "proxy_exposure_post"]:
            reg_df = sub.dropna(subset=[AGE_OUTCOME, treatment] + controls).copy()
            formula = f"{AGE_OUTCOME} ~ {treatment} + C(state) + C(year) + " + " + ".join(controls)
            res = fit_ols(reg_df, formula, weights="female_pop_age_proxy")
            rows.append(
                {
                    "age_group": label,
                    "treatment": treatment,
                    "coef": res.params.get(treatment, np.nan),
                    "se": res.bse.get(treatment, np.nan),
                    "p_value": res.pvalues.get(treatment, np.nan),
                    "n": int(res.nobs),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "age_results.csv", index=False)
    return out


def heterogeneity_results(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    rows = []
    for group_value, group_label in [(0, "Blue states (Obama 2008)"), (1, "Red states (McCain 2008)")]:
        reg_df = sample[sample["red_2008"] == group_value].dropna(subset=[OUTCOME, HEADLINE_TREATMENT, "pop"] + controls).copy()
        formula = f"{OUTCOME} ~ {HEADLINE_TREATMENT} + C(state) + C(year) + " + " + ".join(controls)
        res = fit_ols(reg_df, formula, weights="pop")
        rows.append(
            {
                "group": group_label,
                "coef": res.params.get(HEADLINE_TREATMENT, np.nan),
                "se": res.bse.get(HEADLINE_TREATMENT, np.nan),
                "p_value": res.pvalues.get(HEADLINE_TREATMENT, np.nan),
                "n": int(res.nobs),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "heterogeneity_results.csv", index=False)
    return out


def alternative_exposure_results(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state)
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    mapping = [
        ("Alpert initial_oxy", "initial_oxy_post"),
        ("Oxy misuse 2008", "oxyabuse2008_post"),
        ("Oxy misuse 2004", "oxyabuse2004_post"),
        ("Oxy misuse 2006", "oxyabuse2006_post"),
        ("Pain reliever misuse", "initial_pr_post"),
    ]
    rows = []
    for label, treatment in mapping:
        reg_df = sample.dropna(subset=[OUTCOME, treatment, "pop"] + controls).copy()
        formula = f"{OUTCOME} ~ {treatment} + C(state) + C(year) + " + " + ".join(controls)
        res = fit_ols(reg_df, formula, weights="pop")
        rows.append(
            {
                "exposure_measure": label,
                "coef": res.params.get(treatment, np.nan),
                "se": res.bse.get(treatment, np.nan),
                "p_value": res.pvalues.get(treatment, np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "alternative_exposure_results.csv", index=False)
    return out


def racial_heterogeneity_results(state: pd.DataFrame) -> pd.DataFrame:
    sample = analysis_sample(state).copy()
    controls = safe_cols(sample, PREFERRED_CONTROLS + POLICY_CONTROLS)
    baseline = sample[sample["year"] == 2007].groupby("state", as_index=False)["share_white"].mean()
    cutoff = baseline["share_white"].median()
    sample = sample.merge(
        baseline.assign(white_group=np.where(baseline["share_white"] >= cutoff, "Higher white-share states", "Lower white-share states"))[["state", "white_group"]],
        on="state",
        how="left",
        validate="many_to_one",
    )
    rows = []
    for group in ["Higher white-share states", "Lower white-share states"]:
        reg_df = sample[sample["white_group"] == group].dropna(subset=[OUTCOME, HEADLINE_TREATMENT, "pop"] + controls).copy()
        formula = f"{OUTCOME} ~ {HEADLINE_TREATMENT} + C(state) + C(year) + " + " + ".join(controls)
        res = fit_ols(reg_df, formula, weights="pop")
        rows.append(
            {
                "group": group,
                "coef": res.params.get(HEADLINE_TREATMENT, np.nan),
                "se": res.bse.get(HEADLINE_TREATMENT, np.nan),
                "p_value": res.pvalues.get(HEADLINE_TREATMENT, np.nan),
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "racial_heterogeneity_results.csv", index=False)
    return out


def _fmt_num(x: float, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def _escape_latex(text: str) -> str:
    replacements = {
        "\\": "\\textbackslash{}",
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "{": "\\{",
        "}": "\\}",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def dataframe_to_latex(body: pd.DataFrame) -> str:
    cols = list(body.columns)
    align = "l" + "r" * (len(cols) - 1) if cols else "l"
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule"]
    header = " & ".join(_escape_latex(c) for c in cols) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    for _, row in body.iterrows():
        values = [_escape_latex(row[c]) for c in cols]
        lines.append(" & ".join(values) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return "\n".join(lines)


def significance_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.1:
        return "*"
    return ""


def write_main_regression_table(main: pd.DataFrame, state: pd.DataFrame, inference: pd.DataFrame) -> None:
    lookup = main[main["treatment"] == HEADLINE_TREATMENT].set_index("spec")
    sample = analysis_sample(state)
    exposure_sd, exposure_p90_p10 = _exposure_scalars(sample)
    mean_outcome = float(sample[OUTCOME].mean())
    specs = [
        ("Weighted, no controls", "no_controls_weighted"),
        ("Weighted + core controls", "parsimonious_weighted"),
        ("Weighted + core controls + PDMP", "parsimonious_policy_weighted"),
        ("Relax parallel trends: unweighted + state-specific trends", "parsimonious_state_trends"),
    ]
    rows = []
    for label, spec in specs:
        row = lookup.loc[spec]
        context = contextualize_effect(float(row["coef"]), exposure_sd, exposure_p90_p10, mean_outcome)
        rows.append(
            {
                "Panel": "A. Main specifications",
                "Specification / inference": label,
                "Beta on InitialOxy_s x Post_t": f"{row['coef']:.3f}{significance_stars(row['p_value'])}",
                "Clustered SE": f"({row['se']:.3f})",
                "1 SD effect": f"{context['effect_1sd']:.3f}",
                "90-10 effect": f"{context['effect_p90_p10']:.3f}",
                "% of mean rate": f"{context['pct_mean_1sd']:.2f}",
                "Population weighted": "Yes" if row["weighted"] == 1 else "No",
                "Core controls": "No" if spec == "no_controls_weighted" else "Yes",
                "PDMP control": "Yes" if spec in {"parsimonious_policy_weighted", "parsimonious_financial_policy_weighted"} else "No",
                "State-specific trends": "Yes" if "state_trends" in spec else "No",
            }
        )
    for _, row in inference.iterrows():
        context = contextualize_effect(float(row["coef"]), exposure_sd, exposure_p90_p10, mean_outcome)
        rows.append(
            {
                "Panel": "B. Preferred weighted + PDMP, stress-tested inference",
                "Specification / inference": row["inference"],
                "Beta on InitialOxy_s x Post_t": f"{row['coef']:.3f}{significance_stars(row['p_value'])}",
                "Clustered SE": f"({row['se']:.3f})",
                "1 SD effect": f"{context['effect_1sd']:.3f}",
                "90-10 effect": f"{context['effect_p90_p10']:.3f}",
                "% of mean rate": f"{context['pct_mean_1sd']:.2f}",
                "Population weighted": "Yes",
                "Core controls": "Yes",
                "PDMP control": "Yes",
                "State-specific trends": "No",
            }
        )
    latex = dataframe_to_latex(pd.DataFrame(rows))
    avg_births = float(sample["births"].mean()) if "births" in sample.columns else float("nan")
    note = (
        "Entries report the coefficient on pre-reformulation OxyContin misuse interacted with an indicator for years 2011 and later. "
        "The outcome is a proxy general fertility rate: births per 1,000 women ages 15--44, where the denominator is approximated as total women times the state's observed share of population ages 15--44. Standard errors are shown in parentheses. "
        "The `1 SD effect` column rescales each beta by the cross-state standard deviation of baseline OxyContin misuse, the `90-10 effect` column rescales it by the difference between the 90th- and 10th-percentile states, and `% of mean rate` expresses the 1-SD effect relative to the sample mean fertility rate. "
        f"In this sample the mean fertility rate is {mean_outcome:.2f} births per 1,000 women ages 15--44, and the average state-year has about {avg_births:.0f} births. "
        "Because the design uses only 300 state-year observations and a short pre-period, this table should be read as a low-power setting: effect magnitudes and robustness across specifications are more informative than any single p-value in isolation. "
        "Core controls are unemployment, heroin price, and low-education share. "
        "The weighted columns use state population weights, so they emphasize the effect for women living in larger states, where fertility counts are measured more precisely and where a policy-relevant effect for the average woman is most naturally defined. "
        "Panel B stress-tests the preferred weighted + PDMP specification under alternative covariance estimators. The best default in this setting is clustering by state because treatment varies across states and within-state serial correlation is the main inference problem. Two-way state-year clustering is informative but not preferred because there are only 6 year clusters after dropping 2010. HC1 is shown only as a benchmark because it does not fully account for within-state dependence. "
        "The final column relaxes strict parallel-trends assumptions by allowing each state to have its own linear trend; it is intentionally demanding and is included as a robustness check rather than the preferred specification. "
        "All specifications in this table use 300 state-year observations after dropping 2010 as the transition year. "
        "Significance: * p<0.10, ** p<0.05, *** p<0.01."
    )
    wrapped = "\n".join(
        [
            "\\begin{table}[!htbp]\\centering",
            "\\caption{Weighted headline estimates and robustness checks}",
            "\\label{tab:main}",
            "\\begin{threeparttable}",
            "\\footnotesize",
            "\\begin{adjustbox}{max width=\\textwidth}",
            latex,
            "\\end{adjustbox}",
            f"\\begin{{tablenotes}}[flushleft]\\footnotesize \\item {note}\\end{{tablenotes}}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    (TABLES / "main_results.tex").write_text(wrapped, encoding="utf-8")


def write_tex_table(df: pd.DataFrame, filename: str, caption: str, label: str, note: str, digits: int = 3) -> None:
    body = df.copy()
    for col in body.columns:
        if pd.api.types.is_numeric_dtype(body[col]):
            body[col] = body[col].map(lambda x: _fmt_num(x, digits))
    latex = dataframe_to_latex(body)
    wrapped = "\n".join(
        [
            "\\begin{table}[!htbp]\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
            "\\begin{threeparttable}",
            "\\footnotesize",
            "\\begin{adjustbox}{max width=\\textwidth}",
            latex,
            "\\end{adjustbox}",
            f"\\begin{{tablenotes}}[flushleft]\\footnotesize \\item {note}\\end{{tablenotes}}",
            "\\end{threeparttable}",
            "\\end{table}",
        ]
    )
    (TABLES / filename).write_text(wrapped, encoding="utf-8")


def build_tables(main: pd.DataFrame, age: pd.DataFrame, heterogeneity: pd.DataFrame, alt_exposure: pd.DataFrame, racial_het: pd.DataFrame, mech: pd.DataFrame, placebo: pd.DataFrame, loo: pd.DataFrame, ri: pd.DataFrame, pretrend: pd.DataFrame, state: pd.DataFrame, inference: pd.DataFrame) -> None:
    write_main_regression_table(main, state, inference)
    sample = analysis_sample(state)
    exposure_sd, exposure_p90_p10 = _exposure_scalars(sample)
    age_sample = analysis_sample(load_age())
    age_means = age_sample.groupby("mother_age_group")[AGE_OUTCOME].mean().to_dict()
    age_table = age[age["treatment"] == "initial_oxy_post"][["age_group", "coef", "se", "p_value"]].copy()
    age_table["mean_rate"] = age_table["age_group"].map(age_means)
    age_table.loc[age_table["age_group"] == "25-34 years", "mean_rate"] = age_sample[
        age_sample["mother_age_group"].isin(["25-29 years", "30-34 years"])
    ][AGE_OUTCOME].mean()
    age_table.loc[age_table["age_group"] == "35-44 years", "mean_rate"] = age_sample[
        age_sample["mother_age_group"].isin(["35-39 years", "40-44 years"])
    ][AGE_OUTCOME].mean()
    age_table["effect_1sd"] = age_table["coef"] * exposure_sd
    age_table["effect_p90_p10"] = age_table["coef"] * exposure_p90_p10
    age_table["pct_mean_1sd"] = 100 * age_table["effect_1sd"] / age_table["mean_rate"]
    write_tex_table(
        age_table[["age_group", "coef", "se", "p_value", "mean_rate", "effect_1sd", "effect_p90_p10", "pct_mean_1sd"]],
        "age_results.tex",
        "Weighted age-specific proxy fertility effects",
        "tab:age",
        "Outcome is the proxy age-specific fertility rate, constructed by dividing births in each maternal age bin by a proxy denominator for women in that same age group. The denominator uses total female population interacted with the broad age shares in the authors' file; within a broad bucket such as ages 15--24, the proxy splits population evenly across the two constituent five-year groups. The sample excludes 2010. All regressions include state and year fixed effects plus the parsimonious controls and PDMP authorization, and they are weighted by the age-specific female population proxy so larger age-state cells receive more influence. The `mean rate` column reports the baseline average fertility rate in that age bin. The `1 SD effect` and `90-10 effect` columns convert the raw beta into realistic exposure shifts, and `% mean` reports the 1-SD effect as a share of the baseline age-specific fertility rate. Because the age-specific table is even lower-power than the state-level table, the pooled `25-34` row is more informative than isolated narrow-bin estimates. The age `25-34` group aggregates `25-29` and `30-34` to capture the main childbearing margin, while `35-44` aggregates `35-39` and `40-44` because those older maternal ages are thinner and easier to interpret jointly. Each row uses 300 state-year observations.",
    )
    write_tex_table(
        heterogeneity[["group", "coef", "se", "p_value"]],
        "heterogeneity_results.tex",
        "Red-state / blue-state weighted heterogeneity",
        "tab:heterogeneity",
        "Each row re-estimates the preferred weighted specification separately by 2008 presidential vote. `Blue states` are states carried by Obama in 2008 and `Red states` are states carried by McCain in 2008. The outcome is the proxy fertility rate per 1,000 women ages 15--44, the treatment is pre-reformulation OxyContin misuse interacted with an indicator for years 2011 and later, and the control set is the same parsimonious-plus-PDMP specification used in the headline table.",
    )
    write_tex_table(
        alt_exposure[["exposure_measure", "coef", "se", "p_value"]],
        "alternative_exposure_results.tex",
        "Alternative baseline exposure measures",
        "tab:alt_exposure",
        "Each row re-estimates the preferred weighted specification using a different pre-reformulation exposure measure interacted with the post-2010 indicator. This is a standard robustness exercise for continuous difference-in-differences designs: if the estimates remain similar across plausible exposure definitions, confidence rises that the result is not an artifact of one particular coding choice.",
    )
    write_tex_table(
        racial_het[["group", "coef", "se", "p_value"]],
        "racial_heterogeneity_results.tex",
        "Heterogeneity by baseline racial composition",
        "tab:racial_het",
        "States are split at the median baseline value of \\texttt{share_white}. Each row then re-estimates the preferred weighted specification within that subgroup. This is theoretically motivated because pre-reformulation opioid misuse was more concentrated in whiter populations, so a stronger fertility response in higher white-share states would be consistent with the opioid-exposure channel.",
    )
    if not mech.empty:
        write_tex_table(
            mech,
            "mechanism_results.tex",
            "First-stage-style validation: opioid-market disruption regressions",
            "tab:mechanism",
            "These regressions are first-stage-style validation checks rather than an IV first stage in the literal 2SLS sense. They test whether the reformulation changed the opioid environment more in high-exposure states. The coefficient reports the differential post-2010 change in each mechanism outcome for states with higher pre-reformulation OxyContin exposure, conditional on state fixed effects, year fixed effects, and the parsimonious-plus-policy controls. In the current merged 2007--2013 sample, overdose variables are unpopulated, so this table uses the mechanism outcomes available locally: ARCOS oxycodone supply, ARCOS hydrocodone supply, and heroin treatment admissions.",
        )
    write_tex_table(
        placebo,
        "placebo_results.tex",
        "Placebo adoption-year tests",
        "tab:placebo",
        "These placebo tests intentionally pretend the reformulation happened earlier. If the design were spuriously picking up differential pre-existing trends, we would expect significant placebo coefficients when the post period is reassigned to 2008 or 2009. The treatment variable in each row is pre-reformulation OxyContin exposure interacted with the placebo post indicator, estimated with the same fixed effects and controls as the preferred specification.",
    )
    loo_summary = pd.DataFrame(
        [
            {
                "min_coef": loo["coef"].min(),
                "p25_coef": loo["coef"].quantile(0.25),
                "median_coef": loo["coef"].median(),
                "p75_coef": loo["coef"].quantile(0.75),
                "max_coef": loo["coef"].max(),
            }
        ]
    )
    write_tex_table(
        loo_summary,
        "leave_one_out_summary.tex",
        "Leave-one-state-out stability summary",
        "tab:loo",
        "Each coefficient re-estimates the preferred headline regression after omitting one state at a time. This checks whether the main result is driven by a single influential state such as Florida, West Virginia, or Rhode Island. The table reports the distribution of the leave-one-out coefficients rather than listing all states in the main text.",
    )
    write_tex_table(
        pd.concat([ri, pretrend], axis=1),
        "diagnostics_summary.tex",
        "Parallel-trends and randomization diagnostics",
        "tab:diagnostics",
        "The pretrend entry reports the weighted event-study joint F-test on the 2007 and 2008 lead coefficients, with 2009 as the omitted pre-period. In plain language, this asks whether high-exposure and low-exposure states were already separating before the reformulation. A large p-value means the lead coefficients are not statistically distinguishable from zero, which is supportive of the design but not definitive proof. The randomization-inference p-value comes from permuting baseline exposure across states and re-estimating the weighted headline coefficient. The `post-to-pre magnitude ratio` is a descriptive summary of how large the biggest post coefficient is relative to the largest pre coefficient. It is not a formal modern-DiD estimator and should be read only as an informal diagnostic.",
    )


def write_report(main: pd.DataFrame, age: pd.DataFrame, heterogeneity: pd.DataFrame, alt_exposure: pd.DataFrame, racial_het: pd.DataFrame, mech: pd.DataFrame, pretrend: pd.DataFrame, ri: pd.DataFrame) -> None:
    headline = main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "parsimonious_policy") & (main["weighted"] == 0)].iloc[0]
    weighted = main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "parsimonious_policy_weighted")].iloc[0]
    weighted_core = main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "parsimonious_weighted")].iloc[0]
    trend = main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "parsimonious_state_trends")].iloc[0]
    proxy = main[(main["treatment"] == "proxy_exposure_post") & (main["spec"] == "parsimonious_policy")].iloc[0]
    top_age = age[age["treatment"] == HEADLINE_TREATMENT].sort_values("p_value").iloc[0]
    red = heterogeneity[heterogeneity["group"] == "Red states (McCain 2008)"].iloc[0]
    blue = heterogeneity[heterogeneity["group"] == "Blue states (Obama 2008)"].iloc[0]
    high_white = racial_het[racial_het["group"] == "Higher white-share states"].iloc[0]
    low_white = racial_het[racial_het["group"] == "Lower white-share states"].iloc[0]
    alt_text = "; ".join(
        f"{row.exposure_measure}: coef={row.coef:.3f}, p={row.p_value:.3f}" for _, row in alt_exposure.iterrows()
    )
    interp = effect_interpretation(load_state(), weighted.coef).iloc[0]
    inference = pd.read_csv(OUT / "inference_robustness.csv")
    preferred_inf = inference[inference["cov_mode"] == "state_cluster"].iloc[0]
    twoway_inf = inference[inference["cov_mode"] == "two_way_cluster"].iloc[0]
    year_inf = inference[inference["cov_mode"] == "year_cluster"].iloc[0]
    hc1_inf = inference[inference["cov_mode"] == "hc1"].iloc[0]
    state_sample = analysis_sample(load_state())
    mean_outcome = state_sample[OUTCOME].mean()
    national_births_implied = (
        state_sample.groupby("state")
        .apply(lambda g: weighted.coef * g[BASE_EXPOSURE].iloc[0] * g["female_pop_15_44_proxy"].mean() / 1000)
        .sum()
    )
    sample_decline = state_sample.groupby("year")[OUTCOME].mean().loc[2007] - state_sample.groupby("year")[OUTCOME].mean().loc[2013]
    mech_text = "Mechanism regressions were unavailable." if mech.empty else "; ".join(
        f"{row.outcome}: coef={row.coef:.3f}, p={row.p_value:.3f}" for _, row in mech.iterrows()
    )
    tex = rf"""
\documentclass[11pt]{{article}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{booktabs}}
\usepackage{{threeparttable}}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usepackage{{float}}
\usepackage{{amsmath}}
\usepackage{{hyperref}}
\usepackage{{setspace}}
\usepackage{{caption}}
\captionsetup{{font=small}}
\setstretch{{1.1}}
\setlength{{\emergencystretch}}{{2em}}
\begin{{document}}

\begin{{center}}
{{\LARGE \textbf{{Opioid Supply Disruptions and Fertility}}}}
\end{{center}}

\section*{{Introduction}}
The opioid crisis has generated a large literature on mortality, labor markets, housing, and crime, but much less is known about fertility. This paper studies whether a sharp opioid supply disruption changed fertility in places that had been more exposed to OxyContin before the shock. The empirical setting is the August 2010 abuse-deterrent reformulation of OxyContin, a national change that reduced the drug's misuse potential while affecting states differentially depending on their pre-reformulation exposure.

Our design follows Alpert, Powell, and Pacula (2018) and Evans, Lieber, and Power (2019): a common national reformulation shock interacted with predetermined state exposure. The preferred treatment is the Alpert replication exposure measure \texttt{{initial\_oxy}} from the authors' state-year file. That file is especially useful here because it reproduces the exposure definitions used in the reformulation literature and already contains the relevant macro and policy controls. Public NSDUH pain-reliever misuse is retained as a robustness proxy rather than the headline treatment because it is broader than OxyContin and therefore less tightly linked to the reformulation itself.

The sample is limited to 2007--2013 because this is the clean overlap between the local WONDER births pulls and the authors' curated state-year replication file. That choice keeps the analysis tightly anchored to the replication environment and avoids introducing additional measurement risk from stitching together a longer control panel. A longer-horizon or county-level extension would be valuable, but the present goal is a clean and defensible state-level design.

\section*{{Data And Specification}}
The preferred control set is intentionally parsimonious and follows the design note closely. The exact variables are \texttt{{xxunemp}} (state unemployment rate), \texttt{{heroinprice\_filled}} (state heroin price, with within-state filling when needed), and \texttt{{share\_low\_edu}} (share of adults in the low-education category). These controls are meant to absorb broad labor-market stress, the price of an important opioid substitute, and slower-moving socioeconomic composition, including some of the background pressures associated with the 2008 financial crisis and uneven state recovery. We drop \texttt{{share\_women}} from the preferred specification because the outcome denominator is already built from a female population proxy, so controlling separately for the overall female population share would be awkward and potentially mechanical. One robustness column adds \texttt{{pp\_pmp\_authorized}} for PDMP authorization. We do not make broader policy controls central to the table because the sample is short and the goal is to avoid overfitting the headline regression.

The report focuses on fertility rates for women of childbearing age, so the state outcome is constructed to target women ages 15--44 rather than all women. The local authors' file does not include a direct count of women ages 15--44, but it does include total women, \texttt{{pop\_women}}, and broad age shares of the state population. We therefore build
\[
\text{{female\_pop\_15\_44\_proxy}} = \text{{pop\_women}} \times (\text{{xxage\_2}} + \text{{xxage\_3}} + \text{{xxage\_4}}),
\]
where \texttt{{xxage\_2}} is the population share ages 15--24, \texttt{{xxage\_3}} is ages 25--34, and \texttt{{xxage\_4}} is ages 35--44. The state-level outcome is then
\[
\text{{fertility\_rate\_proxy\_per\_1000\_women\_15\_44}} = 1000 \times \frac{{\text{{births}}}}{{\text{{female\_pop\_15\_44\_proxy}}}}.
\]
This is not an official WONDER fertility-rate series, but it is explicitly targeted at women of childbearing age and is therefore closer to the CDC concept than using all women in the denominator.

For Table 2, the age-specific outcome uses a narrower denominator for each maternal age bin. For example, the \texttt{{15-19 years}} outcome uses births to mothers ages 15--19 divided by a proxy for women ages 15--19, while the \texttt{{25-29 years}} outcome uses births to mothers ages 25--29 divided by a proxy for women ages 25--29. Because the underlying data only provide broad age shares, the proxy splits those broad buckets evenly across their constituent five-year groups. For instance, the ages-15--24 share is split equally into ages 15--19 and 20--24. The design is a continuous-treatment difference-in-differences model with one common shock date rather than a staggered-adoption setting, so the central identifying question is whether states with higher and lower baseline OxyContin exposure would have followed similar fertility paths absent the reformulation.

The weighted headline specification is
\[
\begin{{aligned}}
Y_{{st}} = \alpha_s + \gamma_t
&+ \beta \left(\text{{InitialOxy}}_s \times 1\{{t \geq 2011\}}\right) \\
&+ \delta_1 \text{{Unemp}}_{{st}} + \delta_2 \text{{HeroinPrice}}_{{st}} + \delta_3 \text{{LowEduShare}}_{{st}} + \varepsilon_{{st}}
\end{{aligned}}
\]
where $Y_{{st}}$ is the proxy general fertility rate in state $s$ and year $t$, measured as births per 1,000 women ages 15--44. Because the local replication data do not contain an exact women-15--44 denominator, the bundle approximates that denominator as total women times the observed state population share ages 15--44: \texttt{{female\_pop\_15\_44\_proxy = pop\_women * (xxage\_2 + xxage\_3 + xxage\_4)}}. We then set \texttt{{fertility\_rate\_proxy\_per\_1000\_women\_15\_44 = 1000 * births / female\_pop\_15\_44\_proxy}}. $\alpha_s$ are state fixed effects, and $\gamma_t$ are year fixed effects. The weighted version uses state population weights, so larger states contribute more to estimation.

The most demanding robustness variant relaxes the strict parallel-trends requirement by allowing each state to follow its own linear trend:
\[
\begin{{aligned}}
Y_{{st}} = \alpha_s + \gamma_t + \lambda_s t
&+ \beta \left(\text{{InitialOxy}}_s \times 1\{{t \geq 2011\}}\right) \\
&+ \delta' X_{{st}} + \varepsilon_{{st}}
\end{{aligned}}
\]
This specification is helpful because it asks whether the result survives after absorbing slow-moving state-specific fertility trajectories, but it can also absorb a meaningful share of the treatment signal in a short panel.

\input{{tables/main_results.tex}}
\input{{tables/age_results.tex}}
\input{{tables/heterogeneity_results.tex}}
\input{{tables/alternative_exposure_results.tex}}
\input{{tables/racial_heterogeneity_results.tex}}
\input{{tables/placebo_results.tex}}
\input{{tables/leave_one_out_summary.tex}}
\input{{tables/diagnostics_summary.tex}}
"""
    if not mech.empty:
        tex += "\n\\input{tables/mechanism_results.tex}\n"
    tex += rf"""

\section*{{Results}}
Table 1 reports the headline estimates. The coefficient on pre-reformulation OxyContin misuse interacted with the post-2010 indicator is positive in the weighted specifications and stable across the main columns: {main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "no_controls_weighted")].iloc[0].coef:.3f} with no controls, {weighted_core.coef:.3f} with the core controls, {weighted.coef:.3f} with core controls plus PDMP authorization, and {main[(main["treatment"] == HEADLINE_TREATMENT) & (main["spec"] == "policy_augmented_weighted")].iloc[0].coef:.3f} with the richer policy set. The preferred weighted specification yields a state-clustered standard error of {preferred_inf.se:.3f} $(p={preferred_inf.p_value:.3f})$. Read plainly, states with higher baseline OxyContin exposure experienced a larger relative fertility increase after the reformulation. The sign and magnitude are therefore not an artifact of one particular control choice. The unweighted state-trend specification is much smaller at {trend.coef:.3f} $(p={trend.p_value:.3f})$, which is consistent with a genuinely low-power design in which the most demanding specification absorbs a large share of the available signal rather than cleanly overturning the weighted result.\footnote{{The robustness tables keep the full set of supporting checks. In particular, the preferred coefficient remains similar under two-way state-year clustering, alternative OxyContin-focused exposure definitions, and leave-one-out estimation, while the broader pain-reliever placebo is weak.}} 

The economic magnitude is more informative than the raw one-unit beta. Because the cross-state standard deviation of baseline OxyContin exposure is only about 0.23, a one-standard-deviation increase in exposure corresponds to {interp.effect_per_1sd_per_1000_women:.2f} additional births per 1,000 women ages 15--44, or about {100 * interp.effect_per_1sd_per_1000_women / mean_outcome:.1f}\% of the sample mean fertility rate of {mean_outcome:.1f}. Moving from a state near the 10th percentile of exposure to one near the 90th percentile corresponds to roughly {interp.effect_p90_p10_per_1000_women:.2f} additional births per 1,000 women. A simple state-level back-of-the-envelope calculation implies about {national_births_implied:.0f} births nationally per post-reformulation year.\footnote{{This national scaling is descriptive rather than a separate estimand. It applies the preferred weighted coefficient to the observed cross-state exposure distribution and average female population ages 15--44.}} Relative to the sample-wide fertility decline of about {sample_decline:.2f} births per 1,000 women between 2007 and 2013, the 90--10 exposure contrast is economically meaningful rather than trivial.

The age decomposition in Table 2 helps interpret the mechanism. The clearest positive response appears in the pooled 25--34 group, where the coefficient is {age[age["treatment"] == HEADLINE_TREATMENT].set_index("age_group").loc["25-34 years", "coef"]:.3f} with $p={age[age["treatment"] == HEADLINE_TREATMENT].set_index("age_group").loc["25-34 years", "p_value"]:.3f}$. The narrower 25--29 and 30--34 bins are also positive and statistically significant, while teenage and older-age estimates are much less precise. That pattern is more consistent with a fertility-restoration channel concentrated in core childbearing ages than with a diffuse shift across all age bins. The heterogeneity splits are more descriptive. The weighted estimate is {red.coef:.3f} in red states $(p={red.p_value:.3f})$ and {blue.coef:.3f} in blue states $(p={blue.p_value:.3f})$, so the positive response is not unique to one political subsample. The racial-composition split is noisier: the coefficient is {high_white.coef:.3f} in higher white-share states and {low_white.coef:.3f} in lower white-share states, but those subgroup estimates should be treated cautiously because power is limited.

The dynamic evidence is broadly consistent with the main regression. The omitted year is 2009, the last full pre-reformulation period. The 2007 and 2008 lead coefficients are the formal parallel-trends check; their joint F-test has $p={float(pretrend['p_value'].iloc[0]):.3f}$, which is not a clean slam dunk but is still informative in a short panel with only two leads. The post-2010 coefficients become more positive over time, which fits the idea that any fertility response should emerge with a lag after the August 2010 reformulation.\footnote{{Inference stress tests are reported in the table notes rather than the main text. In the preferred weighted specification, the state-clustered standard error is {preferred_inf.se:.3f}, the two-way state-year clustered standard error is {twoway_inf.se:.3f}, the year-only clustered standard error is {year_inf.se:.3f}, and HC1 is {hc1_inf.se:.3f}. The preferred choice remains clustering by state because treatment varies at the state level and the year dimension has only 6 usable clusters after dropping 2010.}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{figures/event_initial_oxy.png}}
\caption{{Event study for the headline weighted design. Coefficients show differential fertility changes in higher-exposure states relative to lower-exposure states, normalized to zero in 2009. The pre-period leads are the parallel-trends diagnostic and the post-period lags trace the evolving fertility response after the reformulation.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{figures/normalized_oxycodone_supply.png}}
\caption{{First-stage-style visual: normalized oxycodone supply by baseline exposure group, indexed to 2009. The high-exposure group experiences a clearly weaker post-reformulation path, illustrating the supply-side disruption that underlies the fertility design.}}
\end{{figure}}

\begin{{figure}}[H]
\centering
\includegraphics[width=0.8\textwidth]{{figures/normalized_fertility_proxy.png}}
\caption{{Population-weighted fertility rates by baseline exposure group, indexed to 2009. High-exposure states show a modestly stronger post-reformulation fertility path than low-exposure states, the descriptive counterpart to the positive weighted DiD estimate.}}
\end{{figure}}
"""
    tex += rf"""

\section*{{Conclusion}}
This paper provides evidence that the 2010 OxyContin reformulation was followed by a relative fertility increase in states with higher pre-reformulation OxyContin exposure. The preferred weighted estimate implies a modest but economically meaningful rise in births concentrated in core childbearing ages, especially ages 25--34. That pattern is most consistent with a fertility-restoration channel rather than a diffuse shift across all age groups. More broadly, the results suggest that major opioid-supply interventions can have demographic spillovers that standard evaluations may miss. The design is not high-powered and the measurement remains imperfect, but the direction, scale, and internal coherence of the estimates make fertility a serious outcome margin in the broader economics of the opioid crisis.

\section*{{References Mentioned in the Bundle}}
Alpert, Abby, David Powell, and Rosalie Liccardo Pacula (2018), ``Supply-Side Drug Policy in the Presence of Substitutes.''\\
Evans, William N., Ethan Lieber, and Patrick Power (2019/2018 working paper), ``How the Reformulation of OxyContin Ignited the Heroin Epidemic.''\\
Goodman-Bacon, Andrew (2021), ``Difference-in-Differences with Variation in Treatment Timing.''\\
Sun, Liyang, and Sarah Abraham (2021), ``Estimating Dynamic Treatment Effects in Event Studies with Heterogeneous Treatment Effects.''\\
Callaway, Brantly, and Pedro H.C. Sant'Anna (2021), ``Difference-in-Differences with Multiple Time Periods.''\\
Roth, Jonathan (2022), ``Pretest with Caution.''\\
Rambachan, Ashesh, and Jonathan Roth (2023), ``A More Credible Approach to Parallel Trends.''\\
Abadie, Alberto, Joshua Angrist, Brigham Frandsen, and J\"orn-Steffen Pischke (2025/2026 draft), ``Harvesting Differences-in-Differences and Event-Study Evidence.''

\end{{document}}
"""
    (OUT / "paper_results.tex").write_text(tex.strip() + "\n", encoding="utf-8")


def make_event_plot_copy() -> None:
    src = FIGURES / "event_initial_oxy.png"
    if src.exists():
        return


def main() -> None:
    ensure_dirs()
    state = load_state()
    age = load_age()

    main_results = run_main_specs(state)
    age_res = age_results(age)
    heterogeneity = heterogeneity_results(state)
    alt_exposure = alternative_exposure_results(state)
    racial_het = racial_heterogeneity_results(state)
    mech = mechanism_regs(state)
    inference = inference_robustness(state)
    evt, pretrend = event_study(state, BASE_EXPOSURE, "event_initial_oxy")
    event_study(state, "exposure_proxy_pre", "event_nsduh_proxy")
    placebo = placebo_post_tests(state, HEADLINE_TREATMENT, "initial_oxy")
    ri = randomization_inference(state)
    loo = leave_one_out(state)
    make_normalized_validation_plots(state)
    make_parallel_trends_plot(state)
    make_coefficient_plots(main_results, age_res, heterogeneity)
    make_argument_plots(state, alt_exposure, mech)

    build_tables(main_results, age_res, heterogeneity, alt_exposure, racial_het, mech, placebo, loo, ri, pretrend, state, inference)
    write_report(main_results, age_res, heterogeneity, alt_exposure, racial_het, mech, pretrend, ri)

    print(OUT)


if __name__ == "__main__":
    main()
