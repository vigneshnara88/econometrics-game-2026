from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("opioid_fertility_oxycontin")
RAW_WONDER = ROOT / "data" / "raw" / "wonder"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

AUTHORS_DTA = ROOT / "data" / "raw" / "oxycontindata_final.dta"


def load_authors() -> pd.DataFrame:
    df = pd.read_stata(AUTHORS_DTA).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["state"] = df["state"].astype(str).str.strip().str.title()
    df = df[df["state"].ne("")].copy()
    df = df[df["year"].between(2007, 2013, inclusive="both")].copy()
    df["stfips"] = pd.to_numeric(df["stfips"], errors="coerce").astype("Int64")

    # Main treatment definitions and controls already curated by the authors.
    keep = [
        "state",
        "stfips",
        "year",
        "arcos_totaloxycod",
        "arcos_totalhydroc",
        "oxyuse",
        "painrelieveruse",
        "oxyabuse2008",
        "oxyabuse2004",
        "oxyabuse2006",
        "oxyabuse2012",
        "prabuse2008",
        "prabuse2004",
        "initial_oxy",
        "initial_pr",
        "pp_pmp_authorized",
        "pp_must",
        "pp_PILLMILL",
        "pp_medmj",
        "pp_actp",
        "hpi",
        "totpop_SEER",
        "pop",
        "pop_young",
        "pop_mid",
        "pop_old",
        "pop_women",
        "pop_men",
        "pop_white",
        "pop_nwhite",
        "pop_ed1",
        "pop_ed2",
        "xxsex_female",
        "xxunemp",
        "xxage_1",
        "xxage_2",
        "xxage_3",
        "xxage_4",
        "xxage_6",
        "xxed1",
        "xxed2",
        "xxwhite",
        "age_kid",
        "age_adult",
        "age_old",
        "race_white",
        "race_black",
        "race_other",
        "teds_heroin",
        "mort_OPIOIDS",
        "mort_HEROIN",
        "mort_OVERDOSE",
        "mortrate_OPIOIDS_ageadjusted",
        "mortrate_HEROIN_ageadjusted",
        "heroinprice",
    ]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()
    df["post_reform"] = (df["year"] >= 2011).astype(int)
    df["event_time"] = df["year"] - 2010
    df["initial_oxy_post"] = df["initial_oxy"] * df["post_reform"]
    df["initial_pr_post"] = df["initial_pr"] * df["post_reform"]
    df["oxyabuse2008_post"] = df["oxyabuse2008"] * df["post_reform"]
    df["prabuse2008_post"] = df["prabuse2008"] * df["post_reform"]
    df = add_derived_controls(df)
    return df


def add_derived_controls(df: pd.DataFrame) -> pd.DataFrame:
    if {"state", "year"}.issubset(df.columns):
        df = df.sort_values(["state", "year"]).copy()
    denom = df["pop"].where(df["pop"] > 0)

    if "births" in df.columns:
        df["birth_rate_per_1000"] = 1000 * df["births"] / denom
    if {"births", "pop_women"}.issubset(df.columns):
        women_denom = df["pop_women"].where(df["pop_women"] > 0)
        df["fertility_rate_proxy_per_1000_women"] = 1000 * df["births"] / women_denom
    if {"births", "pop_women", "xxage_2", "xxage_3", "xxage_4"}.issubset(df.columns):
        share_15_44 = (
            pd.to_numeric(df["xxage_2"], errors="coerce").clip(lower=0)
            + pd.to_numeric(df["xxage_3"], errors="coerce").clip(lower=0)
            + pd.to_numeric(df["xxage_4"], errors="coerce").clip(lower=0)
        ).clip(lower=0, upper=1)
        df["female_pop_15_44_proxy"] = pd.to_numeric(df["pop_women"], errors="coerce") * share_15_44
        childbearing_denom = df["female_pop_15_44_proxy"].where(df["female_pop_15_44_proxy"] > 0)
        df["fertility_rate_proxy_per_1000_women_15_44"] = 1000 * df["births"] / childbearing_denom

    if "pop_young" in df.columns:
        df["share_young"] = df["pop_young"] / denom
    if "pop_mid" in df.columns:
        df["share_mid"] = df["pop_mid"] / denom
    if "pop_old" in df.columns:
        df["share_old"] = df["pop_old"] / denom
    if "pop_women" in df.columns:
        df["share_women"] = df["pop_women"] / denom
    if "pop_white" in df.columns:
        df["share_white"] = df["pop_white"] / denom
    if "pop_nwhite" in df.columns:
        df["share_nonwhite"] = df["pop_nwhite"] / denom
    if "pop_ed1" in df.columns:
        df["share_low_edu"] = df["pop_ed1"] / denom
    if "pop_ed2" in df.columns:
        df["share_high_edu"] = df["pop_ed2"] / denom

    if "hpi" in df.columns:
        hpi = pd.to_numeric(df["hpi"], errors="coerce")
        df["ln_hpi"] = np.where(hpi > 0, np.log(hpi), np.nan)
        df["hpi_growth_pct"] = 100 * df.groupby("state")["ln_hpi"].diff()
        df["lag_hpi_growth_pct"] = df.groupby("state")["hpi_growth_pct"].shift(1)
    if "pop" in df.columns:
        pop = pd.to_numeric(df["pop"], errors="coerce")
        df["ln_pop"] = np.where(pop > 0, np.log(pop), np.nan)
    if "teds_heroin" in df.columns:
        series = pd.to_numeric(df["teds_heroin"], errors="coerce").fillna(0)
        df["ln_teds_heroin_plus1"] = np.log(series + 1)
    if "xxunemp" in df.columns:
        unemp = pd.to_numeric(df["xxunemp"], errors="coerce")
        df["unemp_change"] = df.groupby("state")["xxunemp"].diff()
        df["lag_unemp_change"] = df.groupby("state")["unemp_change"].shift(1)

    if "heroinprice" in df.columns:
        df["heroinprice_state_mean"] = (
            df.groupby("state")["heroinprice"].transform("mean")
        )
        df["heroinprice_filled"] = df["heroinprice"].fillna(df["heroinprice_state_mean"])

    control_sets = {
        "controls_demo_basic": [
            "share_young",
            "share_old",
            "share_white",
            "share_low_edu",
        ],
        "controls_macro_basic": [
            "xxunemp",
            "ln_hpi",
            "ln_pop",
        ],
        "controls_policy_basic": [
            "pp_pmp_authorized",
            "pp_must",
            "pp_PILLMILL",
            "pp_medmj",
            "pp_actp",
        ],
        "controls_opioid_env": [
            "heroinprice_filled",
            "mortrate_OPIOIDS_ageadjusted",
            "mortrate_HEROIN_ageadjusted",
            "ln_teds_heroin_plus1",
        ],
    }
    for name, cols in control_sets.items():
        df[name] = " | ".join([c for c in cols if c in df.columns])

    return df


def add_age_specific_fertility_proxy(df: pd.DataFrame) -> pd.DataFrame:
    needed = {"mother_age_group", "births", "pop_women", "xxage_1", "xxage_2", "xxage_3", "xxage_4", "xxage_6"}
    if not needed.issubset(df.columns):
        return df

    female_pop = pd.to_numeric(df["pop_women"], errors="coerce")
    age_0_14 = pd.to_numeric(df["xxage_1"], errors="coerce").clip(lower=0)
    age_15_24 = pd.to_numeric(df["xxage_2"], errors="coerce").clip(lower=0)
    age_25_34 = pd.to_numeric(df["xxage_3"], errors="coerce").clip(lower=0)
    age_35_44 = pd.to_numeric(df["xxage_4"], errors="coerce").clip(lower=0)
    age_65_plus = pd.to_numeric(df["xxage_6"], errors="coerce").clip(lower=0)

    # The source file omits an explicit xxage_5 column, so we proxy ages 45-64
    # as the residual population share after the observed age buckets.
    age_45_64 = (1 - age_0_14 - age_15_24 - age_25_34 - age_35_44 - age_65_plus).clip(lower=0)

    proxy_share = pd.Series(np.nan, index=df.index, dtype="float64")
    group = df["mother_age_group"].astype(str)
    proxy_share = proxy_share.mask(group.eq("Under 15 years"), age_0_14)
    proxy_share = proxy_share.mask(group.eq("15-19 years"), 0.5 * age_15_24)
    proxy_share = proxy_share.mask(group.eq("20-24 years"), 0.5 * age_15_24)
    proxy_share = proxy_share.mask(group.eq("25-29 years"), 0.5 * age_25_34)
    proxy_share = proxy_share.mask(group.eq("30-34 years"), 0.5 * age_25_34)
    proxy_share = proxy_share.mask(group.eq("35-39 years"), 0.5 * age_35_44)
    proxy_share = proxy_share.mask(group.eq("40-44 years"), 0.5 * age_35_44)
    # Proxy 45-49 as one quarter of the residual 45-64 bucket.
    proxy_share = proxy_share.mask(group.eq("45-49 years"), 0.25 * age_45_64)

    df["female_pop_age_proxy"] = female_pop * proxy_share
    age_denom = df["female_pop_age_proxy"].where(df["female_pop_age_proxy"] > 0)
    df["age_specific_fertility_rate_proxy_per_1000"] = 1000 * df["births"] / age_denom
    return df


def load_state_births() -> pd.DataFrame:
    df = pd.read_csv(RAW_WONDER / "natality_state_year_births_2007_2016.csv").copy()
    if "Notes" in df.columns:
        df = df[df["Notes"].fillna("") == ""].copy()
    df["state"] = df["State"].astype(str).str.strip().str.title()
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["state_code"] = pd.to_numeric(df["State Code"], errors="coerce").astype("Int64")
    df["births"] = pd.to_numeric(df["Births"], errors="coerce")
    return df[["state", "state_code", "year", "births"]]


def load_age_births() -> pd.DataFrame:
    df = pd.read_csv(RAW_WONDER / "natality_state_year_age_births_2007_2016.csv").copy()
    if "Notes" in df.columns:
        df = df[df["Notes"].fillna("") == ""].copy()
    df["state"] = df["State"].astype(str).str.strip().str.title()
    df["year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["state_code"] = pd.to_numeric(df["State Code"], errors="coerce").astype("Int64")
    df["births"] = pd.to_numeric(df["Births"], errors="coerce")
    df["mother_age_group"] = df["Age of Mother 9"].astype(str).str.strip()
    return df[["state", "state_code", "year", "mother_age_group", "births"]]


def build_master() -> None:
    authors = load_authors()
    state_births = load_state_births()
    age_births = load_age_births()

    overlap_births = state_births[state_births["year"].between(2007, 2013, inclusive="both")].copy()
    overlap_age = age_births[age_births["year"].between(2007, 2013, inclusive="both")].copy()

    master_state = overlap_births.merge(
        authors,
        on=["state", "year"],
        how="inner",
        validate="one_to_one",
    )
    master_state = add_derived_controls(master_state)

    master_age = overlap_age.merge(
        authors,
        on=["state", "year"],
        how="inner",
        validate="many_to_one",
    )
    totals = master_state[["state", "year", "births"]].rename(columns={"births": "state_births_total"})
    master_age = master_age.merge(totals, on=["state", "year"], how="left", validate="many_to_one")
    master_age["birth_share"] = master_age["births"] / master_age["state_births_total"]
    master_age = add_derived_controls(master_age)
    master_age = add_age_specific_fertility_proxy(master_age)

    master_state.to_csv(PROCESSED / "master_state_year_2007_2013.csv", index=False)
    master_age.to_csv(PROCESSED / "master_state_year_age_2007_2013.csv", index=False)

    brief = [
        "# Master Dataset From Authors + WONDER",
        "",
        "Main analysis samples:",
        "- `master_state_year_2007_2013.csv`: state-year births merged to authors' treatment and controls.",
        "- `master_state_year_age_2007_2013.csv`: state-year-age births merged to authors' treatment and controls.",
        "",
        "Recommended main treatment variables:",
        "- `initial_oxy_post`",
        "- `initial_pr_post`",
        "- `oxyabuse2008_post`",
        "- `prabuse2008_post`",
        "",
        "Key existing controls from authors data:",
        "- policy variables: `pp_pmp_authorized`, `pp_must`, `pp_PILLMILL`, `pp_medmj`, `pp_actp`",
        "- demographics: `xxsex_female`, `xxunemp`, `xxage_*`, `xxed*`, `xxwhite`",
        "- state environment: `hpi`, `heroinprice`, overdose mortality variables",
        "",
        "Added derived controls for fertility analysis:",
        "- `share_young`, `share_old`, `share_women`, `share_white`, `share_low_edu`",
        "- `ln_pop`, `ln_hpi`, `xxunemp`",
        "- `heroinprice_filled`, `ln_teds_heroin_plus1`",
        "- `birth_rate_per_1000` in the state-year master file",
        "- `fertility_rate_proxy_per_1000_women` in both master files",
        "- `age_specific_fertility_rate_proxy_per_1000` and `female_pop_age_proxy` in the age-level master file",
        "",
        "Coverage:",
        f"- authors controls/treatment years: {int(authors['year'].min())}-{int(authors['year'].max())}",
        f"- merged main sample years: {int(master_state['year'].min())}-{int(master_state['year'].max())}",
        "",
        "Note:",
        "- WONDER outcomes run through 2016, but the authors' curated treatment/control file stops in 2013.",
        "- Cleanest causal main specification therefore uses 2007-2013 unless we separately extend the controls.",
        "- Age-specific fertility rates use a proxy denominator based on the authors' female population and age-share covariates, because the current local WONDER pulls only include births counts.",
    ]
    (PROCESSED / "master_dataset_brief.md").write_text("\n".join(brief), encoding="utf-8")


if __name__ == "__main__":
    build_master()
