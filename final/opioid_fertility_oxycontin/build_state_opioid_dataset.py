import re
from pathlib import Path

import pandas as pd


ROOT = Path("opioid_fertility_oxycontin")
RAW_WONDER = ROOT / "data" / "raw" / "wonder"
RAW_SAMHSA = ROOT / "data" / "raw" / "samhsa"
PROCESSED = ROOT / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)


def clean_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace("", pd.NA)
        .astype(float)
    )


def build_exposure_table() -> pd.DataFrame:
    tables = pd.read_html(RAW_SAMHSA / "pain_reliever_state_report.html")
    table = tables[4].copy()
    table.columns = [
        "state",
        "pain_reliever_12plus_2009_2010",
        "pain_reliever_12plus_2010_2011",
        "pain_reliever_12_17_2009_2010",
        "pain_reliever_12_17_2010_2011",
        "pain_reliever_18_25_2009_2010",
        "pain_reliever_18_25_2010_2011",
        "pain_reliever_26plus_2009_2010",
        "pain_reliever_26plus_2010_2011",
    ]
    table = table[table["state"] != "Total United States"].copy()
    for col in table.columns[1:]:
        table[col] = clean_numeric(table[col])

    table["exposure_proxy_pre"] = table["pain_reliever_12plus_2009_2010"]
    table["exposure_proxy_post"] = table["pain_reliever_12plus_2010_2011"]
    table["exposure_proxy_change"] = (
        table["exposure_proxy_post"] - table["exposure_proxy_pre"]
    )
    median_exposure = table["exposure_proxy_pre"].median()
    table["high_exposure_proxy"] = (
        table["exposure_proxy_pre"] >= median_exposure
    ).astype(int)

    table.to_csv(PROCESSED / "state_pain_reliever_exposure_proxy.csv", index=False)
    return table


def load_births(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Notes" in df.columns:
        df = df[df["Notes"].fillna("") == ""].copy()
        df = df.drop(columns=["Notes"])
    df["year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["state_code"] = df["State Code"].astype(str).str.zfill(2)
    df["births"] = pd.to_numeric(df["Births"], errors="coerce")
    df["state"] = df["State"]
    df["post_reform"] = (df["year"] >= 2011).astype(int)
    df["event_time"] = df["year"] - 2010
    return df


def build_state_year_dataset(exposure: pd.DataFrame) -> pd.DataFrame:
    births = load_births(RAW_WONDER / "natality_state_year_births_2007_2016.csv")
    merged = births.merge(exposure, on="state", how="left", validate="many_to_one")
    merged["exposure_post"] = merged["exposure_proxy_pre"] * merged["post_reform"]
    cols = [
        "state",
        "state_code",
        "year",
        "births",
        "post_reform",
        "event_time",
        "exposure_proxy_pre",
        "exposure_proxy_post",
        "exposure_proxy_change",
        "high_exposure_proxy",
        "exposure_post",
    ]
    extra = [c for c in merged.columns if c.startswith("pain_reliever_")]
    merged = merged[cols + extra]
    merged.to_csv(PROCESSED / "opioid_state_year_births_panel.csv", index=False)
    return merged


AGE_MAP = {
    "Under 15 years": "under15",
    "15-19 years": "15_19",
    "20-24 years": "20_24",
    "25-29 years": "25_29",
    "30-34 years": "30_34",
    "35-39 years": "35_39",
    "40-44 years": "40_44",
    "45-49 years": "45_49",
    "50 years and over": "50plus",
}


def build_state_year_age_dataset(exposure: pd.DataFrame) -> pd.DataFrame:
    births = load_births(RAW_WONDER / "natality_state_year_age_births_2007_2016.csv")
    births["mother_age_group"] = births["Age of Mother 9"].map(AGE_MAP).fillna(
        births["Age of Mother 9"]
    )
    births["mother_age_code"] = births["Age of Mother 9 Code"]
    merged = births.merge(exposure, on="state", how="left", validate="many_to_one")
    merged["exposure_post"] = merged["exposure_proxy_pre"] * merged["post_reform"]
    cols = [
        "state",
        "state_code",
        "year",
        "mother_age_group",
        "mother_age_code",
        "births",
        "post_reform",
        "event_time",
        "exposure_proxy_pre",
        "exposure_proxy_post",
        "exposure_proxy_change",
        "high_exposure_proxy",
        "exposure_post",
    ]
    extra = [c for c in merged.columns if c.startswith("pain_reliever_")]
    merged = merged[cols + extra]
    merged.to_csv(PROCESSED / "opioid_state_year_age_births_panel.csv", index=False)
    return merged


def write_brief(state_panel: pd.DataFrame, age_panel: pd.DataFrame) -> None:
    lines = [
        "# Opioid-Fertility State Dataset",
        "",
        "This dataset uses CDC WONDER natality births and a public SAMHSA state-level proxy for pre-reformulation opioid exposure.",
        "",
        "## Files",
        "",
        "- `state_pain_reliever_exposure_proxy.csv`: state-level exposure proxy from SAMHSA NSDUH pain-reliever misuse tables.",
        "- `opioid_state_year_births_panel.csv`: state-year births panel, 2007-2016.",
        "- `opioid_state_year_age_births_panel.csv`: state-year-age births panel, 2007-2016.",
        "",
        "## Exposure Definition",
        "",
        "- `exposure_proxy_pre`: 2009-2010 state rate of nonmedical prescription pain-reliever use among persons aged 12+.",
        "- `high_exposure_proxy`: indicator for states at or above the median pre-period exposure.",
        "- `exposure_post`: interaction of pre-period exposure with `post_reform`, where `post_reform = 1` for 2011 onward.",
        "",
        "## Caveat",
        "",
        "This is a public proxy for opioid exposure, not the exact county/state OxyContin pills-per-capita series from ARCOS or the Alpert-Powell-Pacula replication files.",
        "",
        "## Shapes",
        "",
        f"- State-year rows: {len(state_panel)}",
        f"- State-year-age rows: {len(age_panel)}",
        f"- States: {state_panel['state'].nunique()}",
    ]
    (PROCESSED / "dataset_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    exposure = build_exposure_table()
    state_panel = build_state_year_dataset(exposure)
    age_panel = build_state_year_age_dataset(exposure)
    write_brief(state_panel, age_panel)
    print("Wrote processed opioid state datasets:")
    print(PROCESSED / "state_pain_reliever_exposure_proxy.csv")
    print(PROCESSED / "opioid_state_year_births_panel.csv")
    print(PROCESSED / "opioid_state_year_age_births_panel.csv")
    print(PROCESSED / "dataset_brief.md")


if __name__ == "__main__":
    main()
