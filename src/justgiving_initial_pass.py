# Analysis for the JustGiving project
# Prepared by Diogo Koch Alves, PhD
# Date: 2026-03-26

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =========================
# SETTINGS ----
# =========================
DATA_PATH = Path("/Users/dkoch811/Documents/GitHub/Projects/Justgiving/crowdfunding.csv")
PIPELINE_SCRIPT = Path("/Users/dkoch811/Documents/GitHub/Projects/Justgiving/motivation_ensemble_v2.py")
OUTPUT_DIR = Path("/Users/dkoch811/Documents/GitHub/Projects/Justgiving/exploratory_outputs")
SAMPLE_SIZE = None 


# =========================
# HELPERS ----
# =========================
def read_any_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def safe_json_loads(x: Any) -> dict:
    if isinstance(x, dict):
        return x
    if pd.isna(x):
        return {}
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return {}
    return {}


def parse_details_if_needed(df: pd.DataFrame) -> pd.DataFrame:
    if "details" not in df.columns:
        return df.copy()

    details = pd.json_normalize(df["details"].map(safe_json_loads))
    out = pd.concat([df.drop(columns=["details"]), details], axis=1)
    return out


def coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def import_pipeline_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("motivation_ensemble_v2", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_json_list(x: Any) -> list[str]:
    if isinstance(x, list):
        return x
    if pd.isna(x):
        return []
    if isinstance(x, str):
        try:
            parsed = json.loads(x)
            return parsed if isinstance(parsed, list) else []
        except Exception:
            return []
    return []


def tidy_regression(y: pd.Series, x: pd.Series, outcome: str, predictor: str) -> dict:
    dat = pd.DataFrame({"y": y, "x": x}).dropna()

    if len(dat) < 10 or dat["x"].nunique() < 2:
        return {
            "outcome": outcome,
            "motivation": predictor.replace("profile_", ""),
            "n": len(dat),
            "beta": np.nan,
            "se": np.nan,
            "t": np.nan,
            "p": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "r2": np.nan,
            "direction": np.nan,
        }

    X = sm.add_constant(dat["x"])
    model = sm.OLS(dat["y"], X).fit()

    beta = model.params["x"]
    ci_low, ci_high = model.conf_int().loc["x"].tolist()

    return {
        "outcome": outcome,
        "motivation": predictor.replace("profile_", ""),
        "n": int(model.nobs),
        "beta": float(beta),
        "se": float(model.bse["x"]),
        "t": float(model.tvalues["x"]),
        "p": float(model.pvalues["x"]),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "r2": float(model.rsquared),
        "direction": "higher" if beta > 0 else "lower",
    }


def build_preprocessing_audit(
    df: pd.DataFrame,
    story_col: str,
    preprocess_func,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()

    out["story"] = out[story_col].fillna("").astype(str)
    out["raw_story_stripped"] = out["story"].str.strip()
    out["has_nonempty_raw_story"] = out["raw_story_stripped"].ne("")

    out["clean_story_for_key"] = out["story"].apply(preprocess_func)
    out["clean_char_len"] = out["clean_story_for_key"].str.len()
    out["clean_word_count"] = (
        out["clean_story_for_key"]
        .str.split()
        .str.len()
        .fillna(0)
        .astype(int)
    )

    # Actual pipeline rule
    out["is_short_lt10_chars"] = out["clean_char_len"] < 10

    # Diagnostic only
    out["is_short_lt10_words"] = out["clean_word_count"] < 10

    out["story_key"] = (
        out["clean_story_for_key"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    eligible_for_pipeline = ~out["is_short_lt10_chars"]

    out["is_duplicate_story"] = False
    out.loc[eligible_for_pipeline, "is_duplicate_story"] = (
        out.loc[eligible_for_pipeline, "story_key"]
        .duplicated(keep="first")
    )

    out["sent_to_classifier"] = eligible_for_pipeline & ~out["is_duplicate_story"]

    n_total = len(out)
    n_nonempty_raw = int(out["has_nonempty_raw_story"].sum())
    n_ge10_chars = int((~out["is_short_lt10_chars"]).sum())
    n_lt10_chars = int(out["is_short_lt10_chars"].sum())
    n_ge10_words = int((~out["is_short_lt10_words"]).sum())
    n_lt10_words = int(out["is_short_lt10_words"].sum())
    n_duplicates = int(out["is_duplicate_story"].sum())
    n_sent = int(out["sent_to_classifier"].sum())

    audit = pd.DataFrame(
        [
            {
                "stage_type": "pipeline",
                "stage": "raw_rows",
                "rows_remaining": n_total,
                "rows_dropped_this_stage": 0,
                "pct_of_raw": round(100 * n_total / n_total, 2),
                "note": "All rows before preprocessing",
            },
            {
                "stage_type": "diagnostic",
                "stage": "nonempty_raw_story",
                "rows_remaining": n_nonempty_raw,
                "rows_dropped_this_stage": n_total - n_nonempty_raw,
                "pct_of_raw": round(100 * n_nonempty_raw / n_total, 2),
                "note": "Raw story.strip() != ''",
            },
            {
                "stage_type": "pipeline",
                "stage": "clean_story_ge_10_chars",
                "rows_remaining": n_ge10_chars,
                "rows_dropped_this_stage": n_lt10_chars,
                "pct_of_raw": round(100 * n_ge10_chars / n_total, 2),
                "note": "Actual minimum-length filter used by the original pipeline",
            },
            {
                "stage_type": "diagnostic",
                "stage": "clean_story_ge_10_words",
                "rows_remaining": n_ge10_words,
                "rows_dropped_this_stage": n_lt10_words,
                "pct_of_raw": round(100 * n_ge10_words / n_total, 2),
                "note": "Diagnostic only; not used for filtering",
            },
            {
                "stage_type": "pipeline",
                "stage": "after_deduplication",
                "rows_remaining": n_sent,
                "rows_dropped_this_stage": n_duplicates,
                "pct_of_raw": round(100 * n_sent / n_total, 2),
                "note": "Unique cleaned stories sent to classifier before optional sampling",
            },
        ]
    )

    return out, audit


def build_duplicate_story_examples(
    df_audit: pd.DataFrame,
    short_name_col: str,
    max_short_names: int = 5,
) -> pd.DataFrame:
    dup_df = df_audit.loc[
        df_audit["sent_to_classifier"] | df_audit["is_duplicate_story"],
        ["story_key", "story", short_name_col],
    ].copy()

    out = (
        dup_df.groupby("story_key", as_index=False)
        .agg(
            n_rows=(short_name_col, "size"),
            example_short_names=(
                short_name_col,
                lambda x: " | ".join(x.astype(str).head(max_short_names).tolist()),
            ),
            example_story=("story", "first"),
        )
        .query("n_rows > 1")
        .sort_values(["n_rows", "story_key"], ascending=[False, True])
        .reset_index(drop=True)
    )

    out["example_story_preview"] = out["example_story"].str.slice(0, 300)
    return out[
        ["story_key", "n_rows", "example_short_names", "example_story_preview"]
    ]


def build_classification_descriptives(
    classifications: pd.DataFrame,
    categories: list[str],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # Primary category counts
    primary_counts = (
        classifications["primary_category"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("primary_category")
        .reset_index(name="n")
    )
    primary_counts["pct"] = 100 * primary_counts["n"] / len(classifications)
    out["primary_category_counts"] = primary_counts

    # Active-label prevalence
    active_long = (
        classifications["all_categories"]
        .fillna("[]")
        .map(parse_json_list)
        .explode()
        .dropna()
    )

    active_counts = (
        active_long.value_counts()
        .rename_axis("motivation")
        .reset_index(name="n_stories_with_label")
    )

    all_cats_df = pd.DataFrame({"motivation": categories})
    active_counts = all_cats_df.merge(active_counts, on="motivation", how="left")
    active_counts["n_stories_with_label"] = active_counts["n_stories_with_label"].fillna(0).astype(int)
    active_counts["pct_of_classified_stories"] = (
        100 * active_counts["n_stories_with_label"] / len(classifications)
    )
    active_counts = active_counts.sort_values(
        ["n_stories_with_label", "motivation"], ascending=[False, True]
    ).reset_index(drop=True)
    out["active_label_prevalence"] = active_counts

    # Distribution of number of labels
    num_cat_dist = (
        classifications["num_categories"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("num_categories")
        .reset_index(name="n")
    )
    num_cat_dist["pct"] = 100 * num_cat_dist["n"] / len(classifications)
    out["num_categories_distribution"] = num_cat_dist

    # Mean profile weights
    profile_rows = []
    for cat in categories:
        col = f"profile_{cat}"
        if col in classifications.columns:
            profile_rows.append(
                {
                    "motivation": cat,
                    "mean_profile_weight": classifications[col].mean(),
                    "sd_profile_weight": classifications[col].std(),
                }
            )

    profile_summary = (
        pd.DataFrame(profile_rows)
        .sort_values(["mean_profile_weight", "motivation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out["mean_profile_weights"] = profile_summary

    return out


def build_coverage_table(
    raw_rows: int,
    df_audit: pd.DataFrame,
    classification_input: pd.DataFrame,
    classifications: pd.DataFrame,
    merged: pd.DataFrame,
    outcomes: list[str],
    sample_size: int | None,
) -> pd.DataFrame:
    rows = [
        {"metric": "raw_rows", "value": raw_rows},
        {"metric": "nonempty_raw_story_rows", "value": int(df_audit["has_nonempty_raw_story"].sum())},
        {"metric": "rows_surviving_ge10_char_filter", "value": int((~df_audit["is_short_lt10_chars"]).sum())},
        {"metric": "rows_dropped_lt10_chars", "value": int(df_audit["is_short_lt10_chars"].sum())},
        {"metric": "duplicate_story_rows_dropped_pre_classification", "value": int(df_audit["is_duplicate_story"].sum())},
        {"metric": "unique_story_keys_pre_classification", "value": int(len(classification_input))},
        {"metric": "sample_size_requested", "value": "full_run" if sample_size is None else int(sample_size)},
        {"metric": "unique_story_keys_actually_classified", "value": int(classifications["story_key"].nunique())},
        {"metric": "merged_rows_total", "value": int(len(merged))},
        {"metric": "merged_rows_with_classification", "value": int(merged["primary_category"].notna().sum())},
        {"metric": "merged_rows_without_classification", "value": int(merged["primary_category"].isna().sum())},
    ]

    for outcome in outcomes:
        rows.append(
            {
                "metric": f"rows_with_classification_and_nonmissing_{outcome}",
                "value": int(merged.loc[merged["primary_category"].notna(), outcome].notna().sum()),
            }
        )

    return pd.DataFrame(rows)


def build_outcome_descriptives_overall(
    merged: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    rows = []
    dat = merged.loc[merged["primary_category"].notna()].copy()

    for outcome in outcomes:
        s = dat[outcome].dropna()
        rows.append(
            {
                "outcome": outcome,
                "n": int(s.notna().sum()),
                "mean": float(s.mean()) if len(s) else np.nan,
                "sd": float(s.std()) if len(s) else np.nan,
                "median": float(s.median()) if len(s) else np.nan,
                "min": float(s.min()) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def build_outcome_by_primary_category(
    merged: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    dat = merged.loc[merged["primary_category"].notna()].copy()

    pieces = []
    for outcome in outcomes:
        tmp = (
            dat.groupby("primary_category", dropna=False)[outcome]
            .agg(
                n="count",
                mean="mean",
                sd="std",
                median="median",
                min="min",
                max="max",
            )
            .reset_index()
        )
        tmp.insert(0, "outcome", outcome)
        pieces.append(tmp)

    if not pieces:
        return pd.DataFrame(
            columns=["outcome", "primary_category", "n", "mean", "sd", "median", "min", "max"]
        )

    return pd.concat(pieces, ignore_index=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nSTEP 1: Read dataset")
    raw = read_any_table(DATA_PATH)
    print(f"Rows: {len(raw):,}")
    print("Top-level columns:")
    print(raw.columns.tolist())

    print("\nSTEP 2: Unpack JSON if needed")
    df = parse_details_if_needed(raw)
    print(f"Columns after unpacking: {len(df.columns)}")

    short_name_col = choose_first_existing(df, ["short_name", "pageShortName", "shortName"])
    story_col = choose_first_existing(df, ["story", "pageStory"])
    activity_col = choose_first_existing(df, ["activity_type", "activityType"])
    target_col = choose_first_existing(df, ["targetAmount", "fundraisingTarget", "target_amount"])
    raised_col = choose_first_existing(df, ["totalRaisedOnline", "total_raised_online"])
    donations_col = choose_first_existing(df, ["donationCount", "donation_count"])

    print("\nDetected columns:")
    print(
        {
            "short_name_col": short_name_col,
            "story_col": story_col,
            "activity_col": activity_col,
            "target_col": target_col,
            "raised_col": raised_col,
            "donations_col": donations_col,
        }
    )

    required = [short_name_col, story_col, raised_col, donations_col]
    if any(col is None for col in required):
        raise ValueError("Could not detect one or more required columns.")

    if activity_col is None:
        df["activity_type"] = ""
        activity_col = "activity_type"

    if target_col is None:
        df["targetAmount"] = np.nan
        target_col = "targetAmount"

    df[raised_col] = coerce_numeric(df[raised_col])
    df[donations_col] = coerce_numeric(df[donations_col])
    df[target_col] = coerce_numeric(df[target_col])

    print("\nSTEP 3: Prepare unique story file for the original pipeline")
    pipeline_module = import_pipeline_module(PIPELINE_SCRIPT)
    categories = list(pipeline_module.CATEGORIES)

    df_audit, preprocessing_audit = build_preprocessing_audit(
        df=df,
        story_col=story_col,
        preprocess_func=pipeline_module.preprocess_text,
    )

    audit_path = OUTPUT_DIR / "00_preprocessing_audit.csv"
    preprocessing_audit.to_csv(audit_path, index=False)

    print("\nPreprocessing audit:")
    print(preprocessing_audit.to_string(index=False))
    print(f"\nSaved preprocessing audit to: {audit_path}")

    print("\nDetailed counts:")
    print(f"- Raw rows: {len(df_audit):,}")
    print(f"- Non-empty raw stories: {int(df_audit['has_nonempty_raw_story'].sum()):,}")
    print(f"- Dropped for clean story < 10 chars: {int(df_audit['is_short_lt10_chars'].sum()):,}")
    print(f"- Clean stories >= 10 chars: {int((~df_audit['is_short_lt10_chars']).sum()):,}")
    print(f"- Diagnostic only: clean stories < 10 words: {int(df_audit['is_short_lt10_words'].sum()):,}")
    print(f"- Duplicate cleaned stories dropped: {int(df_audit['is_duplicate_story'].sum()):,}")
    print(f"- Unique stories sent to classifier before sampling: {int(df_audit['sent_to_classifier'].sum()):,}")

    duplicate_examples = build_duplicate_story_examples(
        df_audit=df_audit,
        short_name_col=short_name_col,
    )
    duplicate_examples_path = OUTPUT_DIR / "00b_duplicate_story_examples.csv"
    duplicate_examples.to_csv(duplicate_examples_path, index=False)
    print(f"Saved duplicate story examples to: {duplicate_examples_path}")

    classification_input = (
        df_audit.loc[df_audit["sent_to_classifier"], ["story_key", "story", activity_col]]
        .rename(columns={activity_col: "activity_type"})
        .reset_index(drop=True)
    )

    input_path = OUTPUT_DIR / "01_classification_input.csv"
    classification_input.to_csv(input_path, index=False)
    print(f"Unique stories sent to classifier: {len(classification_input):,}")
    print(f"Saved input file to: {input_path}")

    print("\nSTEP 4: Run the original classification pipeline")
    classifications = pipeline_module.run_analysis(
        csv_path=str(input_path),
        story_col="story",
        id_col="story_key",
        sample_size=SAMPLE_SIZE,
        output_dir=str(OUTPUT_DIR),
    )

    classifications_path = OUTPUT_DIR / "02_classifications_clean.csv"
    classifications.to_csv(classifications_path, index=False)
    print(f"Saved classifications to: {classifications_path}")

    print("\nSTEP 4B: Build classification descriptives")
    descriptives = build_classification_descriptives(
        classifications=classifications,
        categories=categories,
    )

    primary_counts_path = OUTPUT_DIR / "02b_primary_category_counts.csv"
    active_prevalence_path = OUTPUT_DIR / "02c_active_label_prevalence.csv"
    num_categories_path = OUTPUT_DIR / "02d_num_categories_distribution.csv"
    profile_weights_path = OUTPUT_DIR / "02e_mean_profile_weights.csv"

    descriptives["primary_category_counts"].to_csv(primary_counts_path, index=False)
    descriptives["active_label_prevalence"].to_csv(active_prevalence_path, index=False)
    descriptives["num_categories_distribution"].to_csv(num_categories_path, index=False)
    descriptives["mean_profile_weights"].to_csv(profile_weights_path, index=False)

    print(f"Saved primary category counts to: {primary_counts_path}")
    print(f"Saved active-label prevalence to: {active_prevalence_path}")
    print(f"Saved num-categories distribution to: {num_categories_path}")
    print(f"Saved mean profile weights to: {profile_weights_path}")

    print("\nPrimary category counts:")
    print(descriptives["primary_category_counts"].to_string(index=False))

    print("\nSTEP 5: Merge classification outputs back to the dataset")
    merged = df_audit.merge(classifications, on="story_key", how="left", suffixes=("", "_clf"))

    print("\nSTEP 6: Create outcome variables")
    merged["log_totalRaisedOnline_plus1"] = np.log(merged[raised_col] + 1)
    merged["log_donationCount_plus1"] = np.log(merged[donations_col] + 1)
    merged["pct_target"] = np.where(
        merged[target_col] > 0,
        merged[raised_col] / merged[target_col],
        np.nan,
    )

    merged_path = OUTPUT_DIR / "03_merged_with_outcomes.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved merged data to: {merged_path}")

    outcomes = [
        "log_totalRaisedOnline_plus1",
        "log_donationCount_plus1",
        "pct_target",
    ]

    print("\nSTEP 6B: Build coverage and raw outcome descriptives")
    coverage_table = build_coverage_table(
        raw_rows=len(raw),
        df_audit=df_audit,
        classification_input=classification_input,
        classifications=classifications,
        merged=merged,
        outcomes=outcomes,
        sample_size=SAMPLE_SIZE,
    )
    coverage_path = OUTPUT_DIR / "03b_classification_coverage.csv"
    coverage_table.to_csv(coverage_path, index=False)
    print(f"Saved coverage table to: {coverage_path}")

    outcome_overall = build_outcome_descriptives_overall(
        merged=merged,
        outcomes=outcomes,
    )
    outcome_overall_path = OUTPUT_DIR / "03c_outcome_descriptives_overall.csv"
    outcome_overall.to_csv(outcome_overall_path, index=False)
    print(f"Saved overall outcome descriptives to: {outcome_overall_path}")

    outcome_by_primary = build_outcome_by_primary_category(
        merged=merged,
        outcomes=outcomes,
    )
    outcome_by_primary_path = OUTPUT_DIR / "03d_outcome_by_primary_category.csv"
    outcome_by_primary.to_csv(outcome_by_primary_path, index=False)
    print(f"Saved outcome descriptives by primary category to: {outcome_by_primary_path}")

    print("\nSTEP 7: Run simple exploratory regressions")
    profile_cols = [c for c in classifications.columns if c.startswith("profile_")]

    reg_rows = []
    for outcome in outcomes:
        for predictor in profile_cols:
            reg_rows.append(tidy_regression(merged[outcome], merged[predictor], outcome, predictor))

    reg_results = pd.DataFrame(reg_rows)
    reg_results["sig"] = np.select(
        [
            reg_results["p"] < 0.001,
            reg_results["p"] < 0.01,
            reg_results["p"] < 0.05,
            reg_results["p"] < 0.10,
        ],
        ["***", "**", "*", "."],
        default="",
    )
    reg_results = reg_results.sort_values(["outcome", "p", "motivation"]).reset_index(drop=True)

    reg_path = OUTPUT_DIR / "04_regression_results.csv"
    reg_results.to_csv(reg_path, index=False)
    print(f"Saved regression results to: {reg_path}")

    print("\nSTEP 8: Build simple summary table")
    summary = reg_results.copy()
    summary["association"] = np.where(
        summary["beta"].isna(),
        "insufficient variation",
        np.where(summary["beta"] > 0, "higher", "lower"),
    )
    summary["summary"] = summary["association"] + " outcome" + summary["sig"].map(
        lambda s: f" ({s})" if s != "" else ""
    )

    summary_table = (
        summary.pivot(index="motivation", columns="outcome", values="summary")
        .reset_index()
        .rename_axis(None, axis=1)
    )

    summary_path = OUTPUT_DIR / "05_summary_table.csv"
    summary_table.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")

    print("\nDone.")
    print("\nMain outputs:")
    print(f"- {audit_path}")
    print(f"- {duplicate_examples_path}")
    print(f"- {classifications_path}")
    print(f"- {primary_counts_path}")
    print(f"- {active_prevalence_path}")
    print(f"- {num_categories_path}")
    print(f"- {profile_weights_path}")
    print(f"- {merged_path}")
    print(f"- {coverage_path}")
    print(f"- {outcome_overall_path}")
    print(f"- {outcome_by_primary_path}")
    print(f"- {reg_path}")
    print(f"- {summary_path}")

    print("\nNote:")
    print(
        "These regressions are run one outcome x one motivation at a time using the profile_ columns. "
        "That is deliberate because the nine profile columns sum to 1 "
        "and would be collinear in a single model."
    )


if __name__ == "__main__":
    main()