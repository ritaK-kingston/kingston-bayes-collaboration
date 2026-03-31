# Analysis for the JustGiving project
# Prepared by Diogo Koch Alves, PhD
# Date: 2026-03-26

from __future__ import annotations

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats


# =========================
# SETTINGS ----
# =========================
OUTPUT_DIR = Path("/Users/dkoch811/Documents/GitHub/Projects/Justgiving/exploratory_outputs")

CLASSIFICATIONS_PATH = OUTPUT_DIR / "02_classifications_clean.csv"
MERGED_PATH = OUTPUT_DIR / "03_merged_with_outcomes.csv"

ANALYSIS_DIR = OUTPUT_DIR / "postclassification_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# HELPERS ----
# =========================
def parse_json_list(x) -> list[str]:
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


def add_sig_stars(df: pd.DataFrame, p_col: str = "p") -> pd.DataFrame:
    df = df.copy()
    df["sig"] = np.select(
        [
            df[p_col] < 0.001,
            df[p_col] < 0.01,
            df[p_col] < 0.05,
            df[p_col] < 0.10,
        ],
        ["***", "**", "*", "."],
        default="",
    )
    return df


def build_classification_descriptives(classifications: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    # infer categories from profile columns
    profile_cols = [c for c in classifications.columns if c.startswith("profile_")]
    categories = [c.replace("profile_", "") for c in profile_cols]

    # 1) Primary category counts
    primary_counts = (
        classifications["primary_category"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("primary_category")
        .reset_index(name="n")
    )
    primary_counts["pct"] = 100 * primary_counts["n"] / len(classifications)
    out["primary_category_counts"] = primary_counts

    # 2) Active-label prevalence
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

    # 3) Distribution of number of labels
    num_cat_dist = (
        classifications["num_categories"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("num_categories")
        .reset_index(name="n")
    )
    num_cat_dist["pct"] = 100 * num_cat_dist["n"] / len(classifications)
    out["num_categories_distribution"] = num_cat_dist

    # 4) Mean profile weights
    profile_rows = []
    for cat in categories:
        col = f"profile_{cat}"
        profile_rows.append(
            {
                "motivation": cat,
                "mean_profile_weight": classifications[col].mean(),
                "sd_profile_weight": classifications[col].std(),
                "min_profile_weight": classifications[col].min(),
                "max_profile_weight": classifications[col].max(),
            }
        )

    profile_summary = (
        pd.DataFrame(profile_rows)
        .sort_values(["mean_profile_weight", "motivation"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out["mean_profile_weights"] = profile_summary

    return out


def build_overall_outcome_descriptives(
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

    return pd.concat(pieces, ignore_index=True)


def rerun_profile_regressions(
    merged: pd.DataFrame,
    outcomes: list[str],
) -> pd.DataFrame:
    profile_cols = [c for c in merged.columns if c.startswith("profile_")]

    reg_rows = []
    for outcome in outcomes:
        for predictor in profile_cols:
            reg_rows.append(tidy_regression(merged[outcome], merged[predictor], outcome, predictor))

    reg_results = pd.DataFrame(reg_rows)
    reg_results = add_sig_stars(reg_results, p_col="p")
    reg_results = reg_results.sort_values(["outcome", "p", "motivation"]).reset_index(drop=True)
    return reg_results


def build_percent_change_table(reg_results: pd.DataFrame) -> pd.DataFrame:
    out = reg_results.copy()

    # For log outcomes only
    out = out.loc[out["outcome"].isin(["log_totalRaisedOnline_plus1", "log_donationCount_plus1"])].copy()

    # Full 1.0 increase in profile weight
    out["pct_change_per_1pt_profile"] = 100 * (np.exp(out["beta"]) - 1)

    # More interpretable: 0.10 increase in profile weight
    out["pct_change_per_10pp_profile"] = 100 * (np.exp(0.10 * out["beta"]) - 1)
    out["pct_change_ci_low_10pp"] = 100 * (np.exp(0.10 * out["ci_low"]) - 1)
    out["pct_change_ci_high_10pp"] = 100 * (np.exp(0.10 * out["ci_high"]) - 1)

    return out[
        [
            "outcome",
            "motivation",
            "n",
            "beta",
            "p",
            "sig",
            "pct_change_per_1pt_profile",
            "pct_change_per_10pp_profile",
            "pct_change_ci_low_10pp",
            "pct_change_ci_high_10pp",
            "r2",
        ]
    ].sort_values(["outcome", "p", "motivation"]).reset_index(drop=True)


def build_pairwise_primary_category_tests(
    merged: pd.DataFrame,
    outcomes: list[str],
    min_n_per_group: int = 20,
) -> pd.DataFrame:
    dat = merged.loc[merged["primary_category"].notna()].copy()

    rows = []

    for outcome in outcomes:
        tmp = dat.loc[dat[outcome].notna(), ["primary_category", outcome]].copy()

        counts = tmp["primary_category"].value_counts()
        valid_cats = counts[counts >= min_n_per_group].index.tolist()
        tmp = tmp.loc[tmp["primary_category"].isin(valid_cats)].copy()

        cats = sorted(tmp["primary_category"].unique().tolist())

        for a, b in combinations(cats, 2):
            xa = tmp.loc[tmp["primary_category"] == a, outcome].dropna()
            xb = tmp.loc[tmp["primary_category"] == b, outcome].dropna()

            if len(xa) < min_n_per_group or len(xb) < min_n_per_group:
                continue

            t_res = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")

            mean_a = xa.mean()
            mean_b = xb.mean()
            diff = mean_a - mean_b

            rows.append(
                {
                    "outcome": outcome,
                    "category_a": a,
                    "category_b": b,
                    "n_a": int(len(xa)),
                    "n_b": int(len(xb)),
                    "mean_a": float(mean_a),
                    "mean_b": float(mean_b),
                    "mean_diff_a_minus_b": float(diff),
                    "t": float(t_res.statistic),
                    "p": float(t_res.pvalue),
                }
            )

    out = pd.DataFrame(rows)

    if len(out) == 0:
        return out

    # FDR within outcome
    out["p_fdr"] = np.nan
    for outcome in out["outcome"].unique():
        mask = out["outcome"] == outcome
        _, p_adj, _, _ = multipletests(out.loc[mask, "p"], method="fdr_bh")
        out.loc[mask, "p_fdr"] = p_adj

    out = add_sig_stars(out, p_col="p_fdr")
    out["better_category"] = np.where(
        out["mean_diff_a_minus_b"] > 0,
        out["category_a"],
        out["category_b"],
    )

    return out.sort_values(["outcome", "p_fdr", "category_a", "category_b"]).reset_index(drop=True)


def build_summary_table(reg_results: pd.DataFrame) -> pd.DataFrame:
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
    return summary_table


def main():
    print("Reading classified files...")
    classifications = pd.read_csv(CLASSIFICATIONS_PATH)
    merged = pd.read_csv(MERGED_PATH)

    print(f"Classified unique stories: {len(classifications):,}")
    print(f"Merged rows: {len(merged):,}")

    outcomes = [
        "log_totalRaisedOnline_plus1",
        "log_donationCount_plus1",
        "pct_target",
    ]

    print("\n1) Classification descriptives")
    descriptives = build_classification_descriptives(classifications)

    primary_counts_path = ANALYSIS_DIR / "a01_primary_category_counts.csv"
    active_prevalence_path = ANALYSIS_DIR / "a02_active_label_prevalence.csv"
    num_categories_path = ANALYSIS_DIR / "a03_num_categories_distribution.csv"
    profile_weights_path = ANALYSIS_DIR / "a04_mean_profile_weights.csv"

    descriptives["primary_category_counts"].to_csv(primary_counts_path, index=False)
    descriptives["active_label_prevalence"].to_csv(active_prevalence_path, index=False)
    descriptives["num_categories_distribution"].to_csv(num_categories_path, index=False)
    descriptives["mean_profile_weights"].to_csv(profile_weights_path, index=False)

    print(descriptives["primary_category_counts"].to_string(index=False))

    print("\n2) Outcome descriptives")
    overall_outcomes = build_overall_outcome_descriptives(merged, outcomes)
    by_primary = build_outcome_by_primary_category(merged, outcomes)

    overall_outcomes_path = ANALYSIS_DIR / "a05_outcome_descriptives_overall.csv"
    by_primary_path = ANALYSIS_DIR / "a06_outcome_by_primary_category.csv"

    overall_outcomes.to_csv(overall_outcomes_path, index=False)
    by_primary.to_csv(by_primary_path, index=False)

    print("\n3) Profile regressions")
    reg_results = rerun_profile_regressions(merged, outcomes)
    reg_results_path = ANALYSIS_DIR / "a07_regression_results.csv"
    reg_results.to_csv(reg_results_path, index=False)

    print("\n4) Percent-change interpretation table")
    pct_change = build_percent_change_table(reg_results)
    pct_change_path = ANALYSIS_DIR / "a08_percent_change_interpretation.csv"
    pct_change.to_csv(pct_change_path, index=False)

    print("\n5) Pairwise comparisons between primary categories")
    pairwise = build_pairwise_primary_category_tests(merged, outcomes, min_n_per_group=20)
    pairwise_path = ANALYSIS_DIR / "a09_pairwise_primary_category_tests.csv"
    pairwise.to_csv(pairwise_path, index=False)

    print("\n6) Compact summary table")
    summary_table = build_summary_table(reg_results)
    summary_path = ANALYSIS_DIR / "a10_summary_table.csv"
    summary_table.to_csv(summary_path, index=False)

    print("\nDone.")
    print("\nSaved:")
    print(f"- {primary_counts_path}")
    print(f"- {active_prevalence_path}")
    print(f"- {num_categories_path}")
    print(f"- {profile_weights_path}")
    print(f"- {overall_outcomes_path}")
    print(f"- {by_primary_path}")
    print(f"- {reg_results_path}")
    print(f"- {pct_change_path}")
    print(f"- {pairwise_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()