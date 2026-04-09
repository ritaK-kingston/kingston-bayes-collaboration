# Analysis for the JustGiving project
# Prepared by Diogo Koch Alves, PhD
# Updated: 2026-04-09

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests


# =========================
# SETTINGS ----
# =========================
BASE_DIR = Path("/Users/dkoch811/Documents/GitHub/Projects/Justgiving")

INITIAL_DIR = BASE_DIR / "JG Initial Analysis"
CLASSIFICATIONS_PATH = INITIAL_DIR / "02_classified_unique_stories.csv"
MERGED_PATH = INITIAL_DIR / "03_merged_with_outcomes.csv"

ANALYSIS_DIR = BASE_DIR / "exploratory_outputs" / "postclassification_analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

print("CLASSIFICATIONS_PATH:", CLASSIFICATIONS_PATH)
print("MERGED_PATH:", MERGED_PATH)
print("ANALYSIS_DIR:", ANALYSIS_DIR)

if not CLASSIFICATIONS_PATH.exists():
    raise FileNotFoundError(f"Missing file: {CLASSIFICATIONS_PATH}")

if not MERGED_PATH.exists():
    raise FileNotFoundError(f"Missing file: {MERGED_PATH}")


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


def choose_first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clean_string_series(series: pd.Series) -> pd.Series:
    out = series.astype(str).str.strip()
    out = out.replace(
        {
            "": np.nan,
            "nan": np.nan,
            "None": np.nan,
            "NaN": np.nan,
            "<NA>": np.nan,
        }
    )
    return out


def nonmissing_string_flag(series: pd.Series) -> pd.Series:
    cleaned = clean_string_series(series)
    return cleaned.notna()


def coerce_bool(series: pd.Series) -> pd.Series:
    def _convert(x):
        if pd.isna(x):
            return pd.NA
        if isinstance(x, bool):
            return x
        if isinstance(x, (int, float)):
            if x == 1:
                return True
            if x == 0:
                return False
            return pd.NA

        s = str(x).strip().lower()
        if s in {"true", "t", "yes", "y", "1"}:
            return True
        if s in {"false", "f", "no", "n", "0"}:
            return False
        return pd.NA

    return series.map(_convert)


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


# =========================
# PAGE ORIGIN + CHARITY DIAGNOSTICS ----
# =========================
def add_page_origin(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()

    charity_id_col = choose_first_existing(out, ["charity.id", "charityId", "charity_id"])
    charity_name_col = choose_first_existing(out, ["charity.name", "charityName", "charity_name"])
    charity_created_col = choose_first_existing(
        out, ["activityCharityCreated", "activity_charity_created"]
    )

    if charity_id_col is None:
        raise ValueError(
            "Could not find charity ID column in merged dataset. "
            "Expected one of: charity.id, charityId, charity_id"
        )

    out["charity_id_clean"] = clean_string_series(out[charity_id_col])
    out["charity_name_clean"] = (
        clean_string_series(out[charity_name_col]) if charity_name_col is not None
        else pd.Series(np.nan, index=out.index, dtype="object")
    )

    out["charity_linked"] = out["charity_id_clean"].notna()

    if charity_created_col is None:
        out["activityCharityCreated_clean"] = pd.Series(pd.NA, index=out.index, dtype="object")
    else:
        out["activityCharityCreated_clean"] = coerce_bool(out[charity_created_col])

    out["page_origin"] = np.select(
        [
            out["charity_linked"] & (out["activityCharityCreated_clean"] == True),
            out["charity_linked"] & (out["activityCharityCreated_clean"] == False),
            ~out["charity_linked"],
        ],
        [
            "charity_created",
            "individual_for_charity",
            "individual_for_self_or_other",
        ],
        default="unclassified",
    )

    out["charity_name_status"] = np.select(
        [
            out["charity_name_clean"].isna(),
            out["charity_name_clean"] == ".",
        ],
        [
            "missing_name",
            "dot_name",
        ],
        default="normal_name",
    )

    return out


def build_page_origin_counts(merged: pd.DataFrame) -> pd.DataFrame:
    out = (
        merged["page_origin"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("page_origin")
        .reset_index(name="n")
    )
    out["pct"] = 100 * out["n"] / len(merged)
    return out


def build_page_origin_by_activity(merged: pd.DataFrame) -> pd.DataFrame:
    activity_col = choose_first_existing(merged, ["activityType", "activity_type"])
    if activity_col is None:
        return pd.DataFrame(columns=["page_origin"])

    return pd.crosstab(
        merged["page_origin"],
        merged[activity_col],
        dropna=False,
    ).reset_index()


def build_charity_outputs(
    merged: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    charity_id_col = choose_first_existing(merged, ["charity.id", "charityId", "charity_id"])
    charity_name_col = choose_first_existing(merged, ["charity.name", "charityName", "charity_name"])

    if charity_name_col is None:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, empty

    tmp = merged.copy()

    if "charity_name_clean" not in tmp.columns:
        tmp["charity_name_clean"] = clean_string_series(tmp[charity_name_col])

    if charity_id_col is not None and "charity_id_clean" not in tmp.columns:
        tmp["charity_id_clean"] = clean_string_series(tmp[charity_id_col])

    if "charity_linked" not in tmp.columns:
        if "charity_id_clean" in tmp.columns:
            tmp["charity_linked"] = tmp["charity_id_clean"].notna()
        else:
            tmp["charity_linked"] = False

    if "charity_name_status" not in tmp.columns:
        tmp["charity_name_status"] = np.select(
            [
                tmp["charity_name_clean"].isna(),
                tmp["charity_name_clean"] == ".",
            ],
            [
                "missing_name",
                "dot_name",
            ],
            default="normal_name",
        )

    unique_charity_names = (
        tmp.loc[tmp["charity_name_clean"].notna(), ["charity_name_clean"]]
        .drop_duplicates()
        .sort_values("charity_name_clean")
        .reset_index(drop=True)
        .rename(columns={"charity_name_clean": "charity_name"})
    )

    charity_name_counts = (
        tmp["charity_name_clean"]
        .value_counts(dropna=False)
        .rename_axis("charity_name")
        .reset_index(name="n_rows")
        .sort_values(["n_rows", "charity_name"], ascending=[False, True], na_position="last")
        .reset_index(drop=True)
    )
    charity_name_counts["pct_rows"] = 100 * charity_name_counts["n_rows"] / len(tmp)

    if charity_id_col is None or "charity_id_clean" not in tmp.columns:
        charity_id_name_lookup = (
            tmp.loc[tmp["charity_name_clean"].notna(), ["charity_name_clean"]]
            .drop_duplicates()
            .sort_values("charity_name_clean")
            .reset_index(drop=True)
            .rename(columns={"charity_name_clean": "charity_name"})
        )
    else:
        charity_id_name_lookup = (
            tmp.loc[tmp["charity_name_clean"].notna(), ["charity_id_clean", "charity_name_clean"]]
            .drop_duplicates()
            .sort_values(["charity_name_clean", "charity_id_clean"])
            .reset_index(drop=True)
            .rename(columns={"charity_id_clean": "charity_id", "charity_name_clean": "charity_name"})
        )

    charity_name_status_by_missing_id = pd.crosstab(
        tmp["charity_name_status"],
        tmp["charity_linked"],
        dropna=False,
    ).reset_index()
    charity_name_status_by_missing_id = charity_name_status_by_missing_id.rename(
        columns={False: "charity_id_missing", True: "charity_id_present"}
    )

    problem_cols = [
        c for c in [
            "charity_name_status",
            "charity_id_clean",
            "charity_name_clean",
            "page_origin",
            "owner",
            "ownerGuid",
            "consumerId",
            "title",
            "pageShortName",
            "activityType",
            "activityCharityCreated",
            "activityCharityCreated_clean",
            "inMemoriam",
            "rememberedPersonSummary.name",
        ] if c in tmp.columns
    ]

    sort_cols = [c for c in ["charity_name_status", "charity_id_clean"] if c in problem_cols]

    charity_problem_rows = tmp.loc[
        tmp["charity_name_status"].isin(["missing_name", "dot_name"]),
        problem_cols,
    ].copy()
    if sort_cols:
        charity_problem_rows = charity_problem_rows.sort_values(sort_cols, na_position="last")
    charity_problem_rows = charity_problem_rows.reset_index(drop=True)

    return (
        unique_charity_names,
        charity_name_counts,
        charity_id_name_lookup,
        charity_name_status_by_missing_id,
        charity_problem_rows,
    )


def build_memorial_outputs(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tmp = merged.copy()

    if "charity_linked" not in tmp.columns:
        charity_id_col = choose_first_existing(tmp, ["charity.id", "charityId", "charity_id"])
        if charity_id_col is not None:
            tmp["charity_id_clean"] = clean_string_series(tmp[charity_id_col])
            tmp["charity_linked"] = tmp["charity_id_clean"].notna()
        else:
            tmp["charity_linked"] = False

    if "page_origin" not in tmp.columns:
        tmp = add_page_origin(tmp)

    activity_col = choose_first_existing(tmp, ["activityType", "activity_type"])
    in_memoriam_col = choose_first_existing(tmp, ["inMemoriam", "in_memoriam"])
    remembered_name_col = choose_first_existing(
        tmp,
        [
            "rememberedPersonSummary.name",
            "remembered_person_summary_name",
        ],
    )

    memory_mask = pd.Series(False, index=tmp.index)

    if activity_col is not None:
        memory_mask = memory_mask | (tmp[activity_col].astype(str).str.strip() == "InMemory")

    if in_memoriam_col is not None:
        memory_mask = memory_mask | (coerce_bool(tmp[in_memoriam_col]) == True)

    if remembered_name_col is not None:
        memory_mask = memory_mask | nonmissing_string_flag(tmp[remembered_name_col])

    tmp["is_memory_row"] = memory_mask

    memorial_linkage_summary = pd.DataFrame(
        [
            {"metric": "all_rows", "value": int(len(tmp))},
            {"metric": "memory_rows", "value": int(tmp["is_memory_row"].sum())},
            {
                "metric": "memory_rows_with_charity_id_present",
                "value": int((tmp["is_memory_row"] & tmp["charity_linked"]).sum()),
            },
            {
                "metric": "memory_rows_with_charity_id_missing",
                "value": int((tmp["is_memory_row"] & ~tmp["charity_linked"]).sum()),
            },
            {
                "metric": "non_memory_rows_with_charity_id_missing",
                "value": int((~tmp["is_memory_row"] & ~tmp["charity_linked"]).sum()),
            },
        ]
    )

    memorial_by_page_origin = pd.crosstab(
        tmp["is_memory_row"],
        tmp["page_origin"],
        dropna=False,
    ).reset_index()

    memorial_example_cols = [
        c for c in [
            "owner",
            "title",
            "pageShortName",
            "activityType",
            "inMemoriam",
            "rememberedPersonSummary.name",
            "charity_id_clean",
            "charity_name_clean",
            "page_origin",
        ] if c in tmp.columns
    ]

    memorial_rows_missing_charity_id = tmp.loc[
        tmp["is_memory_row"] & ~tmp["charity_linked"], memorial_example_cols
    ].copy()
    sort_cols = [c for c in ["activityType", "owner"] if c in memorial_rows_missing_charity_id.columns]
    if sort_cols:
        memorial_rows_missing_charity_id = memorial_rows_missing_charity_id.sort_values(sort_cols, na_position="last")
    memorial_rows_missing_charity_id = memorial_rows_missing_charity_id.reset_index(drop=True)

    return memorial_linkage_summary, memorial_by_page_origin, memorial_rows_missing_charity_id


# =========================
# EXISTING ANALYSIS HELPERS ----
# =========================
def build_classification_descriptives(classifications: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}

    profile_cols = [c for c in classifications.columns if c.startswith("profile_")]
    categories = [c.replace("profile_", "") for c in profile_cols]

    primary_counts = (
        classifications["primary_category"]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("primary_category")
        .reset_index(name="n")
    )
    primary_counts["pct"] = 100 * primary_counts["n"] / len(classifications)
    out["primary_category_counts"] = primary_counts

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

    num_cat_dist = (
        classifications["num_categories"]
        .value_counts(dropna=False)
        .sort_index()
        .rename_axis("num_categories")
        .reset_index(name="n")
    )
    num_cat_dist["pct"] = 100 * num_cat_dist["n"] / len(classifications)
    out["num_categories_distribution"] = num_cat_dist

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


def build_outcome_by_group(
    merged: pd.DataFrame,
    outcomes: list[str],
    group_col: str,
    restrict_to_classified: bool = True,
) -> pd.DataFrame:
    dat = merged.copy()

    if restrict_to_classified:
        dat = dat.loc[dat["primary_category"].notna()].copy()

    dat = dat.loc[dat[group_col].notna()].copy()

    pieces = []
    for outcome in outcomes:
        tmp = (
            dat.groupby(group_col, dropna=False)[outcome]
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


def build_outcome_by_primary_and_page_origin(
    merged: pd.DataFrame,
    outcomes: list[str],
    allowed_page_origins: tuple[str, ...] = ("charity_created", "individual_for_charity"),
) -> pd.DataFrame:
    dat = merged.loc[
        merged["primary_category"].notna() & merged["page_origin"].isin(allowed_page_origins)
    ].copy()

    pieces = []
    for outcome in outcomes:
        tmp = (
            dat.groupby(["primary_category", "page_origin"], dropna=False)[outcome]
            .agg(
                n="count",
                mean="mean",
                sd="std",
                median="median",
                min="min",
                max="max",
            )
            .reset_index()
            .sort_values(["primary_category", "page_origin"])
            .reset_index(drop=True)
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

    out = out.loc[
        out["outcome"].isin(["log_totalRaisedOnline_plus1", "log_donationCount_plus1"])
    ].copy()

    out["pct_change_per_1pt_profile"] = 100 * (np.exp(out["beta"]) - 1)
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


def build_pairwise_tests_for_group(
    merged: pd.DataFrame,
    outcomes: list[str],
    group_col: str,
    min_n_per_group: int = 20,
    restrict_to_classified: bool = True,
) -> pd.DataFrame:
    dat = merged.copy()

    if restrict_to_classified:
        dat = dat.loc[dat["primary_category"].notna()].copy()

    dat = dat.loc[dat[group_col].notna()].copy()

    rows = []

    for outcome in outcomes:
        tmp = dat.loc[dat[outcome].notna(), [group_col, outcome]].copy()

        counts = tmp[group_col].value_counts()
        valid_groups = counts[counts >= min_n_per_group].index.tolist()
        tmp = tmp.loc[tmp[group_col].isin(valid_groups)].copy()

        groups = sorted(tmp[group_col].astype(str).unique().tolist())

        for a, b in combinations(groups, 2):
            xa = tmp.loc[tmp[group_col] == a, outcome].dropna()
            xb = tmp.loc[tmp[group_col] == b, outcome].dropna()

            if len(xa) < min_n_per_group or len(xb) < min_n_per_group:
                continue

            t_res = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")

            mean_a = xa.mean()
            mean_b = xb.mean()
            diff = mean_a - mean_b

            rows.append(
                {
                    "group_var": group_col,
                    "outcome": outcome,
                    "group_a": a,
                    "group_b": b,
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

    out["p_fdr"] = np.nan
    for outcome in out["outcome"].unique():
        mask = out["outcome"] == outcome
        _, p_adj, _, _ = multipletests(out.loc[mask, "p"], method="fdr_bh")
        out.loc[mask, "p_fdr"] = p_adj

    out = add_sig_stars(out, p_col="p_fdr")
    out["better_group"] = np.where(
        out["mean_diff_a_minus_b"] > 0,
        out["group_a"],
        out["group_b"],
    )

    return out.sort_values(["outcome", "p_fdr", "group_a", "group_b"]).reset_index(drop=True)


def build_pairwise_page_origin_within_primary_category(
    merged: pd.DataFrame,
    outcomes: list[str],
    min_n_per_group: int = 20,
) -> pd.DataFrame:
    dat = merged.loc[
        merged["primary_category"].notna() & merged["page_origin"].isin(["charity_created", "individual_for_charity"])
    ].copy()

    rows = []
    for outcome in outcomes:
        tmp = dat.loc[dat[outcome].notna(), ["primary_category", "page_origin", outcome]].copy()

        for primary_cat in sorted(tmp["primary_category"].dropna().astype(str).unique().tolist()):
            sub = tmp.loc[tmp["primary_category"] == primary_cat].copy()
            counts = sub["page_origin"].value_counts()

            if not {"charity_created", "individual_for_charity"}.issubset(set(counts.index)):
                continue
            if counts["charity_created"] < min_n_per_group or counts["individual_for_charity"] < min_n_per_group:
                continue

            xa = sub.loc[sub["page_origin"] == "charity_created", outcome].dropna()
            xb = sub.loc[sub["page_origin"] == "individual_for_charity", outcome].dropna()

            t_res = stats.ttest_ind(xa, xb, equal_var=False, nan_policy="omit")
            mean_a = xa.mean()
            mean_b = xb.mean()

            rows.append(
                {
                    "outcome": outcome,
                    "primary_category": primary_cat,
                    "group_a": "charity_created",
                    "group_b": "individual_for_charity",
                    "n_a": int(len(xa)),
                    "n_b": int(len(xb)),
                    "mean_a": float(mean_a),
                    "mean_b": float(mean_b),
                    "mean_diff_a_minus_b": float(mean_a - mean_b),
                    "t": float(t_res.statistic),
                    "p": float(t_res.pvalue),
                    "better_group": "charity_created" if mean_a > mean_b else "individual_for_charity",
                }
            )

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["p_fdr"] = np.nan
    for outcome in out["outcome"].unique():
        mask = out["outcome"] == outcome
        _, p_adj, _, _ = multipletests(out.loc[mask, "p"], method="fdr_bh")
        out.loc[mask, "p_fdr"] = p_adj

    out = add_sig_stars(out, p_col="p_fdr")
    return out.sort_values(["outcome", "p_fdr", "primary_category"]).reset_index(drop=True)


def _interaction_combo_test(
    model,
    main_name: str,
    interaction_name: str | None,
) -> dict[str, float]:
    if interaction_name is None:
        beta = float(model.params[main_name])
        se = float(model.bse[main_name])
        ci_low, ci_high = model.conf_int().loc[main_name].tolist()
        t = float(model.tvalues[main_name])
        p = float(model.pvalues[main_name])
        return {
            "beta": beta,
            "se": se,
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "t": t,
            "p": p,
        }

    R = np.zeros(len(model.params))
    idx = {name: i for i, name in enumerate(model.params.index)}
    R[idx[main_name]] = 1.0
    R[idx[interaction_name]] = 1.0
    test = model.t_test(R)
    effect = float(test.effect.squeeze())
    se = float(test.sd.squeeze())
    t_val = float(test.tvalue.squeeze())
    p_val = float(test.pvalue.squeeze())
    ci = test.conf_int()
    ci_low = float(np.asarray(ci).reshape(-1, 2)[0, 0])
    ci_high = float(np.asarray(ci).reshape(-1, 2)[0, 1])
    return {
        "beta": effect,
        "se": se,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "t": t_val,
        "p": p_val,
    }


def build_profile_page_origin_interaction_regressions(
    merged: pd.DataFrame,
    outcomes: list[str],
    min_n: int = 30,
) -> pd.DataFrame:
    dat = merged.loc[
        merged["primary_category"].notna() & merged["page_origin"].isin(["charity_created", "individual_for_charity"])
    ].copy()
    dat["page_origin"] = pd.Categorical(
        dat["page_origin"],
        categories=["charity_created", "individual_for_charity"],
    )

    profile_cols = [c for c in dat.columns if c.startswith("profile_")]
    rows = []

    for outcome in outcomes:
        for predictor in profile_cols:
            sub = dat[[outcome, predictor, "page_origin"]].dropna().copy()
            if len(sub) < min_n:
                continue
            if sub[predictor].nunique() < 2 or sub["page_origin"].nunique() < 2:
                continue

            counts = sub["page_origin"].value_counts()
            if counts.min() < min_n:
                continue

            sub = sub.rename(columns={predictor: "x", outcome: "y"})
            model = smf.ols(
                'y ~ x * C(page_origin, Treatment(reference="charity_created"))',
                data=sub,
            ).fit()

            interaction_term = None
            for name in model.params.index:
                if name.startswith("x:C(page_origin"):
                    interaction_term = name
                    break
            page_origin_term = None
            for name in model.params.index:
                if name.startswith("C(page_origin"):
                    page_origin_term = name
                    break

            slope_cc = _interaction_combo_test(model, "x", None)
            slope_ifc = _interaction_combo_test(model, "x", interaction_term)

            interaction_beta = float(model.params.get(interaction_term, np.nan)) if interaction_term else np.nan
            interaction_se = float(model.bse.get(interaction_term, np.nan)) if interaction_term else np.nan
            interaction_t = float(model.tvalues.get(interaction_term, np.nan)) if interaction_term else np.nan
            interaction_p = float(model.pvalues.get(interaction_term, np.nan)) if interaction_term else np.nan
            interaction_ci_low = np.nan
            interaction_ci_high = np.nan
            if interaction_term:
                interaction_ci_low, interaction_ci_high = model.conf_int().loc[interaction_term].tolist()

            intercept_diff_beta = float(model.params.get(page_origin_term, np.nan)) if page_origin_term else np.nan
            intercept_diff_se = float(model.bse.get(page_origin_term, np.nan)) if page_origin_term else np.nan
            intercept_diff_t = float(model.tvalues.get(page_origin_term, np.nan)) if page_origin_term else np.nan
            intercept_diff_p = float(model.pvalues.get(page_origin_term, np.nan)) if page_origin_term else np.nan

            row = {
                "outcome": outcome,
                "motivation": predictor.replace("profile_", ""),
                "predictor": predictor,
                "n": int(model.nobs),
                "n_charity_created": int(counts.get("charity_created", 0)),
                "n_individual_for_charity": int(counts.get("individual_for_charity", 0)),
                "r2": float(model.rsquared),
                "adj_r2": float(model.rsquared_adj),
                "beta_charity_created": slope_cc["beta"],
                "se_charity_created": slope_cc["se"],
                "t_charity_created": slope_cc["t"],
                "p_charity_created": slope_cc["p"],
                "ci_low_charity_created": slope_cc["ci_low"],
                "ci_high_charity_created": slope_cc["ci_high"],
                "beta_individual_for_charity": slope_ifc["beta"],
                "se_individual_for_charity": slope_ifc["se"],
                "t_individual_for_charity": slope_ifc["t"],
                "p_individual_for_charity": slope_ifc["p"],
                "ci_low_individual_for_charity": slope_ifc["ci_low"],
                "ci_high_individual_for_charity": slope_ifc["ci_high"],
                "interaction_beta": interaction_beta,
                "interaction_se": interaction_se,
                "interaction_t": interaction_t,
                "interaction_p": interaction_p,
                "interaction_ci_low": interaction_ci_low,
                "interaction_ci_high": interaction_ci_high,
                "page_origin_intercept_diff_beta": intercept_diff_beta,
                "page_origin_intercept_diff_se": intercept_diff_se,
                "page_origin_intercept_diff_t": intercept_diff_t,
                "page_origin_intercept_diff_p": intercept_diff_p,
            }
            rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out

    out["interaction_sig"] = np.select(
        [
            out["interaction_p"] < 0.001,
            out["interaction_p"] < 0.01,
            out["interaction_p"] < 0.05,
            out["interaction_p"] < 0.10,
        ],
        ["***", "**", "*", "."],
        default="",
    )
    return out.sort_values(["outcome", "interaction_p", "motivation"]).reset_index(drop=True)


def build_interaction_percent_change_table(interaction_results: pd.DataFrame) -> pd.DataFrame:
    out = interaction_results.copy()
    out = out.loc[out["outcome"].isin(["log_totalRaisedOnline_plus1", "log_donationCount_plus1"])].copy()

    def pct(beta):
        return 100 * (np.exp(0.10 * beta) - 1)

    out["pct_change_10pp_charity_created"] = pct(out["beta_charity_created"])
    out["pct_change_10pp_individual_for_charity"] = pct(out["beta_individual_for_charity"])
    out["pct_change_10pp_interaction_gap"] = (
        out["pct_change_10pp_individual_for_charity"] - out["pct_change_10pp_charity_created"]
    )
    out["pct_change_10pp_ci_low_charity_created"] = pct(out["ci_low_charity_created"])
    out["pct_change_10pp_ci_high_charity_created"] = pct(out["ci_high_charity_created"])
    out["pct_change_10pp_ci_low_individual_for_charity"] = pct(out["ci_low_individual_for_charity"])
    out["pct_change_10pp_ci_high_individual_for_charity"] = pct(out["ci_high_individual_for_charity"])

    return out[
        [
            "outcome",
            "motivation",
            "n",
            "beta_charity_created",
            "p_charity_created",
            "beta_individual_for_charity",
            "p_individual_for_charity",
            "interaction_beta",
            "interaction_p",
            "interaction_sig",
            "pct_change_10pp_charity_created",
            "pct_change_10pp_individual_for_charity",
            "pct_change_10pp_interaction_gap",
            "pct_change_10pp_ci_low_charity_created",
            "pct_change_10pp_ci_high_charity_created",
            "pct_change_10pp_ci_low_individual_for_charity",
            "pct_change_10pp_ci_high_individual_for_charity",
            "r2",
            "adj_r2",
        ]
    ].sort_values(["outcome", "interaction_p", "motivation"]).reset_index(drop=True)


def build_interaction_summary_table(interaction_results: pd.DataFrame) -> pd.DataFrame:
    out = interaction_results.copy()
    out["stronger_in"] = np.where(
        out["interaction_beta"] > 0,
        "individual_for_charity",
        np.where(out["interaction_beta"] < 0, "charity_created", "no_difference"),
    )
    out["summary"] = out["stronger_in"] + out["interaction_sig"].map(
        lambda s: f" ({s})" if s else ""
    )
    table = (
        out.pivot(index="motivation", columns="outcome", values="summary")
        .reset_index()
        .rename_axis(None, axis=1)
    )
    return table


def build_group_by_primary_category(
    merged: pd.DataFrame,
    group_col: str,
) -> pd.DataFrame:
    dat = merged.loc[merged["primary_category"].notna()].copy()
    return pd.crosstab(
        dat[group_col],
        dat["primary_category"],
        dropna=False,
    ).reset_index()


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

    return (
        summary.pivot(index="motivation", columns="outcome", values="summary")
        .reset_index()
        .rename_axis(None, axis=1)
    )


# =========================
# MAIN ----
# =========================
def main():
    print("Reading classified files...")
    classifications = pd.read_csv(CLASSIFICATIONS_PATH, low_memory=False)
    merged = pd.read_csv(MERGED_PATH, low_memory=False)

    print(f"Classified unique stories: {len(classifications):,}")
    print(f"Merged rows: {len(merged):,}")

    print("\nAdding page-origin classification from merged dataset...")
    merged = add_page_origin(merged)

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
    by_primary = build_outcome_by_group(
        merged=merged,
        outcomes=outcomes,
        group_col="primary_category",
        restrict_to_classified=True,
    )

    overall_outcomes_path = ANALYSIS_DIR / "a05_outcome_descriptives_overall.csv"
    by_primary_path = ANALYSIS_DIR / "a06_outcome_by_primary_category.csv"

    overall_outcomes.to_csv(overall_outcomes_path, index=False)
    by_primary.to_csv(by_primary_path, index=False)

    print("\n2B) Page-origin descriptives")
    page_origin_counts = build_page_origin_counts(merged)
    page_origin_by_activity = build_page_origin_by_activity(merged)
    by_page_origin_all = build_outcome_by_group(
        merged=merged,
        outcomes=outcomes,
        group_col="page_origin",
        restrict_to_classified=False,
    )
    by_page_origin_classified = build_outcome_by_group(
        merged=merged,
        outcomes=outcomes,
        group_col="page_origin",
        restrict_to_classified=True,
    )
    by_primary_and_page_origin = build_outcome_by_primary_and_page_origin(
        merged=merged,
        outcomes=outcomes,
    )
    primary_by_page_origin = build_group_by_primary_category(merged, "page_origin")

    page_origin_counts_path = ANALYSIS_DIR / "a06b_page_origin_counts.csv"
    page_origin_by_activity_path = ANALYSIS_DIR / "a06c_page_origin_by_activity_type.csv"
    by_page_origin_all_path = ANALYSIS_DIR / "a06d_outcome_by_page_origin_all_rows.csv"
    by_page_origin_classified_path = ANALYSIS_DIR / "a06e_outcome_by_page_origin_classified_rows.csv"
    primary_by_page_origin_path = ANALYSIS_DIR / "a06f_primary_category_by_page_origin.csv"
    merged_with_page_origin_path = ANALYSIS_DIR / "a06g_merged_with_page_origin.csv"
    by_primary_and_page_origin_path = ANALYSIS_DIR / "a06p_outcome_by_primary_category_and_page_origin.csv"

    page_origin_counts.to_csv(page_origin_counts_path, index=False)
    page_origin_by_activity.to_csv(page_origin_by_activity_path, index=False)
    by_page_origin_all.to_csv(by_page_origin_all_path, index=False)
    by_page_origin_classified.to_csv(by_page_origin_classified_path, index=False)
    primary_by_page_origin.to_csv(primary_by_page_origin_path, index=False)
    merged.to_csv(merged_with_page_origin_path, index=False)
    by_primary_and_page_origin.to_csv(by_primary_and_page_origin_path, index=False)

    print(page_origin_counts.to_string(index=False))

    print("\n2C) Charity-name diagnostics")
    (
        unique_charity_names,
        charity_name_counts,
        charity_id_name_lookup,
        charity_name_status_by_missing_id,
        charity_problem_rows,
    ) = build_charity_outputs(merged)

    unique_charity_names_path = ANALYSIS_DIR / "a06h_unique_charity_names.csv"
    charity_name_counts_path = ANALYSIS_DIR / "a06i_charity_name_counts.csv"
    charity_id_name_lookup_path = ANALYSIS_DIR / "a06j_charity_id_name_lookup.csv"
    charity_name_status_by_missing_id_path = ANALYSIS_DIR / "a06k_charity_name_status_by_missing_id.csv"
    charity_problem_rows_path = ANALYSIS_DIR / "a06l_charity_problem_rows.csv"

    unique_charity_names.to_csv(unique_charity_names_path, index=False)
    charity_name_counts.to_csv(charity_name_counts_path, index=False)
    charity_id_name_lookup.to_csv(charity_id_name_lookup_path, index=False)
    charity_name_status_by_missing_id.to_csv(charity_name_status_by_missing_id_path, index=False)
    charity_problem_rows.to_csv(charity_problem_rows_path, index=False)

    print(f"Unique charity names: {len(unique_charity_names):,}")
    if len(charity_name_counts) > 0:
        print("\nTop 20 charity names:")
        print(charity_name_counts.head(20).to_string(index=False))

    print("\n2D) Memorial / in-memory diagnostics")
    (
        memorial_linkage_summary,
        memorial_by_page_origin,
        memorial_rows_missing_charity_id,
    ) = build_memorial_outputs(merged)

    memorial_linkage_summary_path = ANALYSIS_DIR / "a06m_memorial_linkage_summary.csv"
    memorial_by_page_origin_path = ANALYSIS_DIR / "a06n_memorial_by_page_origin.csv"
    memorial_rows_missing_charity_id_path = ANALYSIS_DIR / "a06o_memorial_rows_missing_charity_id.csv"

    memorial_linkage_summary.to_csv(memorial_linkage_summary_path, index=False)
    memorial_by_page_origin.to_csv(memorial_by_page_origin_path, index=False)
    memorial_rows_missing_charity_id.to_csv(memorial_rows_missing_charity_id_path, index=False)

    print(memorial_linkage_summary.to_string(index=False))

    print("\n3) Profile regressions")
    reg_results = rerun_profile_regressions(merged, outcomes)
    reg_results_path = ANALYSIS_DIR / "a07_regression_results.csv"
    reg_results.to_csv(reg_results_path, index=False)

    print("\n4) Percent-change interpretation table")
    pct_change = build_percent_change_table(reg_results)
    pct_change_path = ANALYSIS_DIR / "a08_percent_change_interpretation.csv"
    pct_change.to_csv(pct_change_path, index=False)

    print("\n5) Pairwise comparisons between primary categories")
    pairwise_primary = build_pairwise_tests_for_group(
        merged=merged,
        outcomes=outcomes,
        group_col="primary_category",
        min_n_per_group=20,
        restrict_to_classified=True,
    )
    pairwise_primary_path = ANALYSIS_DIR / "a09_pairwise_primary_category_tests.csv"
    pairwise_primary.to_csv(pairwise_primary_path, index=False)

    print("\n5B) Pairwise comparisons between page-origin groups")
    pairwise_page_origin_all = build_pairwise_tests_for_group(
        merged=merged,
        outcomes=outcomes,
        group_col="page_origin",
        min_n_per_group=20,
        restrict_to_classified=False,
    )
    pairwise_page_origin_classified = build_pairwise_tests_for_group(
        merged=merged,
        outcomes=outcomes,
        group_col="page_origin",
        min_n_per_group=20,
        restrict_to_classified=True,
    )
    pairwise_page_origin_within_primary = build_pairwise_page_origin_within_primary_category(
        merged=merged,
        outcomes=outcomes,
        min_n_per_group=20,
    )

    pairwise_page_origin_all_path = ANALYSIS_DIR / "a09b_pairwise_page_origin_tests_all_rows.csv"
    pairwise_page_origin_classified_path = ANALYSIS_DIR / "a09c_pairwise_page_origin_tests_classified_rows.csv"
    pairwise_page_origin_within_primary_path = ANALYSIS_DIR / "a09d_pairwise_page_origin_tests_within_primary_category.csv"

    pairwise_page_origin_all.to_csv(pairwise_page_origin_all_path, index=False)
    pairwise_page_origin_classified.to_csv(pairwise_page_origin_classified_path, index=False)
    pairwise_page_origin_within_primary.to_csv(pairwise_page_origin_within_primary_path, index=False)

    print("\n6) Compact summary table")
    summary_table = build_summary_table(reg_results)
    summary_path = ANALYSIS_DIR / "a10_summary_table.csv"
    summary_table.to_csv(summary_path, index=False)

    print("\n7) Motivation × page-origin interaction regressions")
    interaction_results = build_profile_page_origin_interaction_regressions(
        merged=merged,
        outcomes=outcomes,
        min_n=30,
    )
    interaction_results_path = ANALYSIS_DIR / "a11_interaction_regression_results.csv"
    interaction_results.to_csv(interaction_results_path, index=False)

    interaction_pct_change = build_interaction_percent_change_table(interaction_results)
    interaction_pct_change_path = ANALYSIS_DIR / "a12_interaction_percent_change_table.csv"
    interaction_pct_change.to_csv(interaction_pct_change_path, index=False)

    interaction_summary = build_interaction_summary_table(interaction_results)
    interaction_summary_path = ANALYSIS_DIR / "a13_interaction_summary_table.csv"
    interaction_summary.to_csv(interaction_summary_path, index=False)

    print("\nDone.")
    print("\nSaved:")
    print(f"- {primary_counts_path}")
    print(f"- {active_prevalence_path}")
    print(f"- {num_categories_path}")
    print(f"- {profile_weights_path}")
    print(f"- {overall_outcomes_path}")
    print(f"- {by_primary_path}")
    print(f"- {page_origin_counts_path}")
    print(f"- {page_origin_by_activity_path}")
    print(f"- {by_page_origin_all_path}")
    print(f"- {by_page_origin_classified_path}")
    print(f"- {primary_by_page_origin_path}")
    print(f"- {merged_with_page_origin_path}")
    print(f"- {by_primary_and_page_origin_path}")
    print(f"- {unique_charity_names_path}")
    print(f"- {charity_name_counts_path}")
    print(f"- {charity_id_name_lookup_path}")
    print(f"- {charity_name_status_by_missing_id_path}")
    print(f"- {charity_problem_rows_path}")
    print(f"- {memorial_linkage_summary_path}")
    print(f"- {memorial_by_page_origin_path}")
    print(f"- {memorial_rows_missing_charity_id_path}")
    print(f"- {reg_results_path}")
    print(f"- {pct_change_path}")
    print(f"- {pairwise_primary_path}")
    print(f"- {pairwise_page_origin_all_path}")
    print(f"- {pairwise_page_origin_classified_path}")
    print(f"- {pairwise_page_origin_within_primary_path}")
    print(f"- {summary_path}")
    print(f"- {interaction_results_path}")
    print(f"- {interaction_pct_change_path}")
    print(f"- {interaction_summary_path}")


if __name__ == "__main__":
    main()
