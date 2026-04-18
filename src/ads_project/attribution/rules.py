from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_ATTRIBUTION_COLUMNS = (
    "campaign",
    "conversion",
    "click",
    "click_pos",
    "click_nb",
    "cost",
    "cpo",
)


def validate_attribution_source(df: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_ATTRIBUTION_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"attribution source is missing required columns: {missing_columns}")


def compute_last_touch_attribution(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            (df["conversion"] == 1) & (df["click"] == 1) & (df["click_pos"] == df["click_nb"] - 1),
            1.0,
            0.0,
        ),
        index=df.index,
        name="last_touch",
    )


def compute_linear_multi_touch_attribution(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.where(
            (df["conversion"] == 1) & (df["click"] == 1) & (df["click_nb"] > 0),
            1.0 / df["click_nb"],
            0.0,
        ),
        index=df.index,
        name="multi_touch_linear",
    )


def compute_time_decay_attribution(
    df: pd.DataFrame,
    *,
    decay_rate: float = 0.5,
) -> pd.Series:
    if not 0 < decay_rate <= 1:
        raise ValueError("decay_rate must be between 0 and 1")

    valid_click_path = (df["conversion"] == 1) & (df["click"] == 1) & (df["click_nb"] > 0)
    gap_to_last = (df["click_nb"] - 1 - df["click_pos"]).clip(lower=0)

    if decay_rate == 1.0:
        normalized_weight = np.where(valid_click_path, 1.0 / df["click_nb"], 0.0)
    else:
        raw_weight = np.power(decay_rate, gap_to_last)
        normalization = (1.0 - np.power(decay_rate, df["click_nb"])) / (1.0 - decay_rate)
        normalized_weight = np.where(valid_click_path, raw_weight / normalization, 0.0)

    return pd.Series(normalized_weight, index=df.index, name="time_decay")


def build_campaign_attribution_report(
    df: pd.DataFrame,
    *,
    decay_rate: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    validate_attribution_source(df)

    attribution_df = df.copy()
    attribution_df["last_touch"] = compute_last_touch_attribution(attribution_df)
    attribution_df["multi_touch_linear"] = compute_linear_multi_touch_attribution(attribution_df)
    attribution_df["time_decay"] = compute_time_decay_attribution(
        attribution_df,
        decay_rate=decay_rate,
    )

    campaign_report = (
        attribution_df.groupby("campaign", observed=True)
        .agg(
            rows=("campaign", "size"),
            clicks=("click", "sum"),
            conversions=("conversion", "sum"),
            total_cost=("cost", "sum"),
            mean_cost=("cost", "mean"),
            mean_cpo=("cpo", "mean"),
            ctr=("click", "mean"),
            last_touch=("last_touch", "sum"),
            multi_touch_linear=("multi_touch_linear", "sum"),
            time_decay=("time_decay", "sum"),
        )
        .reset_index()
    )
    campaign_report["attribution_diff"] = (
        campaign_report["multi_touch_linear"] - campaign_report["last_touch"]
    )
    campaign_report["time_decay_diff_vs_last_touch"] = (
        campaign_report["time_decay"] - campaign_report["last_touch"]
    )
    campaign_report["time_decay_diff_vs_linear"] = (
        campaign_report["time_decay"] - campaign_report["multi_touch_linear"]
    )
    campaign_report["last_touch_share"] = (
        campaign_report["last_touch"] / campaign_report["conversions"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["multi_touch_linear_share"] = (
        campaign_report["multi_touch_linear"] / campaign_report["conversions"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["time_decay_share"] = (
        campaign_report["time_decay"] / campaign_report["conversions"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["roi_proxy_linear"] = (
        campaign_report["multi_touch_linear"] / campaign_report["mean_cpo"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["roi_proxy_time_decay"] = (
        campaign_report["time_decay"] / campaign_report["mean_cpo"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["cost_per_click"] = (
        campaign_report["total_cost"] / campaign_report["clicks"].replace(0, np.nan)
    ).fillna(0.0)
    campaign_report["conversion_rate"] = (
        campaign_report["conversions"] / campaign_report["rows"].replace(0, np.nan)
    ).fillna(0.0)

    campaign_report = campaign_report.sort_values(
        ["conversions", "clicks", "campaign"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    summary = {
        "rows": int(len(attribution_df)),
        "campaigns": int(campaign_report["campaign"].nunique()),
        "total_clicks": int(attribution_df["click"].sum()),
        "total_conversions": int(attribution_df["conversion"].sum()),
        "total_last_touch": float(campaign_report["last_touch"].sum()),
        "total_multi_touch_linear": float(campaign_report["multi_touch_linear"].sum()),
        "total_time_decay": float(campaign_report["time_decay"].sum()),
        "time_decay_rate": float(decay_rate),
    }
    return campaign_report, summary


def build_decision_facing_campaign_report(
    campaign_report: pd.DataFrame,
    *,
    top_campaigns: int,
) -> tuple[pd.DataFrame, dict[str, list[dict[str, float | int]]]]:
    decision_report = campaign_report.copy()
    decision_report["spend_share"] = (
        decision_report["total_cost"] / decision_report["total_cost"].sum()
    ).fillna(0.0)
    decision_report["conversion_share"] = (
        decision_report["conversions"] / decision_report["conversions"].sum()
    ).fillna(0.0)
    decision_report["click_share"] = (
        decision_report["clicks"] / decision_report["clicks"].sum()
    ).fillna(0.0)
    decision_report["attributed_conversion_share_time_decay"] = (
        decision_report["time_decay"] / decision_report["time_decay"].sum()
    ).fillna(0.0)
    decision_report["efficiency_index_time_decay"] = (
        decision_report["attributed_conversion_share_time_decay"]
        / decision_report["spend_share"].replace(0, np.nan)
    ).fillna(0.0)
    decision_report["efficiency_index_linear"] = (
        decision_report["multi_touch_linear_share"] / decision_report["spend_share"].replace(0, np.nan)
    ).fillna(0.0)
    decision_report["roi_proxy_gap_time_decay_vs_linear"] = (
        decision_report["roi_proxy_time_decay"] - decision_report["roi_proxy_linear"]
    )
    decision_report["priority_bucket"] = np.select(
        [
            (decision_report["efficiency_index_time_decay"] >= 1.2) & (decision_report["conversions"] >= 100),
            (decision_report["efficiency_index_time_decay"] <= 0.8) & (decision_report["conversions"] >= 100),
        ],
        [
            "scale_candidate",
            "review_candidate",
        ],
        default="monitor",
    )

    decision_columns = [
        "campaign",
        "priority_bucket",
        "rows",
        "clicks",
        "conversions",
        "total_cost",
        "ctr",
        "conversion_rate",
        "last_touch",
        "multi_touch_linear",
        "time_decay",
        "roi_proxy_linear",
        "roi_proxy_time_decay",
        "roi_proxy_gap_time_decay_vs_linear",
        "spend_share",
        "click_share",
        "conversion_share",
        "attributed_conversion_share_time_decay",
        "efficiency_index_linear",
        "efficiency_index_time_decay",
    ]
    decision_report = decision_report[decision_columns].sort_values(
        ["efficiency_index_time_decay", "conversions", "campaign"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    review_views = {
        "top_scale_candidates": decision_report[decision_report["priority_bucket"] == "scale_candidate"]
        .head(top_campaigns)
        .to_dict(orient="records"),
        "top_review_candidates": decision_report[decision_report["priority_bucket"] == "review_candidate"]
        .head(top_campaigns)
        .to_dict(orient="records"),
        "top_monitor_by_spend": decision_report.sort_values(
            ["spend_share", "campaign"],
            ascending=[False, True],
        )
        .head(top_campaigns)
        .to_dict(orient="records"),
    }
    return decision_report, review_views
