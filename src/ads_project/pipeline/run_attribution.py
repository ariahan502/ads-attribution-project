from __future__ import annotations

import argparse
from pathlib import Path

from ads_project.artifacts import (
    build_run_manifest,
    current_git_commit,
    make_run_dir,
    write_csv,
    write_json,
    write_yaml,
)
from ads_project.attribution import build_campaign_attribution_report, build_decision_facing_campaign_report
from ads_project.config import load_yaml_config
from ads_project.data.io import read_tabular


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline attribution summaries from config.")
    parser.add_argument(
        "--config",
        default="configs/attribution_smoke.yaml",
        help="Path to YAML config for attribution.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_yaml_config(config_path)

    dataset_path = Path(config["dataset_path"])
    output_dir = Path(config.get("output_dir", "artifacts/runs"))
    run_name = str(config.get("run_name", "attribution"))
    top_campaigns = int(config.get("top_campaigns", 20))
    max_rows = config.get("max_rows")
    time_decay_rate = float(config.get("time_decay_rate", 0.5))

    print(f"Loading attribution data from: {dataset_path}")
    df = read_tabular(dataset_path, nrows=int(max_rows) if max_rows is not None else None)
    print(f"Loaded rows: {len(df)}")

    campaign_report, summary = build_campaign_attribution_report(
        df,
        decay_rate=time_decay_rate,
    )
    top_diff = (
        campaign_report.sort_values("attribution_diff", ascending=False)
        .head(top_campaigns)
        .to_dict(orient="records")
    )
    top_time_decay_diff = (
        campaign_report.sort_values("time_decay_diff_vs_linear", ascending=False)
        .head(top_campaigns)
        .to_dict(orient="records")
    )
    top_roi = (
        campaign_report.sort_values("roi_proxy_time_decay", ascending=False)
        .head(top_campaigns)
        .to_dict(orient="records")
    )
    summary_columns = [
        "campaign",
        "rows",
        "clicks",
        "conversions",
        "total_cost",
        "ctr",
        "conversion_rate",
        "last_touch",
        "multi_touch_linear",
        "time_decay",
        "attribution_diff",
        "time_decay_diff_vs_last_touch",
        "time_decay_diff_vs_linear",
        "roi_proxy_linear",
        "roi_proxy_time_decay",
    ]
    campaign_summary = campaign_report[summary_columns].copy()
    decision_report, decision_views = build_decision_facing_campaign_report(
        campaign_report,
        top_campaigns=top_campaigns,
    )

    run_dir = make_run_dir(output_dir, run_name=run_name)
    write_yaml(config, run_dir / "config.yaml")
    write_json(summary, run_dir / "summary.json")
    write_json(
        {
            "campaigns": campaign_summary.to_dict(orient="records"),
        },
        run_dir / "campaign_summary.json",
    )
    write_csv(campaign_summary, run_dir / "campaign_summary.csv")
    write_json(
        {
            "campaigns": decision_report.to_dict(orient="records"),
        },
        run_dir / "campaign_decision_report.json",
    )
    write_csv(decision_report, run_dir / "campaign_decision_report.csv")
    write_json(
        {
            "top_campaigns_by_conversions": campaign_summary.head(top_campaigns).to_dict(orient="records"),
            "top_campaigns_by_diff": top_diff,
            "top_campaigns_by_time_decay_diff_vs_linear": top_time_decay_diff,
            "top_campaigns_by_roi_proxy_time_decay": top_roi,
            **decision_views,
        },
        run_dir / "campaign_report.json",
    )
    manifest = build_run_manifest(
        run_dir=run_dir,
        run_name=run_name,
        pipeline_name="run_attribution",
        config_path=config_path,
        dataset_path=dataset_path,
        train_rows=0,
        validation_rows=None,
        test_rows=len(df),
        metrics=summary,
        validation_metrics=None,
        git_commit=current_git_commit(cwd=Path.cwd()),
        artifacts={
            "config": "config.yaml",
            "summary": "summary.json",
            "campaign_summary_json": "campaign_summary.json",
            "campaign_summary_csv": "campaign_summary.csv",
            "campaign_decision_report_json": "campaign_decision_report.json",
            "campaign_decision_report_csv": "campaign_decision_report.csv",
            "campaign_report": "campaign_report.json",
        },
        extra_metadata={
            "top_campaigns": top_campaigns,
            "report_type": "decision_support_attribution",
            "time_decay_rate": time_decay_rate,
            "schemes": ["last_touch", "multi_touch_linear", "time_decay"],
        },
    )
    write_json(manifest, run_dir / "manifest.json")

    print(f"Campaigns: {summary['campaigns']}")
    print(f"Total conversions: {summary['total_conversions']}")
    print(f"Total last-touch attribution: {summary['total_last_touch']:.6f}")
    print(f"Total linear attribution: {summary['total_multi_touch_linear']:.6f}")
    print(f"Total time-decay attribution: {summary['total_time_decay']:.6f}")
    print(f"Saved attribution artifacts to: {run_dir}")


if __name__ == "__main__":
    main()
