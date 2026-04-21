from __future__ import annotations

import pytest

from ads_project.monitoring import calibration_bin_frame, calibration_drift_report


def test_calibration_drift_report_tracks_summary_deltas() -> None:
    report = calibration_drift_report(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        [0, 1, 1, 1],
        [0.2, 0.5, 0.7, 0.8],
        bins=2,
    )

    assert report["reference_rows"] == 4
    assert report["current_rows"] == 4
    assert report["current_positive_rate"] == pytest.approx(0.75)
    assert report["positive_rate_delta"] == pytest.approx(0.25)
    assert report["calibration_mae_delta"] > 0

    bin_frame = calibration_bin_frame(report)
    assert list(bin_frame.columns) == [
        "bin",
        "reference_rows",
        "current_rows",
        "reference_positives",
        "current_positives",
        "reference_avg_score",
        "current_avg_score",
        "avg_score_delta",
        "reference_actual_rate",
        "current_actual_rate",
        "actual_rate_delta",
    ]


def test_calibration_drift_report_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        calibration_drift_report(
            [0, 1],
            [0.2, 1.2],
            [0, 1],
            [0.2, 0.8],
        )
