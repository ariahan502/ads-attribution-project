from __future__ import annotations

import pandas as pd
import pytest

from ads_project.monitoring import categorical_drift_report, numeric_drift_report


def test_numeric_drift_report_computes_summary_and_psi() -> None:
    reference = pd.DataFrame({"score": [0.1, 0.2, 0.3, 0.4]})
    current = pd.DataFrame({"score": [0.2, 0.3, 0.5, 0.6]})

    report = numeric_drift_report(reference, current, columns=["score"], bins=2)

    assert len(report) == 1
    row = report[0]
    assert row["column"] == "score"
    assert row["kind"] == "numeric"
    assert row["reference_mean"] == pytest.approx(0.25)
    assert row["current_mean"] == pytest.approx(0.4)
    assert row["mean_delta"] == pytest.approx(0.15)
    assert row["psi"] is not None


def test_categorical_drift_report_tracks_new_category_share() -> None:
    reference = pd.DataFrame({"campaign": [1, 1, 2, 2]})
    current = pd.DataFrame({"campaign": [1, 2, 3, 3]})

    report = categorical_drift_report(reference, current, columns=["campaign"], top_n=3)

    assert len(report) == 1
    row = report[0]
    assert row["column"] == "campaign"
    assert row["kind"] == "categorical"
    assert row["new_category_share"] == pytest.approx(0.5)
    assert row["reference_unique"] == 2
    assert row["current_unique"] == 3
    assert row["psi"] is not None


def test_numeric_drift_report_rejects_missing_column() -> None:
    with pytest.raises(ValueError, match="missing drift column"):
        numeric_drift_report(
            pd.DataFrame({"x": [1]}),
            pd.DataFrame({"y": [1]}),
            columns=["x"],
        )
