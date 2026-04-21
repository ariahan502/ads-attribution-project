from __future__ import annotations

import pandas as pd
import pytest

from ads_project.policy import batch_score_summary, build_batch_score_output


def test_build_batch_score_output_ranks_and_flags_recommended_rows() -> None:
    df = pd.DataFrame(
        {
            "uid": [1, 2, 3, 4],
            "timestamp": [10, 20, 30, 40],
            "campaign": [100, 100, 101, 101],
            "true_treatment_effect": [0.1, 0.4, 0.2, 0.3],
        }
    )
    score_df = pd.DataFrame(
        {
            "observational_score": [0.2, 0.6, 0.1, 0.5],
            "doubly_robust_score": [0.1, 0.9, 0.2, 0.8],
        }
    )

    scored = build_batch_score_output(
        df,
        score_df=score_df,
        id_columns=["uid", "timestamp", "campaign"],
        score_cols=["observational_score", "doubly_robust_score"],
        preferred_score_col="doubly_robust_score",
        recommended_top_fraction=0.5,
        optional_columns=["true_treatment_effect"],
    )

    assert scored["uid"].tolist() == [2, 4, 3, 1]
    assert scored["policy_score_rank"].tolist() == [1, 2, 3, 4]
    assert scored["recommended_policy"].tolist() == [1, 1, 0, 0]
    assert scored["policy_score_percentile"].tolist() == [1.0, 0.75, 0.5, 0.25]

    summary = batch_score_summary(
        scored,
        preferred_score_col="doubly_robust_score",
        recommended_top_fraction=0.5,
        true_effect_col="true_treatment_effect",
    )
    assert summary["rows"] == 4
    assert summary["recommended_rows"] == 2
    assert summary["recommended_expected_incremental_conversions"] == pytest.approx(0.7)
    assert summary["recommended_expected_incremental_conversions_per_1k"] == pytest.approx(350.0)


def test_build_batch_score_output_rejects_invalid_top_fraction() -> None:
    with pytest.raises(ValueError, match="recommended_top_fraction"):
        build_batch_score_output(
            pd.DataFrame({"uid": [1]}),
            score_df=pd.DataFrame({"score": [0.1]}),
            id_columns=["uid"],
            score_cols=["score"],
            preferred_score_col="score",
            recommended_top_fraction=0.0,
        )
