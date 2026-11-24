from recon_engine import anomaly, evaluate


def test_detectors_find_most_planted_outliers(frames):
    flagged = anomaly.detect(frames["bank"])
    scores = evaluate.anomaly_scores(flagged, frames["ground_truth"]).set_index("detector")
    assert scores.loc["zscore", "recall"] >= 0.8
    assert scores.loc["zscore", "precision"] >= 0.7
    assert scores.loc["iforest", "recall"] >= 0.8
    assert scores.loc["iforest", "precision"] >= 0.6


def test_flag_columns_are_boolean_and_aligned(frames):
    flagged = anomaly.detect(frames["bank"])
    assert len(flagged) == len(frames["bank"])
    for col in ["flag_zscore", "flag_iforest", "flag_any"]:
        assert flagged[col].dtype == bool
    assert (flagged["flag_any"] >= flagged["flag_zscore"]).all()
