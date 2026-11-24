import pandas as pd
import pytest

from recon_engine import evaluate
from recon_engine.fees import FEE_TIERS, rate_for_turnover, validate_fees


@pytest.mark.parametrize(
    "turnover, expected",
    [
        (0, 0.0120),
        (399_999.99, 0.0120),
        (400_000, 0.0100),
        (899_999.99, 0.0100),
        (900_000, 0.0080),
        (5_000_000, 0.0080),
    ],
)
def test_tier_boundaries(turnover, expected):
    assert rate_for_turnover(turnover) == expected


def test_tiers_are_sorted_ascending():
    thresholds = [t for t, _ in FEE_TIERS]
    assert thresholds == sorted(thresholds)


def test_correct_fee_is_not_flagged_and_wrong_fee_is_flagged():
    rows = []
    for i in range(10):
        rows.append(["2026-01-10 10:00:00", "A", "AZN", 1000.0, 1000.0 * 0.012])  # correct
        rows.append(["2026-01-10 10:00:00", "B", "AZN", 1000.0, 1000.0 * 0.020])  # wrong
    df = pd.DataFrame(rows, columns=["ts", "partner", "currency", "amount", "fee_charged"])
    out = validate_fees(df).set_index("partner")
    assert not out.loc["A", "flagged"]
    assert out.loc["B", "flagged"]
    assert out.loc["B", "fee_diff"] > 0


def test_planted_billing_errors_are_found(frames):
    check = validate_fees(frames["partner"])
    scores = evaluate.fee_scores(check, frames["fee_truth"])
    assert scores["recall"] == 1.0
    assert scores["precision"] == 1.0
