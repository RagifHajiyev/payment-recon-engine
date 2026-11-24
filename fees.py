"""Tiered fee model and independent fee validation.

All rates and FX rates are fictional and for illustration only.
"""

from __future__ import annotations

import pandas as pd

# Illustrative FX rates to AZN.
FX_TO_AZN = {"AZN": 1.0, "USD": 1.70, "EUR": 1.85}

# (minimum monthly partner turnover in AZN, fee rate). Highest matching tier wins.
FEE_TIERS = [(0.0, 0.0120), (400_000.0, 0.0100), (900_000.0, 0.0080)]


def rate_for_turnover(turnover_azn: float) -> float:
    """Return the contractual fee rate for a monthly turnover (AZN)."""
    rate = FEE_TIERS[0][1]
    for threshold, tier_rate in FEE_TIERS:
        if turnover_azn >= threshold:
            rate = tier_rate
    return rate


def to_azn(amount: pd.Series, currency: pd.Series) -> pd.Series:
    """Convert amounts to AZN using the illustrative FX table."""
    return amount * currency.map(FX_TO_AZN).astype(float)


def validate_fees(statement: pd.DataFrame, tolerance: float = 0.01) -> pd.DataFrame:
    """Independently recompute partner fees and flag billing differences.

    ``statement`` needs columns: ts, partner, currency, amount, fee_charged.
    Fees are expected to be turnover-tiered per partner and calendar month.
    A partner-month is flagged when the charged fee differs from the expected
    fee by more than ``tolerance`` (relative).
    """
    d = statement.copy()
    d["month"] = d["ts"].astype(str).str.slice(0, 7)
    d["amount_azn"] = to_azn(d["amount"], d["currency"])
    out = (
        d.groupby(["partner", "month"], as_index=False)
        .agg(turnover_azn=("amount_azn", "sum"), fee_charged=("fee_charged", "sum"))
    )
    out["expected_rate"] = out["turnover_azn"].map(rate_for_turnover)
    out["expected_fee"] = out["turnover_azn"] * out["expected_rate"]
    out["implied_rate"] = out["fee_charged"] / out["turnover_azn"]
    out["fee_diff"] = out["fee_charged"] - out["expected_fee"]
    out["fee_diff_pct"] = out["fee_diff"] / out["expected_fee"]
    out["flagged"] = out["fee_diff_pct"].abs() > tolerance
    return out.round({"turnover_azn": 2, "fee_charged": 2, "expected_fee": 2,
                      "fee_diff": 2, "fee_diff_pct": 4, "implied_rate": 5})
