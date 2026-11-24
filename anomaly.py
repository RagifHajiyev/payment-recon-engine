"""Amount anomaly detection: robust z-score rule plus Isolation Forest."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .fees import to_azn


def _log_amount_and_z(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """log(AZN amount) and its robust z-score within each (partner, currency)."""
    la = np.log1p(to_azn(df["amount"], df["currency"]))
    grp = [df["partner"], df["currency"]]
    med = la.groupby(grp).transform("median")
    mad = (la - med).abs().groupby(grp).transform("median")
    z = (la - med) / (1.4826 * mad.replace(0, np.nan))
    return la, z.fillna(0.0)


def robust_zscore_flags(df: pd.DataFrame, threshold: float = 3.5) -> pd.Series:
    """Flag unusually large amounts using median/MAD on log(AZN amount).

    Computed per (partner, currency) so each segment is judged against its own
    normal range. Only the upper tail is flagged.
    """
    _, z = _log_amount_and_z(df)
    return z > threshold


def isolation_forest_flags(
    df: pd.DataFrame, contamination: float = 0.002, seed: int = 42
) -> pd.Series:
    """Flag anomalies with an Isolation Forest on amount-based features.

    Only features that carry signal are used (log amount and its in-segment
    deviation). Adding uninformative features such as hour of day or partner
    dummies diluted the forest and collapsed precision in testing.
    ``contamination`` is the expected anomaly share and is a tunable assumption.
    """
    la, z = _log_amount_and_z(df)
    feats = pd.DataFrame({"log_amount_azn": la, "segment_z": z})
    model = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=seed, n_jobs=-1
    )
    return pd.Series(model.fit_predict(feats) == -1, index=df.index)


def detect(bank: pd.DataFrame, z_threshold: float = 3.5,
           contamination: float = 0.002, seed: int = 42) -> pd.DataFrame:
    """Return the bank ledger with anomaly flags added."""
    out = bank.copy()
    out["flag_zscore"] = robust_zscore_flags(out, z_threshold)
    out["flag_iforest"] = isolation_forest_flags(out, contamination, seed)
    out["flag_any"] = out["flag_zscore"] | out["flag_iforest"]
    return out
