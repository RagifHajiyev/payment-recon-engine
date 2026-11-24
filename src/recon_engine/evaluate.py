"""Score detection results against the planted ground truth."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from .reconcile import ISSUES


def reconciliation_scores(rec: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Per-issue precision / recall of the reconciliation labels.

    ``outlier_amount`` rows are consistent in all sources, so the correct
    reconciliation label for them is ``matched``.
    """
    t = truth.set_index("ref")["issue_type"].replace({"outlier_amount": "matched"})
    expected = rec["ref"].map(t).fillna("matched")
    labels = ISSUES
    p, r, f, s = precision_recall_fscore_support(
        expected, rec["issue"], labels=labels, zero_division=0
    )
    return pd.DataFrame(
        {"issue": labels, "planted": s, "precision": p.round(4),
         "recall": r.round(4), "f1": f.round(4)}
    )


def fee_scores(fee_check: pd.DataFrame, fee_truth: pd.DataFrame) -> dict:
    """Precision / recall of flagged partner-months vs planted billing errors."""
    flagged = set(map(tuple, fee_check.loc[fee_check["flagged"], ["partner", "month"]].values))
    truth = set(map(tuple, fee_truth[["partner", "month"]].values))
    tp = len(flagged & truth)
    return {
        "planted": len(truth),
        "flagged": len(flagged),
        "precision": round(tp / len(flagged), 4) if flagged else 0.0,
        "recall": round(tp / len(truth), 4) if truth else 1.0,
    }


def anomaly_scores(flagged: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Precision / recall of each anomaly detector for planted amount outliers."""
    actual = flagged["ref"].isin(truth.loc[truth["issue_type"] == "outlier_amount", "ref"])
    rows = []
    for col in ["flag_zscore", "flag_iforest", "flag_any"]:
        pred = flagged[col]
        tp = int((pred & actual).sum())
        rows.append(
            {
                "detector": col.replace("flag_", ""),
                "planted": int(actual.sum()),
                "flagged": int(pred.sum()),
                "precision": round(tp / pred.sum(), 4) if pred.sum() else 0.0,
                "recall": round(tp / actual.sum(), 4) if actual.sum() else 1.0,
            }
        )
    return pd.DataFrame(rows)
