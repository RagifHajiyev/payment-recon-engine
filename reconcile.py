"""Three-way reconciliation: bank ledger vs processor file vs partner statement."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .fees import FX_TO_AZN

ISSUES = [
    "not_in_bank",
    "missing_in_processor",
    "missing_in_partner",
    "amount_mismatch",
    "partner_amount_mismatch",
    "duplicate_in_processor",
]


def reconcile(
    bank: pd.DataFrame,
    processor: pd.DataFrame,
    partner: pd.DataFrame,
    tolerance: float = 0.005,
) -> pd.DataFrame:
    """Match the three sources on ``ref`` and label every record.

    Priority when a record has several problems: not_in_bank,
    missing_in_processor, missing_in_partner, amount_mismatch,
    partner_amount_mismatch, duplicate_in_processor, otherwise matched.
    Duplicate processor rows are removed before matching and reported.
    """
    dup_mask = processor.duplicated("ref", keep="first")
    dup_refs = set(processor.loc[dup_mask, "ref"])
    proc = processor.loc[~dup_mask]

    m = (
        bank[["ref", "ts", "partner", "currency", "amount"]]
        .rename(columns={"ts": "ts_bank", "currency": "ccy_bank", "amount": "bank_amt"})
        .merge(
            proc[["ref", "ts", "currency", "amount"]].rename(
                columns={"ts": "ts_proc", "currency": "ccy_proc", "amount": "proc_amt"}),
            on="ref", how="outer",
        )
        .merge(
            partner[["ref", "partner", "currency", "amount"]].rename(
                columns={"partner": "partner_stmt", "currency": "ccy_part", "amount": "part_amt"}),
            on="ref", how="outer",
        )
    )

    in_bank = m["bank_amt"].notna()
    in_proc = m["proc_amt"].notna()
    in_part = m["part_amt"].notna()
    d_proc = (m["bank_amt"] - m["proc_amt"]).abs()
    d_part = (m["bank_amt"] - m["part_amt"]).abs()

    m["issue"] = np.select(
        [
            ~in_bank,
            ~in_proc,
            ~in_part,
            d_proc > tolerance,
            d_part > tolerance,
            m["ref"].isin(dup_refs),
        ],
        ISSUES,
        default="matched",
    )

    m["partner"] = m["partner"].fillna(m["partner_stmt"])
    m["currency"] = m["ccy_bank"].fillna(m["ccy_proc"]).fillna(m["ccy_part"])
    m["ts"] = m["ts_bank"].fillna(m["ts_proc"])

    # Amount at stake per break, in the record's own currency, then AZN.
    stake = np.select(
        [
            m["issue"].eq("not_in_bank"),
            m["issue"].isin(["missing_in_processor", "missing_in_partner", "duplicate_in_processor"]),
            m["issue"].eq("amount_mismatch"),
            m["issue"].eq("partner_amount_mismatch"),
        ],
        [m["proc_amt"], m["bank_amt"], d_proc, d_part],
        default=0.0,
    )
    m["break_value_azn"] = (stake * m["currency"].map(FX_TO_AZN).astype(float)).round(2)

    cols = ["ref", "ts", "partner", "currency", "bank_amt", "proc_amt", "part_amt",
            "issue", "break_value_azn"]
    return m[cols].sort_values("ref", ignore_index=True)


def summarize(rec: pd.DataFrame) -> pd.DataFrame:
    """Count and AZN value of breaks by issue type."""
    brk = rec[rec["issue"] != "matched"]
    return (
        brk.groupby("issue", as_index=False)
        .agg(records=("ref", "count"), value_azn=("break_value_azn", "sum"))
        .sort_values("records", ascending=False, ignore_index=True)
    )
