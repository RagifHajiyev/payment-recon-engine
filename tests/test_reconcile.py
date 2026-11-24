import pandas as pd

from recon_engine import evaluate
from recon_engine.reconcile import reconcile, summarize


def _bank(rows):
    return pd.DataFrame(rows, columns=["ref", "ts", "partner", "currency", "amount"])


def _side(rows, with_partner=False):
    cols = ["ref", "ts", "partner", "currency", "amount"] if with_partner else \
           ["ref", "ts", "currency", "amount"]
    return pd.DataFrame(rows, columns=cols)


def test_each_issue_type_on_a_tiny_example():
    ts = "2026-01-05 10:00:00"
    bank = _bank([
        ["T1", ts, "A", "AZN", 100.0],   # clean
        ["T2", ts, "A", "AZN", 100.0],   # missing in processor
        ["T3", ts, "A", "AZN", 100.0],   # missing in partner
        ["T4", ts, "A", "AZN", 100.0],   # processor amount differs
        ["T5", ts, "A", "AZN", 100.0],   # partner amount differs
        ["T6", ts, "A", "AZN", 100.0],   # duplicated in processor
    ])
    proc = _side([
        ["T1", ts, "AZN", 100.0], ["T3", ts, "AZN", 100.0],
        ["T4", ts, "AZN", 90.0], ["T5", ts, "AZN", 100.0],
        ["T6", ts, "AZN", 100.0], ["T6", ts, "AZN", 100.0],
        ["X9", ts, "AZN", 55.0],         # not in bank
    ])
    part = _side([
        ["T1", ts, "A", "AZN", 100.0], ["T2", ts, "A", "AZN", 100.0],
        ["T4", ts, "A", "AZN", 100.0], ["T5", ts, "A", "AZN", 95.0],
        ["T6", ts, "A", "AZN", 100.0],
    ], with_partner=True)

    out = reconcile(bank, proc, part).set_index("ref")["issue"].to_dict()
    assert out == {
        "T1": "matched",
        "T2": "missing_in_processor",
        "T3": "missing_in_partner",
        "T4": "amount_mismatch",
        "T5": "partner_amount_mismatch",
        "T6": "duplicate_in_processor",
        "X9": "not_in_bank",
    }


def test_break_value_is_converted_to_azn():
    ts = "2026-01-05 10:00:00"
    bank = _bank([["T1", ts, "A", "USD", 100.0]])
    proc = _side([["T1", ts, "USD", 100.0]])
    part = _side([], with_partner=True)
    rec = reconcile(bank, proc, part)
    assert rec.loc[0, "issue"] == "missing_in_partner"
    assert rec.loc[0, "break_value_azn"] == 170.0  # 100 USD at the illustrative 1.70


def test_tolerance_ignores_tiny_rounding_differences():
    ts = "2026-01-05 10:00:00"
    bank = _bank([["T1", ts, "A", "AZN", 100.00]])
    proc = _side([["T1", ts, "AZN", 100.001]])
    part = _side([["T1", ts, "A", "AZN", 100.00]], with_partner=True)
    assert reconcile(bank, proc, part).loc[0, "issue"] == "matched"


def test_full_dataset_matches_ground_truth_exactly(frames):
    rec = reconcile(frames["bank"], frames["processor"], frames["partner"])
    scores = evaluate.reconciliation_scores(rec, frames["ground_truth"])
    assert (scores["precision"] == 1.0).all()
    assert (scores["recall"] == 1.0).all()
    assert scores["planted"].min() > 0


def test_summary_counts_match_breaks(frames):
    rec = reconcile(frames["bank"], frames["processor"], frames["partner"])
    s = summarize(rec)
    assert s["records"].sum() == (rec["issue"] != "matched").sum()
