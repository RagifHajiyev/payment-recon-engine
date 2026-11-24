import pandas as pd

from recon_engine import generate_data as g


def test_generation_is_reproducible():
    a = g.generate(g.GenConfig(n_txn=2_000, seed=1))
    b = g.generate(g.GenConfig(n_txn=2_000, seed=1))
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key])


def test_different_seed_changes_data():
    a = g.generate(g.GenConfig(n_txn=2_000, seed=1))
    b = g.generate(g.GenConfig(n_txn=2_000, seed=2))
    assert not a["bank"]["amount"].equals(b["bank"]["amount"])


def test_all_planted_issue_types_exist(frames):
    kinds = set(frames["ground_truth"]["issue_type"])
    assert kinds == {
        "missing_in_processor", "missing_in_partner", "duplicate_in_processor",
        "amount_mismatch", "partner_amount_mismatch", "outlier_amount", "not_in_bank",
    }


def test_bank_refs_are_unique_and_sources_have_expected_shape(frames):
    assert frames["bank"]["ref"].is_unique
    assert len(frames["bank"]) == 20_000
    assert frames["processor"]["ref"].duplicated().any()  # planted duplicates
    assert len(frames["fee_truth"]) == 2
