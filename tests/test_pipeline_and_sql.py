from recon_engine import db, pipeline


def test_pipeline_and_sql_views(tmp_path):
    metrics = pipeline.run(
        n_txn=5_000, seed=3, out_dir=tmp_path / "out", data_dir=tmp_path / "data", chart=False
    )
    assert metrics["breaks"] > 0
    assert (tmp_path / "out" / "recon.db").exists()
    assert (tmp_path / "data" / "bank_ledger.csv").exists()

    dbp = tmp_path / "out" / "recon.db"
    unmatched = db.query(dbp, "SELECT * FROM v_unmatched_value")
    assert unmatched["records"].sum() == metrics["breaks"]
    assert abs(unmatched["pct_of_break_value"].sum() - 100.0) < 0.1

    daily = db.query(dbp, "SELECT * FROM v_daily_breaks")
    assert daily["records"].sum() == metrics["breaks"]

    fees = db.query(dbp, "SELECT * FROM v_fee_flags")
    assert len(fees) == metrics["fees"]["flagged"]

    variance = db.query(dbp, "SELECT * FROM v_variance_by_partner_month")
    assert len(variance) > 0
