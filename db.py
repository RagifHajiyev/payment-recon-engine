"""Load results into SQLite and create the reporting views."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

SQL_DIR = Path(__file__).resolve().parents[2] / "sql"


def load(db_path: str | Path, reconciliation: pd.DataFrame,
         fee_check: pd.DataFrame, anomalies: pd.DataFrame,
         views_sql: str | Path | None = None) -> None:
    """(Re)create the SQLite database with result tables and views."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    views_sql = Path(views_sql) if views_sql else SQL_DIR / "views.sql"
    with sqlite3.connect(db_path) as con:
        reconciliation.to_sql("reconciliation", con, index=False)
        fee_check.assign(flagged=fee_check["flagged"].astype(int)).to_sql(
            "fee_check", con, index=False)
        anomalies.assign(
            **{c: anomalies[c].astype(int) for c in anomalies.columns if c.startswith("flag_")}
        ).to_sql("anomalies", con, index=False)
        con.executescript(views_sql.read_text(encoding="utf-8"))


def query(db_path: str | Path, sql: str) -> pd.DataFrame:
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)
