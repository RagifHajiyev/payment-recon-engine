"""End-to-end pipeline: generate -> reconcile -> validate fees -> detect -> report.

Usage:
    python -m recon_engine.pipeline --n 100000 --out output
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import anomaly, db, evaluate, fees, generate_data, reconcile  # noqa: E402


def run(n_txn: int = 100_000, seed: int = 42, out_dir: str | Path = "output",
        data_dir: str | Path = "data", chart: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    frames = generate_data.generate(generate_data.GenConfig(n_txn=n_txn, seed=seed))
    generate_data.write(frames, data_dir)

    rec = reconcile.reconcile(frames["bank"], frames["processor"], frames["partner"])
    summary = reconcile.summarize(rec)
    fee_check = fees.validate_fees(frames["partner"])
    flagged = anomaly.detect(frames["bank"])

    rec_scores = evaluate.reconciliation_scores(rec, frames["ground_truth"])
    fee_scores = evaluate.fee_scores(fee_check, frames["fee_truth"])
    anom_scores = evaluate.anomaly_scores(flagged, frames["ground_truth"])

    rec.to_csv(out / "reconciliation_results.csv", index=False)
    summary.to_csv(out / "break_summary.csv", index=False)
    fee_check.to_csv(out / "fee_validation.csv", index=False)
    flagged[flagged["flag_any"]].to_csv(out / "anomalies.csv", index=False)
    db.load(out / "recon.db", rec, fee_check, flagged)

    metrics = {
        "transactions": n_txn,
        "seed": seed,
        "matched": int((rec["issue"] == "matched").sum()),
        "breaks": int((rec["issue"] != "matched").sum()),
        "break_value_azn": float(rec["break_value_azn"].sum().round(2)),
        "reconciliation": rec_scores.to_dict("records"),
        "fees": fee_scores,
        "anomalies": anom_scores.to_dict("records"),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if chart:
        _chart(summary, fee_check, out / "summary.png")
    return metrics


def _chart(summary: pd.DataFrame, fee_check: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    s = summary.sort_values("records")
    axes[0].barh(s["issue"], s["records"], color="#3b6ea5")
    axes[0].set_title("Reconciliation breaks by type (records)")
    s2 = summary.sort_values("value_azn")
    axes[1].barh(s2["issue"], s2["value_azn"], color="#c0504d")
    axes[1].set_title("Break value by type (AZN)")
    fc = fee_check.copy()
    labels = fc["partner"] + " " + fc["month"].str.slice(5)
    colors = ["#c0504d" if f else "#8aa" for f in fc["flagged"]]
    axes[2].bar(labels, fc["fee_diff_pct"] * 100, color=colors)
    axes[2].set_title("Fee variance vs contract (%), flagged in red")
    axes[2].tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100_000, help="number of base transactions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="output")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    m = run(args.n, args.seed, args.out, args.data)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
