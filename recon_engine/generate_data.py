"""Seeded synthetic data generator with planted, labelled errors.

Produces three sources keyed by ``ref``:
  * bank ledger        (master record)
  * processor file     (clearing/processor view)
  * partner statement  (partner view, including the fee it charged)

Known problems are injected and recorded in ``ground_truth`` so detection
quality can be measured exactly. All data is synthetic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .fees import FX_TO_AZN, rate_for_turnover, to_azn

PARTNERS = ["PartnerA", "PartnerB", "PartnerC", "PartnerD"]
PARTNER_WEIGHTS = [0.40, 0.30, 0.20, 0.10]
CURRENCIES = list(FX_TO_AZN)
CURRENCY_WEIGHTS = [0.70, 0.20, 0.10]


@dataclass
class GenConfig:
    n_txn: int = 100_000
    seed: int = 42
    start: str = "2026-01-01"
    months: int = 3
    n_fee_errors: int = 2
    # Share of base transactions receiving each issue (mutually exclusive).
    issue_rates: dict = field(
        default_factory=lambda: {
            "missing_in_processor": 0.006,
            "missing_in_partner": 0.008,
            "duplicate_in_processor": 0.004,
            "amount_mismatch": 0.005,
            "partner_amount_mismatch": 0.004,
            "outlier_amount": 0.002,
        }
    )
    orphan_rate: float = 0.003  # processor rows with no bank record


def _delta(rng: np.random.Generator, n: int) -> np.ndarray:
    sign = rng.choice([-1.0, 1.0], size=n)
    return sign * rng.uniform(0.5, 25.0, size=n)


def generate(cfg: GenConfig | None = None) -> dict[str, pd.DataFrame]:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_txn

    start = pd.Timestamp(cfg.start)
    seconds = rng.integers(0, cfg.months * 30 * 86_400, size=n)
    base = pd.DataFrame(
        {
            "ref": [f"TX{i:08d}" for i in range(n)],
            "ts": start + pd.to_timedelta(seconds, unit="s"),
            "partner": rng.choice(PARTNERS, size=n, p=PARTNER_WEIGHTS),
            "currency": rng.choice(CURRENCIES, size=n, p=CURRENCY_WEIGHTS),
            "amount": np.round(rng.lognormal(mean=3.8, sigma=0.9, size=n), 2),
        }
    ).sort_values("ts", ignore_index=True)
    base["ref"] = [f"TX{i:08d}" for i in range(n)]

    # --- assign mutually exclusive issues -------------------------------
    names = list(cfg.issue_rates)
    cum = np.cumsum([cfg.issue_rates[k] for k in names])
    u = rng.random(n)
    idx = np.searchsorted(cum, u, side="right")  # == len(names) -> no issue
    issue = np.array(names + [""], dtype=object)[idx]
    base["issue"] = issue

    # Outliers are consistent across all sources (not reconciliation breaks).
    is_out = base["issue"] == "outlier_amount"
    base.loc[is_out, "amount"] = np.round(
        base.loc[is_out, "amount"] * rng.uniform(50, 200, size=is_out.sum()), 2
    )

    ts_str = base["ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
    bank = pd.DataFrame(
        {"ref": base["ref"], "ts": ts_str, "partner": base["partner"],
         "currency": base["currency"], "amount": base["amount"]}
    )
    proc = bank[["ref", "ts", "currency", "amount"]].copy()
    part = bank[["ref", "ts", "partner", "currency", "amount"]].copy()

    # --- tiered fees charged by partner (computed on true turnover) ------
    month = ts_str.str.slice(0, 7)
    azn = to_azn(base["amount"], base["currency"])
    turnover = azn.groupby([base["partner"], month]).transform("sum")
    rate = turnover.map(rate_for_turnover)
    part["fee_charged"] = (azn * rate).round(4)

    # Inject billing errors for a few partner-months.
    groups = (
        pd.DataFrame({"partner": base["partner"], "month": month})
        .drop_duplicates().reset_index(drop=True)
    )
    pick = rng.choice(len(groups), size=min(cfg.n_fee_errors, len(groups)), replace=False)
    fee_truth = []
    for g in pick:
        p, m = groups.loc[g, "partner"], groups.loc[g, "month"]
        mask = (base["partner"] == p) & (month == m)
        correct = float(rate[mask].iloc[0])
        billed = correct + rng.choice([-0.002, 0.002])
        part.loc[mask.values, "fee_charged"] = (azn[mask] * billed).round(4).values
        fee_truth.append({"partner": p, "month": m, "correct_rate": correct,
                          "billed_rate": round(billed, 4)})
    fee_truth = pd.DataFrame(fee_truth)

    # --- apply per-reference issues -------------------------------------
    def refs(kind: str) -> np.ndarray:
        return base.loc[base["issue"] == kind, "ref"].to_numpy()

    drop_proc = refs("missing_in_processor")
    drop_part = refs("missing_in_partner")
    dup_refs = refs("duplicate_in_processor")
    mm_refs = refs("amount_mismatch")
    pmm_refs = refs("partner_amount_mismatch")

    proc = proc[~proc["ref"].isin(drop_proc)].copy()
    part = part[~part["ref"].isin(drop_part)].copy()

    mm = proc["ref"].isin(mm_refs)
    proc.loc[mm, "amount"] = np.round(proc.loc[mm, "amount"] + _delta(rng, mm.sum()), 2)
    pmm = part["ref"].isin(pmm_refs)
    part.loc[pmm, "amount"] = np.round(part.loc[pmm, "amount"] + _delta(rng, pmm.sum()), 2)

    dups = proc[proc["ref"].isin(dup_refs)]
    proc = pd.concat([proc, dups], ignore_index=True)

    n_orph = int(n * cfg.orphan_rate)
    orph = pd.DataFrame(
        {
            "ref": [f"OR{i:08d}" for i in range(n_orph)],
            "ts": (start + pd.to_timedelta(
                rng.integers(0, cfg.months * 30 * 86_400, size=n_orph), unit="s"
            )).strftime("%Y-%m-%d %H:%M:%S"),
            "currency": rng.choice(CURRENCIES, size=n_orph, p=CURRENCY_WEIGHTS),
            "amount": np.round(rng.lognormal(3.8, 0.9, size=n_orph), 2),
        }
    )
    proc = pd.concat([proc, orph], ignore_index=True)

    truth = pd.concat(
        [
            base.loc[base["issue"] != "", ["ref", "issue"]].rename(columns={"issue": "issue_type"}),
            pd.DataFrame({"ref": orph["ref"], "issue_type": "not_in_bank"}),
        ],
        ignore_index=True,
    )
    return {
        "bank": bank,
        "processor": proc.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True),
        "partner": part.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True),
        "ground_truth": truth,
        "fee_truth": fee_truth,
    }


def write(frames: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = {"bank": "bank_ledger", "processor": "processor_file",
             "partner": "partner_statement", "ground_truth": "ground_truth",
             "fee_truth": "fee_truth"}
    for key, fname in names.items():
        frames[key].to_csv(out / f"{fname}.csv", index=False)
