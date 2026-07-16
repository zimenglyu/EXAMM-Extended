#!/usr/bin/env python3
"""Build calendar-based walk-forward splits from the 70/15/15 dataset.

Reconstructs each stock's full chronological series by concatenating its
existing train/val/test files, verifies the calendar is strictly increasing
(no duplicated or out-of-order dates -- 9 of the 50 stocks fail this and must
not be used), then writes calendar splits:

    train = everything <= --cutoff
    val   = --val-year
    test  = --test-year

This mirrors the professor's paper (arXiv 2410.17212) split style: train to
year N-1, validate on year N, trade year N+1.

Usage (pilot):
    python3 scripts/stock_run/build_walkforward_splits.py \
        --cutoff 2021-12-31 --val-year 2022 --test-year 2023 \
        --out datasets/walkforward/pilot_2023 \
        --tickers AKAM ATO CAG COO DECK HBAN JBHT NTRS PKG TRV
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets/701515_split"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", required=True, metavar="YYYY-MM-DD",
                    help="last training date (inclusive)")
    ap.add_argument("--val-year", required=True)
    ap.add_argument("--test-year", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--tickers", nargs="+", required=True)
    ap.add_argument("--min-train-rows", type=int, default=1000)
    args = ap.parse_args()

    if not (args.cutoff < f"{args.val_year}-01-01" < f"{args.test_year}-01-01"):
        fail("expected cutoff < val-year < test-year (chronological, no overlap)")

    out = args.out if args.out.is_absolute() else REPO / args.out
    out.mkdir(parents=True, exist_ok=True)

    print(f"{'ticker':7s} {'train':>6s} {'val':>5s} {'test':>5s}   train range")
    for t in args.tickers:
        parts = []
        cols = None
        for s in ("train", "val", "test"):
            f = SOURCE / f"{t}_{s}.csv"
            if not f.is_file():
                fail(f"{t}: missing source file {f}")
            df = pd.read_csv(f)
            if cols is None:
                cols = list(df.columns)
            elif list(df.columns) != cols:
                fail(f"{t}: column mismatch between splits ({s})")
            parts.append(df)
        full = pd.concat(parts, ignore_index=True)

        d = full["date"].astype(str).to_numpy()
        bad = [i for i in range(len(d) - 1) if not (d[i] < d[i + 1])]
        if bad:
            i = bad[0]
            fail(f"{t}: calendar not strictly increasing at row {i} "
                 f"({d[i]} -> {d[i+1]}) -- duplicated/out-of-order dates; "
                 f"pick a different ticker")

        train = full[full["date"] <= args.cutoff]
        val = full[full["date"].str.startswith(f"{args.val_year}-")]
        test = full[full["date"].str.startswith(f"{args.test_year}-")]

        if len(train) < args.min_train_rows:
            fail(f"{t}: only {len(train)} training rows (min {args.min_train_rows})")
        for name, part, lo, hi in (("val", val, 240, 260), ("test", test, 240, 260)):
            if not (lo <= len(part) <= hi):
                fail(f"{t}: {name} has {len(part)} rows -- expected a full "
                     f"trading year (~250); wrong year or missing data?")
        if len(train) + len(val) + len(test) > len(full):
            fail(f"{t}: splits overlap -- internal error")

        train.to_csv(out / f"{t}_train.csv", index=False)
        val.to_csv(out / f"{t}_val.csv", index=False)
        test.to_csv(out / f"{t}_test.csv", index=False)
        print(f"{t:7s} {len(train):6d} {len(val):5d} {len(test):5d}   "
              f"{train['date'].iloc[0]} .. {train['date'].iloc[-1]}")

    print(f"\nwrote {len(args.tickers)} stocks x 3 splits -> {out}")
    print(f"unused rows between cutoff and val-year (if any) are dropped by design")


if __name__ == "__main__":
    main()
