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

Pure stdlib (no pandas -- cluster login nodes don't have it), and data rows
are passed through byte-faithfully rather than re-parsed.

Usage (pilot):
    python3 scripts/stock_run/build_walkforward_splits.py \
        --cutoff 2021-12-31 --val-year 2022 --test-year 2023 \
        --out datasets/walkforward/pilot_2023 \
        --tickers AKAM ATO CAG COO DECK HBAN JBHT NTRS PKG TRV
"""
import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "datasets/701515_split"


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def read_rows(path):
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            fail(f"{path}: empty file")
        return header, [r for r in reader if r]


def write_rows(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


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
        header = None
        rows = []
        for s in ("train", "val", "test"):
            f = SOURCE / f"{t}_{s}.csv"
            if not f.is_file():
                fail(f"{t}: missing source file {f}")
            h, r = read_rows(f)
            if header is None:
                header = h
            elif h != header:
                fail(f"{t}: column mismatch between splits ({s})")
            rows.extend(r)
        if header[0] != "date":
            fail(f"{t}: first column is '{header[0]}', expected 'date'")

        d = [r[0] for r in rows]
        bad = [i for i in range(len(d) - 1) if not (d[i] < d[i + 1])]
        if bad:
            i = bad[0]
            fail(f"{t}: calendar not strictly increasing at row {i} "
                 f"({d[i]} -> {d[i+1]}) -- duplicated/out-of-order dates; "
                 f"pick a different ticker")

        train = [r for r in rows if r[0] <= args.cutoff]
        val = [r for r in rows if r[0].startswith(f"{args.val_year}-")]
        test = [r for r in rows if r[0].startswith(f"{args.test_year}-")]

        if len(train) < args.min_train_rows:
            fail(f"{t}: only {len(train)} training rows (min {args.min_train_rows})")
        for name, part in (("val", val), ("test", test)):
            if not (240 <= len(part) <= 260):
                fail(f"{t}: {name} has {len(part)} rows -- expected a full "
                     f"trading year (~250); wrong year or missing data?")
        if len(train) + len(val) + len(test) > len(rows):
            fail(f"{t}: splits overlap -- internal error")

        write_rows(out / f"{t}_train.csv", header, train)
        write_rows(out / f"{t}_val.csv", header, val)
        write_rows(out / f"{t}_test.csv", header, test)
        print(f"{t:7s} {len(train):6d} {len(val):5d} {len(test):5d}   "
              f"{train[0][0]} .. {train[-1][0]}")

    print(f"\nwrote {len(args.tickers)} stocks x 3 splits -> {out}")
    print("unused rows between cutoff and val-year (if any) are dropped by design")


if __name__ == "__main__":
    main()
