#!/usr/bin/env python3
"""
Split one core7 panel set into <STOCK>_train / _val / _test.csv for EXAMM, on
the calendar the ONE-NAS runs used, so EXAMM's test span is ONE-NAS's scored span:

    train  rows 0 .. NTW*5-1           the ONE-NAS warm-up (through ~2018-10)
    val    rows NTW*5 .. last 2019 row the unscored online span
    test   2020-01-01 .. end, preceded by WARM earlier rows so the RNN state is
           warm; predictions dated before 2020 are dropped when scoring

The files keep the 7 core7 inputs and the RET_CS target, already scaled, so EXAMM
runs with --normalize none. split_info.csv records each split's panel rows and
dates, to put the dates back on EXAMM's predictions.

Usage (from the EXAMM-Extended directory):
    python3 scripts/data_prep/split_core7_for_examm.py SET
    python3 scripts/data_prep/split_core7_for_examm.py SET --input_root /anvil/projects/x-cis251123/shared/panels_core7
"""
import argparse
import csv
import os
import sys

NTW = {1: 712, 2: 695, 3: 702, 4: 738}   # --num_training_windows of the ONE-NAS eval clock
STEP = 5
COLUMNS = ["RET", "RET_CS_IN", "BA_SPREAD", "ILLIQUIDITY", "REV21_1", "TURN_RATIO", "VOL21", "RET_CS"]
TEST_FROM = "2020-01-01"
WARM = 40


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("set", type=int, choices=sorted(NTW))
    parser.add_argument("--input_root", default=os.path.expanduser("~/ONENAS/datasets/panels_core7"),
                        help="folder holding setN_core7/ (default: ~/ONENAS/datasets/panels_core7)")
    parser.add_argument("--output_root", default="datasets/panels_core7_examm",
                        help="written to <output_root>/setN (default: datasets/panels_core7_examm)")
    args = parser.parse_args()

    src = os.path.join(args.input_root, f"set{args.set}_core7")
    dst = os.path.join(args.output_root, f"set{args.set}")
    if os.path.abspath(src) == os.path.abspath(dst):
        sys.exit("--output_root must differ from --input_root")

    with open(os.path.join(src, "panel_dates.csv"), newline="") as f:
        dates = [row["date"] for row in csv.DictReader(f)]
    cut = NTW[args.set] * STEP
    test_start = next(i for i, d in enumerate(dates) if d >= TEST_FROM)
    if not WARM <= cut < test_start - WARM:
        sys.exit(f"unexpected boundaries: cut {cut}, test start {test_start}")
    splits = {"train": (0, cut), "val": (cut, test_start), "test": (test_start - WARM, len(dates))}

    stocks = sorted(n[:-4] for n in os.listdir(src) if n.endswith(".csv") and not n.startswith("panel_"))
    if not stocks:
        sys.exit(f"no stock csv files in {src}")
    os.makedirs(dst, exist_ok=True)
    for stock in stocks:
        with open(os.path.join(src, f"{stock}.csv"), newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in COLUMNS if c not in reader.fieldnames]
            if missing:
                sys.exit(f"{stock}.csv has no column(s) {missing}")
            rows = list(reader)
        if len(rows) != len(dates):
            sys.exit(f"{stock}.csv has {len(rows)} rows, panel_dates.csv has {len(dates)}")
        for split, (a, b) in splits.items():
            with open(os.path.join(dst, f"{stock}_{split}.csv"), "w", newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(COLUMNS)
                writer.writerows([row[c] for c in COLUMNS] for row in rows[a:b])

    with open(os.path.join(dst, "split_info.csv"), "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["split", "first_row", "last_row", "first_date", "last_date", "rows"])
        for split, (a, b) in splits.items():
            writer.writerow([split, a, b - 1, dates[a], dates[b - 1], b - a])

    print(f"set{args.set}: {len(stocks)} stocks x 3 files -> {dst}")
    for split, (a, b) in splits.items():
        print(f"  {split:5s} rows {a:4d}-{b - 1:4d}  {dates[a]} .. {dates[b - 1]}  ({b - a} rows)")
    print(f"  test includes {WARM} warm-up rows before {TEST_FROM}; scoring keeps predictions from {dates[test_start]} on")


if __name__ == "__main__":
    main()
