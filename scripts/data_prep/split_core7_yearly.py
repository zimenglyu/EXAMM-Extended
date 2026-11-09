#!/usr/bin/env python3
"""
Yearly train/val/test splits of the core7 panels for pooled EXAMM ("Test <YEAR>").

For test year Y:
    train = every row dated <= (Y-2)-12-31
    val   = rows dated in Y-1
    test  = rows dated in Y, plus the WARMUP rows just before Y-01-01 (they overlap
            the end of val) so the network has 40 days of history before the first
            scored day; drop those rows when scoring
Values are copied unchanged from the source panels; one CSV per stock per split,
plus split_info.csv (row ranges and dates) per set.

Usage (from the EXAMM-Extended directory):
    python3 data_prep/split_core7_yearly.py --years 2022 2023
Writes datasets/panels_core7_examm_<Y>/set{1..4}/<STOCK>_{train,val,test}.csv
"""
import argparse
import os
import sys

import pandas as pd

SRC = "/anvil/projects/x-cis251123/shared/panels_core7"
SETS = ["set1", "set2", "set3", "set4"]
WARMUP = 40   # = ONENAS --time_series_length, same warm-up as panels_core7_examm


def fail(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023])
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--out_root", default="datasets")
    a = ap.parse_args()

    for year in a.years:
        out_root = os.path.join(a.out_root, f"panels_core7_examm_{year}")
        if os.path.exists(out_root):
            fail(f"{out_root} already exists; refusing to overwrite it")
        for s in SETS:
            src = os.path.join(a.src, f"{s}_core7")
            dates = pd.read_csv(os.path.join(src, "panel_dates.csv"), usecols=["row", "date"])
            if not (dates["row"].to_numpy() == range(len(dates))).all():
                fail(f"{src}/panel_dates.csv rows are not 0..{len(dates) - 1}")
            d = dates["date"]
            train_last = int(dates.index[d <= f"{year - 2}-12-31"].max())
            val = dates.index[(d >= f"{year - 1}-01-01") & (d <= f"{year - 1}-12-31")]
            test = dates.index[(d >= f"{year}-01-01") & (d <= f"{year}-12-31")]
            if len(val) == 0 or len(test) == 0:
                fail(f"{s}: no rows for val {year - 1} or test {year}")
            ranges = {
                "train": (0, train_last),
                "val": (int(val.min()), int(val.max())),
                "test": (int(test.min()) - WARMUP, int(test.max())),
            }
            if ranges["val"][0] != train_last + 1:
                fail(f"{s}: gap between train and val")

            out = os.path.join(out_root, s)
            os.makedirs(out)
            stocks = sorted(f[:-4] for f in os.listdir(src) if f.endswith(".csv") and not f.startswith("panel_"))
            for t in stocks:
                full = pd.read_csv(os.path.join(src, t + ".csv"))
                if len(full) != len(dates):
                    fail(f"{t} in {s} has {len(full)} rows, panel_dates has {len(dates)}")
                if full.isna().any().any():
                    fail(f"{t} in {s} has empty cells")
                for split, (lo, hi) in ranges.items():
                    full.iloc[lo:hi + 1].to_csv(os.path.join(out, f"{t}_{split}.csv"),
                                                index=False, lineterminator="\n", float_format="%.10g")
            info = pd.DataFrame([{"split": k, "first_row": lo, "last_row": hi,
                                  "first_date": d[lo], "last_date": d[hi], "rows": hi - lo + 1}
                                 for k, (lo, hi) in ranges.items()])
            info.to_csv(os.path.join(out, "split_info.csv"), index=False, lineterminator="\n")
            scored = len(test)
            print(f"{year} {s}: {len(stocks)} stocks | train {d[0]}..{d[train_last]} ({train_last + 1}) | "
                  f"val {d[ranges['val'][0]]}..{d[ranges['val'][1]]} ({len(val)}) | "
                  f"test {d[ranges['test'][0]]}..{d[ranges['test'][1]]} ({scored + WARMUP}, {scored} scored)")


if __name__ == "__main__":
    main()
