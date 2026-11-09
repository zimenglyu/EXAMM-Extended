#!/usr/bin/env python3
"""
Concatenated core7 folds for the pooling ablation ("Concatenated" construction).

For each test year and panel set, the 50 per-stock files of the walk-forward fold
(datasets/panels_core7_examm_<Y>/<set>/<STOCK>_{train,val}.csv, made by
split_core7_yearly.py) are stacked end to end, in alphabetical stock order, into ONE
series per split:

    datasets/panels_core7_examm_cat_<Y>/<set>/ALL_train.csv
    datasets/panels_core7_examm_cat_<Y>/<set>/ALL_val.csv

So EXAMM sees one long series: stock A's whole history, then stock B's, ... With
--time_offset 1 the last day of one stock is followed by the first day of the next, and
the network's memory carries across stocks -- exactly what this construction tests,
compared with Pooled (50 separate series). Values are copied unchanged.

Test is NOT concatenated: all constructions are scored on the same per-stock test files
of panels_core7_examm_<Y>. Each output folder also gets
    stock_order.csv   which rows of ALL_<split>.csv belong to which stock
    split_info.csv    copied from the source fold (dates of train/val/test)

Usage (from the EXAMM-Extended directory, needs pandas):
    python3 data_prep/concat_core7_yearly.py                  # 2022 2023 2024
    python3 data_prep/concat_core7_yearly.py --years 2024
"""
import argparse
import os
import shutil
import sys

import pandas as pd

SETS = ["set1", "set2", "set3", "set4"]
SPLITS = ["train", "val"]


def fail(msg):
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--root", default="datasets")
    a = ap.parse_args()

    for year in a.years:
        out_root = os.path.join(a.root, f"panels_core7_examm_cat_{year}")
        if os.path.exists(out_root):
            fail(f"{out_root} already exists; refusing to overwrite it")
        for s in SETS:
            src = os.path.join(a.root, f"panels_core7_examm_{year}", s)
            if not os.path.isdir(src):
                fail(f"{src} not found -- run data_prep/split_core7_yearly.py --years {year} first")
            stocks = sorted(f[:-len("_train.csv")] for f in os.listdir(src) if f.endswith("_train.csv"))
            if len(stocks) != 50:
                fail(f"{src}: expected 50 stocks, found {len(stocks)}")

            out = os.path.join(out_root, s)
            os.makedirs(out)
            order = []
            summary = []
            for split in SPLITS:
                parts, row, columns = [], 0, None
                for t in stocks:
                    df = pd.read_csv(os.path.join(src, f"{t}_{split}.csv"))
                    if columns is None:
                        columns = list(df.columns)
                    elif list(df.columns) != columns:
                        fail(f"{t}_{split}.csv columns differ from {stocks[0]}_{split}.csv")
                    if df.isna().any().any():
                        fail(f"{t}_{split}.csv has empty cells")
                    order.append({"split": split, "stock": t, "first_row": row, "last_row": row + len(df) - 1,
                                  "rows": len(df)})
                    row += len(df)
                    parts.append(df)
                cat = pd.concat(parts, ignore_index=True)
                cat.to_csv(os.path.join(out, f"ALL_{split}.csv"), index=False,
                           lineterminator="\n", float_format="%.10g")
                summary.append(f"{split} {len(cat)} rows")
            pd.DataFrame(order).to_csv(os.path.join(out, "stock_order.csv"), index=False, lineterminator="\n")
            shutil.copy(os.path.join(src, "split_info.csv"), os.path.join(out, "split_info.csv"))
            print(f"{year} {s}: {len(stocks)} stocks -> " + ", ".join(summary))


if __name__ == "__main__":
    main()
