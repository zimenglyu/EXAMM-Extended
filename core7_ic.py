#!/usr/bin/env python3
"""Rank IC and MSE for core7 genomes, on the test split (default) or the validation split.

Each trading day, the rank correlation between predicted_RET_CS and the realized
next-day RET_CS across the panel's 50 stocks, averaged over the scored days;
MSE is (predicted_RET_CS - expected_RET_CS)^2 over the same days, in RET_CS units.
On test, the 40-day warm-up at the start of each file is dropped; validation has no
warm-up, so every validation day is scored.

--target defaults to RET:
  no --year             -> the RET runs of all three walk-forward folds (2022, 2023, 2024)
  --year 2023           -> the RET runs of the 2023 fold (test_output/core7_examm_ret_2023)
--target RET_CS:
  no --year             -> the 2020-2024 run (test_output/core7_examm): test 2020-2024, val 2018-10..2019
  --year 2023           -> the RET_CS runs of the 2023 fold (test_output/core7_examm_2023)

Use --split val to CHOOSE a genome (e.g. one set per year), --split test to REPORT it.
Run evaluate_rnn on that split first:
    SPLIT=val|test [YEAR=2023] sh scripts/rnn_examples/evaluate_rnn_core7.sh
Usage (keep this file in the EXAMM-Extended folder):
    python3 core7_ic.py                                  # test, RET runs of 2022, 2023 and 2024
    python3 core7_ic.py --year 2023                      # test, RET runs of the 2023 fold
    python3 core7_ic.py --target RET_CS                  # test, the RET_CS 2020-2024 run
    python3 core7_ic.py --year 2023 --target RET_CS      # test, RET_CS runs of the 2023 fold
    python3 core7_ic.py --year 2023 --split val          # validation, same 4 candidates
    python3 core7_ic.py --year 2023 set3/22              # any runs

--target RET: the runs predict next-day RET (robust z of the raw return). Predictions are
converted back to raw returns with the set's median/IQR from panel_meta.json
(raw = RET * iqr + median) and scored against the unclipped RET_raw_<STOCK> in
panel_dates.csv: Rank IC on raw returns, MSE in raw daily-return units (the paper's scale),
and naive_MSE = MSE of always predicting a 0 return, for reference.
"""
import argparse, glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent                     # EXAMM-Extended
PANELS = Path(os.getenv("CORE7_PANELS", ROOT / "datasets" / "panels_core7"))   # <set>_core7/panel_dates.csv

# best repeat per set by validation MSE
DEFAULT_RUNS = {
    None: ["set1/15", "set2/17", "set3/24", "set4/24"],
    2022: ["set1/20", "set2/21", "set3/20", "set4/19"],
    2023: ["set1/22", "set2/20", "set3/21", "set4/19"],
    2024: ["set1/19", "set2/19", "set3/23", "set4/23"],
}
DEFAULT_RUNS_RET = {
    2022: ["set1/21", "set2/16", "set3/22", "set4/24"],
    2023: ["set1/17", "set2/21", "set3/17", "set4/21"],
    2024: ["set1/15", "set2/15", "set3/22", "set4/24"],
}

ap = argparse.ArgumentParser()
ap.add_argument("--year", type=int, choices=[2022, 2023, 2024])
ap.add_argument("--split", choices=["test", "val"], default="test")
ap.add_argument("--target", choices=["RET_CS", "RET"], default="RET")
ap.add_argument("runs", nargs="*")
a = ap.parse_args()

ret = a.target == "RET"
split = a.split
# RET without --year: every walk-forward fold (there is no RET 2020-2024 run)
years = [a.year] if a.year is not None else ([2022, 2023, 2024] if ret else [None])

rows = []
for year, run in [(y, r) for y in years for r in (a.runs or (DEFAULT_RUNS_RET if ret else DEFAULT_RUNS)[y])]:
    results = (f"core7_examm_ret_{year}" if ret else
               "core7_examm" if year is None else f"core7_examm_{year}")
    data = "panels_core7_examm" if year is None else f"panels_core7_examm_{year}"
    # first scored day: test drops its 40-day warm-up; validation scores every day
    start = ("2020-01-01" if year is None else f"{year}-01-01") if split == "test" else "0000"
    s = run.split("/")[0]
    files = sorted(glob.glob(f"{ROOT}/test_output/{results}/{run}/evaluation_{split}/*_{split}_predictions.csv"))
    if len(files) != 50:
        print(f"{run}: found {len(files)} {split} prediction files, expected 50 -- run "
              f"{'' if year is None else f'YEAR={year} '}TARGET={a.target} SPLIT={split} "
              f"sh scripts/rnn_examples/evaluate_rnn_core7.sh")
        continue
    P, A = {}, {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        t = os.path.basename(f)[:-len(f"_{split}_predictions.csv")]
        P[t], A[t] = df[f"predicted_{a.target}"].to_numpy(), df[f"expected_{a.target}"].to_numpy()
    P, A = pd.DataFrame(P), pd.DataFrame(A)            # rows = days, columns = stocks

    # prediction row j forecasts panel row first_row + j + 1
    first = int(pd.read_csv(ROOT / "datasets" / data / s / "split_info.csv")
                .set_index("split").loc[split, "first_row"])
    panel = pd.read_csv(PANELS / f"{s}_core7" / "panel_dates.csv").set_index("row") \
        .loc[first + np.arange(len(P)) + 1]
    dates = panel["date"]
    keep = (dates >= start).to_numpy()                # test: drop the 40-day warm-up

    if ret:
        # back to raw returns, scored against the unclipped raw returns
        meta = PANELS / f"{s}_core7" / "panel_meta.json"
        if not meta.exists():
            raise SystemExit(f"{meta} not found -- needed to convert RET predictions to raw returns")
        st = json.load(open(meta))["stats"]["RET"]
        P = P * st["iqr"] + st["median"]
        A = pd.DataFrame({t: panel[f"RET_raw_{t}"].to_numpy() for t in P.columns})

    ic = P[keep].rank(axis=1).corrwith(A[keep].rank(axis=1), axis=1).dropna()
    rows.append({"year": year or "2020-2024", "split": split, "run": run, "days": len(ic),
                 "first_day": dates[keep].iloc[0], "last_day": dates[keep].iloc[-1],
                 "IC": ic.mean(), "IC_t": ic.mean() / ic.std() * np.sqrt(len(ic)),
                 "IC_positive_days": (ic > 0).mean(),
                 "MSE": float(((P[keep] - A[keep]) ** 2).to_numpy().mean())})
    if ret:
        rows[-1]["naive_MSE"] = float((A[keep] ** 2).to_numpy().mean())

out = pd.DataFrame(rows)
if ret and not out.empty:   # raw-return MSEs are ~1e-4: print in scientific notation
    for c in ("MSE", "naive_MSE"):
        out[c] = out[c].map(lambda v: f"{v:.4e}")
print(out.round(4).to_string(index=False))
