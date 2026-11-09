#!/usr/bin/env python3
"""Paper-style table for pooled EXAMM on the core7 panel sets.

Per test year: Rank IC, MSE, Net %, Sharpe and MDD.
Rank IC and MSE come from ~/EXAMM-Extended/core7_ic.py; Net %, Sharpe and MDD of
the daily long-short come from main_core7.py (Sharpe/MDD need the daily_values
additions in portfolio.py), for each panel set's global best genome in each year.
The naive MSE printed above the table is the MSE of always predicting 0; RET_CS is
a rank-normal score, so it is mean(Phi^-1((i - 0.5) / N)^2) over the N = 50 stocks (~0.9749).
MSE(raw,cal) (RET_CS table only) is a real raw-return forecast error: each RET_CS prediction is
turned into a return forecast a + b * prediction, with a and b fitted by least squares on the
validation year (validation predictions vs. next-day raw returns, pooled over the stocks) and
applied unchanged to the test year. Needs the validation predictions of the same genomes
(SPLIT=val YEAR=<year> sh evaluate_rnn_core7.sh).
Keep this file next to main_core7.py.

Default (--target RET): the RET-target runs (core7_examm_ret_<year>). Rank IC is on raw returns and
MSE is in raw daily-return units (predictions converted with each set's median/IQR from
panel_meta.json, scored against the unclipped raw returns); the line above the table then
gives the naive MSE of always predicting a 0 return, averaged over the 4 sets.

--target RET_CS: the RET_CS-target runs (core7_examm_<year>), with MSE in RET_CS units and
MSE(raw,cal) as described above.

Usage:  python3 core7_table.py                  # RET target (default)
        python3 core7_table.py --target RET_CS  # RET_CS target
"""
import re
import subprocess
import sys
from pathlib import Path
from statistics import NormalDist

import pandas as pd

YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
N_STOCKS = 50
HERE = Path(__file__).resolve().parent
EXAMM = Path.home() / "EXAMM-Extended"
IC_SCRIPT = EXAMM / "core7_ic.py"
NAN = float("nan")
NAIVE_MSE = sum(NormalDist().inv_cdf((i - 0.5) / N_STOCKS) ** 2 for i in range(1, N_STOCKS + 1)) / N_STOCKS
COLUMNS = ["Rank IC", "MSE", "Net(%)", "Sharpe", "MDD(%)"]
FORMATS = ["{:+8.4f}", "{:7.4f}", "{:+8.2f}", "{:+7.2f}", "{:7.2f}"]
WIDTHS = [8, 7, 8, 7, 7]

# default target is RET; --target RET_CS selects the RET_CS runs
args = [a.upper() for a in sys.argv[1:]]
if args not in ([], ["--TARGET", "RET"], ["--TARGET", "RET_CS"]):
    sys.exit("usage: python3 core7_table.py [--target RET|RET_CS]   (default: RET)")
RET = args != ["--TARGET", "RET_CS"]
if RET:   # raw-return MSEs are ~1e-4
    COLUMNS[1], FORMATS[1], WIDTHS[1] = "MSE(raw)", "{:9.3e}", 9
else:     # RET_CS: add the calibrated raw-return MSE next to the RET_CS-unit MSE
    COLUMNS.insert(2, "MSE(raw,cal)"); FORMATS.insert(2, "{:12.3e}"); WIDTHS.insert(2, 12)


def run(*cmd):
    out = subprocess.run(["python3", *map(str, cmd)], capture_output=True, text=True, cwd=HERE)
    return out.stdout + out.stderr


def grab(pattern, text):
    m = re.search(pattern, text)
    return float(m.group(1)) if m else NAN


def naive_raw_mse(s, y):
    """MSE of always predicting a 0 return, in raw daily-return units, over exactly the
    scored test days of set s / year y (the days core7_ic.py scores): mean of RET_raw^2."""
    split = pd.read_csv(EXAMM / "datasets" / f"panels_core7_examm_{y}" / s / "split_info.csv").set_index("split")
    first, last = int(split.loc["test", "first_row"]), int(split.loc["test", "last_row"])
    panel = pd.read_csv(EXAMM / "datasets" / "panels_core7" / f"{s}_core7" / "panel_dates.csv").set_index("row")
    rows = panel.loc[first + 1:last]                    # prediction j forecasts row first + j + 1
    rows = rows[rows["date"] >= f"{y}-01-01"]           # drop the 40-day warm-up
    raw = rows[[c for c in rows.columns if c.startswith("RET_raw_")]].to_numpy()
    return float((raw ** 2).mean())


def predictions_with_raw(s, y, run, split):
    """Predicted RET_CS (rows = days, columns = stocks) of one genome on one split, and the
    actual next-day raw returns of the same stocks and days; test drops the 40-day warm-up.
    Returns None if the prediction files are missing."""
    files = sorted((EXAMM / "test_output" / f"core7_examm_{y}" / run / f"evaluation_{split}")
                   .glob(f"*_{split}_predictions.csv"))
    if len(files) != N_STOCKS:
        return None
    pred = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        pred[f.name[:-len(f"_{split}_predictions.csv")]] = df["predicted_RET_CS"].to_numpy()
    pred = pd.DataFrame(pred)
    info = pd.read_csv(EXAMM / "datasets" / f"panels_core7_examm_{y}" / s / "split_info.csv").set_index("split")
    first = int(info.loc[split, "first_row"])
    panel = pd.read_csv(EXAMM / "datasets" / "panels_core7" / f"{s}_core7" / "panel_dates.csv") \
        .set_index("row").loc[first + pd.RangeIndex(len(pred)) + 1]       # prediction j -> row first + j + 1
    raw = pd.DataFrame({t: panel[f"RET_raw_{t}"].to_numpy() for t in pred.columns})
    keep = (panel["date"] >= (f"{y}-01-01" if split == "test" else "0000")).to_numpy()
    return pred[keep].to_numpy().ravel(), raw[keep].to_numpy().ravel()


def calibrated_raw_mse(s, y, run):
    """Raw-return MSE of RET_CS forecasts turned into return forecasts with
    raw_hat = a + b * predicted_RET_CS, a and b fitted by least squares on the
    validation year (pooled over the stocks) and applied unchanged to the test year."""
    val, test = predictions_with_raw(s, y, run, "val"), predictions_with_raw(s, y, run, "test")
    if val is None or test is None:
        return NAN
    x, r = val
    b = ((x - x.mean()) * (r - r.mean())).sum() / ((x - x.mean()) ** 2).sum()
    a = r.mean() - b * x.mean()
    x, r = test
    return float(((a + b * x - r) ** 2).mean())


res = {(s, y): {c: NAN for c in COLUMNS} for s in SETS for y in YEARS}
best_run = {}
naive = {(s, y): naive_raw_mse(s, y) for s in SETS for y in YEARS}
for y in YEARS:
    for line in run(IC_SCRIPT, "--year", y, "--target", "RET" if RET else "RET_CS").splitlines():
        t = line.split()
        if len(t) >= 10 and t[2].split("/")[0] in SETS:          # year split run ... IC IC_t pos MSE [naive_MSE]
            r = res[(t[2].split("/")[0], y)]
            r[COLUMNS[0]], r[COLUMNS[1]] = float(t[6]), float(t[9])
            best_run[(t[2].split("/")[0], y)] = t[2]
    for s in SETS:
        out = run(HERE / "main_core7.py", s, y, "RET" if RET else "RET_CS")
        r = res[(s, y)]
        r["Net(%)"] = grab(r"Portfolio return for daily_long_short_return is (-?[\d.]+)%", out)
        r["Sharpe"] = grab(r"Sharpe for daily_long_short_return is (-?[\d.]+|nan)", out)
        r["MDD(%)"] = grab(r"MDD for daily_long_short_return is (-?[\d.]+)%", out)
        if not RET:   # RET_CS forecasts turned into raw-return forecasts, calibrated on validation
            if (s, y) in best_run:
                r["MSE(raw,cal)"] = calibrated_raw_mse(s, y, best_run[(s, y)])

rows = [(f"Pooled, {s}", [[res[(s, y)][c] for c in COLUMNS] for y in YEARS]) for s in SETS]
rows.append(("Pooled, mean", [[sum(res[(s, y)][c] for s in SETS) / len(SETS) for c in COLUMNS] for y in YEARS]))

block = sum(WIDTHS) + len(WIDTHS) - 1
naive_line = ", ".join(f"{y} = {sum(naive[(s, y)] for s in SETS) / len(SETS):.3e}" for y in YEARS)
if RET:
    print("RET target -- MSE in raw daily-return units. Naive MSE (always predict a 0 return), "
          f"mean of 4 sets: {naive_line}\n")
else:
    print(f"RET_CS target -- naive MSE (always predict 0, {N_STOCKS} stocks) = {NAIVE_MSE:.4f}.\n"
          f"MSE(raw,cal) = raw-return MSE of a + b x predicted RET_CS, a and b fitted on the validation year "
          f"(needs SPLIT=val predictions);\nnaive raw-return MSE (always predict a 0 return), mean of 4 sets: {naive_line}\n")
    missing = [f"{s}/{y}" for s in SETS for y in YEARS if res[(s, y)]["MSE(raw,cal)"] != res[(s, y)]["MSE(raw,cal)"]]
    if missing:
        print("MSE(raw,cal) is nan where validation predictions are missing; create them with:\n"
              "  for Y in 2022 2023 2024; do SPLIT=val YEAR=$Y TARGET=RET_CS sh ~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7.sh; done\n")
print(f"{'':16}" + "  ".join(f"{'Test ' + str(y):^{block}}" for y in YEARS))
print(f"{'Construction':16}" + "  ".join(" ".join(f"{c:>{w}}" for c, w in zip(COLUMNS, WIDTHS)) for _ in YEARS))
for name, vals in rows:
    print(f"{name:16}" + "  ".join(" ".join(f.format(v) for f, v in zip(FORMATS, cell)) for cell in vals))
