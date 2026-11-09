#!/usr/bin/env python3
"""How long does the Pooled RET forecast's information last? (signal decay)

Each test day t the genome forecasts tomorrow's return (day t+1). This script asks how
well that same forecast ranks the stocks' returns further ahead:

  day-h IC     daily cross-sectional Spearman between the forecast and the return of the
               single day t+h (h = 1 is the usual Rank IC). Shows how long the information
               lasts: if it stays positive for ~10 days, holding ~10 days uses it.
  cum-h IC     the same against the cumulative return of days t+1 .. t+h, i.e. what a
               position opened today and held h days earns.
  top-bottom   average cumulative return over t+1 .. t+h of the K highest-forecast stocks
               minus the K lowest (K = 5), in basis points, before costs: the economic size.

Horizons 1, 2, 3, 5, 10, 15, 20 trading days. Returns are the unclipped raw returns in
panel_dates.csv (a rank correlation does not depend on converting the forecast back to raw
returns). A forecast is scored if its first target day is in the test year (the 40-day
warm-up is dropped); later target days may run into the next year, the outcome is just
the realized return. Days past the end of the panel are dropped.

t-statistics are Newey-West (lag h for the overlapping cumulative horizons, lag 5 for the
single-day ones), because the daily IC series is autocorrelated.

Purely descriptive: it explains the trading results of the existing RET folds; it does
not choose or change any rule of the frozen protocol.

Default: the chosen genome of each RET fold / set (the same ones core7_ic.py scores).
--repeats uses every repeat 15..24 that has test predictions (evaluate_rnn_core7_repeats.sh)
and averages the decay over the repeats of each set / fold.
Keep this file in the EXAMM-Extended folder (next to core7_ic.py).

Usage:  python3 core7_signal_decay.py
        python3 core7_signal_decay.py --repeats
Writes test_output/core7_signal_decay.csv (or core7_signal_decay_repeats.csv).
"""
import argparse
import glob
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent                     # EXAMM-Extended
PANELS = Path(os.getenv("CORE7_PANELS", ROOT / "datasets" / "panels_core7"))
YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
HORIZONS = [1, 2, 3, 5, 10, 15, 20]
K = 5
# chosen genome of each RET fold (lowest validation MSE), as in core7_ic.py
CHOSEN = {
    2022: ["set1/21", "set2/16", "set3/22", "set4/24"],
    2023: ["set1/17", "set2/21", "set3/17", "set4/21"],
    2024: ["set1/15", "set2/15", "set3/22", "set4/24"],
}


def newey_west_t(x, lag):
    """t-statistic of the mean of x with a Newey-West (Bartlett) variance."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan
    e = x - x.mean()
    var = np.sum(e * e) / n     # plain sums: macOS Accelerate matmul can emit spurious warnings
    for k in range(1, min(lag, n - 1) + 1):
        var += 2 * (1 - k / (lag + 1)) * np.sum(e[k:] * e[:-k]) / n
    return x.mean() / np.sqrt(var / n) if var > 0 else np.nan


def daily_spearman(P, R):
    """Cross-sectional Spearman correlation per row of two (days x stocks) arrays."""
    keep = ~(np.isnan(P) | np.isnan(R))
    out = np.full(len(P), np.nan)
    for i in range(len(P)):
        k = keep[i]
        if k.sum() < 10:
            continue
        a = pd.Series(P[i, k]).rank().to_numpy()
        b = pd.Series(R[i, k]).rank().to_numpy()
        a, b = a - a.mean(), b - b.mean()
        d = np.sqrt((a @ a) * (b @ b))
        out[i] = (a @ b) / d if d > 0 else np.nan
    return out


def decay(year, run):
    """Per-day day-h IC, cum-h IC and top-bottom spread of one genome's test forecasts."""
    s = run.split("/")[0]
    files = sorted(glob.glob(f"{ROOT}/test_output/core7_examm_ret_{year}/{run}/evaluation_test/*_test_predictions.csv"))
    if len(files) != 50:
        return None
    pred = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        pred[os.path.basename(f)[:-len("_test_predictions.csv")]] = df["predicted_RET"].to_numpy()
    stocks = sorted(pred)
    P = np.column_stack([pred[t] for t in stocks])                       # days x stocks

    first = int(pd.read_csv(ROOT / "datasets" / f"panels_core7_examm_{year}" / s / "split_info.csv")
                .set_index("split").loc["test", "first_row"])
    panel = pd.read_csv(PANELS / f"{s}_core7" / "panel_dates.csv").set_index("row")
    raw = panel[[f"RET_raw_{t}" for t in stocks]].to_numpy()
    rows = panel.index.to_numpy()
    pos = {r: i for i, r in enumerate(rows)}
    dates = panel["date"].to_numpy()

    target1 = first + np.arange(len(P)) + 1                            # forecast j -> row first + j + 1
    keep = np.array([r in pos and dates[pos[r]] >= f"{year}-01-01" for r in target1])
    P, target1 = P[keep], target1[keep]

    out = []
    for h in HORIZONS:
        day_r = np.full(P.shape, np.nan)
        cum_r = np.full(P.shape, np.nan)
        for i, r in enumerate(target1):
            last = r + h - 1
            if last not in pos:                                         # past the end of the panel
                continue
            block = raw[pos[r]:pos[last] + 1]
            day_r[i] = block[-1]
            cum_r[i] = np.prod(1 + block, axis=0) - 1
        order = np.argsort(P, axis=1)
        top = np.take_along_axis(cum_r, order[:, -K:], axis=1).mean(axis=1)
        bottom = np.take_along_axis(cum_r, order[:, :K], axis=1).mean(axis=1)
        out.append({"horizon": h,
                    "day_ic": daily_spearman(P, day_r),
                    "cum_ic": daily_spearman(P, cum_r),
                    "top_bottom_bp": (top - bottom) * 1e4})
    return out


warnings.filterwarnings("ignore", message="Mean of empty slice")   # days past the panel end

ap = argparse.ArgumentParser()
ap.add_argument("--repeats", action="store_true", help="use every repeat 15..24, averaged per set / fold")
a = ap.parse_args()

rows = []
missing = []
for y in YEARS:
    for s in SETS:
        runs = ([f"{s}/{f}" for f in range(15, 25)] if a.repeats
                else [r for r in CHOSEN[y] if r.startswith(s + "/")])
        per_run = [res for res in (decay(y, r) for r in runs) if res is not None]
        if not per_run:
            missing.append(f"{y} {s}")
            continue
        for k, h in enumerate(HORIZONS):
            # average the per-day series over the repeats (all have the same days)
            day_ic = np.nanmean([r[k]["day_ic"] for r in per_run], axis=0)
            cum_ic = np.nanmean([r[k]["cum_ic"] for r in per_run], axis=0)
            spread = np.nanmean([r[k]["top_bottom_bp"] for r in per_run], axis=0)
            rows.append({"fold": y, "set": s, "horizon": h, "genomes": len(per_run),
                         "day_ic": np.nanmean(day_ic), "day_ic_t": newey_west_t(day_ic, 5),
                         "cum_ic": np.nanmean(cum_ic), "cum_ic_t": newey_west_t(cum_ic, h),
                         "top_bottom_bp": np.nanmean(spread),
                         "top_bottom_t": newey_west_t(spread, h),
                         "_day": day_ic, "_cum": cum_ic, "_spread": spread})

if missing:
    print("no test predictions for: " + ", ".join(missing) + "\n  -> run evaluate_rnn on the test split first"
          + (" (sh scripts/rnn_examples/evaluate_rnn_core7_repeats.sh)" if a.repeats
             else " (SPLIT=test sh scripts/rnn_examples/evaluate_rnn_core7.sh)"))
if not rows:
    raise SystemExit("nothing to score")
res = pd.DataFrame(rows)

# all folds and sets pooled: concatenate the per-day series of every set / fold
pooled = []
for h in HORIZONS:
    part = res[res["horizon"] == h]
    day = np.concatenate(part["_day"].to_list())
    cum = np.concatenate(part["_cum"].to_list())
    spr = np.concatenate(part["_spread"].to_list())
    pooled.append({"horizon": h,
                   "day_ic": np.nanmean(day), "day_ic_t": newey_west_t(day, 5),
                   "cum_ic": np.nanmean(cum), "cum_ic_t": newey_west_t(cum, h),
                   "top_bottom_bp": np.nanmean(spr), "top_bottom_t": newey_west_t(spr, h),
                   "top_bottom_bp_per_day": np.nanmean(spr) / h})
pooled = pd.DataFrame(pooled).set_index("horizon")

res = res.drop(columns=["_day", "_cum", "_spread"])
out = ROOT / "test_output" / ("core7_signal_decay_repeats.csv" if a.repeats else "core7_signal_decay.csv")
res.to_csv(out, index=False)

pd.set_option("display.width", 200)
who = "every repeat 15-24 (averaged per set / fold)" if a.repeats else "the chosen genome of each set / fold"
print(f"Signal decay of the Pooled RET test forecasts, {who}; folds 2022-2024, 4 sets.")
print(f"day_ic = IC with the return of day t+h; cum_ic = IC with the return of days t+1..t+h;")
print(f"top_bottom_bp = top-{K} minus bottom-{K} cumulative return (basis points, before costs). t = Newey-West.\n")
print("All folds and sets pooled:")
print(pooled.round({"day_ic": 4, "day_ic_t": 2, "cum_ic": 4, "cum_ic_t": 2,
                    "top_bottom_bp": 1, "top_bottom_t": 2, "top_bottom_bp_per_day": 2}).to_string())
for col, label in (("day_ic", "day-h IC"), ("cum_ic", "cum-h IC")):
    print(f"\n{label} per fold (4 sets averaged):")
    print(res.pivot_table(index="fold", columns="horizon", values=col).round(4).to_string())
print(f"\nSaved {out}")
