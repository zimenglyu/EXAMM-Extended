#!/usr/bin/env python3
"""Readable plots of the genome frontier (replaces results/core7_genome_frontier.png).

Raw MSE barely differs between genomes (all are close to the naive "always predict 0" MSE), so plotted
as is the 40 genomes squash into a thin strip. Here MSE is shown as its improvement over the naive
forecast, in %:  MSE gain = (naive MSE - MSE) / naive MSE x 100  (higher is better; 0 = no better than
predicting a 0 return). It is a monotone rescaling of MSE, so the frontier and the chosen genomes are
exactly those of core7_genome_frontier.py; nothing is re-chosen or re-traded.

Per test year, two panels (the earlier third panel, MSE vs Rank IC / reversal exposure, was dropped;
the console summary below still prints the MSE-Rank IC correlation):
  1. validation: MSE gain vs net return of the fixed rule (the frontier the genome was chosen on)
  2. test: the same genomes (descriptive)
Each panel prints the Spearman correlation of its two axes. The x axis is zoomed to the bulk of the
genomes; genomes far outside it are drawn at the edge as open markers and counted in the title.

Also writes results/core7_genome_frontier_naive.csv (naive MSE per year and split), which
make_genome_frontier_workbook.py uses for its charts.
Run on the Mac after core7_genome_frontier.py (same --axis as that run):
    cd ~/Financial_toolbox && python3 core7_genome_frontier_plots.py                # MSE vs net return
    cd ~/Financial_toolbox && python3 core7_genome_frontier_plots.py --axis sharpe  # MSE vs Sharpe
    cd ~/Financial_toolbox && python3 core7_genome_frontier_plots.py --axis residual  # MSE vs residual alpha
--rule ls55: plot the daily L/S 5/5 run (same flag as core7_genome_frontier.py; outputs end in _ls55).
x axis: --x rank (default) = MSE percentile rank among the year's 40 candidates (100 = lowest MSE; immune to
outliers); --x gain = MSE gain over naive in %. Both keep MSE's order, so the frontier and choice are unchanged.
"""
import sys

import numpy as np
import pandas as pd

from core7_genome_frontier import (ACOL, ALABEL, AXIS, EXAMM, RESULTS, RULE, SETS, TAG, YEARS, panel, pred_folder,
                                   read_predictions, scored_mask)

SPLITS = ["val", "test"]
table = pd.read_csv(RESULTS / f"core7_genome_frontier{TAG}.csv")
table["chosen"] = table["chosen"].fillna("")
table["repeat"] = table["repeat"].astype(str)


def naive_mse(y, split):
    """MSE of always predicting a 0 return over the scored days of one year / split, mean of the 4 sets."""
    vals = []
    for s in SETS:
        P = None
        for r in [str(r) for r in range(15, 25)]:
            P = read_predictions(pred_folder(y, s, r, s, split), split)
            if P is not None:
                break
        if P is None:
            return np.nan
        rows, keep = scored_mask(y, s, split, len(P))
        raw = rows[[f"RET_raw_{t}" for t in P.columns]].to_numpy()[keep]
        vals.append(float((raw ** 2).mean()))
    return float(np.mean(vals))


naive = pd.DataFrame([{"fold": y, "split": sp, "naive_mse": naive_mse(y, sp)} for y in YEARS for sp in SPLITS])
naive.to_csv(RESULTS / "core7_genome_frontier_naive.csv", index=False)
for sp in SPLITS:
    n = naive[naive["split"] == sp].set_index("fold")["naive_mse"]
    table[f"{sp}_mse_gain_%"] = (table["fold"].map(n) - table[f"{sp}_mse"]) / table["fold"].map(n) * 100


# x axis: MSE as its PERCENTILE RANK among the year's candidates (default; 100 = lowest MSE, 0 = highest), or
# --x gain: MSE gain over naive in % (zoomed to the bulk of the genomes). Both keep the order of MSE, so the
# frontier and the chosen genome are the same either way; the rank axis is immune to outliers.
XMODE = sys.argv[sys.argv.index("--x") + 1].lower() if "--x" in sys.argv else "rank"
if XMODE not in ("rank", "gain"):
    sys.exit("usage: python3 core7_genome_frontier_plots.py [--axis return|sharpe|residual] [--x rank|gain]")
for sp in SPLITS:
    table[f"{sp}_mse_pct"] = table.groupby("fold")[f"{sp}_mse"].rank(ascending=False, pct=True) * 100


def zoom(x):
    """x limits around the bulk of the genomes (3 x IQR beyond the quartiles)."""
    x = x.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    lo, hi = max(x.min(), q1 - 3 * (q3 - q1)), min(x.max(), q3 + 3 * (q3 - q1))
    pad = (hi - lo) * 0.08 or 0.05
    return lo - pad, hi + pad


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
XSUF, XNAME = (("mse_pct", "MSE percentile rank (100 = lowest MSE)") if XMODE == "rank"
               else ("mse_gain_%", "MSE gain over naive (%, higher = lower MSE)"))
COLUMNS = [("val", f"val_{XSUF}", f"val_{ACOL}", ALABEL, "Validation (frontier chosen here)"),
           ("test", f"test_{XSUF}", f"test_{ACOL}", ALABEL, "Test (same genomes)")]
YSHORT = {"return": "net return (%/yr)", "sharpe": "Sharpe ratio", "residual": "residual alpha (%/yr)"}[AXIS]
COLUMNS = [(sp, x, y, YSHORT if y.endswith(ACOL) else lab, h) for sp, x, y, lab, h in COLUMNS]
GREY, NAVY, RED, GREEN = "#b0b0b0", "#1f3864", "#c0392b", "#2e7d32"

fig, axes = plt.subplots(len(YEARS), len(COLUMNS), figsize=(11, 4.3 * len(YEARS)), squeeze=False)
for i, (row_axes, y) in enumerate(zip(axes, YEARS)):
    g = table[table["fold"] == y]
    front = g[g["val_frontier"] == True]
    pick = g[g["chosen"].str.contains("chosen", regex=False)]
    ends = g[g["chosen"].str.contains("frontier end", regex=False) & ~g["chosen"].str.contains("chosen", regex=False)]
    for j, (ax, (sp, xcol, ycol, ylabel, head)) in enumerate(zip(row_axes, COLUMNS)):
        if XMODE == "rank":
            lo, hi = -4, 104
        else:
            lo, hi = zoom(g[xcol])
        X = lambda d: d[xcol].clip(lo, hi)
        ax.scatter(X(g), g[ycol], s=22, c=GREY, edgecolor="none", alpha=0.9, zorder=2)
        f = front.sort_values(xcol)
        # line only where the frontier was formed (validation); on test the same genomes are markers only,
        # since joining them would draw a "frontier" that does not exist there
        ax.plot(X(f), f[ycol], "-o" if sp == "val" else "o", c=NAVY, lw=1.4, ms=5, zorder=3)
        ax.scatter(X(ends), ends[ycol], marker="s", s=55, facecolor="white", edgecolor=GREEN, lw=1.5, zorder=4)
        ax.scatter(X(pick), pick[ycol], marker="*", s=230, c=RED, edgecolor="white", lw=0.6, zorder=5)
        for d, colour in ((pick, RED), (ends, GREEN)):
            for _, r in d.iterrows():
                ax.annotate(f"{r['source_set']}/{r['repeat']}", (min(max(r[xcol], lo), hi), r[ycol]), fontsize=7,
                            color=colour, xytext=(5, 4), textcoords="offset points")
        ax.axhline(0, c="#7f7f7f", lw=0.8, zorder=1)
        ax.set_xlim(lo, hi)
        # empty band at the top of each panel for the rho box, so it never covers a point or its label
        ylo, yhi = ax.get_ylim()
        ax.set_ylim(ylo, yhi + 0.18 * (yhi - ylo))
        rho = g[[xcol, ycol]].dropna().corr(method="spearman").iloc[0, 1]
        ax.text(0.02, 0.97, f"Spearman rho = {rho:+.2f}", transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"))
        if i == 0:
            ax.set_title(head, fontweight="bold")
        if i == len(YEARS) - 1:
            ax.set_xlabel(XNAME)
        ax.set_ylabel(ylabel)
    row_axes[0].annotate(f"Test year {y}\n(validation {y - 1})", xy=(-0.28, 0.5), xycoords="axes fraction",
                         rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
handles = [Line2D([], [], ls="", marker="o", ms=6, color=GREY, label="candidate genomes (40 per year)"),
           Line2D([], [], color=NAVY, marker="o", ms=5, label="validation Pareto frontier"),
           Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=GREEN, mew=1.5, label="frontier ends"),
           Line2D([], [], ls="", marker="*", ms=14, color=RED, label="chosen genome (pre-set rule)")]
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
title_axis = {"return": "net return", "sharpe": "Sharpe", "residual": "residual alpha beyond mean reversion"}[AXIS]
fig.suptitle(f"Genome Pareto frontier: MSE vs {title_axis} of {RULE[0]}\n(one EXAMM genome for all 4 panel sets)",
             fontsize=13, fontweight="bold")
fig.tight_layout(rect=(0.03, 0.04, 1, 0.96))
fig.savefig(RESULTS / f"core7_genome_frontier{TAG}.png", dpi=160)
print(f"Saved {RESULTS / f'core7_genome_frontier{TAG}.png'} (x axis: {XNAME})")
print(f"Saved {RESULTS / 'core7_genome_frontier_naive.csv'}")
print("\nSpearman correlation of validation MSE gain with:")
for y in YEARS:
    g = table[table["fold"] == y]
    c = g[["val_mse_gain_%", f"val_{ACOL}", "val_rank_ic", f"test_{ACOL}"]].corr(method="spearman")["val_mse_gain_%"]
    extra = ""
    if AXIS == "residual":
        cc = g[["val_mse_gain_%", "val_reversal_exposure", "test_mse_gain_%", "test_alpha_%"]].corr(method="spearman")
        extra = (f", val reversal exposure {cc.loc['val_mse_gain_%', 'val_reversal_exposure']:+.2f}"
                 f" | test MSE gain vs test alpha {cc.loc['test_mse_gain_%', 'test_alpha_%']:+.2f}")
    print(f"  {y}: val {AXIS} {c[f'val_{ACOL}']:+.2f}, val Rank IC {c['val_rank_ic']:+.2f}, "
          f"test {AXIS} {c[f'test_{ACOL}']:+.2f}{extra}")
