#!/usr/bin/env python3
"""Does a run's VALIDATION score predict its TEST result? Scatter plots of the 40 runs per test year
(10 repeats x 4 panel sets), from results/core7_main_pareto.csv (core7_main_results.py --pareto); nothing is recomputed.

Columns: validation net return, validation MSE and validation Rank IC, each against the test-year net return.
Each panel's title gives the Spearman rank correlation (rho) and its p-value; rho near 0 = no predictive value.
Writes results/core7_main_pareto_corr.png and prints the correlation table.
Run on the Mac:  cd ~/Financial_toolbox && python3 plot_core7_pareto_corr.py && open results/core7_main_pareto_corr.png
--k 5: the 5/5 results (reads core7_main_k5_pareto.csv, writes core7_main_k5_pareto_corr.png); default 10/10.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import spearmanr

RESULTS = Path(__file__).resolve().parent / "results"
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv[:-1] else 10   # book size (default 10/10, the paper)
EXP = sys.argv[sys.argv.index("--experiment") + 1] if "--experiment" in sys.argv[:-1] else "core7"   # run family
TAG = ("" if K == 10 else f"_k{K}") + ("" if EXP == "core7" else f"_{EXP}")
path = RESULTS / f"core7_main{TAG}_pareto.csv"
if not path.exists():
    raise SystemExit(f"missing {path} -- run: python3 core7_main_results.py --pareto" + ("" if K == 10 else f" --k {K}"))
pt = pd.read_csv(path)

X = [("val_return_%", "validation net return (%)"), ("val_mse", "validation MSE"),
     ("val_rank_ic", "validation Rank IC")]
Y = ("test_return_%", "test net return (%)")
COLOURS = {"set1": "#1f77b4", "set2": "#ff7f0e", "set3": "#2ca02c", "set4": "#d62728"}
years = sorted(pt["test_year"].unique())

rows = []
fig, axes = plt.subplots(len(years), len(X), figsize=(5 * len(X), 3.8 * len(years)), squeeze=False)
for i, year in enumerate(years):
    g = pt[pt["test_year"] == year]
    for j, (col, label) in enumerate(X):
        ax = axes[i][j]
        rho, p = spearmanr(g[col], g[Y[0]])
        rows.append({"test_year": year, "validation score": label, "spearman_rho": rho, "p_value": p, "runs": len(g)})
        for s, h in g.groupby("set"):
            ax.scatter(h[col], h[Y[0]], s=30, c=COLOURS.get(s, "grey"), label=f"V{s[-1]} ({s})", zorder=3)
        chosen = g[g["choice"].fillna("").str.contains("lowest val MSE", regex=False)]
        ax.scatter(chosen[col], chosen[Y[0]], s=110, facecolors="none", edgecolors="black", lw=1.4,
                   label="lowest val MSE (chosen)", zorder=4)
        ax.axhline(0, c="#888888", lw=0.8)
        ax.grid(color="#e6e6e6")
        ax.set_title(f"test {year}: rho = {rho:+.2f} (p = {p:.2f})", fontsize=10)
        if col == "val_mse":
            ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        if i == len(years) - 1:
            ax.set_xlabel(label)
        if j == 0:
            ax.set_ylabel(f"{year}: {Y[1]}")
axes[0][0].legend(fontsize=7, loc="best")
fig.suptitle(f"Does the validation score predict the test result? (each dot = one EXAMM run; retrain every year; L/S {K}/{K})",
             fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = RESULTS / f"core7_main{TAG}_pareto_corr.png"
fig.savefig(out, dpi=150)

tab = pd.DataFrame(rows).pivot(index="validation score", columns="test_year", values="spearman_rho")
pv = pd.DataFrame(rows).pivot(index="validation score", columns="test_year", values="p_value")
pd.set_option("display.width", 160)
print("Spearman correlation of each validation score with the test net return (40 runs per year):")
print(tab.round(2).to_string())
print("\np-values (below 0.05 = unlikely to be chance):")
print(pv.round(2).to_string())
print("\nNote: lower MSE is better, so a useful MSE has a NEGATIVE rho; return and IC should be POSITIVE.")
print(f"Saved {out}")

# ---- within set: removes the level differences between panel sets (e.g. V3 usually high, V1 low) ----
# every score and the test return are ranked WITHIN each (test year, set) group of 10 runs, then the ranks
# are correlated: "inside one set, does the run that is better on validation also do better on test?"
wr = pt.copy()
for col in [c for c, _ in X] + [Y[0]]:
    wr[col] = wr.groupby(["test_year", "set"])[col].rank()
within, per_group = [], []
for col, label in X:
    for year in years + ["all"]:
        g = wr if year == "all" else wr[wr["test_year"] == year]
        rho, p = spearmanr(g[col], g[Y[0]])
        within.append({"validation score": label, "test_year": str(year), "rho": rho, "p": p})
    for (year, s), g in pt.groupby(["test_year", "set"]):
        per_group.append({"validation score": label, "test_year": year, "set": s,
                          "rho": spearmanr(g[col], g[Y[0]])[0]})
within = pd.DataFrame(within)
cols = [str(y) for y in years] + ["all"]
print("\nWITHIN SET (runs ranked inside each set and year, so set differences are removed):")
print("correlation (40 runs per year, 120 for 'all'):")
print(within.pivot(index="validation score", columns="test_year", values="rho").reindex(columns=cols).round(2).to_string())
print("\np-values:")
print(within.pivot(index="validation score", columns="test_year", values="p").reindex(columns=cols).round(2).to_string())
pg = pd.DataFrame(per_group)
print("\nper set and year (10 runs each, too few to be significant alone; shows whether the sign is consistent):")
print(pg.pivot_table(index=["validation score", "set"], columns="test_year", values="rho").round(2).to_string())
useful = {"validation MSE": -1, "validation Rank IC": 1, "validation net return (%)": 1}
print("\ngroups (of 12) where the score points the useful way:")
for label, sign in useful.items():
    r = pg[pg["validation score"] == label]["rho"]
    print(f"  {label}: {(r * sign > 0).sum()} of {len(r)}  (mean rho {r.mean():+.2f})")
