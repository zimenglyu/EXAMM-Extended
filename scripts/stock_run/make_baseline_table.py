#!/usr/bin/env python3
# Emit the per-stock EXAMM baseline results table for the paper, in the paper's
# own reporting convention: BEST-of-runs (the minimum test error over the
# repeated runs per stock -- matching the 2025 Evostar Tables 2-3 "Best MAE" /
# "Best MSE"). Reads collect+evaluate output (test_summary.csv) and writes a
# booktabs LaTeX table plus a plain-text preview -- no hand transcription.
#
# Usage:
#   python3 scripts/stock_run/make_baseline_table.py [test_summary.csv] [out.tex]
# defaults:
#   in : test_output/baseline_best/test_summary.csv
#   out: <same dir>/baseline_table.tex
#
# Columns used: min_test_mae, min_test_mse (best over the stock's runs).

import csv, os, sys

IN  = sys.argv[1] if len(sys.argv) > 1 else "test_output/baseline_best/test_summary.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(IN), "baseline_table.tex")

if not os.path.isfile(IN):
    sys.exit(f"ERROR: {IN} not found -- run collect_best_genomes.sh + evaluate_baseline.sh first")

def sci(x):
    """format like the paper: 2.89e-4 (not 2.89e-04)."""
    m, e = f"{x:.2e}".split("e")
    return f"{m}e{int(e)}"

rows = []
with open(IN) as f:
    for d in csv.DictReader(f):
        rows.append((d["ticker"], float(d["min_test_mae"]), float(d["min_test_mse"])))
rows.sort(key=lambda r: r[0])

sum_mae = sum(r[1] for r in rows)
sum_mse = sum(r[2] for r in rows)

# --- LaTeX (booktabs), matching the paper's table style ---
tex = []
tex.append(r"\begin{table}[t]")
tex.append(r"\centering")
tex.append(r"\caption{EXAMM evolved RNN best-of-runs test error per stock "
           f"({len(rows)} stocks, best over the repeated runs), reproducing the "
           r"individual-stock baseline. Sum is over all stocks.}")
tex.append(r"\label{tab:examm-baseline}")
tex.append(r"\begin{tabular}{lrr}")
tex.append(r"\toprule")
tex.append(r"Stock & Best MAE & Best MSE \\")
tex.append(r"\midrule")
for t, mae, mse in rows:
    tex.append(f"{t} & {mae:.5f} & {sci(mse)} " + r"\\")
tex.append(r"\midrule")
tex.append(f"Sum & {sum_mae:.5f} & {sci(sum_mse)} " + r"\\")
tex.append(r"\bottomrule")
tex.append(r"\end{tabular}")
tex.append(r"\end{table}")
tex = "\n".join(tex) + "\n"

with open(OUT, "w") as f:
    f.write(tex)

# --- plain-text preview to stdout ---
print(f"{'Stock':<8}{'Best MAE':>10}{'Best MSE':>11}")
print("-" * 29)
for t, mae, mse in rows:
    print(f"{t:<8}{mae:>10.5f}{sci(mse):>11}")
print("-" * 29)
print(f"{'Sum':<8}{sum_mae:>10.5f}{sci(sum_mse):>11}")
print()
print(f"LaTeX table written to: {OUT}")
