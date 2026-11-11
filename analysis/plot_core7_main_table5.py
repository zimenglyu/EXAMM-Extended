#!/usr/bin/env python3
"""Your per-panel results next to the paper's Table 5 ("Net return (%) using Model's Predicted Return").

For every panel set (V1-V4 = set1-set4) and test year 2022-2024: your EXAMM net return with the paper's
trading rule -- retrain every year (each year's own fold models; the paper's design) and never retrain (the
2022-fold models) -- next to the paper's EXAMM, buy-and-hold (B&H) and equal-weight (EW) columns.
Reads results/core7_main_per_set.csv (written by core7_main_results.py, per-set tables on by default);
nothing is recomputed. Writes:
  results/core7_main_table5.png   grouped bars per panel and year, one panel of bars per V1-V4
  results/core7_main_table5.xlsx  the table (rows = panel x year, like Table 5) and an Excel bar chart
Run on the Mac after core7_main_results.py:
    cd ~/Financial_toolbox && python3 plot_core7_main_table5.py && open results/core7_main_table5.png
Your "net return" is the annualized return of each test year (about one year of trading days); the paper's
is the return over the test year, so the two are directly comparable up to small day-count differences.
"""
from pathlib import Path

import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"
SETS = ["set1", "set2", "set3", "set4"]
YEARS = [2022, 2023, 2024]
# the paper's Table 5 (net return %), columns B&H, EW, EXAMM
PAPER = {("V1", 2022): (-7.16, -14.26, -4.63), ("V1", 2023): (5.18, -1.67, 21.34), ("V1", 2024): (-4.73, -9.92, 17.19),
         ("V2", 2022): (-11.42, -18.91, 35.22), ("V2", 2023): (15.37, 7.40, 16.92), ("V2", 2024): (12.13, 3.46, 13.80),
         ("V3", 2022): (-11.39, -19.08, 36.68), ("V3", 2023): (14.77, 7.05, -3.82), ("V3", 2024): (13.11, 3.89, -2.88),
         ("V4", 2022): (-10.25, -17.33, -5.43), ("V4", 2023): (9.41, 3.16, 1.84), ("V4", 2024): (6.69, -1.14, 5.96)}

path = RESULTS / "core7_main_per_set.csv"
if not path.exists():
    raise SystemExit(f"missing {path} -- run: python3 core7_main_results.py")
per = pd.read_csv(path)


def mine(fold_of, v, year):
    s = SETS[int(v[1]) - 1]
    r = per[(per["set"] == s) & (per["test_year"] == year) & (per["genomes_from_fold"] == fold_of(year))]
    return float(r["ann_return_%"].iloc[0]) if len(r) else None


rows = []
for v in ["V1", "V2", "V3", "V4"]:
    for y in YEARS:
        bh, ew, ex = PAPER[(v, y)]
        rows.append({"Panel": v, "Test yr": y,
                     "Paper B&H": bh, "Paper EW": ew, "Paper EXAMM": ex,
                     "Yours: retrain every year": mine(lambda yr: yr, v, y),
                     "Yours: never retrain": mine(lambda yr: YEARS[0], v, y)})
table = pd.DataFrame(rows)
table["Retrain minus paper EXAMM"] = table["Yours: retrain every year"] - table["Paper EXAMM"]
mean = table.drop(columns=["Panel", "Test yr"]).mean(numeric_only=True)
table = pd.concat([table, pd.DataFrame([{"Panel": "Mean", "Test yr": "", **mean.to_dict()}])], ignore_index=True)

pd.set_option("display.width", 200)
print("Net return (%) per panel and test year -- paper's Table 5 vs yours (paper's trading rule, costs on):")
print(table.round(2).to_string(index=False))

# ---------------------------------------------------------------- figure ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    series = [("Paper EXAMM", "#1f3864"), ("Yours: retrain every year", "#c0392b"),
              ("Yours: never retrain", "#e59866"), ("Paper EW", "#9e9e9e"), ("Paper B&H", "#cfcfcf")]
    fig, axes = plt.subplots(1, 4, figsize=(18, 5.2), sharey=True)
    width = 0.16
    for ax, v in zip(axes, ["V1", "V2", "V3", "V4"]):
        sub = table[table["Panel"] == v]
        x = np.arange(len(YEARS))
        for i, (col, colour) in enumerate(series):
            vals = sub[col].astype(float).to_numpy()
            ax.bar(x + (i - 2) * width, vals, width, color=colour, label=col, zorder=3)
        ax.axhline(0, color="#555555", lw=0.8, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels([str(y) for y in YEARS])
        ax.set_title(f"Panel {v}", fontweight="bold")
        ax.grid(axis="y", color="#e6e6e6", zorder=0)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Net return (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False)
    fig.suptitle("Net return per panel and test year: paper's Table 5 vs yours (paper's trading rule, costs on)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(RESULTS / "core7_main_table5.png", dpi=160)
    print(f"\nSaved {RESULTS / 'core7_main_table5.png'}")
except ImportError:
    print("\nmatplotlib not installed -- skipped the figure (pip3 install --user matplotlib)")

# ---------------------------------------------------------------- workbook with an Excel chart ----
try:
    from openpyxl import load_workbook
    from openpyxl.chart import BarChart, Reference
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
except ImportError:
    raise SystemExit("openpyxl not installed -- no .xlsx (python3 -m pip install --user openpyxl)")

out = RESULTS / "core7_main_table5.xlsx"
with pd.ExcelWriter(out, engine="openpyxl") as xw:
    table.to_excel(xw, sheet_name="Table 5 compared", index=False, startrow=1)
wb = load_workbook(out)
ws = wb["Table 5 compared"]
ws["A1"] = "Net return (%) per panel (V1-V4) and test year: paper's Table 5 vs yours (paper's trading rule, costs on)"
ws["A1"].font = Font(name="Arial", bold=True, size=13)
edge = Side(style="thin", color="BFBFBF")
border = Border(left=edge, right=edge, top=edge, bottom=edge)
for c in ws[2]:
    c.font, c.fill, c.border = Font(name="Arial", bold=True, color="FFFFFF"), PatternFill("solid", fgColor="1F3864"), border
    c.alignment = Alignment(wrap_text=True, horizontal="center", vertical="center")
ws.row_dimensions[2].height = 34
for row in ws.iter_rows(min_row=3):
    for c in row:
        c.font, c.border = Font(name="Arial", bold=(row[0].value == "Mean")), border
        if isinstance(c.value, float):
            c.number_format = "+0.00;-0.00"
            if c.column >= 3:
                c.fill = PatternFill("solid", fgColor="E2EFDA" if c.value > 0 else "FBE5E5")
for i, w in enumerate([8, 8, 11, 11, 13, 16, 16, 16], start=1):
    ws.column_dimensions[chr(64 + i)].width = w
n = len(table) - 1                                     # rows without the Mean row
chart = BarChart()
chart.type, chart.grouping = "col", "clustered"
chart.title = "Net return (%) per panel and year"
chart.y_axis.title = "Net return (%)"
for col in (5, 6, 7, 4):                               # Paper EXAMM, yours retrain, yours never retrain, Paper EW
    chart.add_data(Reference(ws, min_col=col, min_row=2, max_row=2 + n), titles_from_data=True)
labels = Reference(ws, min_col=1, max_col=2, min_row=3, max_row=2 + n)
chart.set_categories(labels)
chart.height, chart.width = 11, 30
ws.add_chart(chart, f"A{n + 6}")
ws.freeze_panes = "C3"
wb.save(out)
print(f"Saved {out}")
