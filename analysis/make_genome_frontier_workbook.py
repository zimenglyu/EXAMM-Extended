#!/usr/bin/env python3
"""Formatted Excel workbook of the genome frontier experiment (core7_genome_frontier.py).

Reads the three CSVs that script saved (nothing is recomputed) and writes
results/core7_genome_frontier.xlsx:
  Summary          test 2022-2024 joined: chosen genome, both frontier ends, EXAMM per-set default, equal weight
  Per year         test return and Sharpe per year for the same strategies
  Genomes          all 40 candidates per fold: validation and test MSE, Rank IC, return, Sharpe; frontier / chosen flags
  Frontier <year>  validation MSE vs validation return (all genomes, frontier line, chosen genome) and the same on test
  Notes            design and caveats
Run on the Mac after core7_genome_frontier.py and core7_genome_frontier_plots.py (same --axis):
    cd ~/Financial_toolbox && python3 make_genome_frontier_workbook.py                # -> core7_genome_frontier.xlsx
    cd ~/Financial_toolbox && python3 make_genome_frontier_workbook.py --axis sharpe  # -> core7_genome_frontier_sharpe.xlsx
    cd ~/Financial_toolbox && python3 make_genome_frontier_workbook.py --axis residual  # -> core7_genome_frontier_residual.xlsx
    add --rule ls55 (same as the core7_genome_frontier.py run) for the daily L/S 5/5 version (_ls55.xlsx)
--axis residual: y = residual alpha beyond mean reversion (%/yr); Summary / Per year / Genomes add alpha, its
t-stat, beta and reversal exposure against the 10-day z-score reversal benchmark.
Charts: --x rank (default) = MSE percentile rank among the year's 40 candidates; --x gain = MSE gain over naive.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import Reference, ScatterChart, Series
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.drawing.line import LineProperties
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

RESULTS = Path(__file__).resolve().parent / "results"
AXIS = sys.argv[sys.argv.index("--axis") + 1].lower() if "--axis" in sys.argv else "return"
if AXIS not in ("return", "sharpe", "residual"):
    sys.exit("usage: python3 make_genome_frontier_workbook.py [--axis return|sharpe|residual]")
TAG, ACOL = {"return": ("", "return_%"), "sharpe": ("_sharpe", "sharpe"), "residual": ("_residual", "alpha_%")}[AXIS]
YNAME = {"return": "net return (%)", "sharpe": "Sharpe", "residual": "residual alpha (%/yr)"}[AXIS]
RULE_ID = sys.argv[sys.argv.index("--rule") + 1].lower() if "--rule" in sys.argv[:-1] else "overlap"
if RULE_ID not in ("overlap", "ls55"):
    sys.exit("usage: python3 make_genome_frontier_workbook.py [--axis return|sharpe|residual] [--rule overlap|ls55] [--x rank|gain]")
RULE_NAME, RULE_TEXT = {"overlap": ("overlap 5/5 hold 10", "overlapping long-short, top/bottom 5, each day's book held 10 days"),
                        "ls55": ("daily L/S 5/5", "daily long-short, long top 5 / short bottom 5, rebalanced every day "
                                                  "(RET_protocol.md's primary rule, set before any test result)")}[RULE_ID]
TAG += "" if RULE_ID == "overlap" else f"_{RULE_ID}"
OUT = RESULTS / f"core7_genome_frontier{TAG}.xlsx"
F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD = PatternFill("solid", fgColor="E2EFDA"), PatternFill("solid", fgColor="FBE5E5")
PICK = PatternFill("solid", fgColor="FFF2CC")
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)


def need(name):
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} -- run core7_genome_frontier.py first")
    return pd.read_csv(path)


head = need(f"core7_genome_frontier{TAG}_headline.csv")
by_year = need(f"core7_genome_frontier{TAG}_by_year.csv")
genomes = need(f"core7_genome_frontier{TAG}.csv")
YEARS = sorted(genomes["fold"].unique())
# MSE shown as its gain over the naive "always predict 0" forecast, in % (higher = lower MSE): raw MSE barely
# differs between genomes. Needs core7_genome_frontier_plots.py's naive-MSE file; without it, raw MSE is plotted.
naive_path = RESULTS / "core7_genome_frontier_naive.csv"
GAIN = naive_path.exists()
if GAIN:
    nv = pd.read_csv(naive_path)
    for sp in ("val", "test"):
        n = genomes["fold"].map(nv[nv["split"] == sp].set_index("fold")["naive_mse"])
        genomes[f"{sp}_mse_gain_%"] = (n - genomes[f"{sp}_mse"]) / n * 100
# MSE percentile rank among the year's candidates (100 = lowest MSE): the default chart x axis, immune to outliers.
# --x gain charts the MSE gain instead. Both keep MSE's order, so the frontier and the chosen genome do not change.
for sp in ("val", "test"):
    genomes[f"{sp}_mse_pct"] = genomes.groupby("fold")[f"{sp}_mse"].rank(ascending=False, pct=True) * 100
XMODE = sys.argv[sys.argv.index("--x") + 1].lower() if "--x" in sys.argv else "rank"
if XMODE == "rank":
    XV, XT, XLABEL, XSHORT = "val_mse_pct", "test_mse_pct", "MSE percentile rank (100 = lowest MSE)", "MSE pct rank"
elif GAIN:
    XV, XT, XLABEL, XSHORT = "val_mse_gain_%", "test_mse_gain_%", "MSE gain over naive (%, higher = lower MSE)", "MSE gain (%)"
else:
    XV, XT, XLABEL, XSHORT = "val_mse", "test_mse", "MSE (lower is better)", "MSE"

summary = pd.DataFrame({
    "Strategy (genome chosen on validation)": head["strategy"],
    "Annual return (%)": head["ann_return_%"], "Return 95% low": head["ann_return_ci_lo"],
    "Return 95% high": head["ann_return_ci_hi"], "Volatility (%)": head["ann_vol_%"],
    "Sharpe": head["sharpe"], "Sharpe 95% low": head["sharpe_ci_lo"], "Sharpe 95% high": head["sharpe_ci_hi"],
    "PSR": head["psr"], "Max drawdown (%)": head["max_dd_%"], "Genomes used": head["genomes"]})
if AXIS == "residual":
    summary.insert(summary.columns.get_loc("Genomes used"), "Residual alpha (%/yr)", head["alpha_%"])
    summary.insert(summary.columns.get_loc("Genomes used"), "Alpha t-stat", head["alpha_t"])
    summary.insert(summary.columns.get_loc("Genomes used"), "Beta on reversal", head["beta"])

per_year = []
for name, g in by_year.groupby("strategy", sort=False):
    g = g.set_index("year")
    per_year.append({"Strategy": name,
                     **{f"{y} return (%)": g.loc[y, "ann_return_%"] for y in YEARS},
                     **{f"{y} Sharpe": g.loc[y, "sharpe"] for y in YEARS},
                     **({f"{y} residual alpha (%/yr)": g.loc[y, "alpha_%"] for y in YEARS} if AXIS == "residual" else {}),
                     **{f"{y} genome": g.loc[y, "genome"] for y in YEARS}})
per_year = pd.DataFrame(per_year)

cols = {"fold": "Test year", "source_set": "Trained on", "repeat": "Repeat", "genome": "Genome",
        "val_mse": "Val MSE", "val_mse_gain_%": "Val MSE gain over naive (%)",
        "val_mse_pct": "Val MSE percentile (100 = lowest)", "val_rank_ic": "Val Rank IC", "val_return_%": "Val return (%)",
        "val_sharpe": "Val Sharpe", "test_mse": "Test MSE", "test_mse_gain_%": "Test MSE gain over naive (%)",
        "test_mse_pct": "Test MSE percentile (100 = lowest)", "test_rank_ic": "Test Rank IC",
        "test_return_%": "Test return (%)", "test_sharpe": "Test Sharpe", "val_frontier": "On val frontier",
        "chosen": "Picked as"}
if AXIS == "residual":
    extra = {f"{sp}_{k}": f"{lab} {name}" for sp, lab in (("val", "Val"), ("test", "Test"))
             for k, name in (("reversal_exposure", "reversal exposure"), ("beta", "beta on reversal"),
                             ("alpha_%", "residual alpha (%/yr)"), ("alpha_t", "alpha t-stat"))}
    cols = {k: v for k, v in list(cols.items())[:-2]} | extra | {k: cols[k] for k in ("val_frontier", "chosen")}
table = genomes[[c for c in cols if c in genomes]].rename(columns=cols).sort_values(["Test year", "Val MSE"])
table["On val frontier"] = table["On val frontier"].map({True: "Yes", False: ""})
table["Picked as"] = table["Picked as"].fillna("")

notes = [
    "Design (professor's request): only the RET EXAMM runs on the paper's folds -- train <= Y-2, validation Y-1, "
    "test Y for Y = 2022, 2023, 2024. Validation replaces the selection year for every choice; test is reported once.",
    "Candidates per test year: the 40 global best genomes (4 panel sets x 10 runs), each used as ONE genome shared by "
    f"all 4 sets (200 stocks). Trading rule: {RULE_TEXT}, costs on, the 4 sets combined with equal weight.",
    "Pareto frontier on validation: lower validation MSE (raw daily-return units, mean of the 4 sets) and higher "
    "validation net return. Pre-set choice: the frontier genome with the lowest rank sum (MSE rank + return rank); "
    "the two frontier ends (lowest MSE, highest return) are shown for comparison only.",
    "Per-set EXAMM default = each set's own lowest-validation-MSE genome on its own set (the usual paper setup).",
    "Intervals: 95% stationary block bootstrap; PSR = P(true Sharpe > 0), 0.95 the usual bar. Green = Sharpe interval "
    "above zero; red = negative Sharpe; yellow = genomes picked on validation.",
    "Caveats: validation is EXAMM's own fitness year (optimistic); 2022-2024 was examined earlier in this project; "
    "three test years only; cross-set MSE is approximate (each set has its own RET scaling).",
]
if AXIS == "residual":
    notes[2] = ("Pareto frontier on validation: lower validation MSE and higher validation RESIDUAL ALPHA beyond mean "
                "reversion. Pre-set choice: the frontier genome with the lowest rank sum (MSE rank + alpha rank); the two "
                "frontier ends (lowest MSE, highest alpha) are shown for comparison only.")
    notes += [
        "Mean-reversion signal: split-adjusted price rebuilt from RET_raw (cumulative product of 1 + RET_raw over the "
        "whole panel); z_t = (P_t - 10-day mean) / 10-day sample std (ddof 1, days t-9..t, zero/undefined std -> 0); "
        "forecast = -z_t (data up to day t only), aligned and traded exactly like a genome's predictions (same rule, "
        "costs, 4 sets combined) = the reversal benchmark.",
        "Reversal exposure = mean daily cross-sectional Spearman correlation between the genome's forecast and -z on the "
        "scored days, averaged over the 4 sets.",
        "Residual alpha: OLS of the genome's daily combined book return on the reversal benchmark's, same days: "
        "r = a + beta x r_reversal + e; alpha = a x 252 x 100 (%/yr); t-stat uses the plain OLS standard error "
        "(no autocorrelation correction, so it overstates significance for overlapping holdings).",
    ]

sheets = [("Summary", summary, f"Genome Pareto frontier (val MSE vs val {YNAME}): test 2022-2024 joined, {RULE_NAME}"),
          ("Per year", per_year, "Test return and Sharpe per year"),
          ("Genomes", table, "All candidate genomes: validation and test scores")]
RESULTS.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    for name, df, _ in sheets:
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)

wb = load_workbook(OUT)
for name, df, title in sheets:
    ws = wb[name]
    ws["A1"], ws["A1"].font = title, Font(name=F, bold=True, size=13)
    headers = [c.value for c in ws[2]]
    for c in ws[2]:
        c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[2].height = 36
    for row in ws.iter_rows(min_row=3):
        for c in row:
            c.font, c.border = Font(name=F), BORDER
            h = str(headers[c.column - 1])
            if isinstance(c.value, float):
                c.number_format = ("0" if "percentile" in h else "+0.000;-0.000" if "gain" in h else "0.000E+00" if "MSE" in h else "+0.0000;-0.0000" if "IC" in h
                                   else "0.00" if h == "PSR" else "+0.00;-0.00" if "Sharpe" in h or "return" in h.lower()
                                   else "0.00")
        vals = {h: row[i].value for i, h in enumerate(headers)}
        fill = None
        if name == "Genomes" and vals.get("Picked as"):
            fill = PICK
        elif isinstance(vals.get("Sharpe 95% low"), (int, float)) and vals["Sharpe 95% low"] > 0:
            fill = GOOD
        elif isinstance(vals.get("Sharpe"), (int, float)) and vals["Sharpe"] < 0:
            fill = BAD
        if fill:
            for c in row:
                c.fill = fill
    for i, h in enumerate(headers, 1):
        width = max([len(str(h))] + [len(f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value or ""))
                                     for c in ws[get_column_letter(i)][2:]])
        ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 48)
    ws.freeze_panes = "B3"

# one chart sheet per test year: validation frontier and the same genomes on test.
# Charts sit at the top of the sheet (visible on opening); their data table starts below them at DATA_HDR.
# Colours: all genomes = small pale grey, frontier = dark blue diamonds (+ line on validation),
# chosen genome = large red triangle drawn last so it sits on top.
DATA_HDR = 28
STYLE = {"all": ("circle", 6, "BFBFBF", "A6A6A6"), "frontier": ("diamond", 10, "1F4E9A", "1F3864"),
         "chosen": ("triangle", 14, "E03C31", "7F1D1D")}
GRID = GraphicalProperties(ln=LineProperties(solidFill="E7E6E6"))
for y in YEARS:
    g = genomes[genomes["fold"] == y].dropna(subset=["val_mse", "val_return_%"])
    ws = wb.create_sheet(f"Frontier {y}")
    ws["A1"] = (f"Test year {y}: validation {XLABEL} vs validation {YNAME} (left chart) "
                f"and the same genomes on test (right)")
    ws["A1"].font = Font(name=F, bold=True, size=13)
    ws["A2"] = ("Red triangle = genome chosen on validation; blue diamonds = validation Pareto frontier; "
                f"grey = the other candidates. Chart data from row {DATA_HDR}.")
    ws["A2"].font = Font(name=F, italic=True, color="595959")
    blocks = {"all": g, "frontier": g[g["val_frontier"] == True].sort_values(XV),
              "chosen": g[g["chosen"].fillna("").str.contains("chosen")]}
    col = 1
    refs = {}
    for label, part in blocks.items():
        short = XSHORT
        hdr = [f"{label}: val {short}", f"{label}: val {YNAME}", f"{label}: test {short}", f"{label}: test {YNAME}"]
        for j, h in enumerate(hdr):
            c = ws.cell(row=DATA_HDR, column=col + j, value=h)
            c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
            c.alignment = Alignment(wrap_text=True, horizontal="center")
            ws.column_dimensions[get_column_letter(col + j)].width = 13
        for i, (_, r) in enumerate(part.iterrows(), start=DATA_HDR + 1):
            for j, k in enumerate((XV, f"val_{ACOL}", XT, f"test_{ACOL}")):
                c = ws.cell(row=i, column=col + j, value=float(r[k]) if np.isfinite(r[k]) else None)
                c.font = Font(name=F)
                c.number_format = ("0.000E+00" if k.endswith("_mse") else "0" if k.endswith("_pct")
                                   else "+0.00;-0.00")
                if label == "chosen":
                    c.fill = PICK
        refs[label] = (col, DATA_HDR + len(part))
        col += 5
    for split, anchor in (("val", "A3"), ("test", "J3")):
        sc = ScatterChart()
        sc.title = f"{y}: {'validation' if split == 'val' else 'test'} {XSHORT} vs {YNAME}"
        sc.style = 13
        sc.x_axis.title, sc.y_axis.title = f"{split} {XLABEL}", f"{split} {YNAME}"
        # openpyxl >= 3.1 leaves the axes' delete flag unset, which Excel draws as hidden axes (no tick labels)
        sc.x_axis.delete = sc.y_axis.delete = False
        sc.x_axis.majorGridlines.spPr = sc.y_axis.majorGridlines.spPr = GRID
        sc.legend.position = "b"
        for label in ("all", "frontier", "chosen"):
            c0, last = refs[label]
            if last <= DATA_HDR:
                continue
            xo = 0 if split == "val" else 2
            s = Series(Reference(ws, min_col=c0 + xo + 1, min_row=DATA_HDR + 1, max_row=last),
                       Reference(ws, min_col=c0 + xo, min_row=DATA_HDR + 1, max_row=last),
                       title={"all": "other genomes", "frontier": "validation frontier",
                              "chosen": "chosen genome"}[label])
            symbol, size, fill, edge = STYLE[label]
            s.marker.symbol, s.marker.size = symbol, size
            s.marker.graphicalProperties = GraphicalProperties(solidFill=fill)
            s.marker.graphicalProperties.line.solidFill = edge
            if label == "frontier" and split == "val":
                s.graphicalProperties.line.solidFill = fill
                s.graphicalProperties.line.width = 28575   # 2.25 pt
            else:
                s.graphicalProperties.line.noFill = True
            sc.series.append(s)
        if XMODE == "rank":
            sc.x_axis.scaling.min, sc.x_axis.scaling.max = 0, 100
        sc.height, sc.width = 12.5, 21
        ws.add_chart(sc, anchor)

ws = wb["Notes"]
ws.column_dimensions["A"].width = 140
for row in ws.iter_rows():
    for c in row:
        c.font, c.alignment = Font(name=F, bold=(c.row == 1)), Alignment(wrap_text=True)
wb.save(OUT)
print(f"Saved {OUT}")
