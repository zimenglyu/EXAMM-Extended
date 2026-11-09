#!/usr/bin/env python3
"""Excel summary workbook of main_core7_protocol.py (RET_protocol.md with the ORIGINAL 17 rules).

Reads the CSVs main_core7_protocol.py saved (nothing is recomputed) and writes
results/original_rules_summary.xlsx:
  Summary        primary (daily L/S 5/5), walk-forward books and equal weight, with intervals and PSR
  All rules      every rule, test 2022-2024: return, volatility, Sharpe, intervals, PSR, DSR, drawdown, turnover
  Per year       test Sharpe per year for every rule (formula average and years positive)
  L-S grid       test Sharpe of the 16 daily long-short rules as a long count x short count grid
  Walk-forward   the rules chosen each year on the earlier test years and how they did
  Notes          method and caveats
Run on the Mac after main_core7_protocol.py:
    cd ~/Financial_toolbox && python3 make_original_rules_workbook.py
Needs openpyxl (python3 -m pip install --user openpyxl if the import fails).
"""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "original_rules_summary.xlsx"
BASE = "core7_protocol_original_rules"
YEARS = [2022, 2023, 2024]
COUNTS = [3, 5, 10, 15]

F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD = PatternFill("solid", fgColor="E2EFDA"), PatternFill("solid", fgColor="FBE5E5")
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)


def need(name):
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} -- run main_core7_protocol.py first")
    return pd.read_csv(path)


rules = need(f"{BASE}.csv").sort_values("sharpe", ascending=False)
headline = need(f"{BASE}_headline.csv").rename(columns={"Unnamed: 0": "strategy"})
walk = need(f"{BASE}_walkforward.csv")

summary = headline[[c for c in ["strategy", "ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%",
                                "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "psr", "max_dd_%"] if c in headline]]
all_rules = rules[[c for c in ["rule", "ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%", "sharpe",
                               "sharpe_ci_lo", "sharpe_ci_hi", "psr", "dsr", "max_dd_%", "turnover_x"] if c in rules]]
per_year = rules[["rule"] + [f"sharpe_{y}" for y in YEARS]].rename(columns={f"sharpe_{y}": str(y) for y in YEARS})

grid = pd.DataFrame(index=[f"long {L}" for L in COUNTS], columns=[f"short {S}" for S in COUNTS], dtype=float)
by_rule = rules.set_index("rule")
for L in COUNTS:
    for S in COUNTS:
        if f"L/S {L}/{S}" in by_rule.index:
            grid.loc[f"long {L}", f"short {S}"] = by_rule.loc[f"L/S {L}/{S}", "sharpe"]
grid = grid.reset_index().rename(columns={"index": "test Sharpe"})

notes = [
    "RET_protocol.md (frozen 2026-10-31) run with the ORIGINAL 17 rules only: 16 daily long-short rules "
    "(long L / short S, L and S in 3, 5, 10, 15, rebalanced every day) and the equal-weight benchmark.",
    "These rules were set before any test result was seen (the overlapping and banded rules came later).",
    "Forecasts: Pooled EXAMM, RET target, global best genome (lowest validation MSE) per panel set and fold; "
    "test years 2022-2024; the 4 sets (50 stocks each) combined with equal weight. Costs on.",
    "Primary result (set in advance): daily L/S 5/5. Walk-forward: each test year after 2022, the rules ranked by "
    "Sharpe over the earlier test years; top-1 rule and an equal-weight book of the top 3, scored 2023-2024.",
    "Intervals: 95% stationary block bootstrap (mean block 10 days, 2000 resamples, seed 0). PSR = P(true Sharpe > 0), "
    "0.95 the usual bar. DSR = deflated Sharpe ratio with N = 17 rules.",
    "Turnover = dollars bought, sold and shorted per year / average account value.",
    "Caveats: validation is EXAMM's own fitness data, so the genome choice is not tuned for trading; only two scored "
    "walk-forward years; each year uses a different fold's models.",
    f"Source: {BASE}.csv, _headline.csv and _walkforward.csv in Financial_toolbox/results (main_core7_protocol.py).",
]

sheets = [("Summary", summary), ("All rules", all_rules), ("Per year", per_year), ("L-S grid", grid),
          ("Walk-forward", walk)]
titles = {"Summary": "Original 17 rules: headline results, Pooled RET, test 2022-2024, costs on",
          "All rules": "Every rule, test 2022-2024 joined (DSR with N = 17)",
          "Per year": "Test Sharpe per year",
          "L-S grid": "Test Sharpe of the daily long-short rules, 2022-2024 (rows: long count, columns: short count)",
          "Walk-forward": "Walk-forward choices (best Sharpe over the earlier test years) and how they did"}

RESULTS.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    for name, df in sheets:
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)

wb = load_workbook(OUT)
for ws in wb.worksheets:
    hdr = 1
    if ws.title in titles:
        ws["A1"] = titles[ws.title]
        ws["A1"].font = Font(name=F, bold=True, size=13)
        hdr = 2
    for c in ws[hdr]:
        c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[hdr].height = 34
    headers = [c.value for c in ws[hdr]]
    sharpe_col = headers.index("sharpe") + 1 if "sharpe" in headers else None
    lo_col = headers.index("sharpe_ci_lo") + 1 if "sharpe_ci_lo" in headers else None
    for row in ws.iter_rows(min_row=hdr + 1):
        for c in row:
            c.font, c.border = Font(name=F), BORDER
            if isinstance(c.value, float):
                c.number_format = "0.00"
            if ws.title == "L-S grid" and isinstance(c.value, float):
                c.fill = GOOD if c.value > 0 else BAD
        if lo_col and isinstance(row[lo_col - 1].value, (int, float)) and row[lo_col - 1].value > 0:
            for c in row:
                c.fill = GOOD
        elif sharpe_col and isinstance(row[sharpe_col - 1].value, (int, float)) and row[sharpe_col - 1].value < 0:
            for c in row:
                c.fill = BAD
    for i, h in enumerate(headers, 1):
        width = max([len(str(h or ""))] + [len(f"{c.value:.2f}" if isinstance(c.value, float) else str(c.value or ""))
                                            for c in ws[get_column_letter(i)][hdr:]])
        ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 9), 60)
    ws.freeze_panes = ws.cell(row=hdr + 1, column=2)

# Per year: formula columns for the average and the number of positive years
ws = wb["Per year"]
a, b = get_column_letter(2), get_column_letter(1 + len(YEARS))
for col, label in ((2 + len(YEARS), "Average"), (3 + len(YEARS), "Years positive")):
    c = ws.cell(row=2, column=col, value=label)
    c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
    c.alignment = Alignment(wrap_text=True, horizontal="center")
    ws.column_dimensions[get_column_letter(col)].width = 11
for r in range(3, ws.max_row + 1):
    for col, f in ((2 + len(YEARS), f"=AVERAGE({a}{r}:{b}{r})"), (3 + len(YEARS), f'=COUNTIF({a}{r}:{b}{r},">0")')):
        c = ws.cell(row=r, column=col, value=f)
        c.font, c.border = Font(name=F), BORDER
        c.number_format = "0.00" if "AVERAGE" in f else "0"

ws = wb["Notes"]
ws.column_dimensions["A"].width = 150
for row in ws.iter_rows(min_row=2):
    for c in row:
        c.font, c.alignment = Font(name=F), Alignment(wrap_text=True)

wb.calculation.fullCalcOnLoad = True
wb.save(OUT)
print(f"Saved {OUT}")
print("Green: Sharpe 95% interval above zero (or positive Sharpe in the L-S grid); red: negative Sharpe.")
