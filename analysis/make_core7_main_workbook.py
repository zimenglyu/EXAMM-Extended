#!/usr/bin/env python3
"""MAIN WORKFLOW, step 3 (workbook). Renamed from the stale-model workbook script.
Excel workbook of the paper-rule results (core7_main_results.py): nothing is recomputed.

Reads results/core7_main.csv, _headline.csv and (if run with --per-set) _per_set.csv, and,
for comparison, the same files of the trade-every-day run (core7_main_everyday.csv, _headline.csv) if
present. Writes results/core7_main.xlsx:
  Summary        the paper's EXAMM numbers next to yours: retrain every year (the paper's design) and
                 never retrain, per test year and 2022-2024 joined (return, Sharpe, max drawdown)
  Joined         both books 2022-2024 joined, with 95% block-bootstrap intervals and PSR
  Grid           every fold's genomes on every test year computed (Sharpe, return, Rank IC, MSE, days
                 traded); leakage cells (if run with --all) marked
  Rule compared  the paper's rule (trade only when the predicted signs agree) vs trading every day
  Per set        V1-V4 separately, then averaged (paper's Table 7 style), if the per-set file exists
  Notes          method and caveats
Run on the Mac after core7_main_results.py (add --per-set there for the Per set sheet):
    cd ~/Financial_toolbox && python3 make_core7_main_workbook.py
Needs openpyxl (python3 -m pip install --user openpyxl if the import fails).
"""
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / "core7_main.xlsx"
YEARS = [2022, 2023, 2024]
# the paper's EXAMM column (Table 7: average net return across V1-V4; Table 8: model summary)
PAPER = {"2022": 15.46, "2023": 9.07, "2024": 8.52, "mean": 11.02, "ic": 0.0106, "sharpe": 0.74, "mdd": -10.76}
PAPER_EW = {"2022": -17.39, "2023": 3.99, "2024": -0.93, "mean": -4.78}

F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD, PAPER_FILL, LEAK = (PatternFill("solid", fgColor=c) for c in ("E2EFDA", "FBE5E5", "DDEBF7", "FFF2CC"))
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)


def need(name):
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} -- run: python3 core7_main_results.py")
    return pd.read_csv(path)


cells = need("core7_main.csv")
head = need("core7_main_headline.csv").set_index("book")
per_set_path = RESULTS / "core7_main_per_set.csv"
per_set = pd.read_csv(per_set_path) if per_set_path.exists() else None
old_cells = pd.read_csv(RESULTS / "core7_main_everyday.csv") if (RESULTS / "core7_main_everyday.csv").exists() else None
old_head = (pd.read_csv(RESULTS / "core7_main_everyday_headline.csv").set_index("book")
            if (RESULTS / "core7_main_everyday_headline.csv").exists() else None)


def cell(df, fold, year, col):
    r = df[(df["genomes_from_fold"] == fold) & (df["test_year"] == year)]
    return float(r[col].iloc[0]) if len(r) else None


RETRAIN = "retrain every year (standard, the diagonal)"
NEVER = next((b for b in head.index if b.startswith("never retrain")), None)

# ---------------------------------------------------------------- Summary ----
rows = [{"Strategy": "Paper: EXAMM (Tables 7-8)", **{f"{y} return (%)": PAPER[str(y)] for y in YEARS},
         "Mean / joined return (%)": PAPER["mean"], "Sharpe": PAPER["sharpe"], "Max drawdown (%)": PAPER["mdd"],
         "Rank IC": PAPER["ic"]},
        {"Strategy": "Paper: equal weight", **{f"{y} return (%)": PAPER_EW[str(y)] for y in YEARS},
         "Mean / joined return (%)": PAPER_EW["mean"]}]
for name, label, pairs in ((RETRAIN, "Yours: retrain every year (paper's design)", [(y, y) for y in YEARS]),
                           (NEVER, "Yours: never retrain (2022-fold models)", [(YEARS[0], y) for y in YEARS])):
    if name is None or name not in head.index:
        continue
    ics = [cell(cells, f, y, "test_rank_ic") for f, y in pairs]
    rows.append({"Strategy": label, **{f"{y} return (%)": cell(cells, f, y, "ann_return_%") for f, y in pairs},
                 "Mean / joined return (%)": head.loc[name, "ann_return_%"], "Sharpe": head.loc[name, "sharpe"],
                 "Max drawdown (%)": -head.loc[name, "max_dd_%"],
                 "Rank IC": sum(ics) / len(ics) if all(v is not None for v in ics) else None})
summary = pd.DataFrame(rows)

# ---------------------------------------------------------------- Joined ----
joined = head.reset_index()[[c for c in ["book", "ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%",
                                         "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "psr", "max_dd_%"]
                             if c in head.reset_index()]]
joined.columns = ["Book (fixed in advance)", "Annual return (%)", "Return 95% low", "Return 95% high", "Volatility (%)",
                  "Sharpe", "Sharpe 95% low", "Sharpe 95% high", "PSR", "Max drawdown (%)"][:len(joined.columns)]

# ---------------------------------------------------------------- Grid ----
grid_rows = []
for col, label in (("sharpe", "Sharpe"), ("ann_return_%", "Return (%)"), ("test_rank_ic", "Rank IC"),
                   ("test_mse", "MSE (raw daily returns)"), ("days_traded_%", "Days traded (%)")):
    if col not in cells:
        continue
    for f in YEARS:
        row = {"Measure": label, "Genomes from fold": f"{f} (train <= {f - 2}, val {f - 1})"}
        for y in YEARS:
            v = cell(cells, f, y, col)
            row[f"Test {y}"] = v
        grid_rows.append(row)
grid = pd.DataFrame(grid_rows)

# ---------------------------------------------------------------- Rule compared ----
compare = None
if old_cells is not None:
    crow = []
    for f, y in [(y, y) for y in YEARS] + [(YEARS[0], y) for y in YEARS[1:]]:
        crow.append({"Genomes from fold": f, "Test year": y,
                     "Trade every day: return (%)": cell(old_cells, f, y, "ann_return_%"),
                     "Trade every day: Sharpe": cell(old_cells, f, y, "sharpe"),
                     "Paper rule: return (%)": cell(cells, f, y, "ann_return_%"),
                     "Paper rule: Sharpe": cell(cells, f, y, "sharpe"),
                     "Paper rule: days traded (%)": cell(cells, f, y, "days_traded_%") if "days_traded_%" in cells else None})
    for name in (RETRAIN, NEVER):
        if old_head is not None and name in old_head.index and name in head.index:
            crow.append({"Genomes from fold": name, "Test year": "2022-2024",
                         "Trade every day: return (%)": old_head.loc[name, "ann_return_%"],
                         "Trade every day: Sharpe": old_head.loc[name, "sharpe"],
                         "Paper rule: return (%)": head.loc[name, "ann_return_%"],
                         "Paper rule: Sharpe": head.loc[name, "sharpe"]})
    compare = pd.DataFrame(crow)

# ---------------------------------------------------------------- Per set ----
set_table = None
if per_set is not None:
    vname = {s: f"V{i + 1}" for i, s in enumerate(sorted(per_set["set"].unique()))}
    blocks = []
    for label, pairs in (("Retrain every year", [(y, y) for y in YEARS]),
                         ("Never retrain (2022-fold models)", [(YEARS[0], y) for y in YEARS])):
        sel = per_set[[(f, y) in pairs for f, y in zip(per_set["genomes_from_fold"], per_set["test_year"])]]
        if sel.empty:
            continue
        g = sel.pivot(index="test_year", columns="set", values="ann_return_%")
        g.columns = [vname[c] for c in g.columns]
        g["Average V1-V4"] = g.mean(axis=1)
        g.loc["Mean"] = g.mean(axis=0)
        g = g.reset_index().rename(columns={"test_year": "Test year"})
        g.insert(0, "Design", label)
        blocks.append(g)
    if blocks:
        paper_row = pd.DataFrame([{"Design": "Paper: EXAMM", "Test year": y, "Average V1-V4": PAPER[str(y)]} for y in YEARS]
                                 + [{"Design": "Paper: EXAMM", "Test year": "Mean", "Average V1-V4": PAPER["mean"]}])
        set_table = pd.concat(blocks + [paper_row], ignore_index=True)

notes = [
    "Model: Pooled EXAMM, RET target, the 7 core7 inputs; per test year Y the fold train <= Y-2 / validation Y-1; each "
    "panel set's genome = the best of 10 runs by validation MSE (no Pareto frontier).",
    "Trading: the paper's Algorithm 1 (paper_long_short.py): long the 5 highest and short the 5 lowest forecasts, each "
    "side sized current capital / 5, rebalanced ONLY when the 5th-highest forecast is > 0 and the 5th-lowest < 0 "
    "(otherwise the positions are kept). Forecasts converted to raw daily returns first (x IQR + median). Costs on.",
    "The 4 panel sets are combined with equal weight into one portfolio (Summary, Joined, Grid); the Per set sheet "
    "treats V1-V4 separately and averages them, as the paper's Table 7.",
    "Retrain every year = each year's own models (the paper's design). Never retrain = the 2022-fold models (trained "
    "<= 2020, validated on 2021) for 2022-2024. Both are free of leakage.",
    "Grid: cells where a fold's models are scored on an EARLIER year are leakage (only computed with --all) and are "
    "highlighted; they are not test results.",
    "Intervals: 95% stationary block bootstrap; PSR = P(true Sharpe > 0). Paper numbers are from its Tables 7-8.",
    "Caveats: three test years only; wide intervals; the remaining 2024 gap to the paper is most likely the genome "
    "(the paper's own EXAMM runs); short-sale costs follow this repository's stock.py (the paper's version may differ).",
    "Source: core7_main*.csv (core7_main_results.py) and core7_main_everyday*.csv.",
]

sheets = [("Summary", summary, "Paper rule: your EXAMM results next to the paper's (test 2022-2024, costs on)"),
          ("Joined", joined, "Both designs, 2022-2024 joined, with 95% intervals"),
          ("Grid", grid, "Every fold's models on every test year computed (4 sets combined)")]
if compare is not None:
    sheets.append(("Rule compared", compare, "Paper rule (trade only when signs agree) vs trading every day"))
if set_table is not None:
    sheets.append(("Per set", set_table, "Net return (%) per panel set V1-V4 and their average (paper's Table 7 style)"))

RESULTS.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    for name, df, _ in sheets:
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)

wb = load_workbook(OUT)
titles = {name: title for name, _, title in sheets}
for ws in wb.worksheets:
    hdr = 2 if ws.title in titles else 1
    if ws.title in titles:
        ws["A1"] = titles[ws.title]
        ws["A1"].font = Font(name=F, bold=True, size=13)
    headers = [c.value for c in ws[hdr]]
    for c in ws[hdr]:
        c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[hdr].height = 36
    for row in ws.iter_rows(min_row=hdr + 1):
        first = str(row[0].value or "")
        for c in row:
            h = str(headers[c.column - 1])
            c.font, c.border = Font(name=F), BORDER
            if ws.title == "Notes":
                c.alignment = Alignment(wrap_text=True)
            if isinstance(c.value, float):
                c.number_format = ("0.000E+00" if "MSE" in first or "MSE" in h else "+0.0000;-0.0000" if "IC" in first or "IC" in h
                                   else "0.00")
        if first.startswith("Paper"):
            for c in row:
                c.fill = PAPER_FILL
        vals = {h: row[i].value for i, h in enumerate(headers)}
        if ws.title == "Joined":
            lo = vals.get("Sharpe 95% low")
            if isinstance(lo, (int, float)) and lo > 0:
                for c in row:
                    c.fill = GOOD
        if ws.title == "Grid":
            fold = int(str(row[1].value)[:4]) if row[1].value else None
            for i, h in enumerate(headers):
                if str(h).startswith("Test ") and fold and int(str(h)[5:]) < fold and row[i].value is not None:
                    row[i].fill = LEAK
    for i, h in enumerate(headers, 1):
        width = max([len(str(h or ""))] + [len(f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value or ""))
                                            for c in ws[get_column_letter(i)][hdr:]])
        ws.column_dimensions[get_column_letter(i)].width = 140 if ws.title == "Notes" else min(max(width + 2, 10), 48)
    if ws.title != "Notes":
        ws.freeze_panes = ws.cell(row=hdr + 1, column=2)
wb.save(OUT)
print(f"Saved {OUT}")
print("Blue rows: the paper's numbers. Green: Sharpe interval above zero. Yellow (Grid): leakage cells.")
