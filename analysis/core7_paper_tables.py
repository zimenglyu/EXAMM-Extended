#!/usr/bin/env python3
"""Paper-style Excel workbook for the selection-year RET experiment (test 2016-2020), with charts.

Tables in the paper's layout (Table 1/2 = walk-forward folds; Table 3 = Rank IC, MSE and Net (%) per
test year for each construction), here comparing the two genome choices of the Pooled model:
  1 genome   = MAIN: one genome per fold shared by all 4 sets (core7_frontier_sel_single.py)
  4 genomes  = pre-planned check: one genome per panel set and fold (core7_frontier_sel.py)
plus the 2022-2024 RET table (core7_table.py output) and charts. Nothing is re-traded: the numbers come
from the CSVs the runs saved; only the single genome's test Rank IC / MSE on the other sets is scored
here from its saved predictions (same method as core7_sel_ic.py).

Run on the Mac after both runs:
    cd ~/Financial_toolbox && python3 core7_paper_tables.py
Writes results/core7_paper.xlsx. Needs openpyxl (python3 -m pip install --user openpyxl).
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.chart import BarChart, LineChart, Reference, ScatterChart, Series
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
EXAMM = Path.home() / "EXAMM-Extended"
OUT = RESULTS / "core7_paper.xlsx"
YEARS = [2016, 2017, 2018, 2019, 2020]
SETS = ["set1", "set2", "set3", "set4"]
WARMUP = 40                      # unscored warm-up rows before the selection and test years
PRIMARY, NAMED, PROTO = "L/S 5/5", "overlap 5/5 hold 10", "PROTOCOL top-3 book"
MAIN, CHECK = "Pooled, 1 shared genome (main)", "Pooled, 4 genomes per set (pre-planned check)"
CONSTRUCTIONS = {MAIN: "core7_frontier_sel_single", CHECK: "core7_frontier_sel"}   # main construction first

F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)


def need(path):
    if not path.exists():
        raise SystemExit(f"missing {path} -- run the corresponding script first")
    return pd.read_csv(path)


# ---------------- Table 1: folds ----------------
fold_rows = []
for y in YEARS:
    info = {s: pd.read_csv(EXAMM / "datasets" / f"panels_core7_examm_sel_{y}" / s / "split_info.csv").set_index("split")
            for s in SETS}

    def rows(split, scored=False):
        n = [int(i.loc[split, "rows"]) - (WARMUP if scored else 0) for i in info.values()]
        return f"{min(n):,}" if min(n) == max(n) else f"{min(n):,}-{max(n):,}"

    tr = info["set1"].loc["train"]
    fold_rows.append({"Fold": f"Test {y}", "Training rows": rows("train"),
                      "Training period": f"{tr['first_date']} to {tr['last_date']}",
                      "EXAMM validation rows": rows("val"), "EXAMM validation period": str(y - 2),
                      "Selection rows": rows("sel", True), "Selection period": str(y - 1),
                      "Test rows": rows("test", True), "Test period": str(y)})
folds = pd.DataFrame(fold_rows)

# ---------------- Rank IC / MSE per test year ----------------
PANELS = EXAMM / "datasets" / "panels_core7"
warnings.filterwarnings("ignore", message="Mean of empty slice")   # a genome with no predictions on a set
_panel = {}


def panel(s):
    if s not in _panel:
        _panel[s] = pd.read_csv(PANELS / f"{s}_core7" / "panel_dates.csv").set_index("row")
    return _panel[s]


def score(y, target, folder):
    """Test Rank IC and MSE (raw returns) of one genome's predictions on one target set."""
    files = sorted(folder.glob("*_test_predictions.csv"))
    if len(files) != 50:
        return np.nan, np.nan
    data = EXAMM / "datasets" / f"panels_core7_examm_sel_{y}" / target
    first = int(pd.read_csv(data / "split_info.csv").set_index("split").loc["test", "first_row"])
    st = json.load(open(data / "ret_scaling.json"))
    P = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        P[f.name[:-len("_test_predictions.csv")]] = df["predicted_RET"].to_numpy()
    P = pd.DataFrame(P) * st["iqr"] + st["median"]
    rows_ = panel(target).loc[first + np.arange(len(P)) + 1]
    A = pd.DataFrame({t: rows_[f"RET_raw_{t}"].to_numpy() for t in P.columns})
    keep = (rows_["date"] >= f"{y}-01-01").to_numpy()
    ic = P[keep].rank(axis=1).corrwith(A[keep].rank(axis=1), axis=1).dropna()
    return ic.mean(), float(((P[keep] - A[keep]) ** 2).to_numpy().mean())


scores = need(EXAMM / "test_output" / "core7_sel_scores.csv")
scores["repeat"] = scores["repeat"].astype(str)
ic_mse = {}                                   # (construction, year) -> (Rank IC, MSE), mean of the 4 sets
for y in YEARS:
    chosen = pd.read_csv(EXAMM / "test_output" / f"core7_examm_ret_sel_{y}" / "selected_genomes.csv",
                         dtype={"repeat": str})
    m = scores[scores["fold"] == y].merge(chosen[["set", "repeat"]], on=["set", "repeat"])
    ic_mse[(CHECK, y)] = (m["test_IC"].mean(), m["test_MSE"].mean())
    single = EXAMM / "test_output" / f"core7_examm_ret_sel_{y}" / "selected_single_genome.csv"
    if single.exists():
        row = pd.read_csv(single, dtype={"repeat": str}).iloc[0]
        base = EXAMM / "test_output" / f"core7_examm_ret_sel_{y}" / row["source_set"] / row["repeat"]
        per = [score(y, t, base / ("evaluation_test" if t == row["source_set"] else f"evaluation_test_on_{t}"))
               for t in SETS]
        ic_mse[(MAIN, y)] = (np.nanmean([p[0] for p in per]),
                                                             np.nanmean([p[1] for p in per]))

by_year = {c: need(RESULTS / f"{tag}_by_year.csv") for c, tag in CONSTRUCTIONS.items()}
rules = {c: need(RESULTS / f"{tag}.csv") for c, tag in CONSTRUCTIONS.items()}


def yearly(c, rule, col):
    t = by_year[c]
    t = t[(t["rule"] == rule) & (t["split"] == "test")].set_index("year")
    return [float(t.loc[y, col]) if y in t.index else np.nan for y in YEARS]


# Table 3 layout: one row per construction, three columns per test year
t3_rows, t3_ext = [], []
for c in CONSTRUCTIONS:
    net = yearly(c, PRIMARY, "total_%")
    row, ext = {"Construction": c}, {"Construction": c}
    for i, y in enumerate(YEARS):
        ic, mse = ic_mse.get((c, y), (np.nan, np.nan))
        row[f"{y} Rank IC"], row[f"{y} MSE"], row[f"{y} Net (%)"] = ic, mse, net[i]
        ext[f"{y} Net (%) {NAMED}"] = yearly(c, NAMED, "total_%")[i]
        ext[f"{y} Net (%) {PROTO}"] = yearly(c, PROTO, "total_%")[i]
    t3_rows.append(row)
    t3_ext.append(ext)
table3 = pd.DataFrame(t3_rows)
table3b = pd.DataFrame(t3_ext)

# 2022-2024 RET table (core7_table.py output, RET target, global best genome by validation MSE)
t22 = pd.DataFrame([
    ["Pooled, set1", 0.0011, 5.374e-4, -27.56, -0.98, 36.00, -0.0030, 3.352e-4, -11.81, -0.53, 19.57, -0.0202, 3.555e-4, -23.46, -1.06, 24.87],
    ["Pooled, set2", 0.0107, 5.504e-4, -24.80, -0.90, 46.19, -0.0141, 3.599e-4, -25.43, -1.53, 26.22, 0.0256, 3.012e-4, 20.04, 1.09, 13.48],
    ["Pooled, set3", 0.0087, 5.595e-4, -1.49, 0.11, 17.40, -0.0043, 3.345e-4, -25.18, -1.43, 28.80, 0.0155, 3.249e-4, 13.46, 0.74, 12.58],
    ["Pooled, set4", 0.0197, 4.301e-4, -4.17, -0.05, 22.42, 0.0156, 2.562e-4, -1.60, -0.02, 18.58, 0.0015, 2.307e-4, -13.70, -1.07, 20.32],
    ["Pooled, mean", 0.0100, 5.193e-4, -14.51, -0.45, 30.50, -0.0015, 3.214e-4, -16.00, -0.88, 23.29, 0.0056, 3.031e-4, -0.92, -0.08, 17.81],
], columns=["Construction"] + [f"{y} {k}" for y in (2022, 2023, 2024)
                               for k in ("Rank IC", "MSE (raw)", "Net (%)", "Sharpe", "MDD (%)")])

# rule-by-rule comparison and the charts' data
c1, c4 = MAIN, CHECK
r4, r1 = rules[c4].set_index("rule"), rules[c1].set_index("rule")
compare = pd.DataFrame({"1 genome (main): test Sharpe": r1["test_sharpe"], "4 genomes: test Sharpe": r4["test_sharpe"],
                        "1 genome (main): test return (%)": r1["test_ann_return_%"],
                        "4 genomes: test return (%)": r4["test_ann_return_%"],
                        "1 genome (main): turnover (x/yr)": r1["test_turnover_x"],
                        "4 genomes: turnover (x/yr)": r4["test_turnover_x"]})
compare = compare.sort_values("1 genome (main): turnover (x/yr)").reset_index().rename(columns={"index": "rule"})
sharpe_year = pd.DataFrame({"Year": [str(y) for y in YEARS],
                            **{f"{lab} ({short})": yearly(c, rule, "sharpe")
                               for c, short in ((c1, "1 genome, main"), (c4, "4 genomes"))
                               for rule, lab in ((PRIMARY, "Daily L/S 5/5"), (NAMED, "Overlap 5/5 hold 10"),
                                                 (PROTO, "Protocol top-3"))},
                            "Equal weight": yearly(c1, "equal weight", "sharpe")})



# ---------------- Pareto frontiers made from the results (descriptive) ----------------
def pareto(ret, risk):
    """True where no other rule has both a higher (or equal) return and a lower (or equal) risk."""
    ok = ~(np.isnan(ret) | np.isnan(risk))
    out = np.zeros(len(ret), dtype=bool)
    for i in np.where(ok)[0]:
        out[i] = not any(ok[j] and ret[j] >= ret[i] and risk[j] <= risk[i] and (ret[j] > ret[i] or risk[j] < risk[i])
                         for j in range(len(ret)) if j != i)
    return out


def frontier_table(df):
    t = pd.DataFrame({"rule": df["rule"], "family": df.get("family", ""),
                      "Test return (%)": df["test_ann_return_%"], "Test volatility (%)": df["test_ann_vol_%"],
                      "Turnover (x/yr)": df["test_turnover_x"], "Test Sharpe": df["test_sharpe"]})
    r = t["Test return (%)"].to_numpy(float)
    t["On return/volatility frontier"] = np.where(pareto(r, t["Test volatility (%)"].to_numpy(float)), "Yes", "")
    t["On return/turnover frontier"] = np.where(pareto(r, t["Turnover (x/yr)"].to_numpy(float)), "Yes", "")
    return t.sort_values("Turnover (x/yr)").reset_index(drop=True)


frontiers = [("Frontier 2016-20 (1 genome)", frontier_table(rules[c1]),
              "Pareto frontier of the 38 rules, test 2016-2020, 1 shared genome (MAIN). Drawn on test: descriptive"),
             ("Frontier 2016-20 (4 genomes)", frontier_table(rules[c4]),
              "Pareto frontier of the 38 rules, test 2016-2020, 4 genomes per set (pre-planned check). Descriptive")]
paper_folds = RESULTS / "core7_frontier_turnover.csv"
if paper_folds.exists():
    frontiers.append(("Frontier 2022-24 (paper)", frontier_table(pd.read_csv(paper_folds)),
                      "Pareto frontier of the 38 rules, test 2022-2024 (paper's folds, genome by validation MSE)"))

notes = [
    "Paper-style tables for the Pooled EXAMM RET experiment. MAIN construction (listed first): ONE genome per test "
    "year shared by all 4 panel sets, chosen from the 40 candidates (4 sets x 10 runs) by the highest mean Rank IC on "
    "the held-out selection year. Second construction: one genome per panel set (per-set selection-year Rank IC), the "
    "evaluation planned before 2016-2020 was examined; the single-genome design was applied to the same years "
    "afterwards, and both reach the same conclusion. Both choose genomes and rules on the selection year only.",
    "Folds: train <= Y-3, EXAMM validation Y-2, selection Y-1 (never given to EXAMM), test Y. Rows of the selection "
    f"and test years exclude the {WARMUP}-day warm-up (scored days only). Training rows vary by panel set.",
    "Rank IC = mean daily cross-sectional Spearman correlation of forecast and next-day raw return; MSE in raw "
    "daily-return units; both averaged over the 4 panel sets. Net (%) = test-year return of daily long-short 5/5 "
    "(the primary rule), costs on, 4 sets combined (also given for overlap 5/5 hold 10 and the protocol book).",
    "Table 3 (2022-2024) is the earlier core7_table.py RET output (genome by lowest validation MSE).",
    "Charts are built from these tables in the paper's style; the paper's own figures were not available here.",
    "Frontier sheets: a rule is on the frontier when no other rule has both a higher test return and a lower "
    "volatility (or turnover). They are drawn on the test results, so they DESCRIBE the trade-off; they do not "
    "choose a rule (that would use the test data). Rules whose account fell below zero have no return and are left "
    "off. The turnover axis of the charts is logarithmic (about 30x to 1,000x a year).",
]

# ---------------- write ----------------
sheets = [("Table 1 Folds", folds, "Table 1: Walk-forward folds of the selection-year experiment (core7 panels)"),
          ("Table 3 2016-2020", table3, "Table 3: Pooled EXAMM (RET) on the core7 panels, test 2016-2020"),
          ("Table 3b Net by rule", table3b, "Table 3b: Net (%) per test year for the 10-day rule and the protocol book"),
          ("Table 3 2022-2024", t22, "RET results 2022-2024 (core7_table.py, genome by validation MSE)"),
          ("Sharpe by year", sharpe_year, "Test Sharpe per year"),
          ("Rules compared", compare, "Every rule: test Sharpe, return and turnover, 1 genome (main) vs 4 genomes (2016-2020)")]
sheets += frontiers
RESULTS.mkdir(exist_ok=True)
with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
    for name, df, _ in sheets:
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)

wb = load_workbook(OUT)
for name, df, title in sheets:
    ws = wb[name]
    ws["A1"], ws["A1"].font = title, Font(name=F, bold=True, size=13)
    for c in ws[2]:
        c.font, c.fill, c.border = HDR_FONT, HDR_FILL, BORDER
        c.alignment = Alignment(wrap_text=True, vertical="center", horizontal="center")
    ws.row_dimensions[2].height = 40
    for row in ws.iter_rows(min_row=3):
        for c in row:
            c.font, c.border = Font(name=F), BORDER
            head = str(ws.cell(row=2, column=c.column).value)
            if isinstance(c.value, float):
                c.number_format = ("+0.0000;-0.0000" if "Rank IC" in head else "0.000E+00" if "MSE" in head
                                   else "+0.00;-0.00" if "Net" in head or "Sharpe" in head else "0.00")
    for i in range(1, ws.max_column + 1):
        width = max(len(str(ws.cell(row=r, column=i).value or "")) for r in range(2, ws.max_row + 1))
        ws.column_dimensions[get_column_letter(i)].width = min(max(width + 2, 10), 42)
    ws.freeze_panes = "B3"

# chart: Sharpe by year (line)
ws = wb["Sharpe by year"]
ch = LineChart()
ch.title, ch.y_axis.title, ch.x_axis.title = "Test Sharpe by year", "Sharpe", "Test year"
ch.add_data(Reference(ws, min_col=2, max_col=ws.max_column, min_row=2, max_row=2 + len(YEARS)), titles_from_data=True)
ch.set_categories(Reference(ws, min_col=1, min_row=3, max_row=2 + len(YEARS)))
ch.height, ch.width = 10, 22
ws.add_chart(ch, f"A{len(YEARS) + 5}")

# chart: Sharpe per rule, 1 genome (main) vs 4 genomes (bars), rules ordered by turnover
ws = wb["Rules compared"]
n = ws.max_row
bar = BarChart()
bar.type, bar.title = "bar", "Test Sharpe per rule (ordered by turnover, low at top)"
bar.y_axis.title = "Sharpe"
bar.add_data(Reference(ws, min_col=2, max_col=3, min_row=2, max_row=n), titles_from_data=True)
bar.set_categories(Reference(ws, min_col=1, min_row=3, max_row=n))
bar.height, bar.width = 22, 18
ws.add_chart(bar, "I3")

# chart: Sharpe vs turnover (scatter)
sc = ScatterChart()
sc.title, sc.style = "Test Sharpe vs annual turnover", 13
sc.x_axis.title, sc.y_axis.title = "Annual turnover (x account value)", "Test Sharpe"
for ycol, xcol, label in ((2, 6, "1 genome (main)"), (3, 7, "4 genomes")):
    s = Series(Reference(ws, min_col=ycol, min_row=3, max_row=n), Reference(ws, min_col=xcol, min_row=3, max_row=n),
               title=label)
    s.marker.symbol, s.graphicalProperties.line.noFill = "circle", True
    sc.series.append(s)
sc.height, sc.width = 10, 18
ws.add_chart(sc, "T3")

# chart: Table 3 Net (%) per year (bars)
ws = wb["Table 3 2016-2020"]
data_ws = wb.create_sheet("Chart data")
data_ws.append(["Year"] + list(CONSTRUCTIONS))
for i, y in enumerate(YEARS):
    data_ws.append([str(y)] + [table3.loc[k, f"{y} Net (%)"] for k in range(len(table3))])
b2 = BarChart()
b2.title, b2.y_axis.title = "Net (%) of daily L/S 5/5 per test year", "Net (%)"
b2.add_data(Reference(data_ws, min_col=2, max_col=3, min_row=1, max_row=1 + len(YEARS)), titles_from_data=True)
b2.set_categories(Reference(data_ws, min_col=1, min_row=2, max_row=1 + len(YEARS)))
b2.height, b2.width = 9, 18
ws.add_chart(b2, "A7")

for name, t, _ in frontiers:
    ws = wb[name]
    n = len(t) + 2                                   # data rows 3..n
    col0 = len(t.columns) + 2                        # helper columns for the frontier lines, right of the table
    for k, (risk, flag, label) in enumerate((("Turnover (x/yr)", "On return/turnover frontier", "turnover"),
                                             ("Test volatility (%)", "On return/volatility frontier", "volatility"))):
        f = t[t[flag] == "Yes"].sort_values(risk)
        c = col0 + 3 * k
        ws.cell(row=2, column=c, value=f"Frontier ({label}): {risk}")
        ws.cell(row=2, column=c + 1, value=f"Frontier ({label}): return (%)")
        for i, (x, y) in enumerate(zip(f[risk], f["Test return (%)"]), start=3):
            ws.cell(row=i, column=c, value=float(x))
            ws.cell(row=i, column=c + 1, value=float(y))
        for cc in (c, c + 1):
            hc = ws.cell(row=2, column=cc)
            hc.font, hc.fill, hc.border = HDR_FONT, HDR_FILL, BORDER
            hc.alignment = Alignment(wrap_text=True, horizontal="center")
            ws.column_dimensions[get_column_letter(cc)].width = 14
        xcol = list(t.columns).index(risk) + 1
        ycol = list(t.columns).index("Test return (%)") + 1
        sc = ScatterChart()
        sc.title = f"Test return vs {label}" + (" (log scale)" if label == "turnover" else "")
        sc.style = 13
        sc.x_axis.title = risk
        sc.y_axis.title = "Annualized test return (%)"
        if label == "turnover":
            sc.x_axis.scaling.logBase = 10
        dots = Series(Reference(ws, min_col=ycol, min_row=3, max_row=n),
                      Reference(ws, min_col=xcol, min_row=3, max_row=n), title="all rules")
        dots.marker.symbol, dots.graphicalProperties.line.noFill = "circle", True
        sc.series.append(dots)
        if len(f):
            line = Series(Reference(ws, min_col=c + 1, min_row=3, max_row=2 + len(f)),
                          Reference(ws, min_col=c, min_row=3, max_row=2 + len(f)), title="Pareto frontier")
            line.marker.symbol = "diamond"
            sc.series.append(line)
        sc.height, sc.width = 10, 17
        ws.add_chart(sc, f"{get_column_letter(col0 + 7)}{3 + 22 * k}")

ws = wb["Notes"]
ws.column_dimensions["A"].width = 140
for row in ws.iter_rows():
    for c in row:
        c.font, c.alignment = Font(name=F, bold=(c.row == 1)), Alignment(wrap_text=True)
wb.save(OUT)
print(f"Saved {OUT}")
