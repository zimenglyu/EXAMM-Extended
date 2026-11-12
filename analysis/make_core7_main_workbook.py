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
  Table 5        every panel (V1-V4) and test year next to the paper's Table 5 (B&H, EW, EXAMM), with a bar chart
  Pareto         (if core7_main_results.py was run with --pareto) lowest val MSE vs Pareto choice vs highest val
                 return, for retrain every year and never retrain, with the frontier figure
  Pareto repeats every repeat's validation and test scores, frontier membership and labels
  Notes          method and caveats
--k 5: the 5/5 results (core7_main_results.py --k 5) -> results/core7_main_k5.xlsx.
Run on the Mac after core7_main_results.py (add --per-set there for the Per set sheet):
    cd ~/Financial_toolbox && python3 make_core7_main_workbook.py
Needs openpyxl (python3 -m pip install --user openpyxl if the import fails).
"""
import math
import sys
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

RESULTS = Path(__file__).resolve().parent / "results"
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv[:-1] else 10   # book size of the results read
TAG = "" if K == 10 else f"_k{K}"          # default 10/10 (the paper); --k 5 reads core7_main_k5*.csv
OUT = RESULTS / f"core7_main{TAG}.xlsx"
YEARS = [2022, 2023, 2024]
# the paper's EXAMM column (Table 7: average net return across V1-V4; Table 8: model summary)
PAPER = {"2022": 15.46, "2023": 9.07, "2024": 8.52, "mean": 11.02, "ic": 0.0106, "sharpe": 0.74, "mdd": -10.76}
PAPER_EW = {"2022": -17.39, "2023": 3.99, "2024": -0.93, "mean": -4.78}
# the paper's Table 5 (net return % per panel and test year): B&H, EW, EXAMM
PAPER_T5 = {("V1", 2022): (-7.16, -14.26, -4.63), ("V1", 2023): (5.18, -1.67, 21.34), ("V1", 2024): (-4.73, -9.92, 17.19),
            ("V2", 2022): (-11.42, -18.91, 35.22), ("V2", 2023): (15.37, 7.40, 16.92), ("V2", 2024): (12.13, 3.46, 13.80),
            ("V3", 2022): (-11.39, -19.08, 36.68), ("V3", 2023): (14.77, 7.05, -3.82), ("V3", 2024): (13.11, 3.89, -2.88),
            ("V4", 2022): (-10.25, -17.33, -5.43), ("V4", 2023): (9.41, 3.16, 1.84), ("V4", 2024): (6.69, -1.14, 5.96)}

F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD, PAPER_FILL, LEAK = (PatternFill("solid", fgColor=c) for c in ("E2EFDA", "FBE5E5", "DDEBF7", "FFF2CC"))
MEAN_FILL = PatternFill("solid", fgColor="FFF8E1")      # your mean-over-seeds rows (the main comparison)
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)


def need(name):
    path = RESULTS / name
    if not path.exists():
        raise SystemExit(f"missing {path} -- run: python3 core7_main_results.py")
    return pd.read_csv(path)


cells = need(f"core7_main{TAG}.csv")
head = need(f"core7_main{TAG}_headline.csv").set_index("book")
per_set_path = RESULTS / f"core7_main{TAG}_per_set.csv"
per_set = pd.read_csv(per_set_path) if per_set_path.exists() else None
bench = pd.read_csv(RESULTS / f"core7_main{TAG}_benchmarks.csv") if (RESULTS / f"core7_main{TAG}_benchmarks.csv").exists() else None
table8 = pd.read_csv(RESULTS / f"core7_main{TAG}_table8.csv") if (RESULTS / f"core7_main{TAG}_table8.csv").exists() else None
sizes = pd.read_csv(RESULTS / f"core7_main{TAG}_model_size.csv") if (RESULTS / f"core7_main{TAG}_model_size.csv").exists() else None
# yearly returns as the paper reports them: total return over the test year ((C - C0) / C0); older result
# files only have the annualized return
RET_COL = "total_return_%" if "total_return_%" in cells else "ann_return_%"
SET_RET_COL = "total_return_%" if per_set is not None and "total_return_%" in per_set else "ann_return_%"
old_cells = pd.read_csv(RESULTS / f"core7_main_everyday{TAG}.csv") if (RESULTS / f"core7_main_everyday{TAG}.csv").exists() else None
old_head = (pd.read_csv(RESULTS / f"core7_main_everyday{TAG}_headline.csv").set_index("book")
            if (RESULTS / f"core7_main_everyday{TAG}_headline.csv").exists() else None)


def cell(df, fold, year, col):
    r = df[(df["genomes_from_fold"] == fold) & (df["test_year"] == year)]
    return float(r[col].iloc[0]) if len(r) else None


RETRAIN = "retrain every year (standard, the diagonal)"
NEVER = next((b for b in head.index if b.startswith("never retrain")), None)

# ---------------------------------------------------------------- mean over the 10 seeds (the paper's reporting) ----
# core7_main_results.py writes these by default; the comparison sheets below use them FIRST (the paper reports the
# mean over its 10 seeds) and show the single best seed by validation MSE only as a second, labelled row.
_r7p, _r8p = RESULTS / f"core7_main{TAG}_repeats_table7.csv", RESULTS / f"core7_main{TAG}_repeats_table8.csv"
REP7 = pd.read_csv(_r7p, dtype={"test_year": str}) if _r7p.exists() else None
REP8 = pd.read_csv(_r8p) if _r8p.exists() else None
if REP7 is None or REP8 is None:
    print(f"!! {_r7p.name} / {_r8p.name} missing -- rerun core7_main_results.py without --no-mean-repeats; "
          "the sheets show the best seed only")
MEAN_DESIGNS = (("retrain every year", "Yours: retrain every year, MEAN OF 10 SEEDS (paper's reporting)"),
                ("never retrain (fold 2022 models)", "Yours: never retrain, MEAN OF 10 SEEDS"))


def rep7(design, year, col="Average V1-V4"):
    """Mean-over-seeds net return (%) of one design, test year (or 'Mean') and panel column."""
    if REP7 is None:
        return None
    r = REP7[(REP7["design"] == design) & (REP7["test_year"].astype(str) == str(year))]
    return float(r[col].iloc[0]) if len(r) and col in r and pd.notna(r[col].iloc[0]) else None


def rep8(design, col, panel="Average V1-V4"):
    if REP8 is None:
        return None
    r = REP8[(REP8["design"] == design) & (REP8["panel"] == panel)]
    return float(r[col].iloc[0]) if len(r) and col in r else None


# ---------------------------------------------------------------- Summary ----
rows = [{"Strategy": "Paper: EXAMM (Tables 7-8)", **{f"{y} return (%)": PAPER[str(y)] for y in YEARS},
         "Mean / joined return (%)": PAPER["mean"], "Sharpe": PAPER["sharpe"], "Max drawdown (%)": PAPER["mdd"],
         "IC (Pearson)": PAPER["ic"]},
        {"Strategy": "Paper: equal weight", **{f"{y} return (%)": PAPER_EW[str(y)] for y in YEARS},
         "Mean / joined return (%)": PAPER_EW["mean"]}]
for design, label in MEAN_DESIGNS:
    if rep7(design, "Mean") is None:
        continue
    rows.append({"Strategy": label, **{f"{y} return (%)": rep7(design, y) for y in YEARS},
                 "Mean / joined return (%)": rep7(design, "Mean"), "Sharpe": rep8(design, "Sharpe (2022-24)"),
                 "Max drawdown (%)": rep8(design, "MDD (%)"), "IC (Pearson)": rep8(design, "IC (Pearson)"),
                 "Rank IC": rep8(design, "Rank IC (Spearman)")})
for name, label, pairs in ((RETRAIN, "Yours: retrain every year, best seed by val MSE (4 sets combined)", [(y, y) for y in YEARS]),
                           (NEVER, "Yours: never retrain, best seed by val MSE (4 sets combined)", [(YEARS[0], y) for y in YEARS])):
    if name is None or name not in head.index:
        continue
    ics = [cell(cells, f, y, "test_rank_ic") for f, y in pairs]
    ics_p = [cell(cells, f, y, "test_ic_pearson") if "test_ic_pearson" in cells else None for f, y in pairs]
    yearly = [cell(cells, f, y, RET_COL) for f, y in pairs]
    rows.append({"Strategy": label, **{f"{y} return (%)": v for (f, y), v in zip(pairs, yearly)},
                 "Mean / joined return (%)": (sum(yearly) / len(yearly) if all(v is not None for v in yearly)
                                              else head.loc[name, "ann_return_%"]),
                 "2022-2024 annualized (%)": head.loc[name, "ann_return_%"], "Sharpe": head.loc[name, "sharpe"],
                 "Max drawdown (%)": -head.loc[name, "max_dd_%"],
                 "IC (Pearson)": sum(ics_p) / len(ics_p) if all(v is not None for v in ics_p) else None,
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
for col, label in (("sharpe", "Sharpe"), (RET_COL, "Net return (%)"), ("ann_return_%", "Annualized return (%)"),
                   ("test_rank_ic", "Rank IC"),
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
                     "Trade every day: return (%)": cell(old_cells, f, y, RET_COL if RET_COL in old_cells else "ann_return_%"),
                     "Trade every day: Sharpe": cell(old_cells, f, y, "sharpe"),
                     "Paper rule: return (%)": cell(cells, f, y, RET_COL),
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
        g = sel.pivot(index="test_year", columns="set", values=SET_RET_COL)
        g.columns = [vname[c] for c in g.columns]
        g["Average V1-V4"] = g.mean(axis=1)
        g.loc["Mean"] = g.mean(axis=0)
        g = g.reset_index().rename(columns={"test_year": "Test year"})
        g.insert(0, "Design", label)
        blocks.append(g)
    if REP7 is not None:
        mean_blocks = []
        for design, label in MEAN_DESIGNS:
            g = REP7[REP7["design"] == design].drop(columns="design").rename(columns={"test_year": "Test year"})
            if not g.empty:
                g.insert(0, "Design", label)
                mean_blocks.append(g)
        blocks = mean_blocks + [b.assign(Design=b["Design"] + ", best seed by val MSE") for b in blocks]
    if blocks:
        paper_row = pd.DataFrame([{"Design": "Paper: EXAMM", "Test year": y, "Average V1-V4": PAPER[str(y)]} for y in YEARS]
                                 + [{"Design": "Paper: EXAMM", "Test year": "Mean", "Average V1-V4": PAPER["mean"]}])
        set_table = pd.concat(blocks + [paper_row], ignore_index=True)

# ---------------------------------------------------------------- Table 5 (per panel and year) ----
table5 = None
if per_set is not None:
    sets_sorted = sorted(per_set["set"].unique())

    def mine(fold, v, year):
        s = sets_sorted[int(v[1]) - 1]
        r = per_set[(per_set["set"] == s) & (per_set["test_year"] == year) & (per_set["genomes_from_fold"] == fold)]
        return float(r[SET_RET_COL].iloc[0]) if len(r) else None

    def own_bench(v, year, col):
        if bench is None:
            return None
        s = sets_sorted[int(v[1]) - 1]
        r = bench[(bench["set"] == s) & (bench["test_year"] == year)]
        return float(r[col].iloc[0]) if len(r) else None

    t5 = []
    for v in ["V1", "V2", "V3", "V4"]:
        for y in YEARS:
            bh, ew, ex = PAPER_T5[(v, y)]
            mean_r = rep7("retrain every year", y, v)
            t5.append({"Panel": v, "Test yr": y, "Paper B&H": bh, "Paper EW": ew, "Paper EXAMM": ex,
                       "Yours: retrain every year (mean of 10 seeds)": mean_r,
                       "Yours: never retrain (mean of 10 seeds)": rep7("never retrain (fold 2022 models)", y, v),
                       "Retrain (mean of seeds) minus paper EXAMM": (mean_r - ex) if mean_r is not None else None,
                       "Yours: retrain, best seed by val MSE": mine(y, v, y),
                       "Yours: never retrain, best seed by val MSE": mine(YEARS[0], v, y),
                       "Yours: B&H": own_bench(v, y, "bh_total_%"), "Yours: EW": own_bench(v, y, "ew_total_%")})
    table5 = pd.DataFrame(t5)
    means = table5.drop(columns=["Panel", "Test yr"]).mean(numeric_only=True)
    table5 = pd.concat([table5, pd.DataFrame([{"Panel": "Mean", "Test yr": "", **means.to_dict()}])], ignore_index=True)

# ---------------------------------------------------------------- Table 8 / model size ----
table8_sheet = None
if table8 is not None:
    t8 = table8.rename(columns={"portfolio": "Portfolio", "design": "Design"})
    paper8 = pd.DataFrame([{"Design": "Paper: EXAMM (Table 8)", "Portfolio": "Average V1-V4", "IC (Pearson)": PAPER["ic"],
                            "Net (%) mean of years": PAPER["mean"], "Sharpe (2022-24)": PAPER["sharpe"], "MDD (%)": PAPER["mdd"]}])
    cols8 = [c for c in ["Design", "Portfolio", "Rank IC (Spearman)", "IC (Pearson)", "Net (%) mean of years",
                         "Sharpe (2022-24)", "MDD (%)"] if c in t8]
    t8["Design"] = t8["Design"].astype(str) + ", best seed by val MSE"
    parts = [paper8]
    if REP8 is not None:
        m8 = REP8.rename(columns={"panel": "Portfolio", "design": "Design"})
        m8["Design"] = m8["Design"].map(dict(MEAN_DESIGNS)).fillna(m8["Design"])
        parts.append(m8[[c for c in cols8 if c in m8]])
    table8_sheet = pd.concat(parts + [t8[cols8]], ignore_index=True)[cols8]
size_sheet = None
if sizes is not None:
    size_sheet = sizes.rename(columns={"fold": "Test year (fold)", "set": "Set", "repeat": "Repeat",
                                       "enabled_nodes": "Enabled nodes", "enabled_edges": "Enabled edges",
                                       "enabled_recurrent_edges": "Enabled recurrent edges"})
    avg = size_sheet.drop(columns=["Test year (fold)", "Repeat"], errors="ignore").mean(numeric_only=True)
    size_sheet = pd.concat([size_sheet, pd.DataFrame([{"Test year (fold)": "Average", **avg.to_dict()}])], ignore_index=True)


# ---------------------------------------------------------------- Pareto (core7_main_results.py --pareto) ----
pareto_sheet, pareto_rep = None, None
_pdesigns = [("Retrain every year", RESULTS / f"core7_main{TAG}_pareto_summary.csv"),
             ("Never retrain (fold 2022 choice)", RESULTS / f"core7_main{TAG}_pareto_never_retrain.csv")]
_prows = []
for design, path in _pdesigns:
    if not path.exists():
        continue
    ps = pd.read_csv(path)
    for ch in ["lowest val MSE", "Pareto choice", "highest val return"]:
        g = ps[ps["choice"] == ch].set_index("period")
        if g.empty:
            continue
        joined_p = f"{YEARS[0]}-{YEARS[-1]}"
        row = {"Design": design, "Choice": ch}
        for y in YEARS:
            row[f"Net {y} (%)"] = g["net_return_%"].get(str(y))
        if joined_p in g.index:
            j = g.loc[joined_p]
            row.update({"Net mean of years (%)": j["net_return_%"], "Annualized 2022-24 (%)": j.get("ann_return_%"),
                        "Sharpe 2022-24": j["sharpe"], "Sharpe 95% low": j.get("sharpe_ci_lo"),
                        "Sharpe 95% high": j.get("sharpe_ci_hi"), "PSR": j.get("psr"), "Max DD 2022-24 (%)": j["max_dd_%"]})
        for y in YEARS:
            if "repeats" in g and str(y) in g.index and isinstance(g.loc[str(y), "repeats"], str):
                row[f"Repeats {y}"] = g.loc[str(y), "repeats"]
        if design.startswith("Never"):
            row = {k: v for k, v in row.items() if k not in (f"Repeats {YEARS[1]}", f"Repeats {YEARS[2]}")}
        _prows.append(row)
if _prows:
    pareto_sheet = pd.DataFrame(_prows)
if (RESULTS / f"core7_main{TAG}_pareto.csv").exists():
    pareto_rep = pd.read_csv(RESULTS / f"core7_main{TAG}_pareto.csv").rename(columns={
        "test_year": "Test year", "set": "Set", "repeat": "Repeat", "val_mse": "Val MSE", "val_rank_ic": "Val Rank IC",
        "val_return_%": "Val net (%)", "val_sharpe": "Val Sharpe", "test_return_%": "Test net (%)",
        "test_sharpe": "Test Sharpe", "test_max_dd_%": "Test max DD (%)", "test_mse": "Test MSE",
        "test_rank_ic": "Test Rank IC", "val_frontier": "On val frontier", "choice": "Label"})
    pareto_rep["Label"] = pareto_rep["Label"].fillna("")


# ---------------------------------------------------------------- mean over the 10 repeats (--mean-repeats) ----
rep7_sheet, rep8_sheet = None, None
if (RESULTS / f"core7_main{TAG}_repeats_table7.csv").exists():
    r7 = pd.read_csv(RESULTS / f"core7_main{TAG}_repeats_table7.csv").rename(columns={"design": "Design", "test_year": "Test year"})
    r7["Test year"] = r7["Test year"].astype(str)
    paper7 = pd.DataFrame([{"Design": "Paper: EXAMM (Table 5)", "Test year": str(y),
                            **{v: PAPER_T5[(v, y)][2] for v in ["V1", "V2", "V3", "V4"]}} for y in YEARS])
    paper7["Average V1-V4"] = paper7[["V1", "V2", "V3", "V4"]].mean(axis=1)
    pm = paper7[["V1", "V2", "V3", "V4", "Average V1-V4"]].mean()
    paper7 = pd.concat([paper7, pd.DataFrame([{"Design": "Paper: EXAMM (Table 5)", "Test year": "Mean", **pm.to_dict()}])])
    rep7_sheet = pd.concat([paper7, r7], ignore_index=True)
if (RESULTS / f"core7_main{TAG}_repeats_table8.csv").exists():
    r8 = pd.read_csv(RESULTS / f"core7_main{TAG}_repeats_table8.csv").rename(columns={"design": "Design", "panel": "Portfolio"})
    paper8r = pd.DataFrame([{"Design": "Paper: EXAMM (Table 8)", "Portfolio": "Average V1-V4", "IC (Pearson)": PAPER["ic"],
                             "Net (%) mean of years": PAPER["mean"], "Sharpe (2022-24)": PAPER["sharpe"], "MDD (%)": PAPER["mdd"]}])
    rep8_sheet = pd.concat([paper8r, r8], ignore_index=True)[["Design", "Portfolio"] + [c for c in r8.columns
                                                                                         if c not in ("Design", "Portfolio")]]


notes = [
    "MAIN COMPARISON = MEAN OVER THE 10 SEEDS, as the paper reports (\"every configuration is repeated 10 times with "
    "independent seeds, and we report the mean over repeats\"): every seed per panel traded on its own, metrics per seed, "
    "then averaged. Summary, Per set, Table 5 and Table 8 show it first; the single best seed by validation MSE "
    "(the earlier main result) follows as a labelled second row/column. Grid, Joined and Rule compared use the best seed.",
    "Model: Pooled EXAMM, RET target, the 7 core7 inputs; per test year Y the fold train <= Y-2 / validation Y-1; each "
    "panel set's genome = the best of 10 runs by validation MSE (no Pareto frontier).",
    "Trading: the paper's Algorithm 1 (paper_long_short.py): long the K highest and short the K lowest forecasts (K = " + str(K) + " here; the paper uses 10), each "
    "side sized current capital / K, rebalanced ONLY when the K-th-highest forecast is > 0 and the K-th-lowest < 0 "
    "(otherwise the positions are kept). Forecasts converted to raw daily returns first (x IQR + median). Costs on.",
    "The 4 panel sets are combined with equal weight into one portfolio (Summary, Joined, Grid); the Per set sheet "
    "treats V1-V4 separately and averages them, as the paper's Table 7.",
    "Retrain every year = each year's own models (the paper's design). Never retrain = the 2022-fold models (trained "
    "<= 2020, validated on 2021) for 2022-2024. Both are free of leakage.",
    "Grid: cells where a fold's models are scored on an EARLIER year are leakage (only computed with --all) and are "
    "highlighted; they are not test results.",
    "Intervals: 95% stationary block bootstrap; PSR = P(true Sharpe > 0). Paper numbers are from its Tables 5, 7 and 8.",
    "Net return per test year = total return over the test year, (C - C0) / C0, as the paper reports it; the "
    "'2022-2024 annualized' columns are the compound annual return of the joined years. Your B&H: equal dollars in every "
    "stock at the first price + cost, held to the last price - cost; your EW: portfolio.py's equal-weight rule.",
    "Table 8: the paper's IC (+0.0106, its Tables 4 and 8) is the PEARSON IC, so compare it with the IC (Pearson) "
    "column; Rank IC = mean daily cross-sectional Spearman correlation of forecast and next-day return (Pearson shown "
    "too); Sharpe and MDD per panel over 2022-2024, then averaged over V1-V4. Model size: enabled nodes / edges / "
    "recurrent edges of the chosen genomes (gated nodes add internal weights, so the paper's 66 parameters can be higher).",
    "Caveats: three test years only; wide intervals; the remaining 2024 gap to the paper is most likely the genome "
    "(the paper's own EXAMM runs); short-sale costs follow this repository's stock.py (the paper's version may differ).",
    "Pareto (core7_main_results.py --pareto): per test year and set, the 10 repeats scored on the VALIDATION year by MSE "
    "and by the net return of the same trading rule; frontier = repeats no other beats on both; Pareto choice = lowest "
    "sum of the two ranks (ties -> lower MSE), fixed before the test year. Never retrain keeps the fold-2022 choice "
    "(validation 2021) for 2022-2024. Validation and test returns are nearly uncorrelated across repeats, so "
    "differences between the choices are within noise.",
    "Repeats T7 / T8 (core7_main_results.py, on by default): the paper's reporting -- every one of the 10 seeds per "
    "panel traded on its own, metrics per seed, then the MEAN over the seeds; no choice by validation MSE. Sharpe and MDD "
    "per seed over 2022-2024 (retrain every year pairs the seeds by repeat number across folds).",
    "Source: core7_main*.csv (core7_main_results.py) and core7_main_everyday*.csv.",
]

sheets = [("Summary", summary, f"[L/S {K}/{K}] Paper rule: your EXAMM results next to the paper's (test 2022-2024, costs on)"),
          ("Joined", joined, f"[L/S {K}/{K}] Both designs, 2022-2024 joined, with 95% intervals"),
          ("Grid", grid, f"[L/S {K}/{K}] Every fold's models on every test year computed (4 sets combined)")]
if compare is not None:
    sheets.append(("Rule compared", compare, f"[L/S {K}/{K}] Paper rule (trade only when signs agree) vs trading every day"))
if set_table is not None:
    sheets.append(("Per set", set_table, f"[L/S {K}/{K}] Net return (%) per panel set V1-V4 and their average (paper's Table 7 style)"))

if table5 is not None:
    sheets.append(("Table 5", table5, f"[L/S {K}/{K}] Net return (%) per panel (V1-V4) and test year: the paper's Table 5 vs yours"))
if table8_sheet is not None:
    sheets.append(("Table 8", table8_sheet, f"[L/S {K}/{K}] Model summary per panel and averaged over V1-V4 (paper's Table 8 style)"))
if size_sheet is not None:
    sheets.append(("Model size", size_sheet, f"[L/S {K}/{K}] The chosen genomes' size (paper's EXAMM: 66 parameters on average)"))
if pareto_sheet is not None:
    sheets.append(("Pareto", pareto_sheet, f"[L/S {K}/{K}] Choosing each set's genome: lowest validation MSE vs Pareto choice vs highest "
                                           "validation return (paper rule, costs on, 4 sets combined)"))
if pareto_rep is not None:
    sheets.append(("Pareto repeats", pareto_rep, f"[L/S {K}/{K}] Every repeat: validation and test scores, frontier membership, labels"))
if rep7_sheet is not None:
    sheets.append(("Repeats T7", rep7_sheet, f"[L/S {K}/{K}] Net return (%) per panel and year, MEAN over the 10 repeats "
                                             "(the paper's reporting), paper's Table 5 EXAMM on top"))
if rep8_sheet is not None:
    sheets.append(("Repeats T8", rep8_sheet, f"[L/S {K}/{K}] Table 8 style, mean over the 10 repeats, then averaged over V1-V4"))
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
    ws.row_dimensions[hdr].height = 48
    NOT_NUMBERS = {"test year", "test yr", "year", "repeat", "fold", "test year (fold)", "genomes from fold",
                   "days", "rows"}                                 # id columns (whole header names), shown as they are
    for row in ws.iter_rows(min_row=hdr + 1):
        first = str(row[0].value or "")
        mean_row = "MEAN OF 10 SEEDS" in first
        best_row = "best seed" in first
        for c in row:
            h = str(headers[c.column - 1])
            c.font, c.border = Font(name=F), BORDER
            if ws.title == "Notes":
                c.alignment = Alignment(wrap_text=True, vertical="top")
                continue
            if c.column == 1:
                c.alignment = Alignment(wrap_text=True, vertical="center")
            is_num = isinstance(c.value, (int, float)) and not isinstance(c.value, bool)
            if is_num and h.strip().lower() not in NOT_NUMBERS:
                # by the COLUMN header (or the Grid's Measure cell), never by a row label: "best seed by val MSE" is not MSE
                measure = first if ws.title == "Grid" else ""
                if h.startswith("MSE") or " MSE" in h or measure.startswith("MSE"):
                    # plain decimals, never scientific: ~3-4 significant digits whatever the size (0.000230, 0.2262, 54.33)
                    c.number_format = "[<0.01]0.000000;[<10]0.0000;0.00"
                elif "IC" in h.replace("ICE", "") or measure.endswith("IC"):
                    c.number_format = "+0.0000;-0.0000"
                elif "PSR" in h:
                    c.number_format = "0.00"
                elif any(k in h.lower() for k in ("enabled", "connections", "nodes", "edges")):
                    c.number_format = "0" if isinstance(c.value, int) else "0.0"
                else:
                    c.number_format = "+0.00;-0.00" if ("%" in h or "return" in h.lower() or h.startswith(("V", "Average"))
                                                        or "Yours" in h or "Paper" in h or "minus" in h) else "0.00"
                c.alignment = Alignment(horizontal="right", vertical="center")
            elif is_num:
                c.number_format = "0" if float(c.value).is_integer() else "0.00"   # ids; explicit so Excel never picks E+ notation
            if mean_row:
                c.font = Font(name=F, bold=True)
            elif best_row:
                c.font = Font(name=F, italic=True, color="595959")
        if first.startswith("Paper"):
            for c in row:
                c.fill = PAPER_FILL
                c.font = Font(name=F, bold=True)
        if mean_row:
            for c in row:
                c.fill = MEAN_FILL
        if first == "Mean" or str(row[1].value if len(row) > 1 else "") == "Mean":
            for c in row:
                c.font = Font(name=F, bold=True, italic=c.font.italic, color=c.font.color)
                c.border = Border(left=EDGE, right=EDGE, top=Side(style="medium", color="808080"), bottom=EDGE)
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
        col = ws[get_column_letter(i)][hdr:]
        data = max([len(f"{c.value:+.4f}" if isinstance(c.value, float) else str(c.value or "")) for c in col] or [0])
        longest_word = max((len(w) for w in str(h or "").split()), default=0)
        if ws.title == "Notes":
            width = 140
        elif i == 1:
            width = min(max(data, longest_word) + 2, 52)          # labels: as long as needed, wrapped beyond 52
        else:
            width = min(max(data, longest_word, 9) + 3, 22)       # numbers: header wraps instead of widening the column
        ws.column_dimensions[get_column_letter(i)].width = width
    longest_hdr = max((len(str(h or "")) / max(ws.column_dimensions[get_column_letter(i)].width - 2, 1)
                       for i, h in enumerate(headers, 1)), default=1)
    ws.row_dimensions[hdr].height = 16 * max(2, min(5, math.ceil(longest_hdr)))     # room for the wrapped header lines
    if ws.title != "Notes":
        ws.freeze_panes = ws.cell(row=hdr + 1, column=2)
if "Table 5" in wb.sheetnames:                    # bar chart: paper EXAMM, yours (both designs), paper EW
    from openpyxl.chart import BarChart, Reference
    ws = wb["Table 5"]
    n = len(table5) - 1                            # without the Mean row
    chart = BarChart()
    chart.type, chart.grouping = "col", "clustered"
    chart.title = "Net return (%) per panel and test year"
    chart.y_axis.title = "Net return (%)"
    for col in (5, 6, 7, 4):
        chart.add_data(Reference(ws, min_col=col, min_row=2, max_row=2 + n), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, max_col=2, min_row=3, max_row=2 + n))
    chart.height, chart.width = 11, 30
    ws.add_chart(chart, f"A{n + 6}")
    for row in ws.iter_rows(min_row=3):
        for c in row[2:]:
            if isinstance(c.value, float):
                c.fill = GOOD if c.value > 0 else BAD
if "Pareto" in wb.sheetnames:
    ws = wb["Pareto"]
    hdrs = [c.value for c in ws[2]]
    for row in ws.iter_rows(min_row=3):
        for i, h in enumerate(hdrs):
            if str(h).startswith("Net") and isinstance(row[i].value, (int, float)):
                row[i].fill = GOOD if row[i].value > 0 else BAD
        if row[1].value == "lowest val MSE":
            row[1].font = Font(name=F, bold=True)
    png = RESULTS / f"core7_main{TAG}_pareto.png"
    if png.exists():
        try:
            from openpyxl.drawing.image import Image
            img = Image(str(png))
            img.width, img.height = img.width * 0.5, img.height * 0.5
            ws.add_image(img, f"A{ws.max_row + 3}")
        except ImportError:
            print("Pillow not installed -- Pareto figure not embedded (python3 -m pip install --user pillow)")
if "Pareto repeats" in wb.sheetnames:
    ws = wb["Pareto repeats"]
    hdrs = [c.value for c in ws[2]]
    for row in ws.iter_rows(min_row=3):
        vals = {h: row[i].value for i, h in enumerate(hdrs)}
        if vals.get("Label"):
            for c in row:
                c.fill = PatternFill("solid", fgColor="FCE4D6")
        elif vals.get("On val frontier") is True:
            for c in row:
                c.fill = GOOD
wb.save(OUT)
print(f"Saved {OUT}")
print("Blue rows: the paper's numbers. Green: Sharpe interval above zero. Yellow (Grid): leakage cells. "
      "Pareto repeats: orange = a chosen repeat, green = on the validation frontier.")
