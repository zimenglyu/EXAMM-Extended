#!/usr/bin/env python3
"""Genome Pareto frontier with a DRAWDOWN axis: validation MSE vs validation Calmar / Martin ratio, max drawdown or
Ulcer index.

A separate script: core7_genome_frontier.py and its plot / workbook scripts are not changed. Same design as
core7_genome_frontier.py (Pooled EXAMM RET, folds 2022-2024: train <= Y-2, validation Y-1, test Y; 40 candidate
genomes per fold, each ONE genome shared by all 4 panel sets; the fixed trading rule, costs on), only the
second frontier axis differs:
  --axis calmar (default): Calmar ratio = annualized net return / max drawdown (higher is better);
                           keeps drawdown in the choice without rewarding a book that does nothing
  --axis martin          : Martin ratio (Ulcer performance index) = annualized net return / Ulcer index (higher
                           is better); like Calmar but uses every drawdown day instead of the single worst
                           fall, so it is less noisy
  --axis mdd             : max drawdown in % (LOWER is better); risk only -- a genome that barely moves
                           scores well, so read it next to the return
  --axis ulcer           : Ulcer index in % (LOWER is better) = sqrt(mean squared % drawdown below the running
                           peak, every day); depth and length of the falls; risk only, like mdd
Every output also reports max drawdown, Calmar, Ulcer index and Martin ratio, whatever the axis.
Frontier on VALIDATION: lower MSE and better second axis. Choice fixed in advance, as in core7_genome_frontier.py:
the frontier genome with the lowest MSE rank + second-axis rank, ties -> lower validation MSE; the two frontier
ends are recorded. Reported on TEST once (2022-2024 joined and per year): the chosen genome, both frontier ends,
the per-set EXAMM default and equal weight, with 95% block-bootstrap intervals, PSR, max drawdown and Calmar.

Nothing is retraded: it reads the per-genome table (core7_genome_frontier[_ls55].csv) and the trade cache
(results/genome_frontier/trades.npz) that core7_genome_frontier.py wrote, so run that first with the same --rule:
    cd ~/Financial_toolbox && python3 core7_genome_frontier.py --rule ls55
    cd ~/Financial_toolbox && python3 core7_genome_frontier_risk.py --axis calmar --rule ls55
    cd ~/Financial_toolbox && python3 core7_genome_frontier_risk.py --axis martin --rule ls55
    cd ~/Financial_toolbox && python3 core7_genome_frontier_risk.py --axis mdd --rule ls55
    cd ~/Financial_toolbox && python3 core7_genome_frontier_risk.py --axis ulcer --rule ls55
Outputs (results/): core7_genome_frontier_<axis>[_ls55].csv (every genome), _headline.csv, _by_year.csv,
.png (validation frontier, the same genomes on test, validation MSE vs return) and .xlsx (formatted tables).
"""
import sys

AXIS = sys.argv[sys.argv.index("--axis") + 1].lower() if "--axis" in sys.argv[:-1] else "calmar"
RULE_ARG = sys.argv[sys.argv.index("--rule") + 1].lower() if "--rule" in sys.argv[:-1] else "overlap"
if AXIS not in ("calmar", "martin", "mdd", "ulcer") or RULE_ARG not in ("overlap", "ls55"):
    sys.exit("usage: python3 core7_genome_frontier_risk.py [--axis calmar|martin|mdd|ulcer] [--rule overlap|ls55]")
_argv, sys.argv = sys.argv, [sys.argv[0], "--rule", RULE_ARG]   # core7_genome_frontier reads --rule at import
import numpy as np
import pandas as pd

from core7_genome_frontier import (RESULTS, RKEY, RULE, SETS, TAG as BASE_TAG, WORK, YEARS, bootstrap_ci, combine,
                                   metrics, pareto, psr)
sys.argv = _argv

TAG = f"_{AXIS}" + ("" if RULE_ARG == "overlap" else f"_{RULE_ARG}")
YCOL = {"calmar": "calmar", "martin": "martin", "mdd": "max_dd_%", "ulcer": "ulcer_%"}[AXIS]   # val_<YCOL>, test_<YCOL>
HIGHER_BETTER = AXIS in ("calmar", "martin")
YNAME = {"calmar": f"Calmar ratio of {RULE[0]} (return / max drawdown)",
         "martin": f"Martin ratio of {RULE[0]} (return / Ulcer index)",
         "mdd": f"max drawdown of {RULE[0]} (%, lower is better)",
         "ulcer": f"Ulcer index of {RULE[0]} (%, lower is better)"}[AXIS]
YSHORT = {"calmar": "Calmar ratio", "martin": "Martin ratio", "mdd": "max drawdown (%)", "ulcer": "Ulcer index (%)"}[AXIS]


def calmar(m):
    r, dd = m["ann_return_%"], m["max_dd_%"]
    return r / dd if np.isfinite(r) and dd > 1e-9 else np.nan


def ulcer(r):
    """Ulcer index in %: sqrt of the mean squared % drawdown below the running peak of the account, every day."""
    value = np.cumprod(1 + np.asarray(r, dtype=float))
    dd = (1 - value / np.maximum.accumulate(value)) * 100
    return float(np.sqrt(np.mean(dd ** 2)))


def martin(m, u):
    """Martin ratio (Ulcer performance index): annualized net return (%) / Ulcer index (%); no risk-free rate,
    as for the Sharpe of these self-financing long-short books."""
    r = m["ann_return_%"]
    return r / u if np.isfinite(r) and u > 1e-9 else np.nan


def risk(r):
    m = metrics(r)
    u = ulcer(r)
    return {**m, "calmar": calmar(m), "ulcer_%": u, "martin": martin(m, u)}


def need(path, hint):
    if not path.exists():
        sys.exit(f"missing {path} -- {hint}")
    return path


# ---------------------------------------------------------------- inputs ----
base = pd.read_csv(need(RESULTS / f"core7_genome_frontier{BASE_TAG}.csv",
                        f"run: python3 core7_genome_frontier.py --rule {RULE_ARG}"))
base["repeat"] = base["repeat"].astype(str)
cache = dict(np.load(need(WORK / "trades.npz", f"run: python3 core7_genome_frontier.py --rule {RULE_ARG}")))


def book(y, s, r, split, targets=SETS):
    keys = [f"{RKEY}{y}|{s}|{r}|{t}|{split}" for t in targets]
    if not all(k in cache for k in keys):
        return None
    return combine([cache[k] for k in keys])


rows, books = [], {}
for _, b in base.iterrows():
    y, s, r = int(b["fold"]), b["source_set"], b["repeat"]
    row = {k: b[k] for k in ("fold", "source_set", "repeat", "genome", "examm_val_mse", "val_mse", "val_rank_ic",
                             "test_mse", "test_rank_ic") if k in b}
    ok = True
    for split in ("val", "test"):
        bk = book(y, s, r, split)
        if bk is None:
            ok = False
            break
        books[(y, f"{s}/{r}", split)] = bk
        m = risk(bk)
        row.update({f"{split}_return_%": m["ann_return_%"], f"{split}_sharpe": m["sharpe"],
                    f"{split}_max_dd_%": m["max_dd_%"], f"{split}_calmar": m["calmar"],
                    f"{split}_ulcer_%": m["ulcer_%"], f"{split}_martin": m["martin"]})
    if ok:
        rows.append(row)
table = pd.DataFrame(rows).dropna(subset=["val_mse", f"val_{YCOL}"]).reset_index(drop=True)
if table.empty:
    sys.exit("no genome has both validation MSE and trades in the cache -- rerun core7_genome_frontier.py "
             f"--rule {RULE_ARG}")

# ---------------------------------------------------------------- frontier and choice (VALIDATION) ----
sign = 1.0 if HIGHER_BETTER else -1.0           # pareto() maximizes its second argument
table["val_frontier"] = False
table["chosen"] = ""
for y, g in table.groupby("fold"):
    table.loc[g.index, "val_frontier"] = pareto(g["val_mse"].to_numpy(), sign * g[f"val_{YCOL}"].to_numpy())
    g = table.loc[g.index]
    score = (g["val_mse"].rank(method="min") + g[f"val_{YCOL}"].rank(method="min", ascending=not HIGHER_BETTER)
             ).where(g["val_frontier"])
    pick = g.loc[score[score == score.min()].index, "val_mse"].idxmin()
    table.loc[pick, "chosen"] = "chosen (frontier, best rank sum)"
    best_y = g[f"val_{YCOL}"].idxmax() if HIGHER_BETTER else g[f"val_{YCOL}"].idxmin()
    for idx, label in ((g["val_mse"].idxmin(), "frontier end: lowest val MSE"),
                       (best_y, f"frontier end: best val {AXIS}")):
        table.loc[idx, "chosen"] = (table.loc[idx, "chosen"] + "; " if table.loc[idx, "chosen"] else "") + label
for sp in ("val", "test"):
    table[f"{sp}_mse_pct"] = table.groupby("fold")[f"{sp}_mse"].rank(ascending=False, pct=True) * 100


# ---------------------------------------------------------------- report (TEST) ----
def picked(label):
    out = {}
    for y, g in table.groupby("fold"):
        r = g[g["chosen"].str.contains(label, regex=False)].iloc[0]
        out[y] = (f"{r['source_set']}/{r['repeat']}", books[(y, f"{r['source_set']}/{r['repeat']}", "test")])
    return out


strategies = {"CHOSEN genome (val frontier, best rank sum)": picked("chosen"),
              "frontier end: lowest val MSE (EXAMM criterion)": picked("lowest val MSE"),
              f"frontier end: best val {AXIS}": picked(f"best val {AXIS}")}
default = {}
for y in YEARS:          # each set's own lowest-validation-MSE genome on its own set (EXAMM's per-set default)
    g = base[base["fold"] == y]
    per = []
    for t in SETS:
        own = g[g["source_set"] == t]
        if own.empty or own["examm_val_mse"].isna().all():
            break
        r = own.loc[own["examm_val_mse"].idxmin(), "repeat"]
        k = f"{RKEY}{y}|{t}|{r}|{t}|test"
        if k not in cache:
            break
        per.append(cache[k])
    if len(per) == len(SETS):
        default[y] = ("per set", combine(per))
if len(default) == len(YEARS):
    strategies["per-set EXAMM default (each set's lowest val MSE, own set)"] = default
if all(f"{y}|EW|test" in cache for y in YEARS):
    strategies["equal weight (benchmark)"] = {y: ("-", cache[f"{y}|EW|test"]) for y in YEARS}

head, by_year = [], []
for name, per_year in strategies.items():
    joined = np.concatenate([per_year[y][1] for y in YEARS])
    head.append({"strategy": name, **risk(joined), **bootstrap_ci(joined), "psr": psr(joined),
                 "genomes": ", ".join(f"{y}: {per_year[y][0]}" for y in YEARS)})
    for y in YEARS:
        by_year.append({"strategy": name, "year": y, "genome": per_year[y][0], **risk(per_year[y][1])})
head = pd.DataFrame(head).set_index("strategy")
by_year = pd.DataFrame(by_year)

RESULTS.mkdir(exist_ok=True)
out = RESULTS / f"core7_genome_frontier{TAG}"
table.to_csv(f"{out}.csv", index=False)
head.to_csv(f"{out}_headline.csv")
by_year.to_csv(f"{out}_by_year.csv", index=False)

pd.set_option("display.width", 220)
print("=" * 100)
print(f"GENOME PARETO FRONTIER -- validation MSE vs validation {YNAME}")
print("Pooled RET, folds train <= Y-2 / val Y-1 / test Y, one genome for all 4 sets, costs on; chosen on VALIDATION,")
print("reported on TEST 2022-2024 once.")
print("=" * 100)
cols = ["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
        "psr", "max_dd_%", "calmar", "ulcer_%", "martin"]
print(head[[c for c in cols if c in head]].astype(float).round(2).to_string())
print("ci = 95% stationary block bootstrap (mean block 10 days); psr = P(true Sharpe > 0); calmar = return / max drawdown;")
print("ulcer_% = sqrt(mean squared % drawdown, every day); martin = return / ulcer_%.")
print("\nTest max drawdown (%) per year:")
print(by_year.pivot(index="strategy", columns="year", values="max_dd_%").reindex(head.index).round(1).to_string())
print("\nValidation frontier genomes (and how they did on test):")
show = ["fold", "source_set", "repeat", "val_mse", "val_return_%", "val_max_dd_%", "val_calmar", "val_ulcer_%",
        "val_martin", "test_return_%", "test_max_dd_%", "test_calmar", "test_ulcer_%", "test_martin", "chosen"]
print(table[table["val_frontier"]][show].round({c: (7 if c == "val_mse" else 2) for c in show[3:-1]})
      .to_string(index=False))
print("\nSpearman correlation, validation MSE (lower = better) with:")
for y, g in table.groupby("fold"):
    c = g[list(dict.fromkeys(["val_mse", f"val_{YCOL}", "val_return_%", f"test_{YCOL}"]))].corr(method="spearman")["val_mse"]
    print(f"  {y}: val {AXIS} {c[f'val_{YCOL}']:+.2f}, val return {c['val_return_%']:+.2f}, "
          f"test {AXIS} {c[f'test_{YCOL}']:+.2f}  | val {AXIS} vs test {AXIS} "
          f"{g[[f'val_{YCOL}', f'test_{YCOL}']].corr(method='spearman').iloc[0, 1]:+.2f}")

# ---------------------------------------------------------------- figure ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})
    GREY, NAVY, RED, GREEN = "#b0b0b0", "#1f3864", "#c0392b", "#2e7d32"
    panels = [("val_mse_pct", f"val_{YCOL}", YSHORT, "Validation (frontier chosen here)"),
              ("test_mse_pct", f"test_{YCOL}", YSHORT, "Test (same genomes)"),
              ("val_mse_pct", "val_return_%", "net return (%/yr)", "Validation: MSE vs net return")]
    fig, axes = plt.subplots(len(YEARS), 3, figsize=(16, 4.3 * len(YEARS)), squeeze=False)
    for i, (row_axes, y) in enumerate(zip(axes, YEARS)):
        g = table[table["fold"] == y]
        front = g[g["val_frontier"] == True]
        pick = g[g["chosen"].str.contains("chosen", regex=False)]
        ends = g[g["chosen"].str.contains("frontier end", regex=False) & ~g["chosen"].str.contains("chosen", regex=False)]
        for ax, (xc, yc, ylab, title) in zip(row_axes, panels):
            ax.scatter(g[xc], g[yc], s=22, c=GREY, edgecolor="none", zorder=2)
            f = front.sort_values(xc)
            ax.plot(f[xc], f[yc], "-o", c=NAVY, lw=1.4, ms=5, zorder=3)
            ax.scatter(ends[xc], ends[yc], marker="s", s=55, facecolor="white", edgecolor=GREEN, lw=1.5, zorder=4)
            ax.scatter(pick[xc], pick[yc], marker="*", s=230, c=RED, edgecolor="white", lw=0.6, zorder=5)
            for d, colour in ((pick, RED), (ends, GREEN)):
                for _, r in d.iterrows():
                    ax.annotate(f"{r['source_set']}/{r['repeat']}", (r[xc], r[yc]), fontsize=7, color=colour,
                                xytext=(5, 4), textcoords="offset points")
            if not HIGHER_BETTER and yc.endswith(YCOL):
                ax.invert_yaxis()                          # better (smaller drawdown) at the top
            else:
                ax.axhline(0, c="#7f7f7f", lw=0.8, zorder=1)
            ax.set_xlim(-4, 104)
            rho = g[[xc, yc]].dropna().corr(method="spearman").iloc[0, 1]
            ax.text(0.98, 0.03, f"Spearman rho = {rho:+.2f}", transform=ax.transAxes, va="bottom", ha="right", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"))
            if i == 0:
                ax.set_title(title, fontweight="bold")
            if i == len(YEARS) - 1:
                ax.set_xlabel("MSE percentile rank (100 = lowest MSE)")
            ax.set_ylabel(ylab + (" (top = smaller)" if not HIGHER_BETTER and yc.endswith(YCOL) else ""))
        row_axes[0].annotate(f"Test year {y}\n(validation {y - 1})", xy=(-0.28, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=GREY, label="candidate genomes (40 per year)"),
                        Line2D([], [], color=NAVY, marker="o", ms=5, label="validation Pareto frontier"),
                        Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=GREEN, mew=1.5, label="frontier ends"),
                        Line2D([], [], ls="", marker="*", ms=14, color=RED, label="chosen genome (pre-set rule)")],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Genome Pareto frontier: MSE vs {YNAME} (one EXAMM genome for all 4 panel sets)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.97))
    fig.savefig(f"{out}.png", dpi=160)
    print(f"\nSaved {out}.png")
except ImportError:
    print("\nmatplotlib not installed -- skipped the figure (pip3 install --user matplotlib)")

# ---------------------------------------------------------------- workbook ----
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print(f"Saved {out}.csv, _headline.csv, _by_year.csv (openpyxl not installed -- no .xlsx)")
    sys.exit(0)

summary = head.reset_index()[[c for c in ["strategy", "ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%",
                                          "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "psr", "max_dd_%", "calmar", "ulcer_%",
                                          "martin", "genomes"]
                              if c in head.reset_index()]]
summary.columns = ["Strategy (genome chosen on validation)", "Annual return (%)", "Return 95% low", "Return 95% high",
                   "Volatility (%)", "Sharpe", "Sharpe 95% low", "Sharpe 95% high", "PSR", "Max drawdown (%)",
                   "Calmar", "Ulcer index (%)", "Martin ratio", "Genomes used"][:len(summary.columns)]
per_year = []
for name, g in by_year.groupby("strategy", sort=False):
    g = g.set_index("year")
    per_year.append({"Strategy": name, **{f"{y} return (%)": g.loc[y, "ann_return_%"] for y in YEARS},
                     **{f"{y} max drawdown (%)": g.loc[y, "max_dd_%"] for y in YEARS},
                     **{f"{y} Calmar": g.loc[y, "calmar"] for y in YEARS},
                     **{f"{y} Ulcer index (%)": g.loc[y, "ulcer_%"] for y in YEARS},
                     **{f"{y} Martin": g.loc[y, "martin"] for y in YEARS},
                     **{f"{y} Sharpe": g.loc[y, "sharpe"] for y in YEARS}})
per_year = pd.DataFrame(per_year)
gcols = {"fold": "Test year", "source_set": "Trained on", "repeat": "Repeat", "genome": "Genome", "val_mse": "Val MSE",
         "val_mse_pct": "Val MSE percentile (100 = lowest)", "val_return_%": "Val return (%)",
         "val_max_dd_%": "Val max drawdown (%)", "val_calmar": "Val Calmar", "val_ulcer_%": "Val Ulcer index (%)",
         "val_martin": "Val Martin", "val_sharpe": "Val Sharpe",
         "test_mse": "Test MSE", "test_return_%": "Test return (%)", "test_max_dd_%": "Test max drawdown (%)",
         "test_calmar": "Test Calmar", "test_ulcer_%": "Test Ulcer index (%)", "test_martin": "Test Martin",
         "test_sharpe": "Test Sharpe", "val_frontier": "On val frontier",
         "chosen": "Picked as"}
genomes = table[[c for c in gcols if c in table]].rename(columns=gcols).sort_values(["Test year", "Val MSE"])
genomes["On val frontier"] = genomes["On val frontier"].map({True: "Yes", False: ""})
notes = [
    "Frontier axes (validation): lower MSE and " + {"calmar": "higher Calmar ratio (annualized net return / max drawdown)",
                                                     "martin": "higher Martin ratio (annualized net return / Ulcer index)",
                                                     "mdd": "lower max drawdown", "ulcer": "lower Ulcer index"}[AXIS] + ".",
    "Same design as core7_genome_frontier.py: Pooled EXAMM RET, folds train <= Y-2 / validation Y-1 / test Y (2022-2024); "
    "40 candidate genomes per fold (4 sets x 10 runs, each run's lowest-validation-MSE genome), each ONE genome for all 4 sets.",
    f"Trading rule: {RULE[0]}, costs on, the 4 sets combined with equal weight. Nothing retraded: trades from "
    "core7_genome_frontier.py's cache.",
    "Choice fixed in advance: frontier genome with the lowest MSE rank + second-axis rank; ties -> lower validation MSE. "
    "Frontier ends shown for comparison only.",
    "Max drawdown = worst peak-to-trough fall of the account value; Calmar = annualized return / max drawdown (undefined "
    "if the account fell below zero). Both rest on one path over one year, so they are noisy.",
    "Ulcer index = sqrt(mean of the squared % drawdown below the running peak, every day): depth and length of the falls. "
    "Martin ratio = annualized return / Ulcer index (no risk-free rate, as for the Sharpe of these self-financing books); "
    "it uses every drawdown day, so it is less noisy than Calmar.",
    "Intervals: 95% stationary block bootstrap; PSR = P(true Sharpe > 0). Green = Sharpe interval above zero; red = "
    "negative Sharpe; yellow = genomes picked on validation.",
    "Caveats: validation is EXAMM's own fitness year; 2022-2024 examined before in this project; three test years only.",
]
with pd.ExcelWriter(f"{out}.xlsx", engine="openpyxl") as xw:
    for name, df in (("Summary", summary), ("Per year", per_year), ("Genomes", genomes)):
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)
F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD, PICK = (PatternFill("solid", fgColor=c) for c in ("E2EFDA", "FBE5E5", "FFF2CC"))
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)
titles = {"Summary": f"Genome frontier, val MSE vs val {YSHORT}: test 2022-2024 joined, {RULE[0]}, costs on",
          "Per year": "Test return, max drawdown, Calmar, Ulcer index, Martin and Sharpe per year",
          "Genomes": "All candidate genomes: validation and test scores"}
wb = load_workbook(f"{out}.xlsx")
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
        vals = {h: row[i].value for i, h in enumerate(headers)}
        for c in row:
            h = str(headers[c.column - 1])
            c.font, c.border = Font(name=F), BORDER
            if ws.title == "Notes":
                c.alignment = Alignment(wrap_text=True)
            if isinstance(c.value, float):
                c.number_format = ("0.000E+00" if "MSE" in h and "percentile" not in h else "0" if "percentile" in h
                                   else "0.00")
        fill = (PICK if vals.get("Picked as") else
                GOOD if isinstance(vals.get("Sharpe 95% low"), (int, float)) and vals["Sharpe 95% low"] > 0 else
                BAD if isinstance(vals.get("Sharpe"), (int, float)) and vals["Sharpe"] < 0 else None)
        if fill:
            for c in row:
                c.fill = fill
    for i, h in enumerate(headers, 1):
        width = max([len(str(h or ""))] + [len(f"{c.value:.4f}" if isinstance(c.value, float) else str(c.value or ""))
                                            for c in ws[get_column_letter(i)][hdr:]])
        ws.column_dimensions[get_column_letter(i)].width = 140 if ws.title == "Notes" else min(max(width + 2, 10), 48)
    if ws.title != "Notes":
        ws.freeze_panes = ws.cell(row=hdr + 1, column=2)
wb.save(f"{out}.xlsx")
print(f"Saved {out}.csv, _headline.csv, _by_year.csv, .xlsx")
