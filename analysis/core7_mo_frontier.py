#!/usr/bin/env python3
"""Genome frontier of the MULTI-OBJECTIVE EXAMM runs (EXAMM-MO): choice on validation, test reported once.

Steps 6-7 of the multi-objective experiment. The EXAMM-MO search kept and bred genomes by Pareto rank on
validation MSE and a second validation objective (--objective mse_ir: information ratio of a daily long-short
5/5 book against the 10-day z-score reversal strategy; --objective mse_ret: that book's return). Each run saved
its Pareto archive; select_core7_mo_candidates.py pooled them into candidates.csv per fold, and
evaluate_rnn_core7_ret_mo_cross.sh had every candidate forecast all 4 panel sets on validation and test.
This script, per fold (test year Y = 2022, 2023, 2024; validation Y-1):
  1. re-scores every candidate as ONE genome for all 4 sets (200 stocks), in raw daily-return units:
     MSE and Rank IC (as core7_genome_frontier.py), and the fixed trading rule (--rule, default daily L/S 5/5,
     costs on, the 4 sets combined with equal weight): net return, Sharpe, max drawdown, Calmar, Ulcer index,
     Martin ratio, and against the reversal benchmark (the 10-day z-score reversal traded with the same rule):
     beta, residual alpha and information ratio (IR = alpha / residual std x sqrt(252));
  2. VALIDATION Pareto frontier: lower validation MSE vs higher validation IR (--run ir) or net return
     (--run return); one genome per fold chosen by the rule fixed in advance (lowest MSE rank + second-axis
     rank among the frontier genomes, ties -> lower validation MSE); the two frontier ends recorded;
  3. reports on TEST 2022-2024 once (joined and per year): the chosen genome, both frontier ends, the
     MSE-only search's genome-frontier choice and per-set EXAMM default (from core7_genome_frontier.py's
     results, if present), the reversal benchmark and equal weight; 95% block-bootstrap intervals and PSR.
Nothing in core7_genome_frontier.py is changed; its functions are imported.

Run on the Mac after the evaluation (and, for the MSE-only rows, core7_genome_frontier.py --rule ls55):
    cd ~/Financial_toolbox && python3 core7_mo_frontier.py --run ir                 # MSE vs IR runs
    cd ~/Financial_toolbox && python3 core7_mo_frontier.py --run return             # MSE vs return runs
    cd ~/Financial_toolbox && python3 core7_mo_frontier.py --run ir --years 2022    # only the folds evaluated so far
Options: --rule ls55 (default) | overlap. Trades are cached in results/genome_frontier/mo_trades.npz, so a
rerun only trades what is new. Outputs (results/): core7_mo_frontier_<run>[_overlap].csv (every candidate),
_headline.csv, _by_year.csv, .png, .xlsx.
"""
import sys


def arg(name, default):
    return sys.argv[sys.argv.index(name) + 1].lower() if name in sys.argv[:-1] else default


RUN = arg("--run", "ir")
RULE_ARG = arg("--rule", "ls55")
YEARS_ARG = [int(a) for a in sys.argv[sys.argv.index("--years") + 1:] if a.isdigit()] if "--years" in sys.argv else None
if RUN not in ("ir", "return") or RULE_ARG not in ("ls55", "overlap"):
    sys.exit("usage: python3 core7_mo_frontier.py [--run ir|return] [--rule ls55|overlap] [--years 2022 2023 2024]")
_argv, sys.argv = sys.argv, [sys.argv[0], "--rule", RULE_ARG]   # core7_genome_frontier reads --rule at import
import numpy as np
import pandas as pd

from core7_genome_frontier import (EXAMM, RESULTS, RKEY, RULE, SETS, TAG as BASE_TAG, TRADING_DAYS, WORK, YEARS,
                                   bootstrap_ci, combine, daily_returns, forecast_scores, metrics, pareto,
                                   prediction_dir, price_dir, psr, read_predictions, reversal_predictions)
sys.argv = _argv

RUN_DIR = {"ir": "core7_examm_ret_mo", "return": "core7_examm_ret_mo_return"}[RUN]
TAG = f"_{RUN}" + ("" if RULE_ARG == "ls55" else f"_{RULE_ARG}")
YCOL = {"ir": "ir", "return": "return_%"}[RUN]                 # val_<YCOL>, test_<YCOL>
YNAME = {"ir": f"information ratio of {RULE[0]} vs the reversal benchmark",
         "return": f"net return of {RULE[0]} (%/yr)"}[RUN]
YSHORT = {"ir": "information ratio", "return": "net return (%/yr)"}[RUN]
OTHER = {"ir": ("return_%", "net return (%/yr)"), "return": ("ir", "information ratio")}[RUN]
FOLDS = YEARS_ARG or YEARS
OUT = RESULTS / f"core7_mo_frontier{TAG}"


# ---------------------------------------------------------------- measures ----
def risk(r):
    """metrics() plus Calmar, Ulcer index (%) and Martin ratio, as core7_genome_frontier_risk.py."""
    m = metrics(r)
    value = np.cumprod(1 + np.asarray(r, dtype=float))
    dd = (1 - value / np.maximum.accumulate(value)) * 100
    u = float(np.sqrt(np.mean(dd ** 2)))
    ret = m["ann_return_%"]
    return {**m, "calmar": ret / m["max_dd_%"] if np.isfinite(ret) and m["max_dd_%"] > 1e-9 else np.nan,
            "ulcer_%": u, "martin": ret / u if np.isfinite(ret) and u > 1e-9 else np.nan}


def vs_benchmark(r, b):
    """OLS r = a + beta b + e on the same days: beta, residual alpha (%/yr) and information ratio a/sd(e)*sqrt(252)."""
    n = min(len(r), len(b))
    r, b = np.asarray(r[:n], dtype=float), np.asarray(b[:n], dtype=float)
    X = np.column_stack([np.ones(n), b])
    coef = np.linalg.lstsq(X, r, rcond=None)[0]
    e = r - X @ coef
    sd = float(np.sqrt(e @ e / max(n - 2, 1)))
    return {"beta": float(coef[1]), "alpha_%": coef[0] * TRADING_DAYS * 100,
            "ir": coef[0] / sd * np.sqrt(TRADING_DAYS) if sd > 1e-12 else np.nan}


def joined(per_year):
    return np.concatenate([np.asarray(x) for x in per_year])


# ---------------------------------------------------------------- trading with a cache ----
cache_path = WORK / "mo_trades.npz"
WORK.mkdir(parents=True, exist_ok=True)
cache = dict(np.load(cache_path)) if cache_path.exists() else {}
base_cache_path = WORK / "trades.npz"                       # core7_genome_frontier.py's (read only)
base_cache = dict(np.load(base_cache_path)) if base_cache_path.exists() else {}
new_trades = 0


def traded(key, make):
    global new_trades
    if key not in cache:
        cache[key] = make()
        new_trades += 1
        if new_trades % 50 == 0:
            np.savez(cache_path, **cache)
    return cache[key]


def mo_folder(y, s, r, g, t, split):
    return EXAMM / "test_output" / f"{RUN_DIR}_{y}" / s / str(r) / "mo_eval" / str(g) / f"evaluation_{split}_on_{t}"


# ---------------------------------------------------------------- score every candidate ----
rows, books, missing, bench = [], {}, [], {}
for y in FOLDS:
    cpath = EXAMM / "test_output" / f"{RUN_DIR}_{y}" / "candidates.csv"
    if not cpath.exists():
        print(f"!! {y}: no {cpath} -- run select_core7_mo_candidates.py (and the evaluation) first; fold skipped")
        continue
    cands = pd.read_csv(cpath)
    layout = {}
    for _, c in cands.iterrows():
        s, r, g = c["source_set"], str(c["repeat"]), int(c["genome_id"])
        row = {"fold": y, "source_set": s, "repeat": r, "genome_id": g, "file": c["file"],
               "reason": c["reason"], "examm_val_mse": c["val_mse"],
               f"examm_val_{'ir' if 'val_ir' in c else 'return'}": c.get("val_ir", c.get("val_return"))}
        complete = True
        for split in ("val", "test"):
            per, mses, ics = [], [], []
            for t in SETS:
                P = read_predictions(mo_folder(y, s, r, g, t, split), split)
                if P is None or not np.isfinite(P.to_numpy()).all():
                    complete = False
                    break
                layout.setdefault((t, split), (P, s, r, g))
                mse, ic = forecast_scores(P, y, t, split)
                mses.append(mse)
                ics.append(ic)
                per.append(traded(f"{RUN}|{RKEY}{y}|{s}|{r}|{g}|{t}|{split}", lambda: daily_returns(
                    prediction_dir(y, f"mo_{RUN}_{s}", f"{r}_{g}", t, split, P), price_dir(y, t, split),
                    RULE[1], RULE[2])))
            if not complete:
                break
            books[(y, f"{s}/{r}/{g}", split)] = combine(per)
            row.update({f"{split}_mse": float(np.mean(mses)), f"{split}_rank_ic": float(np.mean(ics))})
        if not complete:
            missing.append(f"{y} {s}/{r}/{g}")
            continue
        rows.append(row)

    # benchmarks for this fold: reversal (-z, 10 days) and equal weight, traded on the same files / rule
    for split in ("val", "test"):
        rev, ew = [], []
        for t in SETS:
            if (t, split) not in layout:
                break
            P, s, r, g = layout[(t, split)]
            rev.append(traded(f"REV|{RKEY}{y}|{t}|{split}", lambda: daily_returns(
                prediction_dir(y, "REV", "z10", t, split, reversal_predictions(y, t, split, len(P), list(P.columns))),
                price_dir(y, t, split), RULE[1], RULE[2])))
            ew.append(base_cache[f"{y}|EW|{split}"] if f"{y}|EW|{split}" in base_cache else traded(
                f"EW|{y}|{split}|{t}", lambda: daily_returns(prediction_dir(y, f"mo_{RUN}_{s}", f"{r}_{g}", t, split, P),
                                                             price_dir(y, t, split), "daily_equal_weight_return", {})))
        if len(rev) == len(SETS):
            bench[(y, "reversal", split)] = combine(rev)
            bench[(y, "equal weight", split)] = ew[0] if f"{y}|EW|{split}" in base_cache else combine(ew)
    np.savez(cache_path, **cache)
    print(f"fold {y}: {sum(1 for x in rows if x['fold'] == y)} of {len(cands)} candidates scored")

if missing:
    print(f"!! {len(missing)} candidate(s) left out (predictions missing or invalid on some set / split): "
          + ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else ""))
if not rows:
    sys.exit("no candidate could be scored -- run evaluate_rnn_core7_ret_mo_cross.sh first")
FOLDS = [y for y in FOLDS if any(x["fold"] == y for x in rows)]

for row in rows:
    y, key = row["fold"], f"{row['source_set']}/{row['repeat']}/{row['genome_id']}"
    for split in ("val", "test"):
        m = risk(books[(y, key, split)])
        row.update({f"{split}_return_%": m["ann_return_%"], f"{split}_sharpe": m["sharpe"],
                    f"{split}_max_dd_%": m["max_dd_%"], f"{split}_calmar": m["calmar"],
                    f"{split}_ulcer_%": m["ulcer_%"], f"{split}_martin": m["martin"]})
        if (y, "reversal", split) in bench:
            row.update({f"{split}_{k}": v for k, v in vs_benchmark(books[(y, key, split)],
                                                                    bench[(y, "reversal", split)]).items()})
table = pd.DataFrame(rows).dropna(subset=["val_mse", f"val_{YCOL}"]).reset_index(drop=True)

# ---------------------------------------------------------------- frontier and choice (VALIDATION) ----
table["val_frontier"] = False
table["chosen"] = ""
for y, g in table.groupby("fold"):
    table.loc[g.index, "val_frontier"] = pareto(g["val_mse"].to_numpy(), g[f"val_{YCOL}"].to_numpy())
    g = table.loc[g.index]
    score = (g["val_mse"].rank(method="min") + g[f"val_{YCOL}"].rank(method="min", ascending=False)).where(g["val_frontier"])
    table.loc[g.loc[score[score == score.min()].index, "val_mse"].idxmin(), "chosen"] = "chosen (frontier, best rank sum)"
    for idx, label in ((g["val_mse"].idxmin(), "frontier end: lowest val MSE"),
                       (g[f"val_{YCOL}"].idxmax(), f"frontier end: highest val {RUN}")):
        table.loc[idx, "chosen"] = (table.loc[idx, "chosen"] + "; " if table.loc[idx, "chosen"] else "") + label
for sp in ("val", "test"):
    table[f"{sp}_mse_pct"] = table.groupby("fold")[f"{sp}_mse"].rank(ascending=False, pct=True) * 100


# ---------------------------------------------------------------- report (TEST) ----
def picked(label):
    out = {}
    for y, g in table.groupby("fold"):
        r = g[g["chosen"].str.contains(label, regex=False)].iloc[0]
        key = f"{r['source_set']}/{r['repeat']}/{r['genome_id']}"
        out[y] = (key, books[(y, key, "test")], r["test_mse"])
    return out


strategies = {"CHOSEN multi-objective genome (val frontier, best rank sum)": picked("chosen"),
              "MO frontier end: lowest val MSE": picked("lowest val MSE"),
              f"MO frontier end: highest val {RUN}": picked(f"highest val {RUN}")}

# MSE-only search, from core7_genome_frontier.py's results with the same rule (same 4-set book, same costs)
base_csv = RESULTS / f"core7_genome_frontier{BASE_TAG}.csv"
if base_csv.exists() and base_cache:
    base = pd.read_csv(base_csv)
    base["repeat"] = base["repeat"].astype(str)
    base["chosen"] = base["chosen"].fillna("")
    for name, label in (("MSE-only search: genome-frontier choice (MSE vs return)", "chosen"),
                        ("MSE-only search: lowest val MSE genome", "lowest val MSE")):
        out = {}
        for y in FOLDS:
            hit = base[(base["fold"] == y) & base["chosen"].str.contains(label, regex=False)]
            keys = [f"{RKEY}{y}|{hit.iloc[0]['source_set']}|{hit.iloc[0]['repeat']}|{t}|test" for t in SETS] if len(hit) else []
            if keys and all(k in base_cache for k in keys):
                out[y] = (f"{hit.iloc[0]['source_set']}/{hit.iloc[0]['repeat']}", combine([base_cache[k] for k in keys]),
                          hit.iloc[0]["test_mse"])
        if len(out) == len(FOLDS):
            strategies[name] = out
    default = {}
    for y in FOLDS:
        g, per = base[base["fold"] == y], []
        for t in SETS:
            own = g[g["source_set"] == t]
            k = f"{RKEY}{y}|{t}|{own.loc[own['examm_val_mse'].idxmin(), 'repeat']}|{t}|test" if len(own) else ""
            if k in base_cache:
                per.append(base_cache[k])
        if len(per) == len(SETS):
            default[y] = ("per set", combine(per), np.nan)
    if len(default) == len(FOLDS):
        strategies["MSE-only search: per-set EXAMM default"] = default
else:
    print(f"(no {base_csv.name} / trades.npz -- run core7_genome_frontier.py --rule {RULE_ARG} for the MSE-only rows)")
for name, key in (("reversal benchmark (-z, 10 days)", "reversal"), ("equal weight (benchmark)", "equal weight")):
    if all((y, key, "test") in bench for y in FOLDS):
        strategies[name] = {y: ("-", bench[(y, key, "test")], np.nan) for y in FOLDS}

head, by_year = [], []
rev_test = [bench[(y, "reversal", "test")] for y in FOLDS] if all((y, "reversal", "test") in bench for y in FOLDS) else None
for name, per_year in strategies.items():
    r_all = joined([per_year[y][1] for y in FOLDS])
    reg = {}
    if rev_test is not None:
        n = [min(len(per_year[y][1]), len(rev_test[i])) for i, y in enumerate(FOLDS)]
        reg = vs_benchmark(joined([per_year[y][1][:k] for y, k in zip(FOLDS, n)]),
                           joined([rev_test[i][:k] for i, k in enumerate(n)]))
    head.append({"strategy": name, **risk(r_all), **bootstrap_ci(r_all), "psr": psr(r_all), **reg,
                 "test_mse": float(np.nanmean([per_year[y][2] for y in FOLDS])),
                 "genomes": ", ".join(f"{y}: {per_year[y][0]}" for y in FOLDS)})
    for i, y in enumerate(FOLDS):
        ry = vs_benchmark(per_year[y][1], rev_test[i]) if rev_test is not None else {}
        by_year.append({"strategy": name, "year": y, "genome": per_year[y][0], **risk(per_year[y][1]), **ry,
                        "test_mse": per_year[y][2]})
head = pd.DataFrame(head).set_index("strategy")
by_year = pd.DataFrame(by_year)
RESULTS.mkdir(exist_ok=True)
table.to_csv(f"{OUT}.csv", index=False)
head.to_csv(f"{OUT}_headline.csv")
by_year.to_csv(f"{OUT}_by_year.csv", index=False)

pd.set_option("display.width", 240)
print("=" * 110)
print(f"MULTI-OBJECTIVE EXAMM ({RUN_DIR}): validation MSE vs validation {YNAME}")
print(f"one genome for all 4 sets, {RULE[0]}, costs on; chosen on VALIDATION, reported on TEST {FOLDS} once")
print("=" * 110)
cols = ["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi", "psr",
        "max_dd_%", "martin", "alpha_%", "beta", "ir", "test_mse"]
print(head[[c for c in cols if c in head]].astype(float).round({c: 6 if c == "test_mse" else 2 for c in cols}).to_string())
print("ci = 95% block bootstrap; psr = P(true Sharpe > 0); martin = return / Ulcer index; alpha / beta / ir vs the "
      "reversal benchmark (ir = alpha / residual std x sqrt(252)); test_mse = mean over the years (raw returns).")
print("\nTest Sharpe per year:")
print(by_year.pivot(index="strategy", columns="year", values="sharpe").reindex(head.index).round(2).to_string())
print("\nValidation frontier genomes (and how they did on test):")
show = [c for c in ["fold", "source_set", "repeat", "genome_id", "val_mse", f"val_{YCOL}", "val_return_%", "test_mse",
                    f"test_{YCOL}", "test_return_%", "test_sharpe", "chosen"] if c in table]
show = list(dict.fromkeys(show))
print(table[table["val_frontier"]][show].round({c: (7 if c.endswith("mse") else 3) for c in show}).to_string(index=False))
print("\nSpearman correlations per fold:")
for y, g in table.groupby("fold"):
    c = lambda a, b: g[[a, b]].corr(method="spearman").iloc[0, 1]
    print(f"  {y} ({len(g)} candidates, {int(g['val_frontier'].sum())} on the val frontier): val MSE vs val {RUN} "
          f"{c('val_mse', f'val_{YCOL}'):+.2f} | val vs test {RUN} {c(f'val_{YCOL}', f'test_{YCOL}'):+.2f} | "
          f"val vs test MSE {c('val_mse', 'test_mse'):+.2f}")

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
              ("val_mse_pct", f"val_{OTHER[0]}", OTHER[1], f"Validation: MSE vs {OTHER[1].split(' (')[0]}")]
    fig, axes = plt.subplots(len(FOLDS), 3, figsize=(16, 4.3 * len(FOLDS)), squeeze=False)
    for i, (row_axes, y) in enumerate(zip(axes, FOLDS)):
        g = table[table["fold"] == y]
        front = g[g["val_frontier"] == True]
        pick = g[g["chosen"].str.contains("chosen", regex=False)]
        ends = g[g["chosen"].str.contains("frontier end", regex=False) & ~g["chosen"].str.contains("chosen", regex=False)]
        for ax, (xc, yc, ylab, title) in zip(row_axes, panels):
            if yc not in g:
                ax.set_visible(False)
                continue
            ax.scatter(g[xc], g[yc], s=18, c=GREY, edgecolor="none", zorder=2)
            f = front.sort_values(xc)
            ax.plot(f[xc], f[yc], "-o", c=NAVY, lw=1.4, ms=4, zorder=3)
            ax.scatter(ends[xc], ends[yc], marker="s", s=55, facecolor="white", edgecolor=GREEN, lw=1.5, zorder=4)
            ax.scatter(pick[xc], pick[yc], marker="*", s=230, c=RED, edgecolor="white", lw=0.6, zorder=5)
            for d, colour in ((pick, RED), (ends, GREEN)):
                for _, r in d.iterrows():
                    ax.annotate(f"{r['source_set']}/{r['repeat']}/{r['genome_id']}", (r[xc], r[yc]), fontsize=7,
                                color=colour, xytext=(5, 4), textcoords="offset points")
            ax.axhline(0, c="#7f7f7f", lw=0.8, zorder=1)
            ax.set_xlim(-4, 104)
            rho = g[[xc, yc]].dropna().corr(method="spearman").iloc[0, 1]
            ax.text(0.98, 0.03, f"Spearman rho = {rho:+.2f}", transform=ax.transAxes, va="bottom", ha="right",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"))
            if i == 0:
                ax.set_title(title, fontweight="bold")
            if i == len(FOLDS) - 1:
                ax.set_xlabel("MSE percentile rank among the candidates (100 = lowest MSE)")
            ax.set_ylabel(ylab)
        row_axes[0].annotate(f"Test year {y}\n(validation {y - 1})", xy=(-0.28, 0.5), xycoords="axes fraction",
                             rotation=90, va="center", ha="center", fontsize=11, fontweight="bold")
    fig.legend(handles=[Line2D([], [], ls="", marker="o", ms=6, color=GREY, label="candidate genomes (MO Pareto archives)"),
                        Line2D([], [], color=NAVY, marker="o", ms=5, label="validation Pareto frontier"),
                        Line2D([], [], ls="", marker="s", ms=7, mfc="white", mec=GREEN, mew=1.5, label="frontier ends"),
                        Line2D([], [], ls="", marker="*", ms=14, color=RED, label="chosen genome (pre-set rule)")],
               loc="lower center", ncol=4, frameon=False)
    fig.suptitle(f"Multi-objective EXAMM: MSE vs {YSHORT} of {RULE[0]} (one genome for all 4 panel sets)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0.03, 0.04, 1, 0.97))
    fig.savefig(f"{OUT}.png", dpi=160)
    print(f"\nSaved {OUT}.png")
except ImportError:
    print("\nmatplotlib not installed -- skipped the figure (pip3 install --user matplotlib)")

# ---------------------------------------------------------------- workbook ----
try:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
except ImportError:
    print(f"Saved {OUT}.csv, _headline.csv, _by_year.csv (openpyxl not installed -- no .xlsx)")
    sys.exit(0)

names = {"strategy": "Strategy (genome chosen on validation)", "ann_return_%": "Annual return (%)",
         "ann_return_ci_lo": "Return 95% low", "ann_return_ci_hi": "Return 95% high", "ann_vol_%": "Volatility (%)",
         "sharpe": "Sharpe", "sharpe_ci_lo": "Sharpe 95% low", "sharpe_ci_hi": "Sharpe 95% high", "psr": "PSR",
         "max_dd_%": "Max drawdown (%)", "calmar": "Calmar", "ulcer_%": "Ulcer index (%)", "martin": "Martin ratio",
         "alpha_%": "Residual alpha vs reversal (%/yr)", "beta": "Beta on reversal", "ir": "Information ratio",
         "test_mse": "Test MSE", "genomes": "Genomes used"}
summary = head.reset_index()[[c for c in names if c in head.reset_index()]].rename(columns=names)
per_year = []
for name, g in by_year.groupby("strategy", sort=False):
    g = g.set_index("year")
    per_year.append({"Strategy": name, **{f"{y} return (%)": g.loc[y, "ann_return_%"] for y in FOLDS},
                     **{f"{y} Sharpe": g.loc[y, "sharpe"] for y in FOLDS},
                     **({f"{y} information ratio": g.loc[y, "ir"] for y in FOLDS} if "ir" in g else {}),
                     **{f"{y} max drawdown (%)": g.loc[y, "max_dd_%"] for y in FOLDS},
                     **{f"{y} genome": g.loc[y, "genome"] for y in FOLDS}})
per_year = pd.DataFrame(per_year)
gcols = {"fold": "Test year", "source_set": "Trained on", "repeat": "Repeat", "genome_id": "Genome id",
         "reason": "Why a candidate", "val_mse": "Val MSE", "val_mse_pct": "Val MSE percentile (100 = lowest)",
         "val_rank_ic": "Val Rank IC", "val_return_%": "Val return (%)", "val_sharpe": "Val Sharpe",
         "val_ir": "Val information ratio", "val_beta": "Val beta on reversal", "val_max_dd_%": "Val max drawdown (%)",
         "val_martin": "Val Martin", "test_mse": "Test MSE", "test_rank_ic": "Test Rank IC",
         "test_return_%": "Test return (%)", "test_sharpe": "Test Sharpe", "test_ir": "Test information ratio",
         "test_beta": "Test beta on reversal", "test_max_dd_%": "Test max drawdown (%)", "test_martin": "Test Martin",
         "val_frontier": "On val frontier", "chosen": "Picked as"}
genomes = table[[c for c in gcols if c in table]].rename(columns=gcols).sort_values(["Test year", "Val MSE"])
genomes["On val frontier"] = genomes["On val frontier"].map({True: "Yes", False: ""})
notes = [
    f"Multi-objective EXAMM runs ({RUN_DIR}_<Y>): the search kept and bred genomes by Pareto rank on validation MSE and "
    + ("validation information ratio (daily L/S 5/5 book vs the 10-day z-score reversal strategy, before costs)."
       if RUN == "ir" else "validation return of a daily L/S 5/5 book (before costs)."),
    "Candidates per fold: per source set, the genomes no other genome of its 10 runs' Pareto archives dominated, plus "
    "each run's lowest-validation-MSE genome (select_core7_mo_candidates.py).",
    f"Re-scored here as ONE genome for all 4 sets in raw daily returns: MSE, Rank IC; trading with {RULE[0]}, costs on, "
    "the 4 sets combined with equal weight; information ratio, beta and residual alpha against the reversal strategy "
    "traded the same way.",
    f"Validation frontier: lower MSE vs higher {YSHORT}. Choice fixed in advance: frontier genome with the lowest MSE rank "
    "+ second-axis rank; ties -> lower validation MSE. Frontier ends for comparison only.",
    "MSE-only rows: the ordinary EXAMM runs' genome-frontier choice (MSE vs return), lowest-MSE genome and per-set "
    "default, from core7_genome_frontier.py with the same rule and costs.",
    "Intervals: 95% stationary block bootstrap; PSR = P(true Sharpe > 0). Green = Sharpe interval above zero; red = "
    "negative Sharpe; yellow = genomes picked on validation.",
    "Caveats: validation drives the search and the choice; the in-search objective is before costs; one validation "
    "year is noisy; 2022-2024 examined before in this project; three test years.",
]
with pd.ExcelWriter(f"{OUT}.xlsx", engine="openpyxl") as xw:
    for name, df in (("Summary", summary), ("Per year", per_year), ("Genomes", genomes)):
        df.to_excel(xw, sheet_name=name, index=False, startrow=1)
    pd.DataFrame({"Notes": notes}).to_excel(xw, sheet_name="Notes", index=False)
F = "Arial"
HDR_FILL, HDR_FONT = PatternFill("solid", fgColor="1F3864"), Font(name=F, bold=True, color="FFFFFF")
GOOD, BAD, PICK = (PatternFill("solid", fgColor=c) for c in ("E2EFDA", "FBE5E5", "FFF2CC"))
EDGE = Side(style="thin", color="BFBFBF")
BORDER = Border(left=EDGE, right=EDGE, top=EDGE, bottom=EDGE)
titles = {"Summary": f"Multi-objective EXAMM (MSE vs {YSHORT}): test {FOLDS[0]}-{FOLDS[-1]} joined, {RULE[0]}, costs on",
          "Per year": "Test results per year", "Genomes": "All candidate genomes: validation and test scores"}
wb = load_workbook(f"{OUT}.xlsx")
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
                                   else "+0.0000;-0.0000" if "IC" in h else "0.00")
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
wb.save(f"{OUT}.xlsx")
print(f"Saved {OUT}.csv, _headline.csv, _by_year.csv, .xlsx")
