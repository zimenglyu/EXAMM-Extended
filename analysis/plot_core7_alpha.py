#!/usr/bin/env python3
"""Your own version of the paper's Figure 1 / Table 6: annualized NET alpha with 95% confidence intervals,
computed the way the paper's Sec. 5.3 and 6.4 describe.

Alpha = intercept of the daily regression   r_strategy = alpha + b1 MKT + b2 STR1 + b3 STR5 + e,   x 252, in %.
  r_strategy: the main workflow's daily net returns (paper's Algorithm 1, costs on, per-set genome = lowest val MSE),
              read from the backtest cache -- nothing is retrained or re-traded
  MKT:  equal-weight mean daily return of the panel's own 50 stocks
  STR1: one-day reversal, long the K lowest / short the K highest stocks by their latest daily return
  STR5: the same on the trailing five-day compounded return (K = the book size, 10 by default, as the paper)
  The first 5 days of each test year are dropped (STR5 warm-up), for every specification.
Standard errors: per panel (V1-V4) Newey-West; for the 4 sets the regression is estimated on the STACKED panel
(every panel-day one row, each panel with its own factors) with Driscoll-Kraay standard errors, as the paper does
(robust to serial correlation and to correlation across the 4 panels on the same day).
Specifications (Table 6): MKT, MKT+STR1, MKT+STR1+STR5; the figure uses --spec (default full = MKT+STR1+STR5, the
paper's Figure 1). The figure also shows each alpha's t-value and the t distribution expected if alpha were 0.
Designs: retrain every year (the paper's design) and never retrain (2022 models for 2022-2024).
Options: --experiment paper6 for the paper-input runs, --k 5 for the 5/5 backtest (reads the _k5 results), --spec mkt|str1|full.
Reads results/core7_main[_k5]_per_set.csv and results/genome_frontier/stale_trades_paper[_k5].npz -- run
core7_main_results.py first. Writes results/core7_main[_k5]_alpha.png and _alpha.csv.
Run on the Mac:  cd ~/Financial_toolbox && python3 plot_core7_alpha.py && open results/core7_main_alpha.png
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv[:-1] else 10
SPEC = sys.argv[sys.argv.index("--spec") + 1].lower() if "--spec" in sys.argv[:-1] else "full"
SPECS = {"mkt": ["MKT"], "str1": ["MKT", "STR1"], "full": ["MKT", "STR1", "STR5"]}
SPEC_NAME = {"mkt": "MKT", "str1": "MKT + STR1", "full": "MKT + STR1 + STR5"}
if SPEC not in SPECS:
    raise SystemExit("usage: python3 plot_core7_alpha.py [--k 10|5] [--spec mkt|str1|full]")
EXP = sys.argv[sys.argv.index("--experiment") + 1] if "--experiment" in sys.argv[:-1] else "core7"   # run family
TAG = ("" if K == 10 else f"_k{K}") + ("" if EXP == "core7" else f"_{EXP}")
CACHE = RESULTS / ("genome_frontier" if EXP == "core7" else f"genome_frontier_{EXP}") / f"stale_trades_paper{'' if K == 5 else f'_k{K}'}.npz"
PER_SET = RESULTS / f"core7_main{TAG}_per_set.csv"
SETS = ["set1", "set2", "set3", "set4"]
YEARS = [2022, 2023, 2024]
DAYS, WARMUP = 252, 5

for f in (PER_SET, CACHE):
    if not f.exists():
        raise SystemExit(f"missing {f} -- run: python3 core7_main_results.py" + ("" if K == 10 else f" --k {K}") + ("" if EXP == "core7" else f" --experiment {EXP}"))
per_set = pd.read_csv(PER_SET)
cache = np.load(CACHE)
_fac = {}


def factors(year, s):
    """Daily MKT, STR1, STR5 of one panel and year, aligned with the strategy's daily returns (day t -> t+1)."""
    if (year, s) not in _fac:
        d = CACHE.parent / str(year) / f"prices_{s}_test"
        files = sorted(d.glob("*.csv"))
        if len(files) != 50:
            raise SystemExit(f"missing price files in {d} -- run: python3 core7_main_results.py")
        prc = np.column_stack([pd.read_csv(f, usecols=["PRC"])["PRC"].to_numpy() for f in files])
        R = prc[1:] / prc[:-1] - 1                       # R[t] = return from day t to t+1
        T = len(R)
        mkt = R.mean(axis=1)
        str1, str5 = np.full(T, np.nan), np.full(T, np.nan)
        for t in range(1, T):
            order = np.argsort(R[t - 1], kind="stable")  # latest known daily return, ascending
            str1[t] = R[t, order[:K]].mean() - R[t, order[-K:]].mean()
            if t >= 5:
                trail = np.prod(1 + R[t - 5:t], axis=0) - 1
                order = np.argsort(trail, kind="stable")
                str5[t] = R[t, order[:K]].mean() - R[t, order[-K:]].mean()
        _fac[(year, s)] = {"MKT": mkt, "STR1": str1, "STR5": str5}
    return _fac[(year, s)]


def nw_lags(T):
    return int(math.floor(4 * (T / 100) ** (2 / 9)))


def regress(blocks, cols):
    """Pooled OLS over blocks [(y, X, date_index)]; Driscoll-Kraay SEs (= Newey-West when there is one block)."""
    y = np.concatenate([b[0] for b in blocks])
    X = np.vstack([b[1] for b in blocks])
    dates = np.concatenate([b[2] for b in blocks])
    XtX_inv = np.linalg.inv(X.T @ X)
    coef = XtX_inv @ X.T @ y
    e = y - X @ coef
    u = X * e[:, None]
    uniq, inv = np.unique(dates, return_inverse=True)
    h = np.zeros((len(uniq), X.shape[1]))
    np.add.at(h, inv, u)                                  # sum of moment conditions per date
    L = nw_lags(len(uniq))
    S = h.T @ h
    for lag in range(1, L + 1):
        G = h[lag:].T @ h[:-lag]
        S += (1 - lag / (L + 1)) * (G + G.T)
    se = np.sqrt(np.diag(XtX_inv @ S @ XtX_inv))
    a, se_a = coef[0] * DAYS * 100, se[0] * DAYS * 100
    out = {"alpha_%": a, "ci_lo_%": a - 1.96 * se_a, "ci_hi_%": a + 1.96 * se_a,
           "t_stat": coef[0] / se[0] if se[0] > 0 else np.nan, "days": len(uniq), "rows": len(y)}
    for name, c in zip(cols, coef[1:]):
        out[f"beta_{name}"] = c
    return out


designs = {"retrain every year": lambda y: y, "never retrain (2022 models)": lambda y: YEARS[0]}
rows, missing = [], []
for name, fold_of in designs.items():
    series = {}                                          # (set, year) -> strategy returns and factors, same days
    for s in SETS:
        for y in YEARS:
            sel = per_set[(per_set["set"] == s) & (per_set["test_year"] == y) & (per_set["genomes_from_fold"] == fold_of(y))]
            if sel.empty:
                missing.append(f"{name} {s} {y}")
                continue
            key = f"{fold_of(y)}|{s}|{sel['repeat'].iloc[0]}|{y}"
            if key not in cache:
                missing.append(key)
                continue
            f = factors(y, s)
            n = min(len(cache[key]), len(f["MKT"]))
            series[(s, y)] = {"r": np.asarray(cache[key][:n], float)[WARMUP:],
                              **{k: v[:n][WARMUP:] for k, v in f.items()}}
    groups = [(f"V{i + 1}", [s]) for i, s in enumerate(SETS)] + [("4 sets combined", SETS)]
    for label, sets in groups:
        for period in YEARS + ["2022-2024"]:
            years = YEARS if period == "2022-2024" else [period]
            if not all((s, y) in series for s in sets for y in years):
                continue
            for spec, cols in SPECS.items():
                blocks = []
                for s in sets:
                    r, X, d = [], [], []
                    for y in years:
                        z = series[(s, y)]
                        n = len(z["r"])
                        r.append(z["r"])
                        X.append(np.column_stack([np.ones(n)] + [z[c] for c in cols]))
                        d.append(y * 1000 + np.arange(n))      # day index shared across panels within a year
                    blocks.append((np.concatenate(r), np.vstack(X), np.concatenate(d)))
                rows.append({"design": name, "portfolio": label, "period": str(period), "spec": SPEC_NAME[spec],
                             **regress(blocks, cols)})

if missing:
    print("!! left out (not in the backtest results): " + ", ".join(missing[:8]))
allres = pd.DataFrame(rows)
if allres.empty:
    raise SystemExit("nothing to plot")
allres.to_csv(RESULTS / f"core7_main{TAG}_alpha.csv", index=False)
res = allres[allres["spec"] == SPEC_NAME[SPEC]].copy()
res["beta"] = res["beta_MKT"]
pd.set_option("display.width", 220)
show = ["design", "spec", "alpha_%", "ci_lo_%", "ci_hi_%", "t_stat", "beta_MKT", "beta_STR1", "beta_STR5", "days"]
print(f"Table 6 style -- 4 sets stacked, 2022-2024, Driscoll-Kraay (L/S {K}/{K}); "
      "paper EXAMM: MKT +11.27 (t 2.32), +STR1 +10.96 (t 2.24), +STR1+STR5 +10.64 (t 2.16):")
t6 = allres[(allres["portfolio"] == "4 sets combined") & (allres["period"] == "2022-2024")]
print(t6[[c for c in show if c in t6]].round(3).to_string(index=False))
print(f"\nPer panel, 2022-2024, {SPEC_NAME[SPEC]} (Newey-West):")
print(res[res["period"] == "2022-2024"][["design", "portfolio", "alpha_%", "ci_lo_%", "ci_hi_%", "t_stat"]].round(2)
      .to_string(index=False))
print(f"\nPer test year, 4 sets stacked, {SPEC_NAME[SPEC]}:")
print(res[(res["portfolio"] == "4 sets combined") & (res["period"] != "2022-2024")]
      [["design", "period", "alpha_%", "ci_lo_%", "ci_hi_%", "t_stat"]].round(2).to_string(index=False))

# ---------------------------------------------------------------- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"retrain every year": "#1f3864", "never retrain (2022 models)": "#e59866"}
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, width_ratios=[5, 3], height_ratios=[1, 0.9])
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
for ax, (sub, cats, title) in zip(axes, [
        (res[res["period"] == "2022-2024"], ["V1", "V2", "V3", "V4", "4 sets combined"],
         "2022-2024 joined, per panel"),
        (res[(res["portfolio"] == "4 sets combined") & (res["period"] != "2022-2024")], [str(y) for y in YEARS],
         "Per test year, 4 sets combined")]):
    key = "portfolio" if "per panel" in title else "period"
    x = np.arange(len(cats))
    width = 0.36
    for k, name in enumerate(designs):
        d = sub[sub["design"] == name].set_index(key).reindex(cats)
        pos = x + (k - 0.5) * width
        ax.bar(pos, d["alpha_%"], width, color=COL[name], label=name, zorder=3)
        ax.errorbar(pos, d["alpha_%"], yerr=[d["alpha_%"] - d["ci_lo_%"], d["ci_hi_%"] - d["alpha_%"]],
                    fmt="none", ecolor="black", capsize=4, lw=1.2, zorder=4)
        for xp, (a, hi, lo, t) in zip(pos, d[["alpha_%", "ci_hi_%", "ci_lo_%", "t_stat"]].to_numpy()):
            if np.isfinite(t):                         # t-value written above the interval
                ax.annotate(f"t={t:+.2f}", (xp, hi if a >= 0 else lo), xytext=(0, 4 if a >= 0 else -12),
                            textcoords="offset points", ha="center", fontsize=8,
                            fontweight="bold" if abs(t) >= 1.96 else "normal")
    ax.axhline(0, color="#555555", lw=0.9, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="y", color="#e6e6e6", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_ylabel("Annualized net alpha (%)")
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.925))

# t distribution under "true alpha = 0" (faint, behind), and every portfolio's t-value as a dot on its own row
from scipy.stats import t as t_dist
ax = fig.add_subplot(gs[1, :])
dof = int(res["days"].min()) - 2
crit = t_dist.ppf(0.975, dof)
rows_order = ["V1", "V2", "V3", "V4", "all 4 sets"] + [f"all 4 sets, {y}" for y in YEARS]
marks = res[(res["period"] == "2022-2024") | (res["portfolio"] == "4 sets combined")].copy()
marks["row"] = np.where(marks["period"] == "2022-2024", marks["portfolio"].str.replace("4 sets combined", "all 4 sets"),
                        "all 4 sets, " + marks["period"])
ypos = {r: len(rows_order) - 1 - i for i, r in enumerate(rows_order)}
lim = max(4.0, float(np.nanmax(np.abs(marks["t_stat"]))) + 0.5)
grid = np.linspace(-lim, lim, 600)
dens = t_dist.pdf(grid, dof)
scale = (len(rows_order) - 0.4) / dens.max()
ax.fill_between(grid, -0.6, -0.6 + dens * scale, color="#d9d9d9", alpha=0.55, zorder=1,
                label=f"t distribution if alpha = 0 (df = {dof})")
ax.axvspan(-lim, -crit, color="#c0392b", alpha=0.07, zorder=0)
ax.axvspan(crit, lim, color="#c0392b", alpha=0.07, zorder=0, label=f"significant at 5% (|t| > {crit:.2f})")
for v in (-crit, crit):
    ax.axvline(v, color="#c0392b", ls="--", lw=1, zorder=2)
ax.axvline(0, color="#555555", lw=0.8, zorder=2)
for k, name in enumerate(designs):
    d = marks[marks["design"] == name]
    off = 0.14 if k == 0 else -0.14
    yy = [ypos[r] + off for r in d["row"]]
    ax.scatter(d["t_stat"], yy, s=55, color=COL[name], zorder=4, label=name, edgecolor="white", lw=0.6)
    for tv, y in zip(d["t_stat"], yy):
        ax.annotate(f"{tv:+.2f}", (tv, y), xytext=(7, -3), textcoords="offset points", fontsize=8, color=COL[name],
                    fontweight="bold" if abs(tv) >= crit else "normal", zorder=5)
ax.set_yticks([ypos[r] for r in rows_order])
ax.set_yticklabels([r if r.startswith("all 4 sets,") else (f"{r} (2022-24)") for r in rows_order])
ax.set_ylim(-0.6, len(rows_order) - 0.3)
ax.set_xlim(-lim, lim)
ax.set_xlabel("t-statistic of alpha (alpha / its Newey-West or Driscoll-Kraay standard error)")
ax.set_title("t-value of each alpha on the t distribution expected if the true alpha were 0 "
             "(shaded red = significant at 5%)", fontweight="bold", fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=8, ncol=1)
ax.grid(axis="x", color="#eeeeee", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"Annualized net alpha with 95% confidence intervals\n{SPEC_NAME[SPEC]}; L/S {K}/{K}, paper's trading rule, costs on; 4 sets stacked with Driscoll-Kraay errors", fontweight="bold")
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = RESULTS / f"core7_main{TAG}_alpha.png"
fig.savefig(out, dpi=160)
print(f"\nSaved {out} and {RESULTS / f'core7_main{TAG}_alpha.csv'}")
