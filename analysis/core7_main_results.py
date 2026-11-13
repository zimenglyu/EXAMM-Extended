#!/usr/bin/env python3
"""MAIN WORKFLOW, step 2 (backtest and report). Renamed from the stale-model script.
BOOK SIZE: long K / short K with K = 10 by default, the paper's setting (Sec. 5.2: "the top 10 stocks ... the bottom
10"); --k 5 gives the earlier 5/5 results (outputs then end in _k5, trades cached apart). The paper's IC in its
Tables 4 and 8 (+0.0106) is the PEARSON IC, so compare it with the "IC (Pearson)" columns.
PAPER-STYLE REPORTING: per test year the TOTAL return over the test period ((C - C0) / C0, the paper's "net
return", Algorithm 1 line 17) besides the annualized one; your own buy-and-hold and equal weight per panel and year;
a Table 8 style summary (per panel, then averaged) with Rank IC as Spearman and as Pearson; the chosen models' sizes.
Extra outputs: core7_main_benchmarks.csv, core7_main_table8.csv, core7_main_model_size.csv.
--pareto: also chooses each panel set's model by a validation Pareto frontier (Option A): per test year Y and set,
its 10 EXAMM repeats scored on the VALIDATION year Y-1 by MSE (raw daily returns) and net return of the same
trading rule; frontier = repeats no other repeat beats on both; choice fixed in advance = lowest MSE rank + return
rank (ties -> lower MSE); traded on the test year exactly like the main result (retrain every year). Reported next
to the lowest-validation-MSE choice (the main result), the highest-validation-return end and the AVERAGE OVER
THE 10 RUNS (no choice; the expected result of an arbitrary run). Outputs
core7_main[_everyday]_pareto.csv (every repeat) and _pareto_summary.csv, and core7_main_pareto.png.
Also the NEVER-RETRAIN version (the fold-2022 choice kept for 2022-2024): _pareto_never_retrain.csv. It needs the
other repeats' later-year forecasts: ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh
MEAN OVER THE 10 SEEDS (on by default, --no-mean-repeats to skip): the PAPER'S REPORTING ("every configuration is repeated 10 times with independent seeds, and we
report the mean over the repeats"): every one of the 10 repeats per panel set is traded on its own (no choice by
validation MSE), each year's metrics are computed per repeat, then averaged over the 10 repeats (Tables 5, 7, 8 style),
for retrain every year and never retrain. Needs every repeat's forecasts: evaluation_test/ (standard evaluation) and,
for never retrain, ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh. Outputs core7_main_repeats.csv
(every repeat x year), _repeats_table7.csv and _repeats_table8.csv.
DEFAULTS: the paper's rule (--trade-every-day to switch it off) and the per-set tables (--no-per-set);
the --paper-rule and --per-set descriptions below are now always on.

Stale-model check: does a Pooled EXAMM RET model still trade well in years after the one it was made for?

Standard EXAMM (as main_core7_protocol.py): each fold F = test year (train <= F-2, validation F-1), each
panel set's best genome = the repeat with the lowest validation MSE, traded on its own 50 stocks with
daily L/S 5/5 (long the 5 highest forecasts, short the 5 lowest, rebalanced daily), costs on, the 4 sets
combined with equal weight. Here every fold's best genomes are ALSO traded on the later test years:

    genomes of fold ->   tested on 2022   2023              2024
    2022 (<= 2020, val 2021)   standard   1 year stale       2 years stale
    2023 (<= 2021, val 2022)   -          standard           1 year stale
    2024 (<= 2022, val 2023)   -          -                  standard

Cells below the diagonal are computed only with --all: those genomes trained or were chosen on that year,
so their results contain LEAKAGE (marked "LEAK"). Use them only to compare with a paper, never as a test result.
Reported per cell: test annual return, Sharpe, max drawdown, MSE (raw daily returns) and Rank IC; and,
2022-2024 joined with 95% block-bootstrap intervals and PSR, two books fixed in advance:
  - retrain every year (the standard result: the diagonal);
  - never retrain (the 2022-fold genomes for all three years).
No Pareto frontier is used. Nothing here chooses anything on the test years.

Run on the Mac after the evaluation:
    cd ~/EXAMM-Extended && sh scripts/rnn_examples/evaluate_rnn_core7_main.sh
    cd ~/Financial_toolbox && python3 core7_main_results.py --trade-every-day
    ALL=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh && python3 core7_main_results.py --trade-every-day --all   # all 9 cells
The standard (diagonal) cells reuse the evaluation_test folders the other scripts made; the stale cells use
evaluation_test_y<Y>. Trades are cached in results/genome_frontier/stale_trades.npz.
Outputs: results/core7_main_everyday.csv (every cell) and results/core7_main_everyday_headline.csv.

--per-set: also reports every panel set (V1-V4 = set1-set4) as its OWN portfolio and then averages the four,
as the paper's Tables 7-8 do ("average across V1-V4"), instead of combining the sets into one portfolio:
  Table 7 style: net return per test year, averaged over V1-V4, and their mean over the years;
  Table 8 style: per set over 2022-2024 joined -- Rank IC, net return (mean of the yearly ones), Sharpe,
                 max drawdown -- averaged over V1-V4.
For both books (retrain every year; never retrain). Outputs add core7_main_everyday_per_set.csv.
    python3 core7_main_results.py --trade-every-day --per-set            (add --all for the leakage cells too)

--paper-rule: trade with the paper's Algorithm 1 (paper_long_short.py) instead of portfolio.py's daily L/S 5/5:
the same long 5 / short 5, sizing q = C / K and costs, but a day is traded only if the 5th-highest forecast
is > 0 and the 5th-lowest < 0 (otherwise yesterday's positions are kept), with the forecasts converted to
raw daily returns first (prediction x IQR + median, each set's panel_meta.json) and the account marked to
market every day. Trades cached separately; outputs end in _paper (e.g. core7_main.csv).
    python3 core7_main_results.py
"""
import sys

ALL = "--all" in sys.argv
PARETO = "--pareto" in sys.argv
MEAN_REPEATS = "--no-mean-repeats" not in sys.argv   # mean over the 10 seeds, on by default
K = int(sys.argv[sys.argv.index("--k") + 1]) if "--k" in sys.argv[:-1] else 10   # long K / short K (paper: 10)
PER_SET = "--no-per-set" not in sys.argv
PAPER = "--trade-every-day" not in sys.argv
OUT_TAG = ("" if PAPER else "_everyday") + ("" if K == 10 else f"_k{K}")
_argv, sys.argv = sys.argv, [sys.argv[0], "--rule", "ls55"]   # core7_genome_frontier reads --rule at import
import numpy as np
import pandas as pd

from core7_genome_frontier import (EXAMM, RESULTS, SETS, WORK, YEARS, bootstrap_ci, combine, daily_returns,
                                   final_genome, forecast_scores, metrics, prediction_dir, price_dir, psr,
                                   pareto, read_predictions, ret_stats, scored_mask)
sys.argv = _argv
RULE = (f"daily L/S {K}/{K}", "daily_long_short_return", dict(long_company_number=K, short_company_number=K))

REPEATS = [str(r) for r in range(15, 25)]
TRADE_DAYS = {}                      # (fold, year, set) -> (days traded, days) under --paper-rule


def paper_returns(pred_d, price_d):
    """Daily returns of the paper's Algorithm 1 on one set (paper_long_short.py), costs on."""
    import contextlib
    import io
    from Logger import Logger
    from paper_long_short import paper_long_short_daily_values
    from portfolio import Portfolio
    from stock import Stock
    names = sorted(f.name[:-len("_predictions.csv")] for f in pred_d.glob("*_predictions.csv"))
    logger = Logger()
    logger.set_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio = Portfolio(names)
        portfolio.set_initial_spend_per_stock(100)
        portfolio.set_initial_capital(1000)
        portfolio.set_logger(logger)
        for n in names:
            c = Stock(n)
            c.read_return_prediction(str(pred_d / f"{n}_predictions.csv"))
            c.read_stock_price(str(price_d / f"{n}.csv"))
            portfolio.add_company_to_protfolio(c)
            c.set_logger(logger)
        portfolio.set_use_TC(True)
        portfolio.set_trade_with_bid_ask(False)
        values, trades = paper_long_short_daily_values(portfolio, K, K, sign_check=True)
    return values[1:] / values[:-1] - 1, trades


def total_return(r):
    """Total return over the period in % ((C - C0) / C0, the paper's net return)."""
    return (float(np.prod(1 + np.asarray(r, dtype=float))) - 1) * 100


def pearson_ic(P, year, s):
    """Mean daily cross-sectional PEARSON correlation of the raw forecast with the raw next-day return."""
    st = ret_stats(s)
    rows, keep = scored_mask(year, s, "test", len(P))
    raw = (P * st["iqr"] + st["median"])[keep].reset_index(drop=True)
    actual = pd.DataFrame({t: rows[f"RET_raw_{t}"].to_numpy() for t in P.columns})[keep].reset_index(drop=True)
    return float(raw.corrwith(actual, axis=1).dropna().mean())


def buy_and_hold_total(price_d):
    """Buy-and-hold over the test year: equal dollars in every stock at the first price + cost, sold at the last
    price - cost; total return in %."""
    out = []
    for f in sorted(price_d.glob("*.csv")):
        d = pd.read_csv(f)
        out.append((d["PRC"].iloc[-1] - d["TRAN_COST"].iloc[-1]) / (d["PRC"].iloc[0] + d["TRAN_COST"].iloc[0]) - 1)
    return float(np.mean(out)) * 100


def model_size(fold, s, r):
    """Enabled nodes, edges and recurrent edges of the repeat's final global best (last fitness_log.csv row)."""
    log = pd.read_csv(EXAMM / "test_output" / f"core7_examm_ret_{fold}" / s / r / "fitness_log.csv")
    last = log.iloc[-1]
    return {"enabled_nodes": int(last.iloc[5]), "enabled_edges": int(last.iloc[6]), "enabled_recurrent_edges": int(last.iloc[7])}


def best_repeat(fold, s):
    """The repeat with the lowest validation MSE (last fitness_log.csv row, column 5), as main_core7.py."""
    best = None
    for r in REPEATS:
        log = EXAMM / "test_output" / f"core7_examm_ret_{fold}" / s / r / "fitness_log.csv"
        if log.exists():
            mse = float(pd.read_csv(log).iloc[-1, 4])
            if best is None or mse < best[1]:
                best = (r, mse)
    return best


def pred_dir(fold, s, r, year):
    d = EXAMM / "test_output" / f"core7_examm_ret_{fold}" / s / r
    return d / ("evaluation_test" if year == fold else f"evaluation_test_y{year}")


WORK.mkdir(parents=True, exist_ok=True)
cache_path = WORK / ((("stale_trades_paper" if PAPER else "stale_trades") + ("" if K == 5 else f"_k{K}")) + ".npz")   # per book size
cache = dict(np.load(cache_path)) if cache_path.exists() else {}

cells, books, missing, set_rows, set_books = [], {}, [], [], {}
bench, sizes = {}, []                # (year, set) -> buy-and-hold / equal weight; chosen models' sizes
for fold in YEARS:
    chosen = {s: best_repeat(fold, s) for s in SETS}
    for s in SETS:
        if chosen[s] is not None:
            sizes.append({"fold": fold, "set": s, "repeat": chosen[s][0], **model_size(fold, s, chosen[s][0])})
    for year in [y for y in YEARS if ALL or y >= fold]:
        per, mses, ics, ics_p, used = [], [], [], [], []
        for s in SETS:
            if chosen[s] is None:
                missing.append(f"fold {fold} {s}: no fitness_log.csv")
                break
            r = chosen[s][0]
            P = read_predictions(pred_dir(fold, s, r, year), "test")
            if P is None or not np.isfinite(P.to_numpy()).all():
                missing.append(f"fold {fold} {s}/{r} on {year}: no complete predictions in {pred_dir(fold, s, r, year).name}")
                break
            mse, ic = forecast_scores(P, year, s, "test")
            ic_p = pearson_ic(P, year, s)
            mses.append(mse)
            ics.append(ic)
            ics_p.append(ic_p)
            key = f"{fold}|{s}|{r}|{year}"
            if PAPER:
                st = ret_stats(s)                      # robust z -> raw daily return, for the sign check
                if key not in cache or f"{key}|trades" not in cache:
                    ret, n_trades = paper_returns(prediction_dir(year, f"paper{fold}_{s}", r, s, "test",
                                                                 P * st["iqr"] + st["median"]),
                                                  price_dir(year, s, "test"))
                    cache[key], cache[f"{key}|trades"] = ret, np.array([n_trades, len(ret)])
                TRADE_DAYS[(fold, year, s)] = tuple(int(x) for x in cache[f"{key}|trades"])
            elif key not in cache:
                cache[key] = daily_returns(prediction_dir(year, f"stale{fold}_{s}", r, s, "test", P),
                                           price_dir(year, s, "test"), RULE[1], RULE[2])
            per.append(cache[key])
            set_books[(fold, year, s)] = cache[key]
            if (year, s) not in bench:      # your own benchmarks for this panel and year (independent of the model)
                ew_key = f"EW|{year}|{s}"
                if ew_key not in cache:
                    cache[ew_key] = daily_returns(prediction_dir(year, f"stale{fold}_{s}", r, s, "test", P),
                                                  price_dir(year, s, "test"), "daily_equal_weight_return", {})
                bench[(year, s)] = {"bh_total_%": buy_and_hold_total(price_dir(year, s, "test")),
                                    "ew_total_%": total_return(cache[ew_key])}
            ms = metrics(cache[key])
            set_rows.append({"genomes_from_fold": fold, "test_year": year, "set": s, "repeat": r, "leakage": year < fold,
                             "total_return_%": total_return(cache[key]),
                             "ann_return_%": ms["ann_return_%"], "sharpe": ms["sharpe"], "max_dd_%": ms["max_dd_%"],
                             "test_mse": mse, "test_rank_ic": ic, "test_ic_pearson": ic_p})
            used.append(f"{s}/{r}/{final_genome(EXAMM / 'test_output' / f'core7_examm_ret_{fold}' / s / r)}")
        if len(per) < len(SETS):
            continue
        book = combine(per)
        books[(fold, year)] = book
        m = metrics(book)
        cells.append({"genomes_from_fold": fold, "test_year": year, "years_stale": year - fold,
                      "leakage": year < fold,
                      "total_return_%": total_return(book),
                      "ann_return_%": m["ann_return_%"], "sharpe": m["sharpe"], "max_dd_%": m["max_dd_%"],
                      "test_mse": float(np.mean(mses)), "test_rank_ic": float(np.mean(ics)), "test_ic_pearson": float(np.mean(ics_p)),
                      "days_traded_%": (float(np.mean([TRADE_DAYS[(fold, year, s)][0] / TRADE_DAYS[(fold, year, s)][1]
                                                       for s in SETS])) * 100 if PAPER else np.nan),
                      "genomes": ", ".join(used)})
np.savez(cache_path, **cache)

if missing:
    print("!! left out:\n   " + "\n   ".join(missing))
if not cells:
    sys.exit("nothing to report -- run evaluate_rnn_core7_main.sh (and the standard evaluation) first")
table = pd.DataFrame(cells)
RESULTS.mkdir(exist_ok=True)
table.to_csv(RESULTS / f"core7_main{OUT_TAG}.csv", index=False)

pd.set_option("display.width", 200)
print("=" * 96)
print(f"STALE MODELS -- Pooled RET, per-set best genome (lowest val MSE), "
      f"{f'PAPER Algorithm 1 (L/S {K}/{K}, trade only if signs agree)' if PAPER else RULE[0]}, costs on, 4 sets combined")
print("rows: which fold's genomes; columns: test year; " + ("cells marked * contain LEAKAGE (the genomes trained or "
      "were chosen on that year)" if ALL else "blank = not computed (the genomes saw that year; --all to include)"))
print("=" * 96)
for col, name, nd in (("sharpe", "Test Sharpe", 2), ("total_return_%", "Test net (total) return (%)", 1),
                      ("ann_return_%", "Test annual return (%)", 1),
                      ("test_rank_ic", "Test Rank IC", 4), ("test_mse", "Test MSE (raw daily returns)", 7)):
    grid = table.pivot(index="genomes_from_fold", columns="test_year", values=col).reindex(index=YEARS, columns=YEARS)
    grid.index = [f"fold {f} genomes" for f in grid.index]
    shown = grid.round(nd).astype(object)
    for f in YEARS:
        for y in YEARS:
            if y < f and pd.notna(grid.loc[f"fold {f} genomes", y]):
                shown.loc[f"fold {f} genomes", y] = f"{grid.loc[f'fold {f} genomes', y]:.{nd}f}*"
    print(f"\n{name}:")
    print(shown.to_string(na_rep="-"))

head = []
for name, pairs in (("retrain every year (standard, the diagonal)", [(y, y) for y in YEARS]),
                    (f"never retrain (fold {YEARS[0]} genomes for {YEARS[0]}-{YEARS[-1]})", [(YEARS[0], y) for y in YEARS])):
    if all(p in books for p in pairs):
        r = np.concatenate([books[p] for p in pairs])
        head.append({"book": name, **metrics(r), **bootstrap_ci(r), "psr": psr(r)})
if head:
    head = pd.DataFrame(head).set_index("book")
    head.to_csv(RESULTS / f"core7_main{OUT_TAG}_headline.csv")
    print(f"\n{YEARS[0]}-{YEARS[-1]} joined (both books fixed in advance):")
    print(head[["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "sharpe", "sharpe_ci_lo", "sharpe_ci_hi",
                "psr", "max_dd_%"]].round(2).to_string())
if ALL:
    print("\n* = LEAKAGE: not a valid test result; for comparison with the paper only.")
if PAPER and TRADE_DAYS:
    td = pd.DataFrame([{"fold": f, "year": y, "set": s, "days traded": a, "days": b}
                       for (f, y, s), (a, b) in TRADE_DAYS.items()])
    td["share traded (%)"] = td["days traded"] / td["days"] * 100
    print("\nDays traded under the paper's rule (share of test days, averaged over the 4 sets):")
    print(td.pivot_table(index="fold", columns="year", values="share traded (%)").reindex(index=YEARS, columns=YEARS)
          .round(1).to_string(na_rep="-"))
print(f"\nSaved {RESULTS / f'core7_main{OUT_TAG}.csv'} and _headline.csv")

# ---------------------------------------------------------------- per-set mode (paper's Tables 7-8) ----
if PER_SET:
    per_set = pd.DataFrame(set_rows)
    per_set.to_csv(RESULTS / f"core7_main{OUT_TAG}_per_set.csv", index=False)
    table8_parts = []
    vname = {s: f"V{i + 1}" for i, s in enumerate(SETS)}
    print("\n" + "=" * 96)
    print("PER-SET MODE: each panel set (V1-V4) is its own portfolio, results averaged over V1-V4 (paper style)")
    print("=" * 96)
    for title, pairs in (("never retrain (fold %d genomes)" % YEARS[0], [(YEARS[0], y) for y in YEARS]),
                         ("retrain every year (standard)", [(y, y) for y in YEARS])):
        rows = per_set[per_set.apply(lambda r: (r["genomes_from_fold"], r["test_year"]) in pairs, axis=1)]
        if rows["test_year"].nunique() < len(YEARS) or rows.groupby("test_year")["set"].nunique().min() < len(SETS):
            print(f"\n{title}: incomplete -- skipped")
            continue
        grid = rows.pivot(index="test_year", columns="set", values="total_return_%").reindex(columns=SETS)
        grid.columns = [vname[c] for c in grid.columns]
        grid["Average V1-V4"] = grid.mean(axis=1)
        grid.loc["Mean"] = grid.mean(axis=0)
        print(f"\n{title} -- Table 7 style, net return (%):")
        print(grid.round(2).to_string())
        summary = []
        for s in SETS:
            r = np.concatenate([set_books[(f, y, s)] for f, y in pairs])
            m = metrics(r)
            summary.append({"design": title, "portfolio": vname[s],
                            "Rank IC (Spearman)": rows[rows["set"] == s]["test_rank_ic"].mean(),
                            "IC (Pearson)": rows[rows["set"] == s]["test_ic_pearson"].mean(),
                            "Net (%) mean of years": rows[rows["set"] == s]["total_return_%"].mean(),
                            "Sharpe (2022-24)": m["sharpe"], "MDD (%)": -m["max_dd_%"]})
        summary = pd.DataFrame(summary).set_index("portfolio")
        summary.loc["Average V1-V4"] = summary.drop(columns="design").mean(axis=0)
        summary.loc["Average V1-V4", "design"] = title
        table8_parts.append(summary.reset_index())
        print(f"\n{title} -- Table 8 style (per set, then averaged):")
        print(summary.drop(columns="design").round({"Rank IC (Spearman)": 4, "IC (Pearson)": 4, "Net (%) mean of years": 2,
                                                   "Sharpe (2022-24)": 2, "MDD (%)": 2}).to_string())
    print(f"\nSaved {RESULTS / f'core7_main{OUT_TAG}_per_set.csv'}")
    if table8_parts:
        pd.concat(table8_parts, ignore_index=True).to_csv(RESULTS / f"core7_main{OUT_TAG}_table8.csv", index=False)
        print("\nPaper (Table 8, L/S 10/10): IC (Pearson) +0.0106, Net +11.02 %, Sharpe 0.74, MDD -10.76 %")

# ---------------------------------------------------------------- your own benchmarks and model sizes ----
np.savez(cache_path, **cache)
if bench:
    bt = pd.DataFrame([{"test_year": y, "set": s, **v} for (y, s), v in bench.items()]).sort_values(["set", "test_year"])
    bt.to_csv(RESULTS / f"core7_main{OUT_TAG}_benchmarks.csv", index=False)
    print("\nYour benchmarks, total return over each test year (%), per panel (paper's Table 5 B&H / EW columns):")
    show = bt.pivot(index="test_year", columns="set", values="bh_total_%").round(2)
    show.columns = [f"B&H V{SETS.index(c) + 1}" for c in show.columns]
    show2 = bt.pivot(index="test_year", columns="set", values="ew_total_%").round(2)
    show2.columns = [f"EW V{SETS.index(c) + 1}" for c in show2.columns]
    print(pd.concat([show, show2], axis=1).to_string())
if sizes:
    sz = pd.DataFrame(sizes)
    sz["connections (edges + recurrent)"] = sz["enabled_edges"] + sz["enabled_recurrent_edges"]
    sz.to_csv(RESULTS / f"core7_main{OUT_TAG}_model_size.csv", index=False)
    print("\nChosen models' sizes (enabled nodes / edges / recurrent edges); paper's EXAMM: 66 parameters on average:")
    print(sz.round(1).to_string(index=False))
    print(f"average: {sz['enabled_nodes'].mean():.1f} nodes, {sz['connections (edges + recurrent)'].mean():.1f} connections "
          "(weights on connections; gated nodes such as LSTM/GRU add internal weights, so the paper's parameter count "
          "can be higher)")

# ---------------------------------------------------------------- --pareto: validation Pareto choice per set ----
if PARETO:
    def rule_returns(fold, s, r, year, split, P, tag):
        """Daily returns of the trading rule (paper's rule or trade every day) for one repeat, set and split."""
        key = f"{tag}|{fold}|{s}|{r}|{year}" if split == "val" else f"{fold}|{s}|{r}|{year}"
        if key not in cache:
            if PAPER:
                st = ret_stats(s)
                ret, _ = paper_returns(prediction_dir(year, f"{tag}{fold}_{s}", r, s, split, P * st["iqr"] + st["median"]),
                                       price_dir(year, s, split))
            else:
                ret = daily_returns(prediction_dir(year, f"{tag}e{fold}_{s}", r, s, split, P), price_dir(year, s, split),
                                    RULE[1], RULE[2])
            cache[key] = ret
        return cache[key]

    prow, pmiss = [], []
    for fold in YEARS:
        for s in SETS:
            for r in REPEATS:
                d = EXAMM / "test_output" / f"core7_examm_ret_{fold}" / s / r
                Pv = read_predictions(d / "evaluation_val", "val")
                Pt = read_predictions(d / "evaluation_test", "test")
                if Pv is None or Pt is None or not (np.isfinite(Pv.to_numpy()).all() and np.isfinite(Pt.to_numpy()).all()):
                    pmiss.append(f"{fold} {s}/{r}")
                    continue
                val_mse, val_ic = forecast_scores(Pv, fold, s, "val")
                vr = rule_returns(fold, s, r, fold, "val", Pv, "pval")
                tr = rule_returns(fold, s, r, fold, "test", Pt, "")
                test_mse, test_ic = forecast_scores(Pt, fold, s, "test")
                mv, mt = metrics(vr), metrics(tr)
                prow.append({"test_year": fold, "set": s, "repeat": r, "val_mse": val_mse, "val_rank_ic": val_ic,
                             "val_return_%": total_return(vr), "val_sharpe": mv["sharpe"],
                             "test_return_%": total_return(tr), "test_sharpe": mt["sharpe"], "test_max_dd_%": mt["max_dd_%"],
                             "test_mse": test_mse, "test_rank_ic": test_ic})
    np.savez(cache_path, **cache)
    if pmiss:
        print(f"\n!! --pareto: {len(pmiss)} repeat(s) left out (no complete evaluation_val / evaluation_test predictions): "
              + ", ".join(pmiss[:12]) + (" ..." if len(pmiss) > 12 else ""))
    pt = pd.DataFrame(prow)
    if pt.empty:
        print("\n!! --pareto: no repeat has both validation and test predictions -- run the cross-set evaluation first")
    else:
        pt["val_frontier"] = False
        pt["choice"] = ""
        for (fold, s), g in pt.groupby(["test_year", "set"]):
            pt.loc[g.index, "val_frontier"] = pareto(g["val_mse"].to_numpy(), g["val_return_%"].to_numpy())
            g = pt.loc[g.index]
            score = (g["val_mse"].rank(method="min") + g["val_return_%"].rank(method="min", ascending=False)).where(g["val_frontier"])
            labels = [(g.loc[score[score == score.min()].index, "val_mse"].idxmin(), "Pareto choice"),
                      (g["val_mse"].idxmin(), "lowest val MSE"), (g["val_return_%"].idxmax(), "highest val return")]
            for idx in g.index:                     # one repeat can carry several labels
                pt.loc[idx, "choice"] = "; ".join(lab for i, lab in labels if i == idx)
        pt.to_csv(RESULTS / f"core7_main{OUT_TAG}_pareto.csv", index=False)

        choices = ["lowest val MSE", "Pareto choice", "highest val return"]
        AVG = "average over the 10 runs"      # no choice: the 4-set book of each run number, averaged over the 10

        def average_rows(returns_of):
            """Rows of the 'average over the 10 runs' reference: for every repeat number r the 4-set book (set1-4 run r,
            like the chosen rows), its yearly net return / Sharpe / max drawdown and 2022-2024 joined metrics, then the
            mean over the runs. returns_of(s, r, year) -> daily returns or None."""
            per_run = []
            for r in REPEATS:
                books = []
                for year in YEARS:
                    per = [returns_of(s, r, year) for s in SETS]
                    if any(x is None for x in per):
                        break
                    books.append(combine(per))
                if len(books) < len(YEARS):
                    continue
                m = metrics(np.concatenate(books))
                per_run.append({**{str(y): (total_return(b), metrics(b)["sharpe"], metrics(b)["max_dd_%"])
                                   for y, b in zip(YEARS, books)},
                                "joined": (m["ann_return_%"], m["sharpe"], m["max_dd_%"])})
            if not per_run:
                return []
            out = [{"choice": AVG, "period": str(y),
                    "net_return_%": float(np.mean([x[str(y)][0] for x in per_run])),
                    "sharpe": float(np.mean([x[str(y)][1] for x in per_run])),
                    "max_dd_%": float(np.mean([x[str(y)][2] for x in per_run])),
                    "repeats": f"{len(per_run)} runs (run r of every set, r = {REPEATS[0]}..{REPEATS[-1]})"} for y in YEARS]
            out.append({"choice": AVG, "period": f"{YEARS[0]}-{YEARS[-1]}",
                        "net_return_%": float(np.mean([o["net_return_%"] for o in out])),
                        "ann_return_%": float(np.mean([x["joined"][0] for x in per_run])),
                        "sharpe": float(np.mean([x["joined"][1] for x in per_run])),
                        "max_dd_%": float(np.mean([x["joined"][2] for x in per_run]))})
            return out

        summ = []
        for ch in choices:
            per_year = []
            for fold in YEARS:
                sel = pt[(pt["test_year"] == fold) & pt["choice"].str.contains(ch, regex=False)]
                if sel["set"].nunique() < len(SETS):
                    break
                book = combine([cache[f"{fold}|{row.set}|{row.repeat}|{fold}"] for row in sel.itertuples()])
                per_year.append(book)
                summ.append({"choice": ch, "period": str(fold), "net_return_%": total_return(book),
                             "sharpe": metrics(book)["sharpe"], "max_dd_%": metrics(book)["max_dd_%"],
                             "repeats": ", ".join(f"{row.set}/{row.repeat}" for row in sel.itertuples())})
            if len(per_year) == len(YEARS):
                joined = np.concatenate(per_year)
                m = metrics(joined)
                summ.append({"choice": ch, "period": f"{YEARS[0]}-{YEARS[-1]}",
                             "net_return_%": float(np.mean([x["net_return_%"] for x in summ[-len(YEARS):]])),
                             "ann_return_%": m["ann_return_%"], "sharpe": m["sharpe"], "max_dd_%": m["max_dd_%"],
                             **bootstrap_ci(joined), "psr": psr(joined)})
        def _retrain_returns(s, r, year):
            key = f"{year}|{s}|{r}|{year}"
            return cache[key] if key in cache else None
        summ += average_rows(_retrain_returns)
        summ = pd.DataFrame(summ)
        summ.to_csv(RESULTS / f"core7_main{OUT_TAG}_pareto_summary.csv", index=False)
        print("\n" + "=" * 96)
        print("PARETO CHOICE PER SET (validation MSE vs validation net return of the same rule; retrain every year)")
        print("=" * 96)
        for col, name in (("net_return_%", "Net return (%) -- per year; 2022-2024 = mean of the years"), ("sharpe", "Sharpe")):
            print(f"\n{name}:")
            print(summ.pivot(index="choice", columns="period", values=col).reindex(index=choices + [AVG],
                  columns=[str(y) for y in YEARS] + [f"{YEARS[0]}-{YEARS[-1]}"]).round(2).to_string())
        print("\nFrontier size per test year and set (repeats on the validation frontier):")
        print(pt.groupby(["test_year", "set"])["val_frontier"].sum().unstack().to_string())
        rho = pt.groupby("test_year")[["val_return_%", "test_return_%"]].apply(lambda g: g.corr(method="spearman").iloc[0, 1])
        print("\nSpearman correlation of validation and test net return across the 40 repeats, per test year:")
        print(rho.round(2).to_string())

        # never retrain: the fold-2022 choice (validation 2021) kept for every test year
        F0 = YEARS[0]
        nev, nmiss = [], []
        for ch in choices:
            sel = pt[(pt["test_year"] == F0) & pt["choice"].str.contains(ch, regex=False)]
            if sel["set"].nunique() < len(SETS):
                continue
            per_year = []
            for year in YEARS:
                per = []
                for row in sel.itertuples():
                    P = read_predictions(pred_dir(F0, row.set, row.repeat, year), "test")
                    if P is None or not np.isfinite(P.to_numpy()).all():
                        nmiss.append(f"{row.set}/{row.repeat} on {year}")
                        continue
                    per.append(rule_returns(F0, row.set, row.repeat, year, "test", P, ""))
                if len(per) < len(SETS):
                    break
                book = combine(per)
                per_year.append(book)
                nev.append({"choice": ch, "period": str(year), "net_return_%": total_return(book),
                            "sharpe": metrics(book)["sharpe"], "max_dd_%": metrics(book)["max_dd_%"],
                            "repeats": ", ".join(f"{row.set}/{row.repeat}" for row in sel.itertuples())})
            if len(per_year) == len(YEARS):
                joined = np.concatenate(per_year)
                m = metrics(joined)
                nev.append({"choice": ch, "period": f"{YEARS[0]}-{YEARS[-1]}",
                            "net_return_%": float(np.mean([x["net_return_%"] for x in nev[-len(YEARS):]])),
                            "ann_return_%": m["ann_return_%"], "sharpe": m["sharpe"], "max_dd_%": m["max_dd_%"],
                            **bootstrap_ci(joined), "psr": psr(joined)})
        def _never_returns(s, r, year):
            P = read_predictions(pred_dir(F0, s, r, year), "test")
            if P is None or not np.isfinite(P.to_numpy()).all():
                nmiss.append(f"{s}/{r} on {year}")
                return None
            return rule_returns(F0, s, r, year, "test", P, "")
        nev += average_rows(_never_returns)
        np.savez(cache_path, **cache)
        print("\n" + "=" * 96)
        print(f"PARETO CHOICE PER SET -- NEVER RETRAIN (fold {F0} choice, validation {F0 - 1}, kept for {YEARS[0]}-{YEARS[-1]})")
        print("=" * 96)
        if nmiss:
            print(f"!! no forecasts for {len(set(nmiss))} repeat/year(s): " + ", ".join(sorted(set(nmiss))[:12])
                  + "\n   make them with:  ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh  (in EXAMM-Extended)")
        if nev:
            nev = pd.DataFrame(nev)
            nev.to_csv(RESULTS / f"core7_main{OUT_TAG}_pareto_never_retrain.csv", index=False)
            for col, name in (("net_return_%", "Net return (%) -- per year; 2022-2024 = mean of the years"),
                              ("sharpe", "Sharpe"), ("max_dd_%", "Max drawdown (%)")):
                print(f"\n{name}:")
                print(nev.pivot(index="choice", columns="period", values=col).reindex(
                      index=[c for c in choices + [AVG] if c in set(nev["choice"])],
                      columns=[str(y) for y in YEARS] + [f"{YEARS[0]}-{YEARS[-1]}"]).round(2).to_string())
            print("\nRepeats chosen (fold 2022):")
            print(nev[(nev["period"] == str(F0)) & (nev["choice"] != AVG)].set_index("choice")["repeats"].to_string())
            print(f"Saved {RESULTS / f'core7_main{OUT_TAG}_pareto_never_retrain.csv'}")
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(YEARS), len(SETS), figsize=(16, 3.6 * len(YEARS)), squeeze=False)
            for i, fold in enumerate(YEARS):
                for j, s in enumerate(SETS):
                    ax = axes[i][j]
                    g = pt[(pt["test_year"] == fold) & (pt["set"] == s)].copy()
                    if g.empty:
                        ax.set_visible(False)
                        continue
                    g["mse_pct"] = g["val_mse"].rank(ascending=False, pct=True) * 100
                    ax.scatter(g["mse_pct"], g["val_return_%"], c="#b0b0b0", s=28, zorder=2)
                    f = g[g["val_frontier"] == True].sort_values("mse_pct")
                    ax.plot(f["mse_pct"], f["val_return_%"], "-o", c="#1f3864", ms=5, zorder=3)
                    for lab, mk, col in (("lowest val MSE", "s", "#2e7d32"), ("Pareto choice", "*", "#c0392b")):
                        h = g[g["choice"].str.contains(lab, regex=False)]
                        ax.scatter(h["mse_pct"], h["val_return_%"], marker=mk, s=160 if mk == "*" else 70,
                                   c=col if mk == "*" else "white", edgecolor=col, lw=1.5, zorder=4, label=lab)
                    ax.axhline(0, c="#888888", lw=0.8)
                    ax.set_xlim(0, 105)
                    ax.grid(color="#e6e6e6")
                    if i == 0:
                        ax.set_title(f"V{j + 1} ({s})", fontweight="bold")
                    if j == 0:
                        ax.set_ylabel(f"test {fold}: val {fold - 1} net return (%)")
                    if i == len(YEARS) - 1:
                        ax.set_xlabel("val MSE percentile rank (100 = lowest)")
            axes[0][0].legend(fontsize=8, loc="lower left")
            fig.suptitle("Validation Pareto frontier per panel set: MSE vs net return of the trading rule (10 repeats each)",
                         fontweight="bold")
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            fig.savefig(RESULTS / f"core7_main{OUT_TAG}_pareto.png", dpi=150)
            print(f"\nSaved {RESULTS / f'core7_main{OUT_TAG}_pareto.png'}")
        except ImportError:
            print("\nmatplotlib not installed -- no figure")
        print(f"Saved {RESULTS / f'core7_main{OUT_TAG}_pareto.csv'} and _pareto_summary.csv")

# ---------------------------------------------------------------- mean over the 10 seeds (default; --no-mean-repeats skips it) ----
if MEAN_REPEATS:
    # the paper's Table 5 EXAMM column (net return %, per panel and test year)
    PAPER_T5 = {("V1", 2022): -4.63, ("V1", 2023): 21.34, ("V1", 2024): 17.19, ("V2", 2022): 35.22, ("V2", 2023): 16.92,
                ("V2", 2024): 13.80, ("V3", 2022): 36.68, ("V3", 2023): -3.82, ("V3", 2024): -2.88, ("V4", 2022): -5.43,
                ("V4", 2023): 1.84, ("V4", 2024): 5.96}
    vname = {s: f"V{i + 1}" for i, s in enumerate(SETS)}
    designs = (("retrain every year", lambda y: y), (f"never retrain (fold {YEARS[0]} models)", lambda y: YEARS[0]))
    rrows, rbooks, rmiss = [], {}, []
    for fold_of_name, fold_of in designs:
        for s in SETS:
            for r in REPEATS:
                for year in YEARS:
                    fold = fold_of(year)
                    key = f"{fold}|{s}|{r}|{year}"
                    P = read_predictions(pred_dir(fold, s, r, year), "test")
                    if P is None or not np.isfinite(P.to_numpy()).all():
                        rmiss.append(f"fold {fold} {s}/{r} on {year}")
                        continue
                    if key not in cache:
                        if PAPER:
                            st = ret_stats(s)
                            ret, n_trades = paper_returns(prediction_dir(year, f"paper{fold}_{s}", r, s, "test",
                                                                         P * st["iqr"] + st["median"]),
                                                          price_dir(year, s, "test"))
                            cache[key], cache[f"{key}|trades"] = ret, np.array([n_trades, len(ret)])
                        else:
                            cache[key] = daily_returns(prediction_dir(year, f"stale{fold}_{s}", r, s, "test", P),
                                                       price_dir(year, s, "test"), RULE[1], RULE[2])
                    mse, ic = forecast_scores(P, year, s, "test")
                    m = metrics(cache[key])
                    rbooks[(fold_of_name, s, r, year)] = cache[key]
                    rrows.append({"design": fold_of_name, "fold": fold, "test_year": year, "set": s, "panel": vname[s],
                                  "repeat": r, "total_return_%": total_return(cache[key]), "sharpe": m["sharpe"],
                                  "max_dd_%": m["max_dd_%"], "test_mse": mse, "test_rank_ic": ic,
                                  "test_ic_pearson": pearson_ic(P, year, s)})
        np.savez(cache_path, **cache)
    rep = pd.DataFrame(rrows).drop_duplicates(["design", "set", "repeat", "test_year"])
    print("\n" + "=" * 96)
    print(f"MEAN OVER THE 10 REPEATS, L/S {K}/{K} (the paper's reporting: every seed traded, metrics averaged over "
          "the seeds; no choice by validation)")
    print("=" * 96)
    if rmiss:
        print(f"!! {len(rmiss)} repeat/year(s) without forecasts left out: " + ", ".join(rmiss[:10])
              + (" ..." if len(rmiss) > 10 else "") + "\n   never-retrain forecasts: ALL_REPEATS=1 sh "
              "scripts/rnn_examples/evaluate_rnn_core7_main.sh (in EXAMM-Extended)")
    if rep.empty:
        print("nothing to report")
    else:
        rep.to_csv(RESULTS / f"core7_main{OUT_TAG}_repeats.csv", index=False)
        t7_parts, t8_parts = [], []
        for name, _ in designs:
            d = rep[rep["design"] == name]
            if d.empty:
                continue
            n_rep = d.groupby(["test_year", "set"])["repeat"].nunique()
            mean = d.pivot_table(index="test_year", columns="panel", values="total_return_%", aggfunc="mean")
            sd = d.pivot_table(index="test_year", columns="panel", values="total_return_%", aggfunc="std")
            pos = d.assign(p=d["total_return_%"] > 0).pivot_table(index="test_year", columns="panel", values="p", aggfunc="sum")
            mean["Average V1-V4"] = mean.mean(axis=1)
            mean.loc["Mean"] = mean.mean(axis=0)
            print(f"\n{name} -- Table 5/7 style, net return (%), MEAN over the repeats "
                  f"({n_rep.min()}-{n_rep.max()} repeats per panel and year):")
            print(mean.round(2).to_string())
            print("\n  standard deviation over the repeats (%):")
            print(sd.round(2).to_string())
            print(f"\n  repeats with a positive net return (of {n_rep.max()}):")
            print(pos.astype(int).to_string())
            if name == "retrain every year":
                cmp = mean.drop(index="Mean").drop(columns="Average V1-V4").stack().rename("yours (mean of repeats)").reset_index()
                cmp["paper EXAMM"] = [PAPER_T5.get((v, y)) for y, v in zip(cmp["test_year"], cmp["panel"])]
                cmp["yours - paper"] = cmp["yours (mean of repeats)"] - cmp["paper EXAMM"]
                print("\n  next to the paper's Table 5 EXAMM column:")
                print(cmp.round(2).to_string(index=False))
                print(f"  mean: yours {cmp['yours (mean of repeats)'].mean():.2f} %, paper {cmp['paper EXAMM'].mean():.2f} %")
            t7 = mean.reset_index().rename(columns={"index": "test_year"})
            t7.insert(0, "design", name)
            t7_parts.append(t7)
            per_rep = []                      # per panel and repeat: 2022-2024 joined Sharpe / MDD, mean of the years
            for (s, r), g in d.groupby(["set", "repeat"]):
                if g["test_year"].nunique() < len(YEARS):
                    continue
                joined = np.concatenate([rbooks[(name, s, r, y)] for y in YEARS])
                m = metrics(joined)
                per_rep.append({"panel": vname[s], "repeat": r, "Rank IC (Spearman)": g["test_rank_ic"].mean(),
                                "IC (Pearson)": g["test_ic_pearson"].mean(), "Net (%) mean of years": g["total_return_%"].mean(),
                                "Sharpe (2022-24)": m["sharpe"], "MDD (%)": -m["max_dd_%"]})
            if per_rep:
                pr = pd.DataFrame(per_rep)
                t8 = pr.drop(columns="repeat").groupby("panel").mean()
                t8["Sharpe sd over repeats"] = pr.groupby("panel")["Sharpe (2022-24)"].std()
                t8.loc["Average V1-V4"] = t8.mean(axis=0)
                print(f"\n{name} -- Table 8 style (per repeat, mean over the repeats, then averaged over V1-V4):")
                print(t8.round({"Rank IC (Spearman)": 4, "IC (Pearson)": 4, "Net (%) mean of years": 2, "Sharpe (2022-24)": 2,
                                "MDD (%)": 2, "Sharpe sd over repeats": 2}).to_string())
                t8 = t8.reset_index().rename(columns={"index": "panel"})
                t8.insert(0, "design", name)
                t8_parts.append(t8)
        print("\nPaper (Table 8, L/S 10/10, mean over its repeats): IC (Pearson) +0.0106, Net +11.02 %, Sharpe 0.74, "
              "MDD -10.76 %")
        if t7_parts:
            pd.concat(t7_parts, ignore_index=True).to_csv(RESULTS / f"core7_main{OUT_TAG}_repeats_table7.csv", index=False)
        if t8_parts:
            pd.concat(t8_parts, ignore_index=True).to_csv(RESULTS / f"core7_main{OUT_TAG}_repeats_table8.csv", index=False)
        print(f"Saved {RESULTS / f'core7_main{OUT_TAG}_repeats.csv'}, _repeats_table7.csv and _repeats_table8.csv")
