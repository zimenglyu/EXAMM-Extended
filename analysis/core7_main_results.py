#!/usr/bin/env python3
"""MAIN WORKFLOW, step 2 (backtest and report). Renamed from the stale-model script.
PAPER-STYLE REPORTING: per test year the TOTAL return over the test period ((C - C0) / C0, the paper's "net
return", Algorithm 1 line 17) besides the annualized one; your own buy-and-hold and equal weight per panel and year;
a Table 8 style summary (per panel, then averaged) with Rank IC as Spearman and as Pearson; the chosen models' sizes.
Extra outputs: core7_main_benchmarks.csv, core7_main_table8.csv, core7_main_model_size.csv.
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
PER_SET = "--no-per-set" not in sys.argv
PAPER = "--trade-every-day" not in sys.argv
OUT_TAG = "" if PAPER else "_everyday"
_argv, sys.argv = sys.argv, [sys.argv[0], "--rule", "ls55"]   # core7_genome_frontier reads --rule at import
import numpy as np
import pandas as pd

from core7_genome_frontier import (EXAMM, RESULTS, RULE, SETS, WORK, YEARS, bootstrap_ci, combine, daily_returns,
                                   final_genome, forecast_scores, metrics, prediction_dir, price_dir, psr,
                                   read_predictions, ret_stats, scored_mask)
sys.argv = _argv

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
        values, trades = paper_long_short_daily_values(portfolio, 5, 5, sign_check=True)
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
cache_path = WORK / ("stale_trades_paper.npz" if PAPER else "stale_trades.npz")   # reuse the saved trades
cache = dict(np.load(cache_path)) if cache_path.exists() else {}

cells, books, missing, set_rows, set_books = [], {}, [], [], {}
bench, sizes = {}, []                # (year, set) -> buy-and-hold / equal weight; chosen models' sizes
for fold in YEARS:
    chosen = {s: best_repeat(fold, s) for s in SETS}
    for s in SETS:
        if chosen[s] is not None:
            sizes.append({"fold": fold, "set": s, "repeat": chosen[s][0], **model_size(fold, s, chosen[s][0])})
    for year in [y for y in YEARS if ALL or y >= fold]:
        per, mses, ics, used = [], [], [], []
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
                      "test_mse": float(np.mean(mses)), "test_rank_ic": float(np.mean(ics)),
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
      f"{'PAPER Algorithm 1 (L/S 5/5, trade only if signs agree)' if PAPER else RULE[0]}, costs on, 4 sets combined")
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
        print("\nPaper (Table 8): Rank IC +0.0106, Net +11.02 %, Sharpe 0.74, MDD -10.76 %")

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
