#!/usr/bin/env python3
"""Does the Pooled RET result depend on which genome was chosen? Every repeat, on its own.

For every RET walk-forward fold (2022, 2023, 2024) and panel set (set1-4), each of the 10
repeats (folders 15..24) has its own final global best genome. main_core7.py trades only the one
with the lowest validation MSE -- a choice made on the validation year EXAMM itself evolved on.
Here every repeat is scored and traded SEPARATELY on the test year (no ensemble: one genome,
one forecast, one portfolio each), so the spread across the 10 repeats shows how much the
result depends on that choice, and where the chosen repeat ranks among them.

Per set / fold / repeat:
  * Rank IC on raw returns (same calculation as core7_ic.py: predictions back to raw returns with
    the set's median/IQR, daily cross-sectional Spearman vs the unclipped RET_raw, warm-up dropped)
  * a small FIXED rule list, set before looking (costs on): daily long-short 5/5 (the original
    rule of core7_table.py), overlapping 5/5 and 3/3 held 10 days, 5/5 held 20 days, equal weight;
    annualized return, Sharpe, max drawdown, turnover -- computed as in core7_frontier.py.

Needs, on the Mac, every repeat's global best genome evaluated on test:
    sh ~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_repeats.sh
and each repeat's fitness_log.csv (to know which repeat validation MSE chose).
Builds its own price / prediction files in results/core7_repeats_inputs/ (main_core7.py's files
are not touched) and writes
    results/core7_repeats_by_fold.csv   one row per fold / set / repeat / rule
    results/core7_repeats_joined.csv    2022-2024 joined per set, per repeat folder (folder 15 of
                                        each fold, 16 of each fold, ...) and for the chosen repeats
    results/core7_repeats_summary.csv   median / min / max / fraction positive across the repeats

Usage:  python3 core7_repeats.py
"""
import contextlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from Logger import Logger
from portfolio import Portfolio
from stock import Stock

HERE = Path(__file__).resolve().parent
EXAMM = Path(os.getenv("EXAMM_DIR", Path.home() / "EXAMM-Extended"))
RESULTS = Path(os.getenv("FIN_RESULTS_DIR", HERE / "results"))
INPUTS = RESULTS / "core7_repeats_inputs"
YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
REPEATS = [str(r) for r in range(15, 25)]
TRADING_DAYS = 252
N_STOCKS = 50

# fixed before looking at any repeat: (rule name, strategy, keyword arguments for Portfolio.trade)
RULES = [
    ("L/S 5/5 daily", "daily_long_short_return", dict(long_company_number=5, short_company_number=5)),
    ("overlap 5/5 hold 10", "overlapping_return", dict(long_company_number=5, hold_days=10)),
    ("overlap 3/3 hold 10", "overlapping_return", dict(long_company_number=3, hold_days=10)),
    ("overlap 5/5 hold 20", "overlapping_return", dict(long_company_number=5, hold_days=20)),
    ("equal weight", "daily_equal_weight_return", {}),
]
SELF_VALUED = {"overlapping_return", "banded_return"}


# ------------------------------------------------------------------ inputs
def final_genome(repeat_dir):
    """The repeat's final global best genome: the highest-numbered file (earlier ones are
    superseded bests), the same one main_core7.py uses."""
    genomes = sorted(repeat_dir.glob("global_best_genome_*.bin"), key=lambda g: int(g.stem.split("_")[-1]))
    return genomes[-1].stem if genomes else None


def val_mse(repeat_dir):
    log = repeat_dir / "fitness_log.csv"
    return float(pd.read_csv(log).iloc[-1, 4]) if log.exists() else np.nan


def fold_rows(year, s):
    """Test rows of the panel for one set / fold, from the 40-day warm-up's last day on (that
    day is the entry price of the first trade), as main_core7.py builds them."""
    info = pd.read_csv(EXAMM / "datasets" / f"panels_core7_examm_{year}" / s / "split_info.csv").set_index("split")
    first, last = int(info.loc["test", "first_row"]), int(info.loc["test", "last_row"])
    rows = pd.read_csv(EXAMM / "datasets" / "panels_core7" / f"{s}_core7" / "panel_dates.csv") \
        .set_index("row").loc[first:last]
    skip = int((rows["date"] < f"{year}-01-01").sum()) - 1
    return first, rows, skip


def write_prices(year, s, rows, skip, stocks):
    """Split-adjusted price file per stock, rebuilt from raw returns (as main_core7.py does)."""
    out = INPUTS / f"ret_{year}_{s}_test_prices"
    out.mkdir(parents=True, exist_ok=True)
    r = rows.iloc[skip:]
    for t in stocks:
        growth = (1 + r[f"RET_raw_{t}"]).cumprod()
        prc = r[f"PRC_{t}"].iloc[0] * growth / growth.iloc[0]
        tc = r[f"TC_{t}"] / r[f"PRC_{t}"] * prc
        pd.DataFrame({"date": r["date"].to_numpy(), "PRC": prc.to_numpy(), "BID": (prc - tc).to_numpy(),
                      "ASK": (prc + tc).to_numpy(), "TRAN_COST": tc.to_numpy()}) \
            .to_csv(out / f"{t}.csv", index=False)
    return out


def read_predictions(eval_dir):
    pred = {}
    for f in sorted(eval_dir.glob("*_test_predictions.csv")):
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        pred[f.name[:-len("_test_predictions.csv")]] = df["predicted_RET"].to_numpy()
    return pd.DataFrame(pred)


def rank_ic(pred, year, s, first, rows):
    """Mean daily cross-sectional Spearman of the raw-return forecast vs the realized raw
    return, warm-up dropped -- core7_ic.py's calculation."""
    st = json.load(open(EXAMM / "datasets" / "panels_core7" / f"{s}_core7" / "panel_meta.json"))["stats"]["RET"]
    p = pred * st["iqr"] + st["median"]
    panel = rows.loc[first + np.arange(len(p)) + 1]
    a = pd.DataFrame({t: panel[f"RET_raw_{t}"].to_numpy() for t in p.columns})
    keep = (panel["date"] >= f"{year}-01-01").to_numpy()
    ic = p[keep].rank(axis=1).corrwith(a[keep].rank(axis=1), axis=1).dropna()
    return ic.mean()


def write_predictions(pred, skip, year, s, repeat):
    out = INPUTS / f"ret_{year}_{s}_{repeat}_test_prediction"
    out.mkdir(parents=True, exist_ok=True)
    for t in pred.columns:
        pd.DataFrame({"predicted_RET": pred[t].iloc[skip:].to_numpy()}).to_csv(out / f"{t}_predictions.csv", index=False)
    return out


def load_portfolio(pred_dir, price_dir, stocks):
    logger = Logger()
    logger.set_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio = Portfolio(stocks)
        portfolio.set_initial_spend_per_stock(100)
        portfolio.set_initial_capital(1000)
        portfolio.set_logger(logger)
        for t in stocks:
            company = Stock(t)
            company.read_return_prediction(str(pred_dir / f"{t}_predictions.csv"))
            company.read_stock_price(str(price_dir / f"{t}.csv"))
            portfolio.add_company_to_protfolio(company)
            company.set_logger(logger)
    portfolio.set_use_TC(True)
    portfolio.set_trade_with_bid_ask(False)
    return portfolio


# ------------------------------------------------------------------ trading (as core7_frontier.py)
def daily_returns(portfolio, strategy, kwargs):
    """Daily returns and annual turnover of one strategy run, without changing portfolio.py:
    daily long-short / equal weight use the daily_values portfolio.py records; overlapping is
    valued here from the trades' cash flows plus the open positions each morning."""
    state = {"cash": 0.0, "traded": 0.0, "marks": []}
    restore = []

    def wrap(obj, method, make):
        setattr(obj, method, make(getattr(obj, method)))
        restore.append((obj, method))

    def position_value(company, time):
        return company.get_share() * abs(company.get_stock_price(time))

    for company in portfolio.portfolio_list:
        def clear(original, c=company):
            def f(time):
                state["traded"] += abs(position_value(c, time))
                cash = original(time)
                state["cash"] += cash
                return cash
            return f

        def rebalance(original, c=company):
            def f(target, time):
                state["traded"] += abs(target - position_value(c, time))
                cash = original(target, time)
                state["cash"] += cash
                return cash
            return f

        def opening(original):
            def f(cash_amount, time):
                state["traded"] += abs(cash_amount)
                return original(cash_amount, time)
            return f

        wrap(company, "clear_holdings", clear)
        wrap(company, "rebalance_to", rebalance)
        wrap(company, "buy_stock", opening)
        wrap(company, "short_stock", opening)

    def morning(original):
        def f(time, *a, **k):
            state["marks"].append(portfolio.initial_capital + state["cash"]
                                  + sum(position_value(c, time) for c in portfolio.portfolio_list))
            return original(time, *a, **k)
        return f

    def capture(original):
        def f(*a, **k):
            if strategy in SELF_VALUED:
                state["values"] = state["marks"] + [portfolio.initial_capital + state["cash"]]
            else:
                state["values"] = list(getattr(portfolio, "daily_values", []))
            return original(*a, **k)
        return f

    wrap(portfolio, "get_sorted_return_list", morning)
    wrap(portfolio, "calculate_return", capture)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            portfolio.trade(strategy, **kwargs)
    finally:
        for obj, method in restore:
            delattr(obj, method)
    values = np.asarray(state.get("values", []), dtype=float)
    if len(values) < 3:
        sys.exit(f"{strategy} recorded no daily_values -- portfolio.py needs the daily_values lines "
                 f"in daily_long_short_return and daily_equal_weight_return (added for Sharpe/MDD)")
    turnover = state["traded"] / values.mean() * TRADING_DAYS / (len(values) - 1)
    return values[1:] / values[:-1] - 1, turnover


def metrics(r):
    r = np.asarray(r, dtype=float)
    value = np.cumprod(1 + r)
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {"ann_return_%": (value[-1] ** (TRADING_DAYS / len(r)) - 1) * 100,
            "sharpe": r.mean() * TRADING_DAYS / vol if vol > 1e-12 else np.nan,
            "max_dd_%": (1 - value / np.maximum.accumulate(value)).max() * 100}


# ------------------------------------------------------------------ run
rows_out, series, missing = [], {}, []
for year in YEARS:
    for s in SETS:
        first, rows, skip = fold_rows(year, s)
        price_dir, stocks = None, None
        for repeat in REPEATS:
            repeat_dir = EXAMM / "test_output" / f"core7_examm_ret_{year}" / s / repeat
            eval_dir = repeat_dir / "evaluation_test"
            if len(list(eval_dir.glob("*_test_predictions.csv"))) != N_STOCKS:
                missing.append(f"{year} {s}/{repeat}")
                continue
            pred = read_predictions(eval_dir)
            if stocks is None:
                stocks = sorted(pred.columns)
                price_dir = write_prices(year, s, rows, skip, stocks)
            ic = rank_ic(pred, year, s, first, rows)
            portfolio = load_portfolio(write_predictions(pred, skip, year, s, repeat), price_dir, stocks)
            base = {"fold": year, "set": s, "repeat": repeat, "genome": final_genome(repeat_dir),
                    "val_mse": val_mse(repeat_dir), "rank_ic": ic}
            for name, strategy, kwargs in RULES:
                r, turnover = daily_returns(portfolio, strategy, kwargs)
                series[(year, s, repeat, name)] = r
                rows_out.append({**base, "rule": name, **metrics(r), "turnover_x": turnover})
        print(f"{year} {s}: {sum(1 for x in rows_out if x['fold'] == year and x['set'] == s) // len(RULES)} "
              f"of {len(REPEATS)} repeats done")

if missing:
    print(f"\n{len(missing)} repeats have no test predictions yet (skipped): {', '.join(missing[:12])}"
          + (" ..." if len(missing) > 12 else ""))
    print("  evaluate them with:  sh ~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_repeats.sh")
if not rows_out:
    sys.exit("nothing to report")

by_fold = pd.DataFrame(rows_out)
# the repeat main_core7.py uses: lowest final validation MSE in the set / fold
chosen = by_fold.drop_duplicates(["fold", "set", "repeat"]).dropna(subset=["val_mse"]) \
    .sort_values("val_mse").groupby(["fold", "set"])["repeat"].first()
if by_fold["val_mse"].isna().all():
    print("\nno fitness_log.csv found -- download them to know which repeat validation MSE chose")
by_fold["chosen"] = [chosen.get((f, s)) == r for f, s, r in zip(by_fold["fold"], by_fold["set"], by_fold["repeat"])]
by_fold["sharpe_rank_of_10"] = by_fold.groupby(["fold", "set", "rule"])["sharpe"].rank(ascending=False, method="min")
RESULTS.mkdir(exist_ok=True)
by_fold.to_csv(RESULTS / "core7_repeats_by_fold.csv", index=False)

# 2022-2024 joined per set: the same folder number in every fold, and the chosen repeats
joined = []
for s in SETS:
    paths = {f"folder {r}": {y: r for y in YEARS} for r in REPEATS}
    paths["chosen (val MSE)"] = {y: chosen.get((y, s)) for y in YEARS}
    for label, pick in paths.items():
        for name, *_ in RULES:
            parts = [series.get((y, s, pick[y], name)) for y in YEARS]
            if any(p is None for p in parts):
                continue
            joined.append({"set": s, "path": label, "rule": name, **metrics(np.concatenate(parts))})
joined = pd.DataFrame(joined)
joined.to_csv(RESULTS / "core7_repeats_joined.csv", index=False)


def spread(df, value):
    g = df.groupby(["rule", "fold", "set"])
    out = pd.DataFrame({"n": g[value].count(), "median": g[value].median(), "min": g[value].min(),
                        "max": g[value].max(), "frac_positive": g[value].apply(lambda v: (v > 0).mean())})
    pick = df[df["chosen"]].set_index(["rule", "fold", "set"])
    out["chosen"] = pick[value]
    out["chosen_rank_of_10"] = pick["sharpe_rank_of_10"] if value == "sharpe" else np.nan
    return out


summary = pd.concat({"sharpe": spread(by_fold, "sharpe"), "ann_return_%": spread(by_fold, "ann_return_%")},
                    names=["metric"]).reset_index()
summary.to_csv(RESULTS / "core7_repeats_summary.csv", index=False)

# ------------------------------------------------------------------ printed summary
pd.set_option("display.width", 220)
order = [name for name, *_ in RULES]
ic = by_fold.drop_duplicates(["fold", "set", "repeat"])
ic_tab = ic.groupby(["fold", "set"])["rank_ic"].agg(["median", "min", "max", lambda v: (v > 0).mean()])
ic_tab.columns = ["median", "min", "max", "frac_positive"]
ic_tab["chosen"] = ic[ic["chosen"]].set_index(["fold", "set"])["rank_ic"]
print("\nTest Rank IC across the 10 repeats of each set / fold (chosen = lowest validation MSE):")
print(ic_tab.round(4).to_string())

print("\nTest Sharpe across the 10 repeats of each set / fold, per rule "
      "(chosen_rank_of_10: 1 = best of the 10 on test):")
sh = summary[summary["metric"] == "sharpe"].drop(columns="metric")
sh["rule"] = pd.Categorical(sh["rule"], order)
print(sh.sort_values(["rule", "fold", "set"]).round(2).to_string(index=False))

print("\nAll 40 set-repeats of each fold pooled (test Sharpe):")
pooled = by_fold.groupby(["rule", "fold"]).agg(median=("sharpe", "median"),
                                              frac_positive=("sharpe", lambda v: (v > 0).mean()),
                                              chosen_median=("sharpe", lambda v: np.nan))
pooled["chosen_median"] = by_fold[by_fold["chosen"]].groupby(["rule", "fold"])["sharpe"].median()
print(pooled.reindex(order, level=0).round(2).to_string())

print("\n2022-2024 joined per set (Sharpe): spread over the 10 folder paths vs the chosen repeats:")
if not joined.empty:
    j = joined.copy()
    j["kind"] = np.where(j["path"].str.startswith("chosen"), "chosen", "folder")
    folders = j[j["kind"] == "folder"].groupby(["rule", "set"])["sharpe"]
    jt = pd.DataFrame({"median": folders.median(), "min": folders.min(), "max": folders.max(),
                       "frac_positive": folders.apply(lambda v: (v > 0).mean())})
    jt["chosen"] = j[j["kind"] == "chosen"].set_index(["rule", "set"])["sharpe"]
    print(jt.reindex(order, level=0).round(2).to_string())

print(f"\nSaved {RESULTS / 'core7_repeats_by_fold.csv'}, {RESULTS / 'core7_repeats_joined.csv'}, "
      f"{RESULTS / 'core7_repeats_summary.csv'}")
