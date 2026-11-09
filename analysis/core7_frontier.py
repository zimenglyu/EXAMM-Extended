#!/usr/bin/env python3
"""Return / volatility Pareto frontier of trading rules on the Pooled RET forecasts.

The forecasts are fixed: each panel set's global best Pooled RET genome of each
walk-forward fold (2022, 2023, 2024). What varies is the trading rule: the daily
long-short book of portfolio.py with every combination of long count x short count
(3, 5, 10, 15 each: 16 rules), plus the equal-weight benchmark -- the strategies whose
daily account value portfolio.py already records, so portfolio.py needs no changes.
For every rule:
  1. each set / fold is traded with portfolio.py (transaction costs on), giving a daily
     account value; its daily returns are joined over the three folds into one
     walk-forward series, and the 4 panel sets are combined into one book (equal weight);
  2. that is done on the VALIDATION years (2021-2023) and on the TEST years (2022-2024);
  3. annualized net return, annualized volatility, Sharpe and max drawdown are computed.
The Pareto frontier (highest return for its volatility: no other rule has both a higher
return and a lower volatility) is chosen on VALIDATION, and those rules are reported on
TEST, so the test numbers are not picked with the test years.

Needs, for every set and year, the backtest input files main_core7.py writes:
    python3 main_core7.py <set> <year>       # test  -> results/core7_examm_ret_<year>_<set>_best_test_*
    python3 main_core7.py <set> <year> VAL   # val   -> results/core7_examm_ret_<year>_<set>_best_val_*
and portfolio.py with the daily_values lines in daily_long_short_return and
daily_equal_weight_return (the ones added for Sharpe/MDD).
Writes results/core7_frontier.csv and results/core7_frontier.png.

--no-tc trades with transaction costs off (gross returns), to see how much of the result is
cost drag; it writes results/core7_frontier_noTC.csv and results/core7_frontier_noTC.png instead.

--turnover adds the lower-turnover rules of portfolio.py -- overlapping_return (each day a new
long top-k / short bottom-k book held hold_days, the open books netted) and banded_return (enter
at a rank, exit only past a wider rank) -- and an annual turnover column, and also draws return
against turnover. portfolio.py records no daily values for these two strategies, so this script
computes them itself, without changing portfolio.py: it adds up the cash each trade returns
(rebalance_to / clear_holdings) and, at the start of every day, values the open positions at that
day's price (shorts as a liability). It writes results/core7_frontier_turnover[_noTC].csv/.png.
Turnover = dollars traded (buys + sells, both sides) per year / average account value.

Every run also writes results/core7_frontier<tag>_by_year.csv: the same metrics for each
walk-forward fold on its own (fold Y: validation year Y-1, test year Y), the 4 sets combined,
and prints test Sharpe and return per year for every rule; and
results/core7_frontier<tag>_by_quarter.csv: the same for each calendar quarter (dates from the
price files), printing the test return per quarter. A quarter is ~63 trading days, so its Sharpe
is very noisy (roughly +/- 2); read quarters for where gains and losses happened, not as tests.

Every run also does WALK-FORWARD RULE SELECTION on the test years only. EXAMM evolves its
genomes on validation MSE, so validation is effectively in-sample and a weak guide for choosing
rules; the test years were never seen by EXAMM. For each test year after the first, the rule
with the best Sharpe over the EARLIER test years is chosen (2023: chosen on 2022; 2024: chosen
on 2022-2023) and traded in that year, both as the single best rule and as an equal-weight
book of the top 3. The chosen rules' 2023-2024 results are compared with equal weight and with
the best fixed rule in hindsight (which uses the outcome, so it is only a ceiling). Written to
results/core7_frontier<tag>_walkforward.csv.

Every run also reports how SIGNIFICANT each rule's test result is (test split only):
  - test_sharpe_ci_lo/hi, test_ann_return_ci_lo/hi: 95% bootstrap confidence intervals, from a
    stationary block bootstrap (Politis-Romano, mean block 10 days, 2000 resamples, seed 0) of
    the combined daily book over 2022-2024, so short runs of correlated days stay together;
  - test_psr: probabilistic Sharpe ratio, the probability that the true Sharpe is above 0, given
    the test length and the skewness / kurtosis of the daily returns;
  - test_dsr: deflated Sharpe ratio (Bailey & Lopez de Prado 2014), the same probability but
    against the Sharpe the BEST of N rules would reach by luck alone, where N = the number of
    rules in this run and the spread of luck = the variance of the N test Sharpes. It answers
    "is this still above 0 after allowing for having tried N rules?". N is a lower bound: the
    RET_CS runs and the earlier rule sets were looked at too, so the true number of trials is
    larger (set N_TRIALS below to count them).
The walk-forward books (2023-2024) get the same bootstrap intervals and PSR (no DSR: their rule
is chosen by the walk-forward itself, not picked from the table afterwards).

--target RET_CS uses the RET_CS-target genomes (core7_examm_<year>, the cross-sectional
rank-normal of next-day return) instead of RET; output names get _RET_CS. Needs the RET_CS
backtest inputs:  python3 main_core7.py <set> <year> RET_CS [VAL].

Usage:  python3 core7_frontier.py                       # costs on (net returns)
        python3 core7_frontier.py --no-tc               # costs off (gross returns)
        python3 core7_frontier.py --turnover [--no-tc]  # + overlapping and banded rules
        python3 core7_frontier.py --target RET_CS [--turnover] [--no-tc]
"""
import contextlib
import io
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from Logger import Logger
from portfolio import Portfolio
from stock import Stock

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
SPLITS = ["val", "test"]
TRADING_DAYS = 252

USAGE = "usage: python3 core7_frontier.py [--target RET|RET_CS] [--turnover] [--no-tc]"
ARGS = sys.argv[1:]
TARGET = "RET"
if "--target" in ARGS:
    i = ARGS.index("--target")
    if i + 1 >= len(ARGS) or ARGS[i + 1].upper() not in ("RET", "RET_CS"):
        sys.exit(USAGE)
    TARGET = ARGS[i + 1].upper()
    del ARGS[i:i + 2]
if any(a not in ("--no-tc", "--turnover") for a in ARGS) or len(set(ARGS)) != len(ARGS):
    sys.exit(USAGE)
USE_TC = "--no-tc" not in ARGS
TURNOVER = "--turnover" in ARGS
TAG = ("_turnover" if TURNOVER else "") + ("" if USE_TC else "_noTC") + ("_RET_CS" if TARGET == "RET_CS" else "")


def run_name(year):
    """main_core7.py's run of one walk-forward fold for the chosen target."""
    return f"core7_examm_ret_{year}" if TARGET == "RET" else f"core7_examm_{year}"
COSTS = "costs on (net)" if USE_TC else "costs OFF (gross)"

# (rule name, family, strategy, keyword arguments for Portfolio.trade)
# daily long-short: long the top L and short the bottom S predicted stocks, rebalanced daily
COUNTS = (3, 5, 10, 15)


def family(long_k, short_k):
    return "balanced (L = S)" if long_k == short_k else ("more long (L > S)" if long_k > short_k
                                                          else "more short (L < S)")


RULES = [
    *[(f"L/S {L}/{S}", family(L, S), "daily_long_short_return",
       dict(long_company_number=L, short_company_number=S)) for L in COUNTS for S in COUNTS],
    ("equal weight", "benchmark", "daily_equal_weight_return", {}),
]
if TURNOVER:
    # overlapping: top_k long / bottom_k short, each day's book held hold_days
    RULES += [(f"overlap {k}/{k} hold {h}", "overlapping", "overlapping_return",
               dict(long_company_number=k, hold_days=h)) for k in COUNTS for h in (2, 5, 10, 20)]
    # banded: enter in the top / bottom `entry` ranks, exit once past `exit`
    RULES += [(f"banded {e}-{x}", "banded", "banded_return", dict(entry_rank=e, exit_rank=x))
              for e, x in ((5, 15), (5, 25), (10, 25), (10, 35), (15, 35))]
# strategies whose daily value this script computes itself (portfolio.py records none)
SELF_VALUED = {"overlapping_return", "banded_return"}


def load_portfolio(run, panel_set, split):
    """A Portfolio holding the stocks of one set / fold, with predictions and prices of
    the given split (built by main_core7.py)."""
    pred_dir = RESULTS / f"{run}_{panel_set}_best_{split}_prediction"
    price_dir = RESULTS / f"{run}_{panel_set}_{split}_prices"
    names = sorted(f.name[:-len("_predictions.csv")] for f in pred_dir.glob("*_predictions.csv"))
    if not names or not price_dir.exists():
        year = run.split("_")[-1]
        sys.exit(f"missing {pred_dir.name} -- build it with:\n  python3 main_core7.py {panel_set} {year}"
                 + (" RET_CS" if TARGET == "RET_CS" else "") + (" VAL" if split == "val" else ""))
    logger = Logger()
    logger.set_level("ERROR")
    with contextlib.redirect_stdout(io.StringIO()):
        portfolio = Portfolio(names)
    portfolio.set_initial_spend_per_stock(100)
    portfolio.set_initial_capital(1000)
    portfolio.set_logger(logger)
    with contextlib.redirect_stdout(io.StringIO()):
        add_companies(portfolio, names, pred_dir, price_dir, logger)
    portfolio.set_use_TC(USE_TC)
    portfolio.set_trade_with_bid_ask(False)
    # trading day t uses price row t; daily return k runs from row k to row k + 1
    dates = pd.to_datetime(pd.read_csv(price_dir / f"{names[0]}.csv", usecols=["date"])["date"])
    return portfolio, dates.to_numpy()


def add_companies(portfolio, names, pred_dir, price_dir, logger):
    for name in names:
        company = Stock(name)
        company.read_return_prediction(str(pred_dir / f"{name}_predictions.csv"))
        company.read_stock_price(str(price_dir / f"{name}.csv"))
        portfolio.add_company_to_protfolio(company)
        company.set_logger(logger)


def daily_returns(portfolio, strategy, kwargs):
    """Daily returns and annual turnover of one strategy run. The per-day trading log is kept
    off the screen. Daily long-short / equal weight: the account values portfolio.py records
    in daily_values, copied when calculate_return() is reached. Overlapping / banded: computed
    here from the cash flows of the trades and the open positions valued each morning.
    Nothing is changed in portfolio.py or stock.py; the wrappers are removed after the run."""
    state = {"cash": 0.0, "traded": 0.0, "marks": []}
    restore = []

    def wrap(obj, method, make):
        original = getattr(obj, method)
        setattr(obj, method, make(original))
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
        # called once at the start of every trading day, before that day's trades
        def f(time, *a, **k):
            state["marks"].append(portfolio.initial_capital + state["cash"]
                                  + sum(position_value(c, time) for c in portfolio.portfolio_list))
            return original(time, *a, **k)
        return f

    def capture(original):
        def f(*a, **k):
            if strategy in SELF_VALUED:   # everything is closed by now: the final value is cash
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
        for obj, method in restore:      # back to the class methods
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
    years = len(r) / TRADING_DAYS
    vol = r.std() * np.sqrt(TRADING_DAYS)
    return {"ann_return_%": (value[-1] ** (1 / years) - 1) * 100,
            "ann_vol_%": vol * 100,
            "sharpe": r.mean() * TRADING_DAYS / vol if vol > 1e-12 else np.nan,
            "max_dd_%": (1 - value / np.maximum.accumulate(value)).max() * 100,
            "total_%": (value[-1] - 1) * 100,
            "days": len(r)}


def pareto(df, risk="ann_vol_%"):
    """True for rules that no other rule beats on both higher return and lower risk
    (volatility by default, or turnover)."""
    ret, vol = df["ann_return_%"].to_numpy(), df[risk].to_numpy()
    return [not any((ret[j] >= ret[i]) and (vol[j] <= vol[i]) and (ret[j] > ret[i] or vol[j] < vol[i])
                    for j in range(len(df)) if j != i) for i in range(len(df))]


# ---- significance of a test result (test split only) ----
BOOT_RESAMPLES = 2000
BOOT_BLOCK = 10          # mean block length of the stationary bootstrap, in trading days
BOOT_SEED = 0
N_TRIALS = None          # None: the number of rules in this run; raise it to count other runs looked at
EULER_GAMMA = 0.5772156649015329
PHI = NormalDist()


def stationary_bootstrap(r, resamples=BOOT_RESAMPLES, block=BOOT_BLOCK, seed=BOOT_SEED):
    """Resampled copies (resamples x len(r)) of the daily returns r: blocks of consecutive days
    with random (geometric, mean `block`) lengths and random starts, wrapping around the end."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, n))
    restart = rng.random((resamples, n)) < 1 / block
    idx = np.empty((resamples, n), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    return r[idx]


def bootstrap_ci(r, level=0.95):
    """95% intervals of the annualized Sharpe and the annualized return (%) of daily returns r."""
    boot = stationary_bootstrap(r)
    sd = boot.std(axis=1)
    sharpes = np.where(sd > 1e-12, boot.mean(axis=1) / np.where(sd > 1e-12, sd, 1) * np.sqrt(TRADING_DAYS),
                       np.nan)
    years = boot.shape[1] / TRADING_DAYS
    returns = (np.prod(1 + boot, axis=1) ** (1 / years) - 1) * 100
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"sharpe_ci_lo": np.nanpercentile(sharpes, lo), "sharpe_ci_hi": np.nanpercentile(sharpes, hi),
            "ann_return_ci_lo": np.percentile(returns, lo), "ann_return_ci_hi": np.percentile(returns, hi)}


def sharpe_moments(r):
    """Daily (not annualized) Sharpe, skewness and kurtosis (normal = 3) of daily returns r."""
    r = np.asarray(r, dtype=float)
    sd = r.std()
    if sd <= 1e-12:
        return np.nan, 0.0, 3.0
    z = (r - r.mean()) / sd
    return r.mean() / sd, float((z ** 3).mean()), float((z ** 4).mean())


def probabilistic_sharpe(r, benchmark=0.0):
    """P(true daily Sharpe > benchmark) given len(r) days and the skewness / kurtosis of r
    (Bailey & Lopez de Prado 2012)."""
    sr, skew, kurt = sharpe_moments(r)
    if not np.isfinite(sr):
        return np.nan
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return np.nan
    return PHI.cdf((sr - benchmark) * np.sqrt(len(r) - 1) / np.sqrt(denom))


def deflated_sharpe(r, trial_sharpes, n_trials=None):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014): the probabilistic Sharpe of r against
    the Sharpe expected from the best of n_trials rules with no skill, whose spread is the variance
    of trial_sharpes (daily, not annualized)."""
    trial = np.asarray([x for x in trial_sharpes if np.isfinite(x)], dtype=float)
    n = n_trials or len(trial)
    if n < 2 or len(trial) < 2:
        return np.nan
    expected_max = np.sqrt(trial.var(ddof=1)) * ((1 - EULER_GAMMA) * PHI.inv_cdf(1 - 1 / n)
                                                 + EULER_GAMMA * PHI.inv_cdf(1 - 1 / (n * np.e)))
    return probabilistic_sharpe(r, benchmark=expected_max)


# ---- run every rule on every set / fold, for validation and test ----
series = {(split, name): [] for split in SPLITS for name, *_ in RULES}   # per set: walk-forward daily returns
turnovers = {(split, name): [] for split in SPLITS for name, *_ in RULES}   # per set / fold
yearly = {}   # (split, name, fold year) -> daily returns of each set
yearly_turnover = {}
day_dates = {}   # (split, name, fold year) -> end date of each daily return, per set
for split in SPLITS:
    for s in SETS:
        per_rule = {name: [] for name, *_ in RULES}
        for y in YEARS:
            portfolio, dates = load_portfolio(run_name(y), s, split)
            for name, _, strategy, kwargs in RULES:
                r, turnover = daily_returns(portfolio, strategy, kwargs)
                per_rule[name].append(r)
                turnovers[(split, name)].append(turnover)
                yearly.setdefault((split, name, y), []).append(r)
                day_dates.setdefault((split, name, y), []).append(dates[1:len(r) + 1])
                yearly_turnover.setdefault((split, name, y), []).append(turnover)
            print(f"{split} {s} {y}: {len(RULES)} rules done")
        for name in per_rule:
            series[(split, name)].append(np.concatenate(per_rule[name]))

# ---- one book per rule: the 4 sets combined with equal weight ----
rows = []
test_books = {}   # rule -> combined daily test returns, for the significance columns
for name, family, strategy, kwargs in RULES:
    row = {"rule": name, "family": family}
    for split in SPLITS:
        per_set = series[(split, name)]
        n = min(len(x) for x in per_set)
        book = np.mean([x[:n] for x in per_set], axis=0)
        if split == "test":
            test_books[name] = book
        for k, v in metrics(book).items():
            row[f"{split}_{k}"] = v
        row[f"{split}_turnover_x"] = float(np.mean(turnovers[(split, name)]))
    rows.append(row)
table = pd.DataFrame(rows)

val = table.rename(columns=lambda c: c[4:] if c.startswith("val_") else c)
table["val_frontier"] = pareto(val)
test = table.rename(columns=lambda c: c[5:] if c.startswith("test_") else c)
table["test_frontier"] = pareto(test)
if TURNOVER:   # highest return for the trading it needs
    table["val_turnover_frontier"] = pareto(val, "turnover_x")

# significance of each rule's test result: bootstrap intervals, PSR and DSR
n_trials = N_TRIALS or len(RULES)
trial_sharpes = [sharpe_moments(test_books[name])[0] for name in table["rule"]]
for k in ("sharpe_ci_lo", "sharpe_ci_hi", "ann_return_ci_lo", "ann_return_ci_hi"):
    table[f"test_{k}"] = np.nan
for i, name in enumerate(table["rule"]):
    for k, v in bootstrap_ci(test_books[name]).items():
        table.loc[i, f"test_{k}"] = v
    table.loc[i, "test_psr"] = probabilistic_sharpe(test_books[name])
    table.loc[i, "test_dsr"] = deflated_sharpe(test_books[name], trial_sharpes, n_trials)

RESULTS.mkdir(exist_ok=True)
table.to_csv(RESULTS / f"core7_frontier{TAG}.csv", index=False)

cols = ["ann_return_%", "ann_vol_%", "sharpe", "max_dd_%", "turnover_x"]
flags = ["val_frontier"] + (["val_turnover_frontier"] if TURNOVER else [])
show = table[["rule", "family"] + flags + [f"val_{c}" for c in cols] + [f"test_{c}" for c in cols]]
pd.set_option("display.width", 220)
print(f"\nPooled {TARGET} forecasts, 4 panel sets combined, {COSTS}; validation 2021-2023, test 2022-2024 (walk-forward).")
print("val_frontier = on the return/volatility frontier of the validation years (rules chosen there).")
if TURNOVER:
    print("val_turnover_frontier = on the return/turnover frontier of the validation years.")
print("turnover_x = dollars traded per year / average account value.\n")
print(show.sort_values(["val_frontier", "val_ann_return_%"], ascending=[False, False]).round(2).to_string(index=False))

sig = table[["rule", "test_ann_return_%", "test_ann_return_ci_lo", "test_ann_return_ci_hi", "test_sharpe",
             "test_sharpe_ci_lo", "test_sharpe_ci_hi", "test_psr", "test_dsr"]]
sig.columns = ["rule", "ann_ret_%", "ret_lo", "ret_hi", "sharpe", "sharpe_lo", "sharpe_hi", "PSR", "DSR"]
print(f"\nSignificance of the TEST results (2022-2024, {COSTS}). lo/hi = 95% stationary block bootstrap "
      f"(mean block {BOOT_BLOCK} days, {BOOT_RESAMPLES} resamples).")
print("PSR = probability the true Sharpe is above 0; DSR = the same after allowing for having tried "
      f"N = {n_trials} rules (a lower bound on the trials looked at). Above 0.95 is the usual bar.\n")
print(sig.sort_values("sharpe", ascending=False).round(2).to_string(index=False))
print(f"\nSaved {RESULTS / f'core7_frontier{TAG}.csv'}")

# ---- each walk-forward fold on its own (4 sets combined) ----
year_rows = []
for name, family, strategy, kwargs in RULES:
    for split in SPLITS:
        for y in YEARS:
            per_set = yearly[(split, name, y)]
            n = min(len(x) for x in per_set)
            m = metrics(np.mean([x[:n] for x in per_set], axis=0))
            year_rows.append({"rule": name, "family": family, "split": split, "fold": y,
                              "year": y - 1 if split == "val" else y, **m,
                              "turnover_x": float(np.mean(yearly_turnover[(split, name, y)]))})
by_year = pd.DataFrame(year_rows)
by_year.to_csv(RESULTS / f"core7_frontier{TAG}_by_year.csv", index=False)

order = table.sort_values("test_sharpe", ascending=False)["rule"]
for col, label in (("sharpe", "Sharpe"), ("ann_return_%", "annualized return (%)")):
    wide = by_year.pivot_table(index="rule", columns=["split", "year"], values=col).reindex(order)
    wide.columns = [f"{sp} {yr}" for sp, yr in wide.columns]
    wide = wide[[f"val {y - 1}" for y in YEARS] + [f"test {y}" for y in YEARS]]
    wide["test all"] = table.set_index("rule").loc[order, f"test_{col}"]
    print(f"\n{label} per year (validation year = fold - 1; 4 sets combined), sorted by test Sharpe:")
    print(wide.round(2).to_string())
print(f"\nSaved {RESULTS / f'core7_frontier{TAG}_by_year.csv'}")

# ---- each calendar quarter (4 sets combined) ----
quarter_rows = []
for name, family, strategy, kwargs in RULES:
    for split in SPLITS:
        for y in YEARS:
            per_set, per_dates = yearly[(split, name, y)], day_dates[(split, name, y)]
            n = min(len(x) for x in per_set)
            book = pd.Series(np.mean([x[:n] for x in per_set], axis=0), index=per_dates[0][:n])
            for q, r in book.groupby(book.index.to_period("Q")):
                if len(r) < 20:          # a stray day or two at a fold's edge
                    continue
                vol = r.std() * np.sqrt(TRADING_DAYS)
                value = (1 + r).cumprod()
                quarter_rows.append({"rule": name, "family": family, "split": split, "fold": y,
                                     "quarter": str(q), "days": len(r),
                                     "return_%": (value.iloc[-1] - 1) * 100,
                                     "sharpe": r.mean() * TRADING_DAYS / vol if vol > 1e-12 else np.nan,
                                     "max_dd_%": (1 - value / value.cummax()).max() * 100})
by_quarter = pd.DataFrame(quarter_rows)
by_quarter.to_csv(RESULTS / f"core7_frontier{TAG}_by_quarter.csv", index=False)

wide = by_quarter[by_quarter["split"] == "test"].pivot_table(index="rule", columns="quarter",
                                                             values="return_%").reindex(order)
print("\nTest return (%) per quarter (not annualized; 4 sets combined), sorted by test Sharpe:")
print(wide.round(1).to_string())
print(f"\nSaved {RESULTS / f'core7_frontier{TAG}_by_quarter.csv'}")

# ---- walk-forward rule selection on the test years ----
TOP = 3


def test_book(name, y):
    per_set = yearly[("test", name, y)]
    n = min(len(x) for x in per_set)
    return np.mean([x[:n] for x in per_set], axis=0)


def sharpe(r):
    sd = np.std(r)
    return np.mean(r) / sd * np.sqrt(TRADING_DAYS) if sd > 1e-12 else np.nan


books = {(name, y): test_book(name, y) for name, *_ in RULES for y in YEARS}
choices, picked_top1, picked_top3 = [], [], []
for i, y in enumerate(YEARS[1:], start=1):
    past = YEARS[:i]
    score = pd.Series({name: sharpe(np.concatenate([books[(name, p)] for p in past]))
                       for name, *_ in RULES}).sort_values(ascending=False)
    top = list(score.index[:TOP])
    r1 = books[(top[0], y)]
    n = min(len(books[(t, y)]) for t in top)
    r3 = np.mean([books[(t, y)][:n] for t in top], axis=0)
    picked_top1.append(r1)
    picked_top3.append(r3)
    for k, t in enumerate(top, 1):
        m = metrics(books[(t, y)])
        choices.append({"test_year": y, "chosen_on": "+".join(map(str, past)), "rank": k, "rule": t,
                        "past_sharpe": score[t], "ann_return_%": m["ann_return_%"],
                        "sharpe": m["sharpe"], "max_dd_%": m["max_dd_%"]})
    m3 = metrics(r3)
    choices.append({"test_year": y, "chosen_on": "+".join(map(str, past)), "rank": 0,
                    "rule": f"top-{TOP} book", "past_sharpe": np.nan, "ann_return_%": m3["ann_return_%"],
                    "sharpe": m3["sharpe"], "max_dd_%": m3["max_dd_%"]})
walk = pd.DataFrame(choices)
walk.to_csv(RESULTS / f"core7_frontier{TAG}_walkforward.csv", index=False)

later = YEARS[1:]
summary = {"walk-forward best rule": np.concatenate(picked_top1),
           f"walk-forward top-{TOP} book": np.concatenate(picked_top3),
           "equal weight": np.concatenate([books[("equal weight", y)] for y in later])}
hindsight = max((name for name, *_ in RULES),
                key=lambda nm: sharpe(np.concatenate([books[(nm, y)] for y in later])))
summary[f"best fixed rule in hindsight ({hindsight})"] = np.concatenate([books[(hindsight, y)] for y in later])
print(f"\nWalk-forward rule selection on the TEST years only ({TARGET}, {COSTS}): each year's rule is the")
print("one with the best Sharpe over the earlier test years.")
print(walk.round(2).to_string(index=False))
print(f"\nCombined {later[0]}-{later[-1]}:")
combined = pd.DataFrame({k: {**metrics(v), **bootstrap_ci(v), "psr": probabilistic_sharpe(v)}
                         for k, v in summary.items()}).T
print(combined[["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%", "sharpe",
                "sharpe_ci_lo", "sharpe_ci_hi", "psr", "max_dd_%"]].astype(float).round(2).to_string())
print("ci_lo / ci_hi = 95% stationary block bootstrap; psr = probability the true Sharpe is above 0. "
      "The hindsight rule is chosen on these years, so its interval and PSR are optimistic.")
print(f"\nSaved {RESULTS / f'core7_frontier{TAG}_walkforward.csv'}")

# ---- plot: validation frontier (left), the same rules on test (right) ----
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    families = list(dict.fromkeys(table["family"]))
    markers = dict(zip(families, "osD^vPX*"))
    panels = [("ann_vol_%", "annualized volatility (%)", "val_frontier")]
    if TURNOVER:
        panels.append(("turnover_x", "annual turnover (x account value)", "val_turnover_frontier"))
    fig, axes = plt.subplots(len(panels), 2, figsize=(13, 5.5 * len(panels)), squeeze=False)
    for row_axes, (xcol, xlabel, flag) in zip(axes, panels):
        for ax, split in zip(row_axes, SPLITS):
            for fam in families:
                part = table[table["family"] == fam]
                ax.scatter(part[f"{split}_{xcol}"], part[f"{split}_ann_return_%"], marker=markers[fam],
                           s=45, alpha=0.75, label=fam)
            front = table[table[flag]].sort_values(f"{split}_{xcol}")
            ax.plot(front[f"{split}_{xcol}"], front[f"{split}_ann_return_%"], "k--", lw=1,
                    label="validation frontier rules")
            for _, r in front.iterrows():
                ax.annotate(r["rule"], (r[f"{split}_{xcol}"], r[f"{split}_ann_return_%"]),
                            fontsize=7, xytext=(4, 3), textcoords="offset points")
            ax.axhline(0, color="grey", lw=0.6)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("annualized return (%)")
            ax.set_title("Validation 2021-2023 (frontier chosen here)" if split == "val"
                         else "Test 2022-2024 (same rules)")
        row_axes[0].legend(fontsize=8, loc="best")
    rules = "daily long-short, overlapping and banded rules" if TURNOVER else "daily long-short (long L / short S)"
    fig.suptitle(f"Pooled EXAMM, {TARGET} target: {rules}, {COSTS}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"core7_frontier{TAG}.png", dpi=150)
    print(f"Saved {RESULTS / f'core7_frontier{TAG}.png'}")
except ImportError:
    print("matplotlib not installed -- skipped the plot (pip3 install matplotlib)")
