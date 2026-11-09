import contextlib
import io
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from portfolio import Portfolio
from stock import Stock
from Logger import Logger

# RET_protocol.md, ORIGINAL RULES ONLY: the 16 daily long-short rules (L, S in 3, 5, 10, 15) and the
# equal-weight benchmark -- the rule list from before the overlapping / banded rules were added.
# Same forecasts, engine, costs, primary result (daily L/S 5/5), walk-forward selection and metrics
# as the protocol; the deflated Sharpe ratio counts N = 17 rules.
#
# Uses the prediction / price files main_core7.py builds (RET target, global best genome):
#     python3 main_core7.py set1 2022  ... for set1-set4 and 2022-2024
# Run:  python3 main_core7_protocol.py


def trade_rule(portfolio, strategy, long_company_number, short_company_number):
    """Trade one rule; return its daily returns and annual turnover. The account values are the
    daily_values portfolio.py records (copied when calculate_return() is reached), the turnover is
    the dollars bought, sold and shorted. The per-day trading log is kept off the screen."""
    state = {"traded": 0.0}
    restore = []

    def wrap(obj, method, make):
        setattr(obj, method, make(getattr(obj, method)))
        restore.append((obj, method))

    for company in portfolio.portfolio_list:
        def clear(original, c=company):
            def f(time):
                state["traded"] += abs(c.get_share() * c.get_stock_price(time))
                return original(time)
            return f

        def opening(original):
            def f(cash_amount, time):
                state["traded"] += abs(cash_amount)
                return original(cash_amount, time)
            return f

        wrap(company, "clear_holdings", clear)
        wrap(company, "buy_stock", opening)
        wrap(company, "short_stock", opening)

    def capture(original):
        def f(*a, **k):
            state["values"] = list(getattr(portfolio, "daily_values", []))
            return original(*a, **k)
        return f

    wrap(portfolio, "calculate_return", capture)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            portfolio.trade(strategy, long_company_number, short_company_number)
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
    years = len(r) / TRADING_DAYS
    vol = r.std() * math.sqrt(TRADING_DAYS)
    return {"ann_return_%": (value[-1] ** (1 / years) - 1) * 100 if value[-1] > 0 else np.nan,
            "ann_vol_%": vol * 100,
            "sharpe": r.mean() * TRADING_DAYS / vol if vol > 1e-12 else np.nan,
            "max_dd_%": (1 - value / np.maximum.accumulate(value)).max() * 100}


def bootstrap_ci(r, resamples=2000, block=10, seed=0):
    """95% stationary block bootstrap intervals of the annualized Sharpe and return (%)."""
    r = np.asarray(r, dtype=float)
    n = len(r)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(resamples, n))
    restart = rng.random((resamples, n)) < 1 / block
    idx = np.empty((resamples, n), dtype=int)
    idx[:, 0] = starts[:, 0]
    for t in range(1, n):
        idx[:, t] = np.where(restart[:, t], starts[:, t], (idx[:, t - 1] + 1) % n)
    boot = r[idx]
    sd = boot.std(axis=1)
    sharpes = np.where(sd > 1e-12, boot.mean(axis=1) / np.where(sd > 1e-12, sd, 1) * math.sqrt(TRADING_DAYS), np.nan)
    growth = np.prod(1 + boot, axis=1)
    returns = np.where(growth > 0, (np.abs(growth) ** (TRADING_DAYS / n) - 1) * 100, np.nan)
    return {"sharpe_ci_lo": np.nanpercentile(sharpes, 2.5), "sharpe_ci_hi": np.nanpercentile(sharpes, 97.5),
            "ann_return_ci_lo": np.nanpercentile(returns, 2.5), "ann_return_ci_hi": np.nanpercentile(returns, 97.5)}


def daily_sharpe_moments(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    if sd <= 1e-12:
        return np.nan, 0.0, 3.0
    z = (r - r.mean()) / sd
    return r.mean() / sd, float((z ** 3).mean()), float((z ** 4).mean())


def probabilistic_sharpe(r, benchmark=0.0):
    """P(true daily Sharpe > benchmark) (Bailey & Lopez de Prado 2012)."""
    sr, skew, kurt = daily_sharpe_moments(r)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if not np.isfinite(sr) or denom <= 0:
        return np.nan
    return PHI.cdf((sr - benchmark) * math.sqrt(len(r) - 1) / math.sqrt(denom))


def deflated_sharpe(r, trial_sharpes, n_trials):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014) against the best of n_trials no-skill rules."""
    trial = np.asarray([x for x in trial_sharpes if np.isfinite(x)], dtype=float)
    expected_max = math.sqrt(trial.var(ddof=1)) * ((1 - EULER_GAMMA) * PHI.inv_cdf(1 - 1 / n_trials)
                                                   + EULER_GAMMA * PHI.inv_cdf(1 - 1 / (n_trials * math.e)))
    return probabilistic_sharpe(r, benchmark=expected_max)


def annual_sharpe(r):
    sd = np.std(r)
    return np.mean(r) / sd * math.sqrt(TRADING_DAYS) if sd > 1e-12 else np.nan


TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
PHI = NormalDist()

if __name__ == '__main__':
    years = [2022, 2023, 2024]
    panel_sets = ['set1', 'set2', 'set3', 'set4']
    genome_selection = 'best'          # the global best genome (lowest validation MSE), as main_core7.py
    results_root = Path(__file__).resolve().parent / "results"

    # the original rules: daily long-short, long L / short S, and the equal-weight benchmark
    counts = [3, 5, 10, 15]
    rules = [(f"L/S {L}/{S}", 'daily_long_short_return', L, S) for L in counts for S in counts]
    rules.append(("equal weight", 'daily_equal_weight_return', 5, 5))
    primary = "L/S 5/5"
    top = 3

    initial_capital = 1000
    spend_per_stock = 100
    use_TC = True
    trade_with_bid_ask = False

    # --------------------------------- trading ---------------------------------
    # every rule in every set and test year; the 4 sets are combined with equal weight per year
    logger = Logger()
    logger.set_level('ERROR')
    daily = {}                        # (rule, year) -> list of the 4 sets' daily returns
    turnover = {}                     # rule -> list of annual turnovers (set x year)
    for year in years:
        run = f"core7_examm_ret_{year}"
        for panel_set in panel_sets:
            PREDICTION_DIR = results_root / f"{run}_{panel_set}_{genome_selection}_test_prediction"
            TEST_DIR = results_root / f"{run}_{panel_set}_test_prices"
            stock_names = sorted(f.name[:-len("_predictions.csv")] for f in PREDICTION_DIR.glob("*_predictions.csv"))
            if not stock_names or not TEST_DIR.exists():
                sys.exit(f"missing {PREDICTION_DIR.name} -- build it with:  python3 main_core7.py {panel_set} {year}")

            with contextlib.redirect_stdout(io.StringIO()):
                portfolio = Portfolio(stock_names)
                portfolio.set_initial_spend_per_stock(spend_per_stock)
                portfolio.set_initial_capital(initial_capital)
                portfolio.set_logger(logger)
                for stock_name in stock_names:
                    company = Stock(stock_name)
                    company.read_return_prediction(str(PREDICTION_DIR / f"{stock_name}_predictions.csv"))
                    company.read_stock_price(str(TEST_DIR / f"{stock_name}.csv"))
                    portfolio.add_company_to_protfolio(company)
                    company.set_logger(logger)
            portfolio.set_use_TC(use_TC)
            portfolio.set_trade_with_bid_ask(trade_with_bid_ask)

            for name, strategy, long_company_number, short_company_number in rules:
                r, t = trade_rule(portfolio, strategy, long_company_number, short_company_number)
                daily.setdefault((name, year), []).append(r)
                turnover.setdefault(name, []).append(t)
            print(f"{year} {panel_set}: {len(rules)} rules traded")

    def book(name, year):
        """One year of a rule, the 4 sets combined with equal weight."""
        per_set = daily[(name, year)]
        n = min(len(x) for x in per_set)
        return np.mean([x[:n] for x in per_set], axis=0)

    books = {(name, year): book(name, year) for name, *_ in rules for year in years}
    joined = {name: np.concatenate([books[(name, y)] for y in years]) for name, *_ in rules}

    # --------------------------------- every rule, test 2022-2024 ---------------------------------
    trial_sharpes = [daily_sharpe_moments(joined[name])[0] for name, *_ in rules]
    table = []
    for name, *_ in rules:
        r = joined[name]
        table.append({"rule": name, **metrics(r), "turnover_x": np.mean(turnover[name]), **bootstrap_ci(r),
                      "psr": probabilistic_sharpe(r), "dsr": deflated_sharpe(r, trial_sharpes, len(rules)),
                      **{f"sharpe_{y}": annual_sharpe(books[(name, y)]) for y in years}})
    table = pd.DataFrame(table).sort_values("sharpe", ascending=False)

    # --------------------------------- walk-forward rule selection ---------------------------------
    # each test year after the first: rank the rules by Sharpe over the EARLIER test years
    choices, top1, top3 = [], [], []
    for i, year in enumerate(years[1:], start=1):
        past = years[:i]
        score = pd.Series({name: annual_sharpe(np.concatenate([books[(name, p)] for p in past]))
                           for name, *_ in rules}).sort_values(ascending=False)
        chosen = list(score.index[:top])
        top1.append(books[(chosen[0], year)])
        n = min(len(books[(c, year)]) for c in chosen)
        top3.append(np.mean([books[(c, year)][:n] for c in chosen], axis=0))
        for k, c in enumerate(chosen, 1):
            m = metrics(books[(c, year)])
            choices.append({"test_year": year, "chosen_on": "+".join(map(str, past)), "rank": k, "rule": c,
                            "past_sharpe": score[c], "ann_return_%": m["ann_return_%"], "sharpe": m["sharpe"]})

    later = years[1:]
    headline = {
        f"PRIMARY (pre-set): {primary}, {years[0]}-{years[-1]}": joined[primary],
        f"equal weight, {years[0]}-{years[-1]}": joined["equal weight"],
        f"walk-forward best rule, {later[0]}-{later[-1]}": np.concatenate(top1),
        f"walk-forward top-{top} book, {later[0]}-{later[-1]}": np.concatenate(top3),
        f"equal weight, {later[0]}-{later[-1]}": np.concatenate([books[("equal weight", y)] for y in later]),
    }
    head = pd.DataFrame({k: {**metrics(v), **bootstrap_ci(v), "psr": probabilistic_sharpe(v)}
                         for k, v in headline.items()}).T

    # --------------------------------- report ---------------------------------
    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print(f"RET_protocol.md, ORIGINAL RULES ONLY ({len(rules)} rules: 16 daily long-short + equal weight)")
    print(f"Pooled RET, global best genome per set / fold, test {years[0]}-{years[-1]}, costs "
          f"{'on' if use_TC else 'off'}, 4 sets combined")
    print("=" * 100)
    for k, v in headline.items():
        print(f"Portfolio return for {k} is {metrics(v)['ann_return_%']:.2f}% a year")
    print()
    print(head[["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%", "sharpe",
                "sharpe_ci_lo", "sharpe_ci_hi", "psr", "max_dd_%"]].astype(float).round(2).to_string())
    print("ci_lo / ci_hi = 95% stationary block bootstrap (mean block 10 days, 2000 resamples, seed 0); "
          "psr = P(true Sharpe > 0), 0.95 the usual bar.")

    print("\nWalk-forward choices (best Sharpe over the earlier test years) and how they did:")
    print(pd.DataFrame(choices).round(2).to_string(index=False))

    print(f"\nEvery rule, test {years[0]}-{years[-1]} (dsr = deflated Sharpe ratio, N = {len(rules)} rules):")
    show = ["rule", "ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "sharpe", "sharpe_ci_lo",
            "sharpe_ci_hi", "psr", "dsr", "max_dd_%", "turnover_x"] + [f"sharpe_{y}" for y in years]
    print(table[show].round(2).to_string(index=False))

    results_root.mkdir(exist_ok=True)
    table.to_csv(results_root / "core7_protocol_original_rules.csv", index=False)
    head.to_csv(results_root / "core7_protocol_original_rules_headline.csv")
    pd.DataFrame(choices).to_csv(results_root / "core7_protocol_original_rules_walkforward.csv", index=False)
    print(f"\nSaved {results_root / 'core7_protocol_original_rules.csv'}, _headline.csv and _walkforward.csv")
