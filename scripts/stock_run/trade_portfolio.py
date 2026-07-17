#!/usr/bin/env python3
"""Trade the 50-stock EXAMM baseline through Financial_toolbox portfolio strategies.

Runs the professor's daily long-short strategy (rank all stocks by predicted
return each day, long the top N / short the bottom M) on the held-out test
predictions of each stock's validation-selected genome, and compares against
an equal-weight buy-and-hold of the same universe and the S&P (sprtrn).

Usage:
    python3 scripts/stock_run/trade_portfolio.py \
        --pred-dir test_output/baseline_best/predictions \
        --align suffix --expect-n 50 --long 10 --short 10

    add --oracle   to trade on expected_RET (perfect-foresight upper bound)
    add --use-tc   to charge transaction costs
    add --debug    to watch every trade

Requires predict_selected.sh output (denormalized predictions CSVs).

IMPORTANT: the 50 test CSVs end on the same date but START on different dates.
Her portfolio code indexes every stock by one shared day counter, so unaligned
inputs would silently trade different calendar dates against each other.
--align suffix trims all stocks to the common trailing window and verifies the
trimmed date vectors are identical before trading. Default is --align none,
which refuses to run on unequal lengths.
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- locate the toolbox (same convention as trade_stock.py) -----------------
TOOLBOX = Path(os.getenv("FIN_TOOLBOX_DIR", Path.home() / "Financial_toolbox"))
if not TOOLBOX.is_dir():
    sys.exit(f"Financial_toolbox not found at {TOOLBOX} (set FIN_TOOLBOX_DIR)")
sys.path.insert(0, str(TOOLBOX))
from portfolio import Portfolio  # noqa: E402  (import after path setup)
from stock import Stock          # noqa: E402
from Logger import Logger        # noqa: E402

REPO = Path(__file__).resolve().parents[2]   # EXAMM-Extended root

# strategies Portfolio.trade() can actually dispatch; excludes
# daily_zimeng_long_short_return (dispatched but the method doesn't exist)
STRATEGIES = [
    "daily_long_short_return",
    "daily_long_return",
    "long_short_return",
    "simple_return",
    "portfolio_simple_return",
    "portfolio_shorting_return",
]
# strategies whose long/short counts are actually consumed (and divided by)
COUNTED_STRATEGIES = {"daily_long_short_return", "daily_long_return"}


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def discover_tickers(pred_dir, tickers, exclude):
    available = sorted(p.name[: -len("_test_predictions.csv")]
                       for p in pred_dir.glob("*_test_predictions.csv"))
    if not available:
        fail(f"no *_test_predictions.csv files in {pred_dir}")
    if tickers:
        if len(set(tickers)) != len(tickers):
            fail("duplicate ticker in --tickers")
        missing = sorted(set(tickers) - set(available))
        if missing:
            fail(f"--tickers not found in {pred_dir}: {missing}")
        chosen = sorted(tickers)
    else:
        chosen = available
        manifest = pred_dir / "MANIFEST.csv"
        if manifest.is_file():
            listed = sorted(pd.read_csv(manifest)["ticker"].astype(str))
            if listed != available:
                fail(f"MANIFEST.csv lists {len(listed)} tickers but folder has "
                     f"{len(available)} -- partial download or stale files? "
                     f"missing={sorted(set(listed) - set(available))} "
                     f"extra={sorted(set(available) - set(listed))}")
    if exclude:
        unknown = sorted(set(exclude) - set(chosen))
        if unknown:
            fail(f"--exclude ticker(s) not in universe: {unknown}")
        chosen = [t for t in chosen if t not in set(exclude)]
    if not chosen:
        fail("universe is empty after --exclude")
    return chosen


def check_finite(name, ticker, arr):
    bad = np.flatnonzero(~np.isfinite(arr))
    if len(bad):
        fail(f"{ticker}: non-finite {name} at row {bad[0]} "
             f"(NaN predictions would be silently longed by the strategy)")


def load_stock(ticker, pred_dir, data_dir, oracle):
    pred_file = pred_dir / f"{ticker}_test_predictions.csv"
    test_file = data_dir / f"{ticker}_test.csv"
    if not test_file.is_file():
        fail(f"missing test set {test_file}")

    pred = pd.read_csv(pred_file)
    for col in ("expected_RET", "predicted_RET"):
        if col not in pred.columns:
            fail(f"{ticker}: column {col} missing from {pred_file}")
    test = pd.read_csv(test_file)
    for col in ("date", "RET", "sprtrn", "PRC", "TRAN_COST", "ASK", "BID"):
        if col not in test.columns:
            fail(f"{ticker}: column {col} missing from {test_file}")

    stock = Stock(ticker)
    stock.read_return_prediction(str(pred_file))
    stock.read_stock_price(str(test_file))
    if oracle:  # perfect foresight, without touching her code
        stock.return_prediction = pred["expected_RET"].to_numpy()
        stock.testing_period = len(stock.return_prediction)

    expected = pred["expected_RET"].to_numpy()
    raw_ret = test["RET"].to_numpy()
    n_pred = len(stock.return_prediction)
    n_price = len(stock.stock_price)

    # hard validations, per stock ------------------------------------------
    check_finite("predicted_RET", ticker, pred["predicted_RET"].to_numpy())
    check_finite("expected_RET", ticker, expected)
    check_finite("RET", ticker, raw_ret)       # NaN here would ALSO defeat the
    check_finite("sprtrn", ticker, test["sprtrn"].to_numpy())  # pairing check below
    check_finite("PRC", ticker, stock.stock_price)
    check_finite("TRAN_COST", ticker, stock.transaction_cost)
    check_finite("ASK", ticker, stock.ask_price)
    check_finite("BID", ticker, stock.bid_price)
    if (stock.stock_price <= 0).any():
        row = int(np.flatnonzero(stock.stock_price <= 0)[0])
        fail(f"{ticker}: nonpositive PRC at row {row} (CRSP bid-ask flag?)")
    if n_price != n_pred + 1:
        fail(f"{ticker}: {n_pred} predictions vs {n_price} prices "
             f"(expected prices = predictions + 1)")
    # pairing self-check: expected_RET[i] must equal raw RET[i+1] (they match
    # to ~1e-20 when the prediction file really belongs to this test file).
    # Written as "not <=" so a NaN comparison can never sneak past the check.
    err = np.abs(expected - raw_ret[1:]).max()
    if not (err <= 1e-10):
        fail(f"{ticker}: expected_RET does not match test RET[t+1] "
             f"(max diff {err:.3e}) -- wrong prediction/test pairing?")

    return {
        "ticker": ticker,
        "stock": stock,
        "dates": test["date"].astype(str).to_numpy(),
        "sprtrn": test["sprtrn"].to_numpy(),
        "n_pred": n_pred,
    }


def align_suffix(records):
    """Trim every stock to the common trailing window (all files share the
    same END date but start on different dates)."""
    k = min(r["n_pred"] for r in records)
    for r in records:
        s = r["n_pred"] - k
        stock = r["stock"]
        stock.return_prediction = stock.return_prediction[s:]
        stock.stock_price = stock.stock_price[s:]          # k + 1 prices
        stock.transaction_cost = stock.transaction_cost[s:]
        stock.ask_price = stock.ask_price[s:]
        stock.bid_price = stock.bid_price[s:]
        stock.testing_period = k                            # only set on read!
        r["dates"] = r["dates"][s:]
        r["sprtrn"] = r["sprtrn"][s:]
        r["dropped"] = s
        r["n_pred"] = k
    return k


def verify_market_agreement(records):
    """Every stock's (trimmed) date vector must be identical -- otherwise
    stocks would trade against each other on different calendar days -- and
    their sprtrn columns must agree about the market. Runs in EVERY mode,
    not just --align suffix."""
    ref = records[0]
    for r in records[1:]:
        if not np.array_equal(r["dates"], ref["dates"]):
            i = int(np.flatnonzero(r["dates"] != ref["dates"])[0])
            fail(f"date mismatch across stocks: {r['ticker']}[{i}]="
                 f"{r['dates'][i]} vs {ref['ticker']}[{i}]={ref['dates'][i]}")
        if not np.allclose(r["sprtrn"], ref["sprtrn"], atol=1e-12):
            fail(f"sprtrn mismatch between {r['ticker']} and {ref['ticker']} "
                 f"-- data files disagree about the market")


def benchmarks(records):
    """Equal-weight buy-and-hold of the universe + compounded S&P, both over
    the exact window the strategy trades: PRC[0] -> PRC[-2] (her do-nothing
    convention; do_nothing_return.py's own cross-stock average is broken)."""
    per_stock = {}
    for r in records:
        p = r["stock"].stock_price
        per_stock[r["ticker"]] = (p[-2] - p[0]) / p[0] * 100
    bh = sum(per_stock.values()) / len(per_stock)

    # sprtrn[i] is the market return from day i-1 to day i; the strategy's
    # window PRC[0]..PRC[-2] spans rows 1 .. len(prices)-2
    sp = records[0]["sprtrn"]
    n_prices = len(records[0]["stock"].stock_price)
    sp_return = (np.prod(1.0 + sp[1 : n_prices - 1]) - 1.0) * 100
    return bh, sp_return, per_stock


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, default=REPO / "datasets/701515_split")
    ap.add_argument("--tickers", nargs="+", default=None)
    ap.add_argument("--exclude", nargs="+", default=None)
    ap.add_argument("--capital", type=float, default=1000.0)
    ap.add_argument("--long", type=int, default=10, dest="long_n")
    ap.add_argument("--short", type=int, default=10, dest="short_n")
    ap.add_argument("--strategy", choices=STRATEGIES, default="daily_long_short_return")
    ap.add_argument("--align", choices=["none", "suffix"], default="none")
    ap.add_argument("--use-tc", action="store_true")
    ap.add_argument("--bid-ask", action="store_true")
    ap.add_argument("--oracle", action="store_true",
                    help="trade on expected_RET instead of predictions (upper bound)")
    ap.add_argument("--debug", action="store_true", help="print every trade")
    ap.add_argument("--per-stock", action="store_true",
                    help="print per-stock buy-and-hold table")
    ap.add_argument("--expect-n", type=int, default=None,
                    help="hard-fail unless the final universe has exactly N stocks")
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.pred_dir.is_dir():
        fail(f"--pred-dir {args.pred_dir} is not a directory")
    if args.use_tc and args.bid_ask:
        fail("--use-tc and --bid-ask are mutually exclusive (the toolbox's "
             "if/elif makes TC silently win, so the run would be mislabeled)")

    tickers = discover_tickers(args.pred_dir, args.tickers, args.exclude)
    if args.expect_n is not None and len(tickers) != args.expect_n:
        fail(f"universe has {len(tickers)} stocks, --expect-n {args.expect_n} "
             f"-- partial predictions folder?")

    if args.strategy in COUNTED_STRATEGIES:
        if args.long_n < 1 or args.short_n < 1:
            fail("--long and --short must both be >= 1 "
                 "(the toolbox divides by them)")
        if args.long_n + args.short_n > len(tickers):
            fail(f"--long {args.long_n} + --short {args.short_n} exceeds the "
                 f"universe ({len(tickers)} stocks)")
    if args.strategy == "daily_long_return" and len(tickers) < 30:
        fail("daily_long_return hardcodes a 30-stock universe in the toolbox "
             "(portfolio.py range(30)) -- needs >= 30 stocks")

    records = [load_stock(t, args.pred_dir, args.data_dir, args.oracle)
               for t in tickers]

    lengths = sorted({r["n_pred"] for r in records})
    if len(lengths) > 1:
        if args.align == "none":
            detail = ", ".join(f"{r['ticker']}={r['n_pred']}" for r in records[:6])
            fail(f"test windows differ in length ({detail}, ...) -- the stocks "
                 f"start on different dates. Rerun with --align suffix to trim "
                 f"everything to the common trailing window.")
        k = align_suffix(records)
    else:
        k = lengths[0]
        for r in records:
            r["dropped"] = 0
    verify_market_agreement(records)   # in every mode, aligned or not
    if k < 2:
        fail(f"common window has only {k} prediction day(s) -- the toolbox "
             f"strategies need at least 2")

    # final invariant before handing over to her code (explicit, not assert:
    # asserts vanish under python -O)
    for r in records:
        stock = r["stock"]
        if not (stock.testing_period == len(stock.return_prediction) == k
                and len(stock.stock_price) == k + 1):
            fail(f"{r['ticker']}: internal alignment invariant violated "
                 f"(testing_period={stock.testing_period}, "
                 f"n_pred={len(stock.return_prediction)}, "
                 f"n_price={len(stock.stock_price)}, k={k})")

    logger = Logger("DEBUG" if args.debug else "NONE")
    portfolio = Portfolio(tickers)
    portfolio.set_initial_spend_per_stock(args.capital / len(tickers))
    portfolio.set_initial_capital(args.capital)
    portfolio.set_logger(logger)
    for r in records:
        r["stock"].set_logger(logger)
        portfolio.add_company_to_protfolio(r["stock"])   # (sic)
    portfolio.set_use_TC(args.use_tc)
    portfolio.set_trade_with_bid_ask(args.bid_ask)

    strat_return = portfolio.trade(args.strategy, args.long_n, args.short_n)
    final = args.capital * (1.0 + strat_return / 100.0)
    bh, sp_return, per_stock = benchmarks(records)

    dates = records[0]["dates"]
    mode = "oracle (expected_RET)" if args.oracle else "model (predicted_RET)"
    dropped = {r["ticker"]: r["dropped"] for r in records if r["dropped"]}
    print()
    print(f"universe: {len(tickers)} stocks   window: {dates[0]} .. {dates[-2]} "
          f"({k} trading days)   mode: {mode}")
    if dropped:
        print(f"suffix-aligned: dropped leading rows for {len(dropped)} stocks "
              f"(max {max(dropped.values())})")
    counted = args.strategy in COUNTED_STRATEGIES
    params = f" (long {args.long_n} / short {args.short_n}, " if counted else " ("
    params += f"{'TC on' if args.use_tc else 'TC off'}, "
    params += f"{'bid-ask' if args.bid_ask else 'PRC'} prices)"
    print(f"strategy {args.strategy}{params}")
    print(f"  strategy return : {strat_return:+9.2f}%   "
          f"(final ${final:,.2f} from ${args.capital:,.0f})")
    print(f"  equal-weight B&H: {bh:+9.2f}%")
    print(f"  S&P (sprtrn)    : {sp_return:+9.2f}%")
    if args.strategy in ("daily_long_short_return", "long_short_return",
                         "portfolio_simple_return"):
        # these strategies clear their final positions at the same price they
        # were bought (toolbox reuses the last loop index), so their P&L stops
        # one day earlier than the benchmarks' PRC[0]->PRC[-2] window -- one
        # extra market day for the benchmarks, i.e. conservative for us
        print("  (note: benchmarks include one final day this strategy "
              "clears flat)")

    if args.per_stock:
        print("\nper-stock buy-and-hold over the same window:")
        for t in tickers:
            print(f"  {t:6s} {per_stock[t]:+9.2f}%")

    if args.csv_out:
        row = pd.DataFrame([{
            "strategy": args.strategy, "mode": "oracle" if args.oracle else "model",
            "universe": len(tickers), "days": k,
            "window_start": dates[0], "window_end": dates[-2],
            "long": args.long_n, "short": args.short_n,
            "use_tc": args.use_tc, "bid_ask": args.bid_ask,
            "capital": args.capital, "strategy_return_pct": strat_return,
            "buy_and_hold_pct": bh, "sp_return_pct": sp_return,
        }])
        row.to_csv(args.csv_out, index=False)
        print(f"\nsummary written to {args.csv_out}")


if __name__ == "__main__":
    main()
