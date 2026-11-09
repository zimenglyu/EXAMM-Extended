#!/usr/bin/env python3
"""Genome Pareto frontier on the paper's Train/Val/Test folds (professor's design).

Forecasts: the existing Pooled EXAMM RET runs, folds 2022, 2023, 2024 (train <= Y-2, validation Y-1,
test Y). Candidates per fold: all 40 genomes (4 source panel sets x 10 repeats, each repeat's final
global best), each used as ONE genome shared by all 4 panel sets (200 stocks).

For every candidate, on the VALIDATION year and on the TEST year, over the 4 panel sets:
  - MSE   : mean squared error of the forecast in raw daily-return units (prediction converted with
            the panel set's median / IQR from panel_meta.json, scored against the unclipped raw
            return, as core7_ic.py), averaged over the 4 sets; Rank IC alongside (information);
  - trading: the FIXED rule overlap 5/5 hold 10 (portfolio.py overlapping_return: each day a new
            long-top-5 / short-bottom-5 book held 10 days), costs on, capital 1000, the 4 sets
            combined with equal weight -> net annualized return, Sharpe, max drawdown.
Pareto frontier on VALIDATION: lower MSE and higher net return. One genome per fold is chosen on
VALIDATION only, by a rule fixed before running: among the frontier genomes, the lowest sum of its
rank by validation MSE (ascending) and by validation return (descending), both ranked over all
candidates; ties -> lower validation MSE. The two frontier ends are recorded too (lowest validation
MSE = EXAMM's own criterion; highest validation return).
Reported on TEST once: the chosen genome per fold (2022-2024 joined and per year) with 95% block
bootstrap intervals and PSR, against the two frontier ends, the per-set EXAMM default (each set's own
lowest-validation-MSE genome on its own set) and the equal-weight benchmark.

Needs the predictions of evaluate_rnn_core7_ret_cross.sh (every genome on every set, val and test).
Run on the Mac:   cd ~/Financial_toolbox && python3 core7_genome_frontier.py               # axes: MSE vs net return
                  cd ~/Financial_toolbox && python3 core7_genome_frontier.py --axis sharpe  # axes: MSE vs Sharpe
--axis sharpe builds the frontier and the pre-set choice on validation MSE vs validation SHARPE (risk-adjusted)
instead of net return; everything else is identical and its outputs end in _sharpe (the return version is kept).
                  cd ~/Financial_toolbox && python3 core7_genome_frontier.py --axis residual
--axis residual: MSE vs RESIDUAL ALPHA BEYOND MEAN REVERSION (outputs end in _residual; nothing else changes).
  Fixed in advance:
  1. Mean-reversion signal, per panel set and day t, from prices only up to t: the split-adjusted price is
     rebuilt from RET_raw_<STOCK> over the whole panel (P = cumulative product of 1 + RET_raw, so the window
     has history before the split starts); z_t = (P_t - mean(P_t-9..P_t)) / std(P_t-9..P_t) over 10 trading
     days (sample std, ddof = 1; undefined or zero std -> z = 0); reversal forecast = -z_t.
     It is aligned exactly like a genome's predictions: prediction j is made on panel row first + j and is
     traded / scored for row first + j + 1 (test drops the same warm-up rows). Its prediction files go to
     results/genome_frontier/<year>/REV_z10/ and it is traded with the same rule (overlap 5/5 hold 10, costs
     on, the 4 sets combined with equal weight), cached like the genome trades: the REVERSAL BENCHMARK.
  2. Per genome and split: reversal exposure = mean over the scored days of the daily cross-sectional
     Spearman correlation between the genome's forecast and -z, averaged over the 4 sets; and an OLS
     regression of the genome's combined 4-set daily book return on the reversal benchmark's, same days:
     r_genome = a + b r_reversal + e -> beta = b, residual alpha = a x 252 x 100 (% per year) and its t-stat
     (plain OLS standard error, no autocorrelation correction).
  3. Frontier on validation: lower validation MSE vs higher validation residual alpha; the same pre-set
     choice (lowest rank sum, ties -> lower MSE); frontier ends = lowest validation MSE and highest
     validation residual alpha. The test report adds each strategy's test residual alpha, its t-stat and
     beta against the reversal benchmark, and the reversal benchmark itself as a strategy row.
Outputs: results/core7_genome_frontier.csv (every genome), _headline.csv, _by_year.csv, .png;
--rule ls55: trade with daily long-short 5/5 (long top 5 / short bottom 5, rebalanced daily; RET_protocol.md's
primary rule) instead of overlap 5/5 hold 10, for any --axis; outputs end in _ls55 (e.g. _residual_ls55),
the overlap results and cache entries are left as they are.
intermediate files in results/genome_frontier/ (nothing else in results/ is touched). Trading results
are cached in results/genome_frontier/trades.csv, so a rerun only trades what is new.
"""
import contextlib
import io
import json
import math
import os
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
WORK = RESULTS / "genome_frontier"
EXAMM = Path(os.getenv("EXAMM_DIR", Path.home() / "EXAMM-Extended"))
PANELS = Path(os.getenv("CORE7_PANELS", EXAMM / "datasets" / "panels_core7"))
YEARS = [2022, 2023, 2024]
SETS = ["set1", "set2", "set3", "set4"]
REPEATS = [str(r) for r in range(15, 25)]
SPLITS = ["val", "test"]
# fixed trading rule: overlap 5/5 hold 10 (default) or --rule ls55 = daily long-short 5/5 (RET_protocol.md's
# primary rule, set before any test result was seen); ls55 outputs end in _ls55 and its trades are cached apart
RULES = {"overlap": ("overlap 5/5 hold 10", "overlapping_return", dict(long_company_number=5, hold_days=10)),
         "ls55": ("daily L/S 5/5", "daily_long_short_return", dict(long_company_number=5, short_company_number=5))}
RULE_ID = sys.argv[sys.argv.index("--rule") + 1].lower() if "--rule" in sys.argv[:-1] else "overlap"
if RULE_ID not in RULES:
    sys.exit("usage: python3 core7_genome_frontier.py [--axis return|sharpe|residual] [--rule overlap|ls55]")
RULE = RULES[RULE_ID]
RKEY = "" if RULE_ID == "overlap" else f"{RULE_ID}|"          # cache-key prefix (overlap keys unchanged)
# second frontier axis: validation net return (default) or validation Sharpe (--axis sharpe)
AXIS = "return"
if "--axis" in sys.argv:
    AXIS = sys.argv[sys.argv.index("--axis") + 1].lower() if sys.argv.index("--axis") + 1 < len(sys.argv) else ""
    if AXIS not in ("return", "sharpe", "residual"):
        sys.exit("usage: python3 core7_genome_frontier.py [--axis return|sharpe|residual] [--rule overlap|ls55]")
ACOL = {"return": "return_%", "sharpe": "sharpe", "residual": "alpha_%"}[AXIS]   # column suffix: val_<ACOL>, test_<ACOL>
ALABEL = {"return": f"net return of {RULE[0]} (%/yr)", "sharpe": f"Sharpe of {RULE[0]}",
          "residual": f"residual alpha of {RULE[0]} beyond mean reversion (%/yr)"}[AXIS]
TAG = {"return": "", "sharpe": "_sharpe", "residual": "_residual"}[AXIS] + ("" if RULE_ID == "overlap" else f"_{RULE_ID}")
REV_WINDOW = 10                    # --axis residual: z-score window of the mean-reversion signal (trading days)
REV_NAME = f"mean-reversion benchmark (-z, {REV_WINDOW} days)"
TRADING_DAYS = 252
EULER_GAMMA = 0.5772156649015329
PHI = NormalDist()


# ---------------------------------------------------------------- forecasts ----
_panel = {}


def panel(s):
    if s not in _panel:
        _panel[s] = pd.read_csv(PANELS / f"{s}_core7" / "panel_dates.csv").set_index("row")
    return _panel[s]


def ret_stats(s):
    return json.load(open(PANELS / f"{s}_core7" / "panel_meta.json"))["stats"]["RET"]


def split_first(y, s, split):
    info = pd.read_csv(EXAMM / "datasets" / f"panels_core7_examm_{y}" / s / "split_info.csv").set_index("split")
    return int(info.loc[split, "first_row"]), int(info.loc[split, "last_row"])


def final_genome(repeat_dir):
    """The repeat's final global best: the highest-numbered global_best_genome_*.bin."""
    nums = [int(g.stem.split("_")[-1]) for g in repeat_dir.glob("global_best_genome_*.bin")]
    return f"global_best_genome_{max(nums)}" if nums else None


def pred_folder(y, source, repeat, target, split):
    d = EXAMM / "test_output" / f"core7_examm_ret_{y}" / source / repeat
    return d / (f"evaluation_{split}" if target == source else f"evaluation_{split}_on_{target}")


def read_predictions(folder, split):
    files = sorted(folder.glob(f"*_{split}_predictions.csv"))
    if len(files) != 50:
        return None
    P = {}
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lstrip("#") for c in df.columns]
        P[f.name[:-len(f"_{split}_predictions.csv")]] = df["predicted_RET"].to_numpy()
    return pd.DataFrame(P)


def scored_mask(y, target, split, n):
    """Prediction j forecasts panel row first + j + 1; test drops its warm-up, validation scores every day."""
    first, _ = split_first(y, target, split)
    rows = panel(target).loc[first + np.arange(n) + 1]
    keep = (rows["date"] >= f"{y}-01-01").to_numpy() if split == "test" else np.ones(n, dtype=bool)
    return rows, keep


def forecast_scores(P, y, target, split):
    """MSE (raw daily-return units) and Rank IC of one genome's predictions on one set / split."""
    st = ret_stats(target)
    raw_pred = P * st["iqr"] + st["median"]
    rows, keep = scored_mask(y, target, split, len(P))
    A = pd.DataFrame({t: rows[f"RET_raw_{t}"].to_numpy() for t in P.columns})
    ic = raw_pred[keep].rank(axis=1).corrwith(A[keep].rank(axis=1), axis=1).dropna()
    return float(((raw_pred[keep] - A[keep]) ** 2).to_numpy().mean()), float(ic.mean())


# ---------------------------------------------------------------- mean reversion (--axis residual) ----
_zsig = {}


def reversal_signal(target):
    """-z of the 10-day price z-score for every panel row and stock (data up to that row only)."""
    if target not in _zsig:
        p = panel(target)
        stocks = [c[len("RET_raw_"):] for c in p.columns if c.startswith("RET_raw_")]
        price = (1 + p[[f"RET_raw_{t}" for t in stocks]].fillna(0)).cumprod()
        price.columns = stocks
        mean = price.rolling(REV_WINDOW, min_periods=REV_WINDOW).mean()
        std = price.rolling(REV_WINDOW, min_periods=REV_WINDOW).std(ddof=1)
        z = ((price - mean) / std.where(std > 0)).fillna(0.0)
        _zsig[target] = -z
    return _zsig[target]


def reversal_predictions(y, target, split, n, stocks):
    """The reversal 'forecasts' in genome-prediction layout: row j is made on panel row first + j."""
    first, _ = split_first(y, target, split)
    sig = reversal_signal(target).loc[first + np.arange(n)]
    return pd.DataFrame({t: sig[t].to_numpy() for t in stocks})


def reversal_exposure(P, y, target, split):
    """Mean daily cross-sectional Spearman correlation of a genome's forecast with -z (scored days)."""
    _, keep = scored_mask(y, target, split, len(P))
    R = reversal_predictions(y, target, split, len(P), list(P.columns))
    c = P[keep].reset_index(drop=True).rank(axis=1).corrwith(R[keep].reset_index(drop=True).rank(axis=1), axis=1)
    return float(c.dropna().mean())


def residual_regression(r, b):
    """OLS r = a + beta b + e on the same days: residual alpha (% per year), its t-stat (plain OLS SE), beta."""
    n = min(len(r), len(b))
    r, b = np.asarray(r[:n], dtype=float), np.asarray(b[:n], dtype=float)
    X = np.column_stack([np.ones(n), b])
    coef = np.linalg.lstsq(X, r, rcond=None)[0]
    e = r - X @ coef
    s2 = float(e @ e) / max(n - 2, 1)
    se_a = math.sqrt(s2 * np.linalg.inv(X.T @ X)[0, 0])
    return {"alpha_%": coef[0] * TRADING_DAYS * 100, "alpha_t": coef[0] / se_a if se_a > 1e-12 else np.nan,
            "beta": float(coef[1])}


def joined_regression(per_year, rev_per_year):
    """Residual regression over several years, each year's two series cut to the same length first."""
    rs, bs = [], []
    for r, b in zip(per_year, rev_per_year):
        n = min(len(r), len(b))
        rs.append(np.asarray(r[:n]))
        bs.append(np.asarray(b[:n]))
    return residual_regression(np.concatenate(rs), np.concatenate(bs))


def any_predictions(y, t, split):
    """Any genome's predictions on one set / split (for the benchmarks' file layout)."""
    for s in SETS:
        for r in REPEATS:
            P = read_predictions(pred_folder(y, s, r, t, split), split)
            if P is not None:
                return P, s, r
    sys.exit(f"no {split} predictions at all for {y} {t} -- run evaluate_rnn_core7_ret_cross.sh first")


# ---------------------------------------------------------------- trading ----
def price_dir(y, target, split):
    """Price files of one set / split (the same for every genome), built as main_core7.py does."""
    d = WORK / str(y) / f"prices_{target}_{split}"
    if d.exists() and len(list(d.glob("*.csv"))) == 50:
        return d
    first, last = split_first(y, target, split)
    rows = panel(target).loc[first:last]
    skip = int((rows["date"] < f"{y}-01-01").sum()) - 1 if split == "test" else 0
    rows = rows.iloc[skip:]
    d.mkdir(parents=True, exist_ok=True)
    stocks = [c[len("PRC_"):] for c in rows.columns if c.startswith("PRC_")]
    for s in stocks:
        growth = (1 + rows[f"RET_raw_{s}"]).cumprod()
        prc = rows[f"PRC_{s}"].iloc[0] * growth / growth.iloc[0]
        tc = rows[f"TC_{s}"] / rows[f"PRC_{s}"] * prc
        pd.DataFrame({"date": rows["date"].to_numpy(), "PRC": prc.to_numpy(), "BID": (prc - tc).to_numpy(),
                      "ASK": (prc + tc).to_numpy(), "TRAN_COST": tc.to_numpy()}).to_csv(d / f"{s}.csv", index=False)
    return d


def prediction_dir(y, source, repeat, target, split, P):
    """<STOCK>_predictions.csv of one genome on one set / split, warm-up dropped like main_core7.py."""
    d = WORK / str(y) / f"{source}_{repeat}" / f"pred_{target}_{split}"
    if d.exists() and len(list(d.glob("*_predictions.csv"))) == 50:
        return d
    first, last = split_first(y, target, split)
    rows = panel(target).loc[first:last]
    skip = int((rows["date"] < f"{y}-01-01").sum()) - 1 if split == "test" else 0
    d.mkdir(parents=True, exist_ok=True)
    for s in P.columns:
        pd.DataFrame({"predicted_RET": P[s].to_numpy()[skip:]}).to_csv(d / f"{s}_predictions.csv", index=False)
    return d


def daily_returns(pred_d, price_d, strategy, kwargs):
    """Daily returns of one strategy on one set. Overlapping: the account value is computed here from
    the cash flows of the trades and the open positions valued each morning (portfolio.py records no
    daily values for it); equal weight: portfolio.py's daily_values. Trading log kept off the screen."""
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

    state = {"cash": 0.0, "marks": []}
    restore = []

    def wrap(obj, method, make):
        setattr(obj, method, make(getattr(obj, method)))
        restore.append((obj, method))

    def value(c, time):
        return c.get_share() * abs(c.get_stock_price(time))

    for c in portfolio.portfolio_list:
        def clear(original):
            def f(time):
                cash = original(time)
                state["cash"] += cash
                return cash
            return f

        def rebalance(original):
            def f(target, time):
                cash = original(target, time)
                state["cash"] += cash
                return cash
            return f
        wrap(c, "clear_holdings", clear)
        wrap(c, "rebalance_to", rebalance)

    def morning(original):
        def f(time, *a, **k):
            state["marks"].append(portfolio.initial_capital + state["cash"]
                                  + sum(value(c, time) for c in portfolio.portfolio_list))
            return original(time, *a, **k)
        return f

    def capture(original):
        def f(*a, **k):
            state["values"] = (state["marks"] + [portfolio.initial_capital + state["cash"]]
                               if strategy == "overlapping_return" else list(getattr(portfolio, "daily_values", [])))
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
    v = np.asarray(state.get("values", []), dtype=float)
    if len(v) < 3:
        sys.exit(f"{strategy} recorded no daily values -- portfolio.py needs the daily_values lines "
                 "in daily_equal_weight_return (added for Sharpe/MDD)")
    return v[1:] / v[:-1] - 1


# ---------------------------------------------------------------- statistics ----
def metrics(r):
    r = np.asarray(r, dtype=float)
    value = np.cumprod(1 + r)
    vol = r.std() * math.sqrt(TRADING_DAYS)
    return {"ann_return_%": (value[-1] ** (TRADING_DAYS / len(r)) - 1) * 100 if value[-1] > 0 else np.nan,
            "ann_vol_%": vol * 100,
            "sharpe": r.mean() * TRADING_DAYS / vol if vol > 1e-12 else np.nan,
            "max_dd_%": (1 - value / np.maximum.accumulate(value)).max() * 100}


def bootstrap_ci(r, resamples=2000, block=10, seed=0):
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
    rets = np.where(growth > 0, (np.abs(growth) ** (TRADING_DAYS / n) - 1) * 100, np.nan)
    return {"sharpe_ci_lo": np.nanpercentile(sharpes, 2.5), "sharpe_ci_hi": np.nanpercentile(sharpes, 97.5),
            "ann_return_ci_lo": np.nanpercentile(rets, 2.5), "ann_return_ci_hi": np.nanpercentile(rets, 97.5)}


def psr(r):
    r = np.asarray(r, dtype=float)
    sd = r.std()
    if sd <= 1e-12:
        return np.nan
    sr = r.mean() / sd
    z = (r - r.mean()) / sd
    skew, kurt = float((z ** 3).mean()), float((z ** 4).mean())
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    return PHI.cdf(sr * math.sqrt(len(r) - 1) / math.sqrt(denom)) if denom > 0 else np.nan


def pareto(mse, ret):
    """True where no other genome has both a lower-or-equal MSE and a higher-or-equal return (one strictly)."""
    ok = ~(np.isnan(mse) | np.isnan(ret))
    out = np.zeros(len(mse), dtype=bool)
    for i in np.where(ok)[0]:
        out[i] = not any(ok[j] and mse[j] <= mse[i] and ret[j] >= ret[i] and (mse[j] < mse[i] or ret[j] > ret[i])
                         for j in range(len(mse)) if j != i)
    return out


def choose(g):
    """The pre-set choice (validation only): among frontier genomes, the lowest MSE rank + return (or Sharpe)
    rank (ranked over all candidates); ties -> lower validation MSE. Returns the chosen row's index."""
    rank_mse = g["val_mse"].rank(method="min")
    rank_ret = g[f"val_{ACOL}"].rank(method="min", ascending=False)
    score = (rank_mse + rank_ret).where(g["val_frontier"])
    best = score[score == score.min()].index
    return g.loc[best, "val_mse"].idxmin()


def combine(per_set):
    n = min(len(x) for x in per_set)
    return np.mean([x[:n] for x in per_set], axis=0)


# ---------------------------------------------------------------- main ----
if __name__ == "__main__":
    WORK.mkdir(parents=True, exist_ok=True)
    cache_path = WORK / "trades.npz"
    cache = dict(np.load(cache_path, allow_pickle=False)) if cache_path.exists() else {}
    rows, books, missing = [], {}, []
    for y in YEARS:
        # equal-weight benchmark (independent of the forecasts): once per set / split
        for split in SPLITS:
            key = f"{y}|EW|{split}"
            if key not in cache:
                per = []
                for t in SETS:
                    P = None
                    for s in SETS:                      # any genome's prediction files will do
                        for r in REPEATS:
                            P = read_predictions(pred_folder(y, s, r, t, split), split)
                            if P is not None:
                                pd_ = prediction_dir(y, s, r, t, split, P)
                                break
                        if P is not None:
                            break
                    if P is None:
                        sys.exit(f"no {split} predictions at all for {y} {t} -- run evaluate_rnn_core7_ret_cross.sh first")
                    per.append(daily_returns(pd_, price_dir(y, t, split), "daily_equal_weight_return", {}))
                cache[key] = combine(per)
            books[(y, "equal weight", split)] = cache[key]

        if AXIS == "residual":
            # reversal benchmark: -z traded with the same rule, per set / split, cached like the genomes
            for split in SPLITS:
                per = []
                for t in SETS:
                    key = f"{RKEY}{y}|REV{REV_WINDOW}|{t}|{split}"
                    if key not in cache:
                        P, _, _ = any_predictions(y, t, split)
                        R = reversal_predictions(y, t, split, len(P), list(P.columns))
                        cache[key] = daily_returns(prediction_dir(y, "REV", f"z{REV_WINDOW}", t, split, R),
                                                   price_dir(y, t, split), RULE[1], RULE[2])
                    per.append(cache[key])
                books[(y, "reversal", split)] = combine(per)

        for s in SETS:
            for r in REPEATS:
                d = EXAMM / "test_output" / f"core7_examm_ret_{y}" / s / r
                genome = final_genome(d)
                log = d / "fitness_log.csv"
                row = {"fold": y, "source_set": s, "repeat": r, "genome": genome,
                       "examm_val_mse": float(pd.read_csv(log).iloc[-1, 4]) if log.exists() else np.nan}
                complete = genome is not None
                for split in SPLITS:
                    per, mses, ics, exps = [], [], [], []
                    for t in SETS:
                        P = read_predictions(pred_folder(y, s, r, t, split), split)
                        if P is None or not np.isfinite(P.to_numpy()).all():
                            complete = False
                            break
                        mse, ic = forecast_scores(P, y, t, split)
                        mses.append(mse)
                        ics.append(ic)
                        if AXIS == "residual":
                            exps.append(reversal_exposure(P, y, t, split))
                        key = f"{RKEY}{y}|{s}|{r}|{t}|{split}"
                        if key not in cache:
                            cache[key] = daily_returns(prediction_dir(y, s, r, t, split, P), price_dir(y, t, split),
                                                       RULE[1], RULE[2])
                        per.append(cache[key])
                        books[(y, f"{s}/{r}", split, t)] = cache[key]
                    if not complete:
                        break
                    book = combine(per)
                    books[(y, f"{s}/{r}", split)] = book
                    m = metrics(book)
                    row.update({f"{split}_mse": np.mean(mses), f"{split}_rank_ic": np.mean(ics),
                                f"{split}_return_%": m["ann_return_%"], f"{split}_sharpe": m["sharpe"],
                                f"{split}_max_dd_%": m["max_dd_%"]})
                    if AXIS == "residual":
                        reg = residual_regression(book, books[(y, "reversal", split)])
                        row.update({f"{split}_reversal_exposure": float(np.mean(exps)), f"{split}_beta": reg["beta"],
                                    f"{split}_alpha_%": reg["alpha_%"], f"{split}_alpha_t": reg["alpha_t"]})
                if not complete:
                    missing.append(f"{y} {s}/{r}")
                rows.append(row)
            np.savez(cache_path, **cache)
            print(f"fold {y} {s}: 10 genomes done")

    if missing:
        print(f"!! {len(missing)} genome(s) left out (predictions missing or invalid on some set/split): "
              + ", ".join(missing[:12]) + (" ..." if len(missing) > 12 else ""))
    table = pd.DataFrame(rows)
    table = table.dropna(subset=["val_mse", f"val_{ACOL}"]).reset_index(drop=True)

    # ---- frontier and choice on VALIDATION ----
    table["val_frontier"] = False
    table["chosen"] = ""
    for y, g in table.groupby("fold"):
        table.loc[g.index, "val_frontier"] = pareto(g["val_mse"].to_numpy(), g[f"val_{ACOL}"].to_numpy())
        g = table.loc[g.index]
        table.loc[choose(g), "chosen"] = "chosen (frontier, best rank sum)"
        low = g["val_mse"].idxmin()
        high = g[f"val_{ACOL}"].idxmax()
        for idx, label in ((low, "frontier end: lowest val MSE"), (high, f"frontier end: highest val {AXIS}")):
            table.loc[idx, "chosen"] = (table.loc[idx, "chosen"] + "; " if table.loc[idx, "chosen"] else "") + label
    table["test_frontier"] = False
    for y, g in table.groupby("fold"):
        table.loc[g.index, "test_frontier"] = pareto(g["test_mse"].to_numpy(), g[f"test_{ACOL}"].to_numpy())

    # ---- report on TEST ----
    def picked(label):
        out = {}
        for y, g in table.groupby("fold"):
            row = g[g["chosen"].str.contains(label, regex=False)].iloc[0]
            out[y] = (f"{row['source_set']}/{row['repeat']}", books[(y, f"{row['source_set']}/{row['repeat']}", "test")])
        return out

    strategies = {"CHOSEN genome (val frontier, best rank sum)": picked("chosen"),
                  "frontier end: lowest val MSE (EXAMM criterion)": picked("lowest val MSE"),
                  f"frontier end: highest val {AXIS}": picked(f"highest val {AXIS}")}
    default = {}
    for y in YEARS:          # each set's own lowest-validation-MSE genome on its own set (EXAMM's per-set default)
        g = table[table["fold"] == y]
        per = []
        for t in SETS:
            own = g[g["source_set"] == t]
            if own.empty or own["examm_val_mse"].isna().all():
                per = []
                break
            r = own.loc[own["examm_val_mse"].idxmin(), "repeat"]
            if (y, f"{t}/{r}", "test", t) not in books:
                per = []
                break
            per.append(books[(y, f"{t}/{r}", "test", t)])
        if per:
            default[y] = ("per set", combine(per))
    if len(default) == len(YEARS):
        strategies["per-set EXAMM default (each set's lowest val MSE, own set)"] = default
    strategies["equal weight (benchmark)"] = {y: ("-", books[(y, "equal weight", "test")]) for y in YEARS}
    if AXIS == "residual":
        strategies[REV_NAME] = {y: ("-z", books[(y, "reversal", "test")]) for y in YEARS}

    head, by_year = [], []
    for name, per_year in strategies.items():
        joined = np.concatenate([per_year[y][1] for y in YEARS])
        extra = {}
        if AXIS == "residual":
            extra = joined_regression([per_year[y][1] for y in YEARS], [books[(y, "reversal", "test")] for y in YEARS])
        head.append({"strategy": name, **metrics(joined), **bootstrap_ci(joined), "psr": psr(joined), **extra,
                     "genomes": ", ".join(f"{y}: {per_year[y][0]}" for y in YEARS)})
        for y in YEARS:
            ey = (residual_regression(per_year[y][1], books[(y, "reversal", "test")]) if AXIS == "residual" else {})
            by_year.append({"strategy": name, "year": y, "genome": per_year[y][0], **metrics(per_year[y][1]), **ey})
    head = pd.DataFrame(head).set_index("strategy")
    by_year = pd.DataFrame(by_year)

    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / f"core7_genome_frontier{TAG}.csv", index=False)
    head.to_csv(RESULTS / f"core7_genome_frontier{TAG}_headline.csv")
    by_year.to_csv(RESULTS / f"core7_genome_frontier{TAG}_by_year.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n" + "=" * 100)
    print("GENOME PARETO FRONTIER -- Pooled RET, paper's folds (train <= Y-2, val Y-1, test Y), one genome shared by")
    print(f"all 4 sets; axes: validation MSE (raw returns) vs validation {ALABEL} (costs on);")
    print("genome chosen on VALIDATION only, reported on TEST 2022-2024 once.")
    print("=" * 100)
    cols = ["ann_return_%", "ann_return_ci_lo", "ann_return_ci_hi", "ann_vol_%", "sharpe", "sharpe_ci_lo",
            "sharpe_ci_hi", "psr", "max_dd_%"]
    if AXIS == "residual":
        cols += ["alpha_%", "alpha_t", "beta"]
    print(head[cols].astype(float).round(2).to_string())
    print("ci_lo / ci_hi = 95% stationary block bootstrap (mean block 10 days); psr = P(true Sharpe > 0).")
    if AXIS == "residual":
        print(f"alpha_% / alpha_t / beta: OLS of the daily test return on the {REV_NAME} "
              "(alpha annualized; plain OLS t-stat).")
    print("\nTest Sharpe per year:")
    print(by_year.pivot(index="strategy", columns="year", values="sharpe").reindex(head.index).round(2).to_string())
    show = ["fold", "source_set", "repeat", "val_mse", "val_rank_ic", "val_return_%", "val_sharpe", "test_mse", "test_rank_ic",
            "test_return_%", "test_sharpe", "chosen"]
    rnd = {"val_mse": 7, "test_mse": 7, "val_rank_ic": 4, "test_rank_ic": 4,
           "val_return_%": 2, "val_sharpe": 2, "test_return_%": 2, "test_sharpe": 2}
    if AXIS == "residual":
        show = show[:-1] + ["val_reversal_exposure", "val_alpha_%", "test_reversal_exposure", "test_alpha_%", "chosen"]
        rnd.update({"val_reversal_exposure": 3, "val_alpha_%": 2, "test_reversal_exposure": 3, "test_alpha_%": 2})
    print("\nValidation frontier genomes per fold (and how they did on test):")
    print(table[table["val_frontier"]][show].round(rnd).to_string(index=False))
    print(f"\nSaved {RESULTS / f'core7_genome_frontier{TAG}.csv'}, _headline.csv, _by_year.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(YEARS), 2, figsize=(13, 5 * len(YEARS)), squeeze=False)
        for row_axes, y in zip(axes, YEARS):
            g = table[table["fold"] == y]
            ch = g[g["chosen"].str.contains("chosen", regex=False)]
            for ax, sp in zip(row_axes, SPLITS):
                ax.scatter(g[f"{sp}_mse"], g[f"{sp}_{ACOL}"], c="lightgrey", s=35, label="40 candidate genomes")
                f = g[g["val_frontier"]].sort_values(f"{sp}_mse")
                ax.plot(f[f"{sp}_mse"], f[f"{sp}_{ACOL}"], "o--", c="tab:blue", label="validation frontier genomes")
                ax.scatter(ch[f"{sp}_mse"], ch[f"{sp}_{ACOL}"], marker="*", s=220, c="tab:red", label="chosen", zorder=5)
                for _, r in f.iterrows():
                    ax.annotate(f"{r['source_set']}/{r['repeat']}", (r[f"{sp}_mse"], r[f"{sp}_{ACOL}"]),
                                fontsize=7, xytext=(3, 3), textcoords="offset points")
                ax.axhline(0, c="grey", lw=0.6)
                ax.set_xlabel(f"{'validation' if sp == 'val' else 'test'} MSE (raw daily returns)")
                ax.set_ylabel(ALABEL)
                ax.set_title(f"Fold {y}: {'validation ' + str(y - 1) + ' (frontier chosen here)' if sp == 'val' else 'test ' + str(y) + ' (same genomes)'}")
            row_axes[0].legend(fontsize=8)
        fig.suptitle(f"Pooled EXAMM RET: genome Pareto frontier, MSE vs {ALABEL} (one genome for all 4 sets)")
        fig.tight_layout()
        fig.savefig(RESULTS / f"core7_genome_frontier{TAG}.png", dpi=150)
        print(f"Saved {RESULTS / f'core7_genome_frontier{TAG}.png'}")
    except ImportError:
        print("matplotlib not installed -- skipped the plot (pip3 install --user matplotlib)")
