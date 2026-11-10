#!/usr/bin/env python3
"""The paper's Algorithm 1 (Daily Long-Short Strategy), as a separate function; portfolio.py is not changed.

It is portfolio.py's daily_long_short_return with the sign check that is commented out there switched on:

    each day t: sort the stocks by predicted return
        if the K_L-th highest prediction > 0 and the K_S-th lowest prediction < 0:
            clear all holdings (sell longs, cover shorts)
            q_L = C / K_L, q_S = C / K_S        (C = current capital)
            buy q_L of each of the K_L highest, short q_S of each of the K_S lowest
        else: no trade, yesterday's positions are kept
    clear all holdings at the end

The predictions must be in RAW daily-return units for the "> 0" / "< 0" check (the panels' RET is a robust
z-score, (raw - median) / IQR, so "> 0" there would mean "above the median return"); the caller converts them.
Buying, shorting, clearing and transaction costs are the Stock methods portfolio.py uses (costs on).

Unlike portfolio.py, which records the account value only on days it trades, the account is marked to
market every day (cash + signed value of the open positions), so daily returns, Sharpe and drawdown use
every trading day, including the days without a trade.
"""
import numpy as np


def paper_long_short_daily_values(portfolio, long_n=5, short_n=5, sign_check=True):
    """Account value at the close of every day (plus the final value after clearing), and the number of
    days a trade was made. `portfolio` is a portfolio.Portfolio with its Stocks added and use_TC set."""
    companies = portfolio.portfolio_list
    for c in companies:
        c.reset()
    cash = float(portfolio.initial_capital)
    period = companies[0].testing_period
    preds = np.column_stack([np.asarray(c.return_prediction, dtype=float)[:period] for c in companies])
    held = []
    values, trades = [], 0

    def mark(t):
        return cash + sum(c.get_notional(t) for c in companies)

    for t in range(period - 1):
        values.append(mark(t))
        order = np.argsort(preds[t], kind="stable")          # ascending: lowest first
        longs = [companies[i] for i in order[::-1][:long_n]]
        shorts = [companies[i] for i in order[:short_n]]
        if sign_check and not (preds[t, order[::-1][long_n - 1]] > 0 and preds[t, order[short_n - 1]] < 0):
            continue                                          # no trade today: keep yesterday's positions
        for c in held:
            cash += c.clear_holdings(t)
        capital = cash
        q_long, q_short = capital / long_n, capital / short_n
        for c in longs:
            c.buy_stock(q_long, t)
            cash -= q_long
        for c in shorts:
            cash += c.short_stock(q_short, t)
        held = longs + shorts
        trades += 1
    t = period - 1
    for c in companies:
        cash += c.clear_holdings(t)
    values.append(cash)
    return np.asarray(values, dtype=float), trades
