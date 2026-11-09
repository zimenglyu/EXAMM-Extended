# Frozen evaluation protocol: Pooled EXAMM, RET target, core7 panels

**Frozen on:** 2026-10-31
**Status:** frozen. Nothing below may change after this date. Any later variant (a new rule,
setting, target, selection method or metric) is reported as a separate, exploratory addition and
never replaces a result produced under this protocol.

## 1. Forecasts
- **Model:** Pooled EXAMM, one model per panel set and walk-forward fold, trained on all 50 stocks.
- **Target:** next-day `RET` (robust z of the raw daily return). RET_CS is not part of this
  protocol; RET_CS runs are a side comparison only.
- **Inputs:** `RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21`, `--normalize avg_std_dev`,
  `--time_offset 1`, 10 islands x 10, 10,000 genomes, 10 repeats (folders 15-24) per set and fold
  (`scripts/base_run/core7_examm_yearly_ret.sh`).
- **Panels:** core7 set1-set4 (50 stocks each).
- **Folds:** test year Y in {2022, 2023, 2024}: train <= Y-2, EXAMM validation Y-1, test Y
  (with a 40-day warm-up before the scored test days).

## 2. Genome choice (stated as-is)
- One global best genome per set and fold: the repeat with the **lowest validation MSE**
  (final "Best Val. MSE" in `fitness_log.csv`). This is the rule that was used; it is kept
  unchanged. It is not tuned for trading, and the validation year is also EXAMM's own fitness
  data (see caveats).

## 3. Trading
- **Engine:** Financial_toolbox `portfolio.py` / `stock.py`, initial capital 1000,
  **transaction costs on** (`TRAN_COST` from `panel_dates.csv`), no bid/ask.
- **Book:** each rule is traded in every set and fold; the daily returns of the three test
  years are joined, and the 4 sets are combined with equal weight into one book.
- **Script:** `core7_frontier.py --turnover` (RET is the default target).

## 4. Frozen rule list (38 rules)
1. **Daily long-short** (`daily_long_short_return`), rebalanced every day, long L / short S,
   L, S in {3, 5, 10, 15}: 16 rules.
2. **Equal weight** (`daily_equal_weight_return`): 1 benchmark.
3. **Overlapping** (`overlapping_return`), long top k / short bottom k, each day's book held
   h days, k in {3, 5, 10, 15} x h in {2, 5, 10, 20}: 16 rules.
4. **Banded** (`banded_return`), (entry, exit) in (5, 15), (5, 25), (10, 25), (10, 35),
   (15, 35): 5 rules.

## 5. Results and how they are chosen
- **Primary result (set before any results were seen):** daily long-short **L/S 5/5**, on the
  test years 2022-2024.
- **Rule selection:** walk-forward on the test years only. For each test year after the first,
  rank the 38 rules by Sharpe over the **earlier** test years (2023: chosen on 2022; 2024:
  chosen on 2022-2023) and trade (a) the top-1 rule and (b) an equal-weight book of the top-3
  rules in that year. The scored period is 2023-2024.
- **Exploratory (reported, not used to claim performance):** the other 37 rules, the
  validation frontiers, the per-year and per-quarter tables, the costs-off runs, the
  best-fixed-rule-in-hindsight, and the RET_CS comparison.

## 6. Metrics reported
For the primary result, the walk-forward books and every rule (test split):
- annualized net return, annualized volatility, Sharpe (annualized, risk-free 0),
  maximum drawdown, annual turnover;
- 95% stationary block bootstrap confidence intervals of Sharpe and annualized return
  (mean block 10 days, 2000 resamples, seed 0);
- probabilistic Sharpe ratio (P[true Sharpe > 0]);
- deflated Sharpe ratio with N = 38 trials (a lower bound on the trials looked at);
- comparison with equal weight.
Validation numbers are never reported as performance.

## 7. Caveats (stated with every result)
1. **Validation is not out of sample.** EXAMM uses the validation year to keep each genome's
   best-epoch weights, as its fitness, to steer evolution and to pick the global best genome;
   the best-of-10 repeat choice uses it again, and `avg_std_dev` scaling includes it.
2. **The rule list was shaped by the test years.** The overlapping and banded rules were added
   after seeing that daily long-short lost money on test, so the 2022-2024 results of the
   non-primary rules are optimistic, including the walk-forward books built from them.
3. **Only two scored walk-forward years** (2023-2024, about 500 trading days): a Sharpe has an
   uncertainty of roughly +/-0.7 to +/-1.
4. **Different models each year:** a rule chosen on one year's models is traded with the next
   fold's models.
5. **Multiple testing:** 38 rules here, plus the RET_CS runs and earlier rule sets, were looked
   at; the deflated Sharpe ratio with N = 38 understates the true number of trials.

## 8. What would make it a fully fair test
Run this frozen protocol, unchanged and once, on data nobody has looked at (a 2025 test year or
new panel sets, or earlier untouched test years), and report whatever it gives.
