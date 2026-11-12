# Main workflow (core7, Pooled EXAMM RET, paper's trading rule and reporting)

Design: per test year Y (2022, 2023, 2024) train <= Y-2, validate Y-1, test Y; 10 EXAMM runs (seeds) per panel set
V1-V4; traded with the paper's Algorithm 1 (long 10 / short 10, rebalance only when the 10th-highest forecast > 0 and
the 10th-lowest < 0, forecasts converted to raw returns, costs on). --k 5 = the 5/5 book.
Main comparison with the paper = the MEAN OVER THE 10 SEEDS (as the paper reports); the single best seed by validation
MSE is shown next to it as a labelled second row.

1. Data and EXAMM runs (Anvil, done): scripts/data_prep/split_core7_yearly.py, scripts/base_run/core7_examm_yearly_ret.sh
2. Forecasts (Mac, in ~/EXAMM-Extended; folders already done are skipped):
        sh scripts/rnn_examples/evaluate_rnn_core7_main.sh                  best seed, own + later years
        sh scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh             every seed, own validation + test year
        ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh    every seed, later years (never retrain)
3. Backtest (Mac, in ~/Financial_toolbox):
        python3 core7_main_results.py --pareto            everything: best seed, MEAN OVER THE 10 SEEDS, Pareto choice
   options: --k 5 (5/5 book, outputs end in _k5), --no-mean-repeats, --trade-every-day (no sign check),
            --all (leakage cells, for comparison with the paper only)
4. Extra figures (Mac):
        python3 plot_core7_alpha.py           Figure 1 / Table 6: net alpha vs MKT + STR1 + STR5 (Driscoll-Kraay)
        python3 plot_core7_pareto_corr.py     does a validation score predict the test result?
   (both take --k 5)
5. Workbook (Mac):    python3 make_core7_main_workbook.py && open results/core7_main.xlsx      (--k 5 -> core7_main_k5.xlsx)
   Summary / Per set / Table 5 / Table 8: paper (blue), yours MEAN OF 10 SEEDS (yellow, bold), best seed (grey, italic).
   Repeats T7/T8: the mean over seeds in full. Grid / Joined / Rule compared / Model size: the best seed.

Main result: "retrain every year" (the paper's design). "Never retrain" and the Pareto choice are robustness checks.
The paper's IC (+0.0106, Tables 4 and 8) is the PEARSON IC -- compare it with the IC (Pearson) columns.
Paper: arXiv 2610.07825. Still different from it: the inputs (paper: Return, Volume Change, Bid-Ask Spread, Illiquidity,
Turnover, S&P 500 Return; raw values), 10 backprop epochs (here 20), and the target scaling (here robust z-score);
matching them needs raw CRSP data for the 200 stocks, which is not available.
Code: paper_long_short.py (the paper's Algorithm 1), core7_genome_frontier.py (shared helper functions).
Side experiments (not the main result): core7_genome_frontier*.py (Pareto genome frontier), core7_mo_frontier.py
(multi-objective EXAMM), core7_frontier.py (trading-rule comparison), main_core7_protocol.py (trade every day).
