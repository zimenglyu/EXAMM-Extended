# Main workflow (core7, Pooled EXAMM RET, paper's trading rule)

Design: per test year Y (2022, 2023, 2024) train <= Y-2, validate Y-1, test Y; each panel set's genome = best of
10 runs by validation MSE; traded with the paper's Algorithm 1 (long 5 / short 5, rebalance only when the 5th-highest
forecast > 0 and the 5th-lowest < 0, forecasts converted to raw returns, costs on). No Pareto frontier.

1. Data and EXAMM runs (Anvil, done): scripts/data_prep/split_core7_yearly.py, scripts/base_run/core7_examm_yearly_ret.sh
2. Forecasts (Mac):   cd ~/EXAMM-Extended && sh scripts/rnn_examples/evaluate_rnn_core7_main.sh
3. Backtest (Mac):    cd ~/Financial_toolbox && python3 core7_main_results.py
                      python3 core7_main_results.py --trade-every-day     (comparison: no sign check)
4. Workbook (Mac):    cd ~/Financial_toolbox && python3 make_core7_main_workbook.py && open results/core7_main.xlsx

Main result: "retrain every year" (the paper's design). "Never retrain" is a robustness check.
Code: paper_long_short.py (the paper's Algorithm 1), core7_genome_frontier.py (shared helper functions).
Side experiments (not the main result): core7_genome_frontier*.py (Pareto frontier), core7_mo_frontier.py
(multi-objective EXAMM), core7_frontier.py (trading-rule comparison), main_core7_protocol.py (trade every day).
