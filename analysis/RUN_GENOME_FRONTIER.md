# Genome Pareto frontier on the paper's Train/Val/Test folds (professor's design)

## Design (fixed before running)
- **Forecasts:** the existing Pooled EXAMM RET runs, `test_output/core7_examm_ret_<Y>`, folds
  **2022, 2023, 2024** (train <= Y-2, validation Y-1, test Y with a 40-day warm-up). No new training.
- **Candidates per fold:** all **40 genomes** (4 source panel sets x 10 repeats; each repeat's final
  global best = highest-numbered `global_best_genome_*.bin`), each used as **one genome shared by
  all 4 panel sets** (200 stocks).
- **Axes (per genome, on validation and on test, over the 4 sets):**
  - **MSE** in raw daily-return units (prediction x IQR + median from each set's `panel_meta.json`,
    against the unclipped raw return; as `core7_ic.py`), averaged over the 4 sets;
  - **net annualized return** of the **fixed rule overlap 5/5 hold 10** (`overlapping_return`), costs on,
    capital 1000, the 4 sets combined with equal weight. (Sharpe, max drawdown and Rank IC are reported too.)
- **Pareto frontier on VALIDATION:** lower MSE and higher return.
- **Choice (validation only):** among the frontier genomes, the lowest sum of the rank by validation MSE
  (ascending) and by validation return (descending), both over all 40 candidates; ties -> lower
  validation MSE. The two frontier ends are recorded too: lowest validation MSE (EXAMM's own criterion)
  and highest validation return.
- **Reported on TEST 2022-2024 once:** the chosen genome (joined and per year, 95% block bootstrap
  intervals, PSR), against the two frontier ends, the per-set EXAMM default (each set's own lowest
  validation MSE genome on its own set) and equal weight.
- Validation is also EXAMM's fitness data, so validation numbers are optimistic; only test numbers are
  reported as performance.

## Commands (Terminal on the Mac, not logged in to Anvil)

1. Copy the new scripts:
```bash
scp x-dlopez6@anvil.rcac.purdue.edu:/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh ~/EXAMM-Extended/scripts/rnn_examples/
```
```bash
scp x-dlopez6@anvil.rcac.purdue.edu:/anvil/scratch/x-dlopez6/experiments/Financial_toolbox/core7_genome_frontier.py x-dlopez6@anvil.rcac.purdue.edu:/anvil/scratch/x-dlopez6/experiments/Financial_toolbox/RUN_GENOME_FRONTIER.md ~/Financial_toolbox/
```

2. Only if missing on the Mac (the repeats run already downloaded every RET genome and fitness log;
   the fold files were used before): the validation / test files of the three folds
```bash
rsync -av --include='panels_core7_examm_202[234]/' --include='panels_core7_examm_202[234]/set*/' --include='*_val.csv' --include='*_test.csv' --include='split_info.csv' --exclude='*' x-dlopez6@anvil.rcac.purdue.edu:/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended/datasets/ ~/EXAMM-Extended/datasets/
```
   and, if any genome is reported missing in step 3:
```bash
rsync -av --include='core7_examm_ret_202[234]/' --include='core7_examm_ret_202[234]/set*/' --include='core7_examm_ret_202[234]/set*/*/' --include='global_best_genome_*.bin' --include='fitness_log.csv' --exclude='*' x-dlopez6@anvil.rcac.purdue.edu:/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended/test_output/ ~/EXAMM-Extended/test_output/
```

3. Evaluate every genome on every set, validation and test (960 folders; about 130 already exist:
   the 120 own-set test predictions of the repeats run and the 12 chosen genomes' validation
   predictions, so about 830 new `evaluate_rnn` runs, roughly 2-5 hours). Rerun to resume:
```bash
caffeinate -i sh ~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh
```
   It ends with `evaluated N, already done M, failed 0, missing 0`.

4. The frontier, the choice and the test report (about 960 overlapping backtests + the benchmark,
   roughly 10-20 minutes; results are cached, a rerun only trades what is new):
```bash
cd ~/Financial_toolbox && caffeinate -i python3 core7_genome_frontier.py
```

5. Open the results:
```bash
open ~/Financial_toolbox/results/core7_genome_frontier.png ~/Financial_toolbox/results/core7_genome_frontier_headline.csv ~/Financial_toolbox/results/core7_genome_frontier.csv
```

## Outputs
- `results/core7_genome_frontier.csv` — every genome: validation and test MSE, Rank IC, return,
  Sharpe, max drawdown; `val_frontier`, `test_frontier`, `chosen` (chosen / frontier ends).
- `results/core7_genome_frontier_headline.csv` — chosen genome, frontier ends, per-set EXAMM default,
  equal weight on test 2022-2024 (intervals, PSR).
- `results/core7_genome_frontier_by_year.csv` — the same per test year.
- `results/core7_genome_frontier.png` — per fold: validation MSE vs return (left, frontier and choice)
  and the same genomes on test (right).
- `results/genome_frontier/` — intermediate prediction/price files and the trade cache (`trades.npz`).
  Delete `trades.npz` if predictions are ever re-evaluated.

## --axis residual: MSE vs residual alpha beyond mean reversion

Same 40 genomes, same trades (cache reused); the y axis is the genome's alpha after removing what a plain
10-day z-score mean-reversion strategy explains. Outputs end in `_residual`; the return and Sharpe outputs are untouched.

```bash
cd ~/Financial_toolbox && python3 core7_genome_frontier.py --axis residual
cd ~/Financial_toolbox && python3 core7_genome_frontier_plots.py --axis residual
cd ~/Financial_toolbox && python3 make_genome_frontier_workbook.py --axis residual
```

- Signal: price rebuilt from RET_raw; z_t = (P_t − 10-day mean) / 10-day sample std; forecast = −z_t (data up to t only),
  aligned like the genome predictions and traded with the same rule (overlap 5/5 hold 10, costs on, 4 sets combined)
  = the reversal benchmark (prediction files in results/genome_frontier/<year>/REV_z10/).
- Per genome and split: reversal exposure (mean daily Spearman of forecast with −z, 4 sets averaged) and OLS
  r_genome = a + b·r_reversal → beta = b, residual alpha = a·252·100 %/yr, plain OLS t-stat.
- Frontier: lower val MSE vs higher val residual alpha; same pre-set choice. Test report adds alpha / t / beta and a
  reversal-benchmark row. Plots: val MSE gain vs val alpha, test MSE gain vs test alpha, val MSE gain vs exposure.
- Outputs: results/core7_genome_frontier_residual{.csv,_headline.csv,_by_year.csv,.png,.xlsx}.
