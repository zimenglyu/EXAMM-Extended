# Portfolio Management System

This repository contains a simple portfolio management system written in Python. It allows users to calculate returns on investment for different stocks based on predicted returns and actual prices.

## Files

- `portfolio.py`: Contains the `Portfolio` class responsible for managing financial portfolios.
- `stock.py`: Contains the `Stock` class responsible for managing stocks in portfolios.
- `main.py`: Main script to execute portfolio calculations.
- `README.md`: This documentation file.

## Usage

1. Ensure you have Python installed on your system.
2. Clone this repository to your local machine.
3. Install the required dependencies by running `pip install pandas`.
4. Set environment variables for data and results paths (or use the defaults described below).
5. Define return_strategy in `main.py`.
6. Run `main.py` to execute the portfolio calculations.

## Paths and environment variables

Scripts in this repository avoid hard-coded absolute paths. Set these variables to point to your local data:

- See `.env.example` for a starter template.
- `FIN_DATASETS_DIR`: root directory for datasets (default: `./datasets` relative to repo root)
- `FIN_RESULTS_DIR`: root directory for result outputs (default: `./results` relative to repo root)
- `CRSP_PROCESSOR_DIR`: root directory for the CRSP processor repo (default: `./CRSP_Processor` relative to repo root)
- `EXACT_RESULTS_DIR`: root directory for "exact" experiment outputs (default: `./exact` relative to repo root)
- `FIN_PREDICTION_DIR`, `FIN_TEST_DIR`, `FIN_EXPECTED_DIR`: optional overrides used by specific scripts

## Portfolio Class

The `Portfolio` class contains the following trading strategies:

- `simple_return`: Trade each stock individually based on the predicted return. Each stock has the same initial capital for investment. If predicted return > 0, buy or hold; if predicted return <= 0, sell.
- `portfolio_simple_return`: At time t, sell all stocks with negative predicted return, and use the gained capital to invest all the stocks with positive capital.
- `long_short_return`: Short the stocks with predicted negative return, STILL UNDER DEVELOPMENT.
- `overlapping_return`: Jegadeesh-Titman overlapping portfolios. Each day a
  new long-short book is formed and held `hold_days`, so `hold_days` books
  are open at once, each funded with `1/hold_days` of the capital. The books
  are netted before trading, so a name held by several is held once and only
  the change in its aggregate notional pays the spread. This is the rule the
  ONE-NAS paper reports throughout.
- `banded_return`: Novy-Marx/Velikov buy-hold spread. A name is entered when
  its predicted rank reaches `entry_rank` and held until that rank falls past
  `exit_rank`. The dead band between the two removes the churn that small
  day-to-day rank changes would otherwise cause.
- `conditional_long_short_return`: Algorithm 1 of the ONE-NAS paper. As
  `daily_long_short_return`, but it rebalances only on days when the model
  separates the cross-section by sign. `daily_long_short_return` contains
  these conditions commented out and so trades unconditionally; this is a
  separate method so results from that one stay reproducible.

`overlapping_return` and `banded_return` need `Stock.rebalance_to()`, which
moves a position to a target notional and charges the spread only on the part
that traded. The existing buy/short and sell/return primitives are
all-or-nothing and cannot express a position that merely changes size.

`ensemble.py` carries the paper's prediction-combination step: `rank_mean`
averages the centred ranks of several models (the island champions), and
`mean_pairwise_spearman` reports how correlated those members are, which is
what determines whether averaging them buys anything.
