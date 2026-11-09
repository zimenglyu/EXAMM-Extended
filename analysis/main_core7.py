import os
import sys
from pathlib import Path
import pandas as pd
from portfolio import Portfolio
from stock import Stock
from Logger import Logger
from ensemble import combine_prediction_files

if __name__ == '__main__':
    SEL_YEARS = ('2016', '2017', '2018', '2019', '2020')
    # 'best'     -> the single genome with the lowest validation MSE in the panel set
    #               (selection-year folds: the genome in selected_genomes.csv, chosen on selection-year Rank IC)
    # 'ensemble' -> rank-mean of the final global best genome of every repeat (15..24)
    genome_selection = 'best'
    run = 'core7_examm_ret_2023'
    panel_set = 'set1'
    split = 'test'   # the period that is traded; 'val' trades the validation year instead
    # optional command-line override:  python3 main_core7.py <set> [<year>] [RET|RET_CS] [VAL]
    # the target defaults to RET; RET runs exist for the 2022/2023/2024 folds and the
    # selection-year folds 2016-2020 (core7_examm_ret_sel_<year>, RET only)
    #   python3 main_core7.py set2 2018          ->  run='core7_examm_ret_sel_2018': the genome chosen on the
    #                                                selection year (selected_genomes.csv), traded over 2018
    #   python3 main_core7.py set2 2023 VAL      ->  same genome, traded over its VALIDATION year (2022)
    #   python3 main_core7.py set2               ->  RET backtests of 2022, 2023 and 2024, one after another
    #   python3 main_core7.py set2 2023          ->  panel_set='set2', run='core7_examm_ret_2023'  (RET target)
    #   python3 main_core7.py set2 2023 RET_CS   ->  panel_set='set2', run='core7_examm_2023'      (RET_CS target)
    #   python3 main_core7.py set2 RET_CS        ->  panel_set='set2', run='core7_examm' (2020-2024, RET_CS only)
    if len(sys.argv) > 1:
        panel_set = sys.argv[1]
        extra = [a.upper() for a in sys.argv[2:]]
        if 'VAL' in extra:
            split = 'val'
            extra.remove('VAL')
        years = [a for a in extra if a.isdigit()]
        targets = [a for a in extra if not a.isdigit()]
        if len(years) > 1 or len(targets) > 1 or any(t not in ('RET', 'RET_CS') for t in targets):
            raise ValueError("usage: python3 main_core7.py <set> [<year>] [RET|RET_CS] [VAL]")
        target = targets[0] if targets else 'RET'
        if years and years[0] in SEL_YEARS:
            if target != 'RET':
                raise ValueError(f"the selection-year fold {years[0]} has only RET runs")
            run = f"core7_examm_ret_sel_{years[0]}"
        elif years:
            run = f"core7_examm_ret_{years[0]}" if target == 'RET' else f"core7_examm_{years[0]}"
        elif target == 'RET_CS':
            run = 'core7_examm'
        else:
            # RET without a year: backtest every RET fold, each in its own process
            import subprocess
            status = 0
            for year in ('2022', '2023', '2024'):
                status |= subprocess.run([sys.executable, __file__, panel_set, year, 'RET',
                                          *(['VAL'] if split == 'val' else [])]).returncode
            sys.exit(status)

    repo_root = Path(__file__).resolve().parent
    examm_root = Path(os.getenv("EXAMM_DIR", Path.home() / "EXAMM-Extended"))
    results_root = Path(os.getenv("FIN_RESULTS_DIR", repo_root / "results"))

    runs = {
        'core7_examm':      {'evaluation': 'core7_examm',      'data': 'panels_core7_examm',
                             'prediction_column': 'predicted_RET_CS', 'start': '2020-01-01'},
        'core7_examm_2022': {'evaluation': 'core7_examm_2022', 'data': 'panels_core7_examm_2022',
                             'prediction_column': 'predicted_RET_CS', 'start': '2022-01-01'},
        'core7_examm_2023': {'evaluation': 'core7_examm_2023', 'data': 'panels_core7_examm_2023',
                             'prediction_column': 'predicted_RET_CS', 'start': '2023-01-01'},
        'core7_examm_2024': {'evaluation': 'core7_examm_2024', 'data': 'panels_core7_examm_2024',
                             'prediction_column': 'predicted_RET_CS', 'start': '2024-01-01'},
        # RET target: same data, predicts next-day RET (robust z of the raw return)
        'core7_examm_ret_2022': {'evaluation': 'core7_examm_ret_2022', 'data': 'panels_core7_examm_2022',
                                 'prediction_column': 'predicted_RET', 'start': '2022-01-01'},
        'core7_examm_ret_2023': {'evaluation': 'core7_examm_ret_2023', 'data': 'panels_core7_examm_2023',
                                 'prediction_column': 'predicted_RET', 'start': '2023-01-01'},
        'core7_examm_ret_2024': {'evaluation': 'core7_examm_ret_2024', 'data': 'panels_core7_examm_2024',
                                 'prediction_column': 'predicted_RET', 'start': '2024-01-01'},
    }
    # global best genome per panel set (lowest validation MSE of the 10 repeats), used when
    # the fitness logs are not downloaded
    best_genomes = {
        'core7_examm':          {'set1': '15/global_best_genome_9927',  'set2': '17/global_best_genome_9672',
                                 'set3': '24/global_best_genome_10226', 'set4': '24/global_best_genome_9871'},
        'core7_examm_2022':     {'set1': '20/global_best_genome_17249', 'set2': '21/global_best_genome_15175',
                                 'set3': '20/global_best_genome_9056',  'set4': '19/global_best_genome_9936'},
        'core7_examm_2023':     {'set1': '22/global_best_genome_9872',  'set2': '20/global_best_genome_10309',
                                 'set3': '21/global_best_genome_9694',  'set4': '19/global_best_genome_11164'},
        'core7_examm_2024':     {'set1': '19/global_best_genome_10107', 'set2': '19/global_best_genome_9498',
                                 'set3': '23/global_best_genome_12872', 'set4': '23/global_best_genome_9292'},
        'core7_examm_ret_2022': {'set1': '21/global_best_genome_16410', 'set2': '16/global_best_genome_9708',
                                 'set3': '22/global_best_genome_10627', 'set4': '24/global_best_genome_10937'},
        'core7_examm_ret_2023': {'set1': '17/global_best_genome_9847',  'set2': '21/global_best_genome_9765',
                                 'set3': '17/global_best_genome_10525', 'set4': '21/global_best_genome_9849'},
        'core7_examm_ret_2024': {'set1': '15/global_best_genome_10639', 'set2': '15/global_best_genome_9705',
                                 'set3': '22/global_best_genome_9589',  'set4': '24/global_best_genome_10469'},
    }
    # selection-year folds (EXAMM-Extended data_prep/split_core7_yearly_sel.py): RET was re-scaled
    # per fold, which does not change the ranking the strategies trade on
    for y in SEL_YEARS:
        runs[f'core7_examm_ret_sel_{y}'] = {'evaluation': f'core7_examm_ret_sel_{y}',
                                            'data': f'panels_core7_examm_sel_{y}',
                                            'prediction_column': 'predicted_RET', 'start': f'{y}-01-01'}
    if run not in runs:
        raise ValueError(f"unknown run '{run}'")
    EVALUATION_DIR = examm_root / "test_output" / runs[run]['evaluation'] / panel_set
    DATA_DIR = examm_root / "datasets" / runs[run]['data'] / panel_set
    PANEL_DIR = examm_root / "datasets" / "panels_core7" / f"{panel_set}_core7"
    prediction_column = runs[run]['prediction_column']
    PREDICTION_DIR = results_root / f"{run}_{panel_set}_{genome_selection}_{split}_prediction"
    TEST_DIR = results_root / f"{run}_{panel_set}_{split}_prices"

    # ------------------- pick genome(s) from the validation summary -------------------
    # one row per repeat: its final global best genome and the validation MSE EXAMM logged
    # for it (the last row of fitness_log.csv; the same number evaluate_rnn gives on _val.csv).
    # The test split is never used for choosing.
    summary = []
    for repeat_dir in sorted(p for p in EVALUATION_DIR.iterdir() if p.is_dir()):
        log = repeat_dir / "fitness_log.csv"
        genomes = sorted(repeat_dir.glob("global_best_genome_*.bin"), key=lambda g: int(g.stem.split('_')[-1]))
        if log.exists() and genomes:
            summary.append({'run': repeat_dir.name, 'genome': genomes[-1].stem,
                            'val_mse': float(pd.read_csv(log).iloc[-1, 4])})
    summary = pd.DataFrame(summary)
    if summary.empty and not (genome_selection == 'best' and run.startswith('core7_examm_ret_sel_')):
        if genome_selection != 'best':
            raise FileNotFoundError(f"no repeats with fitness_log.csv and a global_best_genome in {EVALUATION_DIR} "
                                    f"-- download the fitness logs and genomes for {run}/{panel_set}")
        # no fitness logs here: use the recorded global best for this set
        repeat, genome = best_genomes[run][panel_set].split('/')
        summary = pd.DataFrame([{'run': repeat, 'genome': genome, 'val_mse': float('nan')}])
    chosen_on = 'validation MSE'
    if genome_selection == 'best' and run.startswith('core7_examm_ret_sel_'):
        # chosen on the selection year by EXAMM-Extended's core7_sel_ic.py, not by validation MSE
        selected = examm_root / "test_output" / run / "selected_genomes.csv"
        if not selected.exists():
            raise FileNotFoundError(f"{selected} not found -- run python3 core7_sel_ic.py in {examm_root} first")
        pick = pd.read_csv(selected, dtype={'repeat': str}).set_index('set').loc[panel_set]
        members = pd.DataFrame([{'run': pick['repeat'], 'genome': pick['genome'],
                                 'val_mse': float('nan')}])
        chosen_on = 'selection-year Rank IC'
    elif genome_selection == 'best':
        members = summary.nsmallest(1, 'val_mse')
    elif genome_selection == 'ensemble':
        members = summary
    else:
        raise ValueError(f"unknown genome_selection '{genome_selection}'")
    print(f"{run} {panel_set}: using {len(members)} genome(s) ({genome_selection}, chosen on {chosen_on}):")
    print(members.to_string(index=False))
    member_dirs = [EVALUATION_DIR / row.run / f"evaluation_{split}" for row in members.itertuples()]

    # RET_CS is a cross-sectional rank score (not a return); RET is the robust z of the raw
    # return. The strategies trade on the cross-sectional ranking, which neither unit changes,
    # so the prediction goes into 'predicted_RET' unchanged

    # -------- build <STOCK>_predictions.csv and <STOCK>.csv from the test period --------
    # core7 _test.csv files hold only features, so dates, prices, costs and raw returns come
    # from the panel's panel_dates.csv; prediction row j forecasts test row j + 1
    split_info = pd.read_csv(DATA_DIR / "split_info.csv").set_index('split')
    first, last = int(split_info.loc[split, 'first_row']), int(split_info.loc[split, 'last_row'])
    test_rows = pd.read_csv(PANEL_DIR / "panel_dates.csv").set_index('row').loc[first:last]
    # drop the 40-day warm-up before the scored period, but keep its last day as the entry
    # price for the first trade, so the price file keeps one more row than the predictions
    # (validation has no warm-up: all its rows are traded, the first day is the entry price)
    skip = int((test_rows['date'] < runs[run]['start']).sum()) - 1 if split == 'test' else 0

    stock_names = sorted(f.name[:-len(f'_{split}_predictions.csv')]
                         for f in member_dirs[0].glob(f'*_{split}_predictions.csv'))
    if not stock_names and run.startswith('core7_examm_ret_sel_'):
        raise FileNotFoundError(f"no {split} predictions in {member_dirs[0]} -- evaluate them first:\n  "
                                f"YEARS={run.split('_')[-1]} SPLITS={split} sh "
                                f"~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_sel.sh")
    if not stock_names:
        genomes = ' '.join(f"{panel_set}/{row.run}/{row.genome}" for row in members.itertuples())
        year = '' if run == 'core7_examm' else run.split('_')[-1]
        target = 'TARGET=RET ' if '_ret_' in run else ('TARGET=RET_CS ' if year else '')
        raise FileNotFoundError(f"no {split} predictions in {member_dirs[0]} -- evaluate the {run} genomes on SPLIT={split} "
                                f"first:\n  SPLIT={split} {('YEAR=' + year + ' ') if year else ''}{target}sh "
                                f"~/EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7.sh {genomes}")

    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    for stock_name in stock_names:
        member_files = [d / f"{stock_name}_{split}_predictions.csv" for d in member_dirs]
        missing = [f for f in member_files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{missing[0]} -- evaluate the {run} genomes on SPLIT={split} first")
        # Stock.read_return_prediction reads a 'predicted_RET' column
        if len(member_files) == 1:
            predicted = pd.read_csv(member_files[0], usecols=[prediction_column])[prediction_column]
        else:
            predicted = combine_prediction_files([str(f) for f in member_files], column=prediction_column)[prediction_column]
        pd.DataFrame({'predicted_RET': predicted.iloc[skip:].to_numpy()}) \
            .to_csv(PREDICTION_DIR / f"{stock_name}_predictions.csv", index=False)

        # price file: one more row than the predictions (offset 1), plus TRAN_COST.
        # Raw PRC jumps on splits, so the price series is rebuilt from raw daily returns
        # starting at the actual price; TC_<STOCK> is kept as the same fraction of price
        rows = test_rows.iloc[skip:]
        growth = (1 + rows[f'RET_raw_{stock_name}']).cumprod()
        prc = rows[f'PRC_{stock_name}'].iloc[0] * growth / growth.iloc[0]
        tc = rows[f'TC_{stock_name}'] / rows[f'PRC_{stock_name}'] * prc
        pd.DataFrame({'date': rows['date'].to_numpy(), 'PRC': prc.to_numpy(),
                      'BID': (prc - tc).to_numpy(), 'ASK': (prc + tc).to_numpy(),
                      'TRAN_COST': tc.to_numpy()}).to_csv(TEST_DIR / f"{stock_name}.csv", index=False)

    # --------------------------------- trading ---------------------------------
    logger = Logger()
    logger.set_level('INFO')

    spend_per_stock = 100
    initial_capital = 1000
    portfolio = Portfolio(stock_names)
    portfolio.set_initial_spend_per_stock(spend_per_stock)
    portfolio.set_initial_capital(initial_capital)
    portfolio.set_logger(logger)

    for stock_name in stock_names:
        company = Stock(stock_name)
        prediction_file = PREDICTION_DIR / f"{stock_name}_predictions.csv"
        test_file = TEST_DIR / f"{stock_name}.csv"
        company.read_return_prediction(str(prediction_file))
        company.read_stock_price(str(test_file))
        portfolio.add_company_to_protfolio(company)
        company.set_logger(logger)

    # 50 stocks in each core7 panel set, so long/short the top/bottom 5 (10%)
    long_company_number = 5
    short_company_number = 5

    use_TC = True
    trade_with_bid_ask = False
    portfolio.set_use_TC(use_TC)
    portfolio.set_trade_with_bid_ask(trade_with_bid_ask)
    for return_strategy in ['daily_long_short_return','buy_and_hold_return','daily_equal_weight_return']:
        portfolio.trade(return_strategy, long_company_number, short_company_number)
