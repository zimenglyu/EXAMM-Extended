#!/usr/bin/env python3
"""Verify every collected genome reproduces its recorded validation MSE, then
build the corrected (verified-clean) selection.

This formalizes the salvage protocol used on the 50-stock baseline: EXAMM's
saved global-best genomes can be corrupted (fitness/weights mismatch -- see
docs/genome_corruption_bugfix.md; 72/500 baseline genomes, 14.4%), and the
recorded best_validation_mse of a corrupted file is a lie. The only honest
selection input is the RE-MEASURED validation MSE of the saved weights.

For each (ticker,run) in <BEST_DIR>/index.csv:
    run evaluate_rnn on <DATA>/<ticker>_val.csv, record reproduced val MSE.
Writes:
    <BEST_DIR>/val_reproduce.csv    ticker,run,recorded_val_mse,reproduced_val_mse
    <BEST_DIR>/index_corrected.csv  ticker,run,reproduced_val_mse  (best clean
                                    genome per ticker by reproduced val MSE)
Prints the corruption census. Exits nonzero if any ticker has no clean genome.

Usage:
    python3 scripts/stock_run/sweep_verify.py <BEST_DIR> <DATA_DIR>
e.g.
    python3 scripts/stock_run/sweep_verify.py test_output/pilot_2023_best datasets/walkforward/pilot_2023

Run in an interactive/compute job for large collections (one evaluate_rnn per
genome; each is a fast single-threaded inference).
"""
import csv
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "build/rnn_examples/evaluate_rnn"
CLEAN_LO, CLEAN_HI = 0.98, 1.02   # reproduced/recorded ratio band = clean


def fail(msg):
    sys.exit(f"ERROR: {msg}")


def reproduce_mse(bin_file, val_csv):
    with tempfile.TemporaryDirectory() as tmp:
        # stdout=PIPE + universal_newlines instead of capture_output/text:
        # cluster login nodes run Python 3.6
        out = subprocess.run(
            [str(BIN), "--genome_file", str(bin_file),
             "--testing_filenames", str(val_csv),
             "--time_offset", "1", "--output_directory", tmp,
             "--std_message_level", "INFO", "--file_message_level", "NONE"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True)
    for line in out.stdout.splitlines():
        if "MSE:" in line:
            return float(line.rsplit(None, 1)[-1])
    return None


def main():
    if len(sys.argv) != 3:
        fail(f"usage: {sys.argv[0]} <BEST_DIR> <DATA_DIR>")
    best_dir = Path(sys.argv[1])
    data_dir = Path(sys.argv[2])
    if not best_dir.is_absolute():
        best_dir = REPO / best_dir
    if not data_dir.is_absolute():
        data_dir = REPO / data_dir
    index = best_dir / "index.csv"
    if not BIN.is_file():
        fail(f"{BIN} not found -- build evaluate_rnn first")
    if not index.is_file():
        fail(f"{index} not found -- run collect_best_genomes.sh first")

    rows = []
    with open(index, newline="") as f:
        for r in csv.DictReader(f):
            ticker = r["ticker"].strip()
            run = r["run"].strip()
            recorded = float(r["best_validation_mse"])
            bin_file = best_dir / f"{ticker}_run{run}.bin"
            val_csv = data_dir / f"{ticker}_val.csv"
            if not bin_file.is_file():
                fail(f"missing genome {bin_file}")
            if not val_csv.is_file():
                fail(f"missing validation file {val_csv}")
            reproduced = reproduce_mse(bin_file, val_csv)
            if reproduced is None:
                fail(f"evaluate_rnn produced no MSE for {bin_file}")
            rows.append((ticker, run, recorded, reproduced))
            sys.stdout.write(".")
            sys.stdout.flush()
    print(" done")

    with open(best_dir / "val_reproduce.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "run", "recorded_val_mse", "reproduced_val_mse"])
        w.writerows(rows)

    by_ticker = defaultdict(list)
    n_bad = 0
    for ticker, run, rec, rep in rows:
        ratio = rep / rec if rec > 0 else float("inf")
        clean = CLEAN_LO <= ratio <= CLEAN_HI
        n_bad += 0 if clean else 1
        by_ticker[ticker].append((run, rep, ratio, clean))

    print(f"{len(rows)} genomes checked, {n_bad} corrupted "
          f"({n_bad / len(rows) * 100:.1f}%) -- ratio outside "
          f"[{CLEAN_LO}, {CLEAN_HI}]")

    missing = []
    with open(best_dir / "index_corrected.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "run", "reproduced_val_mse"])
        for ticker in sorted(by_ticker):
            clean = [x for x in by_ticker[ticker] if x[3]]
            bad_runs = [x[0] for x in by_ticker[ticker] if not x[3]]
            if bad_runs:
                print(f"  {ticker}: corrupt runs {bad_runs} "
                      f"({len(clean)} clean remain)")
            if not clean:
                missing.append(ticker)
                continue
            run, rep, _, _ = min(clean, key=lambda x: x[1])
            w.writerow([ticker, run, f"{rep:.8f}"])
    print(f"wrote {best_dir / 'val_reproduce.csv'}")
    print(f"wrote {best_dir / 'index_corrected.csv'}")
    if missing:
        fail(f"no clean genome for: {missing} -- rerun those stocks")


if __name__ == "__main__":
    main()
