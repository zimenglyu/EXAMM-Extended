#!/bin/sh
# Evaluate every collected baseline genome on its held-out TEST set -- the
# one-time, paper-comparable evaluation -- and summarize per stock.
#
# Usage: sh scripts/stock_run/evaluate_baseline.sh [BEST_DIR]
#   BEST_DIR default: test_output/baseline_best   (output of collect_best_genomes.sh)
#
# For each (ticker,run) row in <BEST_DIR>/index.csv it runs evaluate_rnn on
# <ticker>_test.csv. The genome .bin carries its normalization stats, and
# write_predictions() DENORMALIZES both the expected and predicted columns
# (rnn/rnn.cxx ~639-645), so the metrics below are already in raw return
# scale -- directly comparable to the paper's per-stock tables. The MSE/MAE
# that evaluate_rnn prints to its own log are in NORMALIZED space; we ignore
# those and compute from the denormalized predictions CSV.
#
# Outputs (written into BEST_DIR):
#   test_results.csv  -- one row per (ticker,run): val_mse + test mae/mse
#                        + predict-zero baseline + directional accuracy
#   test_summary.csv  -- one row per ticker: the VALIDATION-selected run's test
#                        error (the honest single-model number -- picked by
#                        val_mse, never by test), plus mean/std/best over runs
# and prints an overall summary over the stocks.
#
# NOTE: one evaluate_rnn per genome (500 for the full baseline). Each is a fast
# single-threaded inference, but run this in an interactive job (sinteractive)
# or a small sbatch -- not on a login node.

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BEST_DIR=${1:-$REPO/test_output/baseline_best}
DATA=$REPO/datasets/701515_split
BIN=$REPO/build/rnn_examples/evaluate_rnn
INDEX=$BEST_DIR/index.csv

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found -- build evaluate_rnn first" >&2
    exit 1
fi
if [ ! -f "$INDEX" ]; then
    echo "ERROR: $INDEX not found -- run collect_best_genomes.sh first" >&2
    exit 1
fi

RESULTS=$BEST_DIR/test_results.csv
SUMMARY=$BEST_DIR/test_summary.csv
TMP=$BEST_DIR/.eval_tmp
rm -rf "$TMP"
mkdir -p "$TMP"

echo "ticker,run,val_mse,test_mae,test_mse,test_mae_zero,test_mse_zero,dir_acc" > "$RESULTS"

# Read index.csv via redirect (not a pipe) so the loop stays in this shell.
# Discard the header, then one genome per row.
{
    read _header
    while IFS=, read ticker run gen val_mse; do
        [ -n "$ticker" ] || continue
        val_mse=$(printf '%s' "$val_mse" | tr -d '\r')   # tolerate a CRLF index.csv
        bin="$BEST_DIR/${ticker}_run${run}.bin"
        testcsv="$DATA/${ticker}_test.csv"
        if [ ! -f "$bin" ]; then
            echo "WARN: genome $bin missing, skipping" >&2
            continue
        fi
        if [ ! -f "$testcsv" ]; then
            echo "WARN: test set $testcsv missing, skipping" >&2
            continue
        fi

        rm -rf "$TMP"/*
        "$BIN" --genome_file "$bin" --testing_filenames "$testcsv" \
            --time_offset 1 --output_directory "$TMP" \
            --std_message_level ERROR --file_message_level NONE > "$TMP/eval.log" 2>&1

        pred="$TMP/${ticker}_test_predictions.csv"
        if [ ! -f "$pred" ]; then
            echo "WARN: no predictions produced for $ticker run $run (see log in $TMP)" >&2
            continue
        fi

        # predictions are already denormalized -> compute raw-scale metrics directly
        metrics=$(python3 - "$pred" <<'PY'
import csv, sys
with open(sys.argv[1]) as f:
    r = csv.DictReader(f)
    ecol = [c for c in r.fieldnames if c.lstrip('#').startswith('expected_')][0]
    pcol = [c for c in r.fieldnames if c.lstrip('#').startswith('predicted_')][0]
    e, p = [], []
    for row in r:
        e.append(float(row[ecol])); p.append(float(row[pcol]))
n = len(e)
if n == 0:
    sys.exit(1)   # empty predictions -> emit nothing; caller skips with a WARN
mae  = sum(abs(a - b) for a, b in zip(e, p)) / n
mse  = sum((a - b) ** 2 for a, b in zip(e, p)) / n
mae0 = sum(abs(a) for a in e) / n           # predict-zero baseline
mse0 = sum(a * a for a in e) / n
acc  = sum(1 for a, b in zip(e, p) if a * b > 0) / n   # sign agreement
print(f"{mae:.8f},{mse:.8e},{mae0:.8f},{mse0:.8e},{acc:.6f}")
PY
)
        if [ -z "$metrics" ]; then
            echo "WARN: metric computation failed for $ticker run $run" >&2
            continue
        fi
        echo "${ticker},${run},${val_mse},${metrics}" >> "$RESULTS"
    done
} < "$INDEX"

rm -rf "$TMP"

# Per-stock summary + overall report.
python3 - "$RESULTS" "$SUMMARY" <<'PY'
import csv, sys, math
from collections import OrderedDict

results_file, summary_file = sys.argv[1], sys.argv[2]

by_ticker = OrderedDict()
with open(results_file) as f:
    for d in csv.DictReader(f):
        by_ticker.setdefault(d['ticker'], []).append({
            'run':      d['run'],
            'val_mse':  float(d['val_mse']),
            'test_mae': float(d['test_mae']),
            'test_mse': float(d['test_mse']),
            'mae_zero': float(d['test_mae_zero']),
            'mse_zero': float(d['test_mse_zero']),
            'dir_acc':  float(d['dir_acc']),
        })

if not by_ticker:
    sys.exit("ERROR: no rows in " + results_file + " -- every evaluation failed")

def mean(xs):
    return sum(xs) / len(xs)

def std(xs):
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))

# NOTE: min_test_* is the run that did best ON TEST -- optimistic/test-selected,
# informational only. The headline honest number is val_sel_test_* (chosen by
# validation, never by test).
cols = ['ticker', 'n_runs', 'val_sel_run', 'val_sel_test_mae', 'val_sel_test_mse',
        'mean_test_mae', 'std_test_mae', 'mean_test_mse', 'std_test_mse',
        'min_test_mae', 'min_test_mse', 'mean_mae_zero', 'mean_mse_zero', 'mean_dir_acc']
rows = []
vs_mae = []; vs_mse = []; z_mae = []; z_mse = []; acc_all = []
for t, runs in by_ticker.items():
    vsel = min(runs, key=lambda r: r['val_mse'])   # select by VALIDATION, report TEST
    maes = [r['test_mae'] for r in runs]
    mses = [r['test_mse'] for r in runs]
    zmae = mean([r['mae_zero'] for r in runs])
    zmse = mean([r['mse_zero'] for r in runs])
    dacc = mean([r['dir_acc'] for r in runs])
    rows.append([t, len(runs), vsel['run'],
                 vsel['test_mae'], vsel['test_mse'],
                 mean(maes), std(maes), mean(mses), std(mses),
                 min(maes), min(mses), zmae, zmse, dacc])
    vs_mae.append(vsel['test_mae']); vs_mse.append(vsel['test_mse'])
    z_mae.append(zmae); z_mse.append(zmse); acc_all.append(dacc)

with open(summary_file, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(cols)
    w.writerows(rows)

ns = len(by_ticker)
total = sum(len(v) for v in by_ticker.values())
print("stocks evaluated:               %d" % ns)
print("total (ticker,run) evaluations: %d" % total)
if ns != 50:
    print("NOTE: %d stocks (not 50) -- the sum-over-stocks line below is not directly"
          " comparable to the paper's 50-stock total" % ns)
print()
print("validation-selected model per stock (lowest-val run, report its TEST error):")
print("  mean over stocks   MAE %.5f   MSE %.3e" % (mean(vs_mae), mean(vs_mse)))
print("  sum  over stocks   MAE %.5f   MSE %.3e" % (sum(vs_mae), sum(vs_mse)))
print("predict-zero baseline (mean over stocks)   MAE %.5f   MSE %.3e" % (mean(z_mae), mean(z_mse)))
print("directional accuracy  (mean over stocks)   %.4f   (0.5 = coin flip)" % mean(acc_all))
print()
print("per-stock summary written to: %s" % summary_file)
print("per-run results in:           %s" % results_file)
PY
