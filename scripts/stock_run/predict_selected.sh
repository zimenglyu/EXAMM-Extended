#!/bin/sh
# Generate, verify, and package held-out TEST predictions for each stock's
# selected genome -- the input to the trading simulation (trade_portfolio.py).
#
# Usage: sh scripts/stock_run/predict_selected.sh [BEST_DIR] [INDEX_CSV]
#   BEST_DIR  default: test_output/baseline_best
#   INDEX_CSV default: <BEST_DIR>/index_corrected.csv
#             (header ticker,run,...; one row per stock naming the
#              validation-selected VERIFIED-CLEAN genome, built by the
#              val-reproduction sweep -- see docs/genome_corruption_bugfix.md)
#
# For each (ticker,run): runs evaluate_rnn on <ticker>_test.csv and keeps
# <BEST_DIR>/predictions/<ticker>_test_predictions.csv. write_predictions()
# DENORMALIZES expected_RET/predicted_RET back to raw return scale
# (rnn/rnn.cxx), so the CSVs feed Financial_toolbox directly.
#
# Per-output verification (any failure deletes the bad CSV and the script
# exits nonzero -- never trade on a partial predictions folder):
#   - prediction rows == test rows - 1  (--time_offset 1 alignment)
#   - no nan/inf anywhere in the CSV
# On full success writes predictions/MANIFEST.csv (ticker,run,n_pred,md5),
# copies the index for provenance, and tars the folder into ONE file for
# browser download via Open OnDemand:
#   <BEST_DIR>/baseline_best_predictions.tar.gz

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BEST_DIR=${1:-$REPO/test_output/baseline_best}
INDEX=${2:-$BEST_DIR/index_corrected.csv}
DATA=$REPO/datasets/701515_split
BIN=$REPO/build/rnn_examples/evaluate_rnn
PRED_DIR=$BEST_DIR/predictions
MANIFEST=$PRED_DIR/MANIFEST.csv

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found -- build evaluate_rnn first" >&2
    exit 1
fi
if [ ! -f "$INDEX" ]; then
    echo "ERROR: $INDEX not found -- run the val-reproduction sweep first" >&2
    exit 1
fi

# md5 tool differs between Linux (md5sum) and macOS (md5 -q)
if command -v md5sum > /dev/null 2>&1; then
    md5_of() { md5sum "$1" | awk '{print $1}'; }
elif command -v md5 > /dev/null 2>&1; then
    md5_of() { md5 -q "$1"; }
else
    echo "ERROR: neither md5sum nor md5 found" >&2
    exit 1
fi

TAR=$BEST_DIR/baseline_best_predictions.tar.gz
rm -rf "$PRED_DIR"
rm -f "$TAR"   # a stale tar from a previous run must never survive a failed rerun
mkdir -p "$PRED_DIR"
echo "ticker,run,n_pred,md5" > "$MANIFEST"

# data rows = non-blank lines after the header (robust to CRLF, blank lines,
# and a missing trailing newline, unlike wc -l)
expected=$(awk 'NR > 1 && $0 ~ /[^[:space:]\r]/' "$INDEX" | wc -l | tr -d ' ')
n_ok=0
n_fail=0
{
    read _header
    # "|| [ -n ... ]" keeps the last row even when the file has no trailing
    # newline (read returns nonzero there but still fills the variables)
    while IFS=, read ticker run _rest || [ -n "$ticker" ]; do
        ticker=$(printf '%s' "$ticker" | tr -d '\r ')
        run=$(printf '%s' "$run" | tr -d '\r ')
        [ -n "$ticker" ] || continue
        bin="$BEST_DIR/${ticker}_run${run}.bin"
        testcsv="$DATA/${ticker}_test.csv"
        if [ ! -f "$bin" ]; then
            echo "FAIL: genome $bin missing" >&2
            n_fail=$((n_fail + 1)); continue
        fi
        if [ ! -f "$testcsv" ]; then
            echo "FAIL: test set $testcsv missing" >&2
            n_fail=$((n_fail + 1)); continue
        fi

        "$BIN" --genome_file "$bin" --testing_filenames "$testcsv" \
            --time_offset 1 --output_directory "$PRED_DIR" \
            --std_message_level ERROR --file_message_level NONE > /dev/null 2>&1

        pred="$PRED_DIR/${ticker}_test_predictions.csv"
        if [ ! -f "$pred" ]; then
            echo "FAIL: no predictions produced for $ticker run $run" >&2
            n_fail=$((n_fail + 1)); continue
        fi

        # alignment: prediction rows (minus header) == test rows (minus header) - 1
        n_test=$(($(wc -l < "$testcsv") - 1))
        n_pred=$(($(wc -l < "$pred") - 1))
        if [ "$n_pred" -ne "$((n_test - 1))" ]; then
            echo "FAIL: $ticker predictions have $n_pred rows, expected $((n_test - 1))" >&2
            rm -f "$pred"
            n_fail=$((n_fail + 1)); continue
        fi

        # non-finite guard: NaN predictions would be silently LONGED by the
        # trading code (np.argsort sorts NaN last = "highest predicted return").
        # glibc prints nan, -nan, inf, -inf for doubles -- match all four.
        if grep -qiE '(^|,)-?(nan|inf)(,|$)' "$pred"; then
            echo "FAIL: non-finite value in $ticker predictions" >&2
            rm -f "$pred"
            n_fail=$((n_fail + 1)); continue
        fi

        echo "${ticker},${run},${n_pred},$(md5_of "$pred")" >> "$MANIFEST"
        echo "OK: $ticker run $run ($n_pred rows)"
        n_ok=$((n_ok + 1))
    done
} < "$INDEX"

echo "###-------------------###"
echo "predictions: $n_ok ok, $n_fail failed (index expects $expected)"

if [ "$n_fail" -gt 0 ] || [ "$n_ok" -ne "$expected" ]; then
    echo "ERROR: incomplete predictions -- fix failures before trading" >&2
    exit 1
fi

cp "$INDEX" "$PRED_DIR/"   # provenance: which genome produced each file
TAR=$BEST_DIR/baseline_best_predictions.tar.gz
tar -czf "$TAR" -C "$BEST_DIR" predictions
echo "packaged for download:"
echo "  $TAR"
echo "download via https://ondemand.anvil.rcac.purdue.edu -> Files, then extract"
echo "into test_output/baseline_best/ on your machine and run trade_portfolio.py"
