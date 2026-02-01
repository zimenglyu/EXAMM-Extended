#!/bin/sh
# Run evaluate_rnn on all benchmark genomes: for each dataset, run 0-9 genomes
# on that dataset's testing data. Results (MSE, MAE) are printed to the terminal
# and appended to a CSV file (no impact on measured latency/throughput; CSV is
# written from parsed output after each run finishes).
# Usage: from repo root, run:
#   sh scripts/rnn_examples/evaluate_benchmark_rnn.sh
# Requires: build/rnn_examples/evaluate_rnn and genome .bin files in RESULTS_DIR.
#
# Optional env overrides (paths relative to REPO_ROOT or absolute):
#   RESULTS_DIR   directory containing <dataset>/<run>/global_best_genome_<run>.bin or global_best_genome_*.bin (default: results/benchmark_new)
#   EVAL_OUT_DIR  where to write evaluation outputs (default: results/benchmark_eval_minmax)
# Example: evaluate two-layer LSTM genomes from random_two_layer_lstm_1_0.sh:
#   RESULTS_DIR=results/benchmark_two_layer_lstm EVAL_OUT_DIR=results/benchmark_two_layer_lstm_eval sh scripts/rnn_examples/evaluate_benchmark_rnn.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="${RESULTS_DIR:-$REPO_ROOT/results/benchmark_new}"
case "$RESULTS_DIR" in
    /*) ;;
    *) RESULTS_DIR="$REPO_ROOT/$RESULTS_DIR" ;;
esac
BENCHMARKS_DIR="$REPO_ROOT/datasets/benchmarks"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-$REPO_ROOT/results/benchmark_eval_minmax}"
case "$EVAL_OUT_DIR" in
    /*) ;;
    *) EVAL_OUT_DIR="$REPO_ROOT/$EVAL_OUT_DIR" ;;
esac
CSV_FILE="${CSV_FILE:-$EVAL_OUT_DIR/benchmark_results_minmax.csv}"
TMP_OUT="/tmp/evaluate_rnn_out_$$"

cd "$BUILD_DIR" || exit 1
mkdir -p "$EVAL_OUT_DIR"
printf "%s\n" "dataset,run,MSE,MAE,parameter_count,inference_ms,throughput" > "$CSV_FILE"

# dataset_name -> path to test file (relative to build/)
# ETT-small subdatasets live under datasets/benchmarks/ETT-small/
# Using *_test_minmax.csv (minmax-scaled test data)
eval_test_file() {
    case "$1" in
        ETTh1)  echo "../datasets/benchmarks/ETT-small/ETTh1_test_minmax.csv" ;;
        ETTm1)  echo "../datasets/benchmarks/ETT-small/ETTm1_test_minmax.csv" ;;
        exchange_rate) echo "../datasets/benchmarks/exchange_rate/exchange_rate_test_minmax.csv" ;;
        illness) echo "../datasets/benchmarks/illness/illness_test_minmax.csv" ;;
        weather) echo "../datasets/benchmarks/weather/weather_test_minmax.csv" ;;
        electricity) echo "../datasets/benchmarks/electricity/electricity_test_minmax.csv" ;;
        *) echo "" ;;
    esac
}

# Datasets from ETTh (ETTh1) through weather for inference; each uses its test_minmax.csv
for dataset in ETTh1 ETTm1 exchange_rate illness weather; do
    test_file=$(eval_test_file "$dataset")
    if [ -z "$test_file" ]; then
        echo "Skipping unknown dataset: $dataset"
        continue
    fi
    if [ ! -f "$test_file" ]; then
        echo "Skipping $dataset: test file not found: $test_file"
        continue
    fi
    for i in 0 1 2 3 4 5 6 7 8 9; do
        run_dir="$RESULTS_DIR/$dataset/$i/"
        genome_path="$run_dir/global_best_genome_$i.bin"
        if [ ! -f "$genome_path" ]; then
            # Fallback: use the single global_best_genome_*.bin in this run dir (e.g. weather uses genome-id naming)
            genome_path=""
            for bin in "$run_dir"global_best_genome_*.bin; do
                [ -f "$bin" ] || continue
                genome_path="$bin"
                break
            done
        fi
        if [ -z "$genome_path" ] || [ ! -f "$genome_path" ]; then
            echo "[$dataset run=$i] SKIP (genome not found in $run_dir)"
            continue
        fi
        out_dir="$EVAL_OUT_DIR/$dataset/$i"
        mkdir -p "$out_dir"
        echo "---------- $dataset run=$i ----------"
        ./rnn_examples/evaluate_rnn \
            --testing_filenames "$test_file" \
            --time_offset 1 \
            --genome_file "$genome_path" \
            --output_directory "$out_dir" \
            --std_message_level INFO \
            --file_message_level INFO 2>&1 | tee "$TMP_OUT"
        awk -v dataset="$dataset" -v run="$i" '
            /MSE:/ { mse=$NF }
            /MAE:/ { mae=$NF }
            /Parameter count:/ { params=$NF }
            /Inference time \(entire dataset\):/ {
                for (j=1;j<=NF;j++) if ($j ~ /^\([0-9]+\.[0-9]+$/) { gsub(/[()]/,"",$j); infer_ms=$j; break }
            }
            /Throughput:/ { for (k=1;k<=NF;k++) if ($k ~ /^[0-9]+\.[0-9]+$/) { throughput=$k; break } }
            END {
                printf "%s,%s,%s,%s,%s,%s,%s\n", dataset, run, mse+0, mae+0, params+0, infer_ms+0, throughput+0
            }
        ' "$TMP_OUT" >> "$CSV_FILE"
        echo ""
    done
done
rm -f "$TMP_OUT"
echo "All benchmark evaluations finished."
echo "Results CSV: $CSV_FILE"
