#!/bin/sh
# Rename genome .bin files in results/benchmark to a consistent name:
# results/benchmark/DATASET/RUN/global_best_genome_*.bin -> global_best_genome_RUN.bin
# Run from repo root or pass path to results/benchmark as first argument.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="${1:-$REPO_ROOT/results/benchmark}"
cd "$REPO_ROOT" || exit 1

for dataset_dir in "$RESULTS_DIR"/*/; do
    [ -d "$dataset_dir" ] || continue
    dataset=$(basename "$dataset_dir")
    for run_dir in "$dataset_dir"*/; do
        [ -d "$run_dir" ] || continue
        run=$(basename "$run_dir")
        target_name="global_best_genome_${run}.bin"
        for bin in "$run_dir"global_best_genome_*.bin; do
            [ -f "$bin" ] || continue
            current=$(basename "$bin")
            if [ "$current" = "$target_name" ]; then
                break
            fi
            mv "$bin" "$run_dir$target_name"
            echo "Renamed: $run_dir$current -> $target_name"
            break
        done
    done
done
echo "Done renaming genome bin files."
