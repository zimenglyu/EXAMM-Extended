#!/bin/sh
# Collect the best evolved genome from every completed baseline run into one
# flat folder -- the professor's request: save each repeated run's global
# best genome (.bin and .gv). The .txt (equations + best_validation_mse) is
# copied too since it's tiny and needed to pick the best run per stock.
#
# Usage: sh scripts/stock_run/collect_best_genomes.sh [DEST]
#   DEST defaults to test_output/baseline_best
#
# For each test_output/baseline/<TICKER>/run_<R>/ that has a .done marker,
# copies its global_best_genome_*.{bin,gv,txt} to
#   <DEST>/<TICKER>_run<R>.{bin,gv,txt}
# and writes <DEST>/index.csv (ticker,run,generation_id,best_validation_mse).
# Runs without .done, or completed runs missing a genome, are warned and skipped.

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BASE_OUT=${BASE_OUT:-$REPO/test_output/baseline}   # env-overridable (e.g. pilot runs)
DEST=${1:-$REPO/test_output/baseline_best}

if [ ! -d "$BASE_OUT" ]; then
    echo "ERROR: $BASE_OUT not found -- run the baseline first" >&2
    exit 1
fi

mkdir -p "$DEST"
INDEX=$DEST/index.csv
echo "ticker,run,generation_id,best_validation_mse" > "$INDEX"

collected=0
missing_genome=0
not_done=0

for run_dir in "$BASE_OUT"/*/run_*/; do
    [ -d "$run_dir" ] || continue
    run_name=$(basename "$run_dir")            # run_<R>
    R=${run_name#run_}
    S=$(basename "$(dirname "$run_dir")")       # <TICKER>

    if [ ! -f "$run_dir/.done" ]; then
        echo "WARN: $S run $R has no .done (incomplete), skipping" >&2
        not_done=$((not_done + 1))
        continue
    fi

    # usually exactly one global_best per run; if more, take the lowest recorded MSE
    best_txt=$(for f in "$run_dir"/global_best_genome_*.txt; do
        [ -f "$f" ] || continue
        mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$f" | awk '{print $2}')
        [ -n "$mse" ] && echo "$mse $f"
    done | sort -n | head -1 | awk '{print $2}')

    if [ -z "$best_txt" ]; then
        echo "WARN: $S run $R is done but has no global_best_genome_*.txt, skipping" >&2
        missing_genome=$((missing_genome + 1))
        continue
    fi

    base=${best_txt%.txt}                        # .../global_best_genome_<id>
    if [ ! -f "$base.bin" ] || [ ! -f "$base.gv" ]; then
        echo "WARN: $S run $R missing .bin or .gv alongside $(basename "$base"), skipping" >&2
        missing_genome=$((missing_genome + 1))
        continue
    fi

    gen_id=$(basename "$base" | sed 's/global_best_genome_//')
    mse=$(grep -o 'best_validation_mse: [0-9.eE+-]*' "$best_txt" | awk '{print $2}')

    cp "$base.bin" "$DEST/${S}_run${R}.bin"
    cp "$base.gv"  "$DEST/${S}_run${R}.gv"
    cp "$best_txt" "$DEST/${S}_run${R}.txt"
    echo "${S},${R},${gen_id},${mse}" >> "$INDEX"
    collected=$((collected + 1))
done

echo "###-------------------###"
echo "collected: $collected genomes -> $DEST"
echo "index:     $INDEX"
[ "$not_done" -gt 0 ] && echo "WARNING: $not_done run dir(s) had no .done (incomplete)"
[ "$missing_genome" -gt 0 ] && echo "WARNING: $missing_genome completed run(s) had no usable genome"
if [ "$collected" -ne 500 ]; then
    echo "NOTE: expected 500 (50 stocks x 10 runs), got $collected -- check warnings above"
fi
