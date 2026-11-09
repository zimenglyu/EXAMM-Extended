#!/bin/sh
# Evaluate EVERY repeat (folders 15..24) of the core7 Pooled RET runs, not only the one with the
# lowest validation MSE: each repeat's final global best genome (the highest-numbered
# global_best_genome_*.bin in its folder -- earlier ones are superseded bests) is run on the
# chosen split of its own panel set and fold. Each genome is evaluated on its own (no ensemble).
# Works on Anvil and on a Mac: EXAMM-Extended is found from this script's location
# (scripts/rnn_examples/ -> ../..), so keep it in that folder.
#
# Repeats that already have 50 <STOCK>_<SPLIT>_predictions.csv files are skipped, so it can be
# rerun after an interruption, and the 12 genomes evaluated earlier are not redone.
#
# Usage (from anywhere):
#   sh scripts/rnn_examples/evaluate_rnn_core7_repeats.sh                       # test, folds 2022 2023 2024
#   YEARS=2023 sh scripts/rnn_examples/evaluate_rnn_core7_repeats.sh            # one fold
#   SPLIT=val sh scripts/rnn_examples/evaluate_rnn_core7_repeats.sh             # validation instead
#
# Predictions go to test_output/core7_examm_ret_<Y>/<set>/<repeat>/evaluation_<SPLIT>/
SPLIT=${SPLIT:-test}
YEARS=${YEARS:-"2022 2023 2024"}
E=$(cd "$(dirname "$0")/../.." && pwd)

done_n=0; skip_n=0; fail_n=0; missing_n=0
for Y in $YEARS; do
    for s in set1 set2 set3 set4; do
        for r in 15 16 17 18 19 20 21 22 23 24; do
            dir=$E/test_output/core7_examm_ret_$Y/$s/$r
            # final global best = highest genome number in the folder
            genome=$(ls "$dir"/global_best_genome_*.bin 2>/dev/null \
                     | sed 's/.*global_best_genome_\([0-9]*\)\.bin$/\1/' | sort -n | tail -1)
            if [ -z "$genome" ]; then
                echo "!! $Y $s/$r: no global_best_genome_*.bin -- download it first"
                missing_n=$((missing_n + 1)); continue
            fi
            out=$dir/evaluation_$SPLIT
            n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
            if [ "$n" = "50" ]; then
                skip_n=$((skip_n + 1)); continue
            fi
            mkdir -p "$out"
            echo "== $Y RET $s/$r/global_best_genome_$genome ($SPLIT)"
            $E/build/rnn_examples/evaluate_rnn \
            --testing_filenames $E/datasets/panels_core7_examm_$Y/$s/*_$SPLIT.csv \
            --time_offset 1 \
            --genome_file "$dir/global_best_genome_$genome.bin" \
            --output_directory "$out" \
            --std_message_level INFO \
            --file_message_level INFO 2>&1 | grep -E "MSE:|MAE:" | tail -1
            n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
            if [ "$n" = "50" ]; then done_n=$((done_n + 1)); else
                echo "!! $Y $s/$r: only $n of 50 prediction files written"; fail_n=$((fail_n + 1)); fi
        done
    done
done
echo "evaluated $done_n, already done $skip_n, failed $fail_n, genome missing $missing_n"
[ $fail_n -eq 0 ] && [ $missing_n -eq 0 ]
