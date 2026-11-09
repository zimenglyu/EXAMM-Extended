#!/bin/sh
# Genome frontier (professor's design): evaluate EVERY RET genome of the paper's Train/Val/Test folds
# on EVERY panel set, so each genome can be scored as ONE genome shared by all 200 stocks.
#   folds 2022 2023 2024 (test_output/core7_examm_ret_<Y>), source set x repeat 15..24 = 40 genomes
#   per fold, each run on the 4 target sets' _val.csv and _test.csv (datasets/panels_core7_examm_<Y>).
# Genome = the repeat's final global best (highest-numbered global_best_genome_*.bin), as in
# evaluate_rnn_core7_repeats.sh. Each genome is evaluated on its own (no ensemble).
#
# Output folders (evaluate_rnn names its files <STOCK>_<split>_predictions.csv after the input file):
#   own set     -> <repeat>/evaluation_<split>/            (the folders earlier scripts use; reused)
#   other sets  -> <repeat>/evaluation_<split>_on_<target>/
# Folders that already hold 50 prediction files are skipped, so it resumes after an interruption.
# Works on Anvil and on a Mac: EXAMM-Extended is found from this script's location.
#
# Usage:
#   sh scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh                    # all folds, val and test
#   YEARS=2023 sh scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh         # one fold
#   SPLITS=val sh scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh         # validation only
YEARS=${YEARS:-"2022 2023 2024"}
SPLITS=${SPLITS:-"val test"}
SETS="set1 set2 set3 set4"
E=$(cd "$(dirname "$0")/../.." && pwd)

done_n=0; skip_n=0; fail_n=0; missing_n=0
for Y in $YEARS; do
    for s in $SETS; do
        for r in 15 16 17 18 19 20 21 22 23 24; do
            dir=$E/test_output/core7_examm_ret_$Y/$s/$r
            genome=$(ls "$dir"/global_best_genome_*.bin 2>/dev/null \
                     | sed 's/.*global_best_genome_\([0-9]*\)\.bin$/\1/' | sort -n | tail -1)
            if [ -z "$genome" ]; then
                echo "!! $Y $s/$r: no global_best_genome_*.bin -- download it first"
                missing_n=$((missing_n + 1)); continue
            fi
            for t in $SETS; do
                for SPLIT in $SPLITS; do
                    if [ "$t" = "$s" ]; then out=$dir/evaluation_$SPLIT; else out=$dir/evaluation_${SPLIT}_on_$t; fi
                    n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                    if [ "$n" = "50" ]; then
                        skip_n=$((skip_n + 1)); continue
                    fi
                    if ! ls $E/datasets/panels_core7_examm_$Y/$t/*_$SPLIT.csv >/dev/null 2>&1; then
                        echo "!! no $SPLIT files in datasets/panels_core7_examm_$Y/$t -- download them first"
                        missing_n=$((missing_n + 1)); continue
                    fi
                    mkdir -p "$out"
                    echo "== $Y $s/$r/global_best_genome_$genome on $t ($SPLIT)"
                    $E/build/rnn_examples/evaluate_rnn \
                    --testing_filenames $E/datasets/panels_core7_examm_$Y/$t/*_$SPLIT.csv \
                    --time_offset 1 \
                    --genome_file "$dir/global_best_genome_$genome.bin" \
                    --output_directory "$out" \
                    --std_message_level INFO \
                    --file_message_level INFO 2>&1 | grep -E "MSE:|MAE:" | tail -1
                    n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                    if [ "$n" = "50" ]; then done_n=$((done_n + 1)); else
                        echo "!! $Y $s/$r on $t ($SPLIT): only $n of 50 prediction files written"; fail_n=$((fail_n + 1)); fi
                done
            done
        done
    done
done
echo "evaluated $done_n, already done $skip_n, failed $fail_n, missing $missing_n"
[ $fail_n -eq 0 ] && [ $missing_n -eq 0 ]
