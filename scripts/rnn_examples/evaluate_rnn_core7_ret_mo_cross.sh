#!/bin/sh
# Multi-objective genome frontier: evaluate every CANDIDATE genome of the multi-objective EXAMM runs
# (test_output/core7_examm_ret_mo_<Y>/candidates.csv, written by select_core7_mo_candidates.py) on EVERY
# panel set, validation and test, so each genome can be scored as ONE genome shared by all 200 stocks.
# The same evaluation as EXAMM-Extended/scripts/rnn_examples/evaluate_rnn_core7_ret_cross.sh, with the
# candidate list in place of "each repeat's final global best". Multi-objective genome files load in the
# ordinary evaluate_rnn (the extra IR block at the end of the file is simply not read).
#
# Output (evaluate_rnn names its files <STOCK>_<split>_predictions.csv after the input file):
#   <run>/<set>/<repeat>/mo_eval/<genome_id>/evaluation_<split>_on_<target>/
# Folders that already hold 50 prediction files are skipped, so it resumes after an interruption.
# Meant for the Mac (evaluations run there): copy test_output/core7_examm_ret_mo_<Y>/ from Anvil into
# ~/EXAMM-Extended/test_output/ first.
#
# Usage:
#   sh evaluate_rnn_core7_ret_mo_cross.sh                      # all folds, val and test
#   YEARS=2023 sh evaluate_rnn_core7_ret_mo_cross.sh           # one fold
#   EXAMM=~/EXAMM-Extended RUN=core7_examm_ret_mo_test sh evaluate_rnn_core7_ret_mo_cross.sh   # the test run
YEARS=${YEARS:-"2022 2023 2024"}
SPLITS=${SPLITS:-"val test"}
SETS="set1 set2 set3 set4"
E=${EXAMM:-$HOME/EXAMM-Extended}          # evaluate_rnn build and datasets/
RUN=${RUN:-core7_examm_ret_mo}

done_n=0; skip_n=0; fail_n=0; missing_n=0
for Y in $YEARS; do
    base=$E/test_output/${RUN}_$Y
    if [ ! -f "$base/candidates.csv" ]; then
        echo "!! no $base/candidates.csv -- run select_core7_mo_candidates.py first"
        missing_n=$((missing_n + 1)); continue
    fi
    tail -n +2 "$base/candidates.csv" | while IFS=, read -r s r g file rest; do
        echo "$s $r $g $file"
    done > "$base/.candidates.txt"
    while read -r s r g file; do
        dir=$base/$s/$r
        if [ ! -f "$dir/$file" ]; then
            echo "!! $Y $s/$r: no $file -- download it first"
            missing_n=$((missing_n + 1)); continue
        fi
        for t in $SETS; do
            for SPLIT in $SPLITS; do
                out=$dir/mo_eval/$g/evaluation_${SPLIT}_on_$t
                n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                if [ "$n" = "50" ]; then
                    skip_n=$((skip_n + 1)); continue
                fi
                if ! ls $E/datasets/panels_core7_examm_$Y/$t/*_$SPLIT.csv >/dev/null 2>&1; then
                    echo "!! no $SPLIT files in datasets/panels_core7_examm_$Y/$t -- download them first"
                    missing_n=$((missing_n + 1)); continue
                fi
                mkdir -p "$out"
                echo "== $Y $s/$r/$file on $t ($SPLIT)"
                $E/build/rnn_examples/evaluate_rnn \
                --testing_filenames $E/datasets/panels_core7_examm_$Y/$t/*_$SPLIT.csv \
                --time_offset 1 \
                --genome_file "$dir/$file" \
                --output_directory "$out" \
                --std_message_level INFO \
                --file_message_level INFO 2>&1 | grep -E "MSE:|MAE:" | tail -1
                n=$(ls "$out"/*_${SPLIT}_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                if [ "$n" = "50" ]; then done_n=$((done_n + 1)); else
                    echo "!! $Y $s/$r/$file on $t ($SPLIT): only $n of 50 prediction files written"; fail_n=$((fail_n + 1)); fi
            done
        done
    done < "$base/.candidates.txt"
done
echo "evaluated $done_n, already done $skip_n, failed $fail_n, missing $missing_n"
[ $fail_n -eq 0 ] && [ $missing_n -eq 0 ]
