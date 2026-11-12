#!/bin/sh
# MAIN WORKFLOW, step 1 (forecasts). Renamed from the stale-model evaluation script.
# "Stale model" check: evaluate each fold's per-set best RET genome on its own test year AND the LATER test years,
# i.e. a model that is not retrained every year. Only years after the genome's validation year are
# used (a 2022-fold genome, trained <= 2020 and validated on 2021, on the 2023 and 2024 test years;
# a 2023-fold genome on 2024), so no genome is ever scored on data it trained or was chosen on.
#
# Genome per fold and set = the repeat with the lowest validation MSE (last row of fitness_log.csv,
# the same choice as main_core7.py), its final global best (highest-numbered global_best_genome_*.bin).
# Test files: datasets/panels_core7_examm_<Y>/<set>/*_test.csv of the target year Y (with its 40-day
# warm-up, which the backtest drops as usual).
#
# Output: test_output/core7_examm_ret_<F>/<set>/<repeat>/evaluation_test_y<Y>/<STOCK>_test_predictions.csv
#   (own test year Y = F: evaluation_test/, the folder evaluate_rnn_core7.sh and the other scripts use)
# So this one script makes every forecast core7_main_results.py --trade-every-day needs.
# Folders that already hold 50 prediction files are skipped, so it resumes after an interruption.
# Works on Anvil and on a Mac: EXAMM-Extended is found from this script's location.
#
# ALL=1 also evaluates EARLIER test years (2023-fold genomes on 2022, 2024-fold genomes on 2022 and 2023).
# Those genomes trained or were chosen on that year, so their results contain LEAKAGE: use them only to
# compare with a paper that reports such numbers, never as a test result.
#
# Usage:
#   sh scripts/rnn_examples/evaluate_rnn_core7_main.sh            # later years only (no leakage)
#   ALL=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh      # every fold on every test year
#   ALL_REPEATS=1 sh scripts/rnn_examples/evaluate_rnn_core7_main.sh   # also the other 9 repeats of every
#       fold on the later years (needed by core7_main_results.py --pareto for its never-retrain table)
# Then the backtest:  cd ~/Financial_toolbox && python3 core7_main_results.py
SETS="set1 set2 set3 set4"
E=$(cd "$(dirname "$0")/../.." && pwd)

done_n=0; skip_n=0; fail_n=0; missing_n=0
for F in 2022 2023 2024; do
    for Y in 2022 2023 2024; do
        [ "$Y" -lt "$F" ] && [ "${ALL:-0}" != "1" ] && continue # earlier year = leakage, only with ALL=1
        for s in $SETS; do
            base=$E/test_output/core7_examm_ret_$F/$s
            # repeat with the lowest validation MSE (column 5 of the last fitness_log.csv row)
            best=""; best_mse=""
            for r in 15 16 17 18 19 20 21 22 23 24; do
                log=$base/$r/fitness_log.csv
                [ -f "$log" ] || continue
                mse=$(tail -1 "$log" | cut -d, -f5)
                if [ -z "$best" ] || awk "BEGIN{exit !($mse < $best_mse)}"; then best=$r; best_mse=$mse; fi
            done
            if [ -z "$best" ]; then
                echo "!! fold $F $s: no fitness_log.csv in $base/<repeat> -- download the runs first"
                missing_n=$((missing_n + 1)); continue
            fi
            list=$best   # the lowest-val-MSE repeat; with ALL_REPEATS=1 every repeat on the later years
            if [ "${ALL_REPEATS:-0}" = "1" ] && [ "$Y" -ne "$F" ]; then list="15 16 17 18 19 20 21 22 23 24"; fi
            for r in $list; do
                [ -f "$base/$r/fitness_log.csv" ] || continue
                genome=$(ls "$base/$r"/global_best_genome_*.bin 2>/dev/null \
                         | sed 's/.*global_best_genome_\([0-9]*\)\.bin$/\1/' | sort -n | tail -1)
                if [ -z "$genome" ]; then
                    echo "!! fold $F $s/$r: no global_best_genome_*.bin"; missing_n=$((missing_n + 1)); continue
                fi
                if [ "$Y" -eq "$F" ]; then out=$base/$r/evaluation_test; else out=$base/$r/evaluation_test_y$Y; fi
                n=$(ls "$out"/*_test_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                if [ "$n" = "50" ]; then skip_n=$((skip_n + 1)); continue; fi
                if ! ls $E/datasets/panels_core7_examm_$Y/$s/*_test.csv >/dev/null 2>&1; then
                    echo "!! no test files in datasets/panels_core7_examm_$Y/$s -- download them first"
                    missing_n=$((missing_n + 1)); continue
                fi
                mkdir -p "$out"
                echo "== fold $F $s/$r/global_best_genome_$genome on test year $Y"
                $E/build/rnn_examples/evaluate_rnn \
                --testing_filenames $E/datasets/panels_core7_examm_$Y/$s/*_test.csv \
                --time_offset 1 \
                --genome_file "$base/$r/global_best_genome_$genome.bin" \
                --output_directory "$out" \
                --std_message_level INFO \
                --file_message_level INFO 2>&1 | grep -E "MSE:" | tail -1
                n=$(ls "$out"/*_test_predictions.csv 2>/dev/null | wc -l | tr -d ' ')
                if [ "$n" = "50" ]; then done_n=$((done_n + 1)); else
                    echo "!! fold $F $s on $Y: only $n of 50 prediction files written"; fail_n=$((fail_n + 1)); fi
            done
        done
    done
done
echo "evaluated $done_n, already done $skip_n, failed $fail_n, missing $missing_n"
[ $fail_n -eq 0 ] && [ $missing_n -eq 0 ]
