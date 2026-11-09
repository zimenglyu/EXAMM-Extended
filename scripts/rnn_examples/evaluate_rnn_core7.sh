#!/bin/sh
# Evaluate core7 EXAMM genomes on one split of their panel data.
# Works on Anvil and on a Mac: EXAMM-Extended is found from this script's location
# (scripts/rnn_examples/ -> ../..), so keep it in that folder.
#
# TARGET defaults to RET:
#   YEAR unset -> the RET runs of all three folds (2022, 2023, 2024), one after another
#   YEAR=2022  -> the RET runs of that fold: test_output/core7_examm_ret_2022, datasets/panels_core7_examm_2022
# TARGET=RET_CS:
#   YEAR unset -> the 2020-2024 run:      test_output/core7_examm,      datasets/panels_core7_examm
#   YEAR=2022  -> the RET_CS runs of that fold: test_output/core7_examm_2022, datasets/panels_core7_examm_2022
#
# Usage (from anywhere):
#   SPLIT=test sh scripts/rnn_examples/evaluate_rnn_core7.sh                         # RET runs of 2022, 2023, 2024
#   SPLIT=test TARGET=RET_CS sh scripts/rnn_examples/evaluate_rnn_core7.sh           # 4 candidates, 2020-2024 run
#   SPLIT=test YEAR=2023 sh scripts/rnn_examples/evaluate_rnn_core7.sh               # RET runs of the 2023 fold
#   SPLIT=test YEAR=2023 TARGET=RET_CS sh scripts/rnn_examples/evaluate_rnn_core7.sh # RET_CS runs of the 2023 fold
#   SPLIT=test YEAR=2023 TARGET=RET_CS sh scripts/rnn_examples/evaluate_rnn_core7.sh set3/21/global_best_genome_9694
#
# Predictions go to <results>/<set>/<repeat>/evaluation_<SPLIT>/
SPLIT=${SPLIT:-val}
E=$(cd "$(dirname "$0")/../.." && pwd)

# RET (the default) without a YEAR: run every RET fold, each with this same script
if [ -z "${YEAR:-}" ] && [ "${TARGET:-RET}" = "RET" ]; then
    if [ $# -gt 0 ]; then
        echo "naming genomes with the RET target needs YEAR=2022, 2023 or 2024"; exit 1
    fi
    status=0
    for Y in 2022 2023 2024; do
        SPLIT=$SPLIT YEAR=$Y TARGET=RET sh "$0" || status=1
    done
    exit $status
fi

# best repeat per set by validation MSE (final global best genome of that repeat)
case "${YEAR:-}" in
  "")   RESULTS=core7_examm;      DATA=panels_core7_examm
        DEFAULT="set1/15/global_best_genome_9927 set2/17/global_best_genome_9672 set3/24/global_best_genome_10226 set4/24/global_best_genome_9871" ;;
  2022) RESULTS=core7_examm_2022; DATA=panels_core7_examm_2022
        DEFAULT="set1/20/global_best_genome_17249 set2/21/global_best_genome_15175 set3/20/global_best_genome_9056 set4/19/global_best_genome_9936" ;;
  2023) RESULTS=core7_examm_2023; DATA=panels_core7_examm_2023
        DEFAULT="set1/22/global_best_genome_9872 set2/20/global_best_genome_10309 set3/21/global_best_genome_9694 set4/19/global_best_genome_11164" ;;
  2024) RESULTS=core7_examm_2024; DATA=panels_core7_examm_2024
        DEFAULT="set1/19/global_best_genome_10107 set2/19/global_best_genome_9498 set3/23/global_best_genome_12872 set4/23/global_best_genome_9292" ;;
  *)    echo "unknown YEAR '$YEAR' (use 2022, 2023, 2024, or leave unset)"; exit 1 ;;
esac

# RET-target runs: same data, own results folder and global best genomes (default target)
TARGET=${TARGET:-RET}
case "$TARGET" in
  RET_CS) ;;
  RET)
    case "${YEAR:-}" in
      2022) DEFAULT="set1/21/global_best_genome_16410 set2/16/global_best_genome_9708 set3/22/global_best_genome_10627 set4/24/global_best_genome_10937" ;;
      2023) DEFAULT="set1/17/global_best_genome_9847 set2/21/global_best_genome_9765 set3/17/global_best_genome_10525 set4/21/global_best_genome_9849" ;;
      2024) DEFAULT="set1/15/global_best_genome_10639 set2/15/global_best_genome_9705 set3/22/global_best_genome_9589 set4/24/global_best_genome_10469" ;;
      *)    echo "TARGET=RET needs YEAR=2022, 2023 or 2024"; exit 1 ;;
    esac
    RESULTS=core7_examm_ret_$YEAR ;;
  *) echo "unknown TARGET '$TARGET' (use RET_CS or RET)"; exit 1 ;;
esac
GENOMES=${*:-$DEFAULT}

for p in $GENOMES; do
    s=${p%%/*}
    out=$E/test_output/$RESULTS/$(dirname $p)/evaluation_$SPLIT
    mkdir -p $out
    echo "== ${YEAR:-2020-2024} $TARGET $p ($SPLIT)"
    $E/build/rnn_examples/evaluate_rnn \
    --testing_filenames $E/datasets/$DATA/$s/*_$SPLIT.csv \
    --time_offset 1 \
    --genome_file $E/test_output/$RESULTS/$p.bin \
    --output_directory $out \
    --std_message_level INFO \
    --file_message_level INFO 2>&1 | grep -E "MSE:|MAE:"
done
