#!/bin/sh
# baseline runs: 50 stocks, 10 runs each
# 10 islands x 10, repopulation every 200 genomes, 10k max genomes,
# z-score norm, num_mutations 1, bp 10, everything else default
#
# usage: sh scripts/stock_run/baseline_train.sh            (all stocks)
#        sh scripts/stock_run/baseline_train.sh AKAM ATO   (just some)
#
# finished runs get a .done file and are skipped, so rerunning after a
# crash picks up where it left off

REPO=$(cd "$(dirname "$0")/../.." && pwd)
BIN=$REPO/build/mpi/examm_mpi
DATA=$REPO/datasets/701515_split
BASE_OUT=$REPO/test_output/baseline

RUNS=${RUNS:-10}
MAX_GENOMES=${MAX_GENOMES:-10000}
PROCS=${PROCS:-8}
# these two are for the cluster job array (one run per job, launched with srun)
ONLY_RUN=${ONLY_RUN:-}
MPI_LAUNCH=${MPI_LAUNCH:-"mpirun -np $PROCS"}

INPUTS="RET VOL_CHANGE BA_SPREAD ILLIQUIDITY sprtrn TURNOVER"
OUTPUTS="RET"

if [ ! -x "$BIN" ]; then
    echo "ERROR: $BIN not found -- build examm_mpi first" >&2
    exit 1
fi
LAUNCHER=${MPI_LAUNCH%% *}
if ! command -v "$LAUNCHER" > /dev/null 2>&1; then
    echo "ERROR: $LAUNCHER not found" >&2
    exit 1
fi

# stocks from args, or everything in the data folder
if [ $# -gt 0 ]; then
    TICKERS="$@"
else
    TICKERS=$(ls "$DATA" | sed -E 's/_(train|val|test)\.csv$//' | sort -u | grep -v combined_predictors)
fi

# make sure the files exist before starting anything
for S in $TICKERS; do
    for split in train val test; do
        if [ ! -f "$DATA/${S}_${split}.csv" ]; then
            echo "ERROR: $DATA/${S}_${split}.csv does not exist" >&2
            exit 1
        fi
    done
done

N_TICKERS=$(echo $TICKERS | wc -w | tr -d ' ')
mkdir -p "$BASE_OUT"
FAIL_LOG=$BASE_OUT/failures.log

echo "baseline: ${N_TICKERS} stocks x ${RUNS} runs, max_genomes=${MAX_GENOMES}"
echo "results: ${BASE_OUT}"
echo "###-------------------###"

for S in $TICKERS; do
    if [ -n "$ONLY_RUN" ]; then
        RUN=$ONLY_RUN
        RUNS=$ONLY_RUN
    else
        RUN=1
    fi
    while [ $RUN -le $RUNS ]; do
        OUT=$BASE_OUT/$S/run_$RUN
        if [ -f "$OUT/.done" ]; then
            echo "[$S run $RUN/$RUNS] already done, skipping"
            RUN=$((RUN + 1))
            continue
        fi
        # examm appends into old output dirs, so start clean
        rm -rf "$OUT"
        mkdir -p "$OUT"

        echo "[$S run $RUN/$RUNS] started $(date '+%Y-%m-%d %H:%M:%S')"
        $MPI_LAUNCH "$BIN" \
            --training_filenames "$DATA/${S}_train.csv" \
            --validation_filenames "$DATA/${S}_val.csv" \
            --time_offset 1 \
            --input_parameter_names $INPUTS \
            --output_parameter_names $OUTPUTS \
            --number_islands 10 \
            --island_size 10 \
            --max_genomes $MAX_GENOMES \
            --bp_iterations 10 \
            --num_mutations 1 \
            --normalize avg_std_dev \
            --extinction_event_generation_number 200 \
            --islands_to_exterminate 1 \
            --repopulation_method bestGenome \
            --output_directory "$OUT" \
            --std_message_level INFO \
            --file_message_level NONE > "$OUT/train.log" 2>&1

        if [ $? -eq 0 ]; then
            touch "$OUT/.done"
            echo "[$S run $RUN/$RUNS] finished $(date '+%Y-%m-%d %H:%M:%S')"
        else
            echo "$(date '+%Y-%m-%d %H:%M:%S') $S run_$RUN failed, see $OUT/train.log" | tee -a "$FAIL_LOG" >&2
        fi
        RUN=$((RUN + 1))
    done
done

echo "###-------------------###"
DONE_COUNT=$(find "$BASE_OUT" -name .done | wc -l | tr -d ' ')
echo "done: ${DONE_COUNT} finished runs in ${BASE_OUT}"
if [ -f "$FAIL_LOG" ]; then
    echo "some runs failed, see ${FAIL_LOG}:"
    cat "$FAIL_LOG"
fi
