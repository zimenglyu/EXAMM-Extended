#!/bin/bash -l
#SBATCH -J examm_core7_ret
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH --ntasks=128
#SBATCH -t 5:00:00
#SBATCH -o /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.output
#SBATCH -e /anvil/scratch/x-dlopez6/experiments/examm_results/examm_%x_%A_%a.error
#SBATCH --mail-user=rsch_lopedieg@kean.edu
#SBATCH --mail-type=ALL
#
# RET TARGET version of core7_examm_yearly.sh: predicts next-day RET (the raw daily
# return as a robust z-score; raw = RET * IQR + median, stats in panel_meta.json)
# instead of RET_CS (the cross-sectional rank score). Everything else is identical:
# same data (data_prep/split_core7_yearly.py folds: train <= Y-2, val = Y-1, test = Y
# + 40-day warm-up), same inputs, avg_std_dev, repeats 15..24.
# Results go to test_output/core7_examm_ret_<YEAR>/ so they never mix with the RET_CS
# runs in test_output/core7_examm_<YEAR>/. One panel set per array task:
#   sbatch --array=1-4 --export=ALL,YEAR=2022 scripts/base_run/core7_examm_yearly_ret.sh
#   sbatch --array=1-4 --export=ALL,YEAR=2023 scripts/base_run/core7_examm_yearly_ret.sh
#   sbatch --array=1-4 --export=ALL,YEAR=2024 scripts/base_run/core7_examm_yearly_ret.sh
# A folder whose run already finished is skipped, so resubmitting only redoes
# missing or incomplete repeats.

EXAMM="/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended"
YEAR="${YEAR:?set YEAR, e.g. sbatch --export=ALL,YEAR=2022 ...}"
SET="set${SLURM_ARRAY_TASK_ID:-1}"
FOLDERS="15 16 17 18 19 20 21 22 23 24"
MAX_GENOME=10000
DATAPATH="$EXAMM/datasets/panels_core7_examm_$YEAR/$SET"
if ! ls $DATAPATH/*_train.csv >/dev/null 2>&1 || ! ls $DATAPATH/*_val.csv >/dev/null 2>&1; then
    echo "no train/val files in $DATAPATH -- run: python3 data_prep/split_core7_yearly.py --years $YEAR" >&2
    exit 1
fi

INPUT_PARAMETERS="RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21"
OUTPUT_PARAMETERS="RET"            # RET target (RET_CS runs: core7_examm_yearly.sh)

failed=""
for FOLDER in $FOLDERS; do
    exp_name="$EXAMM/test_output/core7_examm_ret_$YEAR/$SET/$FOLDER"

    # skip a repeat that already finished (fitness log reached max genomes)
    if [ -f "$exp_name/fitness_log.csv" ] && \
       [ "$(tail -1 "$exp_name/fitness_log.csv" | cut -d, -f1)" -ge $MAX_GENOME ] 2>/dev/null; then
        echo "$SET folder $FOLDER already finished, skipping"
        continue
    fi

    # retry up to 3 times if examm crashes (segfault in RNN_Genome::add_edge, or
    # "mu or sigma exceeded possible bounds"); each attempt starts from an empty folder
    for attempt in 1 2 3; do
        rm -rf "$exp_name"
        mkdir -p "$exp_name"
        echo "Running EXAMM on $SET (folder $FOLDER, attempt $attempt), results will be saved to: "$exp_name
        echo "###-------------------###"

        # timeout: safety net in case a crashed run hangs (normal runs take ~12-20 min)
        time timeout 45m mpirun -np ${SLURM_NTASKS:-4} $EXAMM/build/mpi/examm_mpi \
        --training_filenames $DATAPATH/*_train.csv \
        --validation_filenames $DATAPATH/*_val.csv \
        --time_offset 1 \
        --input_parameter_names $INPUT_PARAMETERS \
        --output_parameter_names $OUTPUT_PARAMETERS \
        --normalize avg_std_dev \
        --number_islands 10 \
        --island_size 10 \
        --max_genomes $MAX_GENOME \
        --bp_iterations 20 \
        --num_mutations 2 \
        --learning_rate 0.001 \
        --extinction_event_generation_number 500 \
        --repeat_extinction \
        --island_ranking_method "EraseWorst" \
        --repopulation_method "bestGenome" \
        --islands_to_exterminate 1 \
        --save_genome_option the_best \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --output_directory $exp_name \
        --std_message_level ERROR \
        --file_message_level ERROR && break

        echo "$SET folder $FOLDER: attempt $attempt failed" >&2
        if [ $attempt -eq 3 ]; then
            failed="$failed $FOLDER"
        fi
    done
done

if [ -n "$failed" ]; then
    echo "$SET folders that failed all 3 attempts:$failed" >&2
    exit 1
fi
echo "$SET: all folders $FOLDERS finished"
