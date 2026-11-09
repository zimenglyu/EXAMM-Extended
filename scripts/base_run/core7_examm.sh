#!/bin/bash -l
#SBATCH -J examm_core7
#SBATCH -A cis251123
#SBATCH -p wholenode
#SBATCH --ntasks=128
#SBATCH -t 2:00:00
#SBATCH -o examm_%x_%A_%a.output
#SBATCH -e examm_%x_%A_%a.error
#SBATCH --mail-user=rsch_lopedieg@kean.edu
#
# EXAMM on the core7 panels (split by scripts/data_prep/split_core7_for_examm.py).
# One panel set per array task, one repeat per FOLDER:
#   sbatch --array=1-4 --export=ALL,FOLDER=15 scripts/base_run/core7_examm.sh

EXAMM="/anvil/scratch/x-dlopez6/experiments/EXAMM-Extended"
SET="set${SLURM_ARRAY_TASK_ID:-1}"
FOLDER="${FOLDER:-15}"
DATAPATH="$EXAMM/datasets/panels_core7_examm/$SET"

INPUT_PARAMETERS="RET RET_CS_IN BA_SPREAD ILLIQUIDITY REV21_1 TURN_RATIO VOL21"
OUTPUT_PARAMETERS="RET_CS"

exp_name="$EXAMM/test_output/core7_examm/$SET/$FOLDER"
mkdir -p $exp_name
echo "Running EXAMM on $SET (folder $FOLDER), results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np ${SLURM_NTASKS:-4} $EXAMM/build/mpi/examm_mpi \
--training_filenames $DATAPATH/*_train.csv \
--validation_filenames $DATAPATH/*_val.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--normalize avg_std_dev \
--number_islands 10 \
--island_size 10 \
--max_genomes 10000 \
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
--file_message_level ERROR
