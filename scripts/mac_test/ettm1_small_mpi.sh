#!/bin/sh
# Small ETTm1 experiment for Mac testing
# Uses ETTm1 dataset from datasets/benchmarks/ETT-small/
# Settings are intentionally small for fast runs on a local Mac.
#
# ETTm1 columns: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT
#   - OT = Oil Temperature (target for univariate prediction)
#   - All columns used for multivariate -> OT prediction
#
# Data is RAW (not pre-normalized), so we add --normalize min_max

cd build

INPUT_PARAMETERS="HUFL HULL MUFL MULL LUFL LULL OT"
OUTPUT_PARAMETERS="OT"

exp_name="../test_output/mac_test_ettm1"
mkdir -p $exp_name

echo "Running EXAMM on ETTm1 dataset (small Mac test)"
echo "Output -> $exp_name"
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
    --training_filenames ../datasets/benchmarks/ETT-small/ETTm1_train_raw.csv \
    --validation_filenames ../datasets/benchmarks/ETT-small/ETTm1_val_raw.csv \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --time_offset 1 \
    --normalize min_max \
    --sequence_length 96 \
    --number_islands 4 \
    --island_size 5 \
    --max_genomes 200 \
    --bp_iterations 3 \
    --output_directory $exp_name \
    --num_mutations 2 \
    --weight_update adagrad \
    --eps 0.000001 \
    --beta1 0.99 \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --save_genome_option the_best \
    --std_message_level INFO \
    --file_message_level INFO
