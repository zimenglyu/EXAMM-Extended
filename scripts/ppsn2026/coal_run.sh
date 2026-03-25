#!/bin/bash
# PPSN 2026: Coal SHY-EXAMM experiments
# Usage: bash coal_run.sh <condition> <seed>
# Example: bash coal_run.sh shy_s95_k200 3
# Designed for SLURM but can also be run directly.

CONDITION=${1:-baseline}
SEED=${2:-0}
NP=${3:-20}   # number of MPI processes (adjust for cluster)

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_DIR="$REPO_ROOT/datasets/2018_coal"
BUILD_DIR="$REPO_ROOT/build"
OUTPUT_BASE="$REPO_ROOT/results/ppsn2026/coal"

INPUT_PARAMETERS="Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int"
OUTPUT_PARAMETERS="Main_Flm_Int"

# Homeostasis settings per condition
case $CONDITION in
    baseline)
        HOM_ARGS=""
        ;;
    shy_s99_k200)
        HOM_ARGS="--homeostasis_interval 200 --homeostasis_factor 0.99"
        ;;
    shy_s95_k200)
        HOM_ARGS="--homeostasis_interval 200 --homeostasis_factor 0.95"
        ;;
    shy_s90_k200)
        HOM_ARGS="--homeostasis_interval 200 --homeostasis_factor 0.90"
        ;;
    shy_s95_k100)
        HOM_ARGS="--homeostasis_interval 100 --homeostasis_factor 0.95"
        ;;
    shy_s95_k500)
        HOM_ARGS="--homeostasis_interval 500 --homeostasis_factor 0.95"
        ;;
    shy_adaptive_k200)
        HOM_ARGS="--homeostasis_interval 200 --homeostasis_adaptive_target 0.5"
        ;;
    *)
        echo "Unknown condition: $CONDITION"
        echo "Valid: baseline shy_s99_k200 shy_s95_k200 shy_s90_k200 shy_s95_k100 shy_s95_k500 shy_adaptive_k200"
        exit 1
        ;;
esac

EXP_NAME="coal_${CONDITION}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_BASE/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Running: $EXP_NAME"
echo "Output:  $OUTPUT_DIR"
echo "HOM_ARGS: $HOM_ARGS"

cd "$BUILD_DIR"

mpirun -np $NP ./mpi/examm_mpi \
    --training_filenames "$DATASET_DIR/burner_[0-9].csv" \
    --validation_filenames "$DATASET_DIR/burner_1[0-1].csv" \
    --time_offset 1 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 10 \
    --island_size 10 \
    --max_genomes 5000 \
    --bp_iterations 5 \
    --num_mutations 2 \
    --weight_update adagrad \
    --eps 0.000001 \
    --beta1 0.99 \
    --sequence_length 50 \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --save_genome_option the_best \
    --output_directory "$OUTPUT_DIR" \
    --std_message_level INFO \
    --file_message_level INFO \
    --seed $SEED \
    $HOM_ARGS
