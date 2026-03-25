#!/bin/bash
# PPSN 2026: Aviation (C172 NGAFID) SHY-EXAMM experiments
# Usage: bash aviation_run.sh <condition> <seed>
# NOTE: Aviation data is NOT pre-normalized — uses --normalize min_max

CONDITION=${1:-baseline}
SEED=${2:-0}
NP=${3:-20}

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DATASET_DIR="$REPO_ROOT/datasets/2019_ngafid_transfer"
BUILD_DIR="$REPO_ROOT/build"
OUTPUT_BASE="$REPO_ROOT/results/ppsn2026/aviation"

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

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
        exit 1
        ;;
esac

EXP_NAME="aviation_${CONDITION}_seed${SEED}"
OUTPUT_DIR="$OUTPUT_BASE/$EXP_NAME"
mkdir -p "$OUTPUT_DIR"

echo "Running: $EXP_NAME"
echo "Output:  $OUTPUT_DIR"

cd "$BUILD_DIR"

mpirun -np $NP ./mpi/examm_mpi \
    --training_filenames "$DATASET_DIR/c172_file_[1-8].csv" \
    --validation_filenames "$DATASET_DIR/c172_file_9.csv" "$DATASET_DIR/c172_file_10.csv" \
    --time_offset 1 \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --number_islands 10 \
    --island_size 10 \
    --max_genomes 5000 \
    --bp_iterations 5 \
    --num_mutations 2 \
    --normalize min_max \
    --possible_node_types simple UGRNN MGU GRU delta LSTM \
    --save_genome_option the_best \
    --output_directory "$OUTPUT_DIR" \
    --std_message_level INFO \
    --file_message_level INFO \
    --seed $SEED \
    $HOM_ARGS
