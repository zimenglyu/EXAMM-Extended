#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
    printf 'Usage: %s <examm_mpi_executable> <tunnel_port> <new_output_directory>\n' "$0" >&2
    exit 2
fi
executable=$1
port=$2
output_directory=$3
if [[ $output_directory != /* ]]; then output_directory=$PWD/$output_directory; fi
if [[ ! -x $executable ]]; then
    printf 'EXAMM executable is not available: %s\n' "$executable" >&2
    exit 1
fi
if [[ ! $port =~ ^[1-9][0-9]{0,4}$ ]] || (( port > 65535 )); then
    printf 'Port must be a number between 1 and 65535.\n' >&2
    exit 2
fi
if [[ -z $3 || -e $output_directory || -L $output_directory ]]; then
    printf 'Choose a new output directory: %s already exists.\n' "$output_directory" >&2
    exit 1
fi
executable=$(cd "$(dirname "$executable")" && pwd)/$(basename "$executable")
project_directory=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_directory"

exec mpirun -np 2 "$executable" \
    --training_filenames pi_training/fixtures/train.csv \
    --validation_filenames pi_training/fixtures/validation.csv \
    --input_parameter_names x --output_parameter_names x --time_offset 1 \
    --number_islands 1 --island_size 2 --max_genomes 6 --bp_iterations 2 \
    --num_mutations 2 --weight_update adagrad --weight_decay 0.015 --eps 0.000001 \
    --possible_node_types simple --save_genome_option the_best \
    --output_directory "$output_directory" --std_message_level INFO --file_message_level INFO \
    --pi_train_host 127.0.0.1 --pi_train_port "$port"
