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
training_files=(datasets/2018_coal/burner_{0..9}.csv)
validation_files=(datasets/2018_coal/burner_{10..11}.csv)
for data_file in "${training_files[@]}" "${validation_files[@]}"; do
    if [[ ! -r $data_file ]]; then
        printf 'Cannot read dataset file: %s\n' "$data_file" >&2
        exit 1
    fi
done
input_parameters=(
    Conditioner_Inlet_Temp Conditioner_Outlet_Temp Coal_Feeder_Rate Primary_Air_Flow
    Primary_Air_Split System_Secondary_Air_Flow_Total Secondary_Air_Flow Secondary_Air_Split
    Tertiary_Air_Split Total_Comb_Air_Flow Supp_Fuel_Flow Main_Flm_Int
)

exec mpirun -np 2 "$executable" \
    --training_filenames "${training_files[@]}" --validation_filenames "${validation_files[@]}" \
    --input_parameter_names "${input_parameters[@]}" --output_parameter_names Main_Flm_Int --time_offset 1 \
    --number_islands 10 --island_size 10 --max_genomes 2000 --bp_iterations 5 --num_mutations 2 \
    --weight_update adagrad --eps 0.000001 --beta1 0.99 --sequence_length 50 \
    --possible_node_types simple UGRNN MGU GRU delta LSTM --save_genome_option all_best_genomes \
    --output_directory "$output_directory" --std_message_level INFO --file_message_level INFO \
    --pi_train_host 127.0.0.1 --pi_train_port "$port"
