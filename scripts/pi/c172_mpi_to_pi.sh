#!/bin/sh
# Same as scripts/base_run/c172_mpi.sh, plus --send_to_pi so every new global
# best genome is also streamed to a pi_genome_server for evaluation.
# Start scripts/pi/pi_genome_server.sh on the pi before running this.
#
# The pi address defaults to DEFAULT_PI_HOST/DEFAULT_PI_PORT in mpi/examm_mpi.cxx
# (192.168.0.70:5555); add --pi_host <ip> / --pi_port <port> below to override.

cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

exp_name="../test_output/c172_mpi_to_pi"
mkdir -p $exp_name
echo "Running EXAMM MPI on c172 dataset, streaming best genomes to the pi, results will be saved to: "$exp_name
echo "###-------------------###"

mpirun -np 4 ./mpi/examm_mpi \
--training_filenames ../datasets/2019_ngafid_transfer/c172_file_[1-8].csv \
--validation_filenames ../datasets/2019_ngafid_transfer/c172_file_9.csv ../datasets/2019_ngafid_transfer/c172_file_10.csv \
--time_offset 1 \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--number_islands 10 \
--island_size 10 \
--max_genomes 2000 \
--bp_iterations 5 \
--num_mutations 2 \
--normalize min_max \
--output_directory $exp_name \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--send_to_pi \
--std_message_level INFO \
--file_message_level INFO
