#!/bin/sh
# Run this on the Raspberry Pi FIRST. It listens for genomes streamed from
# examm_mpi (started with --send_to_pi) and evaluates each one on the
# test files below. Results are appended to $out_dir/pi_evaluations.csv and
# each genome's predictions go to $out_dir/genome_<id>/.
# --ina219 enables power/energy measurement (needs the INA219 on /dev/i2c-1;
# add --ina219_device <path> to change). Drop the flag if no sensor is wired.

cd build

out_dir="../test_output/pi_server"
mkdir -p $out_dir

./rnn_examples/pi_genome_server \
--port 5555 \
--testing_filenames ../datasets/2019_ngafid_transfer/c172_file_11.csv ../datasets/2019_ngafid_transfer/c172_file_12.csv \
--time_offset 1 \
--output_directory $out_dir \
--save_genomes \
--ina219 \
--std_message_level INFO \
--file_message_level INFO
