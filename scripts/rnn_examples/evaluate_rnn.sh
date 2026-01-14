#!/bin/sh

cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

exp_name="../test_output/c172_mpi"

genome_name="../test_output/c172_mpi/rnn_genome_18.bin"

out_dir="../test_output/c172_mpi/evaluation_results"
mkdir -p $out_dir
echo "Evaluating RNN on c172 dataset, results will be saved to: "$out


./rnn_examples/evaluate_rnn \
--testing_filenames ../datasets/2019_ngafid_transfer/c172_file_12.csv \
--time_offset 1 \
--genome_file $genome_name \
--input_parameter_names $INPUT_PARAMETERS \
--output_parameter_names $OUTPUT_PARAMETERS \
--genome_filename $genome_name \
--output_directory $out_dir \
--std_message_level INFO \
--file_message_level INFO
