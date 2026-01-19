#!/bin/sh

cd build

INPUT_PARAMETERS="AltAGL AltB AltGPS AltMSL BaroA E1_CHT1 E1_CHT2 E1_CHT3 E1_CHT4 E1_EGT1 E1_EGT2 E1_EGT3 E1_EGT4 E1_FFlow E1_OilP E1_OilT E1_RPM FQtyL FQtyR GndSpd IAS LatAc NormAc OAT Pitch Roll TAS VSpd VSpdG WndDr WndSpd"
OUTPUT_PARAMETERS="Pitch"

exp_name="../test_output/c172_mpi"
for i in 0 1 2 3 4 5 6 7 8 9 
do

    genome_name="/Users/zimenglyu/Downloads/demo_new/global_best_genome_${i}.bin"

    out_dir="../test_output/c172_mpi/evaluation_results_new/$i"
    mkdir -p $out_dir
    echo "Evaluating RNN on c172 dataset, results will be saved to: "$out_dir

    ./rnn_examples/evaluate_rnn \
    --testing_filenames ../datasets/2018_coal/burner_11.csv \
    --time_offset 1 \
    --genome_file $genome_name \
    --input_parameter_names $INPUT_PARAMETERS \
    --output_parameter_names $OUTPUT_PARAMETERS \
    --genome_filename $genome_name \
    --output_directory $out_dir \
    --std_message_level INFO \
    --file_message_level INFO
done