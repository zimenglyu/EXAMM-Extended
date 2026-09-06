#!/bin/bash -l
# NOTE the -l flag!
#

# This is an example job file for a Serial Multi-Process job.
# Note that all of the following statements below that begin
# with #SBATCH are actually commands to the SLURM scheduler.
# Please copy this file to your home directory and modify it
# to suit your needs.
# 
# If you need any help, please email rc-help@rit.edu
#

# Name of the job - You'll probably want to customize this.
#SBATCH -J 10k_10

#SBATCH -A <your project name>

# Standard out and Standard Error output files
#SBATCH -o examm_%x_%j.output
#SBATCH -e examm_%x_%j.error

#To send emails, set the adcdress below and remove one of the "#" signs.
#SBATCH --mail-user=<your email address>

# notify on state change: BEGIN, END, FAIL or ALL
#SBATCH --mail-type=ALL


#SBATCH -t 1:0:0
#SBATCH --nodes=1        # Total # of nodes 
#SBATCH --ntasks=4     # Total # of MPI tasks


## Put the job in the "work" partition and request FOUR cores for one task
## "work" is the default partition so it can be omitted without issue.

# # Please not that each node on the cluster is 36 cores
## SBATCH -p debug -n 4


## Job memory requirements in MB
##SBATCH --mem-per-cpu=5000


EXAMM="/home/x-zlyu2/code/EXAMM-Extended"

module --force purge
module load gcc
module load cmake 
module load openmpi
module load libtiff

MAX_GENOME=4000
NUM_ISLAND=10
DATASET="c172"

    for folder in 0 
    do
        # REPOPULATION_METHOD = "bestGenome"
        exp_name="$EXAMM/results/test_new/$DATASET/max_genome_$MAX_GENOME/island_$NUM_ISLAND/$folder"
        mkdir -p $exp_name
        echo "\tIteration: "$exp_name
        echo "\t###-------------------###"

        mpirun -np $SLURM_NTASKS $EXAMM/build/mpi/examm_mpi \
        --training_filenames $EXAMM/datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
        --test_filenames $EXAMM/datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
        --time_offset 1 \
        --input_parameter_names "AltAGL" "AltB" "AltGPS" "AltMSL" "BaroA" "E1_CHT1" "E1_CHT2" "E1_CHT3" "E1_CHT4" "E1_EGT1" "E1_EGT2" "E1_EGT3" "E1_EGT4" "E1_FFlow" "E1_OilP" "E1_OilT" "E1_RPM" "FQtyL" "FQtyR" "GndSpd" "IAS" "LatAc" "NormAc" "OAT" "Pitch" "Roll" "TAS" "VSpd" "VSpdG" "WndDr" "WndSpd"  \
        --output_parameter_names "Pitch" \
        --number_islands $NUM_ISLAND \
        --island_size 10 \
        --max_genomes $MAX_GENOME \
        --bp_iterations 10 \
        --possible_node_types simple UGRNN MGU GRU delta LSTM \
        --normalize min_max \
        --weight_update rmsprop \
        --std_message_level INFO \
        --file_message_level INFO \
        --output_directory $exp_name

    done

