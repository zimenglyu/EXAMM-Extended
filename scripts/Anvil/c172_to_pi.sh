#!/bin/bash -l
# EXAMM MPI run on Anvil that streams every new global best genome to a
# Raspberry Pi for evaluation (see scripts/pi/).
#
# The pi is not reachable from Anvil directly, so it is reached through an
# ssh tunnel via a login node. Before submitting this job, ON THE PI:
#
#   terminal 1:  ./scripts/pi/pi_genome_server.sh
#   terminal 2:  ssh -o ServerAliveInterval=60 -R 5555:localhost:5555 <user>@login03.anvil.rcac.purdue.edu
#
# then submit this job FROM THE SHELL OPENED IN TERMINAL 2 and keep it open.
# LOGIN_NODE below must match the login node used in terminal 2.
#
# One-time: compute nodes must be able to ssh to the login node without a
# password. On Anvil run:  ssh-keygen -t ed25519 -N ""  and
# cat ~/.ssh/id_ed25519.pub >> ~/.ssh/authorized_keys

#SBATCH -J c172_to_pi
#SBATCH -A <your project name>
#SBATCH -o examm_%x_%j.output
#SBATCH -e examm_%x_%j.error
#SBATCH --mail-user=<your email address>
#SBATCH --mail-type=ALL
#SBATCH -t 4:0:0
#SBATCH --nodes=1
#SBATCH --ntasks=16

EXAMM="/home/x-zlyu2/code/EXAMM-Extended"
LOGIN_NODE="login03"
PI_PORT=5555

module --force purge
module load gcc
module load cmake
module load openmpi
module load libtiff

# forward this compute node's $PI_PORT to the login node, where the pi's
# reverse tunnel is listening
ssh -N -o ExitOnForwardFailure=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new -L $PI_PORT:localhost:$PI_PORT $LOGIN_NODE &
TUNNEL_PID=$!
sleep 3
if ! kill -0 $TUNNEL_PID 2>/dev/null; then
    echo "could not open tunnel to $LOGIN_NODE, genomes will not reach the pi"
fi

MAX_GENOME=4000
NUM_ISLAND=10
exp_name="$EXAMM/results/c172_to_pi/max_genome_$MAX_GENOME/island_$NUM_ISLAND/$SLURM_JOB_ID"
mkdir -p $exp_name
echo "results will be saved to: $exp_name"

mpirun -np $SLURM_NTASKS $EXAMM/build/mpi/examm_mpi \
--training_filenames $EXAMM/datasets/2019_ngafid_transfer/c172_file_[1-9].csv \
--validation_filenames $EXAMM/datasets/2019_ngafid_transfer/c172_file_1[0-2].csv \
--time_offset 1 \
--input_parameter_names "AltAGL" "AltB" "AltGPS" "AltMSL" "BaroA" "E1_CHT1" "E1_CHT2" "E1_CHT3" "E1_CHT4" "E1_EGT1" "E1_EGT2" "E1_EGT3" "E1_EGT4" "E1_FFlow" "E1_OilP" "E1_OilT" "E1_RPM" "FQtyL" "FQtyR" "GndSpd" "IAS" "LatAc" "NormAc" "OAT" "Pitch" "Roll" "TAS" "VSpd" "VSpdG" "WndDr" "WndSpd" \
--output_parameter_names "Pitch" \
--number_islands $NUM_ISLAND \
--island_size 10 \
--max_genomes $MAX_GENOME \
--bp_iterations 10 \
--possible_node_types simple UGRNN MGU GRU delta LSTM \
--normalize min_max \
--weight_update rmsprop \
--send_to_pi \
--pi_host 127.0.0.1 \
--pi_port $PI_PORT \
--std_message_level INFO \
--file_message_level INFO \
--output_directory $exp_name

kill $TUNNEL_PID 2>/dev/null
