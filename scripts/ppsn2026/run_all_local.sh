#!/bin/bash
# Generate ALL cluster job commands (do not run — copy-paste to cluster)
# Usage: bash run_all_local.sh > cluster_jobs.txt
# Then inspect cluster_jobs.txt and adapt for SLURM/PBS as needed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDITIONS="baseline shy_s99_k200 shy_s95_k200 shy_s90_k200 shy_s95_k100 shy_s95_k500 shy_adaptive_k200"
DATASETS="coal aviation"
SEEDS="0 1 2 3 4 5 6 7 8 9"

for DATASET in $DATASETS; do
    for CONDITION in $CONDITIONS; do
        for SEED in $SEEDS; do
            echo "bash ${SCRIPT_DIR}/${DATASET}_run.sh ${CONDITION} ${SEED}"
        done
    done
done
