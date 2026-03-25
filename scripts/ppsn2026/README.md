# PPSN 2026 Scripts — Local Mac Debug Only

> **Anvil cluster scripts have moved.**
> All Slurm/HPC job scripts are in the `cluster_scripts` repo:
> `cluster_scripts/anvil/ppsn2026/`

## This directory

Contains only local Mac test/debug scripts for running experiments without a cluster:

- `coal_run.sh` — run a single coal dataset experiment locally
- `aviation_run.sh` — run a single aviation dataset experiment locally
- `run_all_local.sh` — run all conditions locally (slow, debug only)

## Cluster workflow

See `cluster_scripts/anvil/ppsn2026/README.md` for:
- How to submit jobs to Purdue Anvil
- How to collect and analyze results
- Job state tracking (`job_state.json`)
