#!/bin/bash
#
#SBATCH --partition=gpu_min24gb     # Reserved partition
#SBATCH --qos=gpu_min24gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=shorttest_mpi   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"

# Commands / scripts to run - short test with only 200 iterations
python3 train_mpi_inf_3dhp.py -e mpi_inf_3dhp_training -m 200 --use-wandb --wandb-name "MPI-INF-3DHP-Experiment-1"
