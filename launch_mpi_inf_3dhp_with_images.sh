#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=mpi_3dhp_stacked   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
#SBATCH --time=48:00:00            # Maximum time for the job (48 hours)
#SBATCH --mem=32G                  # Memory allocation
#SBATCH --gres=gpu:1               # Request 1 GPU

echo "Running StackedHourglass training on MPI-INF-3DHP dataset"
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0

# Commands / scripts to run
python3 train_mpi_inf_3dhp.py \
    --exp mpi_inf_3dhp_stacked_hourglass_17kpts \
    --max_iters 500 \
    --data_root data/MPI_INF_3DHP/motion3d \
    --mpi_dataset_root /nas-ctm01/datasets/public/mpi_inf_3dhp

echo "Training completed at $(date)"
