#!/bin/bash
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=StackedSimple   # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running simple StackedHourglass training"

# Environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=0

# Run training with simple synthetic images and 2D poses
python train_mpi_inf_3dhp_simple.py \
    --exp pose_mpi_inf_3dhp_simple \
    --data_root data/motion3d \
    --max_iters 500

echo "Training completed"
