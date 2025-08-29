#!/bin/bash
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=StackedTrain    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Running job in reserved partition"

# Environment variables to fix multiprocessing issues
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_LAUNCH_BLOCKING=0

# Run training with real images
python train_mpi_inf_3dhp_with_images.py \
    --exp pose_mpi_inf_3dhp_images \
    --data_root data/motion3d \
    --mpi_dataset_root /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --max_iters 500

echo "Training completed"
