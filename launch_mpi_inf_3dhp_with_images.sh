#!/bin/bash
#SBATCH --job-name=StackedTrain_MPI_Images
#SBATCH --output=slurm_StackedTrain_MPI_Images.%j.out
#SBATCH --error=slurm_StackedTrain_MPI_Images.%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=reserved
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1

echo "Running job in reserved partition"

# Activate conda environment
conda activate tcpformer_env

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run training with real images
python train_mpi_inf_3dhp_with_images.py \
    --exp pose_mpi_inf_3dhp_images \
    --data_root data/motion3d \
    --mpi_dataset_root /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --max_iters 500

echo "Training completed"
