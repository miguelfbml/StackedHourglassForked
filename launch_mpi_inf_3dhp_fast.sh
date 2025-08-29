#!/bin/bash
#
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=MPI_INF_3DHP_Fast    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.

echo "Fast training StackedHourglass with MPI-INF-3DHP (17 keypoints)"

# Fast training with fewer iterations for testing
python train_mpi_inf_3dhp_with_images.py \
    --exp mpi_inf_3dhp_fast_test \
    --data_root data/motion3d \
    --mpi_dataset_root /nas-ctm01/datasets/public/mpi_inf_3dhp \
    --max_iters 50

echo "Fast training completed"
