#!/bin/bash
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=MPI_INF_3DHP_Train    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --mem=32G                  # Memory allocation
#SBATCH --time=48:00:00            # Maximum runtime (48 hours)

echo "=== MPI-INF-3DHP Training Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"

# Load required modules (adjust as needed for your cluster)
# module load cuda/11.8
# module load python/3.8

# Activate your conda environment (adjust as needed)
# source activate your_environment_name

# Check GPU availability
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Configuration
EXP_NAME="mpi_inf_3dhp_experiment_$(date +%Y%m%d_%H%M%S)"
DATA_ROOT="data/motion3d"
MPI_DATASET_ROOT="/nas-ctm01/datasets/public/mpi_inf_3dhp"
MAX_ITERS=500

echo "=== Training Configuration ==="
echo "Experiment name: $EXP_NAME"
echo "Data root: $DATA_ROOT"
echo "MPI dataset root: $MPI_DATASET_ROOT"
echo "Max iterations: $MAX_ITERS"
echo "==================================="

# Check if data directories exist
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data directory $DATA_ROOT does not exist!"
    exit 1
fi

if [ ! -d "$MPI_DATASET_ROOT" ]; then
    echo "ERROR: MPI dataset directory $MPI_DATASET_ROOT does not exist!"
    exit 1
fi

echo "Data directories verified successfully."

# Run training
echo "Starting training..."
python train_mpi_inf_3dhp_with_images.py \
    --exp "$EXP_NAME" \
    --data_root "$DATA_ROOT" \
    --mpi_dataset_root "$MPI_DATASET_ROOT" \
    --max_iters $MAX_ITERS

TRAIN_EXIT_CODE=$?

echo "=== Training Completed ==="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "Training completed successfully!"
    
    # Optional: Copy results to a backup location
    # cp -r exp/$EXP_NAME /path/to/backup/location/
    
    # Display final model info
    if [ -f "exp/$EXP_NAME/checkpoint.pt" ]; then
        echo "Final checkpoint saved at: exp/$EXP_NAME/checkpoint.pt"
        ls -la "exp/$EXP_NAME/"
    fi
    
    # Display training log summary
    if [ -f "exp/$EXP_NAME/eval_log.txt" ]; then
        echo "=== Training Summary ==="
        tail -10 "exp/$EXP_NAME/eval_log.txt"
    fi
else
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "Check the error log for details."
fi

echo "=== Job Finished ==="
