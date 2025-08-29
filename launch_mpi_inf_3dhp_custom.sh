#!/bin/bash
#SBATCH --partition=gpu_min8gb     # Reserved partition
#SBATCH --qos=gpu_min8gb           # QoS level. Must match the partition name. External users must add the suffix "_ext".
#SBATCH --job-name=MPI_INF_3DHP_Custom    # Job name
#SBATCH --output=slurm_%x.%j.out   # File containing STDOUT output
#SBATCH --error=slurm_%x.%j.err    # File containing STDERR output. If ommited, use STDOUT.
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Number of CPU cores
#SBATCH --mem=32G                  # Memory allocation
#SBATCH --time=48:00:00            # Maximum runtime (48 hours)

# Usage: sbatch launch_mpi_inf_3dhp_custom.sh [experiment_name] [max_iters] [batch_size]
# Example: sbatch launch_mpi_inf_3dhp_custom.sh my_experiment 1000 16

echo "=== MPI-INF-3DHP Custom Training Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Parse command line arguments
EXP_NAME=${1:-"mpi_inf_3dhp_default_$(date +%Y%m%d_%H%M%S)"}
MAX_ITERS=${2:-500}
BATCH_SIZE=${3:-16}
DATA_ROOT=${4:-"data/motion3d"}
MPI_DATASET_ROOT=${5:-"/nas-ctm01/datasets/public/mpi_inf_3dhp"}

echo "=== Training Configuration ==="
echo "Experiment name: $EXP_NAME"
echo "Max iterations: $MAX_ITERS"
echo "Batch size: $BATCH_SIZE"
echo "Data root: $DATA_ROOT"
echo "MPI dataset root: $MPI_DATASET_ROOT"
echo "==================================="

# Load required modules (uncomment and adjust as needed)
# module load cuda/11.8
# module load python/3.8

# Activate conda environment (uncomment and adjust as needed)
# source activate your_environment_name

# Check GPU
nvidia-smi

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verify data exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: Data directory $DATA_ROOT does not exist!"
    exit 1
fi

if [ ! -d "$MPI_DATASET_ROOT" ]; then
    echo "ERROR: MPI dataset directory $MPI_DATASET_ROOT does not exist!"
    exit 1
fi

# Create a custom configuration if batch size is different from default
if [ "$BATCH_SIZE" != "16" ]; then
    echo "Custom batch size detected. You may need to modify the config in task/pose_mpi_inf_3dhp_with_images.py"
    echo "Current batch size setting: $BATCH_SIZE"
fi

echo "Starting training with custom parameters..."
python train_mpi_inf_3dhp_with_images.py \
    --exp "$EXP_NAME" \
    --data_root "$DATA_ROOT" \
    --mpi_dataset_root "$MPI_DATASET_ROOT" \
    --max_iters $MAX_ITERS

TRAIN_EXIT_CODE=$?

echo "=== Training Results ==="
echo "End time: $(date)"
echo "Exit code: $TRAIN_EXIT_CODE"

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully!"
    
    # Show final results
    if [ -f "exp/$EXP_NAME/eval_log.txt" ]; then
        echo "📊 Final evaluation results:"
        tail -5 "exp/$EXP_NAME/eval_log.txt"
    fi
    
    if [ -f "exp/$EXP_NAME/checkpoint.pt" ]; then
        echo "💾 Model saved to: exp/$EXP_NAME/checkpoint.pt"
        echo "📁 Experiment directory contents:"
        ls -la "exp/$EXP_NAME/"
    fi
else
    echo "❌ Training failed with exit code: $TRAIN_EXIT_CODE"
fi

echo "=== Job Finished ==="
