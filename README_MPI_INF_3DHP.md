# StackedHourglass for MPI-INF-3DHP Dataset (17 Keypoints)

This repository contains a modified version of the StackedHourglass model adapted to work with 17 keypoints for the MPI-INF-3DHP dataset.

## Changes Made

### 1. Model Configuration
- Modified `task/pose_mpi_inf_3dhp_with_images.py` to support 17 keypoints instead of 16
- Updated `oup_dim` and `num_parts` from 16 to 17
- Adjusted batch size and training parameters for MPI-INF-3DHP

### 2. Data Provider
- Created `data/MPI_INF_3DHP/dp_with_images.py` for loading MPI-INF-3DHP data
- Supports both training and validation datasets
- Handles 17-keypoint pose format
- Includes data augmentation and proper coordinate normalization

### 3. Training Script
- Created `train_mpi_inf_3dhp.py` specifically for MPI-INF-3DHP training
- Supports resuming from checkpoints
- Includes validation during training

## Dataset Structure

The MPI-INF-3DHP dataset should be organized as follows:

```
data/MPI_INF_3DHP/motion3d/
├── data_train_3dhp.npz    # Training poses (3D and 2D)
└── data_test_3dhp.npz     # Test poses (3D and 2D)

/nas-ctm01/datasets/public/mpi_inf_3dhp/
├── S1/Seq1/imageSequence/video_0/    # Training images
├── S1/Seq1/imageSequence/video_1/
├── ...
└── mpi_inf_3dhp_test_set/
    ├── TS1/imageSequence/            # Test images
    ├── TS2/imageSequence/
    └── ...
```

## 17 Keypoint Format

The model uses the following 17-keypoint format (MPI-INF-3DHP standard):

```
0: Spine (H36M root)      9: Right knee
1: Spine4                10: Right ankle  
2: Right shoulder        11: Left hip
3: Right elbow           12: Left knee
4: Right wrist           13: Left ankle
5: Left shoulder         14: Hip (MPI root)
6: Left elbow            15: Spine1
7: Left wrist            16: Head
8: Right hip
```

## Usage

### 1. Test Data Provider
First, test that the data provider works correctly:

```bash
python test_mpi_inf_3dhp_data.py
```

### 2. Training
Start training with the SLURM script:

```bash
sbatch launch_mpi_inf_3dhp_with_images.sh
```

Or run directly:

```bash
python train_mpi_inf_3dhp.py \
    --exp mpi_inf_3dhp_stacked_hourglass_17kpts \
    --max_iters 500 \
    --data_root data/MPI_INF_3DHP/motion3d \
    --mpi_dataset_root /nas-ctm01/datasets/public/mpi_inf_3dhp
```

### 3. Resume Training
To resume from a checkpoint:

```bash
python train_mpi_inf_3dhp.py \
    --continue_exp mpi_inf_3dhp_stacked_hourglass_17kpts \
    --max_iters 500
```

## Key Features

### Data Augmentation
- Random horizontal flipping with proper keypoint swapping
- Color augmentation (hue, saturation, brightness, contrast)
- Coordinate validation and clipping

### Robust Data Loading
- Fallback to synthetic images when real images are missing
- Multiple attempts to load valid samples
- Custom collate function for proper batch handling

### Training Features
- Automatic checkpoint saving every 10,000 iterations
- Validation during training
- Learning rate decay
- GPU support with proper device handling

## Model Architecture

The model uses the standard StackedHourglass architecture with:
- 8 stacked hourglasses
- 256 input/intermediate dimensions
- 17 output heatmaps (one per keypoint)
- 256x256 input resolution
- 64x64 output heatmap resolution

## Files Created/Modified

1. `task/pose_mpi_inf_3dhp_with_images.py` - Task configuration for 17 keypoints
2. `data/MPI_INF_3DHP/dp_with_images.py` - Data provider (fixed errors)
3. `train_mpi_inf_3dhp.py` - Training script
4. `test_mpi_inf_3dhp_data.py` - Data provider test script
5. `launch_mpi_inf_3dhp_with_images.sh` - SLURM launch script

## Troubleshooting

### Common Issues

1. **Missing data files**: Ensure `data_train_3dhp.npz` and `data_test_3dhp.npz` exist in the motion3d folder
2. **Missing images**: The data provider will create synthetic images if real ones are missing
3. **Memory issues**: Reduce batch size in the configuration if you encounter OOM errors
4. **CUDA errors**: Ensure proper GPU setup and CUDA compatibility

### Error Fixes Made

1. **Data format consistency**: Fixed inconsistent return types from `__getitem__`
2. **Coordinate transformation**: Corrected normalization and pixel coordinate conversion
3. **Keypoint swapping**: Fixed left-right keypoint swapping during augmentation
4. **Batch handling**: Added custom collate function for proper tensor creation
5. **Error handling**: Added robust error handling and fallbacks

## Expected Training Time

On a single GPU:
- ~2-3 days for 500k iterations
- Validation every 100 iterations
- Checkpoint saves every 10k iterations

The model should converge and produce reasonable heatmaps for 2D pose estimation on the MPI-INF-3DHP dataset.
