# MPI-INF-3DHP Dataset Setup for StackedHourglass

This directory contains the data provider for training StackedHourglass with the MPI-INF-3DHP dataset (17 keypoints).

## Prerequisites

1. You need to have the MPI-INF-3DHP dataset preprocessed and saved as:
   - `data/motion3d/data_train_3dhp.npz`
   - `data/motion3d/data_test_3dhp.npz`

2. These files should be in the same format as used by TCPFormerForked.

## Usage

To train the StackedHourglass model with MPI-INF-3DHP dataset:

```bash
python train_mpi_inf_3dhp.py --data_root path/to/motion3d --exp mpi_inf_3dhp_experiment
```

## Key Changes from Original MPII Setup

1. **Number of keypoints**: Changed from 16 (MPII) to 17 (MPI-INF-3DHP)
2. **Data loading**: Uses the same data format as TCPFormerForked
3. **Keypoint mapping**: Uses MPI-INF-3DHP joint ordering
4. **Augmentation**: Adapted flip augmentation for 17-joint skeleton

## Joint Ordering (MPI-INF-3DHP)

The 17 joints are ordered as:
0. Head
1. Spine3 (upper thorax)
2. Left shoulder
3. Left elbow  
4. Left wrist
5. Right shoulder
6. Right elbow
7. Right wrist
8. Left hip
9. Left knee
10. Left ankle
11. Right hip
12. Right knee
13. Right ankle
14. Pelvis (root joint)
15. Spine2 (thorax)
16. Neck

Left/right joint pairs for flip augmentation:
- Left: [5, 6, 7, 11, 12, 13] (shoulders, elbows, wrists, hips, knees, ankles)
- Right: [2, 3, 4, 8, 9, 10]
