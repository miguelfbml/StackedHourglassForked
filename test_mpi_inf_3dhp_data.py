#!/usr/bin/env python3
"""
Test script to verify MPI-INF-3DHP data provider works correctly
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_data_provider():
    """Test the MPI-INF-3DHP data provider"""
    
    # Import the task configuration
    import importlib
    task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
    config = task.__config__.copy()
    
    # Override some settings for testing
    config['train']['batchsize'] = 2
    config['train']['num_workers'] = 1
    config['data_root'] = 'data/MPI_INF_3DHP/motion3d'
    config['mpi_dataset_root'] = '/nas-ctm01/datasets/public/mpi_inf_3dhp'
    
    print("Testing MPI-INF-3DHP data provider...")
    print(f"Configuration:")
    print(f"  - Output dimensions: {config['inference']['oup_dim']}")
    print(f"  - Number of parts: {config['inference']['num_parts']}")
    print(f"  - Input resolution: {config['train']['input_res']}")
    print(f"  - Output resolution: {config['train']['output_res']}")
    print(f"  - Batch size: {config['train']['batchsize']}")
    
    # Import data provider
    try:
        data_func = importlib.import_module(config['data_provider'])
        print("✓ Data provider imported successfully")
    except Exception as e:
        print(f"✗ Failed to import data provider: {e}")
        return False
    
    # Test training dataset
    try:
        print("\nTesting training dataset...")
        train_dataset = data_func.Dataset(config, train=True, 
                                        data_root=config['data_root'],
                                        mpi_dataset_root=config['mpi_dataset_root'])
        print(f"✓ Training dataset created with {len(train_dataset)} samples")
        
        # Test a few samples
        for i in range(min(3, len(train_dataset))):
            sample = train_dataset[i]
            print(f"Sample {i}:")
            if isinstance(sample, dict):
                img = sample['imgs']
                heatmaps = sample['heatmaps']
                print(f"  - Image shape: {img.shape}, dtype: {img.dtype}")
                print(f"  - Heatmaps shape: {heatmaps.shape}, dtype: {heatmaps.dtype}")
                print(f"  - Image range: [{img.min():.3f}, {img.max():.3f}]")
                print(f"  - Heatmaps range: [{heatmaps.min():.3f}, {heatmaps.max():.3f}]")
                print(f"  - Active heatmaps: {(heatmaps.max(axis=(1,2)) > 0.1).sum()}/17")
            else:
                print(f"  - Unexpected format: {type(sample)}")
        
    except Exception as e:
        print(f"✗ Training dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test validation dataset
    try:
        print("\nTesting validation dataset...")
        val_dataset = data_func.Dataset(config, train=False,
                                      data_root=config['data_root'], 
                                      mpi_dataset_root=config['mpi_dataset_root'])
        print(f"✓ Validation dataset created with {len(val_dataset)} samples")
        
        # Test a sample
        sample = val_dataset[0]
        if isinstance(sample, dict):
            img = sample['imgs']
            heatmaps = sample['heatmaps']
            print(f"  - Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"  - Heatmaps shape: {heatmaps.shape}, dtype: {heatmaps.dtype}")
        
    except Exception as e:
        print(f"✗ Validation dataset failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test data loader
    try:
        print("\nTesting data loader...")
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config['train']['batchsize'],
            shuffle=True,
            num_workers=0,  # Use 0 for testing
            collate_fn=data_func.custom_collate_fn
        )
        
        batch = next(iter(train_loader))
        print(f"✓ Data loader working")
        if isinstance(batch, dict):
            print(f"  - Batch images shape: {batch['imgs'].shape}")
            print(f"  - Batch heatmaps shape: {batch['heatmaps'].shape}")
        else:
            print(f"  - Unexpected batch format: {type(batch)}")
            
    except Exception as e:
        print(f"✗ Data loader failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed! Data provider is working correctly.")
    return True

def visualize_sample():
    """Visualize a sample from the dataset"""
    import importlib
    task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
    config = task.__config__.copy()
    
    config['data_root'] = 'data/MPI_INF_3DHP/motion3d'
    config['mpi_dataset_root'] = '/nas-ctm01/datasets/public/mpi_inf_3dhp'
    
    data_func = importlib.import_module(config['data_provider'])
    train_dataset = data_func.Dataset(config, train=True, 
                                    data_root=config['data_root'],
                                    mpi_dataset_root=config['mpi_dataset_root'])
    
    sample = train_dataset[0]
    if isinstance(sample, dict):
        img = sample['imgs']
        heatmaps = sample['heatmaps']
        
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        
        # Show image
        axes[0, 0].imshow(img)
        axes[0, 0].set_title('Input Image')
        axes[0, 0].axis('off')
        
        # Show some heatmaps
        for i in range(min(9, heatmaps.shape[0])):
            row = i // 5
            col = (i % 5)
            if row < 2 and col < 5:
                if row == 0 and col == 0:
                    continue
                ax_idx = row * 5 + col
                if ax_idx < 10:
                    ax = axes[row, col]
                    ax.imshow(heatmaps[i], cmap='hot')
                    ax.set_title(f'Joint {i}')
                    ax.axis('off')
        
        # Hide unused subplots
        for i in range(10):
            row = i // 5
            col = i % 5
            if row == 1 and col >= 4:
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig('sample_visualization.png')
        print("Sample visualization saved as 'sample_visualization.png'")

if __name__ == '__main__':
    success = test_data_provider()
    
    if success:
        print("\nAttempting to visualize a sample...")
        try:
            visualize_sample()
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    sys.exit(0 if success else 1)
