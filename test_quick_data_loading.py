#!/usr/bin/env python3
"""
Quick test script to debug MPI-INF-3DHP data loading issues
"""

import os
import sys
import traceback

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

def test_quick():
    """Quick test of data loading"""
    print("=== Quick MPI-INF-3DHP Data Loading Test ===")
    
    # Check if data files exist
    data_root = 'data/motion3d'  # Changed from 'data/MPI_INF_3DHP/motion3d'
    train_file = os.path.join(data_root, "data_train_3dhp.npz")
    test_file = os.path.join(data_root, "data_test_3dhp.npz")
    
    print(f"Checking data files:")
    print(f"  Train file: {train_file} - {'✓' if os.path.exists(train_file) else '✗'}")
    print(f"  Test file: {test_file} - {'✓' if os.path.exists(test_file) else '✗'}")
    
    if not os.path.exists(train_file) or not os.path.exists(test_file):
        print("ERROR: Required data files are missing!")
        return False
    
    # Test importing modules
    try:
        print("\nTesting module imports...")
        import importlib
        
        print("  - Importing task module...")
        task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
        print("  ✓ Task module imported")
        
        print("  - Importing data provider...")
        data_func = importlib.import_module('data.MPI_INF_3DHP.dp_with_images')
        print("  ✓ Data provider imported")
        
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        traceback.print_exc()
        return False
    
    # Test creating a minimal dataset
    try:
        print("\nTesting dataset creation...")
        
        # Create minimal config
        config = {
            'train': {
                'input_res': 256,
                'output_res': 64,
                'batchsize': 1
            },
            'inference': {
                'num_parts': 17
            }
        }
        
        print("  - Creating training dataset...")
        train_dataset = data_func.Dataset(
            config, 
            train=True, 
            data_root=data_root,
            mpi_dataset_root='/nas-ctm01/datasets/public/mpi_inf_3dhp'
        )
        print(f"  ✓ Training dataset created with {len(train_dataset)} samples")
        
        print("  - Creating validation dataset...")
        val_dataset = data_func.Dataset(
            config, 
            train=False, 
            data_root=data_root,
            mpi_dataset_root='/nas-ctm01/datasets/public/mpi_inf_3dhp'
        )
        print(f"  ✓ Validation dataset created with {len(val_dataset)} samples")
        
    except Exception as e:
        print(f"  ✗ Dataset creation error: {e}")
        traceback.print_exc()
        return False
    
    # Test loading a sample
    try:
        print("\nTesting sample loading...")
        if len(train_dataset) > 0:
            print("  - Loading first training sample...")
            sample = train_dataset[0]
            print(f"  ✓ Sample loaded successfully")
            
            if isinstance(sample, dict):
                print(f"    - Images shape: {sample['imgs'].shape}")
                print(f"    - Heatmaps shape: {sample['heatmaps'].shape}")
            else:
                print(f"    - Unexpected format: {type(sample)}")
        else:
            print("  ✗ No samples in training dataset")
            return False
            
    except Exception as e:
        print(f"  ✗ Sample loading error: {e}")
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed! Data loading appears to be working.")
    return True

if __name__ == '__main__':
    success = test_quick()
    sys.exit(0 if success else 1)
