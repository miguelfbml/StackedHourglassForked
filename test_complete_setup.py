#!/usr/bin/env python3
"""
Quick test to verify the complete setup before training
"""

import sys
import os
import numpy as np

def test_complete_setup():
    print("="*60)
    print("COMPLETE SETUP TEST")
    print("="*60)
    
    # 1. Test imports
    print("\n1. Testing imports...")
    try:
        import importlib
        task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
        data_provider = importlib.import_module(task.__config__['data_provider'])
        print("✓ All modules import successfully")
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False
    
    # 2. Test data files
    print("\n2. Testing data files...")
    train_file = "data/motion3d/data_train_3dhp.npz"
    test_file = "data/motion3d/data_test_3dhp.npz"
    
    if os.path.exists(train_file) and os.path.exists(test_file):
        print("✓ NPZ data files exist")
    else:
        print(f"❌ Missing data files: train={os.path.exists(train_file)}, test={os.path.exists(test_file)}")
        return False
    
    # 3. Test image paths
    print("\n3. Testing image paths...")
    mpi_root = "/nas-ctm01/datasets/public/mpi_inf_3dhp"
    
    if os.path.exists(mpi_root):
        print("✓ MPI dataset root exists")
        
        # Check for at least one training sequence
        found_training = False
        for subject in ['S1', 'S2']:
            for seq in ['Seq1', 'Seq2']:
                for cam in ['0', '1', '2']:
                    path = f"{mpi_root}/{subject}/{seq}/imageSequence/video_{cam}"
                    if os.path.exists(path):
                        print(f"✓ Found training images at {path}")
                        found_training = True
                        break
                if found_training:
                    break
            if found_training:
                break
        
        if not found_training:
            print("⚠️ No training images found - will use synthetic images as fallback")
        
        # Check for test images
        test_set_path = f"{mpi_root}/mpi_inf_3dhp_test_set"
        if os.path.exists(test_set_path):
            print("✓ Test set directory exists")
        else:
            print("⚠️ Test set directory not found - validation may use synthetic images")
            
    else:
        print(f"❌ MPI dataset root not found: {mpi_root}")
        print("⚠️ Will use synthetic images as fallback")
    
    # 4. Test data provider instantiation
    print("\n4. Testing data provider...")
    try:
        config = task.__config__.copy()
        config['data_root'] = 'data/motion3d'
        config['mpi_dataset_root'] = mpi_root
        
        # Create a small dataset for testing
        dataset = data_provider.Dataset(config, train=True, data_root='data/motion3d', mpi_dataset_root=mpi_root)
        print(f"✓ Training dataset created with {len(dataset)} samples")
        
        # Test loading one sample
        if len(dataset) > 0:
            img, heatmap = dataset[0]
            print(f"✓ Sample loaded - img shape: {img.shape}, heatmap shape: {heatmap.shape}")
        else:
            print("❌ Dataset is empty")
            return False
            
    except Exception as e:
        print(f"❌ Data provider error: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ SETUP TEST COMPLETED SUCCESSFULLY!")
    print("="*60)
    print("\nYou can now start training with:")
    print("sbatch launch_mpi_inf_3dhp_with_images.sh")
    print("\nOr run directly with:")
    print("python train_mpi_inf_3dhp_with_images.py --exp pose_mpi_inf_3dhp_images")
    
    return True

if __name__ == "__main__":
    test_complete_setup()
