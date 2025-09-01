#!/usr/bin/env python3
"""
Quick test for MPI-INF-3DHP data loading
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_loading():
    print("=== Quick MPI-INF-3DHP Data Loading Test ===")
    
    try:
        # Import the task
        import task.pose_mpi_inf_3dhp_with_images as task_module
        config = task_module.__config__
        
        # Import data provider
        from data.MPI_INF_3DHP.dp_with_images import Dataset
        
        print("✓ Creating train dataset...")
        train_dataset = Dataset(config, train=True)
        
        print("✓ Creating test dataset...")
        test_dataset = Dataset(config, train=False)
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Test dataset: {len(test_dataset)} samples")
        
        if len(train_dataset) > 0:
            # Test loading first sample
            print("\n📝 Testing sample loading...")
            sample = train_dataset[0]
            
            print(f"✓ Sample loaded successfully:")
            print(f"  - Images shape: {sample['imgs'].shape}")
            print(f"  - Heatmaps shape: {sample['heatmaps'].shape}")
            
            print("\n✓ All tests passed! Data loading is working.")
        else:
            print("❌ No training samples found!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_data_loading()