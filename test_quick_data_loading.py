#!/usr/bin/env python3
"""
Quick test script for MPI-INF-3DHP data loading
"""

import sys
import os
import numpy as np

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_data_structure():
    """Test the actual data structure"""
    print("=== Quick MPI-INF-3DHP Data Structure Test ===")
    
    # Check train data structure
    train_file = 'data/motion3d/data_train_3dhp.npz'
    test_file = 'data/motion3d/data_test_3dhp.npz'
    
    for split_name, data_file in [('Train', train_file), ('Test', test_file)]:
        if not os.path.exists(data_file):
            print(f"❌ {data_file} not found")
            continue
            
        print(f"\n📊 {split_name} Data Structure:")
        npz_data = np.load(data_file, allow_pickle=True)
        raw_data = npz_data['data'].item()
        
        sample_count = 0
        
        if split_name == 'Train':
            # Training data: {'S1 Seq1': [...], ...}
            for subject_seq, seq_data in raw_data.items():
                print(f"  {subject_seq}: {len(seq_data)} camera setups")
                for camera_dict in seq_data:
                    for camera_str, camera_data in camera_dict.items():
                        if 'data_2d' in camera_data:
                            frames = len(camera_data['data_2d'])
                            sample_count += frames // 5  # Every 5th frame
                            print(f"    Camera {camera_str}: {frames} frames -> {frames // 5} samples")
        else:
            # Test data: {'TS1': {...}, ...}
            for subject, subject_data in raw_data.items():
                if 'data_2d' in subject_data:
                    frames = len(subject_data['data_2d'])
                    sample_count += frames // 10  # Every 10th frame
                    print(f"  {subject}: {frames} frames -> {frames // 10} samples")
        
        print(f"  Total samples: {sample_count}")

def test_data_loading():
    """Test the data provider"""
    print("\n=== Testing Data Provider ===")
    
    try:
        # Import the config
        import task.pose_mpi_inf_3dhp_with_images as task_module
        
        # Create config
        config = task_module.__config__
        
        # Import data provider
        from data.MPI_INF_3DHP.dp_with_images import Dataset
        
        print("✓ Creating train dataset...")
        train_dataset = Dataset(config, train=True)
        
        print("✓ Creating test dataset...")
        test_dataset = Dataset(config, train=False)
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Test dataset: {len(test_dataset)} samples")
        
        # Test loading first sample
        print("\n📝 Testing sample loading...")
        sample = train_dataset[0]
        
        print(f"✓ Sample loaded successfully:")
        print(f"  - Images shape: {sample['imgs'].shape}")
        print(f"  - Heatmaps shape: {sample['heatmaps'].shape}")
        print(f"  - Subject: {sample['subject']}")
        print(f"  - Sequence: {sample['sequence']}")
        print(f"  - Frame: {sample['frame_idx']}")
        print(f"  - Camera: {sample['camera']}")
        print(f"  - Image path: {sample['imgname']}")
        
        print("\n✓ All tests passed! Data loading appears to be working.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_data_structure()
    test_data_loading()