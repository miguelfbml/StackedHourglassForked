#!/usr/bin/env python3
"""
Test script for MPI-INF-3DHP StackedHourglass setup
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append('.')

def test_data_provider():
    """Test if the data provider can be imported and basic functionality works"""
    print("Testing MPI-INF-3DHP data provider...")
    
    try:
        # Test import
        from data.MPI_INF_3DHP import dp
        print("✓ Data provider imported successfully")
        
        # Test configuration
        config = {
            'train': {
                'input_res': 256,
                'output_res': 64,
                'batchsize': 2,
                'num_workers': 0,
                'use_data_loader': True
            },
            'inference': {
                'num_parts': 17
            },
            'data_root': 'data/motion3d'  # This path might not exist, that's okay for testing
        }
        
        # Test heatmap generation
        heatmap_gen = dp.GenerateHeatmap(64, 17)
        dummy_keypoints = np.random.rand(1, 17, 3) * 64  # Random keypoints
        dummy_keypoints[:, :, 2] = 1  # Set visibility
        heatmaps = heatmap_gen(dummy_keypoints)
        
        print(f"✓ Heatmap generation works. Output shape: {heatmaps.shape}")
        assert heatmaps.shape == (17, 64, 64), f"Expected (17, 64, 64), got {heatmaps.shape}"
        
        print("✓ All basic tests passed!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_task_config():
    """Test if the task configuration can be imported"""
    print("\nTesting MPI-INF-3DHP task configuration...")
    
    try:
        from task import pose_mpi_inf_3dhp
        print("✓ Task configuration imported successfully")
        
        config = pose_mpi_inf_3dhp.__config__
        
        # Check key parameters
        assert config['inference']['num_parts'] == 17, "num_parts should be 17"
        assert config['inference']['oup_dim'] == 17, "oup_dim should be 17"
        assert config['data_provider'] == 'data.MPI_INF_3DHP.dp', "Wrong data provider"
        
        print(f"✓ Configuration is correct:")
        print(f"  - Number of parts: {config['inference']['num_parts']}")
        print(f"  - Output dimension: {config['inference']['oup_dim']}")
        print(f"  - Data provider: {config['data_provider']}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def test_model_compatibility():
    """Test if the model can handle 17 keypoints"""
    print("\nTesting model compatibility...")
    
    try:
        import torch
        from models.posenet import PoseNet
        
        # Create model with 17 output channels
        model = PoseNet(nstack=2, inp_dim=256, oup_dim=17)
        print("✓ Model created with 17 output channels")
        
        # Test forward pass
        dummy_input = torch.randn(1, 256, 256, 3)  # (B, H, W, C)
        output = model(dummy_input)
        
        expected_shape = (1, 2, 17, 64, 64)  # (B, nstack, joints, H, W)
        print(f"✓ Forward pass successful. Output shape: {output.shape}")
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("MPI-INF-3DHP StackedHourglass Setup Test")
    print("=" * 60)
    
    tests = [
        test_data_provider,
        test_task_config,
        test_model_compatibility
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    if all(results):
        print("✓ All tests passed! The setup should work.")
        print("\nNext steps:")
        print("1. Make sure you have the MPI-INF-3DHP data in data/motion3d/")
        print("2. Run: python train_mpi_inf_3dhp.py --data_root data/motion3d")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
