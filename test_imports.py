#!/usr/bin/env python3
"""
Quick test to check if all modules can be imported correctly
"""

import sys
import os

def test_imports():
    print("Testing module imports...")
    
    try:
        # Test task import
        import importlib
        task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
        print("✓ Task module imported successfully")
        
        # Test data provider import
        data_provider = importlib.import_module(task.__config__['data_provider'])
        print("✓ Data provider module imported successfully")
        
        # Test if config is correct
        config = task.__config__
        print(f"✓ Config loaded:")
        print(f"  - Data provider: {config['data_provider']}")
        print(f"  - Network: {config['network']}")
        print(f"  - Number of parts: {config['inference']['num_parts']}")
        print(f"  - Batch size: {config['train']['batchsize']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    
    if success:
        print("\n🎉 All imports successful! You can now run the training.")
        print("\nTo start training:")
        print("sbatch launch_mpi_inf_3dhp_with_images.sh")
    else:
        print("\n❌ Import test failed. Please check the error messages above.")
