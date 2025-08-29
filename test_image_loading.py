#!/usr/bin/env python3
import sys
import os
import numpy as np
import cv2
from imageio import imread
import glob

def test_image_paths(mpi_dataset_root="/nas-ctm01/datasets/public/mpi_inf_3dhp"):
    """Test if we can access MPI-INF-3DHP images"""
    
    print("="*60)
    print("MPI-INF-3DHP Image Loading Test")
    print("="*60)
    
    # Test training data structure
    print("\n1. Testing Training Data Structure:")
    print(f"Dataset root: {mpi_dataset_root}")
    
    # Check if root exists
    if not os.path.exists(mpi_dataset_root):
        print(f"❌ Dataset root does not exist: {mpi_dataset_root}")
        return False
    
    print(f"✓ Dataset root exists")
    
    # Test training data - S1/Seq1/imageSequence/video_0
    found_training_images = False
    for subject in ['S1', 'S2', 'S3', 'S4']:
        for seq in ['Seq1', 'Seq2']:
            for cam in ['0', '1', '2', '4', '5', '6', '7', '8']:
                image_dir = f"{mpi_dataset_root}/{subject}/{seq}/imageSequence/video_{cam}"
                if os.path.exists(image_dir):
                    # Check for images
                    image_files = glob.glob(f"{image_dir}/*.jpg") + glob.glob(f"{image_dir}/*.png")
                    if len(image_files) > 0:
                        print(f"✓ Found {len(image_files)} images in {image_dir}")
                        
                        # Test loading one image
                        try:
                            test_img = imread(image_files[0])
                            print(f"  - Sample image shape: {test_img.shape}")
                            found_training_images = True
                            break
                        except Exception as e:
                            print(f"  - Error loading {image_files[0]}: {e}")
                if found_training_images:
                    break
            if found_training_images:
                break
        if found_training_images:
            break
    
    if not found_training_images:
        print("❌ No training images found")
        # List available directories
        print("\nAvailable directories in dataset root:")
        try:
            for item in os.listdir(mpi_dataset_root):
                if os.path.isdir(os.path.join(mpi_dataset_root, item)):
                    print(f"  - {item}")
        except Exception as e:
            print(f"Error listing directories: {e}")
    
    # Test validation data structure
    print("\n2. Testing Validation Data Structure:")
    test_set_dir = f"{mpi_dataset_root}/mpi_inf_3dhp_test_set"
    
    if os.path.exists(test_set_dir):
        print(f"✓ Test set directory exists: {test_set_dir}")
        
        # Check for test sequences
        found_test_images = False
        try:
            test_sequences = os.listdir(test_set_dir)
            print(f"  Found test sequences: {test_sequences[:5]}...")  # Show first 5
            
            for seq in test_sequences[:3]:  # Check first 3 sequences
                seq_path = f"{test_set_dir}/{seq}/imageSequence"
                if os.path.exists(seq_path):
                    image_files = glob.glob(f"{seq_path}/*.jpg") + glob.glob(f"{seq_path}/*.png")
                    if len(image_files) > 0:
                        print(f"✓ Found {len(image_files)} images in {seq}")
                        
                        # Test loading one image
                        try:
                            test_img = imread(image_files[0])
                            print(f"  - Sample image shape: {test_img.shape}")
                            found_test_images = True
                            break
                        except Exception as e:
                            print(f"  - Error loading {image_files[0]}: {e}")
        except Exception as e:
            print(f"Error listing test sequences: {e}")
        
        if not found_test_images:
            print("❌ No test images found")
    else:
        print(f"❌ Test set directory does not exist: {test_set_dir}")
    
    print("\n" + "="*60)
    if found_training_images or found_test_images:
        print("✓ Image loading test PASSED")
        return True
    else:
        print("❌ Image loading test FAILED")
        return False

def test_data_loading():
    """Test the data loading from npz files"""
    print("\n3. Testing NPZ Data Loading:")
    
    # Test training data
    train_file = "data/motion3d/data_train_3dhp.npz"
    if os.path.exists(train_file):
        try:
            data = np.load(train_file, allow_pickle=True)['data'].item()
            print(f"✓ Loaded training data: {len(data)} sequences")
            
            # Show first sequence info
            first_seq = list(data.keys())[0]
            first_cam = list(data[first_seq][0].keys())[0]
            first_data = data[first_seq][0][first_cam]
            
            print(f"  - First sequence: {first_seq}")
            print(f"  - First camera: {first_cam}")
            print(f"  - 2D data shape: {first_data['data_2d'].shape}")
            print(f"  - 3D data shape: {first_data['data_3d'].shape}")
            
        except Exception as e:
            print(f"❌ Error loading training data: {e}")
    else:
        print(f"❌ Training data file not found: {train_file}")
    
    # Test validation data
    test_file = "data/motion3d/data_test_3dhp.npz"
    if os.path.exists(test_file):
        try:
            data = np.load(test_file, allow_pickle=True)['data'].item()
            print(f"✓ Loaded test data: {len(data)} sequences")
            
            # Show first sequence info
            first_seq = list(data.keys())[0]
            first_data = data[first_seq]
            
            print(f"  - First sequence: {first_seq}")
            print(f"  - 2D data shape: {first_data['data_2d'].shape}")
            print(f"  - 3D data shape: {first_data['data_3d'].shape}")
            print(f"  - Valid frames: {np.sum(first_data['valid'])}/{len(first_data['valid'])}")
            
        except Exception as e:
            print(f"❌ Error loading test data: {e}")
    else:
        print(f"❌ Test data file not found: {test_file}")

if __name__ == "__main__":
    # Test with default path
    success = test_image_paths()
    test_data_loading()
    
    if success:
        print("\n🎉 All tests passed! You can start training with real images.")
        print("\nTo start training, run:")
        print("sbatch launch_mpi_inf_3dhp_with_images.sh")
    else:
        print("\n⚠️ Some tests failed. Please check the image paths.")
