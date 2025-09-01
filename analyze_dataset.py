#!/usr/bin/env python3
"""
Dataset analysis script for MPI-INF-3DHP
"""

import numpy as np
import os
from collections import Counter

def analyze_motion3d_data():
    """Analyze the motion3d .npz files"""
    
    print("=== MPI-INF-3DHP Motion3D Data Analysis ===\n")
    
    # Check the actual files you have
    data_files = [
        'data/motion3d/data_train_3dhp.npz',
        'data/motion3d/data_test_3dhp.npz'
    ]
    
    for data_file in data_files:
        if not os.path.exists(data_file):
            print(f"❌ {data_file} not found")
            continue
            
        print(f"📊 {data_file}:")
        
        try:
            npz_data = np.load(data_file, allow_pickle=True)
            print(f"   NPZ Keys: {list(npz_data.keys())}")
            
            # The data is stored in a single 'data' key as an object
            if 'data' in npz_data:
                data = npz_data['data'].item()  # Extract the dictionary from the object array
                
                if isinstance(data, dict):
                    print(f"   Data keys: {list(data.keys())}")
                    
                    for key in data.keys():
                        item = data[key]
                        if isinstance(item, np.ndarray):
                            print(f"   {key}: shape {item.shape}, dtype {item.dtype}")
                            
                            # Show some sample data
                            if key == 'imgname' and len(item) > 0:
                                print(f"      Sample imgnames: {item[:3]}")
                            elif key == 'joint_2d' and len(item.shape) > 1:
                                print(f"      2D joints shape: {item.shape}")
                                if len(item) > 0:
                                    print(f"      Sample 2D joint: {item[0, 0] if item.shape[1] > 0 else 'None'}")
                            elif key == 'joint_3d' and len(item.shape) > 1:
                                print(f"      3D joints shape: {item.shape}")
                                if len(item) > 0:
                                    print(f"      Sample 3D joint: {item[0, 0] if item.shape[1] > 0 else 'None'}")
                            elif key == 'action' and len(item) > 0:
                                actions = Counter(item)
                                print(f"      Actions: {dict(list(actions.items())[:10])}")  # Show first 10
                            elif key == 'camera' and len(item) > 0:
                                cameras = Counter(item)
                                print(f"      Cameras: {dict(cameras)}")
                            elif key == 'subject' and len(item) > 0:
                                subjects = Counter(item)
                                print(f"      Subjects: {dict(subjects)}")
                        else:
                            print(f"   {key}: {type(item)} - {str(item)[:100]}...")
                else:
                    print(f"   Data is not a dictionary: {type(data)}")
            
            print()
            
        except Exception as e:
            print(f"   Error loading {data_file}: {e}")
            import traceback
            traceback.print_exc()
            print()

def check_image_paths():
    """Check what image paths look like and if they exist"""
    
    print("=== Image Path Analysis ===\n")
    
    data_files = [
        ('Train', 'data/motion3d/data_train_3dhp.npz'),
        ('Test', 'data/motion3d/data_test_3dhp.npz')
    ]
    
    mpi_roots = [
        '/nas-ctm01/datasets/public/mpi_inf_3dhp',
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set'
    ]
    
    for split_name, data_file in data_files:
        if not os.path.exists(data_file):
            continue
            
        print(f"📁 {split_name} Image Paths:")
        
        try:
            npz_data = np.load(data_file, allow_pickle=True)
            
            if 'data' in npz_data:
                data = npz_data['data'].item()
                
                if isinstance(data, dict) and 'imgname' in data:
                    imgnames = data['imgname']
                    print(f"   Total images: {len(imgnames)}")
                    
                    # Check first few image paths
                    found_images = 0
                    for i in range(min(10, len(imgnames))):
                        imgname = imgnames[i]
                        print(f"   [{i}] Image path: {imgname}")
                        
                        # Try different root paths
                        for root in mpi_roots:
                            full_path = os.path.join(root, imgname)
                            if os.path.exists(full_path):
                                print(f"       ✓ Found at: {root}")
                                found_images += 1
                                break
                        else:
                            print(f"       ❌ Not found in any root")
                    
                    print(f"   Found {found_images}/{min(10, len(imgnames))} sample images")
                else:
                    print(f"   No 'imgname' found in data")
            
            print()
            
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            print()

def analyze_dataset_balance():
    """Analyze dataset balance"""
    
    print("=== Dataset Balance Analysis ===\n")
    
    data_files = [
        ('Train', 'data/motion3d/data_train_3dhp.npz'),
        ('Test', 'data/motion3d/data_test_3dhp.npz')
    ]
    
    total_train = 0
    total_test = 0
    
    for split_name, data_file in data_files:
        if not os.path.exists(data_file):
            continue
            
        try:
            npz_data = np.load(data_file, allow_pickle=True)
            
            if 'data' in npz_data:
                data = npz_data['data'].item()
                
                if isinstance(data, dict):
                    # Try different possible keys for number of samples
                    sample_keys = ['imgname', 'joint_2d', 'joint_3d']
                    num_samples = 0
                    
                    for key in sample_keys:
                        if key in data:
                            num_samples = len(data[key])
                            break
                    
                    print(f"📊 {split_name}: {num_samples:,} samples")
                    
                    if split_name == 'Train':
                        total_train = num_samples
                    else:
                        total_test = num_samples
                    
                    # Analyze actions if available
                    if 'action' in data:
                        actions = Counter(data['action'])
                        print(f"   Actions: {dict(list(actions.items())[:10])}")  # Show first 10
                    
                    # Analyze subjects if available
                    if 'subject' in data:
                        subjects = Counter(data['subject'])
                        print(f"   Subjects: {dict(subjects)}")
                    
                    print()
                
        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    if total_train > 0 and total_test > 0:
        ratio = total_train / total_test
        print(f"📈 Train/Test Ratio: {ratio:.1f}:1")
        print(f"   Train: {total_train:,} ({total_train/(total_train+total_test)*100:.1f}%)")
        print(f"   Test:  {total_test:,} ({total_test/(total_train+total_test)*100:.1f}%)")

def check_directory_structure():
    """Check what directories and files actually exist"""
    
    print("=== Directory Structure Check ===\n")
    
    paths_to_check = [
        'data/',
        'data/motion3d/',
        'data/MPI_INF_3DHP/',
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/',
        '/nas-ctm01/datasets/public/mpi_inf_3dhp/mpi_inf_3dhp_test_set/'
    ]
    
    for path in paths_to_check:
        if os.path.exists(path):
            print(f"✓ {path}")
            try:
                contents = os.listdir(path)
                if len(contents) <= 10:
                    print(f"   Contents: {contents}")
                else:
                    print(f"   Contents: {contents[:10]}... ({len(contents)} total)")
            except PermissionError:
                print(f"   (Permission denied)")
        else:
            print(f"❌ {path}")
        print()

if __name__ == '__main__':
    check_directory_structure()
    analyze_motion3d_data()
    check_image_paths()
    analyze_dataset_balance()