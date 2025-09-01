#!/usr/bin/env python3
"""
Dataset analysis script for MPI-INF-3DHP
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

def analyze_dataset():
    """Analyze the MPI-INF-3DHP dataset"""
    
    data_path = 'data/MPI_INF_3DHP'
    
    print("=== MPI-INF-3DHP Dataset Analysis ===\n")
    
    # Analyze each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        h5_path = f'{data_path}/annot/{split}.h5'
        
        if not os.path.exists(h5_path):
            print(f"❌ {split}.h5 not found")
            continue
            
        print(f"📊 {split.upper()} SET:")
        
        with h5py.File(h5_path, 'r') as f:
            # Basic info
            keys = list(f.keys())
            print(f"   Keys: {keys}")
            
            if 'imgname' in f:
                imgnames = f['imgname'][:]
                print(f"   Total samples: {len(imgnames)}")
                
                # Decode imgnames
                if len(imgnames) > 0:
                    sample_names = [name.decode() if isinstance(name, bytes) else name for name in imgnames[:5]]
                    print(f"   Sample names: {sample_names}")
            
            if 'part' in f:
                joints = f['part'][:]
                print(f"   Joint shape: {joints.shape}")
                print(f"   Number of keypoints: {joints.shape[1] if len(joints.shape) > 1 else 'Unknown'}")
                
                # Check for valid joints (non-zero)
                if len(joints.shape) == 3:  # [N, 17, 3] format
                    valid_joints = np.sum(joints[:, :, 2] > 0, axis=1)  # Count visible joints per sample
                    print(f"   Avg visible joints per sample: {np.mean(valid_joints):.1f}")
                    print(f"   Min visible joints: {np.min(valid_joints)}")
                    print(f"   Max visible joints: {np.max(valid_joints)}")
            
            if 'center' in f:
                centers = f['center'][:]
                print(f"   Center shape: {centers.shape}")
            
            if 'scale' in f:
                scales = f['scale'][:]
                print(f"   Scale shape: {scales.shape}")
                print(f"   Scale range: {np.min(scales):.3f} - {np.max(scales):.3f}")
        
        print()

def check_image_availability():
    """Check which images are actually available"""
    
    data_path = 'data/MPI_INF_3DHP'
    
    print("=== Image Availability Check ===\n")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        h5_path = f'{data_path}/annot/{split}.h5'
        
        if not os.path.exists(h5_path):
            continue
            
        print(f"📁 {split.upper()} SET:")
        
        with h5py.File(h5_path, 'r') as f:
            if 'imgname' not in f:
                print("   No imgname found")
                continue
                
            imgnames = f['imgname'][:]
            total_images = len(imgnames)
            found_images = 0
            missing_images = []
            
            for i, imgname in enumerate(imgnames[:100]):  # Check first 100 for speed
                if isinstance(imgname, bytes):
                    imgname = imgname.decode()
                
                img_path = os.path.join(data_path, imgname)
                if os.path.exists(img_path):
                    found_images += 1
                else:
                    missing_images.append(imgname)
                    
                if i % 20 == 0:
                    print(f"   Checked {i+1}/{min(100, total_images)} images...")
            
            print(f"   Found: {found_images}/{min(100, total_images)} images")
            print(f"   Missing rate: {((min(100, total_images) - found_images) / min(100, total_images) * 100):.1f}%")
            
            if missing_images[:3]:
                print(f"   Sample missing: {missing_images[:3]}")
        
        print()

def analyze_synthetic_vs_real():
    """Analyze synthetic vs real images in the dataset"""
    
    data_path = 'data/MPI_INF_3DHP'
    
    print("=== Synthetic vs Real Image Analysis ===\n")
    
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        h5_path = f'{data_path}/annot/{split}.h5'
        
        if not os.path.exists(h5_path):
            continue
            
        print(f"🔍 {split.upper()} SET:")
        
        with h5py.File(h5_path, 'r') as f:
            if 'imgname' not in f:
                continue
                
            imgnames = f['imgname'][:]
            
            synthetic_count = 0
            real_count = 0
            
            for imgname in imgnames:
                if isinstance(imgname, bytes):
                    imgname = imgname.decode()
                
                # Heuristic: synthetic images often contain certain patterns
                if any(pattern in imgname.lower() for pattern in ['synthetic', 'render', 'mpi']):
                    synthetic_count += 1
                else:
                    real_count += 1
            
            total = len(imgnames)
            print(f"   Total: {total}")
            print(f"   Real-like: {real_count} ({real_count/total*100:.1f}%)")
            print(f"   Synthetic-like: {synthetic_count} ({synthetic_count/total*100:.1f}%)")
        
        print()

def plot_dataset_statistics():
    """Plot dataset statistics"""
    
    data_path = 'data/MPI_INF_3DHP'
    splits = ['train', 'valid', 'test']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Collect data
    split_sizes = []
    avg_visible_joints = []
    scale_distributions = []
    
    for split in splits:
        h5_path = f'{data_path}/annot/{split}.h5'
        
        if not os.path.exists(h5_path):
            continue
            
        with h5py.File(h5_path, 'r') as f:
            # Dataset sizes
            if 'imgname' in f:
                split_sizes.append((split, len(f['imgname'][:])))
            
            # Visible joints
            if 'part' in f:
                joints = f['part'][:]
                if len(joints.shape) == 3:
                    visible_per_sample = np.sum(joints[:, :, 2] > 0, axis=1)
                    avg_visible_joints.append((split, visible_per_sample))
            
            # Scale distribution
            if 'scale' in f:
                scales = f['scale'][:]
                scale_distributions.append((split, scales))
    
    # Plot 1: Dataset sizes
    if split_sizes:
        splits_names, sizes = zip(*split_sizes)
        axes[0, 0].bar(splits_names, sizes, color=['blue', 'green', 'red'])
        axes[0, 0].set_title('Dataset Sizes')
        axes[0, 0].set_ylabel('Number of Samples')
        for i, v in enumerate(sizes):
            axes[0, 0].text(i, v + max(sizes)*0.01, str(v), ha='center')
    
    # Plot 2: Visible joints distribution
    if avg_visible_joints:
        for i, (split, visible) in enumerate(avg_visible_joints):
            axes[0, 1].hist(visible, bins=18, alpha=0.7, label=split, 
                           color=['blue', 'green', 'red'][i])
        axes[0, 1].set_title('Visible Joints Distribution')
        axes[0, 1].set_xlabel('Number of Visible Joints')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
    
    # Plot 3: Scale distribution
    if scale_distributions:
        for i, (split, scales) in enumerate(scale_distributions):
            axes[1, 0].hist(scales, bins=50, alpha=0.7, label=split,
                           color=['blue', 'green', 'red'][i])
        axes[1, 0].set_title('Scale Distribution')
        axes[1, 0].set_xlabel('Scale Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
    
    # Plot 4: Summary statistics
    axes[1, 1].axis('off')
    summary_text = "Dataset Summary:\n\n"
    for split, size in split_sizes:
        summary_text += f"{split.upper()}: {size:,} samples\n"
    
    if avg_visible_joints:
        summary_text += "\nAvg Visible Joints:\n"
        for split, visible in avg_visible_joints:
            summary_text += f"{split.upper()}: {np.mean(visible):.1f} ± {np.std(visible):.1f}\n"
    
    axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                   fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('dataset_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset')
    parser.add_argument('--check-images', action='store_true', help='Check image availability')
    parser.add_argument('--synthetic', action='store_true', help='Analyze synthetic vs real')
    parser.add_argument('--plot', action='store_true', help='Plot statistics')
    parser.add_argument('--all', action='store_true', help='Run all analyses')
    
    args = parser.parse_args()
    
    if args.all or args.analyze:
        analyze_dataset()
    
    if args.all or args.check_images:
        check_image_availability()
    
    if args.all or args.synthetic:
        analyze_synthetic_vs_real()
    
    if args.all or args.plot:
        plot_dataset_statistics()
