import cv2
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils.img
import data.MPII.ref as ds

def debug_dataset_samples(dataset_type='train', num_samples=5):
    """
    Debug function to view images and keypoints from the MPII dataset
    
    Args:
        dataset_type: 'train', 'valid', or 'test'
        num_samples: Number of samples to visualize
    """
    print(f"Debugging {dataset_type} dataset samples")
    
    # Load annotation file
    annot_path = os.path.join(ds.annot_dir, f'{dataset_type}.h5')
    print(f"Opening annotation file: {annot_path}")
    
    with h5py.File(annot_path, 'r') as f:
        # Check what's in the h5 file
        print(f"Keys in h5 file: {list(f.keys())}")
        
        # Get number of samples
        total_samples = f['imgname'].shape[0]
        print(f"Total samples: {total_samples}")
        
        # Determine how many samples to show
        samples_to_show = min(num_samples, total_samples)
        print(f"Showing {samples_to_show} samples")
        
        # Define joint names for better understanding
        joint_names = [
            "Right ankle", "Right knee", "Right hip", 
            "Left hip", "Left knee", "Left ankle", 
            "Pelvis", "Thorax", "Neck", "Head", 
            "Right wrist", "Right elbow", "Right shoulder",
            "Left shoulder", "Left elbow", "Left wrist"
        ]
        
        # Iterate over samples
        for i in range(samples_to_show):
            # Get image name and path
            img_name = f['imgname'][i].decode('UTF-8')
            img_path = os.path.join(ds.img_dir, img_name)
            print(f"\nSample {i+1}/{samples_to_show}: {img_path}")
            
            # Read original image
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"ERROR: Could not read image {img_path}")
                continue
            
            # Convert BGR to RGB
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Get keypoints and visibility
            keypoints = f['part'][i]  # shape: (16, 2)
            visibility = f['visible'][i]  # shape: (16,)
            
            # Get center and scale for cropping
            center = f['center'][i]
            scale = f['scale'][i]
            
            print(f"Image shape: {orig_img.shape}")
            print(f"Center: {center}, Scale: {scale}")
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # Plot original image
            ax1.imshow(orig_img)
            ax1.set_title("Original Image")
            ax1.axis('off')
            
            # Draw keypoints on original image
            for j, (x, y) in enumerate(keypoints):
                if visibility[j] > 0:
                    ax1.plot(x, y, 'ro', markersize=6)
                    ax1.text(x+5, y+5, f"{j}", fontsize=9, color='white', 
                             bbox=dict(facecolor='red', alpha=0.5))
            
            # Get cropped image (as used in training)
            input_res = 256  # Assuming input resolution is 256x256
            cropped_img = utils.img.crop(orig_img, center, scale, (input_res, input_res))
            
            # Plot cropped image
            ax2.imshow(cropped_img)
            ax2.set_title("Cropped Image (for Network Input)")
            ax2.axis('off')
            
            # Transform keypoints to cropped image coordinates
            cropped_keypoints = []
            for j, (x, y) in enumerate(keypoints):
                if visibility[j] > 0:
                    transformed = utils.img.transform([x, y], center, scale, (input_res, input_res))
                    cropped_keypoints.append((j, transformed))
                    ax2.plot(transformed[0], transformed[1], 'ro', markersize=6)
                    ax2.text(transformed[0]+5, transformed[1]+5, f"{j}", fontsize=9, 
                             color='white', bbox=dict(facecolor='red', alpha=0.5))
            
            plt.tight_layout()
            plt.show()
            
            # Print keypoint information
            print("\nKeypoint details:")
            print("Index | Joint Name       | Position (x, y)   | Visibility")
            print("-" * 60)
            for j, name in enumerate(joint_names):
                vis_status = "Visible" if visibility[j] > 0 else "Not visible"
                print(f"{j:5d} | {name:16s} | ({keypoints[j][0]:6.1f}, {keypoints[j][1]:6.1f}) | {vis_status}")
            
            # Ask if user wants to see next sample
            if i < samples_to_show - 1:
                user_input = input("\nPress Enter to see next sample or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break

if __name__ == "__main__":
    print("=== MPII Dataset Visualization Debug Tool ===\n")
    
    # Check if directories and files exist
    print(f"Image directory: {ds.img_dir}")
    print(f"Annotation directory: {ds.annot_dir}")
    
    # Check for annotation files
    train_file = os.path.join(ds.annot_dir, 'train.h5')
    valid_file = os.path.join(ds.annot_dir, 'valid.h5')
    test_file = os.path.join(ds.annot_dir, 'test.h5')
    
    print(f"\nAvailable annotation files:")
    print(f"- Train: {'✓' if os.path.exists(train_file) else '✗'}")
    print(f"- Valid: {'✓' if os.path.exists(valid_file) else '✗'}")
    print(f"- Test: {'✓' if os.path.exists(test_file) else '✗'}")
    
    print("\nSelect dataset to visualize:")
    print("1. Training set")
    print("2. Validation set")
    print("3. Test set")
    
    choice = input("\nEnter choice (1-3): ")
    
    dataset_map = {'1': 'train', '2': 'valid', '3': 'test'}
    dataset_type = dataset_map.get(choice, 'train')
    
    num_samples = input("Number of samples to visualize (default: 5): ")
    num_samples = int(num_samples) if num_samples.isdigit() else 5
    
    debug_dataset_samples(dataset_type, num_samples)
