import cv2
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import utils.img without triggering the assertion error in ref.py
import utils.img

# Manually set dataset paths (bypassing the assertion check in ref.py)
class DatasetPaths:
    def __init__(self):
        self.annot_dir = 'data/MPII/annot'
        self.img_dir = 'data/MPII/images'
        
        # Store flipped parts mapping from the ref.py file
        self.flipped_parts = {
            'mpii': [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 15, 14, 13, 12, 11, 10]
        }

# Create an instance of DatasetPaths instead of importing ref.py directly
ds = DatasetPaths()

def check_paths():
    """Check if the data paths exist and suggest fixes if they don't"""
    # Check current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Check if annotation directory exists
    annot_dir_exists = os.path.exists(ds.annot_dir)
    print(f"Annotation directory ({ds.annot_dir}): {'✓ Exists' if annot_dir_exists else '✗ Not found'}")
    
    # Check if image directory exists
    img_dir_exists = os.path.exists(ds.img_dir)
    print(f"Image directory ({ds.img_dir}): {'✓ Exists' if img_dir_exists else '✗ Not found'}")
    
    # Suggest solutions if directories don't exist
    if not annot_dir_exists or not img_dir_exists:
        print("\nPossible solutions:")
        print("1. Make sure you're running the script from the project root directory")
        print("2. Create the missing directories and download the dataset")
        print("3. Modify the paths in the script to match your dataset location")
        
        # Try to find the actual location
        print("\nSearching for potential dataset locations...")
        
        # Look for potential annotation files in the current directory tree
        for root, dirs, files in os.walk('.', topdown=True, maxdepth=3):
            if 'train.h5' in files and 'valid.h5' in files:
                print(f"Found annotation files in: {root}")
                print(f"Consider setting annot_dir to: '{root}'")
                
            # Limit the search depth and exclude certain directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'env']]
        
        return False
    
    return True

def debug_dataset_samples(dataset_type='train', num_samples=5, ignore_missing_images=False):
    """
    Debug function to view images and keypoints from the MPII dataset
    
    Args:
        dataset_type: 'train', 'valid', or 'test'
        num_samples: Number of samples to visualize
        ignore_missing_images: If True, will continue even if individual images are missing
    """
    print(f"Debugging {dataset_type} dataset samples")
    
    # Load annotation file
    annot_path = os.path.join(ds.annot_dir, f'{dataset_type}.h5')
    print(f"Opening annotation file: {annot_path}")
    
    if not os.path.exists(annot_path):
        print(f"ERROR: Annotation file not found: {annot_path}")
        return
    
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
        successful_samples = 0
        for i in range(total_samples):
            if successful_samples >= samples_to_show:
                break
                
            # Get image name and path
            img_name = f['imgname'][i].decode('UTF-8')
            img_path = os.path.join(ds.img_dir, img_name)
            print(f"\nSample {successful_samples+1}/{samples_to_show} (Dataset index: {i}): {img_path}")
            
            # Read original image
            try:
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    raise FileNotFoundError(f"Could not read image {img_path}")
                    
                # Convert BGR to RGB
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"ERROR: {str(e)}")
                if ignore_missing_images:
                    print("Skipping this sample...")
                    continue
                else:
                    print("To continue without this image, run with ignore_missing_images=True")
                    break
            
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
            
            successful_samples += 1
            
            # Ask if user wants to see next sample
            if successful_samples < samples_to_show:
                user_input = input("\nPress Enter to see next sample or 'q' to quit: ")
                if user_input.lower() == 'q':
                    break

def debug_annotations_only(dataset_type='train', num_samples=5):
    """
    Debug function to view annotations without requiring the actual images
    
    Args:
        dataset_type: 'train', 'valid', or 'test'
        num_samples: Number of samples to examine
    """
    print(f"Examining {dataset_type} dataset annotations (without images)")
    
    # Load annotation file
    annot_path = os.path.join(ds.annot_dir, f'{dataset_type}.h5')
    print(f"Opening annotation file: {annot_path}")
    
    if not os.path.exists(annot_path):
        print(f"ERROR: Annotation file not found: {annot_path}")
        return
    
    with h5py.File(annot_path, 'r') as f:
        # Check what's in the h5 file
        print(f"Keys in h5 file: {list(f.keys())}")
        
        # Get number of samples
        total_samples = f['imgname'].shape[0]
        print(f"Total samples: {total_samples}")
        
        # Determine how many samples to show
        samples_to_show = min(num_samples, total_samples)
        print(f"Examining {samples_to_show} samples")
        
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
            # Get image name
            img_name = f['imgname'][i].decode('UTF-8')
            img_path = os.path.join(ds.img_dir, img_name)
            print(f"\nSample {i+1}/{samples_to_show}: {img_path}")
            
            # Get keypoints and visibility
            keypoints = f['part'][i]  # shape: (16, 2)
            visibility = f['visible'][i]  # shape: (16,)
            
            # Get center and scale for cropping
            center = f['center'][i]
            scale = f['scale'][i]
            normalize = f['normalize'][i] if 'normalize' in f else None
            
            print(f"Center: {center}, Scale: {scale}")
            if normalize is not None:
                print(f"Normalize: {normalize}")
            
            # Create a blank canvas to visualize keypoint positions
            canvas = np.ones((500, 500, 3), dtype=np.uint8) * 255
            
            # Calculate min/max coordinates to determine scaling
            valid_points = keypoints[visibility > 0]
            if len(valid_points) > 0:
                min_x, min_y = valid_points.min(axis=0)
                max_x, max_y = valid_points.max(axis=0)
                
                # Scale factor to fit all points on canvas with padding
                scale_x = 400 / max(1, max_x - min_x)
                scale_y = 400 / max(1, max_y - min_y)
                scale_factor = min(scale_x, scale_y)
                
                # Center points on canvas
                offset_x = 50 + (400 - scale_factor * (max_x - min_x)) / 2 - scale_factor * min_x
                offset_y = 50 + (400 - scale_factor * (max_y - min_y)) / 2 - scale_factor * min_y
                
                # Draw connections between joints (skeleton)
                connections = [
                    (0, 1), (1, 2), (2, 6), 
                    (3, 4), (4, 5), (3, 6),
                    (6, 7), (7, 8), (8, 9),
                    (10, 11), (11, 12), (12, 7),
                    (13, 14), (14, 15), (13, 7)
                ]
                
                for conn in connections:
                    if visibility[conn[0]] > 0 and visibility[conn[1]] > 0:
                        pt1 = (int(keypoints[conn[0], 0] * scale_factor + offset_x), 
                               int(keypoints[conn[0], 1] * scale_factor + offset_y))
                        pt2 = (int(keypoints[conn[1], 0] * scale_factor + offset_x), 
                               int(keypoints[conn[1], 1] * scale_factor + offset_y))
                        cv2.line(canvas, pt1, pt2, (200, 200, 200), 2)
                
                # Draw keypoints
                for j, (x, y) in enumerate(keypoints):
                    if visibility[j] > 0:
                        x_canvas = int(x * scale_factor + offset_x)
                        y_canvas = int(y * scale_factor + offset_y)
                        cv2.circle(canvas, (x_canvas, y_canvas), 5, (0, 0, 255), -1)
                        cv2.putText(canvas, str(j), (x_canvas + 5, y_canvas + 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Display the canvas
            plt.figure(figsize=(8, 8))
            plt.imshow(canvas)
            plt.title(f"Keypoints visualization (no image)")
            plt.axis('off')
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
    path_check = check_paths()
    
    # Check for annotation files
    train_file = os.path.join(ds.annot_dir, 'train.h5')
    valid_file = os.path.join(ds.annot_dir, 'valid.h5')
    test_file = os.path.join(ds.annot_dir, 'test.h5')
    
    print(f"\nAvailable annotation files:")
    print(f"- Train: {'✓' if os.path.exists(train_file) else '✗'}")
    print(f"- Valid: {'✓' if os.path.exists(valid_file) else '✗'}")
    print(f"- Test: {'✓' if os.path.exists(test_file) else '✗'}")
    
    print("\nSelect visualization mode:")
    print("1. Full visualization (images + keypoints)")
    print("2. Annotations only (no images required)")
    
    mode_choice = input("\nEnter mode (1-2): ")
    
    print("\nSelect dataset to visualize:")
    print("1. Training set")
    print("2. Validation set")
    print("3. Test set")
    
    choice = input("\nEnter choice (1-3): ")
    
    dataset_map = {'1': 'train', '2': 'valid', '3': 'test'}
    dataset_type = dataset_map.get(choice, 'train')
    
    num_samples = input("Number of samples to visualize (default: 5): ")
    num_samples = int(num_samples) if num_samples.isdigit() else 5
    
    if mode_choice == '2':
        # Annotations only mode
        debug_annotations_only(dataset_type, num_samples)
    else:
        # Full visualization mode with images
        ignore_missing = input("Ignore missing images and continue? (y/n, default: y): ").lower() != 'n'
        debug_dataset_samples(dataset_type, num_samples, ignore_missing)
