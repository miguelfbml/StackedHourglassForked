import cv2
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.img import crop
import data.MPII.ref as ds

def visualize_keypoints(image, keypoints, visibility=None, title=None):
    """
    Visualize keypoints on an image.
    
    Args:
        image: Input RGB image (numpy array)
        keypoints: Array of shape (16, 2) with keypoint coordinates
        visibility: Array of shape (16,) with visibility flags (0=invisible, 1=visible)
        title: Title for the plot
    """
    # Make a copy of the image to draw on
    img_draw = image.copy()
    
    # Define colors for each keypoint (BGR format for OpenCV)
    colors = [
        (255, 0, 0),   # Blue - right ankle
        (255, 85, 0),  # Blue-cyan - right knee
        (255, 170, 0), # Cyan - right hip
        (255, 255, 0), # Yellow - left hip
        (170, 255, 0), # Yellow-green - left knee
        (85, 255, 0),  # Green - left ankle
        (0, 255, 0),   # Green - pelvis
        (0, 255, 85),  # Green-cyan - thorax
        (0, 255, 170), # Cyan - neck
        (0, 255, 255), # Cyan - head
        (0, 170, 255), # Cyan-blue - right wrist
        (0, 85, 255),  # Blue - right elbow
        (0, 0, 255),   # Blue - right shoulder
        (85, 0, 255),  # Blue-magenta - left shoulder
        (170, 0, 255), # Magenta - left elbow
        (255, 0, 255)  # Magenta - left wrist
    ]
    
    # Define the keypoint pairs for skeleton lines
    skeleton_pairs = [
        # Legs
        (0, 1),  # right ankle - right knee
        (1, 2),  # right knee - right hip
        (3, 4),  # left hip - left knee
        (4, 5),  # left knee - left ankle
        # Hips
        (2, 6),  # right hip - pelvis
        (3, 6),  # left hip - pelvis
        # Spine
        (6, 7),  # pelvis - thorax
        (7, 8),  # thorax - neck
        (8, 9),  # neck - head
        # Arms
        (10, 11),  # right wrist - right elbow
        (11, 12),  # right elbow - right shoulder
        (12, 7),   # right shoulder - thorax
        (7, 13),   # thorax - left shoulder
        (13, 14),  # left shoulder - left elbow
        (14, 15)   # left elbow - left wrist
    ]
    
    # Convert image to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    
    # Draw skeleton lines first
    for pair in skeleton_pairs:
        p1, p2 = pair
        if visibility is None or (visibility[p1] > 0 and visibility[p2] > 0):
            pt1 = (int(keypoints[p1, 0]), int(keypoints[p1, 1]))
            pt2 = (int(keypoints[p2, 0]), int(keypoints[p2, 1]))
            cv2.line(img_bgr, pt1, pt2, (255, 255, 255), 2)
    
    # Draw keypoints
    for i, (x, y) in enumerate(keypoints):
        if visibility is None or visibility[i] > 0:
            cv2.circle(img_bgr, (int(x), int(y)), 5, colors[i], -1)
    
    # Convert back to RGB for matplotlib
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Display using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_dataset(dataset_type='train', num_samples=5, start_idx=0):
    """
    Visualize samples from the dataset with keypoints overlaid.
    
    Args:
        dataset_type: 'train' or 'valid'
        num_samples: Number of samples to visualize
        start_idx: Starting index in the dataset
    """
    h5_file = os.path.join(ds.annot_dir, f'{dataset_type}.h5')
    
    with h5py.File(h5_file, 'r') as f:
        total_samples = f['imgname'].shape[0]
        print(f"Total {dataset_type} samples: {total_samples}")
        
        end_idx = min(start_idx + num_samples, total_samples)
        
        for i in range(start_idx, end_idx):
            # Load image
            img_name = f['imgname'][i].decode('UTF-8')
            img_path = os.path.join(ds.img_dir, img_name)
            print(f"Loading image: {img_path}")
            
            # Read the image
            orig_img = cv2.imread(img_path)
            if orig_img is None:
                print(f"Could not read image: {img_path}")
                continue
            
            # Convert from BGR to RGB
            orig_img = orig_img[:, :, ::-1]
            
            # Get center and scale
            center = f['center'][i]
            scale = f['scale'][i]
            print(f"Center: {center}, Scale: {scale}")
            
            # Get keypoints and visibility
            keypoints = f['part'][i]  # Shape: (16, 2)
            visibility = f['visible'][i]  # Shape: (16,)
            
            # Display original image with keypoints
            print(f"Original image shape: {orig_img.shape}")
            visualize_keypoints(orig_img, keypoints, visibility, 
                               f"{dataset_type.capitalize()} Image {i} - Original with Keypoints")
            
            # Get the normalized crop (as used in training)
            input_res = 256  # Assuming input resolution is 256x256
            cropped_img = crop(orig_img, center, scale, (input_res, input_res))
            
            # Transform keypoints to cropped image space
            from utils.img import transform
            cropped_keypoints = np.copy(keypoints)
            for j in range(keypoints.shape[0]):
                if visibility[j] > 0:
                    cropped_keypoints[j] = transform(keypoints[j], center, scale, (input_res, input_res))
            
            visualize_keypoints(cropped_img, cropped_keypoints, visibility,
                               f"{dataset_type.capitalize()} Image {i} - Cropped with Keypoints")
            
            print(f"Keypoint names and positions:")
            joint_names = ["Right ankle", "Right knee", "Right hip", "Left hip", "Left knee", "Left ankle", 
                          "Pelvis", "Thorax", "Neck", "Head", "Right wrist", "Right elbow", 
                          "Right shoulder", "Left shoulder", "Left elbow", "Left wrist"]
            
            for j, name in enumerate(joint_names):
                status = "Visible" if visibility[j] > 0 else "Not visible"
                print(f"{j}: {name} - Position: ({keypoints[j, 0]:.1f}, {keypoints[j, 1]:.1f}) - {status}")
            
            print("-" * 50)

def main():
    print("MPII Dataset Keypoint Visualization")
    print("="*50)
    
    # Check if directories exist
    print(f"Checking directories:")
    print(f"Image directory: {ds.img_dir} - {'Exists' if os.path.exists(ds.img_dir) else 'MISSING'}")
    print(f"Annotation directory: {ds.annot_dir} - {'Exists' if os.path.exists(ds.annot_dir) else 'MISSING'}")
    
    # Check if annotation files exist
    train_file = os.path.join(ds.annot_dir, 'train.h5')
    valid_file = os.path.join(ds.annot_dir, 'valid.h5')
    test_file = os.path.join(ds.annot_dir, 'test.h5')
    
    print(f"Train annotation file: {train_file} - {'Exists' if os.path.exists(train_file) else 'MISSING'}")
    print(f"Valid annotation file: {valid_file} - {'Exists' if os.path.exists(valid_file) else 'MISSING'}")
    print(f"Test annotation file: {test_file} - {'Exists' if os.path.exists(test_file) else 'MISSING'}")
    
    # Show keypoint visualization options
    print("\nChoose an option:")
    print("1. Visualize training samples")
    print("2. Visualize validation samples")
    print("3. Visualize both")
    
    choice = input("Enter your choice (1-3): ")
    
    num_samples = int(input("How many samples to visualize? (default: 3): ") or "3")
    start_idx = int(input("Starting index? (default: 0): ") or "0")
    
    if choice == '1' or choice == '3':
        print("\nVisualizing training samples:")
        visualize_dataset('train', num_samples, start_idx)
    
    if choice == '2' or choice == '3':
        print("\nVisualizing validation samples:")
        visualize_dataset('valid', num_samples, start_idx)

if __name__ == "__main__":
    main()
