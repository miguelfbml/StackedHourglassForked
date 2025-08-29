import cv2
import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Change to project root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(project_root)
print(f"Changed working directory to: {os.getcwd()}")

# Create output directory if it doesn't exist
output_dir = os.path.join(project_root, "output")
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory: {output_dir}")

from utils.img import crop, transform
import data.MPII.ref as ds

def visualize_keypoints(image, keypoints, visibility=None, title=None, save_path=None):
    """
    Visualize keypoints on an image and save to file.
    
    Args:
        image: Input RGB image (numpy array)
        keypoints: Array of shape (16, 2) with keypoint coordinates
        visibility: Array of shape (16,) with visibility flags (0=invisible, 1=visible)
        title: Title for the plot
        save_path: Path to save the annotated image (if None, won't save)
    """
    # Make a copy of the image to draw on
    img_draw = image.copy()
    
    # Ensure image is in correct format
    if img_draw.dtype != np.uint8:
        print(f"Warning: Image has dtype {img_draw.dtype}, converting to uint8")
        if img_draw.max() <= 1.0:
            img_draw = (img_draw * 255).astype(np.uint8)
        else:
            img_draw = img_draw.astype(np.uint8)
    
    print(f"Image shape: {img_draw.shape}, dtype: {img_draw.dtype}, min: {img_draw.min()}, max: {img_draw.max()}")
    
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
    try:
        img_bgr = cv2.cvtColor(img_draw, cv2.COLOR_RGB2BGR)
    except cv2.error as e:
        print(f"Error converting image to BGR: {e}")
        print(f"Image info: shape={img_draw.shape}, dtype={img_draw.dtype}")
        return img_draw  # Return original if conversion fails
    
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
    
    # Save the annotated image if a save path is provided
    if save_path:
        # Add text with image title if provided
        if title:
            # Calculate position for title text (bottom of the image)
            text_pos = (10, img_bgr.shape[0] - 20)
            cv2.putText(img_bgr, title, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Save the BGR image directly with OpenCV
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img_bgr)
        print(f"Saved annotated image to: {save_path}")
    
    # Convert back to RGB for matplotlib display
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def visualize_dataset(dataset_type='train', num_samples=30, start_idx=0, save_images=True):
    """
    Visualize samples from the dataset with keypoints overlaid and save to output directory.
    
    Args:
        dataset_type: 'train' or 'valid'
        num_samples: Number of samples to visualize
        start_idx: Starting index in the dataset
        save_images: Whether to save the images to the output directory
    """
    h5_file = os.path.join(ds.annot_dir, f'{dataset_type}.h5')
    
    # Define joint names for reference
    joint_names = ["Right ankle", "Right knee", "Right hip", "Left hip", "Left knee", "Left ankle", 
                  "Pelvis", "Thorax", "Neck", "Head", "Right wrist", "Right elbow", 
                  "Right shoulder", "Left shoulder", "Left elbow", "Left wrist"]
    
    with h5py.File(h5_file, 'r') as f:
        total_samples = f['imgname'].shape[0]
        print(f"Total {dataset_type} samples: {total_samples}")
        
        end_idx = min(start_idx + num_samples, total_samples)
        print(f"Processing {end_idx - start_idx} samples...")
        
        # Keep track of how many images we've processed
        processed = 0
        
        for i in range(start_idx, end_idx):
            try:
                # Load image
                img_name = f['imgname'][i].decode('UTF-8')
                img_path = os.path.join(ds.img_dir, img_name)
                print(f"Processing image {processed+1}/{end_idx-start_idx}: {img_path}")
                
                # Read the image
                orig_img = cv2.imread(img_path)
                if orig_img is None:
                    print(f"Could not read image: {img_path}")
                    continue
                
                # Convert from BGR to RGB
                orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                
                # Ensure original image is in correct format (uint8)
                if orig_img.dtype != np.uint8:
                    if orig_img.max() <= 1.0:
                        # If values are in [0,1] range, scale to [0,255]
                        orig_img = (orig_img * 255).astype(np.uint8)
                    else:
                        # If values are already in [0,255] range but wrong type
                        orig_img = orig_img.astype(np.uint8)
                
                # Get center and scale
                center = f['center'][i]
                scale = f['scale'][i]
                
                # Get keypoints and visibility
                keypoints = f['part'][i]  # Shape: (16, 2)
                visibility = f['visible'][i]  # Shape: (16,)
                
                # Create output paths
                base_filename = os.path.splitext(os.path.basename(img_name))[0]
                orig_save_path = os.path.join(output_dir, f"{dataset_type}_{base_filename}_original.jpg") if save_images else None
                crop_save_path = os.path.join(output_dir, f"{dataset_type}_{base_filename}_cropped.jpg") if save_images else None
                
                # Process original image with keypoints
                orig_title = f"{dataset_type.capitalize()} {i} - Original"
                annotated_orig = visualize_keypoints(orig_img, keypoints, visibility, orig_title, orig_save_path)
                
                # Get the normalized crop (as used in training)
                input_res = 256  # Assuming input resolution is 256x256
                cropped_img = crop(orig_img, center, scale, (input_res, input_res))
                
                # Ensure cropped image is in correct format (uint8)
                if cropped_img.dtype != np.uint8:
                    if cropped_img.max() <= 1.0:
                        # If values are in [0,1] range, scale to [0,255]
                        cropped_img = (cropped_img * 255).astype(np.uint8)
                    else:
                        # If values are already in [0,255] range but wrong type
                        cropped_img = cropped_img.astype(np.uint8)
                
                # Transform keypoints to cropped image space
                cropped_keypoints = np.copy(keypoints)
                for j in range(keypoints.shape[0]):
                    if visibility[j] > 0:
                        cropped_keypoints[j] = transform(keypoints[j], center, scale, (input_res, input_res))
                
                # Process cropped image with keypoints
                crop_title = f"{dataset_type.capitalize()} {i} - Cropped"
                annotated_crop = visualize_keypoints(cropped_img, cropped_keypoints, visibility, crop_title, crop_save_path)
                
                # Save keypoint information to a text file
                if save_images:
                    info_path = os.path.join(output_dir, f"{dataset_type}_{base_filename}_keypoints.txt")
                    with open(info_path, 'w') as info_file:
                        info_file.write(f"Image: {img_name}\n")
                        info_file.write(f"Center: {center}, Scale: {scale}\n\n")
                        info_file.write("Keypoint details:\n")
                        info_file.write("Index | Joint Name       | Position (x, y)   | Visibility\n")
                        info_file.write("-" * 60 + "\n")
                        for j, name in enumerate(joint_names):
                            vis_status = "Visible" if visibility[j] > 0 else "Not visible"
                            info_file.write(f"{j:5d} | {name:16s} | ({keypoints[j][0]:6.1f}, {keypoints[j][1]:6.1f}) | {vis_status}\n")
                
                processed += 1
            
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
        
        print(f"Processed {processed} images successfully")

def main():
    print("Processing 30 training images with keypoint visualization...")
    
    # Check if directories exist
    if not os.path.exists(ds.img_dir):
        print(f"Error: Image directory not found: {ds.img_dir}")
        return
    if not os.path.exists(ds.annot_dir):
        print(f"Error: Annotation directory not found: {ds.annot_dir}")
        return
    
    print(f"Output directory: {output_dir}")
    
    # Process 30 training samples
    visualize_dataset('train', num_samples=30, start_idx=0, save_images=True)
    print("Processing complete!")

if __name__ == "__main__":
    main()
