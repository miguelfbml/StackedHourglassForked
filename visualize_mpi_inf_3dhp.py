#!/usr/bin/env python3
"""
Visualization script for MPI-INF-3DHP dataset
Creates images with 2D keypoints overlaid to verify annotations match images
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import matplotlib.patches as patches

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def visualize_keypoints():
    print("=== MPI-INF-3DHP 2D Keypoints Visualization ===")
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Import the task and data provider
        import task.pose_mpi_inf_3dhp_with_images as task_module
        config = task_module.__config__
        
        from data.MPI_INF_3DHP.dp_with_images import Dataset
        
        print("✓ Creating dataset...")
        train_dataset = Dataset(config, train=True)
        test_dataset = Dataset(config, train=False)
        
        print(f"✓ Train dataset: {len(train_dataset)} samples")
        print(f"✓ Test dataset: {len(test_dataset)} samples")
        
        # MPI-INF-3DHP 17 keypoint names and connections
        keypoint_names = [
            'Head',         # 0
            'Neck',         # 1
            'RShoulder',    # 2
            'RElbow',       # 3
            'RWrist',       # 4
            'LShoulder',    # 5
            'LElbow',       # 6
            'LWrist',       # 7
            'RHip',         # 8
            'RKnee',        # 9
            'RAnkle',       # 10
            'LHip',         # 11
            'LKnee',        # 12
            'LAnkle',       # 13
            'Pelvis',       # 14
            'Thorax',       # 15
            'Upper Neck'    # 16
        ]
        
        # Define skeleton connections (bone pairs)
        skeleton_connections = [
            (0, 16),   # Head to Upper Neck
            (16, 15),  # Upper Neck to Thorax
            (15, 1),   # Thorax to Neck
            (1, 2),    # Neck to Right Shoulder
            (2, 3),    # Right Shoulder to Right Elbow
            (3, 4),    # Right Elbow to Right Wrist
            (1, 5),    # Neck to Left Shoulder
            (5, 6),    # Left Shoulder to Left Elbow
            (6, 7),    # Left Elbow to Left Wrist
            (15, 14),  # Thorax to Pelvis
            (14, 8),   # Pelvis to Right Hip
            (8, 9),    # Right Hip to Right Knee
            (9, 10),   # Right Knee to Right Ankle
            (14, 11),  # Pelvis to Left Hip
            (11, 12),  # Left Hip to Left Knee
            (12, 13),  # Left Knee to Left Ankle
        ]
        
        # Colors for different body parts
        colors = {
            'head': (255, 0, 0),      # Red
            'torso': (0, 255, 0),     # Green
            'right_arm': (255, 255, 0), # Yellow
            'left_arm': (0, 255, 255),  # Cyan
            'right_leg': (255, 0, 255), # Magenta
            'left_leg': (0, 0, 255),    # Blue
        }
        
        # Assign colors to keypoints
        keypoint_colors = [
            colors['head'],      # 0: Head
            colors['torso'],     # 1: Neck
            colors['right_arm'], # 2: RShoulder
            colors['right_arm'], # 3: RElbow
            colors['right_arm'], # 4: RWrist
            colors['left_arm'],  # 5: LShoulder
            colors['left_arm'],  # 6: LElbow
            colors['left_arm'],  # 7: LWrist
            colors['right_leg'], # 8: RHip
            colors['right_leg'], # 9: RKnee
            colors['right_leg'], # 10: RAnkle
            colors['left_leg'],  # 11: LHip
            colors['left_leg'],  # 12: LKnee
            colors['left_leg'],  # 13: LAnkle
            colors['torso'],     # 14: Pelvis
            colors['torso'],     # 15: Thorax
            colors['head'],      # 16: Upper Neck
        ]
        
        def draw_keypoints_on_image(img, keypoints_2d, title="", save_path=None):
            """Draw 2D keypoints on image"""
            img_vis = img.copy()
            
            # Draw skeleton connections first (lines)
            for start_idx, end_idx in skeleton_connections:
                if start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d):
                    start_pt = keypoints_2d[start_idx]
                    end_pt = keypoints_2d[end_idx]
                    
                    # Only draw if both points are valid
                    if len(start_pt) >= 2 and len(end_pt) >= 2:
                        cv2.line(img_vis, 
                                (int(start_pt[0]), int(start_pt[1])), 
                                (int(end_pt[0]), int(end_pt[1])), 
                                (255, 255, 255), 2)  # White lines
            
            # Draw keypoints (circles)
            for i, (x, y) in enumerate(keypoints_2d):
                if i < len(keypoint_colors):
                    color = keypoint_colors[i]
                    cv2.circle(img_vis, (int(x), int(y)), 5, color, -1)
                    
                    # Add keypoint number
                    cv2.putText(img_vis, str(i), (int(x) + 8, int(y) + 8), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add title
            if title:
                cv2.putText(img_vis, title, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if save_path:
                cv2.imwrite(save_path, img_vis)
                print(f"Saved: {save_path}")
            
            return img_vis
        
        # Visualize a few training samples
        print("\n📸 Visualizing training samples...")
        for i in range(min(10, len(train_dataset))):
            try:
                # Get raw sample data (not processed tensor)
                sample = train_dataset.samples[i]
                
                # Load original image
                img = cv2.imread(sample['img_path'])
                if img is None:
                    print(f"Could not load image: {sample['img_path']}")
                    continue
                
                # Get 2D keypoints (original coordinates)
                joint_2d = sample['joint_2d']
                
                # Create title with sample info
                title = f"Train {i}: {sample['subject']} {sample['sequence']} cam{sample['camera']} frame{sample['frame_idx']}"
                
                # Draw keypoints on original image
                save_path = os.path.join(output_dir, f"train_sample_{i:03d}_original.jpg")
                img_vis = draw_keypoints_on_image(img, joint_2d, title, save_path)
                
                # Also create a resized version (like what the model sees)
                img_resized = cv2.resize(img, (256, 256))
                
                # Scale keypoints to resized image
                scale_x = 256 / img.shape[1]
                scale_y = 256 / img.shape[0]
                joint_2d_scaled = joint_2d.copy()
                joint_2d_scaled[:, 0] *= scale_x
                joint_2d_scaled[:, 1] *= scale_y
                
                title_resized = title + " (256x256)"
                save_path_resized = os.path.join(output_dir, f"train_sample_{i:03d}_resized.jpg")
                draw_keypoints_on_image(img_resized, joint_2d_scaled, title_resized, save_path_resized)
                
            except Exception as e:
                print(f"Error processing training sample {i}: {e}")
        
        # Visualize a few test samples
        print("\n📸 Visualizing test samples...")
        for i in range(min(5, len(test_dataset))):
            try:
                # Get raw sample data
                sample = test_dataset.samples[i]
                
                # Load original image
                img = cv2.imread(sample['img_path'])
                if img is None:
                    print(f"Could not load image: {sample['img_path']}")
                    continue
                
                # Get 2D keypoints
                joint_2d = sample['joint_2d']
                
                # Create title
                title = f"Test {i}: {sample['subject']} frame{sample['frame_idx']}"
                
                # Draw keypoints
                save_path = os.path.join(output_dir, f"test_sample_{i:03d}_original.jpg")
                draw_keypoints_on_image(img, joint_2d, title, save_path)
                
            except Exception as e:
                print(f"Error processing test sample {i}: {e}")
        
        # Create a legend showing keypoint meanings
        print("\n🎨 Creating keypoint legend...")
        legend_img = np.ones((600, 400, 3), dtype=np.uint8) * 255  # White background
        
        y_start = 30
        for i, (name, color) in enumerate(zip(keypoint_names, keypoint_colors)):
            y = y_start + i * 30
            # Draw colored circle
            cv2.circle(legend_img, (30, y), 8, color, -1)
            # Add keypoint number
            cv2.putText(legend_img, str(i), (25, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            # Add keypoint name
            cv2.putText(legend_img, name, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        legend_path = os.path.join(output_dir, "keypoint_legend.jpg")
        cv2.imwrite(legend_path, legend_img)
        print(f"Saved legend: {legend_path}")
        
        print(f"\n✅ Visualization complete!")
        print(f"📁 Check the '{output_dir}' folder for:")
        print(f"   - Training samples with keypoints overlaid")
        print(f"   - Test samples with keypoints overlaid")
        print(f"   - Both original and resized (256x256) versions")
        print(f"   - Keypoint legend showing the 17 keypoint meanings")
        print(f"\n🔍 Inspect the images to verify that:")
        print(f"   - Keypoints are positioned correctly on body parts")
        print(f"   - Skeleton connections make anatomical sense")
        print(f"   - Different subjects/cameras show consistent annotation quality")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    visualize_keypoints()