"""
Image processing utilities for pose estimation
Contains functions for cropping, transforming, and augmenting images
"""

import cv2
import numpy as np
import math

def crop(img, center, scale, res):
    """
    Crop image around center with given scale to target resolution
    
    Args:
        img: Input image (H, W, C)
        center: Center point (x, y)
        scale: Scale factor
        res: Target resolution (width, height)
    
    Returns:
        Cropped and resized image
    """
    if img is None:
        return np.zeros((res[1], res[0], 3), dtype=np.uint8)
    
    # Calculate crop size
    h, w = img.shape[:2]
    center = np.array(center)
    
    # Scale determines the crop size
    crop_size = int(scale * 200)  # 200 is a typical reference size
    
    # Calculate crop boundaries
    x1 = max(0, int(center[0] - crop_size // 2))
    y1 = max(0, int(center[1] - crop_size // 2))
    x2 = min(w, int(center[0] + crop_size // 2))
    y2 = min(h, int(center[1] + crop_size // 2))
    
    # Crop
    cropped = img[y1:y2, x1:x2]
    
    # Resize to target resolution
    if cropped.size > 0:
        resized = cv2.resize(cropped, res, interpolation=cv2.INTER_LINEAR)
    else:
        resized = np.zeros((res[1], res[0], 3), dtype=np.uint8)
    
    return resized

def transform(point, center, scale, res):
    """
    Transform point from original image coordinates to cropped image coordinates
    
    Args:
        point: Point coordinates (x, y)
        center: Center of crop (x, y)
        scale: Scale factor
        res: Target resolution (width, height)
    
    Returns:
        Transformed point coordinates
    """
    point = np.array(point)
    center = np.array(center)
    
    # Calculate crop size
    crop_size = scale * 200
    
    # Transform to crop coordinates
    crop_point = point - center + crop_size / 2
    
    # Scale to target resolution
    scale_factor = res[0] / crop_size
    transformed_point = crop_point * scale_factor
    
    return transformed_point

def get_transform(center, scale, res, rot=0):
    """
    Get transformation matrix for affine transformation
    
    Args:
        center: Center point (x, y)
        scale: Scale factor
        res: Target resolution (width, height)
        rot: Rotation angle in degrees
    
    Returns:
        3x3 transformation matrix
    """
    center = np.array(center)
    
    # Calculate transformation
    h = scale * 200
    t = np.eye(3)
    
    # Translation to center
    t[0, 2] = res[0] * 0.5
    t[1, 2] = res[1] * 0.5
    
    # Scale
    scale_factor = res[0] / h
    t[0, 0] = scale_factor
    t[1, 1] = scale_factor
    
    # Rotation
    if rot != 0:
        rot_rad = rot * math.pi / 180
        cos_rot = math.cos(rot_rad)
        sin_rot = math.sin(rot_rad)
        
        rot_mat = np.array([
            [cos_rot, -sin_rot, 0],
            [sin_rot, cos_rot, 0],
            [0, 0, 1]
        ])
        t = np.dot(t, rot_mat)
    
    # Translation from original center
    t[0, 2] -= center[0] * scale_factor
    t[1, 2] -= center[1] * scale_factor
    
    return t

def kpt_affine(kpt, mat):
    """
    Apply affine transformation to keypoints
    
    Args:
        kpt: Keypoints array of shape (..., 2)
        mat: 2x3 transformation matrix
    
    Returns:
        Transformed keypoints
    """
    kpt = np.array(kpt)
    shape = kpt.shape
    kpt = kpt.reshape(-1, 2)
    
    # Add homogeneous coordinate
    ones = np.ones((kpt.shape[0], 1))
    kpt_homo = np.concatenate([kpt, ones], axis=1)
    
    # Apply transformation
    kpt_transformed = np.dot(kpt_homo, mat.T)
    
    return kpt_transformed.reshape(shape)

def im_to_torch(img):
    """
    Convert image from numpy to torch format
    
    Args:
        img: Image array (H, W, C)
    
    Returns:
        Torch tensor (C, H, W)
    """
    img = img.transpose(2, 0, 1)  # HWC to CHW
    img = img.astype(np.float32) / 255.0
    return img

def torch_to_im(img):
    """
    Convert image from torch to numpy format
    
    Args:
        img: Torch tensor (C, H, W)
    
    Returns:
        Numpy array (H, W, C)
    """
    img = img.transpose(1, 2, 0)  # CHW to HWC
    img = (img * 255.0).astype(np.uint8)
    return img

def color_normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Normalize image colors
    
    Args:
        img: Image array (H, W, C) in range [0, 1]
        mean: Mean values for each channel
        std: Standard deviation for each channel
    
    Returns:
        Normalized image
    """
    mean = np.array(mean).reshape(1, 1, 3)
    std = np.array(std).reshape(1, 1, 3)
    
    img = (img - mean) / std
    return img

def gaussian_blur(img, kernel_size=5):
    """
    Apply Gaussian blur to image
    
    Args:
        img: Input image
        kernel_size: Size of Gaussian kernel
    
    Returns:
        Blurred image
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
