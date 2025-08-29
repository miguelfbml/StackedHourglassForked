import torch
import numpy as np

def calculate_mpjpe(predicted, target):
    """
    Calculate Mean Per Joint Position Error (MPJPE) in pixels
    
    Args:
        predicted: predicted heatmaps (B, 17, H, W)
        target: target heatmaps (B, 17, H, W)
    
    Returns:
        mpjpe: mean per joint position error in pixels
    """
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1]
    
    # Convert heatmaps to 2D coordinates
    pred_coords = heatmaps_to_coords(predicted)  # (B, 17, 2)
    target_coords = heatmaps_to_coords(target)   # (B, 17, 2)
    
    # Calculate L2 distance for each joint
    distances = torch.sqrt(torch.sum((pred_coords - target_coords) ** 2, dim=2))  # (B, 17)
    
    # Calculate MPJPE
    mpjpe = torch.mean(distances)
    
    return mpjpe.item()

def heatmaps_to_coords(heatmaps):
    """
    Convert heatmaps to 2D coordinates using argmax
    
    Args:
        heatmaps: (B, 17, H, W)
    
    Returns:
        coords: (B, 17, 2) coordinates in (x, y) format
    """
    batch_size, num_joints, height, width = heatmaps.shape
    
    # Reshape to (B, 17, H*W)
    heatmaps_reshaped = heatmaps.view(batch_size, num_joints, -1)
    
    # Find max indices
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)  # (B, 17)
    
    # Convert to x, y coordinates
    coords = torch.zeros(batch_size, num_joints, 2, device=heatmaps.device)
    coords[:, :, 0] = max_indices % width   # x coordinate
    coords[:, :, 1] = max_indices // width  # y coordinate
    
    return coords

def calculate_pck(predicted, target, threshold=2.0):
    """
    Calculate Percentage of Correct Keypoints (PCK) at given threshold
    
    Args:
        predicted: predicted heatmaps (B, 17, H, W)
        target: target heatmaps (B, 17, H, W)
        threshold: distance threshold in pixels
    
    Returns:
        pck: percentage of correct keypoints
    """
    batch_size = predicted.shape[0]
    num_joints = predicted.shape[1]
    
    # Convert heatmaps to 2D coordinates
    pred_coords = heatmaps_to_coords(predicted)  # (B, 17, 2)
    target_coords = heatmaps_to_coords(target)   # (B, 17, 2)
    
    # Calculate L2 distance for each joint
    distances = torch.sqrt(torch.sum((pred_coords - target_coords) ** 2, dim=2))  # (B, 17)
    
    # Count correct predictions (within threshold)
    correct = (distances <= threshold).float()
    
    # Calculate PCK
    pck = torch.mean(correct)
    
    return pck.item() * 100  # Return as percentage
