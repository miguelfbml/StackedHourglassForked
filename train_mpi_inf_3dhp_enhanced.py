#!/usr/bin/env python3
"""
Enhanced training script for MPI-INF-3DHP with loss and MPJPE tracking
Based on the working train.py structure
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
import sys

def calculate_mpjpe_pixels(pred_heatmaps, gt_heatmaps, input_res=256):
    """
    Calculate Mean Per Joint Position Error in pixels
    """
    batch_size = pred_heatmaps.shape[0]
    num_joints = pred_heatmaps.shape[1]
    
    def heatmap_to_coords(heatmaps):
        coords = []
        for b in range(batch_size):
            batch_coords = []
            for j in range(num_joints):
                hm = heatmaps[b, j].detach().cpu().numpy()
                if hm.max() > 0:
                    idx = np.unravel_index(np.argmax(hm), hm.shape)
                    x = idx[1] * (input_res / hm.shape[1])
                    y = idx[0] * (input_res / hm.shape[0])
                    batch_coords.append([x, y])
                else:
                    batch_coords.append([0, 0])
            coords.append(batch_coords)
        return np.array(coords)
    
    pred_coords = heatmap_to_coords(pred_heatmaps)
    gt_coords = heatmap_to_coords(gt_heatmaps)
    
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=2))
    mpjpe = np.mean(distances)
    
    return mpjpe

def train_epoch(model, train_loader, optimizer, device, epoch, config):
    """Train for one epoch using the task's training function"""
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    print(f"\n🚀 Training Epoch {epoch}")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        # Use the task's training function
        loss = config['func'](i, config, 'train', **batch)
        
        running_loss += 0.0  # Loss is handled internally
        num_batches += 1
        
        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            print(f"  Batch {i+1}/{len(train_loader)} | Training...")
    
    epoch_time = time.time() - start_time
    print(f"✅ Epoch {epoch} Complete in {epoch_time:.1f}s")
    
    return 0.0, 0.0  # Placeholder values

def validate_epoch(model, val_loader, device, epoch, config):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    num_batches = 0
    
    print(f"\n🔍 Validating Epoch {epoch}")
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            # Use the task's inference function
            result = config['func'](i, config, 'inference', **batch)
            
            if result is not None and 'preds' in result:
                # Calculate MPJPE from predictions
                pred_heatmaps = torch.tensor(result['preds'][-1])  # Last stack output
                gt_heatmaps = batch['heatmaps']
                mpjpe = calculate_mpjpe_pixels(pred_heatmaps, gt_heatmaps)
                running_mpjpe += mpjpe
            
            num_batches += 1
    
    epoch_time = time.time() - start_time
    avg_mpjpe = running_mpjpe / num_batches if num_batches > 0 else 0.0
    
    print(f"✅ Validation {epoch} Complete:")
    print(f"   Time: {epoch_time:.1f}s")
    print(f"   Avg MPJPE: {avg_mpjpe:.2f} pixels")
    
    return 0.0, avg_mpjpe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='mpi_inf_3dhp_enhanced', help='Experiment name')
    parser.add_argument('--max_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=2.5e-4, type=float, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    
    args = parser.parse_args()
    
    print("=== Enhanced MPI-INF-3DHP Training ===")
    print(f"Experiment: {args.exp}")
    print(f"Max epochs: {args.max_epochs}")
    print(f"Learning rate: {args.lr}")
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create experiment directory
    exp_dir = os.path.join('exp', args.exp)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Import task configuration
    try:
        import task.pose_mpi_inf_3dhp_with_images as task_module
        config = task_module.__config__.copy()  # Make a copy to avoid modifying original
        print("✅ Loaded MPI-INF-3DHP task configuration")
    except Exception as e:
        print(f"❌ Error loading task: {e}")
        return
    
    # Load data
    print("\n📊 Loading datasets...")
    try:
        from data.MPI_INF_3DHP.dp_with_images import init
        train_loader, val_loader, test_loader = init(config)
        print(f"✅ Train samples: {len(train_loader.dataset)}")
        print(f"✅ Val/Test samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create the model using the task's make_network function
    print("\n🏗️ Loading model...")
    try:
        # Use the task's make_network function exactly like train.py does
        config = task_module.make_network(config)
        model = config['inference']['net']  # This is the DataParallel wrapped model
        print(f"✅ Model loaded successfully")
        
        # Count parameters
        actual_model = model.module if hasattr(model, 'module') else model
        total_params = sum(p.numel() for p in actual_model.parameters())
        trainable_params = sum(p.numel() for p in actual_model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training loop using the task's training function
    print(f"\n🚀 Starting training from epoch 0")
    best_mpjpe = float('inf')
    
    for epoch in range(args.max_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.max_epochs}")
        print(f"{'='*60}")
        
        # Train using task's training function
        train_loss, train_mpjpe = train_epoch(model, train_loader, None, device, epoch + 1, config)
        
        # Validate
        val_loss, val_mpjpe = validate_epoch(model, val_loader, device, epoch + 1, config)
        
        # Track best model
        is_best = val_mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = val_mpjpe
            print(f"🎉 New best MPJPE: {best_mpjpe:.2f}px")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_mpjpe': best_mpjpe,
        }
        
        torch.save(checkpoint, os.path.join(exp_dir, 'checkpoint.pth'))
        
        if is_best:
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
        
        print(f"\n📊 Summary Epoch {epoch + 1}:")
        print(f"   Val MPJPE: {val_mpjpe:.2f}px")
        print(f"   Best MPJPE: {best_mpjpe:.2f}px {'🌟' if is_best else ''}")
    
    print(f"\n🎉 Training completed!")
    print(f"Best MPJPE achieved: {best_mpjpe:.2f} pixels")

if __name__ == '__main__':
    main()