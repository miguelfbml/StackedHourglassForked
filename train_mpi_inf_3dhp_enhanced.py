#!/usr/bin/env python3
"""
Enhanced training script for MPI-INF-3DHP with loss and MPJPE tracking
"""

import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import time
from torch.utils.tensorboard import SummaryWriter

# Import your existing components
import ref
from utils.utils import adjust_learning_rate
from utils.eval import Evaluation

def calculate_mpjpe_pixels(pred_heatmaps, gt_heatmaps, input_res=256):
    """
    Calculate Mean Per Joint Position Error in pixels
    Args:
        pred_heatmaps: Predicted heatmaps [B, 17, 64, 64]
        gt_heatmaps: Ground truth heatmaps [B, 17, 64, 64]
        input_res: Input image resolution (256)
    Returns:
        MPJPE in pixels
    """
    batch_size = pred_heatmaps.shape[0]
    num_joints = pred_heatmaps.shape[1]
    
    # Convert heatmaps to keypoint coordinates
    def heatmap_to_coords(heatmaps):
        coords = []
        for b in range(batch_size):
            batch_coords = []
            for j in range(num_joints):
                hm = heatmaps[b, j].detach().cpu().numpy()
                if hm.max() > 0:
                    # Find maximum location
                    idx = np.unravel_index(np.argmax(hm), hm.shape)
                    # Scale to input resolution
                    x = idx[1] * (input_res / hm.shape[1])
                    y = idx[0] * (input_res / hm.shape[0])
                    batch_coords.append([x, y])
                else:
                    batch_coords.append([0, 0])
            coords.append(batch_coords)
        return np.array(coords)  # [B, 17, 2]
    
    pred_coords = heatmap_to_coords(pred_heatmaps)
    gt_coords = heatmap_to_coords(gt_heatmaps)
    
    # Calculate euclidean distances
    distances = np.sqrt(np.sum((pred_coords - gt_coords) ** 2, axis=2))  # [B, 17]
    mpjpe = np.mean(distances)
    
    return mpjpe

def train_epoch(model, train_loader, optimizer, criterion, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_mpjpe = 0.0
    num_batches = 0
    
    print(f"\n🚀 Training Epoch {epoch}")
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        imgs = batch['imgs'].to(device)
        heatmaps = batch['heatmaps'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(imgs)
        
        # Calculate loss (outputs is list of predictions from each hourglass)
        total_loss = 0
        for output in outputs:
            total_loss += criterion(output, heatmaps)
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Calculate MPJPE using the final prediction
        final_pred = outputs[-1]  # Last hourglass output
        mpjpe = calculate_mpjpe_pixels(final_pred, heatmaps)
        
        running_loss += total_loss.item()
        running_mpjpe += mpjpe
        num_batches += 1
        
        # Print progress every 100 batches
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / num_batches
            avg_mpjpe = running_mpjpe / num_batches
            print(f"  Batch {i+1}/{len(train_loader)} | Loss: {avg_loss:.4f} | MPJPE: {avg_mpjpe:.2f}px")
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / num_batches
    avg_mpjpe = running_mpjpe / num_batches
    
    print(f"✅ Epoch {epoch} Complete:")
    print(f"   Time: {epoch_time:.1f}s")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Avg MPJPE: {avg_mpjpe:.2f} pixels")
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Train/Loss', avg_loss, epoch)
        writer.add_scalar('Train/MPJPE_pixels', avg_mpjpe, epoch)
    
    return avg_loss, avg_mpjpe

def validate_epoch(model, val_loader, criterion, device, epoch, writer=None):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_mpjpe = 0.0
    num_batches = 0
    
    print(f"\n🔍 Validating Epoch {epoch}")
    start_time = time.time()
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            imgs = batch['imgs'].to(device)
            heatmaps = batch['heatmaps'].to(device)
            
            # Forward pass
            outputs = model(imgs)
            
            # Calculate loss
            total_loss = 0
            for output in outputs:
                total_loss += criterion(output, heatmaps)
            
            # Calculate MPJPE
            final_pred = outputs[-1]
            mpjpe = calculate_mpjpe_pixels(final_pred, heatmaps)
            
            running_loss += total_loss.item()
            running_mpjpe += mpjpe
            num_batches += 1
    
    epoch_time = time.time() - start_time
    avg_loss = running_loss / num_batches
    avg_mpjpe = running_mpjpe / num_batches
    
    print(f"✅ Validation {epoch} Complete:")
    print(f"   Time: {epoch_time:.1f}s")
    print(f"   Avg Loss: {avg_loss:.4f}")
    print(f"   Avg MPJPE: {avg_mpjpe:.2f} pixels")
    
    # Log to tensorboard
    if writer:
        writer.add_scalar('Val/Loss', avg_loss, epoch)
        writer.add_scalar('Val/MPJPE_pixels', avg_mpjpe, epoch)
    
    return avg_loss, avg_mpjpe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default='mpi_inf_3dhp_enhanced', help='Experiment name')
    parser.add_argument('--max_epochs', default=50, type=int, help='Number of epochs')
    parser.add_argument('--lr', default=2.5e-4, type=float, help='Learning rate')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--eval_only', action='store_true', help='Only evaluate')
    parser.add_argument('--data_root', default='data/MPI_INF_3DHP/motion3d', help='Data root')
    parser.add_argument('--mpi_dataset_root', default='/nas-ctm01/datasets/public/mpi_inf_3dhp', help='MPI dataset root')
    
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
    
    # Set up tensorboard
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Import task configuration
    try:
        import task.pose_mpi_inf_3dhp_with_images as task_module
        config = task_module.__config__
        config['data_root'] = args.data_root
        config['mpi_dataset_root'] = args.mpi_dataset_root
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
    
    # Load model
    print("\n🏗️ Loading model...")
    try:
        model = task_module.model
        model = model.to(device)
        print(f"✅ Model loaded: {model.__class__.__name__}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Set up optimizer and loss
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_mpjpe = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(exp_dir, 'checkpoint.pth')
        if os.path.exists(checkpoint_path):
            print(f"📂 Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_mpjpe = checkpoint.get('best_mpjpe', float('inf'))
            print(f"✅ Resumed from epoch {start_epoch}, best MPJPE: {best_mpjpe:.2f}px")
        else:
            print(f"❌ No checkpoint found at {checkpoint_path}")
    
    # Evaluation only
    if args.eval_only:
        print("\n🔍 Evaluation only mode")
        val_loss, val_mpjpe = validate_epoch(model, val_loader, criterion, device, start_epoch, writer)
        return
    
    # Training loop
    print(f"\n🚀 Starting training from epoch {start_epoch}")
    
    for epoch in range(start_epoch, args.max_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{args.max_epochs}")
        print(f"{'='*60}")
        
        # Adjust learning rate
        lr = adjust_learning_rate(optimizer, epoch, args.lr, schedule=[30, 40])
        print(f"Learning rate: {lr}")
        
        # Train
        train_loss, train_mpjpe = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1, writer)
        
        # Validate
        val_loss, val_mpjpe = validate_epoch(model, val_loader, criterion, device, epoch + 1, writer)
        
        # Save checkpoint
        is_best = val_mpjpe < best_mpjpe
        if is_best:
            best_mpjpe = val_mpjpe
            print(f"🎉 New best MPJPE: {best_mpjpe:.2f}px")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_mpjpe': train_mpjpe,
            'val_mpjpe': val_mpjpe,
            'best_mpjpe': best_mpjpe,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(exp_dir, 'checkpoint.pth'))
        
        # Save best model
        if is_best:
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(exp_dir, f'checkpoint_epoch_{epoch + 1}.pth'))
        
        print(f"\n📊 Summary Epoch {epoch + 1}:")
        print(f"   Train Loss: {train_loss:.4f} | Train MPJPE: {train_mpjpe:.2f}px")
        print(f"   Val Loss:   {val_loss:.4f} | Val MPJPE:   {val_mpjpe:.2f}px")
        print(f"   Best MPJPE: {best_mpjpe:.2f}px {'🌟' if is_best else ''}")
    
    print(f"\n🎉 Training completed!")
    print(f"Best MPJPE achieved: {best_mpjpe:.2f} pixels")
    
    writer.close()

if __name__ == '__main__':
    main()