#!/usr/bin/env python3
"""
Training monitoring script for MPI-INF-3DHP StackedHourglass
"""

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def parse_log_file(log_path):
    """Parse training log file to extract loss values"""
    if not os.path.exists(log_path):
        print(f"Log file not found: {log_path}")
        return [], []
    
    iterations = []
    losses = []
    
    with open(log_path, 'r') as f:
        for line in f:
            # Look for lines like: "200:  combined_hm_loss: 0.0010246606543660164"
            match = re.search(r'(\d+):\s+combined_hm_loss:\s+([\d\.]+)', line)
            if match:
                iteration = int(match.group(1))
                loss = float(match.group(2))
                iterations.append(iteration)
                losses.append(loss)
    
    return iterations, losses

def plot_training_progress(exp_name='mpi_inf_3dhp_stacked_hourglass_17kpts'):
    """Plot training progress"""
    log_path = f'exp/{exp_name}/log'
    iterations, losses = parse_log_file(log_path)
    
    if not losses:
        print("No loss data found in log file")
        return
    
    plt.figure(figsize=(12, 6))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(iterations, losses, 'b-', alpha=0.7, label='Training Loss')
    
    # Add moving average
    if len(losses) > 10:
        window = min(50, len(losses) // 10)
        moving_avg = np.convolve(losses, np.ones(window)/window, mode='valid')
        moving_avg_iter = iterations[window-1:]
        plt.plot(moving_avg_iter, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window})')
    
    plt.xlabel('Iteration')
    plt.ylabel('Combined Heatmap Loss')
    plt.title('Training Loss Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for better visualization
    
    # Plot recent progress (last 1000 iterations)
    plt.subplot(1, 2, 2)
    if len(iterations) > 100:
        recent_iter = iterations[-1000:]
        recent_loss = losses[-1000:]
        plt.plot(recent_iter, recent_loss, 'g-', alpha=0.7, label='Recent Training Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Combined Heatmap Loss')
        plt.title('Recent Training Progress')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'exp/{exp_name}/training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\n=== Training Progress Summary ===")
    print(f"Total iterations: {len(iterations)}")
    print(f"Initial loss: {losses[0]:.6f}")
    print(f"Current loss: {losses[-1]:.6f}")
    print(f"Best loss: {min(losses):.6f}")
    print(f"Loss reduction: {((losses[0] - losses[-1]) / losses[0] * 100):.2f}%")
    
    if len(losses) > 100:
        recent_avg = np.mean(losses[-100:])
        older_avg = np.mean(losses[-200:-100]) if len(losses) > 200 else np.mean(losses[:-100])
        trend = "improving" if recent_avg < older_avg else "worsening"
        print(f"Recent trend: {trend} (last 100 vs previous 100 avg)")

def check_experiment_status(exp_name='mpi_inf_3dhp_stacked_hourglass_17kpts'):
    """Check the status of the experiment"""
    exp_path = f'exp/{exp_name}'
    
    print(f"\n=== Experiment Status: {exp_name} ===")
    
    # Check if experiment directory exists
    if not os.path.exists(exp_path):
        print("❌ Experiment directory not found")
        return
    
    # Check for checkpoint
    checkpoint_path = f'{exp_path}/checkpoint.pt'
    if os.path.exists(checkpoint_path):
        print("✅ Checkpoint found")
        # Get checkpoint info
        import torch
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            print(f"   - Epoch: {checkpoint.get('epoch', 'Unknown')}")
            if 'best_val_loss' in checkpoint:
                print(f"   - Best validation loss: {checkpoint['best_val_loss']:.6f}")
        except:
            print("   - Could not load checkpoint details")
    else:
        print("❌ No checkpoint found")
    
    # Check for best model
    best_model_path = f'{exp_path}/best_model.pt'
    if os.path.exists(best_model_path):
        print("✅ Best model saved")
    else:
        print("❌ No best model found")
    
    # Check log file
    log_path = f'{exp_path}/log'
    if os.path.exists(log_path):
        print("✅ Log file found")
        # Get last modification time
        mod_time = datetime.fromtimestamp(os.path.getmtime(log_path))
        print(f"   - Last updated: {mod_time}")
    else:
        print("❌ No log file found")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='mpi_inf_3dhp_stacked_hourglass_17kpts', help='Experiment name')
    parser.add_argument('--plot', action='store_true', help='Plot training progress')
    args = parser.parse_args()
    
    check_experiment_status(args.exp)
    
    if args.plot:
        plot_training_progress(args.exp)
