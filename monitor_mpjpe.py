#!/usr/bin/env python3
"""
Real-time MPJPE monitoring script
Usage: python monitor_mpjpe.py [experiment_name]
"""

import sys
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def plot_mpjpe(exp_name):
    """Plot MPJPE over training iterations"""
    mpjpe_file = f"exp/{exp_name}/mpjpe_log.txt"
    
    if not os.path.exists(mpjpe_file):
        print(f"MPJPE log file not found: {mpjpe_file}")
        return
    
    try:
        # Read the CSV file
        df = pd.read_csv(mpjpe_file)
        
        # Separate training and validation data
        train_data = df[df['Phase'] == 'train']
        valid_data = df[df['Phase'] == 'valid']
        
        # Create the plot
        plt.figure(figsize=(15, 10))
        
        # Plot MPJPE
        plt.subplot(2, 2, 1)
        if len(train_data) > 0:
            plt.plot(train_data['Batch_ID'], train_data['MPJPE'], 'b-', label='Train MPJPE', alpha=0.7)
        if len(valid_data) > 0:
            plt.plot(valid_data['Batch_ID'], valid_data['MPJPE'], 'r-', label='Valid MPJPE', linewidth=2)
        plt.xlabel('Batch ID')
        plt.ylabel('MPJPE (pixels)')
        plt.title('Mean Per Joint Position Error')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot PCK
        plt.subplot(2, 2, 2)
        if len(train_data) > 0:
            plt.plot(train_data['Batch_ID'], train_data['PCK@2px'], 'b-', label='Train PCK@2px', alpha=0.7)
        if len(valid_data) > 0:
            plt.plot(valid_data['Batch_ID'], valid_data['PCK@2px'], 'r-', label='Valid PCK@2px', linewidth=2)
        plt.xlabel('Batch ID')
        plt.ylabel('PCK@2px (%)')
        plt.title('Percentage of Correct Keypoints')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot Loss
        plt.subplot(2, 2, 3)
        if len(train_data) > 0:
            plt.plot(train_data['Batch_ID'], train_data['Loss'], 'b-', label='Train Loss', alpha=0.7)
        if len(valid_data) > 0:
            plt.plot(valid_data['Batch_ID'], valid_data['Loss'], 'r-', label='Valid Loss', linewidth=2)
        plt.xlabel('Batch ID')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for loss
        
        # Show statistics
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Calculate statistics
        stats_text = f"Experiment: {exp_name}\n\n"
        
        if len(train_data) > 0:
            latest_train_mpjpe = train_data['MPJPE'].iloc[-1]
            latest_train_pck = train_data['PCK@2px'].iloc[-1]
            stats_text += f"Latest Train MPJPE: {latest_train_mpjpe:.2f}px\n"
            stats_text += f"Latest Train PCK@2px: {latest_train_pck:.1f}%\n\n"
        
        if len(valid_data) > 0:
            best_valid_mpjpe = valid_data['MPJPE'].min()
            best_valid_pck = valid_data['PCK@2px'].max()
            latest_valid_mpjpe = valid_data['MPJPE'].iloc[-1]
            latest_valid_pck = valid_data['PCK@2px'].iloc[-1]
            
            stats_text += f"Latest Valid MPJPE: {latest_valid_mpjpe:.2f}px\n"
            stats_text += f"Latest Valid PCK@2px: {latest_valid_pck:.1f}%\n\n"
            stats_text += f"Best Valid MPJPE: {best_valid_mpjpe:.2f}px\n"
            stats_text += f"Best Valid PCK@2px: {best_valid_pck:.1f}%\n"
        
        plt.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        
        plt.tight_layout()
        
        # Save the plot
        plot_file = f"exp/{exp_name}/mpjpe_plot.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {plot_file}")
        
        # Also display if running interactively
        plt.show()
        
    except Exception as e:
        print(f"Error plotting MPJPE: {e}")

def monitor_continuously(exp_name, interval=30):
    """Monitor MPJPE continuously and update plot"""
    print(f"Monitoring MPJPE for experiment: {exp_name}")
    print(f"Updating plot every {interval} seconds...")
    print("Press Ctrl+C to stop monitoring")
    
    try:
        while True:
            plot_mpjpe(exp_name)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopping monitoring.")

def main():
    parser = argparse.ArgumentParser(description='Monitor MPJPE during training')
    parser.add_argument('exp_name', help='Experiment name')
    parser.add_argument('--continuous', '-c', action='store_true', help='Monitor continuously')
    parser.add_argument('--interval', '-i', type=int, default=30, help='Update interval in seconds')
    
    args = parser.parse_args()
    
    if args.continuous:
        monitor_continuously(args.exp_name, args.interval)
    else:
        plot_mpjpe(args.exp_name)

if __name__ == "__main__":
    main()
