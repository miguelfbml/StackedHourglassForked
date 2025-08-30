#!/usr/bin/env python3
"""
Training script for StackedHourglass model on MPI-INF-3DHP dataset
Modified to work with 17 keypoints instead of 16
"""

import os
import tqdm
from os.path import dirname

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import torch
import importlib
import argparse
from datetime import datetime
from pytz import timezone
import shutil

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='mpi_inf_3dhp_stacked_hourglass', help='experiment name')
    parser.add_argument('-m', '--max_iters', type=int, default=500, help='max number of iterations (thousands)')
    parser.add_argument('--data_root', type=str, default='data/MPI_INF_3DHP/motion3d', help='path to motion3d data')
    parser.add_argument('--mpi_dataset_root', type=str, default='/nas-ctm01/datasets/public/mpi_inf_3dhp', help='path to MPI-INF-3DHP images')
    args = parser.parse_args()
    return args

def reload(config):
    """
    Load or initialize model's parameters by config from config['opt'].continue_exp
    config['train']['epoch'] records the epoch num
    config['inference']['net'] is the model
    """
    opt = config['opt']

    if opt.continue_exp:
        resume = os.path.join('exp', opt.continue_exp)
        resume_file = os.path.join(resume, 'checkpoint.pt')
        if os.path.isfile(resume_file):
            print("=> loading checkpoint '{}'".format(resume))
            checkpoint = torch.load(resume_file)

            config['inference']['net'].load_state_dict(checkpoint['state_dict'])
            config['train']['optimizer'].load_state_dict(checkpoint['optimizer'])
            config['train']['epoch'] = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(resume))
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(config):
    """Save current model state"""
    exp_path = os.path.join('exp', config['opt'].exp)
    checkpoint_path = os.path.join(exp_path, 'checkpoint.pt')
    
    torch.save({
        'epoch': config['train']['epoch'],
        'state_dict': config['inference']['net'].state_dict(),
        'optimizer': config['train']['optimizer'].state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

def main():
    # Parse arguments
    opt = parse_command_line()
    
    # Update paths in config
    os.environ['DATA_ROOT'] = opt.data_root
    os.environ['MPI_DATASET_ROOT'] = opt.mpi_dataset_root
    
    # Import the task configuration for MPI-INF-3DHP
    task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
    
    # Add opt to config before calling make_network
    task.__config__['opt'] = opt
    task.__config__['data_root'] = opt.data_root
    task.__config__['mpi_dataset_root'] = opt.mpi_dataset_root
    
    exp = task.make_network(task.__config__)
    
    # Update config with command line arguments (redundant but safe)
    exp['opt'] = opt
    exp['data_root'] = opt.data_root
    exp['mpi_dataset_root'] = opt.mpi_dataset_root
    
    # Create experiment directory
    exp_path = os.path.join('exp', opt.exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
        print(f"Created experiment directory: {exp_path}")
    
    # Load or initialize model
    reload(exp)
    
    # Import data provider
    data_func = importlib.import_module(exp['data_provider'])
    
    print("Setting up datasets...")
    print(f"Data root: {opt.data_root}")
    print(f"MPI dataset root: {opt.mpi_dataset_root}")
    
    # Create training dataset
    train_dataset = data_func.Dataset(exp, train=True, 
                                    data_root=opt.data_root,
                                    mpi_dataset_root=opt.mpi_dataset_root)
    print(f"Training dataset size: {len(train_dataset)}")
    
    # Create validation dataset  
    val_dataset = data_func.Dataset(exp, train=False,
                                  data_root=opt.data_root, 
                                  mpi_dataset_root=opt.mpi_dataset_root)
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=exp['train']['batchsize'],
        shuffle=True,
        num_workers=exp['train']['num_workers'],
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=exp['train']['batchsize'],
        shuffle=False, 
        num_workers=exp['train']['num_workers'],
        pin_memory=True
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Training loop
    print("Starting training...")
    print(f"Target iterations: {opt.max_iters * 1000}")
    print(f"Model configuration: {exp['inference']['nstack']} stacks, {exp['inference']['oup_dim']} keypoints")
    
    max_iterations = opt.max_iters * 1000
    current_iter = exp['train']['epoch'] * len(train_loader)
    
    # Set model to training mode
    exp['inference']['net'].train()
    
    while current_iter < max_iterations:
        epoch = current_iter // len(train_loader)
        exp['train']['epoch'] = epoch
        
        print(f"\nEpoch {epoch} (iteration {current_iter}/{max_iterations})")
        
        # Training phase
        exp['inference']['net'].train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for batch_idx, batch_data in enumerate(tqdm.tqdm(train_loader, desc=f"Training epoch {epoch}")):
            current_iter += 1
            
            # Prepare batch data
            if isinstance(batch_data, dict):
                imgs, heatmaps = batch_data['imgs'], batch_data['heatmaps']
            else:
                # Legacy format
                imgs, heatmaps = batch_data
            
            # Move to GPU if available
            device = next(exp['inference']['net'].parameters()).device
            batch_input = {
                'imgs': imgs.to(device),
                'heatmaps': heatmaps.to(device)
            }
            
            # Train step
            exp['func'](current_iter, exp, 'train', **batch_input)
            
            # Validation every valid_iters iterations
            if current_iter % exp['train']['valid_iters'] == 0:
                print(f"\nRunning validation at iteration {current_iter}")
                exp['inference']['net'].eval()
                
                val_loss_sum = 0.0
                val_batches = 0
                
                with torch.no_grad():
                    for val_batch_idx, val_batch_data in enumerate(tqdm.tqdm(val_loader, desc="Validation")):
                        if val_batch_idx >= exp['inference']['num_eval'] // exp['train']['batchsize']:
                            break
                            
                        if isinstance(val_batch_data, dict):
                            val_imgs, val_heatmaps = val_batch_data['imgs'], val_batch_data['heatmaps']
                        else:
                            val_imgs, val_heatmaps = val_batch_data
                            
                        val_batch_input = {
                            'imgs': val_imgs.to(device),
                            'heatmaps': val_heatmaps.to(device)
                        }
                        
                        exp['func'](f"val_{current_iter}_{val_batch_idx}", exp, 'valid', **val_batch_input)
                        val_batches += 1
                
                exp['inference']['net'].train()
            
            # Save checkpoint every 10000 iterations
            if current_iter % 10000 == 0:
                save_checkpoint(exp)
            
            # Learning rate decay
            if current_iter % exp['train']['decay_iters'] == 0 and current_iter > 0:
                for param_group in exp['train']['optimizer'].param_groups:
                    param_group['lr'] = exp['train']['decay_lr']
                print(f"Learning rate decayed to {exp['train']['decay_lr']} at iteration {current_iter}")
            
            if current_iter >= max_iterations:
                break
    
    # Final checkpoint save
    save_checkpoint(exp)
    print(f"Training completed after {current_iter} iterations")

if __name__ == '__main__':
    main()
