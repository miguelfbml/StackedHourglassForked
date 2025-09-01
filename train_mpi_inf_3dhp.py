#!/usr/bin/env python3
"""
Training script for MPI-INF-3DHP dataset
"""

import os
import torch
import importlib
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='mpi_inf_3dhp', help='experiment name')
    parser.add_argument('--max_iters', type=int, default=1000, help='max iterations')
    parser.add_argument('--data_root', type=str, default='data/motion3d', help='data root')
    parser.add_argument('--mpi_dataset_root', type=str, default='/nas-ctm01/datasets/public/mpi_inf_3dhp', help='MPI dataset root')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Starting MPI-INF-3DHP training...")
    print(f"Experiment: {args.exp}")
    print(f"Max iterations: {args.max_iters}")
    print(f"Data root: {args.data_root}")
    print(f"MPI dataset root: {args.mpi_dataset_root}")
    
    # Import task
    task = importlib.import_module('task.pose_mpi_inf_3dhp_with_images')
    config = task.__config__.copy()
    
    # Update config
    config['data_root'] = args.data_root
    config['mpi_dataset_root'] = args.mpi_dataset_root
    
    # Add args to config
    setattr(config, 'opt', args)
    
    # Create network
    print("Creating network...")
    exp = task.make_network(config)
    
    # Import data provider
    data_func = importlib.import_module(exp['data_provider'])
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = data_func.Dataset(exp, train=True, 
                                    data_root=args.data_root,
                                    mpi_dataset_root=args.mpi_dataset_root)
    
    val_dataset = data_func.Dataset(exp, train=False,
                                  data_root=args.data_root,
                                  mpi_dataset_root=args.mpi_dataset_root)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=exp['train']['batchsize'], 
        shuffle=True, num_workers=0, collate_fn=data_func.custom_collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=exp['train']['batchsize'], 
        shuffle=False, num_workers=0, collate_fn=data_func.custom_collate_fn
    )
    
    # Training loop
    print("Starting training...")
    exp['inference']['net'].train()
    
    current_iter = 0
    
    for epoch in range(100):  # Large number to ensure we reach max_iters
        print(f"Epoch {epoch}")
        
        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
            if current_iter >= args.max_iters:
                break
            
            # Train step
            result = exp['func'](current_iter, exp, 'train', **batch_data)
            
            # Validation every 100 iterations
            if current_iter % 100 == 0 and current_iter > 0:
                print(f"\nRunning validation at iteration {current_iter}")
                exp['inference']['net'].eval()
                with torch.no_grad():
                    for val_idx, val_batch in enumerate(val_loader):
                        if val_idx >= 10:  # Only validate on first 10 batches
                            break
                        exp['func'](f"val_{current_iter}_{val_idx}", exp, 'valid', **val_batch)
                exp['inference']['net'].train()
            
            # Save checkpoint every 1000 iterations
            if current_iter % 1000 == 0 and current_iter > 0:
                exp_path = os.path.join('exp', args.exp)
                checkpoint_path = os.path.join(exp_path, f'checkpoint_{current_iter}.pt')
                torch.save({
                    'iteration': current_iter,
                    'state_dict': exp['inference']['net'].state_dict(),
                    'optimizer': exp['train']['optimizer'].state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
            
            current_iter += 1
        
        if current_iter >= args.max_iters:
            break
    
    # Save final model
    exp_path = os.path.join('exp', args.exp)
    final_path = os.path.join(exp_path, 'final_model.pt')
    torch.save({
        'iteration': current_iter,
        'state_dict': exp['inference']['net'].state_dict(),
        'optimizer': exp['train']['optimizer'].state_dict(),
    }, final_path)
    print(f"Final model saved: {final_path}")
    
    print("Training completed!")

if __name__ == '__main__':
    main()