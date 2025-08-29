import os
import tqdm
from os.path import dirname
import argparse
import multiprocessing

# Fix for multiprocessing issues on SLURM/cluster environments
try:
    multiprocessing.set_start_method('spawn', force=True)
except:
    pass

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import torch
import importlib
from datetime import datetime
from pytz import timezone
from utils.evaluation import calculate_mpjpe, calculate_pck

import shutil

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose_mpi_inf_3dhp_simple', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('--data_root', type=str, default='data/motion3d', help='Path to MPI-INF-3DHP data')
    args = parser.parse_args()
    return args

def reload(config):
    """
    load or initialize model's parameters by config from config['opt'].continue_exp
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

def save(config):
    exp_path = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp=='pose_mpi_inf_3dhp_simple' and config['opt'].continue_exp is not None:
        exp_path = os.path.join('exp', config['opt'].continue_exp)
    
    save_file = os.path.join(exp_path, 'checkpoint.pt')
    save_state = config['inference']['net'].state_dict()
    
    torch.save({
        'state_dict': save_state,
        'optimizer': config['train']['optimizer'].state_dict(),
        'epoch': config['train']['epoch']
    }, save_file)

def train(func, data_func, config):
    train_iters = config['train']['train_iters']
    valid_iters = config['train']['valid_iters']
    max_iter = config['opt'].max_iters * 1000
    
    # Create data generators  
    generator = data_func('train')
    valid_generator = data_func('valid')
    
    config['train']['epoch'] = 0
    
    for iteration in tqdm.tqdm(range(1, max_iter)):
        config['train']['epoch'] = iteration
        
        # Training step
        try:
            datas = next(generator)
            func(iteration, config, 'train', **datas)
        except StopIteration:
            generator = data_func('train')
            datas = next(generator)
            func(iteration, config, 'train', **datas)
        
        # Validation every valid_iters iterations
        if iteration % valid_iters == 0:
            print(f"\n=== Validation at iteration {iteration} ===")
            
            # Run validation
            total_val_loss = 0
            val_count = 0
            
            for val_iter in range(10):  # Run a few validation batches
                try:
                    val_datas = next(valid_generator)
                    func(iteration, config, 'valid', **val_datas)
                    val_count += 1
                except StopIteration:
                    valid_generator = data_func('valid')
                    if val_count > 0:
                        break
                    val_datas = next(valid_generator)
                    func(iteration, config, 'valid', **val_datas)
                    val_count += 1
            
            print(f"Completed validation with {val_count} batches")
        
        # Save checkpoint every 1000 iterations
        if iteration % 1000 == 0:
            save(config)
            print(f"Saved checkpoint at iteration {iteration}")

def main():
    opt = parse_command_line()
    
    print("==========\nArgs:")
    for k, v in opt.__dict__.items():
        print(k, ':', v)
    print("==========")
    
    if not os.path.exists('exp'):
        os.mkdir('exp')
    if not os.path.exists(os.path.join('exp', opt.exp)):
        os.mkdir(os.path.join('exp', opt.exp))
    
    exp_path = os.path.join('exp', opt.exp)
    
    # Import and setup task
    task = importlib.import_module('task.pose_mpi_inf_3dhp_simple')
    
    task.__config__['opt'] = opt
    task.__config__['opt'].data_root = opt.data_root
    
    # Setup model
    func = task.make_network(task.__config__)
    
    # Setup data
    data_module = importlib.import_module(task.__config__['data_provider'])
    data_func = data_module.init(task.__config__)
    
    # Load checkpoint if specified
    reload(task.__config__)
    
    print("Starting training...")
    train(func, data_func, task.__config__)

if __name__ == '__main__':
    main()
