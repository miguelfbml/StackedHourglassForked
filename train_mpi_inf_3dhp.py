import os
import tqdm
from os.path import dirname
import time
import threading
from statistics import mean
import numpy as np

import torch.backends.cudnn as cudnn
cudnn.benchmark = True
cudnn.enabled = True

import torch
import importlib
import argparse
from datetime import datetime
from pytz import timezone
import wandb
import pynvml
from ptflops import get_model_complexity_info

import shutil

class GPUUtilizationMonitor:
    def __init__(self, device_idx=0):
        pynvml.nvmlInit()
        self.device = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
        self.utilization_rates = []
        self.memory_usage = []
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def _monitor(self):
        while self.running:
            util = pynvml.nvmlDeviceGetUtilizationRates(self.device)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(self.device)
            self.utilization_rates.append(util.gpu)
            self.memory_usage.append(memory_info.used / memory_info.total * 100)
            time.sleep(0.1)

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        pynvml.nvmlShutdown()

    def get_stats(self):
        avg_util = mean(self.utilization_rates) if self.utilization_rates else 0
        avg_mem = mean(self.memory_usage) if self.memory_usage else 0
        return avg_util, avg_mem

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
                hm = heatmaps[b, j].detach().cpu().numpy() if hasattr(heatmaps, 'detach') else heatmaps[b, j]
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

def compute_flops(model, input_shape, device):
    """Compute FLOPs for the model"""
    model.eval()
    flops, params = get_model_complexity_info(
        model, input_shape[1:], as_strings=False, print_per_layer_stat=False
    )
    return flops, params

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--continue_exp', type=str, help='continue exp')
    parser.add_argument('-e', '--exp', type=str, default='pose', help='experiments name')
    parser.add_argument('-m', '--max_iters', type=int, default=250, help='max number of iterations (thousands)')
    parser.add_argument('--use-wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-name', default=None, type=str, help='WandB run name')
    parser.add_argument('--wandb-project', default='StackedHourglass-MPI-INF-3DHP', type=str, help='WandB project name')
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
            exit(0)

    if 'epoch' not in config['train']:
        config['train']['epoch'] = 0

def save_checkpoint(state, is_best, filename='checkpoint.pt'):
    """
    from pytorch/examples
    """
    basename = dirname(filename)
    if not os.path.exists(basename):
        os.makedirs(basename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pt')

def save(config):
    resume = os.path.join('exp', config['opt'].exp)
    if config['opt'].exp=='pose' and config['opt'].continue_exp is not None:
        resume = os.path.join('exp', config['opt'].continue_exp)
    resume_file = os.path.join(resume, 'checkpoint.pt')

    save_checkpoint({
            'state_dict': config['inference']['net'].state_dict(),
            'optimizer' : config['train']['optimizer'].state_dict(),
            'epoch': config['train']['epoch'],
        }, False, filename=resume_file)
    print('=> save checkpoint')

def train_epoch_with_metrics(train_func, data_func, config, phase):
    """Train/validate one epoch with detailed metrics"""
    gpu_monitor = GPUUtilizationMonitor(device_idx=0)
    gpu_monitor.start()
    
    batch_times = []
    losses = []
    mpjpe_values = []
    
    num_step = config['train']['{}_iters'.format(phase)]
    generator = data_func(phase)
    print('start', phase, config['opt'].exp)

    show_range = range(num_step)
    show_range = tqdm.tqdm(show_range, total=num_step, ascii=True)
    batch_id = num_step * config['train']['epoch']
    
    for i in show_range:
        batch_start = time.perf_counter()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        datas = next(generator)
        outs = train_func(batch_id + i, config, phase, **datas)
        
        # Calculate MPJPE if we have the necessary data
        if 'heatmaps' in datas and phase == 'valid':
            try:
                # Get predictions from the training function
                result = config['func'](batch_id + i, config, 'inference', **datas)
                if result and 'preds' in result:
                    pred_heatmaps = result['preds'][-1]  # Last stack output
                    gt_heatmaps = datas['heatmaps']
                    
                    if not isinstance(pred_heatmaps, torch.Tensor):
                        pred_heatmaps = torch.tensor(pred_heatmaps)
                    
                    mpjpe = calculate_mpjpe_pixels(pred_heatmaps, gt_heatmaps)
                    mpjpe_values.append(mpjpe)
            except Exception as e:
                pass  # Skip MPJPE calculation if it fails
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        batch_time = time.perf_counter() - batch_start
        batch_times.append(batch_time)
    
    gpu_monitor.stop()
    avg_gpu_util, avg_gpu_mem = gpu_monitor.get_stats()
    mean_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0
    mean_mpjpe = sum(mpjpe_values) / len(mpjpe_values) if mpjpe_values else 0
    
    print(f"{phase.capitalize()} Mean Batch Time: {mean_batch_time:.4f} seconds")
    print(f"{phase.capitalize()} GPU Utilization: {avg_gpu_util:.2f}%")
    print(f"{phase.capitalize()} GPU Memory Usage: {avg_gpu_mem:.2f}%")
    if mpjpe_values:
        print(f"{phase.capitalize()} MPJPE: {mean_mpjpe:.2f} pixels")
    
    return mean_batch_time, avg_gpu_util, avg_gpu_mem, mean_mpjpe

def train(train_func, data_func, config, post_epoch=None):
    # Initialize WandB if requested
    if config['opt'].use_wandb:
        wandb.init(
            project=config['opt'].wandb_project,
            name=config['opt'].wandb_name or f"mpi_inf_3dhp_{config['opt'].exp}",
            config={
                'experiment': config['opt'].exp,
                'max_iters': config['opt'].max_iters,
                'batch_size': config['train'].get('batchsize', 'unknown'),
                'learning_rate': config['train'].get('learning_rate', 'unknown'),
            }
        )
    
    # Compute FLOPs for the model
    try:
        model = config['inference']['net']
        actual_model = model.module if hasattr(model, 'module') else model
        # Estimate input shape based on config
        input_shape = (config['train'].get('batchsize', 4), 3, 256, 256)  # [B, C, H, W]
        flops, params = compute_flops(actual_model, input_shape, 'cuda' if torch.cuda.is_available() else 'cpu')
        training_flops = flops * 3  # Approximate: forward + 2x backward
        print(f"Model FLOPs (forward pass): {flops / 1e9:.2f} GFLOPs")
        print(f"Training FLOPs per sample (approx): {training_flops / 1e9:.2f} GFLOPs")
        print(f"Model Parameters: {params:,}")
        
        if config['opt'].use_wandb:
            wandb.config.update({
                'model_flops_gflops': flops / 1e9,
                'training_flops_gflops': training_flops / 1e9,
                'model_parameters': params
            })
    except Exception as e:
        print(f"Could not compute FLOPs: {e}")
        flops = training_flops = 0
    
    best_mpjpe = float('inf')
    
    while True:
        fails = 0
        epoch_start_time = time.perf_counter()
        print('epoch: ', config['train']['epoch'])
        
        if 'epoch_num' in config['train']:
            if config['train']['epoch'] > config['train']['epoch_num']:
                break

        epoch_metrics = {}
        
        for phase in ['train', 'valid']:
            batch_id = config['train']['{}_iters'.format(phase)] * config['train']['epoch']
            if batch_id > config['opt'].max_iters * 1000:
                if config['opt'].use_wandb:
                    wandb.finish()
                return
            
            # Train/validate with detailed metrics
            mean_batch_time, avg_gpu_util, avg_gpu_mem, mean_mpjpe = train_epoch_with_metrics(
                train_func, data_func, config, phase
            )
            
            epoch_metrics[f'{phase}/mean_batch_time'] = mean_batch_time
            epoch_metrics[f'{phase}/gpu_utilization'] = avg_gpu_util
            epoch_metrics[f'{phase}/gpu_memory_usage'] = avg_gpu_mem
            if mean_mpjpe > 0:
                epoch_metrics[f'{phase}/mpjpe'] = mean_mpjpe
                
                # Track best MPJPE
                if phase == 'valid' and mean_mpjpe < best_mpjpe:
                    best_mpjpe = mean_mpjpe
                    epoch_metrics['best_mpjpe'] = best_mpjpe
        
        epoch_time = time.perf_counter() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['epoch'] = config['train']['epoch']
        
        # Add FLOPs info
        if flops > 0:
            epoch_metrics['train/flops_per_sample_gflops'] = training_flops / 1e9
            epoch_metrics['eval/flops_per_sample_gflops'] = flops / 1e9
        
        # Log to WandB
        if config['opt'].use_wandb:
            wandb.log(epoch_metrics, step=config['train']['epoch'])
        
        # Print epoch summary
        print("\n" + "="*80)
        print(f"📊 Epoch {config['train']['epoch']} Summary (Time: {epoch_time:.1f}s):")
        for key, value in epoch_metrics.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        print("="*80 + "\n")
        
        config['train']['epoch'] += 1
        save(config)
    
    if config['opt'].use_wandb:
        wandb.finish()

def init():
    """
    task.__config__ contains the variables that control the training and testing
    make_network builds a function which can do forward and backward propagation
    """
    opt = parse_command_line()
    task = importlib.import_module('task.pose')
    exp_path = os.path.join('exp', opt.exp)
    
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')

    config = task.__config__
    try: os.makedirs(exp_path)
    except FileExistsError: pass

    config['opt'] = opt
    config['data_provider'] = importlib.import_module(config['data_provider'])

    func = task.make_network(config)
    config['func'] = func  # Store function for MPJPE evaluation
    reload(config)
    return func, config

def main():
    func, config = init()
    data_func = config['data_provider'].init(config)
    train(func, data_func, config)
    print(datetime.now(timezone('EST')))

if __name__ == '__main__':
    main()