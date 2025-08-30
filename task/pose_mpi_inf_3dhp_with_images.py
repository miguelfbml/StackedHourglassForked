"""
__config__ contains the options for training and testing the StackedHourglass model
for MPI-INF-3DHP dataset with 17 keypoints
"""
import torch
import numpy as np
from torch import nn
import os
from torch.nn import DataParallel
from utils.misc import make_input, make_output, importNet

__config__ = {
    'data_provider': 'data.MPI_INF_3DHP.dp_with_images',
    'network': 'models.posenet.PoseNet',
    'inference': {
        'nstack': 8,
        'inp_dim': 256,
        'oup_dim': 17,  # Changed from 16 to 17 for MPI-INF-3DHP
        'num_parts': 17,  # Changed from 16 to 17 for MPI-INF-3DHP
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 1000,  # Number of validation examples
        'train_num_eval': 500,  # Number of train examples tested at test time
    },

    'train': {
        'batchsize': 8,  # Reduced from 16 due to larger image size and memory constraints
        'input_res': 256,
        'output_res': 64,
        'train_iters': 2000,  # Increased for MPI-INF-3DHP
        'valid_iters': 100,
        'learning_rate': 1e-3,
        'max_num_people': 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 150000,
        'decay_lr': 2e-4,
        'num_workers': 4,
        'use_data_loader': True,
    },
    
    # MPI-INF-3DHP specific configuration
    'mpi_dataset_root': '/nas-ctm01/datasets/public/mpi_inf_3dhp',
    'data_root': 'data/motion3d',  # Changed from 'data/MPI_INF_3DHP/motion3d'
}

class Trainer(nn.Module):
    """
    The wrapper module that will behave differently for training or testing
    inference_keys specify the inputs for inference
    """
    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.inference_keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs, **inputs):
        inps = {}
        for i in self.inference_keys:
            inps[i] = inputs[i]

        if not self.training:
            return self.model(imgs)
        else:
            combined_hm_preds = self.model(imgs)
            if self.calc_loss is not None:
                return self.calc_loss(combined_hm_preds, inputs)
            else:
                return combined_hm_preds

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']
    
    def calc_loss(*args, **kwargs):
        return make_output(configs, *args, **kwargs)

    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(config['nstack'], config['inp_dim'], config['oup_dim'])

    forward_net = Trainer(poseNet, config['keys'], calc_loss)
    
    def calc_loss(*args, **kwargs):
        return make_output(configs, *args, **kwargs)

    config['net'] = DataParallel(forward_net)

    train_cfg['optimizer'] = torch.optim.Adam(config['net'].parameters(), train_cfg['learning_rate'])

    # Create experiment directory - handle case where 'opt' might not exist yet
    if 'opt' in configs and hasattr(configs['opt'], 'exp'):
        exp_path = os.path.join('exp', configs['opt'].exp)
    else:
        exp_path = os.path.join('exp', 'default_mpi_inf_3dhp')
    
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass # for last input, which is heatmap file name
        net = config['inference']['net']
        config['batch_id'] = batch_id

        if phase != 'inference':
            result = net(inputs['imgs'], **{i: inputs[i] for i in inputs if i!='imgs'})
            num_loss = len(config['train']['loss'])

            losses = result['losses']
            loss = 0
            toprint = '\n{}: '.format(batch_id)
            for i in range(len(config['train']['loss'])):
                loss += config['train']['loss'][i][1] * losses[i].mean() 
                toprint += ' {}: {}'.format(config['train']['loss'][i][0], losses[i].mean().data.item())
                if i < len(config['train']['loss']) - 1:
                    toprint += ' |'
            logger.write(toprint)
            logger.flush()
            print(toprint)
            
            if phase == 'train':
                config['train']['optimizer'].zero_grad()
                loss.backward()
                config['train']['optimizer'].step()
            return None
        else:
            out = {}
            net = net.eval()
            result = net(**inputs)
            net = net.train()
            if type(result)!=list and type(result)!=tuple:
                result = [result]
            out['preds'] = [i.data.cpu().numpy() for i in result]
            return out
    configs['func'] = make_train
    return configs
