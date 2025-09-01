"""
Task configuration for MPI-INF-3DHP dataset with images
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
        'oup_dim': 17,
        'num_parts': 17,
        'increase': 0,
        'keys': ['imgs'],
        'num_eval': 1000,
        'train_num_eval': 500,
    },
    'train': {
        'batchsize': 4,
        'input_res': 256,
        'output_res': 64,
        'train_iters': 2000,
        'valid_iters': 100,
        'learning_rate': 1e-3,
        'max_num_people': 1,
        'loss': [
            ['combined_hm_loss', 1],
        ],
        'decay_iters': 150000,
        'decay_lr': 2e-4,
        'num_workers': 0,
        'use_data_loader': True,
        'sigma': 1,
        'scale_factor': 0.25,
        'rot_factor': 30,
        'label_type': 'Gaussian',
    },
    'data_root': 'data/motion3d',
    'mpi_dataset_root': '/nas-ctm01/datasets/public/mpi_inf_3dhp',
}

class Trainer(nn.Module):
    def __init__(self, model, inference_keys, calc_loss=None):
        super(Trainer, self).__init__()
        self.model = model
        self.inference_keys = inference_keys
        self.calc_loss = calc_loss

    def forward(self, imgs=None, **inputs):
        if imgs is None and 'imgs' in inputs:
            imgs = inputs['imgs']
        
        inps = {}
        labels = {}

        for i in inputs:
            if i in self.inference_keys:
                inps[i] = inputs[i]
            else:
                labels[i] = inputs[i]

        if not self.training:
            return self.model(imgs, **inps)
        else:
            combined_hm_preds = self.model(imgs, **inps)
            if type(combined_hm_preds)!=list and type(combined_hm_preds)!=tuple:
                combined_hm_preds = [combined_hm_preds]
            loss = self.calc_loss(**labels, combined_hm_preds=combined_hm_preds)
            return list(combined_hm_preds) + list([loss])

def make_network(configs):
    train_cfg = configs['train']
    config = configs['inference']
    
    PoseNet = importNet(configs['network'])
    poseNet = PoseNet(config['nstack'], config['inp_dim'], config['oup_dim'])

    def calc_loss(*args, **kwargs):
        return [kwargs['heatmaps']]

    forward_net = Trainer(poseNet, config['keys'], calc_loss)
    config['net'] = DataParallel(forward_net)
    train_cfg['optimizer'] = torch.optim.Adam(config['net'].parameters(), train_cfg['learning_rate'])

    exp_path = os.path.join('exp', configs.get('opt', type('obj', (object,), {'exp': 'mpi_inf_3dhp'})).exp)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    logger = open(os.path.join(exp_path, 'log'), 'a+')

    def make_train(batch_id, config, phase, **inputs):
        for i in inputs:
            try:
                inputs[i] = make_input(inputs[i])
            except:
                pass

        net = config['inference']['net']
        config['batch_id'] = batch_id

        if phase != 'inference':
            result = net(**inputs)
            combined_hm_preds = result[:-1]
            losses = result[-1]

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