import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
import utils.img

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x<0 or y<0 or x>=self.output_res or y>=self.output_res:
                        continue
                    ul = int(x - 3*sigma - 1), int(y - 3*sigma - 1)
                    br = int(x + 3*sigma + 2), int(y + 3*sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[idx, aa:bb,cc:dd] = np.maximum(hms[idx, aa:bb,cc:dd], self.g[a:b,c:d])
        return hms

class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, poses_2d, poses_3d, index, train=True):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.poses_2d = poses_2d
        self.poses_3d = poses_3d
        self.index = index
        self.train = train
        
        # MPI-INF-3DHP specific parameters
        self.kps_left = [5, 6, 7, 11, 12, 13]
        self.kps_right = [2, 3, 4, 8, 9, 10]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        return self.loadData(self.index[idx % len(self.index)])

    def loadData(self, idx):
        # Get 2D poses from the dataset (normalized coordinates)
        pose_2d = self.poses_2d[idx]  # Shape: (n_frames, 17, 3) where last dim is (x, y, confidence)
        pose_3d = self.poses_3d[idx] if self.poses_3d is not None else None
        
        # For now, let's use the center frame for 2D pose estimation
        center_frame = pose_2d.shape[0] // 2
        keypoints_2d = pose_2d[center_frame, :, :2]  # (17, 2)
        
        # Convert normalized coordinates back to pixel coordinates
        # The normalized coordinates are in [-1, 1] range
        keypoints_2d = (keypoints_2d + 1.0) * (self.output_res / 2.0)
        
        # Create a dummy image (for now, we'll generate a black image)
        # In a real implementation, you would load the actual image
        # For now, we create a simple synthetic image with some noise
        img = np.random.normal(0.5, 0.1, (self.input_res, self.input_res, 3)).astype(np.float32)
        img = np.clip(img, 0, 1)
        
        # Data augmentation for training
        if self.train:
            # Random horizontal flip
            if np.random.random() > 0.5:
                img = img[:, ::-1, :]
                keypoints_2d[:, 0] = self.output_res - keypoints_2d[:, 0]
                # Swap left and right keypoints
                keypoints_2d[self.kps_left + self.kps_right] = keypoints_2d[self.kps_right + self.kps_left]
            
            # Random rotation and scale
            center = np.array([self.input_res/2, self.input_res/2])
            scale = max(self.input_res, self.input_res)/200
            
            aug_rot = (np.random.random() * 2 - 1) * 30.
            aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
            scale *= aug_scale
            
            # Apply transformation to keypoints
            mat = utils.img.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]
            
            # Transform keypoints
            keypoints_homogeneous = np.column_stack([keypoints_2d, np.ones(keypoints_2d.shape[0])])
            keypoints_transformed = np.dot(mat, keypoints_homogeneous.T).T
            keypoints_2d = keypoints_transformed
        
        # Reshape keypoints to the expected format for heatmap generation
        keypoints = np.zeros((1, 17, 3))
        keypoints[0, :, :2] = keypoints_2d
        keypoints[0, :, 2] = 1  # Set visibility to 1 for all keypoints
        
        # Generate heatmaps
        heatmaps = self.generateHeatmap(keypoints)
        
        # Convert image to proper format but keep (H, W, C) for the model
        # The PoseNet expects (H, W, C) format and will permute internally
        
        return img.astype(np.float32), heatmaps.astype(np.float32)

def load_mpi_inf_3dhp_data(data_root):
    """Load MPI-INF-3DHP dataset similar to TCPFormerForked"""
    
    # Load training data
    train_data = np.load(os.path.join(data_root, "data_train_3dhp.npz"), allow_pickle=True)['data'].item()
    
    poses_2d_train = []
    poses_3d_train = []
    
    for seq in train_data.keys():
        for cam in train_data[seq][0].keys():
            anim = train_data[seq][0][cam]
            
            data_3d = anim['data_3d']
            # Root-relative coordinates (subtract pelvis - joint 14)
            data_3d[:, :14] -= data_3d[:, 14:15]
            data_3d[:, 15:] -= data_3d[:, 14:15]
            
            data_2d = anim['data_2d']
            # Normalize screen coordinates
            data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
            
            # Add confidence scores
            confidence_scores = np.ones((*data_2d.shape[:2], 1))
            data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
            
            poses_2d_train.append(data_2d)
            poses_3d_train.append(data_3d)
    
    # Load test data
    test_data = np.load(os.path.join(data_root, "data_test_3dhp.npz"), allow_pickle=True)['data'].item()
    
    poses_2d_test = []
    poses_3d_test = []
    
    for seq in test_data.keys():
        anim = test_data[seq]
        
        data_3d = anim['data_3d']
        # Root-relative coordinates
        data_3d[:, :14] -= data_3d[:, 14:15]
        data_3d[:, 15:] -= data_3d[:, 14:15]
        
        data_2d = anim['data_2d']
        
        # Different resolutions for different sequences
        if seq == "TS5" or seq == "TS6":
            width, height = 1920, 1080
        else:
            width, height = 2048, 2048
        
        data_2d[..., :2] = normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
        
        # Add confidence scores
        confidence_scores = np.ones((*data_2d.shape[:2], 1))
        data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
        
        poses_2d_test.append(data_2d)
        poses_3d_test.append(data_3d)
    
    return poses_2d_train, poses_3d_train, poses_2d_test, poses_3d_test

def normalize_screen_coordinates(X, w, h):
    """Normalize screen coordinates to [-1, 1] range"""
    assert X.shape[-1] == 2
    # Normalize to [-1, 1]
    X_normalized = X.copy()
    X_normalized[..., 0] = (X_normalized[..., 0] / w) * 2 - 1
    X_normalized[..., 1] = (X_normalized[..., 1] / h) * 2 - 1
    return X_normalized

def init(config):
    batchsize = config['train']['batchsize']
    
    # Load MPI-INF-3DHP data
    data_root = config.get('data_root', 'data/motion3d')
    poses_2d_train, poses_3d_train, poses_2d_test, poses_3d_test = load_mpi_inf_3dhp_data(data_root)
    
    # Create training and validation splits
    # For simplicity, we'll use the first 80% of training data for training and 20% for validation
    n_train = len(poses_2d_train)
    split_idx = int(0.8 * n_train)
    
    train_indices = list(range(split_idx))
    valid_indices = list(range(split_idx, n_train))
    
    # Create datasets
    train_dataset = Dataset(config, poses_2d_train[:split_idx], poses_3d_train[:split_idx], train_indices, train=True)
    valid_dataset = Dataset(config, poses_2d_train[split_idx:], poses_3d_train[split_idx:], 
                           list(range(len(poses_2d_train[split_idx:]))), train=False)

    use_data_loader = config['train']['use_data_loader']

    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, 
                                                  num_workers=config['train']['num_workers'], pin_memory=False)
    loaders['valid'] = torch.utils.data.DataLoader(valid_dataset, batch_size=batchsize, shuffle=False, 
                                                  num_workers=config['train']['num_workers'], pin_memory=False)

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        for i in range(batchnum):
            try:
                imgs, heatmaps = next(loader)
            except StopIteration:
                # to avoid no data provided by dataloader
                loader = loaders[phase].__iter__()
                imgs, heatmaps = next(loader)
            yield {
                'imgs': imgs,
                'heatmaps': heatmaps,
            }

    return lambda key: gen(key)
