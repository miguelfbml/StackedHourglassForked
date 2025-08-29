import cv2
import torch
import numpy as np
import torch.utils.data
import utils.img
from os.path import join
import glob

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
    def __init__(self, config, train=True, data_root="data/motion3d"):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.train = train
        
        # Load 2D poses from motion3d files (same as TCPFormer)
        self.poses_2d = {}
        self.valid_frames = {}
        
        if train:
            # Training data - use same sequences as TCPFormerForked
            subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
            sequences = ['Seq1', 'Seq2'] 
            cameras = [0, 1, 2, 4, 5, 6, 7, 8]
            
            for subject in subjects:
                for seq in sequences:
                    for cam in cameras:
                        data_file = f"{data_root}/train_{subject}_{seq}_camera_{cam:02d}.npz"
                        if not glob.glob(data_file):
                            continue
                        try:
                            data = np.load(data_file)
                            poses_2d = data['poses_2d']  # Shape: [frames, 17, 2]
                            valid = data['valid_frames'] if 'valid_frames' in data else np.ones(poses_2d.shape[0], dtype=bool)
                            
                            # Store with same key format as TCPFormer
                            key = (subject, seq, cam)
                            self.poses_2d[key] = poses_2d
                            self.valid_frames[key] = valid
                        except:
                            continue
        else:
            # Test data
            test_sequences = ['TS1', 'TS2', 'TS3', 'TS4', 'TS5', 'TS6']
            for seq in test_sequences:
                data_file = f"{data_root}/test_{seq}.npz"
                if not glob.glob(data_file):
                    continue
                try:
                    data = np.load(data_file)
                    poses_2d = data['poses_2d']
                    valid = data['valid_frames'] if 'valid_frames' in data else np.ones(poses_2d.shape[0], dtype=bool)
                    
                    self.poses_2d[seq] = poses_2d
                    self.valid_frames[seq] = valid
                except:
                    continue
        
        # Create frame index for iteration
        self.frame_index = []
        for key in self.poses_2d.keys():
            num_frames = self.poses_2d[key].shape[0]
            valid_frames = self.valid_frames[key] if key in self.valid_frames else np.ones(num_frames, dtype=bool)
            
            # Sample frames based on valid frames
            if train:
                # Use all valid frames for training
                valid_indices = np.where(valid_frames)[0]
            else:
                # Sample every 10th frame for validation
                valid_indices = np.where(valid_frames)[0][::10]
                
            for frame_idx in valid_indices:
                self.frame_index.append((key, frame_idx))
        
        print(f"Loaded {len(self.frame_index)} frames for {'training' if train else 'validation'}")
        
        # Left-right keypoint mapping for H36M/MPI format (17 keypoints)
        self.kps_left = [5, 6, 7, 11, 12, 13]  # Left hip, knee, ankle, shoulder, elbow, wrist
        self.kps_right = [2, 3, 4, 8, 9, 10]   # Right hip, knee, ankle, shoulder, elbow, wrist

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        sequence_key, frame_idx = self.frame_index[idx]
        
        # Get 2D poses from the dataset (normalized coordinates)
        pose_2d = self.poses_2d[sequence_key][frame_idx]  # Shape: (17, 2)
        
        # Generate a simple synthetic image (like original StackedHourglass does)
        img = self.generate_synthetic_image()
        
        # Convert normalized coordinates to pixel coordinates for heatmap generation
        # MPI-INF-3DHP coordinates are normalized to [-1, 1]
        keypoints_2d_pixels = np.zeros_like(pose_2d)
        keypoints_2d_pixels[:, 0] = (pose_2d[:, 0] + 1.0) * (self.output_res / 2.0)
        keypoints_2d_pixels[:, 1] = (pose_2d[:, 1] + 1.0) * (self.output_res / 2.0)
        
        # Data augmentation for training
        if self.train:
            img, keypoints_2d_pixels = self.apply_augmentation(img, keypoints_2d_pixels)
        
        # Prepare keypoints for heatmap generation
        keypoints = np.zeros((1, 17, 3))
        keypoints[0, :, :2] = keypoints_2d_pixels
        keypoints[0, :, 2] = 1.0  # All keypoints are valid
        
        # Generate heatmaps
        heatmaps = self.generateHeatmap(keypoints)
        
        # Convert image to tensor format (H, W, C) -> (C, H, W) 
        img = img.transpose(2, 0, 1)
        
        return img.astype(np.float32), heatmaps.astype(np.float32)

    def generate_synthetic_image(self):
        """Generate a simple synthetic image like the original StackedHourglass"""
        img = np.random.rand(self.input_res, self.input_res, 3) * 0.1 + 0.4
        return img.astype(np.float32)

    def apply_augmentation(self, img, keypoints_2d):
        """Apply data augmentation to image and keypoints"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, ::-1, :]
            keypoints_2d[:, 0] = self.output_res - keypoints_2d[:, 0]
            # Swap left and right keypoints
            keypoints_2d_copy = keypoints_2d.copy()
            keypoints_2d[self.kps_left] = keypoints_2d_copy[self.kps_right]
            keypoints_2d[self.kps_right] = keypoints_2d_copy[self.kps_left]
        
        # Color augmentation
        img = self.preprocess(img)
        
        return img, keypoints_2d

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV)
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:, :, 0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:, :, 1] *= delta_sature
        data[:,:, 1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean

        # clip values
        data = np.clip(data, 0, 1)
        return data

def init(config):
    batchsize = config['train']['batchsize']
    current_path = config['opt'].data_root

    dataset = {
        'train': Dataset(config, train=True, data_root=current_path),
        'valid': Dataset(config, train=False, data_root=current_path)
    }
    
    loaders = {}
    for key in dataset:
        loaders[key] = torch.utils.data.DataLoader(
            dataset[key], 
            batch_size=batchsize, 
            shuffle=(key == 'train'), 
            num_workers=config['train']['num_workers'], 
            pin_memory=False,
            drop_last=True
        )

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        
        for i in range(batchnum):
            try:
                imgs, heatmaps = next(loader)
                yield {
                    'imgs': imgs,
                    'heatmaps': heatmaps,
                }
            except StopIteration:
                # Restart loader if we reach the end
                loader = loaders[phase].__iter__()
                imgs, heatmaps = next(loader)
                yield {
                    'imgs': imgs,
                    'heatmaps': heatmaps,
                }

    return lambda key: gen(key)
