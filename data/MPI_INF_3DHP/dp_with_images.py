import cv2
import sys
import os
import torch
import numpy as np
import torch.utils.data
from imageio import imread
import glob

def preprocess_image(img):
    """Simple image preprocessing - color augmentation"""
    if np.random.random() > 0.5:
        # Random brightness
        img = img * (0.8 + np.random.random() * 0.4)
        img = np.clip(img, 0, 1)
    
    if np.random.random() > 0.5:
        # Random contrast
        mean = np.mean(img)
        img = (img - mean) * (0.8 + np.random.random() * 0.4) + mean
        img = np.clip(img, 0, 1)
    
    return img

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
    def __init__(self, config, train=True, data_root="data/motion3d", mpi_dataset_root="/nas-ctm01/datasets/public/mpi_inf_3dhp"):
        print(f"[DEBUG] Initializing Dataset with train={train}")
        print(f"[DEBUG] data_root: {data_root}")
        print(f"[DEBUG] mpi_dataset_root: {mpi_dataset_root}")
        
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.train = train
        self.data_root = data_root
        self.mpi_dataset_root = mpi_dataset_root
        
        # Load pose data from npz files
        print(f"Loading MPI-INF-3DHP data from {data_root}")
        if train:
            print(f"[DEBUG] Loading training data...")
            self.poses_3d, self.poses_2d = self.load_mpi_inf_3dhp_data(data_root, train=True)
            if not self.poses_2d:
                raise ValueError(f"No training data found in {data_root}")
        else:
            print(f"[DEBUG] Loading validation data...")
            self.poses_3d, self.poses_2d, self.valid_frames = self.load_mpi_inf_3dhp_data(data_root, train=False)
            if not self.poses_2d:
                raise ValueError(f"No validation data found in {data_root}")
        
        print(f"Loaded data for {len(self.poses_2d)} sequences")
        
        # Create index mapping for frames
        print(f"[DEBUG] Creating frame index...")
        self.frame_index = []
        self.sequence_info = {}
        
        if train:
            print(f"[DEBUG] Processing training sequences...")
            # Training data: S1-S8, Seq1-Seq2, cameras 0,1,2,4,5,6,7,8
            for key in self.poses_2d.keys():
                subject_name, seq_name, cam = key
                num_frames = self.poses_2d[key].shape[0]
                
                # Sample every 10th frame to reduce dataset size and avoid missing images
                # Use only 20% of training data to balance with validation set
                max_frame = int(num_frames * 0.2)  # Reduced from 0.8
                sampled_frames = 0
                for frame_idx in range(0, max_frame, 5):  # Sample every 5th frame instead of 10th
                    self.frame_index.append((key, frame_idx))
                    sampled_frames += 1
                    
                print(f"  - {key}: {num_frames} total frames, sampled {sampled_frames}")
                    
                # Store sequence info for image loading
                self.sequence_info[key] = {
                    'subject': subject_name,
                    'sequence': seq_name, 
                    'camera': cam,
                    'image_dir': f"{mpi_dataset_root}/{subject_name}/Seq{seq_name}/imageSequence/video_{cam}"
                }
        else:
            print(f"[DEBUG] Processing validation sequences...")
            # Test data
            for key in self.poses_2d.keys():
                sequence_name = key
                num_frames = self.poses_2d[key].shape[0]
                valid_frame_indices = np.where(self.valid_frames[key])[0] if key in self.valid_frames else range(num_frames)
                
                # Limit to first 80% of frames and sample every 10th frame
                max_frame = int(num_frames * 0.8)
                valid_frame_indices = [f for f in valid_frame_indices if f < max_frame]
                
                sampled_frames = 0
                for frame_idx in valid_frame_indices[::10]:  # Sample every 10th frame
                    self.frame_index.append((key, frame_idx))
                    sampled_frames += 1
                    
                print(f"  - {key}: {num_frames} total frames, {len(valid_frame_indices)} valid, sampled {sampled_frames}")
                    
                # Store sequence info for test image loading
                self.sequence_info[key] = {
                    'sequence': sequence_name,
                    'image_dir': f"{mpi_dataset_root}/mpi_inf_3dhp_test_set/{sequence_name}/imageSequence"
                }
        
        print(f"Loaded {len(self.frame_index)} frames for {'training' if train else 'validation'}")
        print(f"[DEBUG] Dataset initialization complete")
        
        # Left-right keypoint mapping for H36M/MPI format (17 keypoints)
        self.kps_left = [5, 6, 7, 11, 12, 13]  # Left hip, knee, ankle, shoulder, elbow, wrist
        self.kps_right = [2, 3, 4, 8, 9, 10]   # Right hip, knee, ankle, shoulder, elbow, wrist

    def __len__(self):
        return len(self.frame_index)

    def __getitem__(self, idx):
        # Try multiple times to get a valid sample with real image
        max_attempts = 50  # Increased attempts to find real images
        for attempt in range(max_attempts):
            try:
                result = self.loadImage(self.frame_index[(idx + attempt) % len(self.frame_index)])
                if result is not None:
                    return result
            except Exception as e:
                print(f"Error loading sample {idx + attempt}: {e}")
                continue
        
        # If all attempts fail, create a dummy sample to avoid crashing
        print(f"Warning: Could not find real images after {max_attempts} attempts")
        dummy_img = np.zeros((self.input_res, self.input_res, 3), dtype=np.float32)
        dummy_heatmaps = np.zeros((17, self.output_res, self.output_res), dtype=np.float32)
        return {
            'imgs': dummy_img.astype(np.float32), 
            'heatmaps': dummy_heatmaps.astype(np.float32)
        }

    def loadImage(self, data_tuple):
        sequence_key, frame_idx = data_tuple
        
        # Get 2D poses from the dataset (normalized coordinates)
        pose_2d = self.poses_2d[sequence_key][frame_idx]  # Shape: (17, 3) where last dim is (x, y, confidence)
        keypoints_2d = pose_2d[:, :2].copy()  # (17, 2)
        confidence = pose_2d[:, 2].copy()  # (17,)
        
        # Load the actual image
        img = self.load_image_frame(sequence_key, frame_idx)
        
        if img is None:
            # Skip this sample if no real image is available
            # This prevents training on synthetic data which would hurt accuracy
            return None
        else:
            # Resize image to input resolution
            img = cv2.resize(img, (self.input_res, self.input_res))
            img = img.astype(np.float32) / 255.0
        
        # Convert normalized coordinates [-1, 1] to output resolution coordinates [0, output_res]
        keypoints_2d_pixels = np.zeros_like(keypoints_2d)
        keypoints_2d_pixels[:, 0] = (keypoints_2d[:, 0] + 1.0) * (self.output_res / 2.0)
        keypoints_2d_pixels[:, 1] = (keypoints_2d[:, 1] + 1.0) * (self.output_res / 2.0)
        
        # Clamp coordinates to valid range
        keypoints_2d_pixels[:, 0] = np.clip(keypoints_2d_pixels[:, 0], 0, self.output_res - 1)
        keypoints_2d_pixels[:, 1] = np.clip(keypoints_2d_pixels[:, 1], 0, self.output_res - 1)
        
        # Data augmentation for training
        if self.train:
            img, keypoints_2d_pixels = self.apply_augmentation(img, keypoints_2d_pixels)
        
        # Prepare keypoints for heatmap generation - ensure valid coordinates
        keypoints = np.zeros((1, 17, 3))
        keypoints[0, :, :2] = keypoints_2d_pixels
        
        # Set visibility based on coordinate validity and original confidence
        for i in range(17):
            if (keypoints_2d_pixels[i, 0] > 0 and keypoints_2d_pixels[i, 1] > 0 and 
                keypoints_2d_pixels[i, 0] < self.output_res and keypoints_2d_pixels[i, 1] < self.output_res and
                confidence[i] > 0.1):
                keypoints[0, i, 2] = 1.0  # Visible
            else:
                keypoints[0, i, 2] = 0.0  # Not visible
                keypoints[0, i, :2] = 0.0  # Set coordinates to 0 for invisible points
        
        # Generate heatmaps
        heatmaps = self.generateHeatmap(keypoints)
        
        return {
            'imgs': img.astype(np.float32),
            'heatmaps': heatmaps.astype(np.float32)
        }

    def load_image_frame(self, sequence_key, frame_idx):
        """Load actual image frame from MPI-INF-3DHP dataset"""
        try:
            if self.train:
                # Training data structure: S{1-8}/Seq{1-2}/imageSequence/video_{camera}/frame_{frame_idx:06d}.jpg
                subject_name, seq_name, cam = sequence_key
                image_dir = self.sequence_info[sequence_key]['image_dir']
                
                # Try different naming conventions
                possible_names = [
                    f"frame_{frame_idx:06d}.jpg",
                    f"img_{frame_idx:06d}.jpg", 
                    f"{frame_idx:06d}.jpg",
                    f"frame_{frame_idx:05d}.jpg",
                    f"img_{frame_idx:05d}.jpg",
                    f"{frame_idx:05d}.jpg"
                ]
                
                for name in possible_names:
                    image_path = f"{image_dir}/{name}"
                    if os.path.exists(image_path):
                        img = imread(image_path)
                        if len(img.shape) == 2:  # Grayscale
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif img.shape[2] == 4:  # RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        return img
                        
            else:
                # Test data structure: mpi_inf_3dhp_test_set/{sequence}/imageSequence/img_{frame_idx:06d}.jpg
                sequence_name = sequence_key
                image_dir = self.sequence_info[sequence_key]['image_dir']
                
                possible_names = [
                    f"img_{frame_idx:06d}.jpg",
                    f"frame_{frame_idx:06d}.jpg",
                    f"{frame_idx:06d}.jpg",
                    f"img_{frame_idx:05d}.jpg",
                    f"frame_{frame_idx:05d}.jpg",
                    f"{frame_idx:05d}.jpg"
                ]
                
                for name in possible_names:
                    image_path = f"{image_dir}/{name}"
                    if os.path.exists(image_path):
                        img = imread(image_path)
                        if len(img.shape) == 2:  # Grayscale
                            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                        elif img.shape[2] == 4:  # RGBA
                            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                        return img
            
            return None
                
        except Exception as e:
            print(f"Error loading image {sequence_key} frame {frame_idx}: {e}")
            return None

    def apply_augmentation(self, img, keypoints_2d):
        """Apply data augmentation to image and keypoints"""
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, ::-1, :]
            keypoints_2d[:, 0] = self.output_res - 1 - keypoints_2d[:, 0]
            # Swap left and right keypoints
            keypoints_2d_copy = keypoints_2d.copy()
            for i, j in zip(self.kps_left, self.kps_right):
                keypoints_2d[i] = keypoints_2d_copy[j]
                keypoints_2d[j] = keypoints_2d_copy[i]
        
        # Color augmentation
        img = preprocess_image(img)
        
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
        data = np.minimum(np.maximum(data, 0), 1)
        return data

    def load_mpi_inf_3dhp_data(self, data_root, train=True):
        """Load MPI-INF-3DHP dataset similar to TCPFormerForked"""
        
        if train:
            data_file = os.path.join(data_root, "data_train_3dhp.npz")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Training data file not found: {data_file}")
                
            data = np.load(data_file, allow_pickle=True)['data'].item()
            
            out_poses_3d = {}
            out_poses_2d = {}
            
            for seq in data.keys():
                for cam in data[seq][0].keys():
                    anim = data[seq][0][cam]
                    
                    subject_name, seq_name = seq.split(" ")
                    
                    data_3d = anim['data_3d']
                    data_3d[:, :14] -= data_3d[:, 14:15]
                    data_3d[:, 15:] -= data_3d[:, 14:15]
                    out_poses_3d[(subject_name, seq_name, cam)] = data_3d
                    
                    data_2d = anim['data_2d']
                    # Normalize screen coordinates to [-1, 1]
                    data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=2048, h=2048)
                    
                    confidence_scores = np.ones((*data_2d.shape[:2], 1))
                    data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                    
                    out_poses_2d[(subject_name, seq_name, cam)] = data_2d
            
            print(f"Loaded training data for {len(out_poses_2d)} camera sequences")
            return out_poses_3d, out_poses_2d
        else:
            data_file = os.path.join(data_root, "data_test_3dhp.npz")
            if not os.path.exists(data_file):
                raise FileNotFoundError(f"Test data file not found: {data_file}")
                
            data = np.load(data_file, allow_pickle=True)['data'].item()
            
            out_poses_3d = {}
            out_poses_2d = {}
            valid_frame = {}
            
            for seq in data.keys():
                anim = data[seq]
                
                valid_frame[seq] = anim["valid"]
                
                data_3d = anim['data_3d']
                data_3d[:, :14] -= data_3d[:, 14:15]
                data_3d[:, 15:] -= data_3d[:, 14:15]
                out_poses_3d[seq] = data_3d
                
                data_2d = anim['data_2d']
                
                if seq == "TS5" or seq == "TS6":
                    width = 1920
                    height = 1080
                else:
                    width = 2048
                    height = 2048
                data_2d[..., :2] = self.normalize_screen_coordinates(data_2d[..., :2], w=width, h=height)
                
                confidence_scores = np.ones((*data_2d.shape[:2], 1))
                data_2d = np.concatenate((data_2d, confidence_scores), axis=-1)
                out_poses_2d[seq] = data_2d
            
            print(f"Loaded test data for {len(out_poses_2d)} sequences")
            return out_poses_3d, out_poses_2d, valid_frame

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        result = X.copy()
        result[..., 0] = (X[..., 0] / w) * 2 - 1
        result[..., 1] = (X[..., 1] / h) * 2 - 1
        return result


def custom_collate_fn(batch):
    """Custom collate function to handle dictionary format from Dataset.__getitem__"""
    if isinstance(batch[0], dict):
        # Handle dictionary format
        imgs = torch.stack([torch.from_numpy(item['imgs']) for item in batch])
        heatmaps = torch.stack([torch.from_numpy(item['heatmaps']) for item in batch])
        return {
            'imgs': imgs,
            'heatmaps': heatmaps
        }
    else:
        # Handle tuple format (legacy)
        imgs, heatmaps = zip(*batch)
        imgs = torch.stack([torch.from_numpy(img) for img in imgs])
        heatmaps = torch.stack([torch.from_numpy(hm) for hm in heatmaps])
        return {
            'imgs': imgs, 
            'heatmaps': heatmaps
        }


def init(config):
    batchsize = config['train']['batchsize']
    data_root = config.get('data_root', 'data/motion3d')
    mpi_dataset_root = config.get('mpi_dataset_root', '/nas-ctm01/datasets/public/mpi_inf_3dhp')
    
    train_dataset = Dataset(config, train=True, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    # Use test data for validation (same as TCPFormer approach)
    valid_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    test_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)  # Same as valid
    
    dataset = {'train': train_dataset, 'valid': valid_dataset, 'test': test_dataset}
    
    loaders = {}
    for key in dataset:
        shuffle = (key == 'train')
        loaders[key] = torch.utils.data.DataLoader(
            dataset[key], 
            batch_size=batchsize, 
            shuffle=shuffle, 
            num_workers=2 if key == 'train' else 1,  # Reduce workers to avoid issues
            pin_memory=True,
            drop_last=True,  # Drop incomplete batches
            collate_fn=custom_collate_fn  # Use custom collate function
        )

    def gen(phase):
        batchsize = config['train']['batchsize']
        batchnum = config['train']['{}_iters'.format(phase)]
        loader = loaders[phase].__iter__()
        
        for i in range(batchnum):
            try:
                batch = next(loader)
                if isinstance(batch, dict):
                    # New format with dict
                    yield batch
                else:
                    # Legacy format with tuple
                    imgs, heatmaps = batch
                    yield {
                        'imgs': imgs,
                        'heatmaps': heatmaps,
                    }
            except StopIteration:
                # Restart loader if we reach the end
                loader = loaders[phase].__iter__()
                batch = next(loader)
                if isinstance(batch, dict):
                    yield batch
                else:
                    imgs, heatmaps = batch
                    yield {
                        'imgs': imgs,
                        'heatmaps': heatmaps,
                    }

    return lambda key: gen(key)
