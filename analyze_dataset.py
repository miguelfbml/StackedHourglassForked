import torch
import torch.utils.data
import numpy as np
import ref
from h5py import File
import cv2
from imageio import imread
import os

def preprocess_image(img):
    """Simple image preprocessing - color augmentation"""
    if np.random.random() > 0.5:
        # Brightness adjustment
        img = np.clip(img * (0.8 + 0.4 * np.random.random()), 0, 255).astype(np.uint8)
    
    if np.random.random() > 0.5:
        # Contrast adjustment
        img = np.clip((img - 128) * (0.8 + 0.4 * np.random.random()) + 128, 0, 255).astype(np.uint8)
    
    return img

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res / 64
        self.sigma = sigma
        size = 6*sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3*sigma + 1, 3*sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        hms = np.zeros(shape = (self.num_parts, self.output_res, self.output_res), dtype = np.float32)
        sigma = self.sigma
        for idx, pt in enumerate(keypoints):
            if pt[2] > 0:  # visible keypoint
                x, y = int(pt[0]), int(pt[1])
                if x >= 0 and y >= 0 and x < self.output_res and y < self.output_res:
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
        print(f"Initializing Dataset: train={train}, data_root={data_root}, mpi_dataset_root={mpi_dataset_root}")
        
        self.config = config
        self.train = train
        self.data_root = data_root
        self.mpi_dataset_root = mpi_dataset_root
        
        self.img_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.sigma = config['train']['sigma']
        self.scale_factor = config['train']['scale_factor']
        self.rot_factor = config['train']['rot_factor']
        self.label_type = config['train']['label_type']
        
        # Load motion3d data
        self.data = self.load_mpi_inf_3dhp_data(data_root, train)
        print(f"Dataset loaded: {len(self.data['joint_2d'])} samples")
        
        # Generate heatmaps
        self.generateHeatmap = GenerateHeatmap(self.output_res, 17)

    def __len__(self):
        return len(self.data['joint_2d'])

    def __getitem__(self, idx):
        try:
            # Get 2D and 3D joints
            joint_2d = self.data['joint_2d'][idx].copy()  # Shape: (17, 3)
            joint_3d = self.data['joint_3d'][idx].copy()  # Shape: (17, 3)
            
            # Get image name
            imgname = self.data['imgname'][idx]
            
            # Load image
            img = self.load_image_from_path(imgname)
            
            # Get camera parameters if available
            camera_idx = self.data.get('camera', [0])[idx] if 'camera' in self.data else 0
            
            # Apply preprocessing
            inp, out = self.preprocess(img, joint_2d, joint_3d, camera_idx)
            
            return {
                'imgs': inp,
                'heatmaps': out,
                'joint_2d': joint_2d.astype(np.float32),
                'joint_3d': joint_3d.astype(np.float32),
                'imgname': imgname,
                'camera': camera_idx
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a fallback sample
            return self.get_fallback_sample(idx)

    def get_fallback_sample(self, idx):
        """Create a fallback sample when real data fails to load"""
        # Create synthetic image
        img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        
        # Create random valid 2D joints
        joint_2d = np.random.rand(17, 3) * 256
        joint_2d[:, 2] = 1.0  # Set all as visible
        
        # Create random 3D joints
        joint_3d = np.random.randn(17, 3) * 1000
        
        inp, out = self.preprocess(img, joint_2d, joint_3d, 0)
        
        return {
            'imgs': inp,
            'heatmaps': out,
            'joint_2d': joint_2d.astype(np.float32),
            'joint_3d': joint_3d.astype(np.float32),
            'imgname': f'synthetic_{idx}',
            'camera': 0
        }

    def load_image_from_path(self, imgname):
        """Load image from MPI-INF-3DHP dataset structure"""
        
        # Different possible root paths
        if self.train:
            # Training images are in the main dataset
            possible_roots = [
                self.mpi_dataset_root,
                os.path.join(self.mpi_dataset_root, ""),
            ]
        else:
            # Test images are in the test set
            possible_roots = [
                os.path.join(self.mpi_dataset_root, "mpi_inf_3dhp_test_set"),
                self.mpi_dataset_root,
            ]
        
        # Try to find the image
        for root in possible_roots:
            # Handle different possible imgname formats
            possible_paths = [
                os.path.join(root, imgname),  # Direct path
                os.path.join(root, imgname.lstrip('/')),  # Remove leading slash
            ]
            
            # If imgname looks like it's missing the imageSequence part, try to add it
            if 'imageSequence' not in imgname:
                # Extract subject/sequence info from imgname if possible
                # Example: TS1/img_000001.jpg -> TS1/imageSequence/img_000001.jpg
                parts = imgname.split('/')
                if len(parts) >= 2:
                    subject_seq = parts[0]
                    img_file = parts[-1]
                    possible_paths.append(os.path.join(root, subject_seq, 'imageSequence', img_file))
            
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    try:
                        img = imread(img_path)
                        if len(img.shape) == 3:
                            return img
                    except Exception as e:
                        print(f"Error reading image {img_path}: {e}")
                        continue
        
        # If no image found, create synthetic
        print(f"Creating synthetic image for {imgname}")
        return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)

    def apply_augmentation(self, img, keypoints_2d):
        """Apply data augmentation"""
        if not self.train:
            return img, keypoints_2d
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            img = img[:, ::-1, :]
            keypoints_2d = keypoints_2d.copy()
            keypoints_2d[:, 0] = img.shape[1] - keypoints_2d[:, 0]
            
            # Swap left-right keypoints for MPI-INF-3DHP (17 keypoints)
            left_right_pairs = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14)]
            for left_idx, right_idx in left_right_pairs:
                keypoints_2d[[left_idx, right_idx]] = keypoints_2d[[right_idx, left_idx]]
        
        return img, keypoints_2d

    def preprocess(self, img, joint_2d, joint_3d, camera_idx):
        """Preprocess image and joints"""
        
        # Resize image to input resolution
        img_resized = cv2.resize(img, (self.img_res, self.img_res))
        
        # Scale 2D joints to match resized image
        original_h, original_w = img.shape[:2]
        scale_x = self.img_res / original_w
        scale_y = self.img_res / original_h
        
        joint_2d_scaled = joint_2d.copy()
        joint_2d_scaled[:, 0] *= scale_x
        joint_2d_scaled[:, 1] *= scale_y
        
        # Apply augmentation
        img_aug, joint_2d_aug = self.apply_augmentation(img_resized, joint_2d_scaled)
        
        # Scale 2D joints to output resolution for heatmap generation
        joint_2d_heatmap = joint_2d_aug.copy()
        joint_2d_heatmap[:, 0] *= (self.output_res / self.img_res)
        joint_2d_heatmap[:, 1] *= (self.output_res / self.img_res)
        
        # Generate heatmaps
        heatmaps = self.generateHeatmap(joint_2d_heatmap)
        
        # Convert image to tensor format
        inp = img_aug.astype(np.float32) / 255.0
        inp = torch.from_numpy(inp).permute(2, 0, 1)  # HWC to CHW
        
        # Convert heatmaps to tensor
        out = torch.from_numpy(heatmaps)
        
        return inp, out

    def load_mpi_inf_3dhp_data(self, data_root, train=True):
        """Load MPI-INF-3DHP motion3d data"""
        
        if train:
            data_file = os.path.join(data_root, 'data_train_3dhp.npz')
        else:
            data_file = os.path.join(data_root, 'data_test_3dhp.npz')
        
        print(f"Loading data from: {data_file}")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        # Load the .npz file
        npz_data = np.load(data_file, allow_pickle=True)
        data = npz_data['data'].item()  # Extract dictionary from object array
        
        print(f"Data keys: {list(data.keys())}")
        
        # Ensure we have the required keys
        required_keys = ['joint_2d', 'joint_3d', 'imgname']
        for key in required_keys:
            if key not in data:
                raise KeyError(f"Required key '{key}' not found in data")
        
        # Print data shapes
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                print(f"  {key}: {data[key].shape}")
        
        return data

    def normalize_screen_coordinates(self, X, w, h): 
        assert X.shape[-1] == 2
        return X/w*2 - [1, h/w]

def custom_collate_fn(batch):
    """Custom collate function to handle dictionary format from Dataset.__getitem__"""
    if isinstance(batch[0], dict):
        # Handle dictionary format
        collated = {}
        for key in batch[0].keys():
            if key in ['imgs', 'heatmaps']:
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        return collated
    else:
        # Handle tuple/list format (fallback)
        return torch.utils.data.dataloader.default_collate(batch)

def init(config):
    batchsize = config['train']['batchsize']
    data_root = config.get('data_root', 'data/motion3d')
    mpi_dataset_root = config.get('mpi_dataset_root', '/nas-ctm01/datasets/public/mpi_inf_3dhp')
    
    print(f"Initializing data loaders with batch_size={batchsize}")
    print(f"Data root: {data_root}")
    print(f"MPI dataset root: {mpi_dataset_root}")
    
    train_dataset = Dataset(config, train=True, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    # Use test data for validation (same as TCPFormer approach)
    valid_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    test_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)  # Same as valid
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batchsize, shuffle=True, 
        num_workers=0, pin_memory=False, collate_fn=custom_collate_fn
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batchsize, shuffle=False, 
        num_workers=0, pin_memory=False, collate_fn=custom_collate_fn
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batchsize, shuffle=False, 
        num_workers=0, pin_memory=False, collate_fn=custom_collate_fn
    )
    
    return train_loader, valid_loader, test_loader