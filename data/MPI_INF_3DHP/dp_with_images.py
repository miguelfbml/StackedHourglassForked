import torch
import torch.utils.data
import numpy as np
import cv2
from imageio import imread
import os

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
        print(f"Initializing MPI-INF-3DHP Dataset: train={train}")
        
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, 17)
        self.train = train
        self.mpi_dataset_root = mpi_dataset_root
        
        # Load annotations
        self.samples = self.load_annotations(data_root, train)
        print(f"Loaded {len(self.samples)} samples")

    def load_annotations(self, data_root, train):
        """Load MPI-INF-3DHP annotations"""
        if train:
            data_file = os.path.join(data_root, 'data_train_3dhp.npz')
        else:
            data_file = os.path.join(data_root, 'data_test_3dhp.npz')
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading annotations from: {data_file}")
        npz_data = np.load(data_file, allow_pickle=True)
        raw_data = npz_data['data'].item()
        
        samples = []
        total_images_found = 0
        total_images_missing = 0
        
        if train:
            print("\n=== TRAINING DATA ANALYSIS ===")
            print("Processing ALL subjects (S1-S8) and ALL sequences (Seq1-Seq2)...")
            
            # Training data: {'S1 Seq1': [...], 'S2 Seq1': [...], ...}
            for subject_seq, seq_data in raw_data.items():
                print(f"\nProcessing {subject_seq}")
                parts = subject_seq.split(' ')
                subject = parts[0]  # 'S1'
                sequence = parts[1]  # 'Seq1'
                
                subject_images_found = 0
                subject_images_missing = 0
                
                # seq_data is a list containing dictionaries and integers
                if isinstance(seq_data, list):
                    for seq_item in seq_data:
                        # Only process dictionary items (skip integers)
                        if isinstance(seq_item, dict):
                            # This dict has camera keys like '0', '1', etc.
                            for camera_str, camera_data in seq_item.items():
                                if isinstance(camera_str, str) and camera_str.isdigit():
                                    camera_idx = int(camera_str)
                                    
                                    if isinstance(camera_data, dict) and 'data_2d' in camera_data and 'data_3d' in camera_data:
                                        data_2d = camera_data['data_2d']
                                        data_3d = camera_data['data_3d']
                                        
                                        camera_images_found = 0
                                        camera_images_missing = 0
                                        
                                        # Sample every 50th frame to keep dataset manageable
                                        for frame_idx in range(0, len(data_2d), 50):
                                            img_path = self.find_image_path(subject, sequence, frame_idx + 1, camera_idx, train=True)
                                            if img_path and os.path.exists(img_path):
                                                camera_images_found += 1
                                                samples.append({
                                                    'img_path': img_path,
                                                    'joint_2d': data_2d[frame_idx],
                                                    'joint_3d': data_3d[frame_idx],
                                                    'subject': subject,
                                                    'sequence': sequence,
                                                    'frame_idx': frame_idx + 1,
                                                    'camera': camera_idx
                                                })
                                            else:
                                                camera_images_missing += 1
                                        
                                        if camera_images_found > 0 or camera_images_missing > 0:
                                            print(f"  Camera {camera_idx}: {camera_images_found} images found, {camera_images_missing} missing (sampled {len(data_2d)//50} from {len(data_2d)} frames)")
                                        
                                        subject_images_found += camera_images_found
                                        subject_images_missing += camera_images_missing
                
                if subject_images_found > 0 or subject_images_missing > 0:
                    print(f"  {subject_seq} TOTAL: {subject_images_found} images found, {subject_images_missing} missing")
                
                total_images_found += subject_images_found
                total_images_missing += subject_images_missing
                    
        else:
            print("\n=== TEST DATA ANALYSIS ===")
            print("Processing ALL test subjects (TS1-TS6)...")
            
            # Test data: {'TS1': {...}, 'TS2': {...}, ...} - Include all samples
            for subject, subject_data in raw_data.items():
                print(f"\nProcessing {subject}")
                
                if isinstance(subject_data, dict) and 'data_2d' in subject_data and 'data_3d' in subject_data:
                    data_2d = subject_data['data_2d']
                    data_3d = subject_data['data_3d']
                    
                    subject_images_found = 0
                    subject_images_missing = 0
                    
                    # Take every 20th frame for test
                    for frame_idx in range(0, len(data_2d), 20):
                        img_path = self.find_image_path(subject, None, frame_idx + 1, 0, train=False)
                        if img_path and os.path.exists(img_path):
                            subject_images_found += 1
                            samples.append({
                                'img_path': img_path,
                                'joint_2d': data_2d[frame_idx],
                                'joint_3d': data_3d[frame_idx],
                                'subject': subject,
                                'sequence': 'test',
                                'frame_idx': frame_idx + 1,
                                'camera': 0
                            })
                        else:
                            subject_images_missing += 1
                    
                    print(f"  {subject}: {subject_images_found} images found, {subject_images_missing} missing (sampled {len(data_2d)//20} from {len(data_2d)} frames)")
                    total_images_found += subject_images_found
                    total_images_missing += subject_images_missing
        
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Total images found: {total_images_found}")
        print(f"Total images missing: {total_images_missing}")
        print(f"Total samples created: {len(samples)}")
        print(f"Success rate: {total_images_found/(total_images_found+total_images_missing)*100:.1f}%" if (total_images_found+total_images_missing) > 0 else "No images checked")
        
        if train:
            # Count subjects and sequences
            subjects = set()
            sequences = set()
            cameras = set()
            for sample in samples:
                subjects.add(sample['subject'])
                sequences.add(sample['sequence'])
                cameras.add(sample['camera'])
            print(f"Subjects included: {sorted(subjects)}")
            print(f"Sequences included: {sorted(sequences)}")
            print(f"Cameras included: {sorted(cameras)}")
        
        return samples

    def find_image_path(self, subject, sequence, frame_idx, camera_idx, train=True):
        """Find the actual image file path"""
        if train:
            # Training: S1/Seq1/imageFrames/video_0/frame_000001.jpg
            base_path = os.path.join(self.mpi_dataset_root, subject, sequence, "imageFrames", f"video_{camera_idx}")
            img_name = f"frame_{frame_idx:06d}.jpg"
        else:
            # Test: mpi_inf_3dhp_test_set/TS1/imageSequence/img_000001.jpg
            base_path = os.path.join(self.mpi_dataset_root, "mpi_inf_3dhp_test_set", subject, "imageSequence")
            img_name = f"img_{frame_idx:06d}.jpg"
        
        img_path = os.path.join(base_path, img_name)
        return img_path if os.path.exists(img_path) else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            
            # Load image
            img = imread(sample['img_path'])
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            
            # Get 2D keypoints
            joint_2d = sample['joint_2d'].copy()  # Shape: (17, 2)
            
            # Resize image to input resolution
            original_h, original_w = img.shape[:2]
            img_resized = cv2.resize(img, (self.input_res, self.input_res))
            
            # Scale keypoints to match resized image
            scale_x = self.input_res / original_w
            scale_y = self.input_res / original_h
            
            joint_2d_scaled = joint_2d.copy()
            joint_2d_scaled[:, 0] *= scale_x
            joint_2d_scaled[:, 1] *= scale_y
            
            # Add visibility (assume all visible)
            joint_2d_vis = np.ones((17, 3))
            joint_2d_vis[:, :2] = joint_2d_scaled
            
            # Apply random augmentation for training
            if self.train and np.random.random() > 0.5:
                # Random horizontal flip
                img_resized = img_resized[:, ::-1, :]
                joint_2d_vis[:, 0] = self.input_res - joint_2d_vis[:, 0]
                
                # Swap left-right keypoints for MPI-INF-3DHP
                left_right_pairs = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14)]
                for left_idx, right_idx in left_right_pairs:
                    joint_2d_vis[[left_idx, right_idx]] = joint_2d_vis[[right_idx, left_idx]]
            
            # Scale keypoints to output resolution for heatmap generation
            joint_2d_heatmap = joint_2d_vis.copy()
            joint_2d_heatmap[:, 0] *= (self.output_res / self.input_res)
            joint_2d_heatmap[:, 1] *= (self.output_res / self.input_res)
            
            # Generate heatmaps
            heatmaps = self.generateHeatmap(joint_2d_heatmap)
            
            # Convert to tensors
            inp = img_resized.astype(np.float32) / 255.0
            inp = torch.from_numpy(inp).permute(2, 0, 1)  # HWC to CHW
            out = torch.from_numpy(heatmaps)
            
            return {
                'imgs': inp,
                'heatmaps': out
            }
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a synthetic sample as fallback
            synthetic_img = torch.randn(3, self.input_res, self.input_res)
            synthetic_heatmaps = torch.zeros(17, self.output_res, self.output_res)
            return {
                'imgs': synthetic_img,
                'heatmaps': synthetic_heatmaps
            }

def custom_collate_fn(batch):
    """Custom collate function for batch processing"""
    imgs = torch.stack([item['imgs'] for item in batch])
    heatmaps = torch.stack([item['heatmaps'] for item in batch])
    return {
        'imgs': imgs,
        'heatmaps': heatmaps
    }

def init(config):
    """Initialize data loaders"""
    batchsize = config['train']['batchsize']
    data_root = config.get('data_root', 'data/motion3d')
    mpi_dataset_root = config.get('mpi_dataset_root', '/nas-ctm01/datasets/public/mpi_inf_3dhp')
    
    print(f"Initializing MPI-INF-3DHP data loaders")
    print(f"Data root: {data_root}")
    print(f"MPI dataset root: {mpi_dataset_root}")
    
    train_dataset = Dataset(config, train=True, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    valid_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    test_dataset = Dataset(config, train=False, data_root=data_root, mpi_dataset_root=mpi_dataset_root)
    
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