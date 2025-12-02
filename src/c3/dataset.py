import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# Try to import video reading library
try:
    import cv2
except ImportError:
    cv2 = None

class CalligraphyDataset(Dataset):
    def __init__(self, root_dir, video_resolution=(256, 256), image_resolution=(256, 256), frame_skip=1, video_transform=None, image_transform=None, cache_dir=None):
        self.root_dir = root_dir
        self.video_dir = os.path.join(root_dir, 'data', 'video')
        self.image_dir = os.path.join(root_dir, 'data', 'figure')
        self.video_resolution = video_resolution
        self.image_resolution = image_resolution
        self.frame_skip = frame_skip
        self.cache_dir = cache_dir
        
        self.video_transform = video_transform
        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(image_resolution),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            self.image_transform = image_transform
        
        self.samples = []
        self._scan_files()

    def _scan_files(self):
        # Recursively find all video files
        # Assuming structure: data/video/category/name.mp4
        # And corresponding image: data/figure/category/name.jpg (or png)
        
        # This is a simplified scanner. You might need to adjust extensions.
        video_extensions = ['*.mp4', '*.avi', '*.mov']
        for ext in video_extensions:
            for video_path in glob.glob(os.path.join(self.video_dir, '**', ext), recursive=True):
                rel_path = os.path.relpath(video_path, self.video_dir)
                # Construct expected image path
                # Remove extension from video name to find image? 
                # Or assume same name with image extension?
                base_name = os.path.splitext(rel_path)[0]
                
                # Check for image with common extensions
                image_found = False
                for img_ext in ['.jpg', '.png', '.jpeg']:
                    image_path = os.path.join(self.image_dir, base_name + img_ext)
                    if os.path.exists(image_path):
                        self.samples.append((video_path, image_path))
                        image_found = True
                        break
                
                if not image_found:
                    # Try searching in the folder if the structure is slightly different
                    pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, image_path = self.samples[idx]
        
        # Check cache
        if self.cache_dir:
            rel_path = os.path.relpath(video_path, self.video_dir)
            cache_path = os.path.join(self.cache_dir, os.path.splitext(rel_path)[0] + '.pt')
            
            if os.path.exists(cache_path):
                try:
                    data = torch.load(cache_path)
                    data['type'] = 'cached'
                    return data
                except Exception:
                    pass

        # Load Image
        image = Image.open(image_path).convert('RGB')
        if self.image_transform:
            image = self.image_transform(image)
            
        # Load Video
        video_tensor = self._load_video(video_path)
        
        return {'type': 'raw', 'video': video_tensor, 'image': image}

    def _load_video(self, path):
        if cv2 is None:
            raise ImportError("OpenCV (cv2) is required to load videos.")
            
        cap = cv2.VideoCapture(path)
        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % self.frame_skip == 0:
                # BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            frame_count += 1
        cap.release()
        
        if len(frames) == 0:
            # Return dummy or handle error
            return torch.zeros(3, 1, self.video_resolution[0], self.video_resolution[1])

        # Stack frames: (T, H, W, C)
        video = np.stack(frames)
        
        # Convert to Tensor: (C, T, H, W)
        video = torch.from_numpy(video).permute(3, 0, 1, 2).float()
        
        # Normalize to [-1, 1]
        video = (video / 127.5) - 1.0
        
        # Resize if necessary (Cosmos Tokenizer expects specific resolution usually, e.g. 256 or 512)
        # Here we assume the user handles resizing or we add a transform.
        video = torch.nn.functional.interpolate(video, size=self.video_resolution, mode='bilinear', align_corners=False)
        
        return video

def collate_fn(batch):
    # Check type of first element
    if batch[0]['type'] == 'raw':
        videos = [b['video'] for b in batch]
        images = [b['image'] for b in batch]
        
        # Find max T
        max_t = max(v.size(1) for v in videos)
        
        padded_videos = []
        masks = [] # 1 for valid, 0 for pad
        
        for v in videos:
            c, t, h, w = v.size()
            pad_t = max_t - t
            if pad_t > 0:
                # Pad temporal dimension
                padding = torch.zeros(c, pad_t, h, w)
                v_padded = torch.cat([v, padding], dim=1)
                mask = torch.cat([torch.ones(t), torch.zeros(pad_t)])
            else:
                v_padded = v
                mask = torch.ones(t)
            
            padded_videos.append(v_padded)
            masks.append(mask)
            
        padded_videos = torch.stack(padded_videos)
        masks = torch.stack(masks).bool()
        images = torch.stack(images)
        
        return padded_videos, masks, images

    elif batch[0]['type'] == 'cached':
        video_tokens = [b['video_tokens'] for b in batch] # List of (t, h, w)
        image_latents = [b['image_latent'] for b in batch] # List of (C, H, W)
        
        # Pad video tokens
        max_t = max(v.size(0) for v in video_tokens)
        
        padded_tokens = []
        masks = []
        
        # Use 64000 (PAD_TOKEN_ID) for padding if available, else 0
        # We should probably import Config, but let's just use 64000 as it is standard in this project
        pad_val = 64000
        
        for v in video_tokens:
            t, h, w = v.size()
            pad_t = max_t - t
            if pad_t > 0:
                padding = torch.full((pad_t, h, w), pad_val, dtype=v.dtype)
                v_padded = torch.cat([v, padding], dim=0)
                mask = torch.cat([torch.ones(t), torch.zeros(pad_t)])
            else:
                v_padded = v
                mask = torch.ones(t)
                
            padded_tokens.append(v_padded)
            masks.append(mask)
            
        padded_tokens = torch.stack(padded_tokens) # (B, T, H, W)
        masks = torch.stack(masks).bool() # (B, T)
        image_latents = torch.stack(image_latents)
        
        return padded_tokens, masks, image_latents
