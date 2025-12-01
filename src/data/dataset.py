import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np
import cv2

class CalligraphyVideoDataset(Dataset):
    def __init__(self, root_dir, transform=None, frames=16, img_size=128):
        """
        root_dir: Directory containing video files.
        """
        self.root_dir = root_dir
        self.video_files = []
        # Recursively find all video files
        extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
        for ext in extensions:
            self.video_files.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
            
        self.transform = transform
        self.frames = frames
        self.img_size = img_size
        print(f"Found {len(self.video_files)} videos in {root_dir}")
        
    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return torch.zeros(3, self.frames, self.img_size, self.img_size)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
             cap.release()
             return torch.zeros(3, self.frames, self.img_size, self.img_size)

        # Uniform sampling
        indices = np.linspace(0, total_frames - 1, self.frames, dtype=int)
        
        frames_list = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.img_size, self.img_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame)
            else:
                if len(frames_list) > 0:
                    frames_list.append(frames_list[-1])
                else:
                    frames_list.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))
        
        cap.release()
        
        while len(frames_list) < self.frames:
             if len(frames_list) > 0:
                frames_list.append(frames_list[-1])
             else:
                frames_list.append(np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8))

        video_np = np.stack(frames_list)
        
        # Normalize to [-1, 1]
        video_np = video_np.astype(np.float32) / 255.0
        video_np = video_np * 2.0 - 1.0
        
        # (T, H, W, C) -> (C, T, H, W)
        video_tensor = torch.from_numpy(video_np).permute(3, 0, 1, 2)
        
        return video_tensor
