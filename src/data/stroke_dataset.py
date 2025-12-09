import os
import pickle
import random
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .traj_utils import resample_stroke


def center_stroke(stroke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # stroke: (N, 2)
    center = stroke.mean(dim=0) if stroke.numel() > 0 else torch.zeros(2)
    centered = stroke - center
    return centered, center


class StrokeDataset(Dataset):
    """Flat stroke-level dataset for VQ-VAE training.

    Loads pre-processed font pickles and returns centered, resampled strokes.
    """

    def __init__(
        self,
        data_root: str,
        font_names: Optional[List[str]] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        num_points: int = 24,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.num_points = num_points

        if font_names is None:
            candidates = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            font_names = []
            for d in candidates:
                if os.path.exists(os.path.join(data_root, d, f"{d}_strokes.pkl")):
                    font_names.append(d)
            font_names.sort()
        elif isinstance(font_names, str):
            font_names = [font_names]

        self.samples: List[Tuple[str, str, torch.Tensor]] = []
        rng = random.Random(seed)

        for font in font_names:
            stroke_path = os.path.join(data_root, font, f"{font}_strokes.pkl")
            if not os.path.exists(stroke_path):
                continue
            with open(stroke_path, "rb") as f:
                font_data = pickle.load(f)

            keys = sorted(font_data.keys())
            if split in {"train", "val"}:
                rng.shuffle(keys)
                split_idx = int(len(keys) * (1 - val_ratio))
                keys = keys[:split_idx] if split == "train" else keys[split_idx:]

            for key in keys:
                strokes = font_data[key]
                for s in strokes:
                    self.samples.append((font, key, torch.tensor(s, dtype=torch.float32)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        _, key, stroke = self.samples[idx]
        centered, center = center_stroke(stroke)
        resampled = resample_stroke(centered, num_points=self.num_points)
        return {
            "key": key,
            "stroke": resampled,
            "center": center,
        }
