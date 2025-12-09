import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import io
import torchvision.transforms as transforms


class TokenDataset(Dataset):
    """
    Dataset that consumes precomputed stroke tokens and optional centered strokes.
    Each item returns image, token sequence per stroke, centers, and resampled centered strokes.
    """

    def __init__(
        self,
        data_root: str,
        tokens_per_stroke: int,
        font_names: Optional[List[str]] = None,
        split: str = "train",
        val_ratio: float = 0.1,
        transform=None,
        target_size=(256, 256),
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokens_per_stroke = tokens_per_stroke
        self.transform = transform
        self.target_size = target_size

        if font_names is None:
            candidates = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
            font_names = []
            for d in candidates:
                if os.path.exists(os.path.join(data_root, d, f"{d}_tokens.pkl")) and os.path.exists(
                    os.path.join(data_root, d, f"{d}_imgs.pkl")
                ):
                    font_names.append(d)
            font_names.sort()
        elif isinstance(font_names, str):
            font_names = [font_names]

        self.samples: List[Tuple[str, str]] = []
        self.font_tokens: Dict[str, Dict] = {}
        self.font_images: Dict[str, Dict] = {}

        rng = random.Random(seed)
        for font in font_names:
            token_path = os.path.join(data_root, font, f"{font}_tokens.pkl")
            img_path = os.path.join(data_root, font, f"{font}_imgs.pkl")
            if not (os.path.exists(token_path) and os.path.exists(img_path)):
                continue

            with open(token_path, "rb") as f:
                token_data = pickle.load(f)
            with open(img_path, "rb") as f:
                image_data = pickle.load(f)

            keys = sorted(list(set(token_data.keys()) & set(image_data.keys())))
            if split in {"train", "val"}:
                rng.shuffle(keys)
                split_idx = int(len(keys) * (1 - val_ratio))
                keys = keys[:split_idx] if split == "train" else keys[split_idx:]

            self.font_tokens[font] = token_data
            self.font_images[font] = image_data
            for k in keys:
                self.samples.append((font, k))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        font, key = self.samples[idx]
        tok_entry = self.font_tokens[font][key]
        img_bytes = self.font_images[font][key]

        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
            ])(image)

        tokens = tok_entry["tokens"]
        centers = torch.tensor(tok_entry.get("centers", []), dtype=torch.float32)
        strokes_centered = tok_entry.get("strokes_centered", [])
        strokes_centered = torch.tensor(strokes_centered, dtype=torch.float32) if strokes_centered else torch.empty(0)

        return {
            "key": key,
            "font_name": font,
            "image": image,
            "tokens": tokens,
            "centers": centers,
            "strokes_centered": strokes_centered,
        }
