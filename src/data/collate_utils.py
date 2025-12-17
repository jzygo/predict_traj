from typing import Dict, List, Tuple

import torch

from .traj_utils import resample_stroke


def center(stroke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center_pt = stroke.mean(dim=0) if stroke.numel() > 0 else torch.zeros(2)
    return stroke - center_pt, center_pt


def resample_and_stack(strokes: List[torch.Tensor], num_points: int, point_dim: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
    merged_strokes = []
    max_points_per_stroke = 0
    for i, s in enumerate(strokes):
        # if i > 0:
        #     prev_end = strokes[i - 1][-1].clone()
        #     now_start = s[0].clone()
        #     offset = prev_end - now_start
        #     if offset.abs().sum() < 1e-2:
        #         merged_strokes[-1] = torch.cat([merged_strokes[-1], s], dim=0)
        #         # print(f"Merging stroke {i} with previous stroke due to small offset {offset.abs().sum()}")
        #     else:
        #         merged_strokes.append(s)
        # else:
            merged_strokes.append(s)
    resampled = []
    for s in merged_strokes:
        max_points_per_stroke = max(max_points_per_stroke, len(s))
        resampled.append(resample_stroke(s, num_points=num_points, point_dim=point_dim))
    if len(resampled) == 0:
        return torch.zeros(0, num_points, point_dim), torch.zeros(0, point_dim)
    stacked = torch.stack(resampled, dim=0)
    return stacked, max_points_per_stroke


def pad_stroke_batch(batch: List[Dict], num_points: int, max_strokes: int, point_dim: int = 3) -> Dict[str, torch.Tensor]:
    images = torch.stack([item["image"] for item in batch])
    all_resampled = []
    all_masks = []
    lengths = []
    max_points_per_stroke = 0
    for item in batch:
        strokes = item["strokes"]
        resampled, cur_max_points = resample_and_stack(strokes, num_points, point_dim=point_dim)
        max_points_per_stroke = max(max_points_per_stroke, cur_max_points)
        cur_len = resampled.size(0)
        lengths.append(cur_len)

        pad_len = max_strokes - cur_len
        if pad_len > 0:
            pad = torch.zeros(pad_len, num_points, point_dim)
            resampled = torch.cat([resampled, pad], dim=0)

        resampled = resampled[:max_strokes]
        all_resampled.append(resampled)

        mask = torch.zeros(max_strokes + 1, dtype=torch.bool)
        eos_idx = min(cur_len, max_strokes)
        mask[eos_idx + 1 :] = True
        all_masks.append(mask)
    strokes_tensor = torch.stack(all_resampled)
    attn_mask = torch.stack(all_masks)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long).clamp(max=max_strokes)
    return {
        "images": images,
        "strokes": strokes_tensor,
        "stroke_lengths": lengths_tensor,
        "attn_mask": attn_mask,
        "max_points_per_stroke": max_points_per_stroke,
        "key": [item["key"] for item in batch],
        "font_name": [item["font_name"] for item in batch],
    }
