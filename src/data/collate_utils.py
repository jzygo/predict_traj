from typing import Dict, List, Tuple

import torch

from .traj_utils import resample_stroke


def center(stroke: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    center_pt = stroke.mean(dim=0) if stroke.numel() > 0 else torch.zeros(2)
    return stroke - center_pt, center_pt


def resample_and_stack(strokes: List[torch.Tensor], num_points: int, point_dim: int = 3) -> Tuple[torch.Tensor, int]:
    """
    将笔画列表堆叠成张量。
    
    注意：如果笔画已在 FontDataset 中预先重采样到 num_points，则直接堆叠；
    否则仍执行 resample_stroke 进行重采样。
    
    Args:
        strokes: 笔画张量列表，每个形状可能是 (N_i, point_dim) 或已经是 (num_points, point_dim)
        num_points: 每个笔画的目标点数
        point_dim: 每个点的维度 (2 或 3)
    
    Returns:
        stacked: 堆叠后的张量，形状 (num_strokes, num_points, point_dim)
        max_points_per_stroke: 原始笔画中的最大点数（用于参考）
    """
    max_points_per_stroke = 0
    resampled = []
    for s in strokes:
        max_points_per_stroke = max(max_points_per_stroke, len(s))
        # 检查笔画是否已经是目标形状
        if len(s) == num_points and s.shape[-1] == point_dim:
            # 已经重采样过，直接使用
            resampled.append(s if isinstance(s, torch.Tensor) else torch.tensor(s, dtype=torch.float32))
        else:
            # 需要重采样（兼容旧数据或未预处理的情况）
            resampled.append(resample_stroke(s, num_points=num_points, point_dim=point_dim))
    if len(resampled) == 0:
        return torch.zeros(0, num_points, point_dim), 0
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
