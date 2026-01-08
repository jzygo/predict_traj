import torch
import numpy as np

def resample_stroke(stroke, num_points=64, point_dim=3):
    """
    Resamples a single stroke to a fixed number of points.
    Args:
        stroke: np.array or torch.Tensor of shape [N, D] where D is 2 (x, y) or 3 (x, y, r)
        num_points: int
        point_dim: int, dimension of each point (2 or 3)
    Returns:
        torch.Tensor of shape [num_points, point_dim]
    """
    if isinstance(stroke, torch.Tensor):
        stroke = stroke.cpu().numpy()
    else:
        stroke = np.array(stroke)
    
    actual_dim = stroke.shape[1] if len(stroke.shape) > 1 and stroke.shape[0] > 0 else point_dim

    # if len(stroke) > num_points:
    #     print("Warning: stroke has more points than num_points, downsampling may lose data.")
    #     print(f"Stroke length: {len(stroke)}, num_points: {num_points}")
    
    if len(stroke) == 0:
        return torch.zeros(num_points, point_dim, dtype=torch.float32)
        
    if len(stroke) == 1:
        new_traj = np.tile(stroke, (num_points, 1))
    else:
        # Calculate cumulative distance based on (x, y) only
        xy_coords = stroke[:, :2]
        dists = np.linalg.norm(np.diff(xy_coords, axis=0), axis=1)
        cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
        total_dist = cum_dists[-1]
        
        if total_dist < 1e-6:
            new_traj = np.tile(stroke[0], (num_points, 1))
        else:
            # Resample based on arc length
            new_dists = np.linspace(0, total_dist, num_points)
            new_traj = np.zeros((num_points, actual_dim), dtype=float)
            for d in range(actual_dim):
                new_traj[:, d] = np.interp(new_dists, cum_dists, stroke[:, d])
    
    # Ensure output has correct dimension
    if new_traj.shape[1] < point_dim:
        # Pad with zeros if needed (e.g., old 2D data)
        pad = np.zeros((num_points, point_dim - new_traj.shape[1]))
        new_traj = np.hstack([new_traj, pad])
    
    return torch.tensor(new_traj, dtype=torch.float32).view(num_points, point_dim)

def normalize_strokes(strokes, num_points=128):
    """
    Converts a list of strokes (list of tensors/arrays shape [N_i,2]) into a fixed-size action tensor.
    新功能：在相邻两笔之间按照笔画内点的平均距离插入过渡点（所有过渡点 pen=1.0 表示抬笔）。
    Args:
        strokes: List of tensors/arrays, each shape [N_i, 2]
        num_points: Number of points to resample the total trajectory to.
    Returns:
        Tensor of shape [num_points * 3]
    """
    # 先把所有笔画统一为 numpy array，计算每笔的平均相邻点距离
    proc_strokes = []
    mean_steps = []
    for stroke in strokes:
        if isinstance(stroke, torch.Tensor):
            stroke = stroke.cpu().numpy()
        else:
            stroke = np.array(stroke)
        proc_strokes.append(stroke)
        if len(stroke) >= 2:
            dists = np.linalg.norm(np.diff(stroke, axis=0), axis=1)
            mean_steps.append(float(np.mean(dists)) if len(dists) > 0 else 0.0)
        else:
            mean_steps.append(0.0)

    # 全局回退平均步长（用于缺失/为0时的备用值）
    positive_means = [m for m in mean_steps if m > 1e-9]
    global_mean = float(np.mean(positive_means)) if len(positive_means) > 0 else 1.0

    trajectory = []

    # 遍历每一笔并构建带 pen 状态的序列，同时在笔间插入过渡点
    for idx, stroke in enumerate(proc_strokes):
        if stroke is None or len(stroke) == 0:
            continue

        # 将该笔的点加入轨迹，笔内点 pen=0，笔的最后一个点 pen=1（表示抬笔点）
        for i, pt in enumerate(stroke):
            pen = 0.0
            if i == len(stroke) - 1:
                pen = 1.0
            trajectory.append([float(pt[0]), float(pt[1]), pen])

        # 如果不是最后一笔且下一笔有点，则在两笔之间插入过渡点（全部为 pen=1.0）
        # 过渡点数量根据两笔内点的平均距离决定：ceil(distance / avg_step)
        if idx < len(proc_strokes) - 1:
            next_stroke = proc_strokes[idx + 1]
            if next_stroke is None or len(next_stroke) == 0:
                continue

            last_pt = stroke[-1].astype(float)
            next_first = next_stroke[0].astype(float)
            d = np.linalg.norm(next_first - last_pt)

            # 如果两点几乎重合，则不插（避免生成重复点）
            if d < 1e-8:
                continue

            mean_prev = mean_steps[idx] if mean_steps[idx] > 1e-9 else None
            mean_next = mean_steps[idx + 1] if mean_steps[idx + 1] > 1e-9 else None

            # 优先使用两笔的均值；若都不可用则用 global_mean
            candidates = [v for v in (mean_prev, mean_next) if v is not None]
            if len(candidates) > 0:
                avg_step = float(np.mean(candidates))
            else:
                avg_step = global_mean

            # 避免 avg_step 为 0
            if avg_step <= 1e-9:
                avg_step = global_mean if global_mean > 1e-9 else 1.0

            num_insert = int(np.ceil(d / avg_step))
            # 至少插入1个过渡点（如果距离不为0）
            num_insert = max(1, num_insert)

            # 插值（均匀），注意不包括端点（端点已在轨迹里）
            for k in range(1, num_insert + 1):
                t = k / (num_insert + 1.0)
                inter = (1.0 - t) * last_pt + t * next_first
                trajectory.append([float(inter[0]), float(inter[1]), 1.0])  # 抬笔状态

    trajectory = np.array(trajectory, dtype=float)

    if len(trajectory) == 0:
        return torch.zeros(num_points * 3, dtype=torch.float32)

    # 2. Resample to fixed length (same逻辑)
    current_len = len(trajectory)
    if current_len == 1:
        new_traj = np.tile(trajectory, (num_points, 1))
    else:
        old_indices = np.linspace(0, 1, current_len)
        new_indices = np.linspace(0, 1, num_points)

        new_traj = np.zeros((num_points, 3), dtype=float)
        new_traj[:, 0] = np.interp(new_indices, old_indices, trajectory[:, 0])
        new_traj[:, 1] = np.interp(new_indices, old_indices, trajectory[:, 1])
        new_traj[:, 2] = np.interp(new_indices, old_indices, trajectory[:, 2])
        
        # Threshold pen state
        new_traj[:, 2] = (new_traj[:, 2] > 0.5).astype(float)

    return torch.tensor(new_traj.flatten(), dtype=torch.float32), current_len


def recover_strokes(action, num_points=128):
    """
    Recovers strokes from the action tensor for visualization.
    修改点：遇到 pen > 0.5 的点视为抬笔（lift），并且**不**把该点加入笔画数组中，
    这样插入的过渡点（pen-up）不会出现在恢复的笔画坐标里。
    """
    traj = action.reshape(num_points, 3)

    strokes = []
    current_stroke = []

    for pt in traj:
        x, y, pen = float(pt[0]), float(pt[1]), float(pt[2])
        # 如果 pen 表示抬笔，则结束当前笔（但不把这个 pen-up 点加入笔的数据）
        if pen > 0.5:
            if len(current_stroke) > 0:
                strokes.append(np.array(current_stroke))
                current_stroke = []
            # 否则只是一个抬笔点，跳过
        else:
            current_stroke.append([x, y])

    # 最后一笔（如果没有在末尾遇到抬笔）
    if len(current_stroke) > 0:
        strokes.append(np.array(current_stroke))

    return strokes


def resample_stroke_by_dist(stroke, step_size=0.005, point_dim=3):
    """
    Resamples a single stroke such that consecutive points are approximately step_size apart.
    Args:
        stroke: np.array or torch.Tensor of shape [N, D] where D is 2 (x, y) or 3 (x, y, r)
        step_size: float, the desired arc length distance between points
        point_dim: int, dimension of each point (2 or 3)
    Returns:
        torch.Tensor of shape [M, point_dim] where M depends on the stroke length and step_size
    """
    if isinstance(stroke, torch.Tensor):
        stroke = stroke.cpu().numpy()
    else:
        stroke = np.array(stroke)
    
    actual_dim = stroke.shape[1] if len(stroke.shape) > 1 and stroke.shape[0] > 0 else point_dim

    if len(stroke) == 0:
        return torch.zeros(0, point_dim, dtype=torch.float32)
        
    if len(stroke) == 1:
        new_traj = stroke.reshape(1, actual_dim)
        if new_traj.shape[1] < point_dim:
            pad = np.zeros((1, point_dim - new_traj.shape[1]))
            new_traj = np.hstack([new_traj, pad])
        return torch.tensor(new_traj, dtype=torch.float32)
    
    # Calculate cumulative distance based on (x, y) only
    xy_coords = stroke[:, :2]
    dists = np.linalg.norm(np.diff(xy_coords, axis=0), axis=1)
    cum_dists = np.insert(np.cumsum(dists), 0, 0.0)
    total_dist = cum_dists[-1]
    
    if total_dist < 1e-6:
        new_traj = stroke[0].reshape(1, actual_dim)
    else:
        # Determine number of points to result in segment length approx step_size
        num_points = max(2, int(np.round(total_dist / step_size)) + 1)
        
        new_dists = np.linspace(0, total_dist, num_points)
        new_traj = np.zeros((num_points, actual_dim), dtype=float)
        for d in range(actual_dim):
            new_traj[:, d] = np.interp(new_dists, cum_dists, stroke[:, d])
            
    # Ensure output has correct dimension
    if new_traj.shape[1] < point_dim:
        pad = np.zeros((new_traj.shape[0], point_dim - new_traj.shape[1]))
        new_traj = np.hstack([new_traj, pad])
    
    return torch.tensor(new_traj, dtype=torch.float32)

