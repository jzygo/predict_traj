import torch
import numpy as np

def normalize_strokes(strokes, num_points=128):
    """
    Converts a list of strokes (list of tensors) into a fixed-size action tensor.
    Format: [x, y, pen_state] flattened.
    
    Args:
        strokes: List of tensors/arrays, each shape [N_i, 2]
        num_points: Number of points to resample the total trajectory to.
        
    Returns:
        Tensor of shape [num_points * 3]
    """
    # 1. Flatten to a single sequence with pen state
    # pen_state: 0 for drawing, 1 for lifting (end of stroke)
    trajectory = []
    
    for stroke in strokes:
        if isinstance(stroke, torch.Tensor):
            stroke = stroke.numpy()
        else:
            stroke = np.array(stroke)
            
        if len(stroke) == 0:
            continue
            
        for i, point in enumerate(stroke):
            pen = 0.0
            if i == len(stroke) - 1:
                pen = 1.0
            trajectory.append([point[0], point[1], pen])
            
    trajectory = np.array(trajectory)
    
    if len(trajectory) == 0:
        return torch.zeros(num_points * 3)
        
    # 2. Resample to fixed length
    # We separate x, y, and pen for interpolation
    current_len = len(trajectory)
    
    if current_len == 1:
        # Handle single point case by repeating
        new_traj = np.tile(trajectory, (num_points, 1))
    else:
        old_indices = np.linspace(0, 1, current_len)
        new_indices = np.linspace(0, 1, num_points)
        
        new_traj = np.zeros((num_points, 3))
        # Interpolate x, y
        new_traj[:, 0] = np.interp(new_indices, old_indices, trajectory[:, 0])
        new_traj[:, 1] = np.interp(new_indices, old_indices, trajectory[:, 1])
        # Interpolate pen state (can be treated as probability)
        new_traj[:, 2] = np.interp(new_indices, old_indices, trajectory[:, 2])
        
    # 3. Flatten
    # [num_points, 3] -> [num_points * 3]
    return torch.tensor(new_traj.flatten(), dtype=torch.float32)

def recover_strokes(action, num_points=128):
    """
    Recovers strokes from the action tensor for visualization.
    """
    # Reshape [num_points * 3] -> [num_points, 3]
    traj = action.reshape(num_points, 3)
    
    strokes = []
    current_stroke = []
    
    for point in traj:
        x, y, pen = point
        current_stroke.append([x, y])
        
        # If pen > 0.5, consider it a lift
        if pen > 0.5:
            strokes.append(np.array(current_stroke))
            current_stroke = []
            
    if len(current_stroke) > 0:
        strokes.append(np.array(current_stroke))
        
    return strokes
