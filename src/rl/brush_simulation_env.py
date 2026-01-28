# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
"""
毛笔仿真环境（无机械臂版本）

功能：
1. 读取笔画数据
2. 计算坐标缩放到工作空间
3. 插值处理轨迹
4. 根据坐标和笔杆倾斜角度计算每一步的笔杆旋转
5. 根据每一步的坐标和旋转进行物理仿真
6. 提取画布并渲染

不依赖机械臂IK求解，直接使用轨迹坐标驱动毛笔仿真
"""

import argparse
import math
import os
import io
import pickle
import sys
import time
import random
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import warp as wp

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入毛笔物理仿真模块
from simulation.model.brush import Brush
from simulation.xpbd_warp_diff import XPBDSimulator
from simulation.config import (
    BRUSH_UP_POSITION,
    WRITING_WORLD_SIZE,
    BRUSH_RADIUS,
    BRUSH_MAX_LENGTH,
    BRUSH_LENGTH_RATIO,
    TILT_ANGLE,
    MAGIC_NUMBER_B,
    MAGIC_NUMBER_K,
    MAX_GRAVITY,
)

# 导入轨迹工具
from data.traj_utils import resample_stroke, resample_stroke_by_dist

# 尝试导入 matplotlib 用于绘图
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def setup_best_gpu():
    """自动选择显存最多的GPU"""
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return

        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): 
                continue
            parts = line.split(',')
            if len(parts) >= 2:
                idx = parts[0].strip()
                free_mem = parts[1].strip()
                gpu_memory.append((int(idx), int(free_mem)))

        if gpu_memory:
            best_gpu = max(gpu_memory, key=lambda x: x[1])
            gpu_idx = best_gpu[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            print(f"Auto-selected GPU {gpu_idx} with {best_gpu[1]} MB free memory.")

    except Exception as e:
        print(f"Failed to auto-select GPU: {e}")


# ==============================================================================
# 数据加载函数
# ==============================================================================

def load_font_data(data_root: str, char_key: str = None) -> Tuple[List, any, str]:
    """
    加载字体数据
    
    Args:
        data_root: 数据根目录
        char_key: 要加载的字符 key，如果为 None 则随机选择
        
    Returns:
        (strokes, image, key): 笔画列表、图像和字符 key
    """
    from PIL import Image

    strokes_path = os.path.join(data_root, "strokes.pkl")
    imgs_path = os.path.join(data_root, "images.pkl")

    if not os.path.exists(strokes_path):
        print(f"Warning: Strokes file not found: {strokes_path}")
        return [], None, None

    if not os.path.exists(imgs_path):
        print(f"Warning: Images file not found: {imgs_path}")
        return [], None, None

    # 加载笔画数据
    with open(strokes_path, 'rb') as f:
        strokes_data = pickle.load(f)

    # 加载图像数据
    with open(imgs_path, 'rb') as f:
        images_data = pickle.load(f)

    # 获取共同的 key
    common_keys = sorted(list(set(strokes_data.keys()) & set(images_data.keys())))
    
    if char_key is None:
        char_key = random.choice(common_keys)
    elif char_key not in common_keys:
        print(f"Warning: char_key '{char_key}' not found, using random key")
        char_key = random.choice(common_keys)
    
    # 解码图像
    img_bytes = images_data[char_key]
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # 处理笔画数据：r 从 [-1, 1] 反归一化
    stroke_result = []
    for stroke in strokes_data[char_key]:
        stroke_temp = stroke.copy() if hasattr(stroke, 'copy') else list(stroke)
        for i in range(len(stroke_temp)):
            stroke_temp[i][2] = (stroke_temp[i][2] + 1) / 2 * 20 / 256 * WRITING_WORLD_SIZE * 4
        stroke_result.append(stroke_temp)
    
    return stroke_result, image, char_key


# ==============================================================================
# 坐标变换和几何计算函数
# ==============================================================================

def compute_brush_depth_from_radius(target_radius: float,
                                    brush_radius: float = BRUSH_RADIUS,
                                    brush_max_length: float = BRUSH_MAX_LENGTH,
                                    brush_length_ratio: float = BRUSH_LENGTH_RATIO) -> float:
    """
    根据目标切面半径计算笔刷下压深度
    """
    # 限制目标半径不超过毛笔最大半径
    tan_alpha = brush_radius / (brush_max_length * brush_length_ratio)
    alpha = math.atan(tan_alpha)
    max_r = brush_radius * math.sqrt(
        math.cos(alpha + math.radians(TILT_ANGLE)) / math.cos(alpha - math.radians(TILT_ANGLE)))
    target_radius = min(target_radius, max_r)
    target_radius = max(target_radius, 0)

    H = brush_max_length * brush_length_ratio
    theta_rad = math.radians(TILT_ANGLE)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    term_inside_sqrt = (H ** 2 / BRUSH_RADIUS ** 2) * (cos_theta ** 2) - (sin_theta ** 2)
    term1 = H * cos_theta
    term2 = target_radius * math.sqrt(term_inside_sqrt)
    d = term1 - term2 * MAGIC_NUMBER_K + brush_max_length - H

    return d


def compute_radius_from_brush_depth(d: float,
                                    brush_radius: float = BRUSH_RADIUS,
                                    brush_max_length: float = BRUSH_MAX_LENGTH,
                                    brush_length_ratio: float = BRUSH_LENGTH_RATIO) -> float:
    """
    根据笔刷深度反推接触半径
    """
    H = brush_max_length * brush_length_ratio
    theta_rad = math.radians(TILT_ANGLE)
    cos_theta = math.cos(theta_rad)
    sin_theta = math.sin(theta_rad)

    term_inside_sqrt = (H ** 2 / brush_radius ** 2) * (cos_theta ** 2) - (sin_theta ** 2)
    
    if term_inside_sqrt <= 0:
        return 0.0
        
    sqrt_val = math.sqrt(term_inside_sqrt)
    term1 = H * cos_theta
    
    numerator = term1 + brush_max_length - H - d
    denominator = sqrt_val * MAGIC_NUMBER_K
    
    if denominator == 0:
        return 0.0
        
    calculated_r = numerator / denominator

    # 限制范围
    tan_alpha = brush_radius / (brush_max_length * brush_length_ratio)
    alpha = math.atan(tan_alpha)
    
    cos_denominator = math.cos(alpha - math.radians(TILT_ANGLE))
    if cos_denominator == 0:
        max_r = brush_radius
    else:
        max_r = brush_radius * math.sqrt(
            math.cos(alpha + math.radians(TILT_ANGLE)) / cos_denominator)

    final_r = min(calculated_r, max_r)
    final_r = max(final_r, 0.0)

    return final_r


def normalize_strokes_to_workspace(strokes: list,
                                   center: np.ndarray = np.array([0.45, 0.0, 0.25]),
                                   size: float = 0.15,
                                   offset: np.ndarray = None) -> list:
    """
    将笔画坐标归一化并映射到工作空间
    
    Args:
        strokes: 原始笔画列表 [[x, y, r], ...]
        center: 书写区域中心点
        size: 书写区域大小（正方形边长的一半）
        offset: 额外的位移偏移量 [dx, dy, dz]
        
    Returns:
        归一化后的笔画列表，每个点是 [x, y, z]
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])
    if not strokes:
        return []

    # 找到所有笔画点的边界
    all_points = []
    for stroke in strokes:
        for pt in stroke:
            all_points.append([pt[0], pt[1]])
    all_points = np.array(all_points)

    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy
    max_range = max(range_xy[0], range_xy[1])

    if max_range < 1e-6:
        max_range = 1.0

    center_xy = (min_xy + max_xy) / 2.0
    base_z = center[2] + offset[2]

    normalized_strokes = []
    for stroke in strokes:
        stroke_np = np.array(stroke, dtype=np.float32)
        workspace_points = []
        
        for pt in stroke_np:
            # 归一化 x, y 坐标
            normalized_x = (pt[0] - center_xy[0]) / (max_range / 2.0)
            normalized_y = (pt[1] - center_xy[1]) / (max_range / 2.0)

            robot_x = center[0] + normalized_x * size + offset[0]
            robot_y = center[1] - normalized_y * size + offset[1]  # y 翻转

            # 根据目标半径计算 z 高度
            if len(pt) >= 3:
                target_radius = float(pt[2])
                depth = compute_brush_depth_from_radius(target_radius)
                robot_z = base_z + depth
            else:
                robot_z = base_z + BRUSH_MAX_LENGTH

            workspace_points.append([robot_x, robot_y, robot_z])

        normalized_strokes.append(np.array(workspace_points, dtype=np.float32))

    return normalized_strokes


# ==============================================================================
# 笔杆旋转计算函数
# ==============================================================================

def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """四元数乘法 (x, y, z, w) 格式"""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])


def quat_slerp(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """四元数球面线性插值"""
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot

    if dot > 0.9995:
        result = q1 + t * (q2 - q1)
        return result / np.linalg.norm(result)

    theta_0 = math.acos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = math.sin(theta)
    sin_theta_0 = math.sin(theta_0)

    s1 = math.cos(theta) - dot * sin_theta / sin_theta_0
    s2 = sin_theta / sin_theta_0

    return s1 * q1 + s2 * q2


def compute_tilt_rotation(movement_dir: np.ndarray, 
                          tilt_angle: float = math.radians(TILT_ANGLE),
                          base_rotation: np.ndarray = None) -> np.ndarray:
    """
    基于运动方向计算倾斜后的笔杆旋转四元数
    
    Args:
        movement_dir: 运动方向向量 (x, y, z)
        tilt_angle: 倾斜角度（弧度）
        base_rotation: 基准旋转四元数 (x, y, z, w)，默认为单位四元数
        
    Returns:
        旋转四元数 (x, y, z, w)
    """
    if base_rotation is None:
        base_rotation = np.array([0.0, 0.0, 0.0, 1.0])
    
    dir_xy = np.array([movement_dir[0], movement_dir[1], 0.0])
    dir_norm = np.linalg.norm(dir_xy)
    if dir_norm < 1e-6:
        dir_xy = np.array([1.0, 0.0, 0.0])
    else:
        dir_xy = dir_xy / dir_norm

    # 倾斜轴垂直于运动方向
    tilt_axis = np.array([-dir_xy[1], dir_xy[0], 0.0])

    half_angle = tilt_angle / 2.0
    sin_half = math.sin(half_angle)
    cos_half = math.cos(half_angle)

    tilt_quat = np.array([
        tilt_axis[0] * sin_half,
        tilt_axis[1] * sin_half,
        tilt_axis[2] * sin_half,
        cos_half
    ])

    result = quat_multiply(tilt_quat, base_rotation)
    return result


# ==============================================================================
# 毛笔构建函数
# ==============================================================================

def build_brush(root_position: torch.Tensor = None, 
                tangent_vector: torch.Tensor = None) -> Brush:
    """
    创建毛笔模型
    
    Args:
        root_position: 毛笔根部位置
        tangent_vector: 毛笔指向方向
        
    Returns:
        Brush 对象
    """
    if root_position is None:
        root_position = torch.tensor([0.0, 0.0, BRUSH_UP_POSITION], dtype=torch.float64)

    if tangent_vector is None:
        tangent_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)

    brush = Brush(
        radius=BRUSH_RADIUS,
        max_length=BRUSH_MAX_LENGTH,
        max_hairs=500,
        max_particles_per_hair=20,
        thickness=0.005,
        root_position=root_position,
        tangent_vector=tangent_vector,
        fixed_length_ratio=1 / 10,
        free_length_ratio=1 - BRUSH_LENGTH_RATIO
    )
    return brush


# ==============================================================================
# 毛笔仿真环境类
# ==============================================================================

class BrushSimulationEnv:
    """
    毛笔仿真环境（无机械臂版本）
    
    直接使用轨迹坐标和计算的旋转驱动毛笔物理仿真
    """
    
    def __init__(self, 
                 offset: np.ndarray = None,
                 xpbd_iterations: int = 200,
                 resample_dist: float = None,
                 fps: int = 60,
                 canvas_resolution: int = 256,
                 output_dir: str = "rl_output"):
        """
        初始化仿真环境
        
        Args:
            offset: 坐标偏移量
            xpbd_iterations: XPBD 求解器迭代次数
            resample_dist: 弧长重采样距离，None 表示不重采样
            fps: 仿真帧率
            canvas_resolution: 画布分辨率
            output_dir: 输出目录
        """
        self.fps = fps
        self.frame_dt = 1.0 / fps
        self.xpbd_iterations = xpbd_iterations
        self.resample_dist = resample_dist
        self.canvas_resolution = canvas_resolution
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 坐标偏移
        self.coord_offset = np.array(offset) if offset is not None else np.array([0.0, 0.0, 0.0])
        
        # 毛笔参数
        self.tilt_angle = math.radians(TILT_ANGLE)
        self.lift_height = 0.08  # 抬笔高度
        
        # 书写区域参数
        self.writing_center = np.array([0.45, 0.0, 0.25])
        self.writing_size = WRITING_WORLD_SIZE
        
        # 基准旋转（竖直向下）
        self.base_rotation = np.array([0.0, 0.0, 0.0, 1.0])
        
        # 仿真状态
        self.brush = None
        self.brush_simulator = None
        self.brush_initialized = False
        
        # 数据
        self.strokes = []
        self.raw_strokes = []
        self.reference_image = None
        self.char_key = None
        
        # 画布边界
        self.canvas_bounds = None
        self.paper_z_threshold = 0.25
        
        # 平滑参数
        self._smoothed_direction = np.array([1.0, 0.0, 0.0])
        self._current_target_quat = None
        self.quat_smooth_alpha = 0.08
        self.max_dir_change = math.radians(8.0)
        
        print(f"[BrushSimulationEnv] Initialized with offset={self.coord_offset}")
    
    def load_data(self, data_root: str, char_key: str = None):
        """
        加载笔画数据
        
        Args:
            data_root: 数据根目录
            char_key: 字符 key
        """
        raw_strokes, self.reference_image, self.char_key = load_font_data(data_root, char_key)
        
        if not raw_strokes:
            print("[Error] No stroke data loaded!")
            return False
        
        print(f"[Data] Loaded character '{self.char_key}' with {len(raw_strokes)} strokes")
        
        # 弧长重采样
        if self.resample_dist is not None:
            print(f"[Resample] Resampling strokes with step size {self.resample_dist}")
            resampled_strokes = []
            for i, stroke in enumerate(raw_strokes):
                stroke_np = np.array(stroke)
                original_len = len(stroke_np)
                resampled = resample_stroke_by_dist(stroke_np, step_size=self.resample_dist, point_dim=3)
                resampled_strokes.append(resampled.numpy().tolist())
                print(f"  Stroke {i + 1}: {original_len} -> {len(resampled)} points")
            raw_strokes = resampled_strokes
        
        self.raw_strokes = raw_strokes
        
        # 归一化到工作空间
        self.strokes = normalize_strokes_to_workspace(
            raw_strokes,
            center=self.writing_center,
            size=self.writing_size,
            offset=self.coord_offset
        )
        
        # 计算画布边界
        c_off = self.coord_offset
        min_x = self.writing_center[0] - (self.writing_size + BRUSH_MAX_LENGTH / 2) * 1.1 + c_off[0]
        max_x = self.writing_center[0] + (self.writing_size + BRUSH_MAX_LENGTH / 2) * 1.1 + c_off[0]
        min_y = self.writing_center[1] - (self.writing_size + BRUSH_MAX_LENGTH / 2) * 1.1 + c_off[1]
        max_y = self.writing_center[1] + (self.writing_size + BRUSH_MAX_LENGTH / 2) * 1.1 + c_off[1]
        self.canvas_bounds = (min_x, max_x, min_y, max_y)
        
        self.paper_z_threshold = 0.25 + self.coord_offset[2]
        
        print(f"[Data] Canvas bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")
        
        return True
    
    def initialize_brush_simulation(self, initial_position: np.ndarray = None):
        """
        初始化毛笔物理仿真器
        
        Args:
            initial_position: 初始毛笔根部位置
        """
        if initial_position is None:
            if len(self.strokes) > 0:
                first_point = self.strokes[0][0]
                initial_position = np.array([first_point[0], first_point[1], 
                                             first_point[2] + self.lift_height])
            else:
                initial_position = np.array([0.4, 0.0, 0.5])
        
        print(f"[Brush] Initializing at position: {initial_position}")
        
        # 创建毛笔
        self.brush = build_brush(
            root_position=torch.tensor(initial_position, dtype=torch.float64)
        )
        self.brush.gen_hairs()
        self.brush.gen_constraints()
        
        print(f"[Brush] Created with {len(self.brush.particles)} particles, "
              f"{len(self.brush.hairs)} hairs, {len(self.brush.constraints)} constraints")
        
        # 估算仿真步数
        total_points = sum(len(s) for s in self.strokes)
        # 每个点约需要多帧，加上抬笔过渡
        num_sim_steps = max(6000, total_points * 5 + len(self.strokes) * 100)
        
        min_x, max_x, min_y, max_y = self.canvas_bounds
        
        # 创建 XPBD 仿真器
        self.brush_simulator = XPBDSimulator(
            dt=self.frame_dt,
            substeps=10,
            iterations=self.xpbd_iterations,
            num_steps=num_sim_steps,
            batch_size=1,
            gravity=torch.tensor([0.0, 0.0, -MAX_GRAVITY], dtype=torch.float64),
            dis_compliance=1e-10,
            variable_dis_compliance=1e-8,
            bending_compliance_range=(1e-5, 1e-5),
            alignment_compliance_range=(1e-5, 1e-5),
            damping=0.3,
            canvas_resolution=self.canvas_resolution,
            canvas_min_x=min_x,
            canvas_max_x=max_x,
            canvas_min_y=min_y,
            canvas_max_y=max_y,
        )
        
        # 加载毛笔到仿真器
        self.brush_simulator.load_brush(self.brush)
        
        self.brush_initialized = True
        print("[Brush] Simulation initialized successfully!")
    
    def compute_rotation_from_direction(self, 
                                         current_pos: np.ndarray,
                                         prev_pos: np.ndarray = None) -> np.ndarray:
        """
        根据运动方向计算笔杆旋转
        
        Args:
            current_pos: 当前位置
            prev_pos: 上一个位置
            
        Returns:
            旋转四元数 (x, y, z, w)
        """
        if prev_pos is not None:
            movement = current_pos - prev_pos
            movement[2] = 0.0  # 只考虑 XY 平面的运动方向
            move_norm = np.linalg.norm(movement)
            
            if move_norm > 1e-6:
                desired_dir = movement / move_norm
                
                # 限制方向变化
                current_dir = self._smoothed_direction
                cur_norm = np.linalg.norm(current_dir)
                if cur_norm < 1e-6:
                    current_dir = desired_dir
                else:
                    current_dir = current_dir / cur_norm
                
                dot_val = float(np.clip(np.dot(current_dir, desired_dir), -1.0, 1.0))
                angle = math.acos(dot_val)
                
                if angle > self.max_dir_change:
                    t = self.max_dir_change / angle
                    new_dir = current_dir * (1 - t) + desired_dir * t
                else:
                    new_dir = desired_dir
                
                new_norm = np.linalg.norm(new_dir)
                if new_norm > 1e-6:
                    self._smoothed_direction = new_dir / new_norm
        
        # 计算倾斜旋转
        raw_target_rotation = compute_tilt_rotation(
            self._smoothed_direction, 
            self.tilt_angle,
            self.base_rotation
        )
        
        # 平滑旋转
        if self._current_target_quat is None:
            self._current_target_quat = raw_target_rotation
        else:
            self._current_target_quat = quat_slerp(
                self._current_target_quat, raw_target_rotation, self.quat_smooth_alpha
            )
        
        return self._current_target_quat.copy()
    
    def step_simulation(self, position: np.ndarray, rotation: np.ndarray):
        """
        执行一步物理仿真
        
        Args:
            position: 毛笔根部位置 (x, y, z)
            rotation: 笔杆旋转四元数 (x, y, z, w)
        """
        if not self.brush_initialized:
            print("[Warning] Brush simulation not initialized!")
            return
        
        current_pos = torch.tensor(position, dtype=torch.float64)
        
        # 转换旋转格式：从 (x, y, z, w) 到仿真器需要的格式
        # 并翻转方向（笔杆指向和笔头指向相反）
        x, y, z, w = rotation[0], rotation[1], rotation[2], rotation[3]
        flipped_w = -x
        flipped_x = w
        flipped_y = z
        flipped_z = -y
        absolute_rotation = torch.tensor(
            [flipped_w, flipped_x, flipped_y, flipped_z],
            dtype=torch.float64
        )
        
        # 执行仿真步
        self.brush_simulator.step_with_absolute_transform(current_pos, absolute_rotation)
    
    def get_canvas(self) -> np.ndarray:
        """
        获取当前画布图像
        
        Returns:
            画布数组 (canvas_resolution, canvas_resolution)
        """
        if self.brush_simulator is None:
            return np.zeros((self.canvas_resolution, self.canvas_resolution))
        
        try:
            raw_tensor = wp.to_torch(self.brush_simulator.ink_canvas)
            res = self.brush_simulator.canvas_resolution
            image_tensor = raw_tensor[0].reshape((res, res))
            
            # 应用 sigmoid 归一化
            image_tensor = (torch.sigmoid(image_tensor) - 0.5) * 2.0
            canvas_2d = image_tensor.cpu().detach().numpy()
            
            # 反转颜色（墨迹为黑色）
            return 1.0 - canvas_2d
        except Exception as e:
            print(f"[Error] Failed to get canvas: {e}")
            return np.zeros((self.canvas_resolution, self.canvas_resolution))
    
    def reset_brush_state(self, position: np.ndarray, rotation: np.ndarray):
        """
        重置毛笔状态到指定位置和旋转
        
        Args:
            position: 新的根部位置
            rotation: 新的旋转四元数 (x, y, z, w)
        """
        if self.brush is None or self.brush_simulator is None:
            return
        
        current_root_pos = torch.tensor(position, dtype=torch.float64)
        
        # 计算笔头指向方向
        q = torch.tensor([rotation[0], rotation[1], rotation[2], rotation[3]], dtype=torch.float64)
        x, y, z, w = q[0], q[1], q[2], q[3]
        
        current_root_rot = torch.stack([
            -2.0 * (x * z + y * w),
            2.0 * (x * w - y * z),
            2.0 * (x * x + y * y) - 1.0
        ])
        
        if current_root_rot[2] > 0:
            current_root_rot = -current_root_rot
        
        self.brush = build_brush(current_root_pos, current_root_rot)
        self.brush_simulator.load_brush(self.brush, keep_ink_canvas=True)
    
    def run_simulation(self, 
                       points_per_frame: float = 0.5,
                       lift_speed: float = 0.02,
                       prepare_speed: float = 0.01,
                       save_intermediate: bool = False) -> np.ndarray:
        """
        运行完整的书写仿真
        
        Args:
            points_per_frame: 每帧前进的点数
            lift_speed: 抬笔速度
            prepare_speed: 准备阶段速度
            save_intermediate: 是否保存中间结果
            
        Returns:
            最终画布图像
        """
        if not self.brush_initialized:
            print("[Error] Brush simulation not initialized!")
            return np.zeros((self.canvas_resolution, self.canvas_resolution))
        
        print(f"[Simulation] Starting simulation with {len(self.strokes)} strokes...")
        
        # 状态变量
        current_stroke_idx = 0
        current_point_idx = 0.0
        point_accumulator = 0.0
        
        is_lifting = False
        lift_progress = 0.0
        lift_start_pos = None
        lift_end_pos = None
        
        is_preparing = True
        prepare_progress = 0.0
        prepare_start_pos = None
        prepare_end_pos = None
        
        prev_pos = None
        frame_count = 0
        
        # 初始化第一笔的准备阶段
        if len(self.strokes) > 0:
            start_pt = self.strokes[0][0]
            prepare_start_pos = np.array([start_pt[0], start_pt[1], start_pt[2] + self.lift_height])
            prepare_end_pos = np.array([start_pt[0], start_pt[1], start_pt[2]])
            
            # 初始化运动方向
            if len(self.strokes[0]) >= 2:
                p0 = np.array(self.strokes[0][0])
                p1 = np.array(self.strokes[0][1])
                dir_vec = p1 - p0
                dir_vec[2] = 0.0
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-6:
                    self._smoothed_direction = dir_vec / norm
        
        while current_stroke_idx < len(self.strokes):
            frame_count += 1
            
            # 计算当前位置
            if is_preparing:
                # 准备阶段：从起点上方下降到起点
                t = prepare_progress
                t = t * t * (3 - 2 * t)  # smoothstep
                
                current_pos = prepare_start_pos * (1 - t) + prepare_end_pos * t
                
                prepare_progress += prepare_speed
                if prepare_progress >= 1.0:
                    is_preparing = False
                    prepare_progress = 0.0
                    prev_pos = None  # 重置，避免落笔瞬间的突变
                
            elif is_lifting:
                # 抬笔过渡
                t = lift_progress
                
                if t < 0.3:
                    # 抬起阶段
                    lift_t = t / 0.3
                    lift_t = lift_t * lift_t * (3 - 2 * lift_t)
                    current_pos = np.array([
                        lift_start_pos[0],
                        lift_start_pos[1],
                        lift_start_pos[2] + self.lift_height * lift_t
                    ])
                elif t < 0.7:
                    # 平移阶段
                    move_t = (t - 0.3) / 0.4
                    move_t = move_t * move_t * (3 - 2 * move_t)
                    current_pos = np.array([
                        lift_start_pos[0] + (lift_end_pos[0] - lift_start_pos[0]) * move_t,
                        lift_start_pos[1] + (lift_end_pos[1] - lift_start_pos[1]) * move_t,
                        lift_start_pos[2] + self.lift_height
                    ])
                else:
                    # 落下阶段
                    drop_t = (t - 0.7) / 0.3
                    drop_t = drop_t * drop_t * (3 - 2 * drop_t)
                    current_pos = np.array([
                        lift_end_pos[0],
                        lift_end_pos[1],
                        lift_end_pos[2] + self.lift_height * (1 - drop_t)
                    ])
                
                lift_progress += lift_speed
                if lift_progress >= 1.0:
                    is_lifting = False
                    lift_progress = 0.0
                    current_point_idx = 0.0
                    
                    # 进入下一笔的准备阶段
                    if current_stroke_idx < len(self.strokes):
                        start_pt = self.strokes[current_stroke_idx][0]
                        prepare_start_pos = np.array([start_pt[0], start_pt[1], start_pt[2] + self.lift_height])
                        prepare_end_pos = np.array([start_pt[0], start_pt[1], start_pt[2]])
                        
                        # 更新运动方向
                        if len(self.strokes[current_stroke_idx]) >= 2:
                            p0 = np.array(self.strokes[current_stroke_idx][0])
                            p1 = np.array(self.strokes[current_stroke_idx][1])
                            dir_vec = p1 - p0
                            dir_vec[2] = 0.0
                            norm = np.linalg.norm(dir_vec)
                            if norm > 1e-6:
                                self._smoothed_direction = dir_vec / norm
                        
                        is_preparing = True
                        
                        # 重置毛笔状态
                        rotation = self.compute_rotation_from_direction(prepare_start_pos, None)
                        self.reset_brush_state(prepare_start_pos, rotation)
            else:
                # 正常书写
                current_stroke = self.strokes[current_stroke_idx]
                point_idx = min(int(current_point_idx), len(current_stroke) - 1)
                
                # 插值
                if point_idx < len(current_stroke) - 1:
                    t = current_point_idx - int(current_point_idx)
                    p1 = current_stroke[point_idx]
                    p2 = current_stroke[point_idx + 1]
                    current_pos = p1 * (1 - t) + p2 * t
                else:
                    current_pos = current_stroke[point_idx]
                
                # 推进点索引
                point_accumulator += points_per_frame
                if point_accumulator >= 1.0:
                    advance = int(point_accumulator)
                    point_accumulator -= advance
                    current_point_idx += advance
                else:
                    current_point_idx += points_per_frame
                
                # 检查是否完成当前笔画
                if current_point_idx >= len(current_stroke) - 1:
                    current_stroke_idx += 1
                    
                    if save_intermediate:
                        self._save_intermediate_result(f"stroke_{current_stroke_idx}")
                    
                    if current_stroke_idx >= len(self.strokes):
                        print(f"[Simulation] Writing complete! Total frames: {frame_count}")
                        break
                    else:
                        # 开始抬笔
                        is_lifting = True
                        lift_progress = 0.0
                        lift_start_pos = current_stroke[-1].copy()
                        lift_end_pos = self.strokes[current_stroke_idx][0].copy()
                        print(f"[Simulation] Stroke {current_stroke_idx} complete, "
                              f"lifting to stroke {current_stroke_idx + 1}")
            
            # 计算旋转
            rotation = self.compute_rotation_from_direction(current_pos, prev_pos)
            
            # 执行仿真步
            self.step_simulation(current_pos, rotation)
            
            prev_pos = current_pos.copy()
            
            # 进度输出
            if frame_count % 100 == 0:
                print(f"[Simulation] Frame {frame_count}, "
                      f"stroke {current_stroke_idx + 1}/{len(self.strokes)}, "
                      f"pos=({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.4f})")
        
        print(f"[Simulation] Completed in {frame_count} frames")
        
        return self.get_canvas()
    
    def _save_intermediate_result(self, suffix: str):
        """保存中间结果"""
        if not HAS_MATPLOTLIB:
            return
        
        canvas = self.get_canvas()
        output_path = self.output_dir / f"{self.char_key}_{suffix}.png"
        
        plt.figure(figsize=(6, 6))
        plt.imshow(canvas, cmap='gray', vmin=0, vmax=1)
        plt.title(f"{self.char_key} - {suffix}")
        plt.axis('off')
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[Save] Intermediate result saved to: {output_path}")
    
    def save_comparison_image(self, suffix: str = "_final"):
        """
        保存对比图像
        
        Args:
            suffix: 文件名后缀
        """
        if not HAS_MATPLOTLIB:
            print("[Warning] matplotlib not available, skipping save")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 子图1：真值图片
        ax1 = axes[0]
        if self.reference_image is not None:
            ax1.imshow(self.reference_image)
            ax1.set_title(f"Ground Truth - {self.char_key}", fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "No reference image", ha='center', va='center', fontsize=14)
            ax1.set_title("Ground Truth (N/A)", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # 子图2：仿真画布
        ax2 = axes[1]
        canvas = self.get_canvas()
        ax2.imshow(canvas, cmap='gray', vmin=0, vmax=1)
        ax2.set_title("Brush Physics Ink Canvas", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # 子图3：轨迹可视化
        ax3 = axes[2]
        if self.strokes and len(self.strokes) > 0:
            base_z = 0.25 + self.coord_offset[2]
            colors = plt.cm.jet(np.linspace(0, 1, len(self.strokes)))
            
            vec_compute_radius = np.vectorize(compute_radius_from_brush_depth)
            all_points = []
            
            for i, stroke in enumerate(self.strokes):
                stroke_np = np.array(stroke)
                xy = stroke_np[:, :2]
                z_vals = stroke_np[:, 2]
                if len(xy) == 0:
                    continue
                
                all_points.append(xy)
                
                d_inputs = z_vals - base_z
                r_vals = vec_compute_radius(d_inputs)
                r_vals = np.maximum(r_vals, 0)
                
                # 绘制带半径的圆
                circle_patches = []
                for j in range(len(xy)):
                    if r_vals[j] > 1e-5:
                        circle = patches.Circle((xy[j, 0], xy[j, 1]), r_vals[j])
                        circle_patches.append(circle)
                
                if circle_patches:
                    from matplotlib.collections import PatchCollection
                    p = PatchCollection(circle_patches, alpha=0.3, facecolor=colors[i], edgecolor='none')
                    ax3.add_collection(p)
                
                # 轨迹线
                if len(xy) > 1:
                    ax3.plot(xy[:, 0], xy[:, 1], color='gray', alpha=0.5, linewidth=0.5, zorder=2)
                
                # 标记起点
                ax3.text(xy[0, 0], xy[0, 1], str(i + 1), color='red', fontsize=10, fontweight='bold', zorder=3)
            
            ax3.set_aspect('equal')
            ax3.set_title(f"Visualized Radius - {self.char_key}", fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            
            if all_points:
                all_points_np = np.concatenate(all_points)
                x_min, x_max = all_points_np[:, 0].min(), all_points_np[:, 0].max()
                y_min, y_max = all_points_np[:, 1].min(), all_points_np[:, 1].max()
                margin = 0.02
                ax3.set_xlim(x_min - margin, x_max + margin)
                ax3.set_ylim(y_min - margin, y_max + margin)
        else:
            ax3.text(0.5, 0.5, "No stroke data", ha='center', va='center', fontsize=14)
            ax3.set_title("Stroke Visualization (N/A)", fontsize=12, fontweight='bold')
        
        offset_str = f"({self.coord_offset[0]:.2f}, {self.coord_offset[1]:.2f}, {self.coord_offset[2]:.2f})"
        fig.suptitle(f"Simulation Result - {self.char_key} | Offset: {offset_str}",
                     fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{self.char_key}{suffix}.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"[Save] Comparison image saved to: {output_path}")
        plt.close()


# ==============================================================================
# 命令行接口
# ==============================================================================

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Brush simulation without robotic arm"
    )
    parser.add_argument(
        '--data_root',
        type=str,
        default=None,
        help='Path to data directory containing strokes.pkl and images.pkl'
    )
    parser.add_argument(
        '--char_key',
        type=str,
        default=None,
        help='Character key to load (default: random)'
    )
    parser.add_argument(
        '--offset',
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=('X', 'Y', 'Z'),
        help='Coordinate offset (default: 0.0 0.0 0.0)'
    )
    parser.add_argument(
        '--xpbd_iterations',
        type=int,
        default=200,
        help='Number of XPBD solver iterations (default: 200)'
    )
    parser.add_argument(
        '--resample_dist',
        type=float,
        default=None,
        help='Arc length resampling distance (default: None)'
    )
    parser.add_argument(
        '--canvas_resolution',
        type=int,
        default=256,
        help='Canvas resolution (default: 256)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='rl_output',
        help='Output directory (default: rl_output)'
    )
    parser.add_argument(
        '--points_per_frame',
        type=float,
        default=0.5,
        help='Points to advance per frame (default: 0.5)'
    )
    parser.add_argument(
        '--save_intermediate',
        action='store_true',
        help='Save intermediate results after each stroke'
    )
    
    return parser.parse_args()


def find_data_root() -> str:
    """查找数据目录"""
    possible_paths = [
        "/data1/jizy/output/infer_output",
        "/home/jizy/project/video_tokenizer/data/test/debug",
        str(PROJECT_ROOT / "data" / "debug"),
        str(PROJECT_ROOT / "data" / "infer_output"),
        str(PROJECT_ROOT.parent / "data" / "debug"),
        str(PROJECT_ROOT.parent / "data" / "infer_output"),
        "data/debug",
        "data/infer_output",
    ]
    
    for path in possible_paths:
        strokes_file = os.path.join(path, "strokes.pkl")
        imgs_file = os.path.join(path, "images.pkl")
        if os.path.exists(strokes_file) and os.path.exists(imgs_file):
            return path
    
    return None


def main():
    """主函数"""
    setup_best_gpu()
    
    args = parse_args()
    
    # 查找数据目录
    data_root = args.data_root
    if data_root is None:
        data_root = find_data_root()
        if data_root is None:
            print("[Error] Could not find data directory!")
            print("Please specify --data_root")
            return
    
    print(f"[Main] Using data root: {data_root}")
    
    # 创建仿真环境
    env = BrushSimulationEnv(
        offset=args.offset,
        xpbd_iterations=args.xpbd_iterations,
        resample_dist=args.resample_dist,
        canvas_resolution=args.canvas_resolution,
        output_dir=args.output_dir
    )
    
    # 加载数据
    if not env.load_data(data_root, args.char_key):
        print("[Error] Failed to load data!")
        return
    
    # 初始化仿真
    env.initialize_brush_simulation()
    
    # 运行仿真
    time_start = time.time()
    
    canvas = env.run_simulation(
        points_per_frame=args.points_per_frame,
        save_intermediate=args.save_intermediate
    )
    
    time_end = time.time()
    print(f"[Main] Total simulation time: {time_end - time_start:.2f} seconds")
    
    # 保存最终结果
    env.save_comparison_image("_final_comparison")
    
    print("[Main] Done!")


if __name__ == "__main__":
    main()
