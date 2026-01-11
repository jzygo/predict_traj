# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import math
import os
import io
import pickle
import sys

def setup_best_gpu():
    import subprocess
    try:
        # Query GPU memory usage using nvidia-smi
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,nounits,noheader'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            return

        gpu_memory = []
        for line in result.stdout.strip().split('\n'):
            if not line.strip(): continue
            parts = line.split(',')
            if len(parts) >= 2:
                idx = parts[0].strip()
                free_mem = parts[1].strip()
                gpu_memory.append((int(idx), int(free_mem)))

        # Select GPU with the most free memory
        if gpu_memory:
            best_gpu = max(gpu_memory, key=lambda x: x[1])
            gpu_idx = best_gpu[0]
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)
            print(f"Auto-selected GPU {gpu_idx} with {best_gpu[1]} MB free memory.")

    except Exception as e:
        print(f"Failed to auto-select GPU: {e}")

setup_best_gpu()

import numpy as np
import warp as wp
import torch
from pathlib import Path

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.utils import create_cylinder_mesh
from config import MAX_GRAVITY

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 导入毛笔物理仿真模块
from simulation.model.brush import Brush
from simulation.xpbd_warp_diff import XPBDSimulator, identity_quaternion
from simulation.config import BRUSH_UP_POSITION

# 导入轨迹工具（用于弧长重采样）
from data.traj_utils import resample_stroke, resample_stroke_by_dist

# 尝试导入 matplotlib 用于绘图，如果不可用则禁用可视化
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

WRITING_WORLD_SIZE = 0.06


def load_strokes_from_pkl(pkl_path: str, char_key: str = None) -> tuple:
    """
    从 pkl 文件加载笔画数据
    
    Args:
        pkl_path: strokes pkl 文件路径
        char_key: 要加载的字符 key，如 'GB45'。如果为 None，返回第一个有笔画的字符
        
    Returns:
        (strokes, key): 笔画列表和字符 key
        strokes 是一个列表，每个元素是一个笔画的点序列 [[x1,y1], [x2,y2], ...]
    """
    with open(pkl_path, 'rb') as f:
        strokes_data = pickle.load(f)
    
    if char_key is not None and char_key in strokes_data:
        return strokes_data[char_key], char_key
    
    # 返回第一个有笔画的字符
    for key in sorted(strokes_data.keys()):
        if len(strokes_data[key]) > 0:
            return strokes_data[key], key
    
    return [], None


def load_font_data(data_root: str) -> tuple:
    """
    参考 font_dataset.py 的方式加载字体数据
    
    Args:
        data_root: 数据根目录，包含 {font_name}_strokes.pkl 和 {font_name}_imgs.pkl
        font_name: 字体名称
        char_key: 要加载的字符 key。如果为 None，返回第一个有笔画的字符
        
    Returns:
        (strokes, image, key): 笔画列表、图像(PIL Image)和字符 key
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
    import random
    char_key = random.choice(common_keys)
    char_key = "下载 (1)"
    # char_key = "debug"
    # 解码图像
    img_bytes = images_data[char_key]
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    # r 从 [-1, 1] 反归一化
    stroke_result = []
    max_r = -1.0
    for stroke in strokes_data[char_key]:
        stroke_temp = stroke
        for i in range(len(stroke_temp)):
            stroke_temp[i][2] = (stroke_temp[i][2] + 1) / 2 * 20 / 256 * WRITING_WORLD_SIZE * 4
            if stroke_temp[i][2] > max_r:
                max_r = stroke_temp[i][2]
        stroke_result.append(stroke_temp)
    # for stroke in stroke_result:
    #     for i in range(len(stroke)):
    #         stroke[i][2] = stroke[i][2] / max_r * BRUSH_RADIUS
    # return stroke_result, image, char_key
    return [stroke_result[1], stroke_result[8]], image, char_key


def build_brush(root_position: torch.Tensor = None, stick_length: float = 0.3) -> Brush:
    """
    参考 robot_brush_pipeline.py 创建毛笔模型
    
    Args:
        root_position: 毛笔根部位置 (与木棍端点对齐)
        stick_length: 木棍长度
        
    Returns:
        Brush 对象
    """
    if root_position is None:
        root_position = torch.tensor([0.0, 0.0, BRUSH_UP_POSITION], dtype=torch.float64)
    
    brush = Brush(
        radius=BRUSH_RADIUS,
        max_length=BRUSH_MAX_LENGTH,
        max_hairs=500,
        max_particles_per_hair=30,
        thickness=0.005,  # 毛发粗细
        root_position=root_position,
        tangent_vector=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),  # 向下
        length_ratio=BRUSH_LENGTH_RATIO,
    )
    return brush


# 毛笔几何参数（与 build_brush 保持一致）
BRUSH_RADIUS = 0.0055  # 毛笔最大半径
BRUSH_MAX_LENGTH = 0.047  # 毛笔最大长度
BRUSH_LENGTH_RATIO = 9 / 10  # 锥形部分占比
TILT_ANGLE = 0.0
MAGIC_NUMBER_B = 0.0
MAGIC_NUMBER_K = 1.0


def compute_brush_depth_from_radius(target_radius: float,
                                    brush_radius: float = BRUSH_RADIUS,
                                    brush_max_length: float = BRUSH_MAX_LENGTH,
                                    brush_length_ratio: float = BRUSH_LENGTH_RATIO) -> float:
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

    d = term1 - term2 * MAGIC_NUMBER_K + (brush_max_length - H) + brush_max_length / (WRITING_WORLD_SIZE * 2) * MAGIC_NUMBER_B # 最后一个数字是经验参数

    # 计算下压深度：d = r * max_length * length_ratio / radius
    # depth = target_radius * brush_max_length * brush_length_ratio / brush_radius

    return d


def normalize_strokes_to_workspace(strokes: list, 
                                    center: np.ndarray = np.array([0.45, 0.0, 0.25]),
                                    size: float = 0.15,
                                    offset: np.ndarray = None,
                                    brush_radius: float = BRUSH_RADIUS,
                                    brush_max_length: float = BRUSH_MAX_LENGTH,
                                    brush_length_ratio: float = BRUSH_LENGTH_RATIO) -> list:
    """
    将笔画坐标归一化并映射到机械臂工作空间
    
    Args:
        strokes: 原始笔画列表，每个笔画是 [[x, y, r], ...] 的坐标
                 其中 x, y 是像素坐标，r 是目标切面半径（毛笔下压后与纸面接触的圆形半径）
        center: 书写区域中心点 (机械臂坐标系)
        size: 书写区域大小（正方形边长的一半）
        offset: 额外的位移偏移量 [dx, dy, dz]
        brush_radius: 毛笔最大半径（用于计算下压深度）
        brush_max_length: 毛笔最大长度（用于计算下压深度）
        brush_length_ratio: 毛笔锥形部分占比（用于计算下压深度）
        
    Returns:
        归一化后的笔画列表，每个点是 [x, y, z] 的机械臂坐标
        其中 z 是根据目标切面半径计算出的笔尖高度（= 书写平面高度 - 下压深度）
    """
    if offset is None:
        offset = np.array([0.0, 0.0, 0.0])
    if not strokes:
        return []
    
    # 找到所有笔画点的边界（只使用 x, y 坐标）
    all_points = []
    for stroke in strokes:
        for pt in stroke:
            # 只取前两个坐标 (x, y) 用于归一化计算
            all_points.append([pt[0], pt[1]])
    all_points = np.array(all_points)
    
    min_xy = all_points.min(axis=0)
    max_xy = all_points.max(axis=0)
    range_xy = max_xy - min_xy
    max_range = max(range_xy[0], range_xy[1])
    
    if max_range < 1e-6:
        max_range = 1.0
    
    # 归一化到 [-1, 1] 范围
    center_xy = (min_xy + max_xy) / 2.0
    
    # 书写平面基准高度
    base_z = center[2] + offset[2]
    
    normalized_strokes = []
    for stroke in strokes:
        stroke_np = np.array(stroke, dtype=np.float32)
        
        # 映射到机械臂工作空间
        # 原始坐标: x 向右, y 向下 (图像坐标系)
        # 机械臂坐标: x 向前, y 向左, z 向上
        # 映射: 图像 x -> 机械臂 x, 图像 y -> 机械臂 -y (翻转)
        workspace_points = []
        for pt in stroke_np:
            # 归一化 x, y 坐标
            normalized_x = (pt[0] - center_xy[0]) / (max_range / 2.0)
            normalized_y = (pt[1] - center_xy[1]) / (max_range / 2.0)
            
            # 图像 x 映射到机械臂 x
            # 图像 y 映射到机械臂 -y (左右镜像，使字不反)
            robot_x = center[0] + normalized_x * size + offset[0]
            robot_y = center[1] - normalized_y * size + offset[1]  # 注意这里翻转
            
            # 从第三维提取目标切面半径，计算下压深度和目标高度
            # 毛笔根部在上方，笔尖在下方，控制点位于毛笔根部
            # 当笔尖刚接触纸面时：z = base_z + brush_max_length
            # 当下压 depth 时：z = base_z + brush_max_length - depth
            # 这样 z 始终高于纸面(base_z)，且 z 越低表示下压越深
            if len(pt) >= 3:
                target_radius = float(pt[2])
                # 根据切面半径计算下压深度
                depth = compute_brush_depth_from_radius(
                    target_radius,
                    brush_radius=brush_radius,
                    brush_max_length=brush_max_length,
                    brush_length_ratio=brush_length_ratio
                )
                robot_z = base_z + depth
            else:
                # 如果没有第三维数据，使用笔尖刚接触纸面的高度
                robot_z = base_z + brush_max_length
            
            workspace_points.append([robot_x, robot_y, robot_z])
        
        normalized_strokes.append(np.array(workspace_points, dtype=np.float32))
    
    return normalized_strokes


def visualize_strokes_to_file(raw_strokes: list, 
                               normalized_strokes: list, 
                               char_key: str,
                               output_dir: str = "stroke_visualizations") -> None:
    """
    将原始笔画和归一化后的笔画分别可视化并保存为 PNG 图片
    
    Args:
        raw_strokes: 原始笔画列表（像素坐标）
        normalized_strokes: 归一化后的笔画列表（机械臂坐标）
        char_key: 字符 key（用于文件名）
        output_dir: 输出目录（支持相对路径和绝对路径）
    """
    if not HAS_MATPLOTLIB:
        print(f"Warning: matplotlib not available, skipping stroke visualization for {char_key}")
        return
    
    # 创建输出目录（支持相对和绝对路径）
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取绝对路径以便调试
    abs_output_path = output_path.resolve()
    print(f"[Visualization] Output directory: {abs_output_path}")
    
    try:
        # ========== 可视化 1：原始笔画（像素坐标） ==========
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # 获取原始笔画的边界
        all_raw_points = []
        for stroke in raw_strokes:
            all_raw_points.extend(stroke)
        all_raw_points = np.array(all_raw_points)
        
        min_xy = all_raw_points.min(axis=0)
        max_xy = all_raw_points.max(axis=0)
        
        # 绘制原始笔画
        colors = plt.cm.jet(np.linspace(0, 1, len(raw_strokes)))
        for i, stroke in enumerate(raw_strokes):
            stroke_np = np.array(stroke)
            if len(stroke_np) > 1:
                ax.plot(stroke_np[:, 0], stroke_np[:, 1], 
                       color=colors[i], linewidth=2.5, label=f"Stroke {i+1}", zorder=2)
            # 标记起点
            ax.scatter(stroke_np[0, 0], stroke_np[0, 1], 
                      color=colors[i], s=80, marker='o', edgecolors='white', 
                      linewidths=2, zorder=3)
        
        # 添加边界框
        rect = patches.Rectangle((min_xy[0], min_xy[1]), 
                                 max_xy[0]-min_xy[0], max_xy[1]-min_xy[1],
                                 linewidth=2, edgecolor='red', facecolor='none', 
                                 linestyle='--', zorder=1)
        ax.add_patch(rect)
        
        ax.set_aspect('equal')
        ax.set_title(f"Raw Strokes (Pixel Coordinates) - {char_key}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        raw_output_path = abs_output_path / f"{char_key}_raw_strokes.png"
        plt.savefig(str(raw_output_path), dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved raw strokes to: {raw_output_path}")
        plt.close()
        
        # ========== 可视化 2：机械臂坐标系下的笔画 ==========
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        colors = plt.cm.jet(np.linspace(0, 1, len(normalized_strokes)))
        for i, stroke in enumerate(normalized_strokes):
            stroke_np = np.array(stroke)
            # 只取 x, y（忽略 z）
            xy = stroke_np[:, :2]
            if len(xy) > 1:
                ax.plot(xy[:, 0], xy[:, 1], 
                       color=colors[i], linewidth=2.5, label=f"Stroke {i+1}", zorder=2)
            # 标记起点
            ax.scatter(xy[0, 0], xy[0, 1], 
                      color=colors[i], s=80, marker='o', edgecolors='white', 
                      linewidths=2, zorder=3)
        
        ax.set_aspect('equal')
        ax.set_title(f"Normalized Strokes (Robot Coordinates) - {char_key}", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (robot coords)")
        ax.set_ylabel("Y (robot coords)")
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        norm_output_path = abs_output_path / f"{char_key}_normalized_strokes.png"
        plt.savefig(str(norm_output_path), dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved normalized strokes to: {norm_output_path}")
        plt.close()
        
        # ========== 可视化 3：对比图（左为原始，右为归一化） ==========
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
        
        # 左图：原始笔画
        all_raw_points = []
        for stroke in raw_strokes:
            all_raw_points.extend(stroke)
        all_raw_points = np.array(all_raw_points)
        min_xy = all_raw_points.min(axis=0)
        max_xy = all_raw_points.max(axis=0)
        
        colors = plt.cm.jet(np.linspace(0, 1, len(raw_strokes)))
        for i, stroke in enumerate(raw_strokes):
            stroke_np = np.array(stroke)
            if len(stroke_np) > 1:
                ax1.plot(stroke_np[:, 0], stroke_np[:, 1], 
                        color=colors[i], linewidth=2, label=f"S{i+1}")
            ax1.scatter(stroke_np[0, 0], stroke_np[0, 1], 
                       color=colors[i], s=60, marker='o', edgecolors='white', linewidths=1.5)
        
        rect1 = patches.Rectangle((min_xy[0], min_xy[1]), 
                                  max_xy[0]-min_xy[0], max_xy[1]-min_xy[1],
                                  linewidth=1.5, edgecolor='red', facecolor='none', linestyle='--')
        ax1.add_patch(rect1)
        ax1.set_aspect('equal')
        ax1.set_title(f"Raw (pixels)", fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9, loc='upper left')
        
        # 右图：归一化笔画
        for i, stroke in enumerate(normalized_strokes):
            stroke_np = np.array(stroke)
            xy = stroke_np[:, :2]
            if len(xy) > 1:
                ax2.plot(xy[:, 0], xy[:, 1], 
                        color=colors[i], linewidth=2, label=f"S{i+1}")
            
            # 标注每个点
            ax2.scatter(xy[:, 0], xy[:, 1], 
                       color=colors[i], s=20, marker='.', alpha=0.8)
            
            # 标注起点
            ax2.scatter(xy[0, 0], xy[0, 1], 
                       color=colors[i], s=60, marker='o', edgecolors='white', linewidths=1.5)
        
        ax2.set_aspect('equal')
        ax2.set_title(f"Normalized (robot coords)", fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9, loc='upper left')
        
        fig.suptitle(f"Strokes Comparison - {char_key}", fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        comp_output_path = abs_output_path / f"{char_key}_comparison.png"
        plt.savefig(str(comp_output_path), dpi=150, bbox_inches='tight')
        print(f"[Visualization] Saved comparison visualization to: {comp_output_path}")
        plt.close()
        
        print(f"[Visualization] All visualizations completed successfully!")
        
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()


class Example:
    def __init__(self, viewer, offset: np.ndarray = None, xpbd_iterations: int = 200, debug_flag: bool = False, resample_dist: float = None):
        # Frame timing
        self.fps = 60
        
        # 弧长重采样参数
        self.resample_dist = resample_dist
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.xpbd_iterations = xpbd_iterations
        self.debug_flag = debug_flag

        self.viewer = viewer
        
        # ------------------------------------------------------------------
        # 命令行参数：坐标偏移量
        # ------------------------------------------------------------------
        if offset is None:
            self.coord_offset = np.array([0.0, 0.0, 0.0])  # 默认偏移
        else:
            self.coord_offset = np.array(offset)
        print(f"[Config] Coordinate offset: {self.coord_offset}")
        
        # ------------------------------------------------------------------
        # 物理仿真开关
        # ------------------------------------------------------------------
        self.enable_brush_physics = True  # 是否启用毛笔物理仿真
        self.brush = None
        self.brush_simulator = None
        self.brush_initialized = False
        
        # ------------------------------------------------------------------
        # 存储原始笔画数据用于最终可视化
        # ------------------------------------------------------------------
        self.raw_strokes = None
        self.recorded_joints = []

        # ------------------------------------------------------------------
        # 加载笔画数据 (参考 font_dataset.py)
        # ------------------------------------------------------------------
        # 尝试多个可能的数据路径
        possible_data_roots = [
            "/home/jizy/project/video_tokenizer/data/test/debug",
            "../../data/infer_output_zhao",
            "../../data/debug",
            "../../data/infer_output_ou",
            "/data1/jizy/output/infer_output",
            "../../data/infer_output",
            "/data1/jizy/font_out/FZSJSJW",  # predict_traj/data
            str(PROJECT_ROOT / "data"),
            "data",
            "../data",
            "../../data",
        ]
        
        data_root = None
        for path in possible_data_roots:
            strokes_file = os.path.join(path, "strokes.pkl")
            imgs_file = os.path.join(path, "images.pkl")
            if os.path.exists(strokes_file) and os.path.exists(imgs_file):
                data_root = path
                break
        
        if data_root is None:
            print("Warning: Could not find font data files. Using default trajectory.")
            print(f"Searched paths: {possible_data_roots}")
            self.use_font_data = False
            self.strokes = []
            self.char_key = None
            self.reference_image = None
            self.raw_strokes = None
        else:
            print(f"Loading font data from: {data_root}")
            # 使用新的加载函数
            raw_strokes, self.reference_image, self.char_key = load_font_data(
                data_root
            )
            print(f"[Debug] raw_strokes loaded: {len(raw_strokes)} strokes")
            
            # 如果指定了重采样距离，则对每个笔画进行弧长重采样
            if self.resample_dist is not None:
                print(f"[Resample] Resampling strokes with step size {self.resample_dist} based on arc length")
                resampled_strokes = []
                for i, stroke in enumerate(raw_strokes):
                    stroke_np = np.array(stroke)
                    stroke_r_record = ""
                    for point in stroke_np:
                        stroke_r_record += f"{point[2]:.6f}, "
                    print(f"[Stroke Info] Stroke {i}: r range = {stroke_np[:,2].min():.4f} to {stroke_np[:,2].max():.4f}, r values = [{stroke_r_record}]")
                    original_len = len(stroke_np)
                    # 使用 resample_stroke_by_dist 进行弧长重采样（只使用 x, y 坐标）
                    resampled = resample_stroke_by_dist(stroke_np, step_size=self.resample_dist, point_dim=2)
                    resampled_strokes.append(resampled.numpy().tolist())
                    print(f"  Stroke {i+1}: {original_len} -> {len(resampled)} points")
                raw_strokes = resampled_strokes
                print(f"[Resample] Resampling completed")
            
            # 存储原始笔画数据用于最终可视化
            self.raw_strokes = raw_strokes
            
            if raw_strokes:
                # 书写区域中心和偏移
                writing_center = np.array([0.45, 0.0, 0.25])
                writing_size = WRITING_WORLD_SIZE  # 书写区域半大小 (24cm wide square)
                
                # 计算墨迹画布的边界 (Workspace Bounds)
                c_off = self.coord_offset
                
                min_x = writing_center[0] - writing_size * 1.1 + c_off[0]
                max_x = writing_center[0] + writing_size * 1.1 + c_off[0]
                min_y = writing_center[1] - writing_size * 1.1 + c_off[1]
                max_y = writing_center[1] + writing_size * 1.1 + c_off[1]
                
                self.canvas_bounds = (min_x, max_x, min_y, max_y)
                
                # 将笔画映射到机械臂工作空间（应用坐标偏移）
                # 第三维(r)会被转换为下压深度，从而得到目标 z 高度
                self.strokes = normalize_strokes_to_workspace(
                    raw_strokes,
                    center=writing_center,  # 书写区域中心
                    size=writing_size,  # 书写区域大小
                    offset=self.coord_offset  # 应用命令行指定的偏移
                )
                self.use_font_data = True
                
                # 书写平面高度（纸面基准高度，用于墨迹渲染）
                self.ink_plane_z = 0.25 + self.coord_offset[2]
                
                # 导出笔画可视化
                print(f"[Debug] Calling visualize_strokes_to_file with char_key={self.char_key}")
                visualize_strokes_to_file(raw_strokes, self.strokes, self.char_key)
                
                print(f"Loaded character '{self.char_key}' with {len(self.strokes)} strokes")
                print(f"[Brush Config] radius={BRUSH_RADIUS:.4f}, max_length={BRUSH_MAX_LENGTH:.4f}, length_ratio={BRUSH_LENGTH_RATIO:.2f}")
                print(f"[Paper Height] ink_plane_z={self.ink_plane_z:.4f}")
                for i, s in enumerate(self.strokes):
                    z_values = [pt[2] for pt in s]
                    z_min, z_max = min(z_values), max(z_values)
                    depth_min = - self.ink_plane_z + z_max  # 最小下压深度
                    depth_max = - self.ink_plane_z + z_min  # 最大下压深度
                    print(f"  Stroke {i+1}: {len(s)} points, z range=[{z_min:.4f}, {z_max:.4f}], depth range=[{depth_min:.4f}, {depth_max:.4f}]")
                    
                # 显示参考图像信息
                if self.reference_image is not None:
                    print(f"Reference image size: {self.reference_image.size}")
            else:
                print("[Debug] No raw_strokes found")
                self.use_font_data = False
                self.strokes = []
                self.reference_image = None
                self.raw_strokes = None
                self.canvas_bounds = (0.3, 0.6, -0.15, 0.15)

        # ------------------------------------------------------------------
        # 轨迹控制状态
        # ------------------------------------------------------------------
        self.current_stroke_idx = 0  # 当前笔画索引
        self.current_point_idx = 0   # 当前笔画中的点索引
        self.is_lifting = False      # 是否正在抬笔
        self.lift_progress = 0.0     # 抬笔进度 [0, 1]
        self.lift_speed = 0.02      # 抬笔速度（越小越慢）
        self.lift_height = 0.08      # 抬笔高度
        self.is_preparing = False    # 笔画开始前的准备（姿态调整+下降）
        self.prepare_progress = 0.0  # 准备进度 [0, 1]
        self.prepare_speed = 0.01   # 准备阶段推进速度（越小越慢，可拉长调整时间）
        self.points_per_frame = 0.1  # 每帧前进的点数（控制书写速度）; 越小越慢 (0.5=原速, 0.1=5倍慢)
        self.point_accumulator = 0.0 # 点累加器（用于非整数速度）
        
        # 存储当前目标位置（用于抬笔过渡）
        self._lift_start_pos = None
        self._lift_end_pos = None
        self._prepare_start_pos = None
        self._prepare_end_pos = None

        # 缓存下一笔的期望运动方向（用于提前设定姿态）
        self._next_stroke_dir = np.array([1.0, 0.0, 0.0])
        
        # 书写完成标志
        self.writing_complete = False
        self._final_saved = False  # 标记是否已保存最终图片

        # ------------------------------------------------------------------
        # 墨迹轨迹参数
        # ------------------------------------------------------------------
        self.ink_trail = []  # 存储墨迹点位置
        # 用“距离采样 + 插值补点”替代按帧采样，避免轨迹离散
        self.ink_step_distance = 0.003  # 墨迹点间距（米）；越小越密
        self.ink_min_move = 0.0008  # 小于该距离不记录，抑制抖动
        self._last_ink_pos = None  # 上一次记录的墨迹点（np.array shape=(3,)）
        self.max_ink_points = 5000  # 最大墨迹点数量
        # 墨迹写在固定书写平面上（统一 Z），再加一点偏移避免 z-fighting
        self.ink_plane_z = 0.0  # 默认书写平面高度；若加载了字体数据会自动覆盖
        self.ink_z_offset = -0.001  # 相对书写平面的Z偏移

        # ------------------------------------------------------------------
        # 木棍参数 (相对于末端执行器)
        # ------------------------------------------------------------------
        self.stick_length = 0.3  # 木棍长度 30cm
        self.stick_radius = 0.005  # 木棍半径 0.5cm
        self.stick_offset = wp.vec3(0.0, 0.0, self.stick_length)
        
        # ------------------------------------------------------------------
        # 毛笔物理仿真参数
        # ------------------------------------------------------------------
        self.brush_length = 0.08  # 毛笔长度 8cm
        self._prev_brush_root_pos = None  # 上一帧毛笔根部位置
        self.brush_sim_substeps = 1  # 物理仿真子步数
        self.brush_particles_positions = None  # 毛笔粒子位置（用于渲染）
        
        # ------------------------------------------------------------------
        # 渲染降采样参数（仿真保持完整粒子数，仅渲染时降采样以提升性能）
        # 仿真: max_hairs=500, max_particles_per_hair=20 (共10000粒子)
        # 渲染: render_hairs=100, render_particles_per_hair=10 (共1000粒子)
        # ------------------------------------------------------------------
        self.render_downsample_enabled = True  # 是否启用渲染降采样
        self.render_hairs = 300  # 渲染时使用的毛发数量
        self.render_particles_per_hair = 20  # 渲染时每根毛发的粒子数量
        self._render_hair_indices = None  # 缓存：用于渲染的毛发索引
        self._render_particle_indices = None  # 缓存：用于渲染的粒子索引（相对于每根毛发）
        self._render_global_indices = None  # 缓存：用于渲染的全局粒子索引（预计算，避免每帧循环）

        # ------------------------------------------------------------------
        # 笔杆倾斜参数
        # ------------------------------------------------------------------
        self.tilt_angle = math.radians(TILT_ANGLE)

        # 用于平滑方向变化的参数
        self._prev_target = None
        self._smoothed_direction = np.array([1.0, 0.0, 0.0])
        
        # 用于平滑旋转目标的参数
        self._current_target_quat = None
        self.quat_smooth_alpha = 0.08
        
        # 笔杆旋转状态跟踪（用于毛笔物理仿真）
        self._prev_brush_rotation = None  # 上一帧的笔杆旋转四元数 (x, y, z, w)
        self._brush_use_rotation = True   # 是否启用笔杆旋转仿真
        
        # 用于平滑关节角的参数
        self._prev_joint_q = None
        self.joint_smooth_alpha = 0.4  # 增大平滑系数，使关节变化更平缓（0.5=50% 新值）
        
        # 关节速度限制（防止关节突变）
        self.max_joint_velocity = 1  # 弧度/秒，限制单帧关节变化
        self.max_joint_delta = self.max_joint_velocity * self.frame_dt  # 单帧最大变化
        
        # 关节加加速度平滑（使关节加速度连续）
        self._prev_joint_velocity = None
        self.max_joint_acceleration = 3.0  # 弧度/秒²
        self.max_accel_delta = self.max_joint_acceleration * self.frame_dt
        
        # 目标位置/旋转的平滑增强
        self.target_pos_smooth_alpha = 0.1  # 目标位置平滑（更新缓慢）
        self._smoothed_target_pos = None

        # 方向变化限制，防止笔画中途大幅转向导致环路
        self.max_dir_change = math.radians(8.0)  # 每帧最大方向改变（弧度）

        # ------------------------------------------------------------------
        # IK 权重/迭代参数
        # ------------------------------------------------------------------
        self.pos_weight = 20.0
        self.rot_weight = 5.0
        self.joint_limit_weight = 1.0

        # ------------------------------------------------------------------
        # Build Franka robot + ground
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            # Path("/data1/jizy/newton-assets/franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        franka.add_ground_plane()

        self.graph = None
        self.model = franka.finalize()
        self.viewer.set_model(self.model)

        # States
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # 创建木棍圆柱体 mesh
        # ------------------------------------------------------------------
        self._create_stick_mesh()
        self._create_tip_mesh()

        # ------------------------------------------------------------------
        # End effector configuration
        # ------------------------------------------------------------------
        self.ee_index = 10  # fr3_hand_tcp

        # 获取并保存初始末端执行器姿态
        body_q_np = self.state.body_q.numpy()
        ee_rot = body_q_np[self.ee_index, 3:7]
        self.base_rotation = np.array([ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3]])
        self.target_rotation = wp.vec4(ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3])


        # ------------------------------------------------------------------
        # IK setup
        # ------------------------------------------------------------------
        initial_target = self._get_initial_target()

        self.pos_obj = ik.IKPositionObjective(
            link_index=self.ee_index,
            link_offset=self.stick_offset,
            target_positions=wp.array([initial_target], dtype=wp.vec3),
            weight=self.pos_weight,
        )

        self.enable_rotation_objective = True
        self.rot_obj = ik.IKRotationObjective(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array([self.target_rotation], dtype=wp.vec4),
            weight=self.rot_weight,
        )

        self.obj_joint_limits = ik.IKJointLimitObjective(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=self.joint_limit_weight,
        )

        self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.ik_iters = 200
        self.solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[self.pos_obj, self.obj_joint_limits] + 
                       ([self.rot_obj] if self.enable_rotation_objective else []),
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianMode.ANALYTIC,
        )

        # ------------------------------------------------------------------
        # 机械臂 Warm Up：在仿真开始前将机械臂移动到第一个笔画起点上方
        # 这样机械臂一加载好，关节角度就已经是目标位置的结果
        # ------------------------------------------------------------------
        # self._warmup_robot_to_initial_position()
        # ------------------------------------------------------------------
        # 初始化毛笔物理仿真（需要在 ee_index 设置之后）
        # ------------------------------------------------------------------
        if self.enable_brush_physics:
            self._initialize_brush_simulation()

        # 初始：如果有字体数据，先在起笔上方进行姿态准备（需在 base_rotation 定义后）
        if self.use_font_data and len(self.strokes) > 0:
            self._enter_prepare_for_stroke(0)

    def _warmup_robot_to_initial_position(self):
        """
        机械臂 Warm Up：通过多次 IK 迭代将机械臂移动到第一个笔画起点上方
        使得仿真开始时机械臂已经处于正确位置，避免初始移动过程
        """
        if not self.use_font_data or len(self.strokes) == 0:
            print("[Warm Up] No font data, skipping warm up")
            return
        
        print("[Warm Up] Starting robot warm up to initial position...")
        
        # 获取初始目标位置（第一个笔画第一个点上方）
        initial_target = self._get_initial_target()
        
        # 计算初始姿态方向（基于第一笔的运动方向）
        if len(self.strokes[0]) >= 2:
            p0 = np.array(self.strokes[0][0], dtype=np.float32)
            p1 = np.array(self.strokes[0][1], dtype=np.float32)
            dir_vec = p1 - p0
            dir_vec[2] = 0.0
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                dir_vec = dir_vec / norm
            else:
                dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        
        # 计算目标旋转姿态
        target_rotation = self._compute_tilt_rotation(dir_vec)
        
        # 更新 IK 目标
        self.pos_obj.set_target_position(0, initial_target)
        if self.enable_rotation_objective:
            self.rot_obj.set_target_rotation(0, target_rotation)
        
        # 执行多次 IK 求解，确保收敛
        warmup_iterations = 100  # warm up 的外层迭代次数
        warmup_ik_iters = 1000   # 每次 warm up 的 IK 迭代次数
        
        for i in range(warmup_iterations):
            try:
                self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count), device=self.viewer.device)
            except TypeError:
                self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))
            
            self.solver.step(self.joint_q, self.joint_q, iterations=warmup_ik_iters)
            
            # 获取求解结果并直接设置为模型关节角
            solved_q = self.joint_q.numpy().reshape(-1)
            self.model.joint_q = wp.array(solved_q, dtype=wp.float32, device=self.viewer.device)
            
            # 更新前向运动学
            newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        
        # 获取最终关节角度
        final_q = self.model.joint_q.numpy()
        
        # 初始化平滑参数为 warm up 结果，避免仿真开始时的突变
        self._prev_joint_q = final_q.copy()
        self._prev_joint_velocity = np.zeros_like(final_q)
        
        # 初始化目标位置平滑参数
        self._smoothed_target_pos = np.array([float(initial_target[0]), float(initial_target[1]), float(initial_target[2])])
        self._prev_target = self._smoothed_target_pos.copy()
        
        # 初始化旋转平滑参数
        self._current_target_quat = np.array([target_rotation[0], target_rotation[1], target_rotation[2], target_rotation[3]])
        
        # 验证末端执行器位置
        body_q_np = self.state.body_q.numpy()
        ee_pos = body_q_np[self.ee_index, 0:3]
        
        # 计算木棍末端位置（需要考虑旋转）
        ee_rot = body_q_np[self.ee_index, 3:7]
        stick_offset_np = np.array([0.0, 0.0, self.stick_length])
        stick_tip_pos = ee_pos + self._rotate_vector(stick_offset_np, ee_rot)
        
        target_np = np.array([float(initial_target[0]), float(initial_target[1]), float(initial_target[2])])
        distance = np.linalg.norm(stick_tip_pos - target_np)
        
        print(f"[Warm Up] Completed!")
        print(f"  Target position: [{target_np[0]:.4f}, {target_np[1]:.4f}, {target_np[2]:.4f}]")
        print(f"  Stick tip position: [{stick_tip_pos[0]:.4f}, {stick_tip_pos[1]:.4f}, {stick_tip_pos[2]:.4f}]")
        print(f"  Position error: {distance:.6f} m")

    def _get_initial_target(self) -> wp.vec3:
        """获取初始目标位置"""
        if self.use_font_data and len(self.strokes) > 0:
            first_point = self.strokes[0][0]
            # 初始位置在第一笔起点上方（抬笔状态）
            return wp.vec3(first_point[0], first_point[1], first_point[2] + self.lift_height)
        else:
            return wp.vec3(0.4, 0.0, 0.5)

    # ----------------------------------------------------------------------
    # 笔画轨迹获取
    # ----------------------------------------------------------------------
    def get_current_trajectory_point(self) -> tuple:
        """
        获取当前轨迹点
        
        Returns:
            (position, is_pen_down): 位置和是否落笔状态
        """
        if not self.use_font_data or len(self.strokes) == 0:
            # 没有字体数据时，使用默认圆形轨迹
            return self._get_default_trajectory_point(), True
        
        if self.writing_complete:
            # 书写完成，保持在最后位置上方
            last_stroke = self.strokes[-1]
            last_point = last_stroke[-1]
            return wp.vec3(last_point[0], last_point[1], last_point[2] + self.lift_height), False
        
        # 处理抬笔过渡
        if self.is_lifting:
            return self._get_lifting_position(), False

        # 笔画开始准备阶段（姿态调整+下降），仍然保持抬笔
        if self.is_preparing:
            return self._get_preparing_position(), False
        
        # 正常书写
        current_stroke = self.strokes[self.current_stroke_idx]
        point_idx = min(int(self.current_point_idx), len(current_stroke) - 1)
        
        # 在两个点之间插值，实现平滑运动
        if point_idx < len(current_stroke) - 1:
            t = self.current_point_idx - int(self.current_point_idx)
            p1 = current_stroke[point_idx]
            p2 = current_stroke[point_idx + 1]
            pos = p1 * (1 - t) + p2 * t
        else:
            pos = current_stroke[point_idx]
        
        return wp.vec3(pos[0], pos[1], pos[2]), True

    def _get_lifting_position(self) -> wp.vec3:
        """计算抬笔过程中的位置"""
        if self._lift_start_pos is None or self._lift_end_pos is None:
            return wp.vec3(0.4, 0.0, 0.5 + self.lift_height)
        
        # 使用平滑的抬笔曲线（先快速抬起，再平移，最后缓慢落下）
        t = self.lift_progress
        
        # 分三段：抬起(0-0.3)、平移(0.3-0.7)、落下(0.7-1.0)
        if t < 0.3:
            # 抬起阶段
            lift_t = t / 0.3
            lift_t = lift_t * lift_t * (3 - 2 * lift_t)  # smoothstep
            x = self._lift_start_pos[0]
            y = self._lift_start_pos[1]
            z = self._lift_start_pos[2] + self.lift_height * lift_t
        elif t < 0.7:
            # 平移阶段
            move_t = (t - 0.3) / 0.4
            move_t = move_t * move_t * (3 - 2 * move_t)  # smoothstep
            x = self._lift_start_pos[0] + (self._lift_end_pos[0] - self._lift_start_pos[0]) * move_t
            y = self._lift_start_pos[1] + (self._lift_end_pos[1] - self._lift_start_pos[1]) * move_t
            z = self._lift_start_pos[2] + self.lift_height
        else:
            # 落下阶段
            drop_t = (t - 0.7) / 0.3
            drop_t = drop_t * drop_t * (3 - 2 * drop_t)  # smoothstep
            x = self._lift_end_pos[0]
            y = self._lift_end_pos[1]
            z = self._lift_end_pos[2] + self.lift_height * (1 - drop_t)
        
        return wp.vec3(x, y, z)

    def _enter_prepare_for_stroke(self, stroke_idx: int):
        """进入指定笔画的准备阶段：先抬笔到起点上方并设定姿态"""
        if stroke_idx >= len(self.strokes):
            return

        # 确保基准旋转存在（初始化早期可能尚未设置）
        if not hasattr(self, "base_rotation") or self.base_rotation is None:
            self.base_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

        start_pt = self.strokes[stroke_idx][0]
        self.is_preparing = True
        self.prepare_progress = 0.0
        self._prepare_start_pos = np.array([start_pt[0], start_pt[1], start_pt[2] + self.lift_height], dtype=np.float32)
        self._prepare_end_pos = np.array([start_pt[0], start_pt[1], start_pt[2]], dtype=np.float32)

        # 提前设置运动方向，用于姿态倾斜在落笔前就收敛
        if len(self.strokes[stroke_idx]) >= 2:
            p0 = np.array(self.strokes[stroke_idx][0], dtype=np.float32)
            p1 = np.array(self.strokes[stroke_idx][1], dtype=np.float32)
            dir_vec = p1 - p0
            dir_vec[2] = 0.0
            norm = np.linalg.norm(dir_vec)
            if norm > 1e-6:
                dir_vec = dir_vec / norm
            else:
                dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            dir_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        self._smoothed_direction = dir_vec.copy()
        self._next_stroke_dir = dir_vec.copy()
        self._prev_target = np.array([start_pt[0], start_pt[1], start_pt[2]], dtype=np.float32)

        # 预设当前旋转目标，使 IK 在抬笔时即可收敛到新方向
        raw_target_rotation = self._compute_tilt_rotation(self._smoothed_direction)
        self._current_target_quat = np.array([
            raw_target_rotation[0], raw_target_rotation[1], raw_target_rotation[2], raw_target_rotation[3]
        ])
        
        # 重置毛笔状态
        if stroke_idx != 0:
            self._reset_brush_state()

    def _get_preparing_position(self) -> wp.vec3:
        """笔画开始前的姿态调整/下降位置"""
        if self._prepare_start_pos is None or self._prepare_end_pos is None:
            return wp.vec3(0.4, 0.0, 0.5 + self.lift_height)

        t = self.prepare_progress
        t = t * t * (3 - 2 * t)  # smoothstep

        x = self._prepare_start_pos[0] * (1 - t) + self._prepare_end_pos[0] * t
        y = self._prepare_start_pos[1] * (1 - t) + self._prepare_end_pos[1] * t
        z = self._prepare_start_pos[2] * (1 - t) + self._prepare_end_pos[2] * t

        return wp.vec3(x, y, z)

    def _get_default_trajectory_point(self) -> wp.vec3:
        """默认圆形轨迹"""
        center = np.array([0.4, 0.0, 0.5])
        radius = 0.1
        speed = 0.5
        angle = self.sim_time * speed
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        z = center[2]
        return wp.vec3(x, y, z)

    def _advance_trajectory(self):
        """推进轨迹状态"""
        if not self.use_font_data or self.writing_complete:
            return

        # 笔画开始前的准备阶段：先在起笔上方调整姿态，再下降到书写面
        if self.is_preparing:
            self.prepare_progress += self.prepare_speed  # 准备阶段速度
            if self.prepare_progress >= 1.0:
                self.prepare_progress = 0.0
                self.is_preparing = False
                # 准备完成后，重置墨迹采样起点，避免落笔瞬间的抖动
                self._last_ink_pos = None
            return
        
        if self.is_lifting:
            # 抬笔过程
            self.lift_progress += self.lift_speed  # 抬笔速度
            if self.lift_progress >= 1.0:
                self.is_lifting = False
                self.lift_progress = 0.0
                self.current_point_idx = 0.0
                # 抵达下一笔起点后，进入准备阶段（姿态调整+下降），仍保持抬笔
                if self.current_stroke_idx < len(self.strokes):
                    self._enter_prepare_for_stroke(self.current_stroke_idx)
            return
        
        # 正常书写，推进点索引
        self.point_accumulator += self.points_per_frame
        if self.point_accumulator >= 1.0:
            advance = int(self.point_accumulator)
            self.point_accumulator -= advance
            self.current_point_idx += advance
        else:
            self.current_point_idx += self.points_per_frame
        
        current_stroke = self.strokes[self.current_stroke_idx]
        
        # 检查是否完成当前笔画
        if self.current_point_idx >= len(current_stroke) - 1:
            self.current_stroke_idx += 1
            
            # 保存当前进度对比图
            self._save_final_comparison_image(suffix=f"_stroke_{self.current_stroke_idx}")

            if self.current_stroke_idx >= len(self.strokes):
                # 所有笔画完成
                self.writing_complete = True
                print(f"Writing complete! Character '{self.char_key}' finished.")
            else:
                # 开始抬笔过渡到下一笔
                self.is_lifting = True
                self.lift_progress = 0.0
                
                # 记录抬笔起点和终点
                self._lift_start_pos = current_stroke[-1].copy()
                self._lift_end_pos = self.strokes[self.current_stroke_idx][0].copy()
                
                print(f"Stroke {self.current_stroke_idx} complete, lifting to stroke {self.current_stroke_idx + 1}")

    # ----------------------------------------------------------------------
    # 笔杆姿态计算（与 example_ik_brush_trajectory.py 相同）
    # ----------------------------------------------------------------------
    def _compute_tilt_rotation(self, movement_dir: np.ndarray) -> wp.vec4:
        """基于运动方向计算倾斜后的目标姿态"""
        dir_xy = np.array([movement_dir[0], movement_dir[1], 0.0])
        dir_norm = np.linalg.norm(dir_xy)
        if dir_norm < 1e-6:
            dir_xy = np.array([1.0, 0.0, 0.0])
        else:
            dir_xy = dir_xy / dir_norm

        tilt_axis = np.array([-dir_xy[1], dir_xy[0], 0.0])
        
        half_angle = self.tilt_angle / 2.0
        sin_half = math.sin(half_angle)
        cos_half = math.cos(half_angle)
        
        tilt_quat = np.array([
            tilt_axis[0] * sin_half,
            tilt_axis[1] * sin_half,
            tilt_axis[2] * sin_half,
            cos_half
        ])
        
        result = self._quat_multiply(tilt_quat, self.base_rotation)
        return wp.vec4(result[0], result[1], result[2], result[3])

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """四元数乘法"""
        x1, y1, z1, w1 = q1
        x2, y2, z2, w2 = q2
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])

    def _quat_slerp(self, q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
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

    # ----------------------------------------------------------------------
    # Simulation
    # ----------------------------------------------------------------------
    def simulate(self):
        try:
            self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count), device=self.viewer.device)
        except TypeError:
            self.joint_q = wp.array(self.model.joint_q, shape=(1, self.model.joint_coord_count))

        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)

        solved_q = self.joint_q.numpy().reshape(-1)
        
        # ==================================================================
        # 关节平滑与限制：多级平滑机制
        # ==================================================================
        
        if self._prev_joint_q is not None:
            # 第一步：速度限制（防止关节突变）
            delta_q = solved_q - self._prev_joint_q
            delta_norm = np.linalg.norm(delta_q)
            
            if delta_norm > 1e-6:
                # 计算单帧最大允许变化
                max_allowed_delta = self.max_joint_delta * len(delta_q)
                
                if delta_norm > max_allowed_delta:
                    # 如果变化太大，限制变化幅度
                    scale = max_allowed_delta / delta_norm
                    delta_q = delta_q * scale
                    solved_q = self._prev_joint_q + delta_q
            
            # 第二步：加速度限制（使关节加速度平滑）
            if self._prev_joint_velocity is not None:
                velocity = delta_q  # 当前速度
                accel = velocity - self._prev_joint_velocity  # 加速度
                accel_norm = np.linalg.norm(accel)
                
                if accel_norm > 1e-6:
                    max_allowed_accel = self.max_accel_delta * len(accel)
                    
                    if accel_norm > max_allowed_accel:
                        # 限制加速度
                        scale = max_allowed_accel / accel_norm
                        accel = accel * scale
                        velocity = self._prev_joint_velocity + accel
                        solved_q = self._prev_joint_q + velocity
                
                self._prev_joint_velocity = velocity.copy()
            else:
                self._prev_joint_velocity = delta_q.copy()
            
            # 第三步：时间滤波（低通滤波平滑关节角）
            # 这是额外的平滑层，使关节变化更加平缓
            smoothed_q = self._prev_joint_q * (1 - self.joint_smooth_alpha) + solved_q * self.joint_smooth_alpha
            
            self._prev_joint_q = smoothed_q.copy()
            self.model.joint_q = wp.array(smoothed_q, dtype=wp.float32, device=self.viewer.device)
        else:
            self._prev_joint_q = solved_q.copy()
            self._prev_joint_velocity = np.zeros_like(solved_q)
            self.model.joint_q = wp.array(solved_q, dtype=wp.float32, device=self.viewer.device)

        # Record current joint configuration
        current_q = self.model.joint_q.numpy().flatten()
        self.recorded_joints.append(current_q)

    def _update_targets(self):
        """更新 IK 目标"""
        # 获取当前轨迹点
        target_pos, is_pen_down = self.get_current_trajectory_point()
        
        # 平滑目标位置（使目标位置变化不会造成关节突变）
        current_pos = np.array([float(target_pos[0]), float(target_pos[1]), float(target_pos[2])])
        
        if self._smoothed_target_pos is None:
            self._smoothed_target_pos = current_pos.copy()
        else:
            # 缓慢平滑更新目标位置，避免 IK 目标突变
            self._smoothed_target_pos = (
                self._smoothed_target_pos * (1 - self.target_pos_smooth_alpha) +
                current_pos * self.target_pos_smooth_alpha
            )
        
        # 计算运动方向（用于笔杆倾斜），限制每帧方向改变，避免笔画中途大幅转向形成环
        if self._prev_target is not None:
            movement = current_pos - self._prev_target
            movement[2] = 0.0
            move_norm = np.linalg.norm(movement)
            
            if move_norm > 1e-6:
                desired_dir = movement / move_norm
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
                else:
                    self._smoothed_direction = np.array([1.0, 0.0, 0.0])
        
        self._prev_target = current_pos.copy()

        # 更新位置目标为平滑后的位置
        smoothed_target = wp.vec3(self._smoothed_target_pos[0], self._smoothed_target_pos[1], self._smoothed_target_pos[2])
        self.pos_obj.set_target_position(0, smoothed_target)
        self._current_target = smoothed_target
        self._is_pen_down = is_pen_down

        # 更新旋转目标
        if self.enable_rotation_objective:
            raw_target_rotation = self._compute_tilt_rotation(self._smoothed_direction)
            raw_quat = np.array([raw_target_rotation[0], raw_target_rotation[1], 
                                 raw_target_rotation[2], raw_target_rotation[3]])
            
            if self._current_target_quat is None:
                self._current_target_quat = raw_quat
            else:
                self._current_target_quat = self._quat_slerp(
                    self._current_target_quat, raw_quat, self.quat_smooth_alpha
                )
            
            target_rotation = wp.vec4(
                self._current_target_quat[0], self._current_target_quat[1],
                self._current_target_quat[2], self._current_target_quat[3]
            )
            self.rot_obj.set_target_rotation(0, target_rotation)

        # 推进轨迹
        self._advance_trajectory()

    def _rotate_vector(self, v, q):
        """使用四元数旋转向量"""
        qx, qy, qz, qw = q
        t = 2.0 * np.cross([qx, qy, qz], v)
        return v + qw * t + np.cross([qx, qy, qz], t)

    def _create_stick_mesh(self):
        """创建木棍的圆柱体 mesh"""
        half_height = self.stick_length / 2.0
        vertices, indices = create_cylinder_mesh(self.stick_radius, half_height, up_axis=2)

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.viewer.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.viewer.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.viewer.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.viewer.device)

        self.viewer.log_mesh("stick_mesh", points, indices, normals, uvs)

        self.stick_xforms = wp.zeros(1, dtype=wp.transform, device=self.viewer.device)
        self.stick_scales = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.stick_colors = wp.array([[0.6, 0.4, 0.2]], dtype=wp.vec3, device=self.viewer.device)
        self.stick_materials = wp.zeros(1, dtype=wp.int32, device=self.viewer.device)

        self._create_target_mesh()

    def _create_target_mesh(self):
        """创建目标点的球体 mesh"""
        from newton.utils import create_sphere_mesh
        vertices, indices = create_sphere_mesh(0.01)

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.viewer.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.viewer.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.viewer.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.viewer.device)

        self.viewer.log_mesh("target_mesh", points, indices, normals, uvs)

        self.target_xforms = wp.zeros(1, dtype=wp.transform, device=self.viewer.device)
        self.target_scales = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.target_colors = wp.array([[1.0, 0.2, 0.2]], dtype=wp.vec3, device=self.viewer.device)
        self.target_materials = wp.zeros(1, dtype=wp.int32, device=self.viewer.device)

    def _create_tip_mesh(self):
        """创建笔尖位置的调试球"""
        from newton.utils import create_sphere_mesh
        vertices, indices = create_sphere_mesh(0.008)

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.viewer.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.viewer.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.viewer.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.viewer.device)

        self.viewer.log_mesh("tip_mesh", points, indices, normals, uvs)

        self.tip_xforms = wp.zeros(1, dtype=wp.transform, device=self.viewer.device)
        self.tip_scales = wp.array([[1.0, 1.0, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        # 落笔时蓝色，抬笔时灰色
        self.tip_colors = wp.array([[0.2, 0.7, 1.0]], dtype=wp.vec3, device=self.viewer.device)
        self.tip_materials = wp.zeros(1, dtype=wp.int32, device=self.viewer.device)

        # 创建墨迹点的 mesh（更小的球体）
        self._create_ink_mesh()

    def _create_ink_mesh(self):
        """创建墨迹点的球体 mesh"""
        from newton.utils import create_sphere_mesh
        vertices, indices = create_sphere_mesh(0.004)  # 墨迹点比笔尖小

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.viewer.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.viewer.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.viewer.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.viewer.device)

        self.viewer.log_mesh("ink_mesh", points, indices, normals, uvs)
        
        # 创建毛笔粒子 mesh（用于渲染毛笔物理仿真粒子）
        self._create_brush_particle_mesh()

    def _create_brush_particle_mesh(self):
        """创建毛笔粒子的球体 mesh"""
        from newton.utils import create_sphere_mesh
        vertices, indices = create_sphere_mesh(0.0005)  # 毛笔粒子

        points = wp.array(vertices[:, 0:3], dtype=wp.vec3, device=self.viewer.device)
        normals = wp.array(vertices[:, 3:6], dtype=wp.vec3, device=self.viewer.device)
        uvs = wp.array(vertices[:, 6:8], dtype=wp.vec2, device=self.viewer.device)
        indices = wp.array(indices, dtype=wp.int32, device=self.viewer.device)

        self.viewer.log_mesh("brush_particle_mesh", points, indices, normals, uvs)

    # ----------------------------------------------------------------------
    # 毛笔物理仿真相关方法
    # ----------------------------------------------------------------------
    def _initialize_brush_simulation(self):
        """初始化毛笔和 XPBD 物理仿真器"""
        print("[Brush Physics] Initializing brush simulation...")
        
        # 获取初始木棍端点位置作为毛笔根部位置
        initial_brush_root = self._get_initial_brush_root_position()
        print(f"[Brush Physics] Initial brush root position: {initial_brush_root}")
        
        # 创建毛笔模型
        self.brush = build_brush(
            root_position=torch.tensor(initial_brush_root, dtype=torch.float64),
            stick_length=self.stick_length
        )
        
        # 生成毛笔毛发和约束
        self.brush.gen_hairs()
        self.brush.gen_constraints()

        # 记录初始粒子相对于根部的偏移（此时笔是竖直向下的）
        self._initial_particle_offsets = []
        for p in self.brush.particles:
            # 使用 clone() 确保保存的是副本
            offset = p.position - self.brush.root_position
            self._initial_particle_offsets.append(offset.clone())
        
        print(f"[Brush Physics] Brush created with {len(self.brush.particles)} particles")
        print(f"[Brush Physics] Brush has {len(self.brush.hairs)} hairs")
        print(f"[Brush Physics] Brush has {len(self.brush.constraints)} constraints")
        
        # 计算总步数（基于轨迹点数）
        total_trajectory_points = 0
        if self.use_font_data and len(self.strokes) > 0:
            for stroke in self.strokes:
                total_trajectory_points += len(stroke)
            # 加上抬笔过渡的估计帧数
            total_trajectory_points += len(self.strokes) * 100  # 每个笔画间约 100 帧过渡
        else:
            total_trajectory_points = 10000  # 默认轨迹
        
        # 确保最小步数
        num_sim_steps = max(total_trajectory_points * 2, 1000)
        
        min_x, max_x, min_y, max_y = self.canvas_bounds
        print(f"[Brush Physics] Canvas Bounds: X[{min_x:.3f}, {max_x:.3f}], Y[{min_y:.3f}, {max_y:.3f}]")

        # 创建 XPBD 仿真器
        self.brush_simulator = XPBDSimulator(
            dt=self.frame_dt,
            substeps=self.brush_sim_substeps,
            iterations=self.xpbd_iterations,
            num_steps=num_sim_steps,
            batch_size=1,
            gravity=torch.tensor([0.0, 0.0, -MAX_GRAVITY], dtype=torch.float64),
            dis_compliance=1e-8,
            variable_dis_compliance=1e-8,
            bending_compliance=1e-7,
            damping=0.3,
            canvas_resolution=256,
            canvas_min_x=min_x,
            canvas_max_x=max_x,
            canvas_min_y=min_y,
            canvas_max_y=max_y,
        )
        
        # 加载毛笔到仿真器
        self.brush_simulator.load_brush(self.brush)
        
        # 初始化位移序列（使用零位移初始化，稍后每帧动态更新）
        initial_displacements = torch.zeros((1, num_sim_steps, 3), dtype=torch.float64)
        
        # 初始化旋转序列（如果启用旋转仿真）
        if self._brush_use_rotation:
            # 创建单位四元数序列 (w, x, y, z) 格式
            initial_rotations = identity_quaternion((num_sim_steps,), dtype=torch.float64)
            self.brush_simulator.load_displacements(initial_displacements[0], initial_rotations)
        else:
            self.brush_simulator.load_displacements(initial_displacements[0])
        
        # 记录当前毛笔根部位置
        self._prev_brush_root_pos = np.array(initial_brush_root, dtype=np.float64)
        
        # 初始化笔杆旋转状态
        initial_rotation = self._get_stick_rotation()
        self._prev_brush_rotation = initial_rotation.copy()
        
        # 仿真时间步计数器
        self.brush_sim_step = 0
        
        self.brush_initialized = True
        print("[Brush Physics] Brush simulation initialized successfully!")
        
        # 输出渲染降采样配置信息
        if self.render_downsample_enabled:
            sim_total = self.brush.max_hairs * self.brush.max_particles_per_hair
            render_hairs = min(self.render_hairs, self.brush.max_hairs)
            render_particles = min(self.render_particles_per_hair, self.brush.max_particles_per_hair)
            render_total = render_hairs * render_particles
            print(f"[Brush Rendering] Downsampling enabled:")
            print(f"  Simulation: {self.brush.max_hairs} hairs × {self.brush.max_particles_per_hair} particles = {sim_total} total")
            print(f"  Rendering:  {render_hairs} hairs × {render_particles} particles = {render_total} total")
            print(f"  Reduction ratio: {render_total / sim_total * 100:.1f}%")

    def _reset_brush_state(self):
        """重置毛笔状态：粒子回归初始直线形态，并根据当前笔杆方向旋转"""
        if self.brush is None or self.brush_simulator is None:
            return
            
        # 获取当前笔杆位置（作为毛笔根部）
        current_root_pos = torch.tensor(self._get_stick_tip_position(), dtype=torch.float64)
        
        # 使用当前笔杆的实际旋转，而不是目标旋转
        # 这样能确保重置后的物理状态与当前画面中的笔杆状态一致，避免非固定粒子产生突变位移
        # _get_stick_rotation 返回 [x, y, z, w]
        current_stick_quat = self._get_stick_rotation()
        q = torch.tensor([current_stick_quat[0], current_stick_quat[1], current_stick_quat[2], current_stick_quat[3]], dtype=torch.float64)
        
        # 重置所有粒子位置
        for i, p in enumerate(self.brush.particles):
            if i >= len(self._initial_particle_offsets):
                break
                
            offset = self._initial_particle_offsets[i]
            
            # 应用四元数旋转：v_rot = v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v)
            uv = torch.cross(q[:3], offset)
            uuv = torch.cross(q[:3], uv)
            rotated_offset = offset + 2.0 * (q[3] * uv + uuv)
            
            p.position = current_root_pos + rotated_offset
            # 重置速度为0
            p.velocity = torch.zeros_like(p.velocity)
            
        # 同步更新 Brush 对象的根部属性
        self.brush.root_position = current_root_pos
        
        # 同时更新 tangent_vector
        tangent_init = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64) # 初始向下
        uv_t = torch.cross(q[:3], tangent_init)
        uuv_t = torch.cross(q[:3], uv_t)
        self.brush.tangent_vector = tangent_init + 2.0 * (q[3] * uv_t + uuv_t)
        
        # 重新加载到 Simulator 以重置 GPU 状态 (positions, velocities, constraints)
        # 传入 keep_ink_canvas=True 避免清除之前的墨迹
        self.brush_simulator.load_brush(self.brush, keep_ink_canvas=True)

    def _get_initial_brush_root_position(self) -> list:
        """获取初始毛笔根部位置（对齐木棍端点）"""
        # 使用初始目标位置作为参考
        if self.use_font_data and len(self.strokes) > 0:
            first_point = self.strokes[0][0]
            # 毛笔根部在目标点上方（与木棍端点对齐）
            return [first_point[0], first_point[1], first_point[2] + self.lift_height]
        else:
            return [0.4, 0.0, 0.5 + self.lift_height]

    def _update_brush_simulation(self):
        """更新毛笔物理仿真 - 将毛笔根部与木棍端点对齐，并同步笔杆旋转"""
        if not self.brush_initialized or self.brush_simulator is None:
            return
        
        # 获取当前木棍端点位置
        stick_tip_pos = self._get_stick_tip_position()
        current_pos = np.array([float(stick_tip_pos[0]), float(stick_tip_pos[1]), float(stick_tip_pos[2])], dtype=np.float64)
        
        # 计算位移（当前位置 - 上一帧位置）
        if self._prev_brush_root_pos is not None:
            displacement = current_pos - self._prev_brush_root_pos
        else:
            displacement = np.zeros(3, dtype=np.float64)
        
        self._prev_brush_root_pos = current_pos.copy()
        
        # 限制位移幅度防止数值不稳定
        max_displacement = 0.05  # 最大 5cm 每帧
        disp_norm = np.linalg.norm(displacement)
        if disp_norm > max_displacement:
            displacement = displacement * (max_displacement / disp_norm)
        
        # 计算旋转增量（如果启用旋转仿真）
        delta_rotation = None
        if self._brush_use_rotation and self.brush_simulator.use_rotation:
            current_rotation = self._get_stick_rotation()
            if self._prev_brush_rotation is not None:
                delta_rotation = self._compute_delta_rotation(self._prev_brush_rotation, current_rotation)
            else:
                # 第一帧使用单位四元数（无旋转）
                delta_rotation = np.array([1.0, 0.0, 0.0, 0.0])  # (w, x, y, z)
            self._prev_brush_rotation = current_rotation.copy()
        
        # 更新仿真器中的位移和旋转
        if self.brush_sim_step < self.brush_simulator.num_steps - 1:
            # 将位移转换为 warp 格式并应用
            disp_tensor = torch.tensor(displacement, dtype=torch.float64).unsqueeze(0)
            
            # 直接更新 displacement 数组中的当前步
            # 注意：需要 clone 和 detach 来避免 in-place 操作错误
            if self.brush_simulator.displacement is not None:
                disp_wp = wp.to_torch(self.brush_simulator.displacement).clone().detach()
                disp_wp[0, self.brush_sim_step] = disp_tensor
                self.brush_simulator.displacement = wp.from_torch(disp_wp, dtype=wp.vec3d, requires_grad=True)
            
            # 更新旋转序列（如果启用旋转仿真）
            if delta_rotation is not None and self.brush_simulator.delta_rotation_seq is not None:
                rot_tensor = torch.tensor(delta_rotation, dtype=torch.float64).unsqueeze(0)
                rot_wp = wp.to_torch(self.brush_simulator.delta_rotation_seq).clone().detach()
                rot_wp[0, self.brush_sim_step] = rot_tensor
                self.brush_simulator.delta_rotation_seq = wp.from_torch(rot_wp, dtype=wp.vec4d, requires_grad=True)
            
            # 执行一步仿真
            try:
                self.brush_simulator.step(self.brush_sim_step, self.brush_sim_step + 1)
                self.brush_sim_step += 1
            except Exception as e:
                print(f"[Brush Physics] Simulation step failed: {e}")
        
        # 提取粒子位置用于渲染
        self._extract_brush_particles_for_rendering()

    def _extract_brush_particles_for_rendering(self):
        """提取毛笔粒子位置用于渲染（支持降采样以提升渲染性能）"""
        if self.brush_simulator is None or self.brush_simulator.positions is None:
            return
        
        try:
            # 从 warp 数组获取粒子位置
            positions_tensor = wp.to_torch(self.brush_simulator.positions)
            # positions shape: (batch, num_steps, num_particles, 3) -> 我们取 batch=0, step=current
            step_idx = min(self.brush_sim_step - 1, positions_tensor.shape[1] - 1)
            all_particles_pos = positions_tensor[0, step_idx].cpu().detach().numpy()
            
            # 如果启用降采样，只提取部分粒子用于渲染
            if self.render_downsample_enabled and self.brush is not None:
                self.brush_particles_positions = self._downsample_particles_for_rendering(all_particles_pos)
            else:
                self.brush_particles_positions = all_particles_pos
        except Exception as e:
            print(f"[Brush Physics] Failed to extract particle positions: {e}")
            self.brush_particles_positions = None
    
    def _build_render_particle_indices(self):
        """
        构建并缓存用于渲染的全局粒子索引数组
        
        只在初始化或参数改变时调用一次，后续直接使用缓存的索引数组进行快速提取
        """
        if self.brush is None:
            return
        
        # 获取仿真参数
        sim_hairs = self.brush.max_hairs  # 仿真毛发数量 (500)
        sim_particles_per_hair = self.brush.max_particles_per_hair  # 仿真每根毛发粒子数 (20)
        
        # 渲染参数
        render_hairs = min(self.render_hairs, sim_hairs)  # 渲染毛发数量 (100)
        render_particles = min(self.render_particles_per_hair, sim_particles_per_hair)  # 渲染每根毛发粒子数 (10)
        
        # 均匀选择毛发索引
        self._render_hair_indices = np.linspace(0, sim_hairs - 1, render_hairs, dtype=np.int32)
        
        # 均匀选择粒子索引（相对于每根毛发）
        self._render_particle_indices = np.linspace(0, sim_particles_per_hair - 1, render_particles, dtype=np.int32)
        
        # 预计算全局粒子索引数组（关键优化：避免每帧循环计算）
        # 对于每根选中的毛发，计算其选中粒子的全局索引
        global_indices = []
        for hair_idx in self._render_hair_indices:
            hair_start = hair_idx * sim_particles_per_hair
            for particle_idx in self._render_particle_indices:
                global_indices.append(hair_start + particle_idx)
        
        self._render_global_indices = np.array(global_indices, dtype=np.int32)
        
        print(f"[Brush Rendering] Built render indices: {len(self._render_global_indices)} particles")
    
    def _downsample_particles_for_rendering(self, all_particles_pos: np.ndarray) -> np.ndarray:
        """
        对粒子进行降采样用于渲染
        
        仿真使用完整的粒子数（如 500根毛发 × 20粒子 = 10000粒子），
        但渲染时只使用一部分（如 100根毛发 × 10粒子 = 1000粒子）以提升性能。
        
        Args:
            all_particles_pos: 所有粒子的位置数组 shape=(num_particles, 3)
            
        Returns:
            降采样后的粒子位置数组
        """
        # 检查是否需要构建索引缓存
        if self._render_global_indices is None:
            self._build_render_particle_indices()
        
        # 使用缓存的全局索引直接提取粒子位置（numpy高级索引，一次性完成）
        valid_indices = self._render_global_indices[self._render_global_indices < len(all_particles_pos)]
        return all_particles_pos[valid_indices].astype(np.float32)

    def _render_brush_particles(self):
        """渲染毛笔粒子"""
        if self.brush_particles_positions is None or len(self.brush_particles_positions) == 0:
            return
        
        n_particles = len(self.brush_particles_positions)
        
        # 创建变换数组
        brush_xforms_list = []
        brush_colors_list = []
        
        for i, pos in enumerate(self.brush_particles_positions):
            tf = wp.transform(
                wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                wp.quat_identity()
            )
            brush_xforms_list.append(tf)
            # 毛笔颜色：深棕色到黑色的渐变（模拟蘸墨效果）
            z_ratio = max(0, min(1, pos[2] / self.brush_length))
            # r = 0.15 * (1 - z_ratio) + 0.05 * z_ratio
            # g = 0.1 * (1 - z_ratio) + 0.03 * z_ratio
            # b = 0.05 * (1 - z_ratio) + 0.02 * z_ratio
            r = 1.0
            g = 1.0
            b = 1.0
            brush_colors_list.append([r, g, b])
        
        brush_xforms = wp.array(brush_xforms_list, dtype=wp.transform, device=self.viewer.device)
        brush_scales = wp.array([[1.0, 1.0, 1.0]] * n_particles, dtype=wp.vec3, device=self.viewer.device)
        brush_colors = wp.array(brush_colors_list, dtype=wp.vec3, device=self.viewer.device)
        brush_materials = wp.zeros(n_particles, dtype=wp.int32, device=self.viewer.device)
        
        self.viewer.log_instances(
            "brush_particles_instance",
            "brush_particle_mesh",
            brush_xforms,
            brush_scales,
            brush_colors,
            brush_materials,
        )

    def _update_ink_trail(self, tip_pos, is_pen_down):
        """更新墨迹轨迹（按距离采样，并在段内插值补点）"""
        if not is_pen_down:
            self._last_ink_pos = None
            return

        current = np.array(
            [float(tip_pos[0]), float(tip_pos[1]), float(self.ink_plane_z) + self.ink_z_offset],
            dtype=np.float32,
        )

        if self._last_ink_pos is None:
            self.ink_trail.append(current)
            self._last_ink_pos = current
            return

        delta = current - self._last_ink_pos
        dist = float(np.linalg.norm(delta))

        if dist < self.ink_min_move:
            return

        step = max(float(self.ink_step_distance), 1e-6)
        n_new = int(dist / step)

        # 至少补 1 个点，确保连续
        n_new = max(n_new, 1)

        for i in range(1, n_new + 1):
            t = i / (n_new + 0.0)
            p = self._last_ink_pos * (1.0 - t) + current * t
            self.ink_trail.append(p)

        # 维持最大点数（保留最新的）
        if len(self.ink_trail) > self.max_ink_points:
            excess = len(self.ink_trail) - self.max_ink_points
            del self.ink_trail[:excess]

        self._last_ink_pos = current

    def _render_ink_trail(self):
        """渲染墨迹轨迹"""
        if len(self.ink_trail) == 0:
            return
        
        n_points = len(self.ink_trail)
        
        # 创建变换数组
        ink_xforms = wp.zeros(n_points, dtype=wp.transform, device=self.viewer.device)
        ink_scales = wp.array([[1.0, 1.0, 1.0]] * n_points, dtype=wp.vec3, device=self.viewer.device)
        ink_colors = wp.array([[0.05, 0.05, 0.05]] * n_points, dtype=wp.vec3, device=self.viewer.device)  # 深黑色墨迹
        ink_materials = wp.zeros(n_points, dtype=wp.int32, device=self.viewer.device)
        
        # 设置每个墨迹点的位置
        xforms_np = np.zeros((n_points, 7), dtype=np.float32)
        for i, pos in enumerate(self.ink_trail):
            xforms_np[i, 0:3] = pos  # 位置
            xforms_np[i, 3:7] = [0.0, 0.0, 0.0, 1.0]  # 单位四元数
        
        # 转换为 warp transform 数组
        ink_xforms_list = []
        for i in range(n_points):
            tf = wp.transform(
                wp.vec3(xforms_np[i, 0], xforms_np[i, 1], xforms_np[i, 2]),
                wp.quat_identity()
            )
            ink_xforms_list.append(tf)
        
        ink_xforms = wp.array(ink_xforms_list, dtype=wp.transform, device=self.viewer.device)
        
        self.viewer.log_instances(
            "ink_trail_instance",
            "ink_mesh",
            ink_xforms,
            ink_scales,
            ink_colors,
            ink_materials,
        )

    def _get_stick_tip_position(self):
        """计算实际木棍端点位置"""
        body_q_np = self.state.body_q.numpy()
        ee_pos = body_q_np[self.ee_index, :3]
        ee_rot = body_q_np[self.ee_index, 3:7]

        stick_offset_np = np.array([self.stick_offset[0], self.stick_offset[1], self.stick_offset[2]])
        tip_pos = ee_pos + self._rotate_vector(stick_offset_np, ee_rot)
        return wp.vec3(tip_pos[0], tip_pos[1], tip_pos[2])

    def _get_stick_rotation(self) -> np.ndarray:
        """获取木棍（笔杆）的当前旋转四元数
        
        Returns:
            四元数数组 (x, y, z, w) 格式，与 newton/warp 的格式一致
        """
        body_q_np = self.state.body_q.numpy()
        ee_rot = body_q_np[self.ee_index, 3:7]  # (x, y, z, w) 格式
        return ee_rot.copy()

    def _compute_delta_rotation(self, prev_quat: np.ndarray, curr_quat: np.ndarray) -> np.ndarray:
        """计算从上一帧到当前帧的增量旋转四元数
        
        Args:
            prev_quat: 上一帧的四元数 (x, y, z, w) 格式
            curr_quat: 当前帧的四元数 (x, y, z, w) 格式
            
        Returns:
            增量旋转四元数 (w, x, y, z) 格式（用于 XPBD 仿真器）
            delta_rotation = curr_quat * inverse(prev_quat)
        """
        # 计算 prev_quat 的逆（共轭）
        prev_quat_inv = np.array([-prev_quat[0], -prev_quat[1], -prev_quat[2], prev_quat[3]])
        
        # delta = curr * prev_inv
        # 四元数乘法 (x1,y1,z1,w1) * (x2,y2,z2,w2)
        x1, y1, z1, w1 = curr_quat
        x2, y2, z2, w2 = prev_quat_inv
        
        delta_xyzw = np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2
        ])
        
        # 转换为 (w, x, y, z) 格式供 XPBD 仿真器使用
        delta_wxyz = np.array([delta_xyzw[3], delta_xyzw[0], delta_xyzw[1], delta_xyzw[2]])
        
        return delta_wxyz

    def _get_stick_transform(self):
        """计算木棍的世界变换"""
        body_q_np = self.state.body_q.numpy()
        ee_pos = body_q_np[self.ee_index, :3]
        ee_rot = body_q_np[self.ee_index, 3:7]

        half_offset = np.array([0.0, 0.0, self.stick_length / 2.0])
        stick_center = ee_pos + self._rotate_vector(half_offset, ee_rot)

        return wp.transform(
            wp.vec3(stick_center[0], stick_center[1], stick_center[2]),
            wp.quat(ee_rot[0], ee_rot[1], ee_rot[2], ee_rot[3])
        )

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self._update_targets()
        self.simulate()
        
        # 更新毛笔物理仿真
        if self.enable_brush_physics and self.brush_initialized and not self.debug_flag:
            self._update_brush_simulation()
        
        self.sim_time += self.frame_dt
        
        # 检测书写完成，调用 test_final 保存结果
        if self.writing_complete and not self._final_saved:
            self._final_saved = True
            self.test_final()

    def test_final(self):
        """仿真结束时保存对比图片"""
        self._save_final_comparison_image()
        self._save_joint_trajectory()

    def _save_joint_trajectory(self):
        """保存关节轨迹到文件"""
        if not self.recorded_joints:
            print("[Final] No joint data recorded.")
            return
            
        try:
            output_path = "stroke_visualizations/joint_trajectory.pt"
            # Convert list of arrays to a single numpy array
            joints_array = np.array(self.recorded_joints)
            # Save as torch tensor
            torch.save(torch.from_numpy(joints_array), output_path)
            print(f"[Final] Saved joint trajectory to: {os.path.abspath(output_path)} with shape {joints_array.shape}")
        except Exception as e:
            print(f"[Final] Failed to save joint trajectory: {e}")

    def _save_final_comparison_image(self, suffix: str = "_final_comparison"):
        """
        保存最终对比图片，包含三个子图：
        1. 数据中的真值图片（reference_image）
        2. 毛笔物理仿真器的 ink_canvas
        3. 数据轨迹经过处理后的笔画可视化
        """
        if not HAS_MATPLOTLIB:
            print("[Final] matplotlib not available, skipping final comparison image")
            return
        
        print(f"[Final] Saving comparison image ({suffix})...")
        
        # 创建输出目录
        output_dir = Path("stroke_visualizations")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建三列子图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        char_key = self.char_key if self.char_key else "unknown"
        
        # ========== 子图1：真值图片 ==========
        ax1 = axes[0]
        if self.reference_image is not None:
            # 将图片转换为 numpy 数组以便后续处理
            ref_img_np = np.array(self.reference_image)
            
            # 如果是白底黑字，尝试裁剪到文字区域
            try:
                # 转换为灰度图计算边界
                if ref_img_np.ndim == 3:
                    gray_img = np.mean(ref_img_np, axis=2)
                else:
                    gray_img = ref_img_np
                
                # 寻找非白色像素 (假设阈值为 240)
                non_white_indices = np.where(gray_img < 240)
                
                if len(non_white_indices[0]) > 0:
                    y_min, y_max = np.min(non_white_indices[0]), np.max(non_white_indices[0])
                    x_min, x_max = np.min(non_white_indices[1]), np.max(non_white_indices[1])
                    
                    # 增加一点边缘留白 (padding)
                    pad = 10
                    h, w = gray_img.shape
                    y_min = max(0, y_min - pad)
                    y_max = min(h, y_max + pad)
                    x_min = max(0, x_min - pad)
                    x_max = min(w, x_max + pad)
                    
                    # 裁剪图片
                    cropped_img = ref_img_np[y_min:y_max, x_min:x_max]
                    ax1.imshow(cropped_img)
                else:
                    # 如果找不到黑色像素，显示原图
                    ax1.imshow(self.reference_image)
            except Exception as e:
                print(f"Warning: Failed to crop reference image: {e}")
                ax1.imshow(self.reference_image)
                
            ax1.set_title(f"Ground Truth - {char_key}", fontsize=12, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "No reference image", ha='center', va='center', fontsize=14)
            ax1.set_title("Ground Truth (N/A)", fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        # ========== 子图2：毛笔物理仿真 ink_canvas ==========
        ax2 = axes[1]
        ink_canvas_img = None
        
        if self.enable_brush_physics and self.brush_simulator is not None:
            try:
                # 获取 ink_canvas 数据（参照标准处理方法）
                raw_tensor = wp.to_torch(self.brush_simulator.ink_canvas)  # shape: (batch_size, canvas_size)
                
                # 处理batch维度的数据
                batch_size = raw_tensor.shape[0]
                res = self.brush_simulator.canvas_resolution
                
                if batch_size > 1:
                    # 保存第一个batch的结果
                    image_tensor = raw_tensor[0].reshape((res, res))
                    print(f"[Final] Batch size: {batch_size}, using first batch result")
                else:
                    # 兼容单batch情况
                    image_tensor = raw_tensor[0].reshape((res, res))
                
                # 应用 sigmoid 归一化处理
                image_tensor = (torch.sigmoid(image_tensor) - 0.5) * 2.0
                canvas_2d = image_tensor.cpu().detach().numpy()
                
                # 反转颜色（墨迹为黑色）
                ink_canvas_img = 1.0 - canvas_2d
                ax2.imshow(ink_canvas_img, cmap='gray', vmin=0, vmax=1)
                ax2.set_title(f"Brush Physics Ink Canvas", fontsize=12, fontweight='bold')
            except Exception as e:
                print(f"[Final] Failed to get ink canvas: {e}")
                ax2.text(0.5, 0.5, f"Error: {e}", ha='center', va='center', fontsize=10)
                ax2.set_title("Ink Canvas (Error)", fontsize=12, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "Brush physics disabled", ha='center', va='center', fontsize=14)
            ax2.set_title("Ink Canvas (Disabled)", fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # ========== 子图3：处理后的轨迹数据 ==========
        ax3 = axes[2]
        if self.strokes and len(self.strokes) > 0:
            # 计算 r 值进行可视化
            base_z = 0.25 + self.coord_offset[2]
            
            from matplotlib.collections import PatchCollection
            
            # 为每个笔画分配颜色
            colors = plt.cm.jet(np.linspace(0, 1, len(self.strokes)))

            all_points = []

            for i, stroke in enumerate(self.strokes):
                stroke_np = np.array(stroke)
                xy = stroke_np[:, :2]
                z_vals = stroke_np[:, 2]
                if len(xy) == 0: continue
                
                all_points.append(xy)
                
                # 计算 r 值 (保持原有逻辑)
                depths = - (z_vals - base_z - BRUSH_MAX_LENGTH)
                r_vals = (depths) * BRUSH_RADIUS / (BRUSH_MAX_LENGTH * BRUSH_LENGTH_RATIO)
                r_vals = np.maximum(r_vals, 0) # 确保半径非负
                
                # 创建圆的 Patch 集合
                circle_patches = []
                record_stroke_r = ""
                for j in range(len(xy)):
                    if r_vals[j] > 1e-5: # 忽略太小的圆
                        record_stroke_r += f"{r_vals[j]:.4f}, "
                        circle = patches.Circle((xy[j, 0], xy[j, 1]), r_vals[j])
                        circle_patches.append(circle)
                print(f"[Final] Stroke {i+1} radii: {record_stroke_r}")
                
                if circle_patches:
                    # 使用 PatchCollection 提高性能
                    p = PatchCollection(circle_patches, alpha=0.3, facecolor=colors[i], edgecolor='none')
                    ax3.add_collection(p)
                
                # 绘制中心轨迹线
                if len(xy) > 1:
                    ax3.plot(xy[:, 0], xy[:, 1], color='gray', alpha=0.5, linewidth=0.5, zorder=2)
                
                # 标记起点
                ax3.text(xy[0, 0], xy[0, 1], str(i+1), color='red', fontsize=10, fontweight='bold', zorder=3)

            ax3.set_aspect('equal')
            ax3.set_title(f"Visualized Radius (r) - {char_key}", fontsize=12, fontweight='bold')
            ax3.set_xlabel("X (robot coords)")
            ax3.set_ylabel("Y (robot coords)")
            ax3.grid(True, alpha=0.3)
            
            # 手动更新坐标轴范围 (add_collection 不会自动更新)
            if all_points:
                all_points_np = np.concatenate(all_points)
                x_min, x_max = all_points_np[:, 0].min(), all_points_np[:, 0].max()
                y_min, y_max = all_points_np[:, 1].min(), all_points_np[:, 1].max()
                margin = 0.02
                ax3.set_xlim(x_min - margin, x_max + margin)
                ax3.set_ylim(y_min - margin, y_max + margin)

        else:
            ax3.text(0.5, 0.5, "No stroke data", ha='center', va='center', fontsize=14)
            ax3.set_title("Normalized Strokes (N/A)", fontsize=12, fontweight='bold')
        
        # 添加总标题
        offset_str = f"({self.coord_offset[0]:.2f}, {self.coord_offset[1]:.2f}, {self.coord_offset[2]:.2f})"
        fig.suptitle(f"Simulation Result Comparison - {char_key} | Offset: {offset_str}", 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存图片
        output_path = output_dir / f"{char_key}{suffix}.png"
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        print(f"[Final] Saved comparison image to: {output_path.resolve()}")
        plt.close()

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.viewer.log_state(self.state)

        # 渲染木棍
        stick_tf = self._get_stick_transform()
        self.stick_xforms.fill_(stick_tf)
        self.viewer.log_instances(
            "stick_instance",
            "stick_mesh",
            self.stick_xforms,
            self.stick_scales,
            self.stick_colors,
            self.stick_materials,
        )

        # 渲染目标点
        if hasattr(self, '_current_target'):
            target_tf = wp.transform(self._current_target, wp.quat_identity())
            self.target_xforms.fill_(target_tf)
            
            # 根据落笔状态改变目标点颜色
            if hasattr(self, '_is_pen_down') and self._is_pen_down:
                self.target_colors = wp.array([[1.0, 0.2, 0.2]], dtype=wp.vec3, device=self.viewer.device)
            else:
                self.target_colors = wp.array([[0.5, 0.5, 0.5]], dtype=wp.vec3, device=self.viewer.device)
            
            # self.viewer.log_instances(
            #     "target_instance",
            #     "target_mesh",
            #     self.target_xforms,
            #     self.target_scales,
            #     self.target_colors,
            #     self.target_materials,
            # )

            # 渲染笔尖（木棍端点）
            tip_pos = self._get_stick_tip_position()
            tip_tf = wp.transform(tip_pos, wp.quat_identity())
            self.tip_xforms.fill_(tip_tf)

            # 根据落笔状态改变笔尖颜色
            if hasattr(self, '_is_pen_down') and self._is_pen_down:
                self.tip_colors = wp.array([[0.2, 0.7, 1.0]], dtype=wp.vec3, device=self.viewer.device)
            else:
                self.tip_colors = wp.array([[0.7, 0.7, 0.7]], dtype=wp.vec3, device=self.viewer.device)
            #
            # self.viewer.log_instances(
            #     "tip_instance",
            #     "tip_mesh",
            #     self.tip_xforms,
            #     self.tip_scales,
            #     self.tip_colors,
            #     self.tip_materials,
            # )
            
            # 渲染毛笔粒子（物理仿真）
            if self.enable_brush_physics and self.brush_initialized and not self.debug_flag:
                self._render_brush_particles()

            # 更新并渲染墨迹轨迹（使用毛笔最低点或木棍端点）
            if self.enable_brush_physics and self.brush_particles_positions is not None:
                # 使用毛笔最低粒子作为墨迹参考点
                brush_tip_pos = self._get_brush_tip_position()
                if brush_tip_pos is not None:
                    self._update_ink_trail(brush_tip_pos, self._is_pen_down)
            else:
                self._update_ink_trail(tip_pos, self._is_pen_down)
            # self._render_ink_trail()

            # 每秒打印状态
            if int(self.sim_time * self.fps) % self.fps == 0:
                dx = float(tip_pos[0] - self._current_target[0])
                dy = float(tip_pos[1] - self._current_target[1])
                dz = float(tip_pos[2] - self._current_target[2])
                err = math.sqrt(dx * dx + dy * dy + dz * dz)
                
                status = "PEN_DOWN" if self._is_pen_down else "LIFTING"
                stroke_info = f"stroke {self.current_stroke_idx + 1}/{len(self.strokes)}" if self.use_font_data else "default"
                
                brush_info = ""
                if self.enable_brush_physics and self.brush_initialized:
                    brush_info = f"  brush_step={self.brush_sim_step}"
                    if self.brush_particles_positions is not None:
                        # 获取毛笔最低点 z 坐标
                        min_z = np.min(self.brush_particles_positions[:, 2])
                        brush_info += f"  brush_min_z={min_z:.4f}"
                
                print(f"[{status}] [{(self.sim_time / self.frame_dt):.2f} frames] {stroke_info}  err={err:.4f}m  "
                      f"tilt_dir=({self._smoothed_direction[0]:.2f},{self._smoothed_direction[1]:.2f}){brush_info}")

        self.viewer.end_frame()
        wp.synchronize()

    def _get_brush_tip_position(self):
        """获取毛笔最低粒子位置（作为笔尖）"""
        if self.brush_particles_positions is None or len(self.brush_particles_positions) == 0:
            return None
        
        # 找到 z 坐标最小的粒子
        min_z_idx = np.argmin(self.brush_particles_positions[:, 2])
        pos = self.brush_particles_positions[min_z_idx]
        return wp.vec3(float(pos[0]), float(pos[1]), float(pos[2]))


def parse_custom_args():
    """解析自定义命令行参数"""
    parser = argparse.ArgumentParser(
        description="Calligraphy writing simulation with robotic arm",
        add_help=False  # 避免与 newton.examples.init() 的参数冲突
    )
    parser.add_argument(
        '--offset', 
        type=float, 
        nargs=3, 
        default=[0.0, 0.0, 0.0],
        metavar=('X', 'Y', 'Z'),
        help='Coordinate offset to apply to loaded trajectory (default: 0.0 0.0 -0.1)'
    )
    parser.add_argument(
        '--xpbd_iterations',
        type=int,
        default=200,
        help='Number of iterations for XPBD solver (default: 200)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Precompute total simulation frames based on trajectory data'
    )
    parser.add_argument(
        '--resample_dist',
        type=float,
        default=None,
        metavar='D',
        help='Target arc length distance between intervals for resampling strokes. If not specified, no resampling is performed.'
    )
    
    # 只解析已知参数，其余留给 newton.examples.init()
    known_args, remaining_args = parser.parse_known_args()

    if known_args.debug:
        # iterations = 1
        known_args.xpbd_iterations = 1
    
    # 恢复 sys.argv 以便 newton.examples.init() 处理剩余参数
    sys.argv = [sys.argv[0]] + remaining_args
    
    return known_args


if __name__ == "__main__":
    # 先解析自定义参数
    custom_args = parse_custom_args()
    # 然后初始化 newton viewer
    viewer, args = newton.examples.init()
    
    # 创建示例，传入坐标偏移参数和重采样参数
    example = Example(
        viewer, 
        offset=custom_args.offset, 
        xpbd_iterations=custom_args.xpbd_iterations, 
        debug_flag=custom_args.debug,
        resample_dist=custom_args.resample_dist
    )
    
    newton.examples.run(example, args)