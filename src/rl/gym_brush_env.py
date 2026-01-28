# SPDX-FileCopyrightText: Copyright (c) 2025
# SPDX-License-Identifier: Apache-2.0
"""
Gym 兼容的毛笔仿真环境

提供标准的 Gym 接口，用于强化学习训练
"""

import math
import numpy as np
import torch
import warp as wp
from typing import Optional, Tuple, Dict, Any

try:
    import gymnasium as gym
    from gymnasium import spaces
    HAS_GYM = True
except ImportError:
    try:
        import gym
        from gym import spaces
        HAS_GYM = True
    except ImportError:
        HAS_GYM = False
        print("[Warning] gymnasium/gym not installed, GymBrushEnv will not be available")

from .brush_simulation_env import (
    BrushSimulationEnv,
    normalize_strokes_to_workspace,
    compute_tilt_rotation,
    quat_slerp,
    BRUSH_RADIUS,
    BRUSH_MAX_LENGTH,
    BRUSH_LENGTH_RATIO,
    TILT_ANGLE,
    WRITING_WORLD_SIZE,
)


if HAS_GYM:
    class GymBrushEnv(gym.Env):
        """
        Gym 兼容的毛笔仿真环境
        
        观测空间:
            - 当前笔刷位置 (3,)
            - 当前笔刷旋转 (4,) 四元数
            - 目标轨迹点 (3,)
            - 上一帧画布特征 (可选)
            
        动作空间:
            - 位置增量 (3,) 或绝对位置 (3,)
            - 旋转增量 (4,) 或绝对旋转 (4,)
            
        奖励:
            - 轨迹跟踪误差
            - 画布相似度
            - 平滑度奖励
        """
        
        metadata = {'render_modes': ['human', 'rgb_array']}
        
        def __init__(self,
                     data_root: str = None,
                     canvas_resolution: int = 64,
                     max_steps: int = 500,
                     action_type: str = 'delta',  # 'delta' or 'absolute'
                     reward_type: str = 'sparse',  # 'sparse' or 'dense'
                     xpbd_iterations: int = 50,
                     render_mode: str = None):
            """
            初始化 Gym 环境
            
            Args:
                data_root: 数据目录
                canvas_resolution: 画布分辨率
                max_steps: 最大步数
                action_type: 动作类型 ('delta' 增量, 'absolute' 绝对)
                reward_type: 奖励类型 ('sparse' 稀疏, 'dense' 密集)
                xpbd_iterations: XPBD 迭代次数
                render_mode: 渲染模式
            """
            super().__init__()
            
            self.data_root = data_root
            self.canvas_resolution = canvas_resolution
            self.max_steps = max_steps
            self.action_type = action_type
            self.reward_type = reward_type
            self.render_mode = render_mode
            
            # 创建基础仿真环境
            self.sim_env = BrushSimulationEnv(
                xpbd_iterations=xpbd_iterations,
                canvas_resolution=canvas_resolution,
                output_dir="gym_output"
            )
            
            # 定义动作空间
            if action_type == 'delta':
                # 位置增量 + 旋转增量（欧拉角）
                self.action_space = spaces.Box(
                    low=np.array([-0.01, -0.01, -0.01, -0.1, -0.1, -0.1]),
                    high=np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1]),
                    dtype=np.float32
                )
            else:
                # 绝对位置 + 绝对旋转（四元数）
                self.action_space = spaces.Box(
                    low=np.array([0.3, -0.2, 0.2, -1, -1, -1, -1]),
                    high=np.array([0.6, 0.2, 0.4, 1, 1, 1, 1]),
                    dtype=np.float32
                )
            
            # 定义观测空间
            # 位置(3) + 旋转(4) + 目标位置(3) + 速度(3) = 13
            obs_dim = 13
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32
            )
            
            # 状态变量
            self.current_pos = np.zeros(3)
            self.current_rot = np.array([0, 0, 0, 1.0])  # 单位四元数
            self.prev_pos = np.zeros(3)
            self.target_trajectory = []
            self.current_target_idx = 0
            self.step_count = 0
            
            self._initialized = False
        
        def _load_random_character(self):
            """加载随机字符数据"""
            if self.data_root is None:
                # 生成简单的测试轨迹
                self.target_trajectory = self._generate_test_trajectory()
            else:
                if self.sim_env.load_data(self.data_root):
                    # 将所有笔画展平为单一轨迹
                    self.target_trajectory = []
                    for stroke in self.sim_env.strokes:
                        self.target_trajectory.extend(stroke.tolist())
                else:
                    self.target_trajectory = self._generate_test_trajectory()
        
        def _generate_test_trajectory(self) -> list:
            """生成测试轨迹（简单的正弦曲线）"""
            num_points = 100
            trajectory = []
            for i in range(num_points):
                t = i / (num_points - 1)
                x = 0.4 + 0.1 * t
                y = 0.05 * np.sin(2 * np.pi * t)
                z = 0.27
                trajectory.append([x, y, z])
            return trajectory
        
        def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
            """
            重置环境
            
            Returns:
                observation, info
            """
            super().reset(seed=seed)
            
            # 加载新的字符数据
            self._load_random_character()
            
            # 初始化仿真器
            if len(self.target_trajectory) > 0:
                initial_pos = np.array(self.target_trajectory[0])
                initial_pos[2] += 0.05  # 从上方开始
            else:
                initial_pos = np.array([0.4, 0.0, 0.3])
            
            if not self._initialized:
                self.sim_env.initialize_brush_simulation(initial_pos)
                self._initialized = True
            else:
                # 重置毛笔状态
                self.current_rot = np.array([0, 0, 0, 1.0])
                self.sim_env.reset_brush_state(initial_pos, self.current_rot)
            
            self.current_pos = initial_pos.copy()
            self.prev_pos = initial_pos.copy()
            self.current_target_idx = 0
            self.step_count = 0
            
            obs = self._get_observation()
            info = {'target_trajectory_length': len(self.target_trajectory)}
            
            return obs, info
        
        def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
            """
            执行一步动作
            
            Args:
                action: 动作向量
                
            Returns:
                observation, reward, terminated, truncated, info
            """
            self.step_count += 1
            
            # 解析动作
            if self.action_type == 'delta':
                # 增量动作
                pos_delta = action[:3]
                rot_delta = action[3:6]  # 欧拉角增量
                
                self.prev_pos = self.current_pos.copy()
                self.current_pos = self.current_pos + pos_delta
                
                # 将欧拉角增量转换为旋转
                self.current_rot = self._apply_euler_rotation(self.current_rot, rot_delta)
            else:
                # 绝对动作
                self.prev_pos = self.current_pos.copy()
                self.current_pos = action[:3]
                self.current_rot = action[3:7]
                self.current_rot = self.current_rot / (np.linalg.norm(self.current_rot) + 1e-8)
            
            # 执行仿真
            self.sim_env.step_simulation(self.current_pos, self.current_rot)
            
            # 更新目标索引
            if self.current_target_idx < len(self.target_trajectory) - 1:
                self.current_target_idx += 1
            
            # 计算奖励
            reward = self._compute_reward()
            
            # 检查终止条件
            terminated = self.current_target_idx >= len(self.target_trajectory) - 1
            truncated = self.step_count >= self.max_steps
            
            obs = self._get_observation()
            info = {
                'step': self.step_count,
                'target_idx': self.current_target_idx,
                'position_error': self._get_position_error()
            }
            
            return obs, reward, terminated, truncated, info
        
        def _get_observation(self) -> np.ndarray:
            """获取当前观测"""
            # 当前目标
            if self.current_target_idx < len(self.target_trajectory):
                target = np.array(self.target_trajectory[self.current_target_idx])
            else:
                target = np.array(self.target_trajectory[-1])
            
            # 速度
            velocity = self.current_pos - self.prev_pos
            
            obs = np.concatenate([
                self.current_pos,     # 3
                self.current_rot,     # 4
                target,               # 3
                velocity,             # 3
            ]).astype(np.float32)
            
            return obs
        
        def _compute_reward(self) -> float:
            """计算奖励"""
            if len(self.target_trajectory) == 0:
                return 0.0
            
            # 位置误差
            pos_error = self._get_position_error()
            
            if self.reward_type == 'sparse':
                # 稀疏奖励：只有接近目标时才给正奖励
                if pos_error < 0.005:
                    return 1.0
                else:
                    return -0.01
            else:
                # 密集奖励
                reward = -pos_error * 10.0  # 位置误差惩罚
                
                # 平滑度奖励
                velocity = self.current_pos - self.prev_pos
                smoothness_reward = -np.linalg.norm(velocity) * 0.1
                
                return reward + smoothness_reward
        
        def _get_position_error(self) -> float:
            """获取位置误差"""
            if self.current_target_idx < len(self.target_trajectory):
                target = np.array(self.target_trajectory[self.current_target_idx])
            else:
                target = np.array(self.target_trajectory[-1])
            
            return np.linalg.norm(self.current_pos - target)
        
        def _apply_euler_rotation(self, quat: np.ndarray, euler_delta: np.ndarray) -> np.ndarray:
            """应用欧拉角增量到四元数"""
            # 简化实现：小角度近似
            roll, pitch, yaw = euler_delta
            
            # 构建增量四元数
            cr = np.cos(roll / 2)
            sr = np.sin(roll / 2)
            cp = np.cos(pitch / 2)
            sp = np.sin(pitch / 2)
            cy = np.cos(yaw / 2)
            sy = np.sin(yaw / 2)
            
            delta_quat = np.array([
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
                cr * cp * cy + sr * sp * sy
            ])
            
            # 四元数乘法
            x1, y1, z1, w1 = quat
            x2, y2, z2, w2 = delta_quat
            
            result = np.array([
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            ])
            
            return result / (np.linalg.norm(result) + 1e-8)
        
        def render(self):
            """渲染当前状态"""
            if self.render_mode == 'rgb_array':
                return self.sim_env.get_canvas()
            elif self.render_mode == 'human':
                # 简单的文本输出
                print(f"Step {self.step_count}: pos={self.current_pos}, target_idx={self.current_target_idx}")
            return None
        
        def close(self):
            """关闭环境"""
            pass
        
        def get_canvas(self) -> np.ndarray:
            """获取当前画布"""
            return self.sim_env.get_canvas()


# 如果 gym 不可用，创建一个占位符类
if not HAS_GYM:
    class GymBrushEnv:
        def __init__(self, *args, **kwargs):
            raise ImportError("gymnasium or gym is required for GymBrushEnv. "
                            "Please install with: pip install gymnasium")
