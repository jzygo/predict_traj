# RL 模块初始化文件
"""
RL module for brush simulation environment

This module provides a brush simulation environment that does not depend on
robotic arm IK solving. It directly uses trajectory coordinates and computed
rotations to drive the brush physics simulation.

主要组件:
- BrushSimulationEnv: 核心仿真环境，直接控制毛笔位置和旋转
- GymBrushEnv: Gym 兼容的 RL 训练环境（需要 gymnasium/gym）

使用示例:
    # 直接使用仿真环境
    from rl import BrushSimulationEnv
    env = BrushSimulationEnv()
    env.load_data("path/to/data")
    env.initialize_brush_simulation()
    canvas = env.run_simulation()
    
    # 使用 Gym 环境进行 RL 训练
    from rl import GymBrushEnv
    env = GymBrushEnv(data_root="path/to/data")
    obs, info = env.reset()
    obs, reward, done, truncated, info = env.step(action)
"""

from .brush_simulation_env import (
    BrushSimulationEnv,
    load_font_data,
    normalize_strokes_to_workspace,
    compute_brush_depth_from_radius,
    compute_radius_from_brush_depth,
    compute_tilt_rotation,
    build_brush,
    quat_multiply,
    quat_slerp,
)

# 尝试导入 Gym 环境（可选依赖）
try:
    from .gym_brush_env import GymBrushEnv
    _HAS_GYM_ENV = True
except ImportError:
    _HAS_GYM_ENV = False
    GymBrushEnv = None

__all__ = [
    'BrushSimulationEnv',
    'load_font_data',
    'normalize_strokes_to_workspace',
    'compute_brush_depth_from_radius',
    'compute_radius_from_brush_depth',
    'compute_tilt_rotation',
    'build_brush',
    'quat_multiply',
    'quat_slerp',
]

if _HAS_GYM_ENV:
    __all__.append('GymBrushEnv')
