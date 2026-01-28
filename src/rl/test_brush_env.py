#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试毛笔仿真环境

运行方式:
    python test_brush_env.py --data_root ../../data/debug
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import time
import numpy as np

from rl import BrushSimulationEnv


def test_basic_simulation(data_root: str = None, char_key: str = None):
    """测试基本的仿真流程"""
    print("=" * 60)
    print("测试基本仿真流程")
    print("=" * 60)
    
    # 创建环境
    env = BrushSimulationEnv(
        offset=[0.0, 0.0, 0.0],
        xpbd_iterations=100,
        resample_dist=0.005,
        canvas_resolution=256,
        output_dir="test_output"
    )
    
    # 加载数据
    if data_root is None:
        # 尝试查找数据目录
        possible_paths = [
            str(PROJECT_ROOT / "data" / "debug"),
            str(PROJECT_ROOT / "data" / "infer_output"),
            str(PROJECT_ROOT.parent / "data" / "debug"),
            str(PROJECT_ROOT.parent / "data" / "infer_output"),
        ]
        
        import os
        for path in possible_paths:
            if os.path.exists(os.path.join(path, "strokes.pkl")):
                data_root = path
                break
        
        if data_root is None:
            print("[Error] 找不到数据目录，请指定 --data_root")
            return False
    
    print(f"使用数据目录: {data_root}")
    
    if not env.load_data(data_root, char_key):
        print("[Error] 加载数据失败")
        return False
    
    # 初始化仿真
    env.initialize_brush_simulation()
    
    # 运行仿真
    print("\n开始仿真...")
    time_start = time.time()
    
    canvas = env.run_simulation(
        points_per_frame=0.5,
        save_intermediate=True
    )
    
    time_end = time.time()
    print(f"仿真完成，耗时: {time_end - time_start:.2f} 秒")
    
    # 保存结果
    env.save_comparison_image("_test_result")
    
    print(f"画布形状: {canvas.shape}")
    print(f"画布值范围: [{canvas.min():.3f}, {canvas.max():.3f}]")
    
    return True


def test_step_by_step_control(data_root: str = None):
    """测试逐步控制"""
    print("\n" + "=" * 60)
    print("测试逐步控制")
    print("=" * 60)
    
    env = BrushSimulationEnv(
        xpbd_iterations=50,
        canvas_resolution=128,
        output_dir="test_output"
    )
    
    # 使用简单的测试轨迹
    # 创建一个简单的直线轨迹
    num_points = 50
    test_trajectory = []
    for i in range(num_points):
        t = i / (num_points - 1)
        x = 0.4 + 0.1 * t
        y = 0.0
        z = 0.27  # 固定高度
        test_trajectory.append([x, y, z])
    
    print(f"测试轨迹: {len(test_trajectory)} 个点")
    
    # 手动设置数据（不从文件加载）
    env.strokes = [np.array(test_trajectory, dtype=np.float32)]
    env.canvas_bounds = (0.3, 0.6, -0.1, 0.1)
    
    # 初始化
    initial_pos = np.array([0.4, 0.0, 0.35])
    env.initialize_brush_simulation(initial_pos)
    
    # 逐步执行
    print("开始逐步控制...")
    prev_pos = initial_pos.copy()
    
    for i, point in enumerate(test_trajectory):
        current_pos = np.array(point)
        
        # 计算旋转
        rotation = env.compute_rotation_from_direction(current_pos, prev_pos)
        
        # 执行仿真步
        env.step_simulation(current_pos, rotation)
        
        prev_pos = current_pos.copy()
        
        if i % 10 == 0:
            print(f"  步骤 {i}/{len(test_trajectory)}, 位置: ({current_pos[0]:.3f}, {current_pos[1]:.3f}, {current_pos[2]:.3f})")
    
    # 获取画布
    canvas = env.get_canvas()
    print(f"最终画布形状: {canvas.shape}")
    
    return True


def test_rotation_computation():
    """测试旋转计算"""
    print("\n" + "=" * 60)
    print("测试旋转计算")
    print("=" * 60)
    
    from rl import compute_tilt_rotation, quat_multiply, quat_slerp
    
    # 测试不同运动方向的旋转
    directions = [
        np.array([1.0, 0.0, 0.0]),   # 向前
        np.array([0.0, 1.0, 0.0]),   # 向左
        np.array([-1.0, 0.0, 0.0]),  # 向后
        np.array([0.0, -1.0, 0.0]),  # 向右
        np.array([0.707, 0.707, 0.0]),  # 45度
    ]
    
    base_rot = np.array([0.0, 0.0, 0.0, 1.0])
    
    for direction in directions:
        rotation = compute_tilt_rotation(direction, base_rotation=base_rot)
        print(f"方向 {direction} -> 四元数 {rotation}")
    
    # 测试插值
    q1 = compute_tilt_rotation(np.array([1.0, 0.0, 0.0]))
    q2 = compute_tilt_rotation(np.array([0.0, 1.0, 0.0]))
    
    print("\n四元数插值测试:")
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        q_interp = quat_slerp(q1, q2, t)
        print(f"  t={t:.2f}: {q_interp}")
    
    return True


def test_gym_env():
    """测试 Gym 环境（如果可用）"""
    print("\n" + "=" * 60)
    print("测试 Gym 环境")
    print("=" * 60)
    
    try:
        from rl import GymBrushEnv
        if GymBrushEnv is None:
            print("GymBrushEnv 不可用 (需要 gymnasium/gym)")
            return True
    except ImportError as e:
        print(f"无法导入 GymBrushEnv: {e}")
        return True
    
    # 创建环境
    env = GymBrushEnv(
        data_root=None,  # 使用测试轨迹
        canvas_resolution=64,
        max_steps=100,
        action_type='delta'
    )
    
    # 重置
    obs, info = env.reset()
    print(f"初始观测形状: {obs.shape}")
    print(f"初始 info: {info}")
    
    # 执行几步随机动作
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if terminated or truncated:
            break
    
    print(f"执行 {step + 1} 步后累计奖励: {total_reward:.4f}")
    
    env.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="测试毛笔仿真环境")
    parser.add_argument('--data_root', type=str, default=None, help='数据目录路径')
    parser.add_argument('--char_key', type=str, default=None, help='字符 key')
    parser.add_argument('--test', type=str, default='all', 
                        choices=['all', 'basic', 'step', 'rotation', 'gym'],
                        help='要运行的测试')
    
    args = parser.parse_args()
    
    results = {}
    
    if args.test in ['all', 'rotation']:
        results['rotation'] = test_rotation_computation()
    
    if args.test in ['all', 'step']:
        results['step'] = test_step_by_step_control(args.data_root)
    
    if args.test in ['all', 'basic']:
        results['basic'] = test_basic_simulation(args.data_root, args.char_key)
    
    if args.test in ['all', 'gym']:
        results['gym'] = test_gym_env()
    
    # 打印结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"  {name}: {status}")


if __name__ == "__main__":
    main()
