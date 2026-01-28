# 毛笔仿真环境 (RL Module)

这个模块提供了一个不依赖机械臂的毛笔物理仿真环境，可用于强化学习训练或直接仿真书法书写。

## 主要特点

- **无机械臂依赖**: 直接使用轨迹坐标和旋转驱动毛笔物理仿真，无需 IK 求解
- **独立笔杆旋转计算**: 根据运动方向自动计算笔杆倾斜角度
- **完整的物理仿真**: 使用 XPBD 物理引擎模拟毛笔毛发行为
- **画布渲染**: 实时生成墨迹画布图像
- **Gym 兼容接口**: 提供标准的 Gym 环境接口（可选）

## 文件结构

```
rl/
├── __init__.py              # 模块入口
├── brush_simulation_env.py  # 核心仿真环境
├── gym_brush_env.py         # Gym 兼容环境（可选）
├── test_brush_env.py        # 测试脚本
└── README.md                # 本文档
```

## 快速开始

### 基本使用

```python
from rl import BrushSimulationEnv

# 创建仿真环境
env = BrushSimulationEnv(
    offset=[0.0, 0.0, 0.0],      # 坐标偏移
    xpbd_iterations=200,          # XPBD 迭代次数
    resample_dist=0.005,          # 弧长重采样距离
    canvas_resolution=256,        # 画布分辨率
    output_dir="output"           # 输出目录
)

# 加载数据
env.load_data("path/to/data", char_key="某字")

# 初始化仿真
env.initialize_brush_simulation()

# 运行完整仿真
canvas = env.run_simulation(
    points_per_frame=0.5,         # 每帧前进点数
    save_intermediate=True        # 保存中间结果
)

# 保存对比图
env.save_comparison_image("_result")
```

### 逐步控制

```python
import numpy as np
from rl import BrushSimulationEnv

env = BrushSimulationEnv()
env.initialize_brush_simulation(initial_position=np.array([0.4, 0.0, 0.3]))

# 逐步执行仿真
prev_pos = None
trajectory = [...]  # 轨迹点列表

for point in trajectory:
    current_pos = np.array(point)
    
    # 根据运动方向计算旋转
    rotation = env.compute_rotation_from_direction(current_pos, prev_pos)
    
    # 执行一步仿真
    env.step_simulation(current_pos, rotation)
    
    prev_pos = current_pos

# 获取画布
canvas = env.get_canvas()
```

### Gym 环境（强化学习）

```python
from rl import GymBrushEnv

# 创建 Gym 环境
env = GymBrushEnv(
    data_root="path/to/data",
    canvas_resolution=64,
    max_steps=500,
    action_type='delta',      # 'delta' 或 'absolute'
    reward_type='dense'       # 'sparse' 或 'dense'
)

# 标准 Gym 接口
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        break

canvas = env.get_canvas()
env.close()
```

## 命令行使用

```bash
# 直接运行仿真
python brush_simulation_env.py --data_root ../../data/debug --char_key "某字"

# 运行测试
python test_brush_env.py --data_root ../../data/debug --test all
```

### 命令行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_root` | str | None | 数据目录路径 |
| `--char_key` | str | None | 字符 key（默认随机） |
| `--offset` | float×3 | 0 0 0 | 坐标偏移量 |
| `--xpbd_iterations` | int | 200 | XPBD 迭代次数 |
| `--resample_dist` | float | None | 弧长重采样距离 |
| `--canvas_resolution` | int | 256 | 画布分辨率 |
| `--output_dir` | str | rl_output | 输出目录 |
| `--points_per_frame` | float | 0.5 | 每帧前进点数 |
| `--save_intermediate` | flag | False | 保存中间结果 |

## API 参考

### BrushSimulationEnv

核心仿真环境类。

#### 方法

- `load_data(data_root, char_key=None)`: 加载笔画数据
- `initialize_brush_simulation(initial_position=None)`: 初始化物理仿真
- `run_simulation(...)`: 运行完整仿真
- `step_simulation(position, rotation)`: 执行单步仿真
- `compute_rotation_from_direction(current_pos, prev_pos)`: 计算笔杆旋转
- `get_canvas()`: 获取当前画布
- `reset_brush_state(position, rotation)`: 重置毛笔状态
- `save_comparison_image(suffix)`: 保存对比图

### 工具函数

- `load_font_data(data_root, char_key)`: 加载字体数据
- `normalize_strokes_to_workspace(strokes, center, size, offset)`: 归一化笔画到工作空间
- `compute_brush_depth_from_radius(radius)`: 从目标半径计算下压深度
- `compute_tilt_rotation(movement_dir, tilt_angle, base_rotation)`: 计算倾斜旋转
- `quat_multiply(q1, q2)`: 四元数乘法
- `quat_slerp(q1, q2, t)`: 四元数球面插值

## 数据格式

### 输入数据

数据目录应包含：
- `strokes.pkl`: 笔画数据字典 `{char_key: [stroke1, stroke2, ...]}`
  - 每个 stroke 是 `[[x, y, r], ...]` 格式的点列表
  - `r` 是目标切面半径（归一化到 [-1, 1]）
- `images.pkl`: 参考图像字典 `{char_key: image_bytes}`

### 输出

- 画布图像: numpy 数组 `(resolution, resolution)`，值域 [0, 1]
- 对比图: PNG 图像，包含真值、仿真结果和轨迹可视化

## 依赖

必需：
- numpy
- torch
- warp

可选：
- matplotlib（用于可视化）
- gymnasium/gym（用于 Gym 环境）
- PIL（用于图像处理）

## 与原版的区别

相比 `example_ik_brush_trajectory.py`：

1. **移除机械臂**: 不再需要 Newton 机械臂模型和 IK 求解
2. **独立旋转计算**: 根据轨迹自动计算笔杆旋转，不依赖末端执行器姿态
3. **简化接口**: 直接使用位置和旋转驱动仿真
4. **RL 友好**: 提供标准 Gym 接口，支持强化学习训练
5. **更快执行**: 移除 IK 求解开销，仿真更高效

## 许可证

Apache-2.0
