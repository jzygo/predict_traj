import os
import sys
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image
import matplotlib
import torch
import warp as wp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

from config import BRUSH_UP_POSITION, MAX_GRAVITY
from simulation.model.brush import Brush
from simulation.model.constraints import VariableDistanceConstraint, FixedPointConstraint, DistanceConstraint, BendingConstraint

# 使用无界面后端，便于子进程并行渲染
matplotlib.use("Agg")

PLANE_Z = 0.25

def _render_single_frame(args: Tuple[int, torch.Tensor, Path]):
    """在子进程中渲染单帧并保存为 PNG。

    参数:
    - args: (frame_index, coords(Nx3), out_dir)
    """
    i, coords, out_dir = args
    # Convert tensor to numpy if needed for matplotlib
    if isinstance(coords, torch.Tensor):
        coords = coords.numpy()
    # 每个子进程独立创建 Figure，互不干扰
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(
        coords[:, 0], coords[:, 1], coords[:, 2],
        c=coords[:, 2], cmap='viridis', s=10, alpha=0.8
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Step {i+1}')
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_zlim(0, 1.0)
    # 保持颜色条一致性可选：不同帧范围可能不同，这里按全局固定不易实现，使用每帧自适应
    # 保存帧
    out_path = out_dir / f"frame_{i:05d}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return str(out_path)


@wp.kernel
def solve_ground_collision(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_step: int,
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        restitution: wp.float64,
        friction: wp.float64,
        num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] <= 0.0:  # Fixed particles
        return

    pos = positions[batch_id, time_step, tid]
    vel = velocities[batch_id, tid]

    # Ground collision (z = 0)
    if pos[2] < wp.float64(0.0):
        # Correct position
        positions[batch_id, time_step, tid] = wp.vec3d(pos[0], pos[1], wp.float64(0.0))

        # Apply restitution and friction
        vel_normal = vel[2]
        vel_tangent = wp.vec3d(vel[0], vel[1], wp.float64(0.0))

        if vel_normal < wp.float64(0.0):
            vel_normal = -vel_normal * restitution
            vel_tangent = vel_tangent * (wp.float64(1.0) - friction)
            velocities[batch_id, tid] = wp.vec3d(vel_tangent[0], vel_tangent[1], vel_normal)


@wp.kernel
def step_time_id(time_id: int):
    pass
    # time_id += 1


@wp.kernel
def compute_l1_loss(
        sim_image: wp.array(dtype=wp.float64, ndim=2),
        ref_image: wp.array(dtype=wp.float64),
        height: int,
        width: int,
        batch_id: int,
        loss: wp.array(dtype=wp.float64)
):
    tid = wp.tid()
    if tid >= height * width:
        return

    sim_pixel = (wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-sim_image[batch_id, tid])) - wp.float64(0.5)) * wp.float64(2)
    diff = sim_pixel - ref_image[tid]
    scale = wp.float64(1.0) / (wp.float64(height) * wp.float64(width))
    wp.atomic_add(loss, 0, wp.abs(diff) * scale)


@wp.kernel
def integrate_particles(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        external_forces: wp.array(dtype=wp.vec3d, ndim=2),
        dt: wp.float64,
        num_particles: int,
        time_id: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] > wp.float64(0.0):
        force = external_forces[batch_id, tid]

        max_force = wp.float64(100.0)
        force_magnitude = wp.length(force)
        if force_magnitude > max_force:
            force = force * (max_force / force_magnitude)

        velocity_delta = force * inv_masses[batch_id, tid] * dt
        new_velocity = velocities[batch_id, tid] + velocity_delta

        max_velocity = wp.float64(10.0)
        velocity_magnitude = wp.length(new_velocity)
        if velocity_magnitude > max_velocity:
            new_velocity = new_velocity * (max_velocity / velocity_magnitude)

        velocities[batch_id, tid] = new_velocity

        displacement = new_velocity * dt
        max_displacement = wp.float64(1.0)
        displacement_magnitude = wp.length(displacement)
        if displacement_magnitude > max_displacement:
            displacement = displacement * (max_displacement / displacement_magnitude)

        positions[batch_id, time_id, tid] = positions[batch_id, time_id, tid] + displacement


@wp.kernel
def apply_step_displacement_to_fixed_positions(
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        disp_seq: wp.array(dtype=wp.vec3d, ndim=2),
        time_id: int,
        num_fixed: int
):
    batch_id, tid = wp.tid()
    if tid >= num_fixed:
        return
    d = disp_seq[batch_id, time_id]
    fixed_positions[batch_id, tid] = fixed_positions[batch_id, tid] + d


@wp.func
def quat_multiply(q1: wp.vec4d, q2: wp.vec4d) -> wp.vec4d:
    """四元数乘法: q1 * q2"""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return wp.vec4d(
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    )


@wp.func
def quat_conjugate(q: wp.vec4d) -> wp.vec4d:
    """四元数共轭"""
    return wp.vec4d(q[0], -q[1], -q[2], -q[3])


@wp.func
def quat_rotate_vector(q: wp.vec4d, v: wp.vec3d) -> wp.vec3d:
    """使用四元数旋转向量: q * v * q^-1"""
    # 将向量转换为纯四元数 (0, v)
    v_quat = wp.vec4d(wp.float64(0.0), v[0], v[1], v[2])
    # q * v * q^-1
    q_conj = quat_conjugate(q)
    result = quat_multiply(quat_multiply(q, v_quat), q_conj)
    return wp.vec3d(result[1], result[2], result[3])


@wp.kernel
def apply_rotation_displacement_to_fixed_positions(
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_local_coords: wp.array(dtype=wp.vec3d, ndim=1),  # 固定粒子相对于笔杆中心的局部坐标
        brush_center: wp.array(dtype=wp.vec3d, ndim=2),  # 每个batch每个时间步的笔杆中心位置
        rotation_seq: wp.array(dtype=wp.vec4d, ndim=2),  # 每个batch每个时间步的累积旋转四元数
        time_id: int,
        num_fixed: int
):
    """根据笔杆旋转计算每个固定粒子的新位置
    
    每个固定粒子的新位置 = 笔杆中心位置 + 旋转后的局部坐标
    """
    batch_id, tid = wp.tid()
    if tid >= num_fixed:
        return
    
    # 获取当前时间步的笔杆中心和累积旋转
    center = brush_center[batch_id, time_id]
    quat = rotation_seq[batch_id, time_id]  # 累积旋转四元数 (w, x, y, z)
    
    # 获取该固定粒子的局部坐标
    local_coord = fixed_local_coords[tid]
    
    # 使用四元数旋转局部坐标
    rotated_local = quat_rotate_vector(quat, local_coord)
    
    # 计算新的世界坐标
    fixed_positions[batch_id, tid] = center + rotated_local


@wp.kernel
def update_brush_center_and_rotation(
        brush_center: wp.array(dtype=wp.vec3d, ndim=2),
        rotation_seq: wp.array(dtype=wp.vec4d, ndim=2),
        disp_seq: wp.array(dtype=wp.vec3d, ndim=2),  # 平移位移
        delta_rotation_seq: wp.array(dtype=wp.vec4d, ndim=2),  # 增量旋转四元数序列
        time_id: int
):
    """更新笔杆中心位置和累积旋转
    
    中心位置累加平移位移
    旋转通过四元数乘法累积
    """
    batch_id = wp.tid()
    
    # 更新中心位置
    if time_id == 0:
        brush_center[batch_id, time_id] = brush_center[batch_id, time_id] + disp_seq[batch_id, time_id]
    else:
        brush_center[batch_id, time_id] = brush_center[batch_id, time_id - 1] + disp_seq[batch_id, time_id]
    
    # 更新累积旋转: new_rotation = delta_rotation * prev_rotation
    delta_rot = delta_rotation_seq[batch_id, time_id]
    if time_id == 0:
        # 第一帧直接使用增量旋转
        rotation_seq[batch_id, time_id] = delta_rot
    else:
        prev_rot = rotation_seq[batch_id, time_id - 1]
        rotation_seq[batch_id, time_id] = quat_multiply(delta_rot, prev_rot)


@wp.kernel
def apply_prev_pos_to_now_pos(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        inv_mass: wp.array(dtype=wp.float64, ndim=2),
        disp_seq: wp.array(dtype=wp.vec3d, ndim=2),
        time_id: int,
        num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return
    d = disp_seq[batch_id, time_id]
    if inv_mass[batch_id, tid] > wp.float64(0.0):
        if time_id == 0:
            positions[batch_id, time_id, tid] = positions[batch_id, time_id, tid] + d / wp.float64(1e6)
        else:
            positions[batch_id, time_id, tid] = positions[batch_id, time_id - 1, tid] + d / wp.float64(1e6)
    else:
        if time_id == 0:
            positions[batch_id, time_id, tid] = positions[batch_id, time_id, tid]
        else:
            positions[batch_id, time_id, tid] = positions[batch_id, time_id - 1, tid] + d


@wp.kernel
def record_ink_collision(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_step: int,
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        ink_canvas: wp.array(dtype=wp.float64, ndim=2),
        canvas_resolution: int,
        num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] <= 0.0:
        return
    if time_step == 0:
        return
    pos = positions[batch_id, time_step, tid]
    prev_pos = positions[batch_id, time_step - 1, tid]

    if (prev_pos[2] > wp.float64(0.0) and pos[2] <= wp.float64(0.0)) or (pos[2] <= wp.float64(0.0)):
        x_canvas = wp.clamp(pos[0], wp.float64(0.0), wp.float64(1.0))
        y_canvas = wp.clamp(pos[1], wp.float64(0.0), wp.float64(1.0))

        pixel_x = int(x_canvas * wp.float64(canvas_resolution - 1))
        pixel_y = int(y_canvas * wp.float64(canvas_resolution - 1))

        pixel_x = wp.clamp(pixel_x, 0, canvas_resolution - 1)
        pixel_y = wp.clamp(pixel_y, 0, canvas_resolution - 1)

        pixel_idx = pixel_y * canvas_resolution + pixel_x
        wp.atomic_add(ink_canvas, batch_id, pixel_idx, wp.float64(1.0))


@wp.kernel
def record_ink_collision_soft(
    positions: wp.array(dtype=wp.vec3d, ndim=3),
    time_step: int,
    inv_masses: wp.array(dtype=wp.float64, ndim=2),
    ink_canvas: wp.array(dtype=wp.float64, ndim=2),
    canvas_resolution: int,
    num_particles: int,
    sigma_px: wp.float64,
    radius_px: int,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    pos = positions[batch_id, time_step, tid]

    if not pos[2] <= wp.float64(0.002) + wp.float64(PLANE_Z):
        return

    # xy coordinate mapping
    px = wp.clamp(pos[0], wp.float64(min_x), wp.float64(max_x))
    py = wp.clamp(pos[1], wp.float64(min_y), wp.float64(max_y))

    resm1 = wp.float64(canvas_resolution - 1)
    
    # Map x from [min_x, max_x] to [0, resm1] (Left -> Right)
    gx = (px - wp.float64(min_x)) / (wp.float64(max_x) - wp.float64(min_x)) * resm1
    
    # Map y from [min_y, max_y] to [resm1, 0] (Top -> Bottom, assuming image y-axis points down)
    # Note: original code did `gy = resm1 - ...`, consistent with y-axis pointing up in world space
    gy = (wp.float64(max_y) - py) / (wp.float64(max_y) - wp.float64(min_y)) * resm1

    abs_z = wp.abs(pos[2])
    dynamic_sigma = wp.float64(2.0) / (wp.exp(wp.float64(10.0) * abs_z))
    base_px = int(wp.floor(gx))
    base_py = int(wp.floor(gy))

    radius_px = radius_px * int(dynamic_sigma)

    px0 = base_px - radius_px
    py0 = base_py - radius_px
    if px0 < 0:
        px0 = 0
    if py0 < 0:
        py0 = 0
    px1 = base_px + radius_px
    py1 = base_py + radius_px
    max_idx = canvas_resolution - 1
    if px1 > max_idx:
        px1 = max_idx
    if py1 > max_idx:
        py1 = max_idx

    sigma_eff = sigma_px * dynamic_sigma

    inv_two_sigma2 = wp.float64(1.0) / (wp.float64(2.0) * sigma_eff * sigma_eff + wp.float64(1e-12))

    for yy in range(py0, py1 + 1):
        cy = wp.float64(yy) + wp.float64(0.5)
        dy = gy - cy
        for xx in range(px0, px1 + 1):
            cx = wp.float64(xx) + wp.float64(0.5)
            dx = gx - cx
            d2 = dx * dx + dy * dy
            w = wp.exp(-d2 * inv_two_sigma2)
            pixel_idx = yy * canvas_resolution + xx
            wp.atomic_add(ink_canvas, batch_id, pixel_idx, w)

@wp.kernel
def solve_distance_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_id: int,
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        constraint_indices: wp.array(dtype=wp.vec2i),
        rest_distances: wp.array(dtype=wp.float64),
        compliances: wp.array(dtype=wp.float64),
        lagrange_multipliers: wp.array(dtype=wp.float64),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_constraints: int
):
    batch_id, tid = wp.tid()
    if tid >= num_constraints:
        return

    idx = constraint_indices[tid]
    i, j = idx[0], idx[1]

    xi = positions[batch_id, time_id, i]
    xj = positions[batch_id, time_id, j]

    diff = xi - xj
    current_distance = wp.length(diff)

    epsilon = wp.float64(1e-15)
    if current_distance < wp.float64(epsilon):
        return

    violation = current_distance - rest_distances[tid]
    n = diff / (current_distance + epsilon)

    w_sum = inv_masses[batch_id, i] + inv_masses[batch_id, j]
    if w_sum < wp.float64(epsilon):
        return

    alpha = wp.max(compliances[tid] / (dt * dt), epsilon)
    denom = w_sum + alpha

    numerator = -(violation + alpha * lagrange_multipliers[tid])
    delta_lambda = numerator / denom

    max_correction = wp.float64(0.1) * rest_distances[tid]
    delta_lambda = wp.clamp(delta_lambda, -max_correction, max_correction)

    lagrange_multipliers[tid] = lagrange_multipliers[tid] + delta_lambda

    correction = delta_lambda * n
    correction_magnitude = wp.length(correction)

    if correction_magnitude > max_correction:
        correction = correction * (max_correction / correction_magnitude)

    if inv_masses[batch_id, i] > wp.float64(0.0):
        ci = inv_masses[batch_id, i] * correction
        wp.atomic_add(deltas_x, batch_id, i, ci[0])
        wp.atomic_add(deltas_y, batch_id, i, ci[1])
        wp.atomic_add(deltas_z, batch_id, i, ci[2])
    if inv_masses[batch_id, j] > wp.float64(0.0):
        cj = -inv_masses[batch_id, j] * correction
        wp.atomic_add(deltas_x, batch_id, j, cj[0])
        wp.atomic_add(deltas_y, batch_id, j, cj[1])
        wp.atomic_add(deltas_z, batch_id, j, cj[2])

@wp.kernel
def process_ink_canvas(
        ink_canvas: wp.array(dtype=wp.float64, ndim=2),
        canvas_resolution: int
):
    batch_id, tid = wp.tid()
    if tid >= canvas_resolution * canvas_resolution:
        return
    # sigmod
    v = ink_canvas[batch_id, tid]
    ink_canvas[batch_id, tid] = (wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-v)) - wp.float64(0.5)) * wp.float64(2)

@wp.kernel
def solve_variable_distance_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_id: int,
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        constraint_indices: wp.array(dtype=wp.vec2i),
        rest_distances: wp.array(dtype=wp.float64),
        base_compliances: wp.array(dtype=wp.float64),
        lagrange_multipliers: wp.array(dtype=wp.float64),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        z_threshold: wp.float64,
        weakness_factor: wp.float64,
        num_constraints: int
):
    """
    可变距离约束求解器 - 基于z坐标的动态约束强度
    z离0越近，约束越弱；离开一定距离后，约束强度固定为普通距离约束强度
    """
    batch_id, tid = wp.tid()
    if tid >= num_constraints:
        return

    idx = constraint_indices[tid]
    i, j = idx[0], idx[1]

    # Get current positions
    xi = positions[batch_id, time_id, i]
    xj = positions[batch_id, time_id, j]

    # Calculate constraint violation
    diff = xi - xj
    current_distance = wp.length(diff)

    epsilon = wp.float64(1e-15)
    # Numerical stability check
    if current_distance < wp.float64(epsilon):
        return

    violation = current_distance - rest_distances[tid]
    n = diff / current_distance


    # Calculate denominator for constraint resolution
    w_sum = inv_masses[batch_id, i] + inv_masses[batch_id, j]
    if w_sum < wp.float64(epsilon):
        return

    # 计算基于z坐标的动态compliance
    # 取两个点的平均z坐标作为参考
    avg_z = (xi[2] + xj[2]) * wp.float64(0.5)
    abs_z = wp.abs(avg_z)

    # 动态调整compliance：z离0越近，compliance越大（约束越弱）
    dynamic_compliance = base_compliances[tid]
    if abs_z < z_threshold:
        # 在阈值范围内，线性插值增加compliance
        z_ratio = wp.float64(1.0) - (abs_z / z_threshold)  # z=0时为1, z=threshold时为0
        dynamic_compliance = base_compliances[tid] * (wp.float64(1.0) + weakness_factor * z_ratio)
        # dynamic_compliance = z_ratio * 1e-1 + base_compliances[tid]

    # XPBD compliance model with numerical stability
    alpha = wp.max(dynamic_compliance / (dt * dt), epsilon)
    denom = w_sum + alpha

    # Calculate delta lambda with clamping
    numerator = -(violation + alpha * lagrange_multipliers[tid])
    delta_lambda = numerator / denom

    # Clamp delta lambda to prevent explosive corrections
    max_correction = wp.float64(0.1) * rest_distances[tid]
    delta_lambda = wp.clamp(delta_lambda, -max_correction, max_correction)

    # Update lagrange multiplier
    lagrange_multipliers[tid] = lagrange_multipliers[tid] + delta_lambda

    # Apply position corrections with stability check
    correction = delta_lambda * n
    correction_magnitude = wp.length(correction)

    # Limit correction magnitude
    if correction_magnitude > max_correction:
        correction = correction * (max_correction / correction_magnitude)

    if inv_masses[batch_id, i] > wp.float64(0.0):
        ci = inv_masses[batch_id, i] * correction
        wp.atomic_add(deltas_x, batch_id, i, ci[0])
        wp.atomic_add(deltas_y, batch_id, i, ci[1])
        wp.atomic_add(deltas_z, batch_id, i, ci[2])
    if inv_masses[batch_id, j] > wp.float64(0.0):
        cj = -inv_masses[batch_id, j] * correction
        wp.atomic_add(deltas_x, batch_id, j, cj[0])
        wp.atomic_add(deltas_y, batch_id, j, cj[1])
        wp.atomic_add(deltas_z, batch_id, j, cj[2])


@wp.kernel
def solve_bending_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_id: int,
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        constraint_indices: wp.array(dtype=wp.vec3i),  # (i, j, k) 三个粒子索引
        rest_distances: wp.array(dtype=wp.float64),  # i到k的直线距离
        compliances: wp.array(dtype=wp.float64),
        lagrange_multipliers: wp.array(dtype=wp.float64),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_constraints: int
):
    """
    弯曲距离约束求解器 - 约束首尾两点的距离以保持直线形态
    
    对于三个连续粒子 i, j, k，约束 |i-k| = rest_distance
    其中 rest_distance = |i-j| + |j-k|（直线时的距离）
    
    这是一个简单的距离约束，比角度约束更加数值稳定
    """
    batch_id, tid = wp.tid()
    if tid >= num_constraints:
        return

    idx = constraint_indices[tid]
    i = idx[0]
    j = idx[1]  # 中间点（不直接参与此约束，但用于记录）
    k = idx[2]

    # 获取首尾两点的位置
    xi = positions[batch_id, time_id, i]
    xk = positions[batch_id, time_id, k]

    # 计算当前距离
    diff = xi - xk
    current_distance = wp.length(diff)

    epsilon = wp.float64(1e-10)
    if current_distance < epsilon:
        return

    # 约束违背量
    rest_dist = rest_distances[tid]
    violation = current_distance - rest_dist

    # 如果违背量很小，跳过
    if wp.abs(violation) < epsilon:
        return

    # 梯度方向（归一化的连接向量）
    n = diff / current_distance

    # 计算有效质量
    w_i = inv_masses[batch_id, i]
    w_k = inv_masses[batch_id, k]
    w_sum = w_i + w_k

    if w_sum < epsilon:
        return

    # XPBD compliance
    alpha = compliances[tid] / (dt * dt)
    if alpha < epsilon:
        alpha = epsilon
    denom = w_sum + alpha

    # 计算拉格朗日乘子增量
    prev_lambda = lagrange_multipliers[tid]
    numerator = -(violation + alpha * prev_lambda)
    delta_lambda = numerator / denom

    # 限制最大修正量
    max_correction = wp.float64(0.1) * rest_dist
    delta_lambda = wp.clamp(delta_lambda, -max_correction, max_correction)

    # 更新拉格朗日乘子
    lagrange_multipliers[tid] = prev_lambda + delta_lambda

    # 应用位置修正
    correction = delta_lambda * n

    if w_i > wp.float64(0.0):
        ci = w_i * correction
        wp.atomic_add(deltas_x, batch_id, i, ci[0])
        wp.atomic_add(deltas_y, batch_id, i, ci[1])
        wp.atomic_add(deltas_z, batch_id, i, ci[2])

    if w_k > wp.float64(0.0):
        ck = -w_k * correction
        wp.atomic_add(deltas_x, batch_id, k, ck[0])
        wp.atomic_add(deltas_y, batch_id, k, ck[1])
        wp.atomic_add(deltas_z, batch_id, k, ck[2])


@wp.kernel
def apply_and_clear_deltas(
    positions: wp.array(dtype=wp.vec3d, ndim=3),
    time_id: int,
    deltas_x: wp.array(dtype=wp.float64, ndim=2),
    deltas_y: wp.array(dtype=wp.float64, ndim=2),
    deltas_z: wp.array(dtype=wp.float64, ndim=2),
    num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return
    dx = deltas_x[batch_id, tid]
    dy = deltas_y[batch_id, tid]
    dz = deltas_z[batch_id, tid]
    if dx != wp.float64(0.0) or dy != wp.float64(0.0) or dz != wp.float64(0.0):
        positions[batch_id, time_id, tid] = positions[batch_id, time_id, tid] + wp.vec3d(dx, dy, dz)
    deltas_x[batch_id, tid] = wp.float64(0.0)
    deltas_y[batch_id, tid] = wp.float64(0.0)
    deltas_z[batch_id, tid] = wp.float64(0.0)


@wp.kernel
def solve_plane_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_id: int,
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        plane_normals: wp.array(dtype=wp.vec3d),
        plane_distances: wp.array(dtype=wp.float64),
        compliances: wp.array(dtype=wp.float64),
        lagrange_multipliers: wp.array(dtype=wp.float64, ndim=2),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_particles: int,
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] <= wp.float64(0.0):  # Fixed particles
        return

    pos = positions[batch_id, time_id, tid]

    # Check against all planes
    plane_idx = 0
    normal = plane_normals[plane_idx]
    distance = plane_distances[plane_idx]

    # Calculate signed distance to plane
    signed_distance = wp.dot(pos, normal) - distance

    # Only apply constraint if particle is on the negative side (violating constraint)
    if signed_distance < wp.float64(0.0):
        # Constraint violation (negative distance means penetration)
        violation = signed_distance

        # Gradient is just the plane normal
        grad = normal

        # Calculate denominator for constraint resolution
        w = inv_masses[batch_id, tid] * wp.dot(grad, grad)
        if w < wp.float64(1e-5):
            return

        # XPBD compliance model
        alpha = wp.max(compliances[plane_idx] / (dt * dt), wp.float64(1e-15))
        denom = w + alpha

        # Calculate delta lambda
        lambda_idx = tid + plane_idx
        numerator = -(violation + alpha * lagrange_multipliers[batch_id, lambda_idx])
        delta_lambda = numerator / denom

        # Clamp delta lambda to prevent explosive corrections
        max_correction = wp.float64(0.1)
        delta_lambda = wp.clamp(delta_lambda, -max_correction, max_correction)

        # Update lagrange multiplier
        lagrange_multipliers[batch_id, lambda_idx] = lagrange_multipliers[batch_id, lambda_idx] + delta_lambda

        # Apply position correction
        correction = inv_masses[batch_id, tid] * delta_lambda * grad

        # Limit correction magnitude
        correction_magnitude = wp.length(correction)
        if correction_magnitude > max_correction:
            correction = correction * (max_correction / correction_magnitude)

        # Accumulate corrections atomically
        wp.atomic_add(deltas_x, batch_id, tid, correction[0])
        wp.atomic_add(deltas_y, batch_id, tid, correction[1])
        wp.atomic_add(deltas_z, batch_id, tid, correction[2])



@wp.kernel
def apply_fixed_constraints_gradient_preserving(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_step: int,
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_indices: wp.array(dtype=int),
        num_fixed: int,
        stiffness: wp.float64  # 软约束强度 (0,1]，越接近1越“硬”
):
    batch_id, tid = wp.tid()
    if tid >= num_fixed:
        return

    idx = fixed_indices[tid]
    # 计算当前位置与目标位置的差异
    current_pos = positions[batch_id, time_step, idx]
    target_pos = fixed_positions[batch_id, tid]
    diff = target_pos - current_pos

    # 软约束：只移动 diff 的 stiffness 比例
    new_pos = current_pos + diff * stiffness
    positions[batch_id, time_step, idx] = new_pos

    # 速度软阻尼：保留一部分原速度，防止数值震荡，同时保持梯度流动
    v = velocities[batch_id, idx]
    one = wp.float64(1.0)
    clamped_stiff = wp.clamp(stiffness, wp.float64(0.0), one)
    # 速度趋向 0，但不是直接清零：v_new = (1 - clamped_stiff)*v
    velocities[batch_id, idx] = v * (one - clamped_stiff)


@wp.kernel
def update_velocities(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        time_step: int,
        disp_seq: wp.array(dtype=wp.vec3d, ndim=2),
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles or time_step == 0:
        return

    if time_step == 0 and inv_masses[batch_id, tid] > wp.float64(0.0):
        velocities[batch_id, tid] = disp_seq[batch_id, time_step] / dt
        return

    if inv_masses[batch_id, tid] > wp.float64(0.0):
        velocities[batch_id, tid] = (positions[batch_id, time_step, tid] - positions[batch_id, time_step - 1, tid]) / dt

@wp.kernel
def apply_velocity_damping_and_clamp(
    velocities: wp.array(dtype=wp.vec3d, ndim=2),
    damping: wp.float64,
    max_velocity: wp.float64,
    num_particles: int
):
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    v = velocities[batch_id, tid]
    one = wp.float64(1.0)
    zero = wp.float64(0.0)
    damping_factor = wp.clamp(one - damping, zero, one)
    v = v * damping_factor

    mag = wp.length(v)
    if mag > max_velocity:
        v = v * (max_velocity / mag)
    velocities[batch_id, tid] = v


class XPBDSimulator:
    def __init__(self,
                 dt: float = 1.0 / 60.0,
                 substeps: int = 5,
                 iterations: int = 15,
                 num_steps: int = 5,
                 batch_size: int = 1,
                 gravity: torch.Tensor = torch.tensor([0.0, 0.0, -9.81], dtype=torch.float64),
                 dis_compliance: float = 1e-8,  # 降低compliance使毛发内部距离约束更硬
                 variable_dis_compliance: float = 0.01,
                 bending_compliance: float = 1e-6,  # 弯曲约束compliance，值越小毛发越硬
                 damping: float = 0.3,  # 增加阻尼以提高稳定性
                 restitution: float = 0.5,
                 friction: float = 0.3,
                 device: Optional[str] = None,
                 canvas_resolution: int = 512,
                 variable_z_threshold: float = 0.05,  # z坐标阈值，超过此值约束强度不再变化
                 variable_weakness_factor: float = 10.0,  # 接近z=0时的弱化倍数
                 splat_sigma_px: float = 1.0,  # 高斯核标准差（以像素为单位）
                 splat_radius_px: int = 1,  # 溅射半径（像素，2->5x5 邻域）
                 fixed_stiffness: float = 0.6,  # 软固定约束的收敛强度 (0,1]，越大越“硬”
                 canvas_min_x: float = 0.0,
                 canvas_max_x: float = 1.0,
                 canvas_min_y: float = 0.0,
                 canvas_max_y: float = 1.0
                 ):

        # Timing and physical parameters
        self.dt = dt
        self.substeps = substeps
        self.iterations = iterations
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.prev_step = 0

        self.sub_dt = dt / substeps
        self.gravity = gravity
        self.dis_compliance = dis_compliance
        self.variable_dis_compliance = variable_dis_compliance
        self.bending_compliance = bending_compliance
        self.damping = damping
        self.restitution = restitution
        self.friction = friction

        self.time_id = 0  # 当前时间步计数器

        # 可变距离约束的特殊参数
        self.variable_z_threshold = variable_z_threshold
        self.variable_weakness_factor = variable_weakness_factor
        self.splat_sigma_px = splat_sigma_px
        self.splat_radius_px = splat_radius_px
        self.fixed_stiffness = fixed_stiffness

        self.min_x = canvas_min_x
        self.max_x = canvas_max_x
        self.min_y = canvas_min_y
        self.max_y = canvas_max_y

        # Device selection (prefer CUDA if available)
        self.device = self._select_device(device)

        # Simulation state
        self.num_particles = 0
        self.particles_initialized = False

        # Ink rendering parameters
        self.canvas_resolution = canvas_resolution
        self.ink_canvas = None

        # Device arrays (Warp)
        self.positions = None
        self.prev_positions = None
        self.velocities = None
        self.inv_masses = None
        self.external_forces = None

        self.sim_image = None
        self.ref_image = None
        self.loss = None

        self.displacement = None  # 用户输入的位移场

        # Per-particle accumulated corrections (to avoid concurrent writes)
        self._delta_x = None
        self._delta_y = None
        self._delta_z = None

        # Constraint arrays (host-side collections for potential future use)
        self.distance_constraints = []
        self.bending_constraints = []
        self.fixed_constraints = []

        # Warp arrays for constraints
        self.distance_indices = None
        self.distance_rest_lengths = None
        self.distance_compliances = None
        self.distance_lambdas = None

        # 可变距离约束数组
        self.variable_distance_indices = None
        self.variable_distance_rest_lengths = None
        self.variable_distance_compliances = None
        self.variable_distance_lambdas = None

        # 弯曲约束数组
        self.bending_indices = None
        self.bending_rest_distances = None
        self.bending_compliances = None
        self.bending_lambdas = None

        self.fixed_indices = None
        self.fixed_positions = None
        
        # 笔杆旋转相关数组
        self.fixed_local_coords = None  # 固定粒子相对于笔杆中心的局部坐标
        self.brush_center = None  # 笔杆中心位置 (batch_size, num_steps, 3)
        self.rotation_seq = None  # 累积旋转四元数序列 (batch_size, num_steps, 4)
        self.delta_rotation_seq = None  # 增量旋转四元数序列 (batch_size, num_steps, 4)
        self.use_rotation = False  # 是否使用旋转模式
        self.initial_brush_center = None  # 初始笔杆中心位置

        # Plane constraint arrays (for z=0 plane)
        self.plane_normals = None
        self.plane_distances = None
        self.plane_compliances = None
        self.plane_lambdas = None
        self.num_planes = 0

        # Cached counts to avoid repeated host-device sync
        self.num_distance_constraints = 0
        self.num_variable_distance_constraints = 0
        self.num_bending_constraints = 0
        self.num_fixed_constraints = 0
        self._fixed_indices_host = None

    def set_loss(self, ref_image: np.ndarray):
        ref_tensor = torch.tensor(ref_image, dtype=torch.float64).to(self.device)
        if torch.max(ref_tensor) > 1.0:
            ref_tensor = ref_tensor / 255.0
        if ref_tensor.dim() == 2:
            ref_tensor = ref_tensor.flatten()
        self.ref_image = wp.from_torch(ref_tensor, dtype=wp.float64)
        self.loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)

    def _select_device(self, preferred: Optional[str]) -> str:
        if preferred:
            return preferred
        # Try CUDA, else fallback to CPU
        try:
            wp.get_device("cuda")
            return "cuda"
        except Exception:
            return "cpu"

    def load_displacements(self, displacements: torch.Tensor, rotations: torch.Tensor = None):
        """加载位移序列和可选的旋转序列
        
        Args:
            displacements: 平移位移序列 (num_steps, 3) 或 (batch_size, num_steps, 3)
            rotations: 增量旋转四元数序列 (num_steps, 4) 或 (batch_size, num_steps, 4)
                      四元数格式为 (w, x, y, z)，表示每一步相对于上一步的增量旋转
                      如果为 None，则使用原来的纯平移模式
        """
        if displacements.dim() == 2:
            displacements = displacements.unsqueeze(0).repeat(self.batch_size, 1, 1)
        displacements = displacements.to(self.device)
        self.displacement = wp.from_torch(displacements, dtype=wp.vec3d, requires_grad=True)
        
        if rotations is not None:
            self.use_rotation = True
            if rotations.dim() == 2:
                rotations = rotations.unsqueeze(0).repeat(self.batch_size, 1, 1)
            rotations = rotations.to(self.device)
            self.delta_rotation_seq = wp.from_torch(rotations, dtype=wp.vec4d, requires_grad=True)
            
            # 初始化累积旋转序列为单位四元数
            identity_quats = torch.zeros((self.batch_size, self.num_steps, 4), dtype=torch.float64, device=self.device)
            identity_quats[:, :, 0] = 1.0  # w = 1, x = y = z = 0 表示单位四元数
            self.rotation_seq = wp.from_torch(identity_quats, dtype=wp.vec4d, requires_grad=True)
            
            # 初始化笔杆中心位置序列
            if self.initial_brush_center is not None:
                brush_centers = self.initial_brush_center.unsqueeze(0).unsqueeze(0).repeat(
                    self.batch_size, self.num_steps, 1
                ).to(self.device)
                self.brush_center = wp.from_torch(brush_centers, dtype=wp.vec3d, requires_grad=True)
        else:
            self.use_rotation = False

    def load_brush(self, brush: Brush):
        """Load particles and constraints from brush"""
        # Generate brush structure if not done
        if not brush.hairs:
            brush.gen_hairs()
            brush.gen_constraints()

        self.num_particles = len(brush.particles)

        self.time_id = 0
        
        # 存储初始笔杆中心位置（用于旋转计算）
        self.initial_brush_center = brush.root_position.clone().to(torch.float64)

        # Extract particle data using torch tensors
        positions = torch.zeros((self.batch_size, self.num_steps, self.num_particles, 3), dtype=torch.float64).to(self.device)
        velocities = torch.zeros((self.batch_size, self.num_particles, 3), dtype=torch.float64).to(self.device)
        inv_masses = torch.ones((self.batch_size, self.num_particles), dtype=torch.float64).to(self.device)

        for b in range(self.batch_size):
            for i, particle in enumerate(brush.particles):
                positions[b, 0, i] = torch.tensor(particle.position, dtype=torch.float64)
                velocities[b, i] = torch.tensor(particle.velocity, dtype=torch.float64)
                inv_masses[b, i] = 1.0

        self.positions = wp.from_torch(positions, dtype=wp.vec3d, requires_grad=True)
        self.velocities = wp.from_torch(velocities, dtype=wp.vec3d)
        self.inv_masses = wp.from_torch(inv_masses, dtype=wp.float64)

        # External forces (gravity) using torch
        forces = self.gravity.unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_particles, 1).to(torch.float64).to(self.device)
        self.external_forces = wp.from_torch(forces, dtype=wp.vec3d)

        # Allocate zeroed delta buffers for atomic accumulation
        self._delta_x = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)
        self._delta_y = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)
        self._delta_z = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)

        # Initialize ink canvas (1x1 canvas mapped to canvas_resolution x canvas_resolution pixels)
        canvas_size = self.canvas_resolution * self.canvas_resolution
        # Enable gradient tracking on the ink canvas for autodiff
        self.ink_canvas = wp.zeros((self.batch_size, canvas_size), dtype=wp.float64, device=self.device, requires_grad=True)

        # Extract constraints (会设置 fixed_local_coords)
        self._extract_constraints(brush.constraints, brush.root_position)

        # Setup plane constraints (z=0 plane)
        self._setup_plane_constraints()

        self.particles_initialized = True

    def _extract_constraints(self, constraints, brush_root_position: torch.Tensor = None):
        """Extract and categorize constraints
        
        Args:
            constraints: 约束列表
            brush_root_position: 笔杆中心位置，用于计算固定粒子的局部坐标
        """
        distance_data = []
        variable_distance_data = []
        bending_data = []
        fixed_data = []

        for constraint in constraints:
            if isinstance(constraint, DistanceConstraint):
                i = constraint.point_a.particle_id
                j = constraint.point_b.particle_id
                rest_dist = constraint.rest_distance
                compliance = self.dis_compliance

                distance_data.append((i, j, rest_dist, compliance))

            elif isinstance(constraint, VariableDistanceConstraint):
                # 单独处理可变距离约束
                i = constraint.point_a.particle_id
                j = constraint.point_b.particle_id
                rest_dist = constraint.rest_distance
                compliance = self.variable_dis_compliance

                variable_distance_data.append((i, j, rest_dist, compliance))

            elif isinstance(constraint, FixedPointConstraint):
                i = constraint.point.particle_id
                pos = constraint.point.position
                fixed_data.append((i, pos))

            elif isinstance(constraint, BendingConstraint):
                # 处理弯曲距离约束 - 约束首尾两点的距离
                i = constraint.point_a.particle_id
                j = constraint.point_b.particle_id  # 中间点（记录但不直接参与约束）
                k = constraint.point_c.particle_id
                rest_dist = constraint.rest_distance
                compliance = self.bending_compliance
                bending_data.append((i, j, k, rest_dist, compliance))

        # Create distance constraint arrays using torch
        if distance_data:
            num_dist = len(distance_data)
            dist_indices = torch.tensor([(d[0], d[1]) for d in distance_data], dtype=torch.int32).to(self.device)
            dist_lengths = torch.tensor([d[2] for d in distance_data], dtype=torch.float64).to(self.device)
            # Clamp compliance values to prevent numerical issues
            dist_compliances = torch.tensor([d[3] for d in distance_data], dtype=torch.float64).to(self.device)
            dist_lambdas = torch.zeros(num_dist, dtype=torch.float64).to(self.device)

            self.distance_indices = wp.from_torch(dist_indices, dtype=wp.vec2i)
            self.distance_rest_lengths = wp.from_torch(dist_lengths, dtype=wp.float64)
            self.distance_compliances = wp.from_torch(dist_compliances, dtype=wp.float64)
            self.distance_lambdas = wp.from_torch(dist_lambdas, dtype=wp.float64)
            self.num_distance_constraints = num_dist

        # Create variable distance constraint arrays using torch
        if variable_distance_data:
            num_var_dist = len(variable_distance_data)
            var_dist_indices = torch.tensor([(d[0], d[1]) for d in variable_distance_data], dtype=torch.int32).to(
                self.device)
            var_dist_lengths = torch.tensor([d[2] for d in variable_distance_data], dtype=torch.float64).to(self.device)
            var_dist_compliances = torch.tensor([d[3] for d in variable_distance_data], dtype=torch.float64).to(
                self.device)
            var_dist_lambdas = torch.zeros(num_var_dist, dtype=torch.float64).to(self.device)

            self.variable_distance_indices = wp.from_torch(var_dist_indices, dtype=wp.vec2i)
            self.variable_distance_rest_lengths = wp.from_torch(var_dist_lengths, dtype=wp.float64)
            self.variable_distance_compliances = wp.from_torch(var_dist_compliances, dtype=wp.float64)
            self.variable_distance_lambdas = wp.from_torch(var_dist_lambdas, dtype=wp.float64)
            self.num_variable_distance_constraints = num_var_dist

        # Create bending constraint arrays using torch
        if bending_data:
            num_bending = len(bending_data)
            # 使用 vec3i 存储三个粒子的索引 (i, j, k)，其中 j 是中间点
            bending_indices = torch.tensor([(d[0], d[1], d[2]) for d in bending_data], dtype=torch.int32).to(self.device)
            bending_rest_distances = torch.tensor([d[3] for d in bending_data], dtype=torch.float64).to(self.device)
            bending_compliances = torch.tensor([d[4] for d in bending_data], dtype=torch.float64).to(self.device)
            bending_lambdas = torch.zeros(num_bending, dtype=torch.float64).to(self.device)

            self.bending_indices = wp.from_torch(bending_indices, dtype=wp.vec3i)
            self.bending_rest_distances = wp.from_torch(bending_rest_distances, dtype=wp.float64)
            self.bending_compliances = wp.from_torch(bending_compliances, dtype=wp.float64)
            self.bending_lambdas = wp.from_torch(bending_lambdas, dtype=wp.float64)
            self.num_bending_constraints = num_bending

        # Create fixed constraint arrays using torch
        if fixed_data:
            num_fixed = len(fixed_data)
            fixed_indices = torch.tensor([f[0] for f in fixed_data], dtype=torch.int32).to(self.device)
            # Stack the tensors instead of using torch.tensor()
            fixed_positions = torch.stack([f[1] for f in fixed_data]).to(torch.float64).to(self.device)
            fixed_positions = fixed_positions.unsqueeze(0).repeat(self.batch_size, 1, 1)

            self.fixed_indices = wp.from_torch(fixed_indices, dtype=wp.int32)
            self.fixed_positions = wp.from_torch(fixed_positions, dtype=wp.vec3d)
            self.num_fixed_constraints = num_fixed
            # Cache a host copy to avoid repeated device->host syncs
            self._fixed_indices_host = fixed_indices.cpu().numpy()
            
            # 计算并存储固定粒子相对于笔杆中心的局部坐标（用于旋转）
            if brush_root_position is not None:
                brush_center = brush_root_position.to(torch.float64).to(self.device)
                # 每个固定粒子的局部坐标 = 粒子位置 - 笔杆中心
                fixed_pos_stack = torch.stack([f[1] for f in fixed_data]).to(torch.float64).to(self.device)
                local_coords = fixed_pos_stack - brush_center
                self.fixed_local_coords = wp.from_torch(local_coords, dtype=wp.vec3d)

            # Set inverse mass to zero for fixed particles
            # Convert warp array to torch tensor, modify, then convert back
            inv_masses_tensor = wp.to_torch(self.inv_masses)
            for b in range(self.batch_size):
                for idx in fixed_indices:
                    inv_masses_tensor[b, idx] = 0.0
            self.inv_masses = wp.from_torch(inv_masses_tensor.to(self.device), dtype=wp.float64)

    def _setup_plane_constraints(self):
        """Setup plane constraints for z=0 plane penetration prevention"""
        # Define z=0 plane using torch
        plane_normals = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64).to(self.device)  # Normal pointing up
        plane_distances = torch.tensor([PLANE_Z], dtype=torch.float64).to(self.device)  # Distance from origin
        plane_compliances = torch.tensor([1e-9], dtype=torch.float64).to(
            self.device)  # Zero compliance for hard constraint

        self.num_planes = 1

        # Create device arrays
        self.plane_normals = wp.from_torch(plane_normals, dtype=wp.vec3d)
        self.plane_distances = wp.from_torch(plane_distances, dtype=wp.float64)
        self.plane_compliances = wp.from_torch(plane_compliances, dtype=wp.float64)

        # Create lagrange multiplier array (one per particle per plane)
        plane_lambdas = torch.zeros((self.batch_size, self.num_particles * self.num_planes), dtype=torch.float64).to(self.device)
        self.plane_lambdas = wp.from_torch(plane_lambdas, dtype=wp.float64)

    def compute_wp_loss(self, batch_id: int = 0):
        self.loss = wp.zeros(1, dtype=wp.float64, device=self.device, requires_grad=True)
        wp.launch(
            compute_l1_loss,
            dim=self.canvas_resolution * self.canvas_resolution,
            inputs=[
                self.ink_canvas,
                self.ref_image,
                self.canvas_resolution,
                self.canvas_resolution,
                batch_id,
            ],
            outputs=[self.loss],
            device=self.device
        )

    def step(self, start: int, end: int):
        for time_step in range(start, end):
                wp.launch(
                    apply_prev_pos_to_now_pos,
                    dim=(self.batch_size, self.num_particles),
                    inputs=[
                        self.positions,
                        self.inv_masses,
                        self.displacement,
                        time_step,
                        self.num_particles
                    ],
                    device=self.device
                )
                
                if self.use_rotation and self.brush_center is not None:
                    # 使用旋转模式：先更新笔杆中心和累积旋转，然后应用到固定粒子
                    wp.launch(
                        update_brush_center_and_rotation,
                        dim=self.batch_size,
                        inputs=[
                            self.brush_center,
                            self.rotation_seq,
                            self.displacement,
                            self.delta_rotation_seq,
                            time_step
                        ],
                        device=self.device
                    )
                    # 根据旋转计算每个固定粒子的新位置
                    wp.launch(
                        apply_rotation_displacement_to_fixed_positions,
                        dim=(self.batch_size, self.num_fixed_constraints),
                        inputs=[
                            self.fixed_positions,
                            self.fixed_local_coords,
                            self.brush_center,
                            self.rotation_seq,
                            time_step,
                            self.num_fixed_constraints
                        ],
                        device=self.device
                    )
                else:
                    # 原有的纯平移模式
                    wp.launch(
                        apply_step_displacement_to_fixed_positions,
                        dim=(self.batch_size, self.num_fixed_constraints),
                        inputs=[
                            self.fixed_positions,
                            self.displacement,
                            time_step,
                            self.num_fixed_constraints
                        ],
                        device=self.device
                    )
                self._substep(time_step)

    def _substep(self, time_step):
        wp.launch(
            integrate_particles,
            dim=(self.batch_size,  self.num_particles),
            inputs=[
                self.positions,
                self.velocities,
                self.inv_masses,
                self.external_forces,
                self.sub_dt,
                self.num_particles,
                time_step
            ],
            device=self.device
        )

        for _ in range(self.iterations):
            if self.distance_indices is not None and self.num_distance_constraints > 0:
                wp.launch(
                    solve_distance_constraints,
                    dim=(self.batch_size, self.num_distance_constraints),
                    inputs=[
                        self.positions,
                        time_step,
                        self.inv_masses,
                        self.distance_indices,
                        self.distance_rest_lengths,
                        self.distance_compliances,
                        self.distance_lambdas,
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.sub_dt,
                        self.num_distance_constraints
                    ],
                    device=self.device
                )

                # Solve variable distance constraints
            if self.variable_distance_indices is not None and self.num_variable_distance_constraints > 0:
                wp.launch(
                    solve_variable_distance_constraints,
                    dim=(self.batch_size, self.num_variable_distance_constraints),
                    inputs=[
                        self.positions,
                        time_step,
                        self.inv_masses,
                        self.variable_distance_indices,
                        self.variable_distance_rest_lengths,
                        self.variable_distance_compliances,
                        self.variable_distance_lambdas,
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.sub_dt,
                        self.variable_z_threshold,
                        self.variable_weakness_factor,
                        self.num_variable_distance_constraints
                    ],
                    device=self.device
                )

            # Solve bending constraints for hair stiffness
            # if self.bending_indices is not None and self.num_bending_constraints > 0:
            #     wp.launch(
            #         solve_bending_constraints,
            #         dim=(self.batch_size, self.num_bending_constraints),
            #         inputs=[
            #             self.positions,
            #             time_step,
            #             self.inv_masses,
            #             self.bending_indices,
            #             self.bending_rest_distances,
            #             self.bending_compliances,
            #             self.bending_lambdas,
            #             self._delta_x,
            #             self._delta_y,
            #             self._delta_z,
            #             self.sub_dt,
            #             self.num_bending_constraints
            #         ],
            #         device=self.device
            #     )

                # Apply accumulated deltas and clear for next iteration
            wp.launch(
                apply_and_clear_deltas,
                dim=(self.batch_size,self.num_particles),
                inputs=[
                    self.positions,
                    time_step,
                    self._delta_x,
                    self._delta_y,
                    self._delta_z,
                    self.num_particles
                ],
                device=self.device
            )

            # Solve plane constraints (z=0 plane penetration prevention)
            if self.plane_normals is not None and self.num_planes > 0:
                pass
                wp.launch(
                    solve_plane_constraints,
                    dim=(self.batch_size, self.num_particles),
                    inputs=[
                        self.positions,
                        time_step,
                        self.inv_masses,
                        self.plane_normals,
                        self.plane_distances,
                        self.plane_compliances,
                        self.plane_lambdas,
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.sub_dt,
                        self.num_particles,
                    ],
                    device=self.device
                )

                # Apply plane constraint corrections
                wp.launch(
                    apply_and_clear_deltas,
                    dim=(self.batch_size, self.num_particles),
                    inputs=[
                        self.positions,
                        time_step,
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.num_particles
                    ],
                    device=self.device
                )

            # Apply fixed constraints
            if self.fixed_indices is not None and self.num_fixed_constraints > 0:
                wp.launch(
                    apply_fixed_constraints_gradient_preserving,
                    dim=(self.batch_size, self.num_fixed_constraints),
                    inputs=[
                        self.positions,
                        time_step,
                        self.velocities,
                        self.fixed_positions,
                        self.fixed_indices,
                        self.num_fixed_constraints,
                        wp.float64(self.fixed_stiffness)
                    ],
                    device=self.device
                )

        wp.launch(
            record_ink_collision_soft,
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.positions,
                time_step,
                self.inv_masses,
                self.ink_canvas,
                self.canvas_resolution,
                self.num_particles,
                wp.float64(self.splat_sigma_px),
                int(self.splat_radius_px),
                self.min_x,
                self.max_x,
                self.min_y,
                self.max_y
            ],
            device=self.device
        )

        wp.launch(
            update_velocities,
            dim=(self.batch_size,self.num_particles),
            inputs=[
                self.positions,
                time_step,
                self.displacement,
                self.velocities,
                self.inv_masses,
                self.sub_dt,
                self.num_particles
            ],
            device=self.device
        )

        # Apply damping & clamp in Warp to keep autodiff graph intact
        wp.launch(
            apply_velocity_damping_and_clamp,
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.velocities,
                wp.float64(self.damping),
                wp.float64(10.0),  # max velocity magnitude
                self.num_particles
            ],
            device=self.device
        )

    def get_ink_canvas_batch(self, batch_id: int = None):
        if batch_id is None:
            return wp.to_torch(self.ink_canvas)
        else:
            canvas_torch = wp.to_torch(self.ink_canvas)
            return canvas_torch[batch_id]
    
    def get_positions_batch(self, batch_id: int = None):
        if batch_id is None:
            return wp.to_torch(self.positions)
        else:
            pos_torch = wp.to_torch(self.positions)
            return pos_torch[batch_id]


def axis_angle_to_quaternion(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """将轴角表示转换为四元数
    
    Args:
        axis: 旋转轴，形状为 (..., 3)，会被自动归一化
        angle: 旋转角度（弧度），形状为 (...) 或标量
        
    Returns:
        四元数，形状为 (..., 4)，格式为 (w, x, y, z)
    """
    # 归一化轴
    axis = axis / (torch.norm(axis, dim=-1, keepdim=True) + 1e-12)
    
    # 处理 angle 维度
    if angle.dim() == 0:
        angle = angle.unsqueeze(0)
    if angle.dim() < axis.dim():
        angle = angle.unsqueeze(-1)
    
    half_angle = angle / 2.0
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    
    # 四元数格式: (w, x, y, z)
    w = cos_half.squeeze(-1)
    xyz = axis * sin_half
    
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)


def euler_to_quaternion(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """将欧拉角转换为四元数 (ZYX 顺序)
    
    Args:
        roll: 绕 X 轴旋转角度（弧度）
        pitch: 绕 Y 轴旋转角度（弧度）
        yaw: 绕 Z 轴旋转角度（弧度）
        
    Returns:
        四元数，形状为 (..., 4)，格式为 (w, x, y, z)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return torch.stack([w, x, y, z], dim=-1)


def identity_quaternion(batch_shape: tuple = (), dtype=torch.float64, device=None) -> torch.Tensor:
    """创建单位四元数（无旋转）
    
    Args:
        batch_shape: 批次形状，如 (batch_size, num_steps)
        dtype: 数据类型
        device: 设备
        
    Returns:
        单位四元数，形状为 (*batch_shape, 4)，格式为 (w, x, y, z) = (1, 0, 0, 0)
    """
    shape = batch_shape + (4,)
    quat = torch.zeros(shape, dtype=dtype, device=device)
    quat[..., 0] = 1.0  # w = 1
    return quat


def create_rotation_sequence(
    angles: torch.Tensor,
    axis: torch.Tensor = None,
    rotation_type: str = "axis_angle"
) -> torch.Tensor:
    """创建旋转四元数序列，用于毛笔笔杆旋转
    
    Args:
        angles: 旋转角度序列
            - 如果 rotation_type == "axis_angle": 形状为 (num_steps,) 或 (batch_size, num_steps)
            - 如果 rotation_type == "euler": 形状为 (num_steps, 3) 或 (batch_size, num_steps, 3)，
              分别表示 (roll, pitch, yaw)
        axis: 旋转轴，仅用于 axis_angle 模式，形状为 (3,)
              默认为 [0, 0, 1]（绕 Z 轴旋转）
        rotation_type: "axis_angle" 或 "euler"
        
    Returns:
        增量旋转四元数序列，形状为 (num_steps, 4) 或 (batch_size, num_steps, 4)
        格式为 (w, x, y, z)
    """
    if rotation_type == "axis_angle":
        if axis is None:
            axis = torch.tensor([0.0, 0.0, 1.0], dtype=angles.dtype, device=angles.device)
        
        if angles.dim() == 1:
            # (num_steps,) -> (num_steps, 4)
            axis_expanded = axis.unsqueeze(0).expand(angles.shape[0], -1)
            return axis_angle_to_quaternion(axis_expanded, angles)
        else:
            # (batch_size, num_steps) -> (batch_size, num_steps, 4)
            axis_expanded = axis.unsqueeze(0).unsqueeze(0).expand(angles.shape[0], angles.shape[1], -1)
            return axis_angle_to_quaternion(axis_expanded, angles.unsqueeze(-1).expand(-1, -1, 3)[..., 0])
    
    elif rotation_type == "euler":
        if angles.dim() == 2:
            # (num_steps, 3)
            return euler_to_quaternion(angles[:, 0], angles[:, 1], angles[:, 2])
        else:
            # (batch_size, num_steps, 3)
            return euler_to_quaternion(angles[..., 0], angles[..., 1], angles[..., 2])
    
    else:
        raise ValueError(f"Unknown rotation_type: {rotation_type}")
