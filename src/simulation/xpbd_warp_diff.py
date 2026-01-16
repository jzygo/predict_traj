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
from simulation.model.constraints import VariableDistanceConstraint, FixedPointConstraint, DistanceConstraint, \
    BendingConstraint, AlignmentConstraint, CollisionConstraint

# 使用无界面后端，便于子进程并行渲染
matplotlib.use("Agg")

PLANE_Z = 0.25


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

    sim_pixel = (wp.float64(1.0) / (wp.float64(1.0) + wp.exp(-sim_image[batch_id, tid])) - wp.float64(
        0.5)) * wp.float64(2)
    diff = sim_pixel - ref_image[tid]
    scale = wp.float64(1.0) / (wp.float64(height) * wp.float64(width))
    wp.atomic_add(loss, 0, wp.abs(diff) * scale)


@wp.kernel
def copy_now_to_prev(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        num_particles: int
):
    """将 now_position (1) 拷贝到 prev_position (0)，用于下一个时间步开始前"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return
    positions[batch_id, 0, tid] = positions[batch_id, 1, tid]


@wp.kernel
def integrate_particles(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        external_forces: wp.array(dtype=wp.vec3d, ndim=2),
        dt: wp.float64,
        num_particles: int
):
    """积分粒子位置。始终在 now_position (1) 上操作"""
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

        # 固定使用 time_step=1 (now_position)
        positions[batch_id, 1, tid] = positions[batch_id, 1, tid] + displacement


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
def apply_absolute_transform_to_fixed_positions(
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_local_coords: wp.array(dtype=wp.vec3d, ndim=1),  # 固定粒子相对于笔杆中心的局部坐标
        brush_center: wp.array(dtype=wp.vec3d, ndim=1),  # 每个batch的笔杆中心位置
        absolute_rotation: wp.array(dtype=wp.vec4d, ndim=1),  # 每个batch的绝对旋转四元数 (w, x, y, z)
        num_fixed: int
):
    """根据绝对位置和绝对旋转直接计算每个固定粒子的世界坐标

    每个固定粒子的新位置 = 笔杆中心位置 + 旋转后的局部坐标
    
    Args:
        fixed_positions: 固定粒子位置数组 (batch_size, num_fixed, 3)
        fixed_local_coords: 固定粒子相对于笔杆中心的初始局部坐标 (num_fixed, 3)
        brush_center: 当前笔杆中心的绝对位置 (batch_size, 3)
        absolute_rotation: 当前笔杆的绝对旋转四元数 (batch_size, 4)，格式为 (w, x, y, z)
        num_fixed: 固定粒子数量
    """
    batch_id, tid = wp.tid()
    if tid >= num_fixed:
        return

    # 获取当前笔杆中心和绝对旋转
    center = brush_center[batch_id]
    quat = absolute_rotation[batch_id]  # 绝对旋转四元数 (w, x, y, z)

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

    brush_center[batch_id, time_id] = brush_center[batch_id, time_id] + disp_seq[batch_id, time_id]

    # 更新累积旋转: new_rotation = delta_rotation * prev_rotation
    delta_rot = delta_rotation_seq[batch_id, time_id]
    prev_rot = rotation_seq[batch_id, time_id]
    rotation_seq[batch_id, time_id] = quat_multiply(delta_rot, prev_rot)


@wp.kernel
def apply_prev_pos_to_now_pos(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_indices: wp.array(dtype=int),
        inv_mass: wp.array(dtype=wp.float64, ndim=2),
        num_particles: int
):
    """将固定位置应用到 now_position (1)"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    idx = fixed_indices[tid]
    # 固定使用 time_step=1 (now_position)
    positions[batch_id, 1, idx] = fixed_positions[batch_id, tid]


import warp as wp


@wp.kernel
def record_ink_collision_hard(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        ink_canvas: wp.array(dtype=wp.float64, ndim=2),
        canvas_resolution: int,
        num_particles: int,
        sigma_px: wp.float64,  # 保留参数以维持接口兼容，但在函数内不使用
        radius_px: int,  # 保留参数以维持接口兼容，但在函数内不使用
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float
):
    """记录墨迹碰撞（硬边）。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    # 固定使用 time_step=1 (now_position)
    pos = positions[batch_id, 1, tid]

    # 这里的 PLANE_Z 需要确保在外部定义或作为常量传入，假设原逻辑依然需要判断高度
    # 注意：如果原本是在全局变量里定义的 PLANE_Z，请确保 kernel 能访问到
    if not pos[2] <= wp.float64(0.0005) + wp.float64(PLANE_Z):
        return

    # 1. 坐标映射 (保持不变)
    px = wp.clamp(pos[0], wp.float64(min_x), wp.float64(max_x))
    py = wp.clamp(pos[1], wp.float64(min_y), wp.float64(max_y))

    resm1 = wp.float64(canvas_resolution - 1)

    # Map x from [min_x, max_x] to [0, resm1]
    gx = (px - wp.float64(min_x)) / (wp.float64(max_x) - wp.float64(min_x)) * resm1

    # Map y from [min_y, max_y] to [resm1, 0]
    gy = (wp.float64(max_y) - py) / (wp.float64(max_y) - wp.float64(min_y)) * resm1

    # 2. 计算具体的像素整数索引 (修改部分)
    # 使用 round 找到最近的像素中心，而不是 floor
    ix = int(wp.round(gx))
    iy = int(wp.round(gy))

    # 3. 边界检查与写入 (修改部分)
    # 尽管前面做了 clamp，但 round 可能会导致坐标正好溢出 (例如 511.9 -> 512)，所以必须检查
    if ix >= 0 and ix < canvas_resolution and iy >= 0 and iy < canvas_resolution:
        pixel_idx = iy * canvas_resolution + ix

        # 直接添加固定强度 1.0，或者你可以根据需要添加 inv_masses[batch_id, tid]
        wp.atomic_add(ink_canvas, batch_id, pixel_idx, wp.float64(1.0))

@wp.kernel
def solve_distance_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
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
    """距离约束求解器。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_constraints:
        return

    idx = constraint_indices[tid]
    i, j = idx[0], idx[1]

    # 固定使用 time_step=1 (now_position)
    xi = positions[batch_id, 1, i]
    xj = positions[batch_id, 1, j]

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
def solve_alignment_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        constraint_indices: wp.array(dtype=wp.vec3i),
        rest_ratios: wp.array(dtype=wp.float64),
        compliances: wp.array(dtype=wp.float64),
        lagrange_multipliers: wp.array(dtype=wp.float64),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_constraints: int
):
    """对齐约束求解器。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_constraints:
        return

    idx = constraint_indices[tid]
    i, j, k = idx[0], idx[1], idx[2]

    # Get positions: i, j are fixed (start, end), k is free
    # 固定使用 time_step=1 (now_position)
    p_start = positions[batch_id, 1, i]
    p_end = positions[batch_id, 1, j]
    p_free = positions[batch_id, 1, k]

    # Axis vector from start to end
    axis = p_end - p_start
    axis_sq = wp.dot(axis, axis)
    eps = wp.float64(1e-12)

    if axis_sq < eps:
        return

    # New projection definition: Use stored ratio to determine target position along the line
    ratio = rest_ratios[tid]

    # Calculate target position
    p_proj = p_start + axis * ratio

    limit_plane_z = wp.float64(0.25)
    if p_proj[2] < limit_plane_z:
        if wp.abs(axis[2]) > eps:
            t_int = (limit_plane_z - p_start[2]) / axis[2]
            p_int = p_start + axis * t_int

            # distance between original projected point and intersection
            dist_p_i = wp.length(p_proj - p_int)

            # direction from intersection to projected point in xy plane
            dir_x = p_proj[0] - p_int[0]
            dir_y = p_proj[1] - p_int[1]
            mag_xy = wp.sqrt(dir_x * dir_x + dir_y * dir_y)

            if mag_xy > eps:
                scale = dist_p_i / mag_xy
                p_proj = wp.vec3d(p_int[0] + dir_x * scale, p_int[1] + dir_y * scale, limit_plane_z)
            else:
                p_proj = p_int

        # Distance vector from projection to free point
        diff = (p_free - p_proj) * wp.float64(1.2)
        dist = wp.length(diff)
    else:
        # Distance vector from projection to free point
        diff = p_free - p_proj
        dist = wp.length(diff)

    if dist < eps:
        return

    # Direction of correction (pulling free particle towards line)
    n = diff / dist

    w_free = inv_masses[batch_id, k]
    # For fixed particles, inv_mass is zero, so we only consider w_free
    w_sum = w_free

    if w_sum < eps:
        return

    alpha = compliances[tid] / (dt * dt)
    denom = w_sum + alpha

    # Delta lambda
    delta_lambda = -(dist + alpha * lagrange_multipliers[tid]) / denom
    lagrange_multipliers[tid] = lagrange_multipliers[tid] + delta_lambda

    # Correction
    correction = delta_lambda * n * w_free

    # We only update the free particle
    wp.atomic_add(deltas_x, batch_id, k, correction[0])
    wp.atomic_add(deltas_y, batch_id, k, correction[1])
    wp.atomic_add(deltas_z, batch_id, k, correction[2])


@wp.kernel
def apply_and_clear_deltas(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        num_particles: int
):
    """应用并清除位移增量。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return
    dx = deltas_x[batch_id, tid]
    dy = deltas_y[batch_id, tid]
    dz = deltas_z[batch_id, tid]
    if dx != wp.float64(0.0) or dy != wp.float64(0.0) or dz != wp.float64(0.0):
        # 固定使用 time_step=1 (now_position)
        positions[batch_id, 1, tid] = positions[batch_id, 1, tid] + wp.vec3d(dx, dy, dz)
    deltas_x[batch_id, tid] = wp.float64(0.0)
    deltas_y[batch_id, tid] = wp.float64(0.0)
    deltas_z[batch_id, tid] = wp.float64(0.0)


@wp.kernel
def solve_plane_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
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
    """平面约束求解器。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] <= wp.float64(0.0):  # Fixed particles
        return

    # 固定使用 time_step=1 (now_position)
    pos = positions[batch_id, 1, tid]

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
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_positions: wp.array(dtype=wp.vec3d, ndim=2),
        fixed_indices: wp.array(dtype=int),
        num_fixed: int,
        stiffness: wp.float64  # 软约束强度 (0,1]，越接近1越“硬”
):
    """应用固定约束（保持梯度）。固定使用 time_step=1 (now_position)"""
    batch_id, tid = wp.tid()
    if tid >= num_fixed:
        return

    idx = fixed_indices[tid]
    # 固定使用 time_step=1 (now_position)
    positions[batch_id, 1, idx] = fixed_positions[batch_id, tid]


@wp.kernel
def update_velocities(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        velocities: wp.array(dtype=wp.vec3d, ndim=2),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        num_particles: int
):
    """更新速度。使用固定的 time_step=0 表示 prev_position，time_step=1 表示 now_position"""
    batch_id, tid = wp.tid()
    if tid >= num_particles:
        return

    if inv_masses[batch_id, tid] > wp.float64(0.0):
        # 使用 now_position (1) - prev_position (0) 计算速度
        velocities[batch_id, tid] = (positions[batch_id, 1, tid] - positions[batch_id, 0, tid]) / dt


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


@wp.kernel
def detect_collisions(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        hair_particle_start: wp.array(dtype=int),  # 每根毛发的起始粒子索引
        hair_particle_count: wp.array(dtype=int),  # 每根毛发的粒子数量
        collision_pairs: wp.array(dtype=wp.vec2i, ndim=2),  # 输出：碰撞粒子对 (batch, max_pairs, 2)
        collision_count: wp.array(dtype=int, ndim=1),  # 输出：每个batch的碰撞对数量
        collision_radius: wp.float64,
        num_hairs: int,
        max_collision_pairs: int
):
    """检测不同毛发之间的粒子碰撞
    
    为了避免同一根毛发内部的粒子碰撞（它们已经有距离约束），
    只检测不同毛发之间的碰撞。
    """
    batch_id, pair_id = wp.tid()
    
    # pair_id 编码了两根毛发的索引 (hair_i, hair_j)，其中 hair_i < hair_j
    # 总共有 C(num_hairs, 2) = num_hairs * (num_hairs - 1) / 2 对
    total_pairs = (num_hairs * (num_hairs - 1)) / 2
    if wp.float64(pair_id) >= total_pairs:
        return
    
    # 使用数学公式直接从 pair_id 计算 hair_i 和 hair_j
    # 对于上三角矩阵索引，pair_id 与 (hair_i, hair_j) 的关系:
    # pair_id = hair_i * num_hairs - hair_i * (hair_i + 1) / 2 + (hair_j - hair_i - 1)
    # 使用求根公式反解 hair_i:
    # hair_i = floor((2*n - 1 - sqrt((2*n-1)^2 - 8*pair_id)) / 2)
    n = wp.float64(num_hairs)
    k = wp.float64(pair_id)
    
    # 计算 hair_i: 使用二次方程求根公式
    # 从 k = i*(n-1) - i*(i-1)/2 + (j-i-1) 反解
    # 简化后: i = floor((2n - 1 - sqrt((2n-1)^2 - 8k)) / 2)
    discriminant = (wp.float64(2.0) * n - wp.float64(1.0)) * (wp.float64(2.0) * n - wp.float64(1.0)) - wp.float64(8.0) * k
    hair_i_float = (wp.float64(2.0) * n - wp.float64(1.0) - wp.sqrt(discriminant)) / wp.float64(2.0)
    hair_i = int(wp.floor(hair_i_float))
    
    # 计算 hair_j: 基于 hair_i 计算偏移
    # pair_id = hair_i * (num_hairs - 1) - hair_i * (hair_i - 1) / 2 + (hair_j - hair_i - 1)
    # 简化: pair_id = hair_i * num_hairs - hair_i * (hair_i + 1) / 2 + hair_j - hair_i - 1
    i_f = wp.float64(hair_i)
    base_idx = int(i_f * n - i_f * (i_f + wp.float64(1.0)) / wp.float64(2.0))
    hair_j = hair_i + 1 + (pair_id - base_idx)
    
    # 边界检查
    if hair_i < 0 or hair_i >= num_hairs - 1 or hair_j <= hair_i or hair_j >= num_hairs:
        return
    
    # 获取两根毛发的粒子范围
    start_i = hair_particle_start[hair_i]
    count_i = hair_particle_count[hair_i]
    start_j = hair_particle_start[hair_j]
    count_j = hair_particle_count[hair_j]
    
    collision_dist = collision_radius * wp.float64(2.0)
    collision_dist_sq = collision_dist * collision_dist
    
    # 检测两根毛发之间的所有粒子对
    for pi in range(count_i):
        particle_i = start_i + pi
        # 跳过固定粒子
        if inv_masses[batch_id, particle_i] <= wp.float64(0.0):
            continue
            
        pos_i = positions[batch_id, 1, particle_i]
        
        for pj in range(count_j):
            particle_j = start_j + pj
            # 跳过固定粒子
            if inv_masses[batch_id, particle_j] <= wp.float64(0.0):
                continue
                
            pos_j = positions[batch_id, 1, particle_j]
            diff = pos_i - pos_j
            dist_sq = wp.dot(diff, diff)
            
            if dist_sq < collision_dist_sq and dist_sq > wp.float64(1e-12):
                # 发现碰撞，原子性地添加到碰撞对列表
                idx = wp.atomic_add(collision_count, batch_id, 1)
                if idx < max_collision_pairs:
                    collision_pairs[batch_id, idx] = wp.vec2i(particle_i, particle_j)


@wp.kernel
def solve_collision_constraints(
        positions: wp.array(dtype=wp.vec3d, ndim=3),
        inv_masses: wp.array(dtype=wp.float64, ndim=2),
        collision_pairs: wp.array(dtype=wp.vec2i, ndim=2),
        collision_count: wp.array(dtype=int, ndim=1),
        collision_radius: wp.float64,
        compliance: wp.float64,
        deltas_x: wp.array(dtype=wp.float64, ndim=2),
        deltas_y: wp.array(dtype=wp.float64, ndim=2),
        deltas_z: wp.array(dtype=wp.float64, ndim=2),
        dt: wp.float64,
        max_collision_pairs: int
):
    """碰撞约束求解器
    
    对于每个检测到的碰撞对，如果距离小于碰撞距离，则施加排斥约束将粒子推开。
    """
    batch_id, tid = wp.tid()
    
    count = collision_count[batch_id]
    if tid >= count or tid >= max_collision_pairs:
        return
    
    pair = collision_pairs[batch_id, tid]
    i = pair[0]
    j = pair[1]
    
    pos_i = positions[batch_id, 1, i]
    pos_j = positions[batch_id, 1, j]
    
    diff = pos_i - pos_j
    dist = wp.length(diff)
    
    collision_dist = collision_radius * wp.float64(2.0)
    
    epsilon = wp.float64(1e-12)
    if dist < epsilon or dist >= collision_dist:
        return
    
    # 计算穿透深度（负值表示穿透）
    penetration = dist - collision_dist
    
    # 方向：从 j 指向 i
    n = diff / (dist + epsilon)
    
    w_i = inv_masses[batch_id, i]
    w_j = inv_masses[batch_id, j]
    w_sum = w_i + w_j
    
    if w_sum < epsilon:
        return
    
    # XPBD 约束求解
    alpha = compliance / (dt * dt)
    denom = w_sum + alpha
    
    # delta_lambda = -penetration / denom (简化版本，不维护拉格朗日乘子，因为碰撞是瞬时约束)
    delta_lambda = -penetration / denom
    
    # 限制最大修正量
    max_correction = collision_dist * wp.float64(0.5)
    delta_lambda = wp.clamp(delta_lambda, -max_correction, max_correction)
    
    # 计算位置修正
    correction = delta_lambda * n
    
    # 根据质量比例分配修正量
    if w_i > wp.float64(0.0):
        corr_i = correction * (w_i / w_sum)
        wp.atomic_add(deltas_x, batch_id, i, corr_i[0])
        wp.atomic_add(deltas_y, batch_id, i, corr_i[1])
        wp.atomic_add(deltas_z, batch_id, i, corr_i[2])
    
    if w_j > wp.float64(0.0):
        corr_j = -correction * (w_j / w_sum)
        wp.atomic_add(deltas_x, batch_id, j, corr_j[0])
        wp.atomic_add(deltas_y, batch_id, j, corr_j[1])
        wp.atomic_add(deltas_z, batch_id, j, corr_j[2])


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
                 variable_z_threshold: float = 0.005,  # z坐标阈值，超过此值约束强度不再变化
                 variable_weakness_factor: float = 10.0,  # 接近z=0时的弱化倍数
                 splat_sigma_px: float = 1.0,  # 高斯核标准差（以像素为单位）
                 splat_radius_px: int = 1,  # 溅射半径（像素，2->5x5 邻域）
                 fixed_stiffness: float = 0.6,  # 软固定约束的收敛强度 (0,1]，越大越“硬”
                 alignment_compliance_range: Tuple[float, float] = (1e-4, 1e-4),  # 对齐约束compliance范围 (root, tip)
                 collision_radius: float = 0.0005,  # 碰撞半径，粒子间距离小于2*collision_radius时产生碰撞
                 collision_compliance: float = 1e-6,  # 碰撞约束的柔度
                 enable_collision: bool = True,  # 是否启用碰撞检测
                 max_collision_pairs: int = 10000,  # 每个batch最大碰撞对数量
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
        self.alignment_compliance_range = alignment_compliance_range
        self.collision_radius = collision_radius
        self.collision_compliance = collision_compliance
        self.enable_collision = enable_collision
        self.max_collision_pairs = max_collision_pairs
        self.damping = damping
        self.restitution = restitution
        self.friction = friction

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

        # Alignment constraint arrays
        self.alignment_indices = None
        self.alignment_rest_ratios = None
        self.alignment_compliances = None
        self.alignment_lambdas = None

        self.fixed_indices = None
        self.fixed_positions = None

        # 笔杆旋转相关数组
        self.fixed_local_coords = None  # 固定粒子相对于笔杆中心的局部坐标
        self.brush_center = None  # 笔杆中心位置 (batch_size, num_steps, 3)
        self.rotation_seq = None  # 累积旋转四元数序列 (batch_size, num_steps, 4)
        self.delta_rotation_seq = None  # 增量旋转四元数序列 (batch_size, num_steps, 4)
        self.use_rotation = False  # 是否使用旋转模式
        self.initial_brush_center = None  # 初始笔杆中心位置

        # 绝对变换模式相关数组
        self.current_brush_center = None  # 当前笔杆中心位置 (batch_size, 3)
        self.current_absolute_rotation = None  # 当前绝对旋转四元数 (batch_size, 4)

        # Plane constraint arrays (for z=0 plane)
        self.plane_normals = None
        self.plane_distances = None
        self.plane_compliances = None
        self.plane_lambdas = None
        self.num_planes = 0

        # 碰撞约束数组
        self.hair_particle_start = None  # 每根毛发的起始粒子索引
        self.hair_particle_count = None  # 每根毛发的粒子数量
        self.collision_pairs = None  # 碰撞粒子对
        self.collision_count = None  # 每个batch的碰撞对数量
        self.num_hairs = 0  # 毛发数量

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

    def load_brush(self, brush: Brush, keep_ink_canvas: bool = False):
        """Load particles and constraints from brush"""
        # Generate brush structure if not done
        if not brush.hairs:
            brush.gen_hairs()
            brush.gen_constraints()

        self.num_particles = len(brush.particles)

        # 存储初始笔杆中心位置（用于旋转计算）
        self.initial_brush_center = brush.root_position.clone().to(torch.float64)

        # Extract particle data using torch tensors
        # 只需要 2 个时间步：time_step=0 表示 prev_position，time_step=1 表示 now_position
        positions = torch.zeros((self.batch_size, 2, self.num_particles, 3), dtype=torch.float64).to(
            self.device)
        velocities = torch.zeros((self.batch_size, self.num_particles, 3), dtype=torch.float64).to(self.device)
        inv_masses = torch.ones((self.batch_size, self.num_particles), dtype=torch.float64).to(self.device)

        for b in range(self.batch_size):
            for i, particle in enumerate(brush.particles):
                # 同时初始化 prev_position (0) 和 now_position (1)
                positions[b, 0, i] = torch.tensor(particle.position, dtype=torch.float64)
                positions[b, 1, i] = torch.tensor(particle.position, dtype=torch.float64)
                velocities[b, i] = torch.tensor(particle.velocity, dtype=torch.float64)
                inv_masses[b, i] = 1.0

        self.positions = wp.from_torch(positions, dtype=wp.vec3d, requires_grad=True)
        self.velocities = wp.from_torch(velocities, dtype=wp.vec3d)
        self.inv_masses = wp.from_torch(inv_masses, dtype=wp.float64)

        # External forces (gravity) using torch
        forces = self.gravity.unsqueeze(0).unsqueeze(0).repeat(self.batch_size, self.num_particles, 1).to(
            torch.float64).to(self.device)
        self.external_forces = wp.from_torch(forces, dtype=wp.vec3d)

        # Allocate zeroed delta buffers for atomic accumulation
        self._delta_x = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)
        self._delta_y = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)
        self._delta_z = wp.zeros((self.batch_size, self.num_particles), dtype=wp.float64, device=self.device)

        # Initialize ink canvas (1x1 canvas mapped to canvas_resolution x canvas_resolution pixels)
        if not keep_ink_canvas or self.ink_canvas is None:
            canvas_size = self.canvas_resolution * self.canvas_resolution
            # Enable gradient tracking on the ink canvas for autodiff
            self.ink_canvas = wp.zeros((self.batch_size, canvas_size), dtype=wp.float64, device=self.device,
                                       requires_grad=True)

        # Extract constraints (会设置 fixed_local_coords)
        self._extract_constraints(brush.constraints, brush)

        # Setup plane constraints (z=0 plane)
        self._setup_plane_constraints()

        # Setup collision detection data structures
        if self.enable_collision:
            self._setup_collision_data(brush)

        self.particles_initialized = True

    def _extract_constraints(self, constraints, brush: Brush = None):
        """Extract and categorize constraints

        Args:
            constraints: 约束列表
            brush: Brush 对象，用于获取 root_position 和 tangent_vector 来计算固定粒子的局部坐标
        """
        brush_root_position = brush.root_position if brush is not None else None
        brush_tangent_vector = brush.tangent_vector if brush is not None else None
        distance_data = []
        variable_distance_data = []
        bending_data = []
        alignment_data = []
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

            elif isinstance(constraint, AlignmentConstraint):
                i = constraint.point_fixed_start.particle_id
                j = constraint.point_fixed_end.particle_id
                k = constraint.point_free.particle_id
                ratio = constraint.rest_ratio

                # 计算 variable compliance
                # relative_location: 0.0 (near root) -> 1.0 (tip)
                comp_start, comp_end = self.alignment_compliance_range
                t = constraint.relative_location
                compliance = comp_start + (comp_end - comp_start) * t

                alignment_data.append((i, j, k, ratio, compliance))

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
            bending_indices = torch.tensor([(d[0], d[1], d[2]) for d in bending_data], dtype=torch.int32).to(
                self.device)
            bending_rest_distances = torch.tensor([d[3] for d in bending_data], dtype=torch.float64).to(self.device)
            bending_compliances = torch.tensor([d[4] for d in bending_data], dtype=torch.float64).to(self.device)
            bending_lambdas = torch.zeros(num_bending, dtype=torch.float64).to(self.device)

            self.bending_indices = wp.from_torch(bending_indices, dtype=wp.vec3i)
            self.bending_rest_distances = wp.from_torch(bending_rest_distances, dtype=wp.float64)
            self.bending_compliances = wp.from_torch(bending_compliances, dtype=wp.float64)
            self.bending_lambdas = wp.from_torch(bending_lambdas, dtype=wp.float64)
            self.num_bending_constraints = num_bending

        # Create alignment constraint arrays using torch
        if alignment_data:
            num_ali = len(alignment_data)
            ali_indices = torch.tensor([(d[0], d[1], d[2]) for d in alignment_data], dtype=torch.int32).to(self.device)
            ali_ratios = torch.tensor([d[3] for d in alignment_data], dtype=torch.float64).to(self.device)
            ali_compliances = torch.tensor([d[4] for d in alignment_data], dtype=torch.float64).to(self.device)
            ali_lambdas = torch.zeros(num_ali, dtype=torch.float64).to(self.device)

            self.alignment_indices = wp.from_torch(ali_indices, dtype=wp.vec3i)
            self.alignment_rest_ratios = wp.from_torch(ali_ratios, dtype=wp.float64)
            self.alignment_compliances = wp.from_torch(ali_compliances, dtype=wp.float64)
            self.alignment_lambdas = wp.from_torch(ali_lambdas, dtype=wp.float64)
            self.num_alignment_constraints = num_ali

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
            # 注意：这里的局部坐标需要是初始状态（竖直向下）下的坐标
            # 如果毛笔已经旋转，需要将局部坐标反向旋转回初始状态
            if brush_root_position is not None:
                brush_center = brush_root_position.to(torch.float64).to(self.device)
                # 每个固定粒子的局部坐标 = 粒子位置 - 笔杆中心
                fixed_pos_stack = torch.stack([f[1] for f in fixed_data]).to(torch.float64).to(self.device)
                local_coords = fixed_pos_stack - brush_center
                
                # 如果毛笔有切向量信息，需要将当前坐标反向旋转回初始状态（竖直向下）
                # 这样 absolute_rotation 才能正确地从初始状态旋转到目标状态
                if brush_tangent_vector is not None:
                    from simulation.model.brush import rotation_matrix_from_vectors
                    tangent = brush_tangent_vector.to(torch.float64)
                    vertical_down = torch.tensor([0., 0., -1.], dtype=torch.float64)
                    
                    # 如果当前切向量不是竖直向下，需要反向旋转
                    if not torch.allclose(tangent, vertical_down, atol=1e-6):
                        # 计算从 vertical_down 到 tangent 的旋转矩阵
                        rot_mat = rotation_matrix_from_vectors(vertical_down, tangent)
                        # 逆旋转矩阵 = 转置（对于正交旋转矩阵）
                        inv_rot_mat = rot_mat.T.to(self.device)
                        # 将每个局部坐标反向旋转
                        local_coords = (inv_rot_mat @ local_coords.T).T
                
                # 确保张量是连续的，warp.from_torch 要求连续张量
                self.fixed_local_coords = wp.from_torch(local_coords.contiguous(), dtype=wp.vec3d)

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
        plane_lambdas = torch.zeros((self.batch_size, self.num_particles * self.num_planes), dtype=torch.float64).to(
            self.device)
        self.plane_lambdas = wp.from_torch(plane_lambdas, dtype=wp.float64)

    def _setup_collision_data(self, brush: Brush):
        """Setup data structures for collision detection between hairs
        
        为碰撞检测设置数据结构，记录每根毛发的粒子范围信息。
        """
        self.num_hairs = len(brush.hairs)
        
        if self.num_hairs < 2:
            # 至少需要2根毛发才能进行碰撞检测
            self.enable_collision = False
            return
        
        # 构建每根毛发的粒子起始索引和数量
        hair_starts = []
        hair_counts = []
        
        for hair in brush.hairs:
            if len(hair.particles) > 0:
                # 获取该毛发第一个粒子的ID作为起始索引
                start_idx = hair.particles[0].particle_id
                count = len(hair.particles)
                hair_starts.append(start_idx)
                hair_counts.append(count)
        
        self.num_hairs = len(hair_starts)
        
        if self.num_hairs < 2:
            self.enable_collision = False
            return
        
        # 转换为torch tensor并传输到设备
        hair_starts_tensor = torch.tensor(hair_starts, dtype=torch.int32).to(self.device)
        hair_counts_tensor = torch.tensor(hair_counts, dtype=torch.int32).to(self.device)
        
        self.hair_particle_start = wp.from_torch(hair_starts_tensor, dtype=wp.int32)
        self.hair_particle_count = wp.from_torch(hair_counts_tensor, dtype=wp.int32)
        
        # 分配碰撞对数组
        collision_pairs = torch.zeros((self.batch_size, self.max_collision_pairs, 2), dtype=torch.int32).to(self.device)
        collision_count = torch.zeros(self.batch_size, dtype=torch.int32).to(self.device)
        
        self.collision_pairs = wp.from_torch(collision_pairs, dtype=wp.vec2i)
        self.collision_count = wp.from_torch(collision_count, dtype=wp.int32)

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
            # 移除了此处的 copy_now_to_prev，移至 _substep 内部以确保 XPBD 在 substeps > 1 时速度计算正确

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
            # 将固定粒子的位置应用到 now_position
            wp.launch(
                apply_prev_pos_to_now_pos,
                dim=(self.batch_size, self.num_fixed_constraints),
                inputs=[
                    self.positions,
                    self.fixed_positions,
                    self.fixed_indices,
                    self.inv_masses,
                    self.num_particles
                ],
                device=self.device
            )
            for j in range(self.substeps):
                self._substep()

    def _substep(self):
        """执行一个子步骤。所有位置操作都使用固定的 time_step=1 (now_position)"""
        # [Fix] 必须在每个子步开始时更新 prev_position，否则 update_velocities 会使用错误的参考位置导致爆炸
        wp.launch(
            copy_now_to_prev,
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.positions,
                self.num_particles
            ],
            device=self.device
        )

        wp.launch(
            integrate_particles,
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.positions,
                self.velocities,
                self.inv_masses,
                self.external_forces,
                self.sub_dt,
                self.num_particles
            ],
            device=self.device
        )
        if self.alignment_lambdas is not None:
            self.alignment_lambdas.zero_()
        if self.distance_lambdas is not None:
            self.distance_lambdas.zero_()
        if self.plane_lambdas is not None:
            self.plane_lambdas.zero_()
        if self.bending_lambdas is not None:
            self.bending_lambdas.zero_()
        # return
        for _ in range(self.iterations):
            if self.distance_indices is not None and self.num_distance_constraints > 0:
                wp.launch(
                    solve_distance_constraints,
                    dim=(self.batch_size, self.num_distance_constraints),
                    inputs=[
                        self.positions,
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

            if self.alignment_indices is not None and self.num_alignment_constraints > 0:
                wp.launch(
                    solve_alignment_constraints,
                    dim=(self.batch_size, self.num_alignment_constraints),
                    inputs=[
                        self.positions,
                        self.inv_masses,
                        self.alignment_indices,
                        self.alignment_rest_ratios,
                        self.alignment_compliances,
                        self.alignment_lambdas,
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.sub_dt,
                        self.num_alignment_constraints
                    ],
                    device=self.device
                )

            # Apply accumulated deltas and clear for next iteration
            wp.launch(
                apply_and_clear_deltas,
                dim=(self.batch_size, self.num_particles),
                inputs=[
                    self.positions,
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
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.num_particles
                    ],
                    device=self.device
                )

            # Solve collision constraints (hair-hair collision)
            if self.enable_collision and self.num_hairs >= 2:
                # 重置碰撞计数
                self.collision_count.zero_()
                
                # 检测碰撞对
                num_hair_pairs = (self.num_hairs * (self.num_hairs - 1)) // 2
                wp.launch(
                    detect_collisions,
                    dim=(self.batch_size, num_hair_pairs),
                    inputs=[
                        self.positions,
                        self.inv_masses,
                        self.hair_particle_start,
                        self.hair_particle_count,
                        self.collision_pairs,
                        self.collision_count,
                        wp.float64(self.collision_radius),
                        self.num_hairs,
                        self.max_collision_pairs
                    ],
                    device=self.device
                )
                
                # 求解碰撞约束
                wp.launch(
                    solve_collision_constraints,
                    dim=(self.batch_size, self.max_collision_pairs),
                    inputs=[
                        self.positions,
                        self.inv_masses,
                        self.collision_pairs,
                        self.collision_count,
                        wp.float64(self.collision_radius),
                        wp.float64(self.collision_compliance),
                        self._delta_x,
                        self._delta_y,
                        self._delta_z,
                        self.sub_dt,
                        self.max_collision_pairs
                    ],
                    device=self.device
                )
                
                # Apply collision constraint corrections
                wp.launch(
                    apply_and_clear_deltas,
                    dim=(self.batch_size, self.num_particles),
                    inputs=[
                        self.positions,
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
                        self.velocities,
                        self.fixed_positions,
                        self.fixed_indices,
                        self.num_fixed_constraints,
                        wp.float64(self.fixed_stiffness)
                    ],
                    device=self.device
                )

        wp.launch(
            record_ink_collision_hard,
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.positions,
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
            dim=(self.batch_size, self.num_particles),
            inputs=[
                self.positions,
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

    def update_fixed_positions_from_transform(
            self,
            brush_center: torch.Tensor,
            absolute_rotation: torch.Tensor
    ):
        """根据绝对位置和绝对旋转直接更新固定粒子位置
        
        Args:
            brush_center: 笔杆中心的绝对位置，形状为 (3,) 或 (batch_size, 3)
            absolute_rotation: 笔杆的绝对旋转四元数，形状为 (4,) 或 (batch_size, 4)
                              格式为 (w, x, y, z)
        """
        if self.fixed_local_coords is None or self.num_fixed_constraints == 0:
            return

        # 处理输入维度
        if brush_center.dim() == 1:
            brush_center = brush_center.unsqueeze(0).repeat(self.batch_size, 1)
        if absolute_rotation.dim() == 1:
            absolute_rotation = absolute_rotation.unsqueeze(0).repeat(self.batch_size, 1)

        # 转换为设备上的张量
        brush_center = brush_center.to(torch.float64).to(self.device)
        absolute_rotation = absolute_rotation.to(torch.float64).to(self.device)

        # 转换为 warp 数组
        self.current_brush_center = wp.from_torch(brush_center, dtype=wp.vec3d)
        self.current_absolute_rotation = wp.from_torch(absolute_rotation, dtype=wp.vec4d)

        # 使用 kernel 计算固定粒子的新位置
        wp.launch(
            apply_absolute_transform_to_fixed_positions,
            dim=(self.batch_size, self.num_fixed_constraints),
            inputs=[
                self.fixed_positions,
                self.fixed_local_coords,
                self.current_brush_center,
                self.current_absolute_rotation,
                self.num_fixed_constraints
            ],
            device=self.device
        )

        # 将固定粒子的位置应用到 now_position
        wp.launch(
            apply_prev_pos_to_now_pos,
            dim=(self.batch_size, self.num_fixed_constraints),
            inputs=[
                self.positions,
                self.fixed_positions,
                self.fixed_indices,
                self.inv_masses,
                self.num_particles
            ],
            device=self.device
        )

    def step_with_absolute_transform(
            self,
            brush_center: torch.Tensor,
            absolute_rotation: torch.Tensor
    ):
        """使用绝对变换执行一步仿真
        
        与 step() 不同，此方法直接使用绝对位置和旋转来更新固定粒子，
        而不是通过增量位移和旋转累积。
        
        Args:
            brush_center: 笔杆中心的绝对位置，形状为 (3,) 或 (batch_size, 3)
            absolute_rotation: 笔杆的绝对旋转四元数，形状为 (4,) 或 (batch_size, 4)
                              格式为 (w, x, y, z)
        """
        # 直接更新固定粒子位置
        self.update_fixed_positions_from_transform(brush_center, absolute_rotation)

        # 执行物理仿真子步骤
        for j in range(self.substeps):
            self._substep()
