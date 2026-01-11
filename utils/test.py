import warp as wp
import warp.render
import numpy as np
import math

# 初始化 Warp
wp.init()


# ==========================================
# 1. 配置与物理参数
# ==========================================
class SimConfig:
    num_strands = 2000  # 毛发根数
    points_per_strand = 20  # 每根毛发的段数
    segment_length = 0.1  # 每段长度

    # XPBD 参数
    dt = 1.0 / 60.0  # 时间步长
    substeps = 10  # 子步数 (提高稳定性)

    gravity = wp.vec3(0.0, -9.8, 0.0)
    damping = 0.05  # 空气阻力

    # 约束柔顺度 (Compliance alpha = 1 / stiffness)
    # alpha = 0 表示无限硬 (刚性)
    alpha_stretch = 0.0001  # 拉伸柔顺度
    alpha_bend = 0.01  # 弯曲柔顺度

    # 碰撞体 (头部)
    head_radius = 1.5
    head_center = wp.vec3(0.0, 0.0, 0.0)


cfg = SimConfig()
total_points = cfg.num_strands * cfg.points_per_strand


# ==========================================
# 2. Warp Kernel 定义 (GPU 并行计算核心)
# ==========================================

# ------------------------------------------------
# 初始化：在球面上生成毛发
# ------------------------------------------------
@wp.kernel
def init_hair(
        points: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        segment_length: float,
        num_strands: int,
        points_per_strand: int,
        head_radius: float
):
    tid = wp.tid()

    # 计算当前点属于哪根毛发 (strand_idx) 以及是第几段 (local_idx)
    strand_idx = tid // points_per_strand
    local_idx = tid % points_per_strand

    # 使用 Hash 生成随机分布的根节点位置 (上半球)
    state = wp.rand_init(42, strand_idx)

    # 随机球面坐标 (极坐标)
    theta = wp.randf(state) * 2.0 * math.pi
    phi = wp.randf(state) * 0.5 * math.pi  # 0 到 pi/2，只在上半球

    x = head_radius * wp.sin(phi) * wp.cos(theta)
    y = head_radius * wp.cos(phi)  # Y轴向上
    z = head_radius * wp.sin(phi) * wp.sin(theta)

    root_pos = wp.vec3(x, y, z)
    normal = wp.normalize(root_pos)

    # 沿法线方向生长
    pos = root_pos + normal * (float(local_idx) * segment_length)

    points[tid] = pos
    velocities[tid] = wp.vec3(0.0, 0.0, 0.0)

    # 根节点质量无限大 (inv_mass = 0)，其余点为 1.0
    if local_idx == 0:
        inv_mass[tid] = 0.0
    else:
        inv_mass[tid] = 1.0


# ------------------------------------------------
# 显式积分 (Semi-Implicit Euler) - 第一步预测
# ------------------------------------------------
@wp.kernel
def predict_positions(
        points: wp.array(dtype=wp.vec3),
        predicted_points: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        gravity: wp.vec3,
        dt: float,
        damping: float
):
    tid = wp.tid()

    if inv_mass[tid] == 0.0:
        predicted_points[tid] = points[tid]
        return

    # v = v + g * dt
    vel = velocities[tid] + gravity * dt
    vel = vel * (1.0 - damping)  # 简单阻力

    # p_pred = p + v * dt
    pos = points[tid] + vel * dt

    predicted_points[tid] = pos
    velocities[tid] = vel  # 临时存储更新后的速度


# ------------------------------------------------
# XPBD 距离约束 (处理拉伸)
# ------------------------------------------------
@wp.kernel
def solve_stretch(
        predicted_points: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        segment_length: float,
        alpha: float,
        dt: float,
        points_per_strand: int
):
    # 以 Strand 为单位并行，或者以约束为单位并行。
    # 为了避免通过 Shared Memory 处理复杂的依赖，这里我们简单地每个点处理它和它父节点的约束
    # 注意：这在严格并行下是 Gauss-Seidel 风格的红黑迭代更佳，这里为了代码简洁
    # 采用每个线程处理一条边 (idx, idx-1)

    tid = wp.tid()
    local_idx = tid % points_per_strand

    if local_idx == 0:
        return  # 根节点没有父节点

    idx1 = tid - 1
    idx2 = tid

    p1 = predicted_points[idx1]
    p2 = predicted_points[idx2]

    w1 = inv_mass[idx1]
    w2 = inv_mass[idx2]

    if w1 + w2 == 0.0:
        return

    # C(x) = |p1 - p2| - rest_len
    n = p1 - p2
    dist = wp.length(n)

    if dist > 0.0:
        n = n / dist

    rest_len = segment_length
    C = dist - rest_len

    # XPBD Compliance correction
    # lambda = -C / (sum(w) + alpha / dt^2)
    compliance = alpha / (dt * dt)
    lagrange = -C / (w1 + w2 + compliance)

    correction = n * lagrange

    # 应用位置修正 (原子操作在某些情况下是必要的，但在这里由于链式结构，
    # 若使用 Graph Coloring 会更好。但在高阻尼和子步下，直接写入通常能获得可信的视觉效果)
    # 为了严谨，我们通常不对 p1 (父节点) 施加回弹力的全部影响，或者在这一步仅更新 p2 (Forward Euler 风格)
    # 但标准的 PBD 是双向更新。

    # 简化：由于是毛发，主要关注 p2 被 p1 拉住
    if w1 > 0.0:
        wp.atomic_add(predicted_points, idx1, correction * w1)
    if w2 > 0.0:
        wp.atomic_add(predicted_points, idx2, -correction * w2)


# ------------------------------------------------
# XPBD 弯曲约束 (使用简单的跳点距离约束)
# ------------------------------------------------
# 通过约束 i 和 i+2 之间的距离来模拟弯曲刚度
@wp.kernel
def solve_bend(
        predicted_points: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        segment_length: float,
        alpha: float,
        dt: float,
        points_per_strand: int
):
    tid = wp.tid()
    local_idx = tid % points_per_strand

    # 需要至少隔一个点：i 和 i+2
    if local_idx < 2:
        return

    idx1 = tid - 2
    idx2 = tid

    p1 = predicted_points[idx1]
    p2 = predicted_points[idx2]

    w1 = inv_mass[idx1]
    w2 = inv_mass[idx2]

    if w1 + w2 == 0.0:
        return

    n = p1 - p2
    dist = wp.length(n)
    if dist > 0.0:
        n = n / dist

    # i 和 i+2 的理想距离是 2 * segment_length (笔直状态)
    rest_len = segment_length * 2.0
    C = dist - rest_len

    compliance = alpha / (dt * dt)
    lagrange = -C / (w1 + w2 + compliance)

    correction = n * lagrange

    if w1 > 0.0:
        wp.atomic_add(predicted_points, idx1, correction * w1)
    if w2 > 0.0:
        wp.atomic_add(predicted_points, idx2, -correction * w2)


# ------------------------------------------------
# 碰撞约束 (头部 SDF)
# ------------------------------------------------
@wp.kernel
def solve_collision(
        predicted_points: wp.array(dtype=wp.vec3),
        inv_mass: wp.array(dtype=float),
        head_center: wp.vec3,
        head_radius: float
):
    tid = wp.tid()
    if inv_mass[tid] == 0.0:
        return

    pos = predicted_points[tid]

    # 简单的球体 SDF
    diff = pos - head_center
    dist = wp.length(diff)

    # 如果在球内部
    if dist < head_radius:
        # 投影到表面
        normal = diff / dist
        correction = normal * (head_radius - dist)
        predicted_points[tid] = pos + correction


# ------------------------------------------------
# 更新速度与位置
# ------------------------------------------------
@wp.kernel
def update_state(
        points: wp.array(dtype=wp.vec3),
        predicted_points: wp.array(dtype=wp.vec3),
        velocities: wp.array(dtype=wp.vec3),
        dt: float
):
    tid = wp.tid()

    # XPBD 速度更新: v = (x_new - x_old) / dt
    new_pos = predicted_points[tid]
    old_pos = points[tid]

    velocities[tid] = (new_pos - old_pos) / dt
    points[tid] = new_pos


# ==========================================
# 3. 主程序
# ==========================================

def run_simulation():
    # 1. 内存分配
    points = wp.zeros(total_points, dtype=wp.vec3, device="cuda")
    predicted_points = wp.zeros(total_points, dtype=wp.vec3, device="cuda")
    velocities = wp.zeros(total_points, dtype=wp.vec3, device="cuda")
    inv_mass = wp.zeros(total_points, dtype=float, device="cuda")

    # 2. 初始化场景
    wp.launch(
        kernel=init_hair,
        dim=total_points,
        inputs=[points, velocities, inv_mass, cfg.segment_length, cfg.num_strands, cfg.points_per_strand,
                cfg.head_radius],
        device="cuda"
    )

    # 3. 渲染设置 (USD)
    usd_path = "hair_simulation_xpbd.usd"
    renderer = wp.render.UsdRenderer(usd_path)

    # 创建渲染所需的 shape 数组 (lines)
    # Warp USD 渲染线段需要定义索引，这里我们将每根毛发连成线
    indices_cpu = []
    for s in range(cfg.num_strands):
        base = s * cfg.points_per_strand
        for i in range(cfg.points_per_strand - 1):
            indices_cpu.append(base + i)
            indices_cpu.append(base + i + 1)

    indices = wp.array(indices_cpu, dtype=int, device="cuda")

    print(f"Start simulation: {cfg.num_strands} strands, {total_points} particles.")

    # 4. 仿真循环
    frame_dt = 1.0 / 60.0
    sim_dt = frame_dt / cfg.substeps

    for frame in range(120):  # 渲染 120 帧 (2秒)

        # XPBD 子步循环
        for step in range(cfg.substeps):

            # A. 预测 (Integrate)
            wp.launch(
                kernel=predict_positions,
                dim=total_points,
                inputs=[points, predicted_points, velocities, inv_mass, cfg.gravity, sim_dt, cfg.damping],
                device="cuda"
            )

            # B. 约束求解 (Solve Constraints)
            # 迭代多次以提高刚度收敛性，XPBD 依赖非线性求解
            solve_iters = 2
            for _ in range(solve_iters):
                # 距离约束
                wp.launch(
                    kernel=solve_stretch,
                    dim=total_points,
                    inputs=[predicted_points, inv_mass, cfg.segment_length, cfg.alpha_stretch, sim_dt,
                            cfg.points_per_strand],
                    device="cuda"
                )
                # 弯曲约束
                wp.launch(
                    kernel=solve_bend,
                    dim=total_points,
                    inputs=[predicted_points, inv_mass, cfg.segment_length, cfg.alpha_bend, sim_dt,
                            cfg.points_per_strand],
                    device="cuda"
                )
                # 碰撞约束
                wp.launch(
                    kernel=solve_collision,
                    dim=total_points,
                    inputs=[predicted_points, inv_mass, cfg.head_center, cfg.head_radius],
                    device="cuda"
                )

            # C. 更新状态
            wp.launch(
                kernel=update_state,
                dim=total_points,
                inputs=[points, predicted_points, velocities, sim_dt],
                device="cuda"
            )

        # 渲染当前帧到 USD
        time = frame * frame_dt
        renderer.begin_frame(time)

        # 渲染头部球体参考
        renderer.render_sphere("head", cfg.head_center, wp.quat_identity(), cfg.head_radius, color=(0.8, 0.8, 0.8))

        # 渲染毛发 (Lines)
        renderer.render_line_list(
            name="hair_strands",
            vertices=wp.to_torch(points),
            indices=wp.to_torch(indices),
            color=wp.vec3(0.1, 0.1, 0.1),
            radius=0.02
        )

        renderer.end_frame()

        if frame % 20 == 0:
            print(f"Frame {frame} completed.")

    renderer.save()
    print(f"Simulation finished. Saved to {usd_path}")


if __name__ == "__main__":
    run_simulation()