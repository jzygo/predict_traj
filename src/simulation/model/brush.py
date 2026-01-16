from random import random

import torch

from simulation.model.constraints import DistanceConstraint, FixedPointConstraint, AngleConstraint, \
    VariableDistanceConstraint, BendingConstraint, AlignmentConstraint


def rotation_matrix_from_vectors(vec1, vec2):
    """ 计算两个向量之间的旋转矩阵 """
    u = vec1 / torch.norm(vec1)
    v = vec2 / torch.norm(vec2)

    if torch.allclose(u, v, atol=1e-6):
        return torch.eye(3, dtype=torch.float64)
    if torch.allclose(u, -v, atol=1e-6):
        # 180度旋转，选取任意垂直轴
        if torch.abs(u[0]) > 0.9:
            temp = torch.tensor([0, 1, 0], dtype=torch.float64)
        else:
            temp = torch.tensor([1, 0, 0], dtype=torch.float64)
        axis = torch.cross(u, temp)
        axis = axis / torch.norm(axis)
        # Rodrigues for 180
        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=torch.float64)
        return torch.eye(3, dtype=torch.float64) + 2 * (K @ K)

    # Standard formula
    w = torch.cross(u, v)
    c = torch.dot(u, v)
    K = torch.tensor([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=torch.float64)

    # R = I + K + K^2 * (1 - c) / s^2  => 1/(1+c)
    R = torch.eye(3, dtype=torch.float64) + K + (K @ K) * (1 / (1 + c))
    return R


class Hair:
    def __init__(self, length, flexibility, thickness, root_position, tangent_vector):
        self.length = length
        self.flexibility = flexibility
        self.thickness = thickness
        self.root_position = root_position.clone().detach()
        self.tangent_vector = tangent_vector.clone().detach()
        self.tangent_vector = self.tangent_vector / torch.norm(self.tangent_vector)  # Normalize the tangent vector
        self.particles = []
        self.constraints = []

    def gen_particles(self, num_particles):
        for i in range(num_particles):
            position = self.root_position + (self.tangent_vector * (i / num_particles) * self.length)
            velocity = self.tangent_vector * self.flexibility
            particle = Particle(position, velocity)
            self.particles.append(particle)

    def gen_constraints(self):
        # 固定顶部约 10% 的粒子（至少固定 1 个，即根粒子）
        fixed_count = max(1, int(torch.ceil(torch.tensor(0.2 * len(self.particles), dtype=torch.float64)).item()))

        for i in range(len(self.particles) - 1):
            point_a = self.particles[i]
            point_b = self.particles[i + 1]
            rest_distance = torch.norm(point_b.position - point_a.position).item()
            dis_constraint = DistanceConstraint(point_a, point_b, rest_distance, self.flexibility)
            self.constraints.append(dis_constraint)
            if i + 2 < len(self.particles):
                # 使用弯曲距离约束代替角度约束，更加数值稳定
                # 休息距离 = |a-b| + |b-c|，即直线时 a 到 c 的距离
                point_c = self.particles[i + 2]
                dist_ab = torch.norm(point_b.position - point_a.position).item()
                dist_bc = torch.norm(point_c.position - point_b.position).item()
                rest_dist_ac = dist_ab + dist_bc
                bending_constraint = BendingConstraint(
                    self.particles[i],
                    self.particles[i + 1],
                    self.particles[i + 2],
                    rest_distance=rest_dist_ac
                )
                self.constraints.append(bending_constraint)

    def apply_displacement(self, displacement):
        displacement = torch.tensor(displacement, dtype=torch.float64) if not isinstance(displacement,
                                                                                         torch.Tensor) else displacement
        self.root_position += displacement
        for particle in self.particles:
            particle.position += displacement

    def apply_rotation(self, rotation_matrix, pivot_point=None):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64) if not isinstance(rotation_matrix,
                                                                                               torch.Tensor) else rotation_matrix

        # 如果没有指定旋转中心，默认不移动 root_position (或认为只旋转方向)
        # 但如果有 particles，通常需要围绕某个点旋转
        # 兼容旧逻辑：如果不传 pivot，只转 tangent (旧代码没有转 particles，可能有 bug 或用意是不转位置)
        # 这里我们总是更新 tangent
        self.tangent_vector = rotation_matrix @ self.tangent_vector

        if pivot_point is not None:
            pivot_point = torch.tensor(pivot_point, dtype=torch.float64) if not isinstance(pivot_point,
                                                                                           torch.Tensor) else pivot_point

            # 旋转 root_position
            # new_pos = pivot + R @ (old_pos - pivot)
            self.root_position = pivot_point + rotation_matrix @ (self.root_position - pivot_point)

            # 旋转所有 particles
            for particle in self.particles:
                particle.position = pivot_point + rotation_matrix @ (particle.position - pivot_point)
                particle.velocity = rotation_matrix @ particle.velocity


class Particle:
    def __init__(self, position, velocity):
        self.position = torch.tensor(position, dtype=torch.float64) if not isinstance(position,
                                                                                      torch.Tensor) else position.clone()
        self.velocity = torch.tensor(velocity, dtype=torch.float64) if not isinstance(velocity,
                                                                                      torch.Tensor) else velocity.clone()
        self.particle_id = -1


class Brush:
    def __init__(self, radius, max_length, max_hairs, max_particles_per_hair, thickness, root_position=None,
                 length_ratio=5 / 6, tangent_vector=None, normal_vector=None):
        self.radius = radius
        self.max_length = max_length
        self.length_ratio = length_ratio
        self.max_hairs = max_hairs
        self.max_particles_per_hair = max_particles_per_hair
        self.thickness = thickness

        self.hairs = []
        self.particles = []
        self.constraints = []

        if root_position is None:
            root_position = torch.tensor([0, 0, max_length], dtype=torch.float64)
        if tangent_vector is None:
            tangent_vector = torch.tensor([0, 0, -1], dtype=torch.float64)
        if normal_vector is None:
            normal_vector = torch.tensor([1, 0, 0], dtype=torch.float64)

        self.root_position = torch.tensor(root_position, dtype=torch.float64) if not isinstance(root_position,
                                                                                                torch.Tensor) else root_position.clone().to(
            dtype=torch.float64)
        self.tangent_vector = torch.tensor(tangent_vector, dtype=torch.float64) if not isinstance(tangent_vector,
                                                                                                  torch.Tensor) else tangent_vector.clone().to(
            dtype=torch.float64)
        self.normal_vector = torch.tensor(normal_vector, dtype=torch.float64) if not isinstance(normal_vector,
                                                                                                torch.Tensor) else normal_vector.clone().to(
            dtype=torch.float64)

    def gen_hairs(self):
        target_tangent = self.tangent_vector.clone()
        vertical_tangent = torch.tensor([0., 0., -1.], dtype=torch.float64)

        # 暂时将 tangent 设为竖直向下，以便在此坐标系下生成毛发
        self.tangent_vector = vertical_tangent

        for i in range(self.max_hairs):
            current_radius = random() * self.radius
            current_angle = random() * 2 * torch.pi
            x = current_radius * torch.cos(torch.tensor(current_angle, dtype=torch.float64)).item()
            y = current_radius * torch.sin(torch.tensor(current_angle, dtype=torch.float64)).item()
            length = self.max_length * (1 - (current_radius / self.radius)) * self.length_ratio + self.max_length * (
                        1 - self.length_ratio)

            flexibility = (1 - (current_radius / self.radius)) * 0.5 + 0.5
            hair = Hair(length, flexibility, self.thickness,
                        self.root_position + torch.tensor([x, y, 0], dtype=torch.float64),
                        self.tangent_vector)
            hair.gen_particles(self.max_particles_per_hair)
            fixed_particles = []
            for particle in hair.particles:
                dis = torch.norm(particle.position - hair.root_position).item()
                if dis < 1 / 10 * self.max_length:
                    self.constraints.append(FixedPointConstraint(particle))
                    fixed_particles.append(particle)

            if len(fixed_particles) >= 2:
                p_start = fixed_particles[0]
                p_end = fixed_particles[-1]
                dist_start_end = torch.norm(p_end.position - p_start.position).item() + 1e-6

                # Identify non-fixed particles that follow the fixed ones
                # Assuming hair.particles is ordered from root to tip
                start_idx = hair.particles.index(p_end) + 1
                num_free_particles = len(hair.particles) - start_idx
                
                for k in range(start_idx, len(hair.particles)):
                    dist_start_free = torch.norm(hair.particles[k].position - p_start.position).item()
                    ratio = dist_start_free / dist_start_end
                    
                    # Calculate relative location (0.0 at root-end/free-start, 1.0 at tip)
                    if num_free_particles > 1:
                        relative_loc = (k - start_idx) / (num_free_particles - 1)
                    else:
                        relative_loc = 1.0
                        
                    self.constraints.append(AlignmentConstraint(p_start, p_end, hair.particles[k], ratio, relative_loc))

            self.hairs.append(hair)
            hair.gen_constraints()
            self.particles.extend(hair.particles)
            self.constraints.extend(hair.constraints)

        for i in range(len(self.particles)):
            self.particles[i].particle_id = i

        # 如果目标方向不是竖直向下，则旋转
        if not torch.allclose(target_tangent, vertical_tangent):
            rot_mat = rotation_matrix_from_vectors(vertical_tangent, target_tangent)
            self.apply_rotation(rot_mat)

            # 确保切向量完全对齐（消除浮点误差）
            self.tangent_vector = target_tangent

    def gen_constraints(self):
        """Generate VariableDistanceConstraints across hairs using Delaunay triangulation.
        
        基于毛发第一层粒子的Delaunay三角剖分构造约束：
        - 取每根毛发的第一层粒子
        - 投影到垂直于切向量的局部平面进行三角剖分
        - 对于三角形中相邻的两个顶点（毛发），从第一层开始逐层添加VariableDistanceConstraint
        - 直到其中一根毛发没有该层的粒子为止
        """
        from scipy.spatial import Delaunay

        hairs = self.hairs
        n = len(hairs)
        if n < 3:  # Delaunay需要至少3个点
            return

        # 构造局部坐标系用于投影
        # z轴为切向量，x轴为法向量 (如果它们不垂直，需Schmidt正交化)
        z_axis = self.tangent_vector / torch.norm(self.tangent_vector)
        x_axis = self.normal_vector.clone()
        # 确保 x_axis 垂直于 z_axis
        x_axis = x_axis - torch.dot(x_axis, z_axis) * z_axis
        if torch.norm(x_axis) < 1e-6:
            # 如果法向量平行于切向量（异常），随便找一个垂直轴
            if torch.abs(z_axis[0]) < 0.9:
                x_axis = torch.tensor([1, 0, 0], dtype=torch.float64)
            else:
                x_axis = torch.tensor([0, 1, 0], dtype=torch.float64)
            x_axis = torch.cross(x_axis, z_axis)
        x_axis = x_axis / torch.norm(x_axis)
        y_axis = torch.cross(z_axis, x_axis)

        # 提取每根毛发第一层粒子的局部xy坐标
        first_layer_points = []
        for hair in hairs:
            if len(hair.particles) > 0:
                first_particle = hair.particles[0]
                # 计算相对于 Brush root 的向量
                rel_pos = first_particle.position - self.root_position
                # 投影到局部平面的 xy
                x_val = torch.dot(rel_pos, x_axis).item()
                y_val = torch.dot(rel_pos, y_axis).item()
                first_layer_points.append([x_val, y_val])

        if len(first_layer_points) < 3:
            return

        # 转换为torch tensor，但为了兼容Delaunay仍需要numpy
        points_tensor = torch.tensor(first_layer_points, dtype=torch.float64)
        points = points_tensor.numpy()  # 暂时转为numpy用于Delaunay，但主要存储使用tensor

        # 执行Delaunay三角剖分
        try:
            tri = Delaunay(points)
        except:
            # 如果三角剖分失败，回退到原始逻辑
            return

        # 获取所有三角形的边
        edges = set()
        for simplex in tri.simplices:
            # 每个三角形有3条边
            for i in range(3):
                for j in range(i + 1, 3):
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)

        # 为每条边上的毛发对添加逐层约束
        for hair_i, hair_j in edges:
            if hair_i >= len(hairs) or hair_j >= len(hairs):
                continue

            hair_a = hairs[hair_i]
            hair_b = hairs[hair_j]

            # 逐层添加约束，直到其中一根毛发没有该层粒子
            max_layers = min(len(hair_a.particles), len(hair_b.particles))

            for layer in range(max_layers):
                # 50% 概率跳过，增加随机性
                if random() < 0.8:
                    continue
                particle_a = hair_a.particles[layer]
                particle_b = hair_b.particles[layer]
                rest_dist = torch.norm(particle_b.position - particle_a.position).item()
                self.constraints.append(VariableDistanceConstraint(particle_a, particle_b, rest_dist))

    def apply_displacement(self, displacement):
        displacement = torch.tensor(displacement, dtype=torch.float64) if not isinstance(displacement,
                                                                                         torch.Tensor) else displacement
        self.root_position += displacement
        for hair in self.hairs:
            hair.apply_displacement(displacement)

    def apply_rotation(self, rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64) if not isinstance(rotation_matrix,
                                                                                               torch.Tensor) else rotation_matrix
        for hair in self.hairs:
            # 使用 brush.root_position 作为旋转中心
            hair.apply_rotation(rotation_matrix, pivot_point=self.root_position)
        self.tangent_vector = rotation_matrix @ self.tangent_vector
        self.normal_vector = rotation_matrix @ self.normal_vector


if __name__ == "__main__":
    # Test code
    b = Brush(radius=1.0, max_length=1.0, max_hairs=10, max_particles_per_hair=5, thickness=0.1,
              tangent_vector=torch.tensor([1.0, 1.0, 0.0]))
    b.gen_hairs()  # Should rotate to diagonal
    b.gen_constraints()  # Should work without error
    print("Particles:", len(b.particles))
    print("Constraints:", len(b.constraints))
    # Check direction
    dirs = []
    for h in b.hairs:
        if len(h.particles) > 1:
            # check direction of particles
            v = h.particles[-1].position - h.particles[0].position
            v = v / (torch.norm(v) + 1e-6)
            dirs.append(v)
    if dirs:
        avg_dir = torch.stack(dirs).mean(dim=0)
        avg_dir = avg_dir / torch.norm(avg_dir)
        print("Avg dir:", avg_dir)
        target_dir = b.tangent_vector / torch.norm(b.tangent_vector)
        print("Target:", target_dir)
        if torch.dot(avg_dir, target_dir) > 0.9:
            print("Direction Check: PASS")
        else:
            print("Direction Check: FAIL")
    else:
        print("No particles to check direction")
