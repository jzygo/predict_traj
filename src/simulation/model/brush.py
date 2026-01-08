from random import random

import torch

from simulation.model.constraints import DistanceConstraint, FixedPointConstraint, AngleConstraint, VariableDistanceConstraint, BendingConstraint


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
        displacement = torch.tensor(displacement, dtype=torch.float64) if not isinstance(displacement, torch.Tensor) else displacement
        self.root_position += displacement
        for particle in self.particles:
            particle.position += displacement
    
    def apply_rotation(self, rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64) if not isinstance(rotation_matrix, torch.Tensor) else rotation_matrix
        self.tangent_vector = rotation_matrix @ self.tangent_vector


class Particle:
    def __init__(self, position, velocity):
        self.position = torch.tensor(position, dtype=torch.float64) if not isinstance(position, torch.Tensor) else position.clone()
        self.velocity = torch.tensor(velocity, dtype=torch.float64) if not isinstance(velocity, torch.Tensor) else velocity.clone()
        self.particle_id = -1


class Brush:
    def __init__(self, radius, max_length, max_hairs, max_particles_per_hair, thickness, root_position=None, length_ratio=5/6, tangent_vector=None, normal_vector=None):
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

        self.root_position = torch.tensor(root_position, dtype=torch.float64) if not isinstance(root_position, torch.Tensor) else root_position.clone()
        self.tangent_vector = torch.tensor(tangent_vector, dtype=torch.float64) if not isinstance(tangent_vector, torch.Tensor) else tangent_vector.clone()
        self.normal_vector = torch.tensor(normal_vector, dtype=torch.float64) if not isinstance(normal_vector, torch.Tensor) else normal_vector.clone()

    def gen_hairs(self):
        for i in range(self.max_hairs):
            current_radius = random() * self.radius
            current_angle = random() * 2 * torch.pi
            x = current_radius * torch.cos(torch.tensor(current_angle, dtype=torch.float64)).item()
            y = current_radius * torch.sin(torch.tensor(current_angle, dtype=torch.float64)).item()
            length = self.max_length * (1 - (current_radius / self.radius)) * self.length_ratio + self.max_length * (1 - self.length_ratio)

            flexibility = (1 - (current_radius / self.radius)) * 0.5 + 0.5
            hair = Hair(length, flexibility, self.thickness, 
                        self.root_position + torch.tensor([x, y, 0], dtype=torch.float64),
                        self.tangent_vector)
            hair.gen_particles(self.max_particles_per_hair)
            for particle in hair.particles:
                dis = torch.norm(particle.position - hair.root_position).item()
                if dis < 1/10 * self.max_length:
                    self.constraints.append(FixedPointConstraint(particle))
            self.hairs.append(hair)
            hair.gen_constraints()
            self.particles.extend(hair.particles)
            self.constraints.extend(hair.constraints)
        
        for i in range(len(self.particles)):
            self.particles[i].particle_id = i

    def gen_constraints(self):
        """Generate VariableDistanceConstraints across hairs using Delaunay triangulation.
        
        基于毛发第一层粒子的Delaunay三角剖分构造约束：
        - 取每根毛发的第一层粒子（z坐标一致，xy坐标不同）
        - 基于xy坐标进行Delaunay三角剖分
        - 对于三角形中相邻的两个顶点（毛发），从第一层开始逐层添加VariableDistanceConstraint
        - 直到其中一根毛发没有该层的粒子为止
        """
        from scipy.spatial import Delaunay
        
        hairs = self.hairs
        n = len(hairs)
        if n < 3:  # Delaunay需要至少3个点
            return
            
        # 提取每根毛发第一层粒子的xy坐标
        first_layer_points = []
        for hair in hairs:
            if len(hair.particles) > 0:
                first_particle = hair.particles[0]
                # 只使用xy坐标进行三角剖分
                first_layer_points.append([first_particle.position[0].item(), first_particle.position[1].item()])
            
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
        displacement = torch.tensor(displacement, dtype=torch.float64) if not isinstance(displacement, torch.Tensor) else displacement
        self.root_position += displacement
        for hair in self.hairs:
            hair.apply_displacement(displacement)
    
    def apply_rotation(self, rotation_matrix):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=torch.float64) if not isinstance(rotation_matrix, torch.Tensor) else rotation_matrix
        for hair in self.hairs:
            root_to_hair = hair.root_position - self.root_position
            root_to_hair = rotation_matrix @ root_to_hair
            hair.root_position = self.root_position + root_to_hair
            hair.apply_rotation(rotation_matrix)
        self.tangent_vector = rotation_matrix @ self.tangent_vector
        self.normal_vector = rotation_matrix @ self.normal_vector

