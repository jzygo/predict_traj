

class DistanceConstraint:
    def __init__(self, point_a, point_b, rest_distance, flexibility=1.0):
        self.point_a = point_a
        self.point_b = point_b
        self.flexibility = flexibility
        self.rest_distance = rest_distance

class AngleConstraint:
    def __init__(self, point_a, point_b, point_c, rest_angle):
        self.point_a = point_a
        self.point_b = point_b
        self.point_c = point_c
        self.rest_angle = rest_angle

class FixedPointConstraint:
    def __init__(self, point):
        self.point = point

class VariableDistanceConstraint:
    def __init__(self, point_a, point_b, rest_distance):
        self.point_a = point_a
        self.point_b = point_b
        self.rest_distance = rest_distance


class BendingConstraint:
    """弯曲约束 - 用于保持毛发直线形态
    
    对于三个连续粒子 a, b, c，约束 a 和 c 之间的距离
    等于 |a-b| + |b-c|（即直线时的距离）
    这比角度约束更加数值稳定
    
    对于毛笔毛发仿真：
    - relative_location 表示约束在毛发上的相对位置 (0.0=根部, 1.0=尖端)
    - 根部 compliance 较小（硬），保持笔头形状稳定
    - 尖端 compliance 较大（软），允许自然弯曲和飘逸效果
    """
    def __init__(self, point_a, point_b, point_c, rest_distance, relative_location=0.0):
        self.point_a = point_a  # 第一个端点
        self.point_b = point_b  # 中间点
        self.point_c = point_c  # 第二个端点
        self.rest_distance = rest_distance  # a到c的直线距离
        self.relative_location = relative_location  # 在毛发上的相对位置 (0.0=根部, 1.0=尖端)

class AlignmentConstraint:
    """对齐约束 - 使非固定粒子趋向于两固定粒子构成的直线或锥形收拢位置
    
    根据粒子是否被压在纸面上，动态选择目标位置：
    - 未被压在纸面：收拢成锥形（使用 cone_target_ratio 和 initial_radial_offset）
    - 被压在纸面：向固定粒子延长线位置移动（使用 rest_ratio）
    """
    def __init__(self, point_fixed_start, point_fixed_end, point_free, rest_ratio, relative_location=0.0, 
                 cone_target_ratio=None, initial_radial_offset=None):
        self.point_fixed_start = point_fixed_start
        self.point_fixed_end = point_fixed_end
        self.point_free = point_free
        self.rest_ratio = rest_ratio  # 延长线目标：free点到fixed_start点的距离比例 (离fixed_start的距离 / |start-end|)
        self.relative_location = relative_location  # 粒子在毛发上的相对位置 (0.0=根部固定区域结束点, 1.0=尖端)
        # 锥形目标：粒子在锥形上的位置比例，None 表示使用 rest_ratio
        self.cone_target_ratio = cone_target_ratio if cone_target_ratio is not None else rest_ratio
        # 初始径向偏移：毛发相对于笔心中心轴的偏移向量 (x, y, z)
        # 用于计算锥形目标位置
        self.initial_radial_offset = initial_radial_offset if initial_radial_offset is not None else (0.0, 0.0, 0.0)


class CollisionConstraint:
    """碰撞约束 - 防止粒子之间相互穿透
    
    当两个粒子之间的距离小于它们的碰撞半径之和时，产生排斥力将它们推开。
    这是一个动态约束，每个时间步都需要重新检测哪些粒子对发生碰撞。
    """
    def __init__(self, collision_radius: float = 0.001, compliance: float = 1e-6):
        """
        Args:
            collision_radius: 每个粒子的碰撞半径，两粒子碰撞距离阈值为 2 * collision_radius
            compliance: 碰撞约束的柔度，值越小约束越硬
        """
        self.collision_radius = collision_radius
        self.compliance = compliance


