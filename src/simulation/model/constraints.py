

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
    """
    def __init__(self, point_a, point_b, point_c, rest_distance):
        self.point_a = point_a  # 第一个端点
        self.point_b = point_b  # 中间点
        self.point_c = point_c  # 第二个端点
        self.rest_distance = rest_distance  # a到c的直线距离

class AlignmentConstraint:
    """对齐约束 - 使非固定粒子趋向于两固定粒子构成的直线
    """
    def __init__(self, point_fixed_start, point_fixed_end, point_free, rest_ratio, relative_location=0.0):
        self.point_fixed_start = point_fixed_start
        self.point_fixed_end = point_fixed_end
        self.point_free = point_free
        self.rest_ratio = rest_ratio # 初始状态下, free点到fixed_start点的距离比例 (离fixed_start的距离 / |start-end|)
        self.relative_location = relative_location  # 粒子在毛发上的相对位置 (0.0=根部固定区域结束点, 1.0=尖端)


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


