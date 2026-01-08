import torch


def check_joint_limits(pt_file_path):
    """
    读取pt文件并校验Franka机械臂关节限制
    """

    # 1. 定义关节限制 (根据图片硬编码)
    # 单位: rad
    # 对应关节: [1, 2, 3, 4, 5, 6, 7]
    q_max_limits = torch.tensor([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
    q_min_limits = torch.tensor([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])

    print(f"正在加载文件: {pt_file_path} ...")

    try:
        # 2. 读取 PT 文件
        # 假设数据形状为 (N, 9) 或类似的序列结构
        data = torch.load(pt_file_path, map_location='cpu')

        # 检查数据维度
        if data.dim() != 2 or data.shape[1] < 7:
            print(f"错误: 数据形状不符合预期。期望形状为 (N, >=7)，实际形状为 {data.shape}")
            return False

        # 3. 数据预处理
        # "舍弃最后两个维度" -> 取前7列
        # 假设输入是 (N, 9)，我们需要的是 (N, 7)
        joint_positions = data[:, :7]

        num_frames = joint_positions.shape[0]
        print(f"检测到 {num_frames} 帧数据，正在进行校验...")

        # 4. 向量化比较 (利用广播机制)
        # 检查是否小于最小值
        violation_min = joint_positions < q_min_limits
        # 检查是否大于最大值
        violation_max = joint_positions > q_max_limits

        # 合并违规情况
        any_violations = violation_min | violation_max

        # 5. 输出结果
        if not any_violations.any():
            print("\n✅ 校验通过：所有关节数据均在限制范围内。")
            return True
        else:
            print("\n❌ 校验失败：发现关节数据超出限制！")

            # 统计违规详情
            total_violations = any_violations.sum().item()
            print(f"总共有 {total_violations} 个数据点违规。")

            # 找出具体是哪些帧、哪些关节违规
            # 获取非零元素的索引 (row_idx, joint_idx)
            viol_indices = torch.nonzero(any_violations, as_tuple=False)

            print("\n--- 违规详情示例 (前 5 个) ---")
            print(f"{'帧序号':<10} {'关节序号(1-7)':<15} {'当前值':<12} {'范围':<20} {'类型'}")

            count = 0
            for idx in viol_indices:

                frame_i, joint_i = idx[0].item(), idx[1].item()
                val = joint_positions[frame_i, joint_i].item()

                # 确定是超上限还是下限
                limit_min = q_min_limits[joint_i].item()
                limit_max = q_max_limits[joint_i].item()

                v_type = "低于下限" if val < limit_min else "高于上限"

                print(f"{frame_i} {joint_i + 1} {val:.4f} [{limit_min:.4f}, {limit_max:.4f}]   {v_type}")
                count += 1

            return False

    except Exception as e:
        print(f"处理文件时发生错误: {e}")
        return False


# ==========================================
# 测试代码 (生成一个假的 .pt 文件来演示)
# ==========================================
if __name__ == "__main__":

    test_filename = "../src/simulation/stroke_visualizations/joint_trajectory.pt"

    # 2. 运行校验函数
    check_joint_limits(test_filename)