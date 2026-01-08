import os
import json
import pickle
import io
import random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from scipy import ndimage


def compute_max_black_radius(binary_img: np.ndarray, points: np.ndarray, max_radius: int = 20) -> np.ndarray:
    """
    计算每个轨迹点处仅包含黑色像素的最大圆半径。
    
    Args:
        binary_img: 二值化图像，黑色像素为 True，白色为 False，shape (H, W)
        points: 轨迹点坐标，shape (N, 2)，以像素为单位 (x, y)
        max_radius: 半径上限
        
    Returns:
        radii: 每个点对应的最大半径，shape (N,)
    """
    H, W = binary_img.shape
    N = len(points)
    radii = np.zeros(N, dtype=np.float32)
    
    # 计算距离变换：每个黑色像素到最近白色像素的距离
    # 首先反转：白色区域为 True，黑色区域为 False
    white_mask = ~binary_img
    # distance_transform_edt 计算每个 False 位置到最近 True 位置的距离
    # 我们需要黑色像素到白色像素的距离，所以用 white_mask
    dist_to_white = ndimage.distance_transform_edt(~white_mask)
    
    for i, (x, y) in enumerate(points):
        # 将坐标四舍五入到最近的整数像素
        px, py = int(round(x)), int(round(y))
        
        # 边界检查
        if px < 0 or px >= W or py < 0 or py >= H:
            radii[i] = 0.0
            continue
        
        # 获取该点到最近白色像素的距离
        r = dist_to_white[py, px]
        radii[i] = min(r, max_radius)
    
    return radii

class FontDataset(Dataset):
    """
    用于加载由 process_font_data.py 生成的字体数据集。
    包含图像数据 (JPG bytes) 和 笔画数据 (归一化坐标)。
    支持多字体加载及训练集/验证集划分。
    """
    def __init__(self, data_root, font_names=None, split='all', val_ratio=0.1, target_size=(256, 256), seed=42, max_radius=20, num_points_per_stroke=64):
        """
        Args:
            data_root (str): 数据集根目录
            font_names (list or str, optional): 字体名称列表。如果为 None，则扫描 data_root 下的所有文件夹。
            split (str): 'all', 'train', 或 'val'。
            val_ratio (float): 验证集比例。
            transform (callable, optional): 应用于图像的 transform
            target_size (tuple): 图像调整大小的目标尺寸 (width, height)
            seed (int): 随机种子，用于确保划分的一致性。
            max_radius (int): 计算 r 时的半径上限，默认 20 像素
            num_points_per_stroke (int): 每个笔画重采样的点数，默认 64
        """
        self.data_root = data_root
        self.target_size = target_size
        self.max_radius = max_radius
        self.num_points_per_stroke = num_points_per_stroke
        
        # 确定要加载的字体列表
        if font_names is None:
            # 扫描目录下所有可能的字体文件夹
            if os.path.exists(data_root):
                candidates = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
                font_names = []
                for d in candidates:
                    # 简单的检查：是否存在对应的 pkl 文件
                    # if os.path.exists(os.path.join(data_root, d, f"FZSJSJW_strokes.pkl")) and \
                    if d in ["FZBYSJW", "FZJZJW", "FZKATJW", "FZPSHJW", "FZShangKJW", "FZShouHJW", "FZYTJW", "FZYTK", "FZZDXJW", "FZZHYJW"]:
                        continue
                    if os.path.exists(os.path.join(data_root, d, f"{d}_strokes.pkl")) and \
                       os.path.exists(os.path.join(data_root, d, f"{d}_imgs.pkl")):
                        font_names.append(d)
                font_names.sort() # 排序以确保顺序一致
            else:
                font_names = []
                print(f"Warning: data_root {data_root} does not exist.")
        elif isinstance(font_names, str):
            font_names = [font_names]
            
        self.font_names = font_names
        self.samples = [] # list of (font_name, key)
        self.font_data = {} # {font_name: {'images': ..., 'strokes': ...}}
        
        print(f"Initializing FontDataset with split='{split}', val_ratio={val_ratio}, fonts={len(self.font_names)}")
        
        for font_name in self.font_names:
            font_dir = os.path.join(data_root, font_name)
            strokes_path = os.path.join(font_dir, f"{font_name}_strokes.pkl")
            pkl_path = os.path.join(font_dir, f"{font_name}_imgs.pkl")
            
            try:
                # 加载笔画数据
                with open(strokes_path, 'rb') as f:
                    strokes_data = pickle.load(f)
                # 加载图像数据
                with open(pkl_path, 'rb') as f:
                    images_data = pickle.load(f)
            except Exception as e:
                print(f"Error loading font {font_name}: {e}")
                continue
                
            # 确保键值对齐
            keys = sorted(list(set(strokes_data.keys()) & set(images_data.keys())))
            
            # 划分训练集和验证集
            if split != 'all':
                # 使用局部 Random 实例以避免影响全局状态，并确保可复现性
                rng = random.Random(seed)
                rng.shuffle(keys)
                
                split_idx = int(len(keys) * (1 - val_ratio))
                if split == 'train':
                    keys = keys[:split_idx]
                elif split == 'val':
                    keys = keys[split_idx:]
            
            # 存储数据
            self.font_data[font_name] = {
                'images': images_data,
                'strokes': strokes_data
            }
            
            
            for k in keys:
                if len(self.font_data[font_name]['strokes'][k]) > 0:
                    self.samples.append((font_name, k))
                    

        print(f"Successfully loaded {len(self.samples)} samples from {len(self.font_data)} fonts.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        font_name, key = self.samples[idx]
        
        # 获取数据
        img_bytes = self.font_data[font_name]['images'][key]
        strokes = self.font_data[font_name]['strokes'][key]
        
        # 处理图像
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        w, h = image.size
        max_dim = max(w, h)
        if w != h:
            new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
            new_img.paste(image, (0, 0))
            image = new_img

        # 默认转换: Resize + ToTensor
        default_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(), # 归一化到 [0, 1]
        ])
        image_tensor = default_transform(image)
        
        # 将图像 resize 后用于计算 r
        resized_image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        resized_np = np.array(resized_image)
        
        # 二值化：将 RGB 转灰度，然后阈值分割
        # 灰度 < 128 为黑色 (True)，否则为白色 (False)
        gray = np.mean(resized_np, axis=2)
        binary_img = gray < 128  # 黑色像素为 True
        
        # 计算从原始坐标到 target_size 的缩放比例
        scale = self.target_size[0] / max_dim  # 假设 target_size 是正方形
        
        # 处理笔画：先重采样 (x, y)，再计算 r
        strokes_tensors = []
        for s in strokes:
            stroke_np = np.array(s, dtype=np.float32)
            
            if len(stroke_np) == 0:
                # 空笔画：返回全零张量
                strokes_tensors.append(torch.zeros(self.num_points_per_stroke, 3, dtype=torch.float32))
                continue
            
            # 步骤1：先对 (x, y) 坐标进行重采样
            # 使用 resample_stroke 函数，只处理 2D 坐标
            from .traj_utils import resample_stroke
            resampled_xy = resample_stroke(stroke_np[:, :2], num_points=self.num_points_per_stroke, point_dim=2)
            resampled_xy = resampled_xy.numpy()  # shape: (num_points_per_stroke, 2)
            
            # 步骤2：将重采样后的归一化坐标 (原始像素坐标 / max_dim) 转换为 target_size 像素坐标
            # resampled_xy 是原始像素坐标，需要映射到 target_size
            scaled_coords = resampled_xy * scale  # 映射到 target_size 像素坐标
            
            # 步骤3：根据重采样后的坐标计算每个点的最大黑色圆半径
            radii = compute_max_black_radius(binary_img, scaled_coords, self.max_radius)
            
            # 步骤4：将坐标归一化到 [-1, 1]
            normalized_coords = resampled_xy / max_dim * 2.0 - 1.0
            
            # 步骤5：将半径归一化到 [-1, 1]，r 上限为 max_radius
            normalized_radii = radii / self.max_radius * 2.0 - 1.0
            
            # 合并为 (num_points_per_stroke, 3) 的张量：(x, y, r)
            stroke_with_r = np.column_stack([normalized_coords, normalized_radii])
            strokes_tensors.append(torch.tensor(stroke_with_r, dtype=torch.float32))

        return {
            'key': key,
            'font_name': font_name,
            'image': image_tensor,
            'strokes': strokes_tensors
        }

def visualize_sample(dataset, idx=None):
    """
    可视化数据集中的一个样本：将图像和对应的笔画轨迹叠加显示。
    支持显示 r 维度（用圆圈半径表示）。
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    for i in range(len(dataset)):
        sample = dataset[i]
        if sample['key'] == 'GB45':
            print("Successfully found sample with key 'GB45'")
            break
    key = sample['key']
    font_name = sample.get('font_name', 'Unknown')
    image = sample['image']
    strokes = sample['strokes']
    
    print(f"Visualizing sample index: {idx}, Font: {font_name}, Key: {key}")
    print(f"Number of strokes: {len(strokes)}")
    
    plt.figure(figsize=(10, 10))
    
    # 如果是 Tensor (C, H, W)，转换为 numpy (H, W, C)
    if isinstance(image, torch.Tensor):
        disp_img = image.permute(1, 2, 0).numpy()
        # 如果做了 Normalize，这里可能需要反归一化，这里假设只是 ToTensor [0, 1]
        disp_img = np.clip(disp_img, 0, 1)
    else:
        disp_img = image
    
    # 显示图像作为背景
    # extent 参数将图像映射到 [-1, 1] 的坐标范围，与笔画轨迹对齐
    plt.imshow(disp_img, extent=[-1, 1, -1, 1], alpha=0.6)  # alpha 控制图像透明度
    
    # 绘制边框 [-1, 1]
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.5, linewidth=2)
    
    colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))
    
    # 绘制笔画轨迹
    for i, stroke in enumerate(strokes):
        if isinstance(stroke, torch.Tensor):
            stroke = stroke.numpy()
        else:
            stroke = np.array(stroke)
            
        if len(stroke) == 0:
            continue
        
        # 检查是否有 r 维度
        has_r = stroke.shape[1] >= 3
        
        # 绘制笔画路径，加粗线条
        plt.plot(stroke[:, 0], -stroke[:, 1], 
                label=f'Stroke {i+1}', 
                color=colors[i], 
                linewidth=3)  # 加粗线条
        
        # 标出起点（更大更明显）
        plt.scatter(stroke[0, 0], -stroke[0, 1], 
                   color=colors[i], s=50, marker='o', 
                   edgecolors='white', linewidths=2, zorder=5)
        
        # 标出过程点
        plt.scatter(stroke[:, 0], -stroke[:, 1], 
                   color=colors[i], s=5, zorder=4)
        
        # 如果有 r 维度，用圆圈表示半径
        if has_r:
            max_radius_normalized = dataset.max_radius / (dataset.target_size[0] / 2)  # 转换为 [-1, 1] 坐标系
            for j in range(0, len(stroke), max(1, len(stroke) // 5)):  # 每隔几个点画一个圆
                # r 从 [-1, 1] 反归一化到像素值，再转换到 [-1, 1] 坐标系
                r_normalized = (stroke[j, 2] + 1) / 2 * max_radius_normalized
                if r_normalized > 0.01:  # 只画有意义的半径
                    circle = plt.Circle((stroke[j, 0], -stroke[j, 1]), r_normalized, 
                                        fill=False, color=colors[i], alpha=0.5, linewidth=1)
                    plt.gca().add_patch(circle)

    plt.title(f"Overlayed: {key} ({font_name})", fontsize=14, fontweight='bold')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True, alpha=0.3)
    plt.axis('equal')  # 保持纵横比
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("visual_data.png", dpi=150, bbox_inches='tight')
    plt.show()

def custom_collate_fn(batch):
    """
    自定义 collate_fn，用于处理变长的笔画数据。
    """
    keys = [item['key'] for item in batch]
    font_names = [item.get('font_name', '') for item in batch]
    images = torch.stack([item['image'] for item in batch])
    
    # 笔画是变长的，无法直接 stack，保持列表形式
    strokes = [item['strokes'] for item in batch]
    
    return {
        'keys': keys,
        'font_names': font_names,
        'images': images,
        'strokes': strokes
    }

if __name__ == "__main__":
    # 配置路径 (请根据实际情况修改)
    # 假设数据在 /data1/jizy/font_out，或者当前目录下的 data/font_out
    # 这里使用一个示例路径，用户运行前需要修改
    DATA_ROOT = r'/data1/jizy/font_out' 
    
    # 检查路径是否存在，如果不存在则提示用户
    if not os.path.exists(DATA_ROOT):
        print(f"Warning: Path {DATA_ROOT} does not exist.")
        print("Please edit DATA_ROOT in the script to point to your data.")
        # 为了演示，我们创建一个 dummy dataset 或者退出
        # exit()
    else:
        # 1. 创建数据集 (自动扫描所有字体)
        dataset = FontDataset(data_root=DATA_ROOT, split='train')
        
        # 2. 验证可视化
        if len(dataset) > 0:
            print("Visualizing a random sample...")
            visualize_sample(dataset)
            
            # 3. 测试 DataLoader
            dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=custom_collate_fn)
            
            print("\nTesting DataLoader iteration...")
            for batch in dataloader:
                print(f"Batch keys: {batch['keys']}")
                print(f"Batch fonts: {batch['font_names']}")
                print(f"Batch images shape: {batch['images'].shape}")
                print(f"Batch strokes length: {len(batch['strokes'])}")
                break
