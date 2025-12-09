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

class FontDataset(Dataset):
    """
    用于加载由 process_font_data.py 生成的字体数据集。
    包含图像数据 (JPG bytes) 和 笔画数据 (归一化坐标)。
    支持多字体加载及训练集/验证集划分。
    """
    def __init__(self, data_root, font_names=None, split='all', val_ratio=0.1, transform=None, target_size=(256, 256), seed=42):
        """
        Args:
            data_root (str): 数据集根目录
            font_names (list or str, optional): 字体名称列表。如果为 None，则扫描 data_root 下的所有文件夹。
            split (str): 'all', 'train', 或 'val'。
            val_ratio (float): 验证集比例。
            transform (callable, optional): 应用于图像的 transform
            target_size (tuple): 图像调整大小的目标尺寸 (width, height)
            seed (int): 随机种子，用于确保划分的一致性。
        """
        self.data_root = data_root
        self.transform = transform
        self.target_size = target_size
        
        # 确定要加载的字体列表
        if font_names is None:
            # 扫描目录下所有可能的字体文件夹
            if os.path.exists(data_root):
                candidates = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
                font_names = []
                for d in candidates:
                    # 简单的检查：是否存在对应的 pkl 文件
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
        
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换: Resize + ToTensor
            default_transform = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(), # 归一化到 [0, 1]
            ])
            image = default_transform(image)
            
        # 处理笔画
        strokes_tensors = [torch.tensor(s, dtype=torch.float32) for s in strokes]
        
        return {
            'key': key,
            'font_name': font_name,
            'image': image,
            'strokes': strokes_tensors
        }

def visualize_sample(dataset, idx=None):
    """
    可视化数据集中的一个样本：显示图像和对应的笔画轨迹。
    """
    if idx is None:
        idx = random.randint(0, len(dataset) - 1)
    
    sample = dataset[idx]
    key = sample['key']
    font_name = sample.get('font_name', 'Unknown')
    image = sample['image']
    strokes = sample['strokes']
    
    print(f"Visualizing sample index: {idx}, Font: {font_name}, Key: {key}")
    print(f"Number of strokes: {len(strokes)}")
    
    plt.figure(figsize=(12, 6))
    
    # 1. 显示图像
    plt.subplot(1, 2, 1)
    # 如果是 Tensor (C, H, W)，转换为 numpy (H, W, C)
    if isinstance(image, torch.Tensor):
        disp_img = image.permute(1, 2, 0).numpy()
        # 如果做了 Normalize，这里可能需要反归一化，这里假设只是 ToTensor [0, 1]
        disp_img = np.clip(disp_img, 0, 1)
    else:
        disp_img = image
        
    plt.imshow(disp_img)
    plt.title(f"Image: {key} ({font_name})")
    plt.axis('off')
    
    # 2. 显示笔画
    plt.subplot(1, 2, 2)
    
    # 绘制边框 [-1, 1]
    plt.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], 'k--', alpha=0.5)
    
    colors = plt.cm.jet(np.linspace(0, 1, len(strokes)))
    
    for i, stroke in enumerate(strokes):
        if isinstance(stroke, torch.Tensor):
            stroke = stroke.numpy()
        else:
            stroke = np.array(stroke)
            
        if len(stroke) == 0:
            continue
            
        # process_font_data.py 中的归一化:
        # nx = (x / width) * 2 - 1
        # ny = (y / height) * 2 - 1
        # 图像坐标系通常 Y 轴向下。
        # Matplotlib 默认 Y 轴向上。
        # 为了让笔画看起来和图像方向一致（如果是汉字），我们需要翻转 Y 轴显示，
        # 或者在绘制时取反 Y。
        # 这里我们假设数据中的 -1 是顶部，1 是底部 (对应图像坐标)。
        # 在 plot 中，为了让 -1 在顶部，我们需要 invert_yaxis() 或者绘制 -y。
        
        plt.plot(stroke[:, 0], -stroke[:, 1], label=f'Stroke {i+1}', color=colors[i])
        # 标出起点
        plt.scatter(stroke[0, 0], -stroke[0, 1], color=colors[i], s=20, marker='o')
        # 标出过程点
        plt.scatter(stroke[:, 0], -stroke[:, 1], color=colors[i], s=2)

    plt.title(f"Strokes: {key} (Normalized)")
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.grid(True)
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("output/visual_data.png")

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
