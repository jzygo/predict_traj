import os
import json
import pickle
import argparse
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from PIL import Image

def save_image(args):
    """
    保存单张图片的辅助函数
    """
    key, img_bytes, save_dir = args
    
    try:
        # 尝试识别图像格式并保存
        # 直接写入原始字节流是最快的，且无损
        # 我们先尝试用 PIL 读取一下以确定后缀名（如果需要），或者直接存为 jpg/png
        # 为了最大化速度和兼容性，我们先解析一次
        image = Image.open(io.BytesIO(img_bytes))
        ext = image.format.lower() if image.format else 'jpg'
        
        save_path = os.path.join(save_dir, f"{key}.{ext}")
        
        # 直接写入原始字节，避免重编码带来的质量损失和性能开销
        with open(save_path, 'wb') as f:
            f.write(img_bytes)
            
        return True
    except Exception as e:
        print(f"Error saving image {key}: {e}")
        return False

def process_font(font_name, source_root, target_root, num_workers=16):
    """
    处理单个字体的数据
    """
    source_font_dir = os.path.join(source_root, font_name)
    target_font_dir = os.path.join(target_root, font_name)
    target_images_dir = os.path.join(target_font_dir, "images")
    
    # 1. 准备目录
    os.makedirs(target_images_dir, exist_ok=True)
    
    json_path = os.path.join(source_font_dir, f"{font_name}.json")
    pkl_path = os.path.join(source_font_dir, f"{font_name}_imgs.pkl")
    
    if not os.path.exists(json_path) or not os.path.exists(pkl_path):
        print(f"Skipping {font_name}: Missing json or pkl file.")
        return

    print(f"Processing font: {font_name}...")

    # 2. 处理笔画数据 (Strokes)
    # 将 JSON 转换为 PyTorch 格式 (.pt)，极大减少内存占用并加速加载
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            strokes_data = json.load(f)
        
        # 转换为更紧凑的格式
        # 假设 strokes_data 是 {key: [[x,y], ...]} 或者是 {key: [stroke1, stroke2...]}
        # 我们将其中的数值列表预先转换为 Tensor (如果长度不固定，只能存为 list of Tensors)
        # 为了通用性，我们保存为字典: {key: [Tensor(stroke1), Tensor(stroke2)...]}
        
        compact_strokes = {}
        for key, strokes in strokes_data.items():
            # 将每个笔画转换为 float32 tensor，不仅省内存，训练时也省去了转换时间
            compact_strokes[key] = [torch.tensor(s, dtype=torch.float32) for s in strokes]
            
        # 保存为 .pt 文件
        target_pt_path = os.path.join(target_font_dir, "strokes.pt")
        torch.save(compact_strokes, target_pt_path)
        print(f"Saved compact strokes to {target_pt_path}")
            
    except Exception as e:
        print(f"Error processing strokes for {font_name}: {e}")
        return

    # 3. 处理图像数据 (Images)
    try:
        with open(pkl_path, 'rb') as f:
            images_data = pickle.load(f)
            
        # 准备任务列表
        tasks = []
        for key, img_bytes in images_data.items():
            # 确保只处理在 strokes 中存在的 key (对齐数据)
            if key in strokes_data:
                tasks.append((key, img_bytes, target_images_dir))
        
        # 多线程保存图片
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(save_image, tasks), 
                total=len(tasks), 
                desc=f"Saving images for {font_name}",
                unit="img"
            ))
            
    except Exception as e:
        print(f"Error processing images for {font_name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Convert Font Dataset to Image Files")
    parser.add_argument("--source_root", type=str, default=r"/data1/jizy/font_out", help="原始数据根目录 (包含 pkl/json)")
    parser.add_argument("--target_root", type=str, default=r"/data1/jizy/font_data", help="目标数据根目录")
    parser.add_argument("--num_workers", type=int, default=16, help="线程数")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source_root):
        print(f"Error: Source root {args.source_root} does not exist.")
        return

    # 扫描所有字体
    font_names = [d for d in os.listdir(args.source_root) if os.path.isdir(os.path.join(args.source_root, d))]
    font_names.sort()
    
    print(f"Found {len(font_names)} fonts in {args.source_root}")
    print(f"Target directory: {args.target_root}")
    
    for font_name in font_names:
        process_font(font_name, args.source_root, args.target_root, args.num_workers)
        
    print("\nAll done! Dataset conversion completed.")
    print(f"New dataset structure in {args.target_root}:")
    print("  font_name/")
    print("    strokes.json")
    print("    images/")
    print("      key1.jpg")
    print("      key2.jpg")
    print("      ...")

if __name__ == "__main__":
    main()
