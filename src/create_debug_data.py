import numpy as np
import pickle
import os
import io
from PIL import Image

def create_debug_data():
    # --- 1. 创建 images.pkl ---
    # value是一个全部为白色的256*256图片
    # 通常图像使用 uint8 类型，范围 0-255。白色即为 255。
    # 假设图像是 RGB 3通道
    image = np.full((256, 256, 3), 255, dtype=np.uint8)
    
    # 参考 infer.py，将图片转为 bytes 存储
    pil_img = Image.fromarray(image)
    with io.BytesIO() as output:
        pil_img.save(output, format="JPEG")
        img_bytes = output.getvalue()
    
    images_dict = {"debug": img_bytes}
    
    with open('../data/debug/images.pkl', 'wb') as f:
        pickle.dump(images_dict, f)
    print(f"已创建 images.pkl，包含 key='debug' 的全白图片 (JPEG bytes)")

    # --- 2. 创建 strokes.pkl ---
    # value是一个三维向量序列，整个序列48个点
    num_points = 48
    
    # 第一个维度从-1均匀变化到1
    dim0 = np.linspace(-1, 1, num_points)
    
    # 第二个维度保持为0
    dim1 = np.zeros(num_points)
    
    # 第三个维度从-1均匀变化到1
    dim2 = np.linspace(-1, 1, num_points)
    
    # 堆叠成 (48, 3) 的形状
    # 使用 float32 是常见做法，如果需要 float64 可以去掉 .astype
    stroke = np.stack([dim0, dim1, dim2], axis=1).astype(np.float32)
    
    strokes_dict = {"debug": [stroke]}
    
    with open('../data/debug/strokes.pkl', 'wb') as f:
        pickle.dump(strokes_dict, f)
    print(f"已创建 strokes.pkl，包含 key='debug' 的轨迹数据，形状: {stroke.shape}")
    print("第一个点:", stroke[0])
    print("中间点:", stroke[num_points//2])
    print("最后点:", stroke[-1])

if __name__ == "__main__":
    create_debug_data()
