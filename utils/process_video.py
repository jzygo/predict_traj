import cv2
import os
import numpy as np
import argparse

def process_video(target_frames=None):
    # 定义输入和输出路径
    input_path = os.path.join('../data', 'linqiange', 'bi.mp4')
    output_dir = '../debug'
    output_path = os.path.join(output_dir, 'output_grid.mp4')

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"Error: Input file not found at {input_path}")
        return

    # 打开视频
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 获取视频属性
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 3
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Original Video: {width}x{height} @ {fps}fps, {total_frames} frames")

    # 定义四宫格中每个子画面的大小 (原视频宽高的一半)
    # 这样拼起来后，总分辨率与原视频一致
    sub_w = width // 2
    sub_h = height // 2
    
    # 如果原视频宽高是奇数，//2 可能会导致拼接后少1像素，这里简单处理，输出视频宽高设为 sub_w * 2, sub_h * 2
    out_w = sub_w * 2
    out_h = sub_h * 2

    # 定义视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    # 准备重采样索引
    indices = None
    if target_frames is not None:
        indices = np.linspace(0, total_frames - 1, target_frames).astype(int)
        print(f"Resampling video to {target_frames} frames.")

    current_cap_idx = -1
    last_valid_frame = None

    def frame_generator():
        nonlocal current_cap_idx, last_valid_frame
        if indices is not None:
            for t_idx in indices:
                while current_cap_idx < t_idx:
                    ret, frame = cap.read()
                    if not ret:
                        return
                    current_cap_idx += 1
                    last_valid_frame = frame
                if current_cap_idx < t_idx: # Video ended early
                    return
                yield last_valid_frame
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                yield frame

    frame_idx = 0
    total_to_process = len(indices) if indices is not None else total_frames

    for frame in frame_generator():
        # 1. 原分辨率 (缩放到 sub_w, sub_h 以便拼接)
        # 注意：这里其实是把原图缩小了一半显示。
        # 如果想保留原图细节，可能需要裁剪或者输出更大的视频。
        # 但为了四宫格对比，通常是缩小显示。
        frame_orig_view = cv2.resize(frame, (sub_w, sub_h))
        
        # 辅助函数：下采样再上采样
        def process_resolution(img, target_resolution):
            # 计算缩放比例，保持长宽比
            # 这里假设 target_resolution 是短边的目标大小
            h, w = img.shape[:2]
            scale = target_resolution / min(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            
            # 下采样
            down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            # 上采样回视图大小 (sub_w, sub_h)
            # 注意：为了展示低分辨率效果，我们先放大回原图大小(或者接近原图大小)，然后再缩小到视图大小？
            # 或者直接从 down resize 到 (sub_w, sub_h)
            # 直接 resize 到 (sub_w, sub_h) 即可
            up = cv2.resize(down, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST) 
            # 使用 INTER_NEAREST 可以看到马赛克效果，INTER_LINEAR 会模糊
            # 用户通常希望看到“低分辨率”的像素感，或者模糊感。这里选用 NEAREST 可能会更直观地展示“分辨率低”的含义（像素块大）。
            # 如果想要模糊效果，可以改用 INTER_LINEAR
            return up

        # 2. 128分辨率
        frame_128 = process_resolution(frame, 128)

        # 3. 64分辨率
        frame_64 = process_resolution(frame, 64)

        # 4. 32分辨率
        frame_32 = process_resolution(frame, 32)

        # 拼接
        # Top row: Original, 128
        top_row = np.hstack((frame_orig_view, frame_128))
        # Bottom row: 64, 32
        bottom_row = np.hstack((frame_64, frame_32))
        # Combine
        grid_frame = np.vstack((top_row, bottom_row))

        # 添加文字标签
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 1
        color = (255, 255, 255)
        thickness = 2
        
        cv2.putText(grid_frame, "Original", (10, 30), font, scale, color, thickness)
        cv2.putText(grid_frame, "128 px", (sub_w + 10, 30), font, scale, color, thickness)
        cv2.putText(grid_frame, "64 px", (10, sub_h + 30), font, scale, color, thickness)
        cv2.putText(grid_frame, "32 px", (sub_w + 10, sub_h + 30), font, scale, color, thickness)

        out.write(grid_frame)

        frame_idx += 1
        if frame_idx % 50 == 0:
            print(f"Processed {frame_idx}/{total_to_process} frames")

    cap.release()
    out.release()
    print(f"Done. Saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_frames', type=int, default=64, help='Target number of frames')
    args = parser.parse_args()
    process_video(target_frames=args.target_frames)
