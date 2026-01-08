import cv2


def speed_up_video_cv2(input_path, output_path, speed_factor):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("无法打开视频文件")
        return

    # 获取原视频属性
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 定义编解码器 (mp4v 兼容性较好)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"原视频帧数: {total_frames}, 开始处理...")

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 核心逻辑：只写入第 N 帧
        # 例如加速10倍，则每10帧只取1帧写入，保持原FPS播放，视觉上就是加速了
        if count % speed_factor == 0:
            out.write(frame)

        count += 1
        if count % 100 == 0:
            print(f"进度: {count}/{total_frames}", end='\r')

    cap.release()
    out.release()
    print("\n完成 (无音频模式)")


if __name__ == "__main__":
    speed_up_video_cv2("input2.mp4", "output2_cv2_20x.mp4", 20)