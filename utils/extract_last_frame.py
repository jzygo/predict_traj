import os
import sys
import glob
import numpy as np
from tqdm import tqdm
import mediapy as media
import concurrent.futures
import shutil
import subprocess

def process_video_task(args):
    video_path, source_dir, target_dir = args
    
    # Determine output path
    rel_path = os.path.relpath(video_path, source_dir)
    base_name = os.path.splitext(rel_path)[0]
    output_rel_path = base_name + ".png"
    output_path = os.path.join(target_dir, output_rel_path)

    # Create directories
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Method 1: Try ffmpeg command line (Low memory)
    # This is preferred as it avoids loading the video into Python memory
    if shutil.which('ffmpeg'):
        try:
            # -sseof -1: Seek to 1 second before EOF
            # -update 1: Overwrite the output file with each new frame decoded
            # -q:v 2: High quality jpeg/png
            # This effectively saves the last frame (or one of the last frames)
            cmd = [
                'ffmpeg', '-y',
                '-sseof', '-1',
                '-i', video_path,
                '-update', '1',
                '-q:v', '2',
                '-loglevel', 'error',
                output_path
            ]
            # On Windows, creationflags=0x08000000 (CREATE_NO_WINDOW) prevents popup windows
            creationflags = 0x08000000 if sys.platform == 'win32' else 0
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, creationflags=creationflags)
            
            # Verify output exists
            if os.path.exists(output_path):
                return None
        except Exception:
            # Fallback if ffmpeg fails
            pass

    # Method 2: Try OpenCV (Low memory)
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frame_count > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite(output_path, frame)
                    cap.release()
                    return None
            cap.release()
    except ImportError:
        pass
    except Exception:
        pass

    # Method 3: Fallback to mediapy (High memory warning)
    try:
        # Read video
        video = media.read_video(video_path)
        
        if len(video) == 0:
            return f"Warning: Video {video_path} has no frames. Skipping."

        # Get last frame
        last_frame = video[-1]

        # Save image
        media.write_image(output_path, last_frame)
        return None
    except Exception as e:
        return f"Error processing {video_path}: {e}"

def main():
    # Determine paths
    # Assuming script is in project/utils/ or project/
    # We want to find project/data/video and project/data/figure
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # If script is in utils/, parent is project root
    project_root = os.path.dirname(current_dir)
    
    # Check if we are in the right place (simple check)
    if not os.path.exists(os.path.join(project_root, 'data')):
        # Maybe script is in project root?
        if os.path.exists(os.path.join(current_dir, 'data')):
            project_root = current_dir
        else:
            print(f"Error: Could not locate 'data' directory relative to {current_dir}")
            sys.exit(1)

    DATA_VIDEO_DIR = os.path.join(project_root, 'data', 'video')
    DATA_FIGURE_DIR = os.path.join(project_root, 'data', 'figure')

    print(f"Source Video Directory: {DATA_VIDEO_DIR}")
    print(f"Target Figure Directory: {DATA_FIGURE_DIR}")

    if not os.path.exists(DATA_VIDEO_DIR):
        print(f"Error: Source directory {DATA_VIDEO_DIR} does not exist.")
        sys.exit(1)

    # Find all video files (mp4, avi, mov, etc.)
    extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in extensions:
        video_files.extend(glob.glob(os.path.join(DATA_VIDEO_DIR, '**', ext), recursive=True))
    
    if not video_files:
        print("No video files found.")
        return

    print(f"Found {len(video_files)} videos. Processing...")

    # Detect CPU count
    max_workers = os.cpu_count() or 1
    print(f"Detected {max_workers} CPUs. Using {max_workers} threads for processing.")

    # Prepare arguments
    tasks = [(vp, DATA_VIDEO_DIR, DATA_FIGURE_DIR) for vp in video_files]

    # Run with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_video_task, tasks), total=len(tasks)))

    # Print errors
    for res in results:
        if res:
            print(res)

    print("Done.")

if __name__ == "__main__":
    main()
