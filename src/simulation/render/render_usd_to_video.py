import sys
import os
import math
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import tempfile
import shutil
import glob
import threading
import time

# --- 全局配置 ---
OUTPUT_DIR_BASE = "/home/jizy/project/video_tokenizer/src/simulation/render/output_frames"
RESOLUTION = (1280, 720)
USE_GPU = True
# 并行进程数 - 根据GPU内存占用20%计算，可同时运行5个进程
NUM_PARALLEL_PROCESSES = 5

# GPU分配锁，用于线程安全地分配GPU
gpu_lock = threading.Lock()


def get_gpu_memory_info():
    """获取所有GPU的显存使用情况，返回按空闲显存排序的GPU列表"""
    try:
        # 使用nvidia-smi查询GPU显存信息
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.total,memory.used,memory.free", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode != 0:
            print("Warning: nvidia-smi failed, defaulting to GPU 0")
            return [{'index': 0, 'total': 0, 'used': 0, 'free': float('inf')}]
        
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append({
                        'index': int(parts[0]),
                        'total': int(parts[1]),
                        'used': int(parts[2]),
                        'free': int(parts[3])
                    })
        
        # 按空闲显存降序排序（空闲越多越优先）
        gpu_info.sort(key=lambda x: x['free'], reverse=True)
        return gpu_info
        
    except FileNotFoundError:
        print("Warning: nvidia-smi not found, defaulting to GPU 0")
        return [{'index': 0, 'total': 0, 'used': 0, 'free': float('inf')}]
    except Exception as e:
        print(f"Warning: Error getting GPU info: {e}, defaulting to GPU 0")
        return [{'index': 0, 'total': 0, 'used': 0, 'free': float('inf')}]


def select_best_gpu():
    """选择当前显存最空闲的GPU"""
    with gpu_lock:
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            best_gpu = gpu_info[0]
            print(f"   Selected GPU {best_gpu['index']} (Free: {best_gpu['free']}MB / Total: {best_gpu['total']}MB)")
            return best_gpu['index']
        return 0


def get_available_gpus():
    """获取所有可用GPU列表，按空闲显存排序"""
    gpu_info = get_gpu_memory_info()
    return [g['index'] for g in gpu_info]


def assign_gpus_to_workers(num_workers):
    """为所有worker预分配GPU，使用轮询方式均匀分配"""
    gpu_info = get_gpu_memory_info()
    if not gpu_info:
        return [0] * num_workers
    
    # 按空闲显存排序的GPU列表
    available_gpus = [g['index'] for g in gpu_info]
    
    # 轮询分配GPU给每个worker
    assignments = []
    for i in range(num_workers):
        gpu_idx = available_gpus[i % len(available_gpus)]
        assignments.append(gpu_idx)
    
    return assignments


def print_gpu_status():
    """打印所有GPU的状态"""
    gpu_info = get_gpu_memory_info()
    print("\n📊 GPU Status:")
    for gpu in gpu_info:
        usage_pct = (gpu['used'] / gpu['total'] * 100) if gpu['total'] > 0 else 0
        bar_len = 20
        filled = int(bar_len * usage_pct / 100)
        bar = '█' * filled + '░' * (bar_len - filled)
        print(f"   GPU {gpu['index']}: [{bar}] {usage_pct:.1f}% used ({gpu['used']}MB / {gpu['total']}MB, Free: {gpu['free']}MB)")
    print()

def get_args():
    """解析命令行参数"""
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser(description="Blender Multi-Process Renderer")
    parser.add_argument("--usd_path", type=str, required=True, help="Path to USD file")
    parser.add_argument("--start", type=int, default=None, help="Start frame override")
    parser.add_argument("--end", type=int, default=None, help="End frame override")
    
    # [新增] 步长参数，默认为 1 (不跳帧)
    parser.add_argument("--step", type=int, default=1, help="Frame step (1=all frames, 10=every 10th frame)")
    
    # [修改] 视频输出参数，现在默认为 output.mp4，因为我们总是输出视频
    parser.add_argument("--video_name", type=str, default="output.mp4", help="Output video filename (e.g. 'result.mp4')")
    
    parser.add_argument("--probe", action="store_true", help="Only detect frame range and exit")
    
    parser.add_argument("--use_ffmpeg", action="store_true", help="Render images first then combine with ffmpeg")
    
    # [新增] 多进程并行渲染参数
    parser.add_argument("--parallel", action="store_true", help="Enable multi-process parallel rendering to maximize GPU utilization")
    parser.add_argument("--num_workers", type=int, default=NUM_PARALLEL_PROCESSES, help="Number of parallel render processes (default: 5, ~20%% GPU memory each)")
    
    # [新增] 内部使用的worker模式参数
    parser.add_argument("--worker_mode", action="store_true", help="Internal: Run as a worker process")
    parser.add_argument("--worker_output_dir", type=str, default="/home/jizy/project/video_tokenizer/src/simulation/render/output_frames/temp_frames", help="Internal: Worker output directory for frames")
    parser.add_argument("--gpu_id", type=int, default=None, help="Internal: Specify which GPU to use for rendering")

    return parser.parse_args(argv)


def is_blender_environment():
    """检查是否在Blender环境中运行"""
    try:
        import bpy
        return True
    except ImportError:
        return False


def probe_frame_range(usd_path):
    """探测USD文件的帧范围（通过启动Blender子进程）"""
    script_path = os.path.abspath(__file__)
    
    cmd = [
        "/home/jizy/tool/blender/blender", "--background", "--python", script_path,
        "--", "--usd_path", usd_path, "--probe"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    for line in result.stdout.split('\n'):
        if line.startswith("PROBE_RESULT:"):
            parts = line.split(':')
            return int(parts[1]), int(parts[2])
    
    print("Warning: Could not detect frame range, using default 1-100")
    return 1, 100


def split_frame_ranges(start, end, step, num_workers):
    """将帧范围分割成多个子范围用于并行渲染"""
    # 计算实际需要渲染的帧数
    frames = list(range(start, end + 1, step))
    total_frames = len(frames)
    
    if total_frames == 0:
        return []
    
    # 每个worker分配的帧数
    frames_per_worker = max(1, total_frames // num_workers)
    
    ranges = []
    for i in range(num_workers):
        worker_start_idx = i * frames_per_worker
        if i == num_workers - 1:
            # 最后一个worker处理剩余所有帧
            worker_end_idx = total_frames - 1
        else:
            worker_end_idx = min((i + 1) * frames_per_worker - 1, total_frames - 1)
        
        if worker_start_idx <= worker_end_idx:
            worker_frames = frames[worker_start_idx:worker_end_idx + 1]
            ranges.append({
                'worker_id': i,
                'start': worker_frames[0],
                'end': worker_frames[-1],
                'frames': worker_frames
            })
    
    return ranges


def run_worker_process(worker_info, usd_path, step, temp_dir, assigned_gpu):
    """启动一个Blender worker进程渲染指定帧范围"""
    worker_id = worker_info['worker_id']
    start = worker_info['start']
    end = worker_info['end']
    
    # 使用预分配的GPU
    gpu_id = assigned_gpu
    
    worker_output_dir = os.path.join(temp_dir, f"worker_{worker_id}")
    os.makedirs(worker_output_dir, exist_ok=True)
    
    # 创建日志文件
    log_file = os.path.join(worker_output_dir, "render.log")
    
    script_path = os.path.abspath(__file__)
    
    cmd = [
        "/home/jizy/tool/blender/blender", "--background", "--python", script_path,
        "--",
        "--usd_path", usd_path,
        "--start", str(start),
        "--end", str(end),
        "--step", str(step),
        "--worker_mode",
        "--worker_output_dir", worker_output_dir,
        "--gpu_id", str(gpu_id),
        "--use_ffmpeg"  # worker模式下总是输出图片序列
    ]
    
    print(f"[Worker {worker_id}] Starting: frames {start}-{end} on GPU {gpu_id}")
    print(f"[Worker {worker_id}] Log file: {log_file}")
    
    # 设置环境变量让Blender使用指定的GPU
    env = os.environ.copy()
    
    # CUDA (用于Cycles)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # NVIDIA GPU选择
    env['__NV_PRIME_RENDER_OFFLOAD'] = '1'
    env['__NV_PRIME_RENDER_OFFLOAD_PROVIDER'] = 'NVIDIA-G0'
    env['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'
    env['NVIDIA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        # 使用Popen以便实时跟踪进程
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                env=env,
                text=True
            )
            
            # 等待进程完成，定期检查状态
            while process.poll() is None:
                # 检查输出文件数量
                frame_count = len(glob.glob(os.path.join(worker_output_dir, "frame_*.png")))
                if frame_count > 0:
                    print(f"[Worker {worker_id}] Progress: {frame_count} frames rendered")
                time.sleep(10)  # 每10秒检查一次
            
            return_code = process.returncode
        
        # 读取日志文件检查错误
        with open(log_file, 'r') as log_f:
            log_content = log_f.read()
        
        if return_code != 0:
            print(f"[Worker {worker_id}] ❌ FAILED with return code {return_code}")
            # 显示最后100行日志
            log_lines = log_content.split('\n')
            print(f"[Worker {worker_id}] Last 50 lines of log:")
            for line in log_lines[-50:]:
                print(f"  {line}")
            return False, worker_id, worker_output_dir
        
        # 检查是否有渲染输出
        frame_files = glob.glob(os.path.join(worker_output_dir, "frame_*.png"))
        if not frame_files:
            print(f"[Worker {worker_id}] ⚠️ WARNING: No frames rendered!")
            print(f"[Worker {worker_id}] Log content:")
            for line in log_content.split('\n')[-30:]:
                print(f"  {line}")
            return False, worker_id, worker_output_dir
        
        print(f"[Worker {worker_id}] ✅ Completed: {len(frame_files)} frames rendered")
        return True, worker_id, worker_output_dir
        
    except Exception as e:
        print(f"[Worker {worker_id}] ❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False, worker_id, worker_output_dir


def combine_frames_to_video(temp_dir, output_path, fps, start_frame):
    """将所有worker渲染的帧合并成视频"""
    # 创建合并目录
    combined_dir = os.path.join(temp_dir, "combined")
    os.makedirs(combined_dir, exist_ok=True)
    
    # 收集所有帧并按帧号重命名
    worker_dirs = sorted(glob.glob(os.path.join(temp_dir, "worker_*")))
    
    for worker_dir in worker_dirs:
        frame_files = glob.glob(os.path.join(worker_dir, "frame_*.png"))
        for frame_file in frame_files:
            # 复制到合并目录
            shutil.copy(frame_file, combined_dir)
    
    # 使用ffmpeg合并
    input_pattern = os.path.join(combined_dir, "frame_%04d.png")
    
    command = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", str(start_frame),
        "-i", input_pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",  # 高质量
        output_path
    ]
    
    try:
        subprocess.check_call(command)
        print(f"✅ Video saved to: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Error combining video: {e}")
        return False


def run_parallel_render(args):
    """主控制器：并行渲染模式"""
    print("=" * 60)
    print("🚀 Multi-Process Parallel Rendering Mode")
    print(f"   Workers: {args.num_workers} (each ~20% GPU memory)")
    print("=" * 60)
    
    # 显示GPU状态
    print_gpu_status()
    
    # 1. 探测帧范围
    print("\n[Step 1] Detecting frame range...")
    detected_start, detected_end = probe_frame_range(args.usd_path)
    
    start = args.start if args.start is not None else detected_start
    end = args.end if args.end is not None else detected_end
    step = args.step
    
    print(f"   Frame range: {start} - {end} (step: {step})")
    
    # 2. 分割帧范围
    print(f"\n[Step 2] Splitting work among {args.num_workers} workers...")
    frame_ranges = split_frame_ranges(start, end, step, args.num_workers)
    
    for fr in frame_ranges:
        print(f"   Worker {fr['worker_id']}: frames {fr['start']}-{fr['end']} ({len(fr['frames'])} frames)")
    
    # 3. 创建临时目录（固定在输出路径下）
    temp_dir = os.path.join(OUTPUT_DIR_BASE, "temp_output")
    if os.path.exists(temp_dir):
        # 清理旧的临时目录
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"\n[Step 3] Temp directory: {temp_dir}")
    
    # 4. 预分配GPU给每个worker
    print(f"\n[Step 4] Assigning GPUs to workers...")
    gpu_assignments = assign_gpus_to_workers(len(frame_ranges))
    for i, gpu_id in enumerate(gpu_assignments):
        print(f"   Worker {i} -> GPU {gpu_id}")
    
    # 5. 并行启动所有worker
    print(f"\n[Step 5] Launching {len(frame_ranges)} parallel render processes...")
    
    results = []
    # 使用ThreadPoolExecutor而不ProcessPoolExecutor
    # 因为实际的并行是通过外部Blender进程实现的
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(
                run_worker_process, 
                fr, 
                args.usd_path, 
                step, 
                temp_dir,
                gpu_assignments[fr['worker_id']]  # 传入预分配的GPU
            ): fr['worker_id'] for fr in frame_ranges
        }
        
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                success, wid, output_dir = future.result()
                results.append((success, wid, output_dir))
            except Exception as e:
                print(f"[Worker {worker_id}] Exception: {e}")
                results.append((False, worker_id, None))
    
    # 6. 检查结果
    all_success = all(r[0] for r in results)
    if not all_success:
        print("\n❌ Some workers failed!")
        for success, wid, _ in results:
            if not success:
                print(f"   Worker {wid} failed")
    
    # 7. 合并视频
    print("\n[Step 6] Combining frames into video...")
    
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    video_filename = args.video_name if args.video_name else "output.mp4"
    output_path = os.path.join(OUTPUT_DIR_BASE, video_filename)
    
    # 默认帧率
    fps = 24
    
    success = combine_frames_to_video(temp_dir, output_path, fps, start)
    
    # 8. 清理临时文件
    print("\n[Step 7] Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
        print("   ✅ Temp files cleaned")
    except Exception as e:
        print(f"   ⚠️ Could not clean temp dir: {e}")
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Parallel rendering completed successfully!")
    else:
        print("⚠️ Rendering completed with some issues")
    print("=" * 60)

def auto_detect_frame_range():
    """(保持原逻辑不变) 根据导入的物体自动检测动画范围"""
    import bpy
    
    try:
        from pxr import Usd
    except ImportError:
        Usd = None
    
    min_frame = 999999
    max_frame = -999999
    found_anim = False

    if bpy.data.cache_files:
        found_anim = True
        min_frame = bpy.context.scene.frame_start
        max_frame = bpy.context.scene.frame_end
        print(f"Detected USD Cache. Using Scene Range: {min_frame} - {max_frame}")

    if not found_anim:
        for obj in bpy.data.objects:
            for mod in obj.modifiers:
                if mod.type == 'MESH_SEQUENCE_CACHE' and mod.cache_file:
                    found_anim = True
                    min_frame = bpy.context.scene.frame_start
                    max_frame = bpy.context.scene.frame_end
                    break
            if not found_anim:
                for const in obj.constraints:
                    if const.type == 'TRANSFORM_CACHE' and const.cache_file:
                        found_anim = True
                        min_frame = bpy.context.scene.frame_start
                        max_frame = bpy.context.scene.frame_end
                        break
            if found_anim: break

    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.action:
            for fcurve in obj.animation_data.action.fcurves:
                if fcurve.keyframe_points:
                    found_anim = True
                    k_start = fcurve.keyframe_points[0].co.x
                    k_end = fcurve.keyframe_points[-1].co.x
                    if k_start < min_frame: min_frame = k_start
                    if k_end > max_frame: max_frame = k_end

    start = 1
    end = 100
    if found_anim:
        start = int(math.floor(min_frame))
        end = int(math.ceil(max_frame))
    else:
        print("No animation keyframes or USD caches found; defaulting to 1-100 frames.")
        pass

    return start, end

def setup_scene(args):
    import bpy
    
    # 1. 重置场景
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. 导入 USD
    if os.path.exists(args.usd_path):
        bpy.ops.wm.usd_import(filepath=args.usd_path, import_cameras=True, import_lights=True)
    else:
        print(f"Error: USD file not found at {args.usd_path}")
        sys.exit(1)

    # --- 设置特定对象的材质颜色 ---
    def get_or_create_material(name, color_rgb, roughness=0.5):
        """获取或创建指定颜色的材质"""
        mat = bpy.data.materials.get(name)
        if not mat:
            mat = bpy.data.materials.new(name=name)
            mat.use_nodes = True
            bsdf = mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs['Base Color'].default_value = (*color_rgb, 1.0)
                bsdf.inputs['Roughness'].default_value = roughness
                # 针对黑色墨水增加一点光泽控制
                if color_rgb == (0.0, 0.0, 0.0):
                    try:
                        bsdf.inputs['Specular IOR Level'].default_value = 0.2
                    except:
                        pass 
        return mat

    def apply_material_recursive(root_name, material):
        """
        查找名为 root_name 的物体，并将 material 应用于它及其所有子孙物体
        """
        root_obj = bpy.data.objects.get(root_name)
        
        if not root_obj:
            print(f"Warning: Root object '{root_name}' not found in scene.")
            return

        print(f"--- Applying material '{material.name}' to hierarchy under '{root_name}' ---")
        
        # 递归函数遍历子节点
        def _recursive_apply(obj):
            # 如果是网格物体，应用材质
            if obj.type == 'MESH':
                if obj.data.materials:
                    obj.data.materials.clear()
                obj.data.materials.append(material)
                # print(f"  -> Applied to {obj.name}") # 调试用，防止输出太多可注释掉
            
            # 继续遍历子物体
            for child in obj.children:
                _recursive_apply(child)

        # 开始递归
        _recursive_apply(root_obj)

    # 1. 定义材质
    # 黑色 (用于 brush_particles)
    mat_black = get_or_create_material("Mat_Black", (0.0, 0.0, 0.0), roughness=0.2)
    # 棕色 (用于 stick)
    mat_brown = get_or_create_material("Mat_Brown", (0.6, 0.4, 0.2), roughness=0.6)

    # 2. 应用材质到指定层级
    # 将 brush_particles_instance 下所有物体设为黑色
    apply_material_recursive("brush_particles_instance", mat_black)
    
    # 将 stick_instance 下所有物体设为棕色
    apply_material_recursive("stick_instance", mat_brown)

    # ----------------------------------------

    scene = bpy.context.scene

    
    # --- 帧范围设定逻辑 ---
    detected_start, detected_end = auto_detect_frame_range()
    
    if args.probe:
        print(f"PROBE_RESULT:{detected_start}:{detected_end}")
        return False

    scene.frame_start = args.start if args.start is not None else detected_start
    scene.frame_end = args.end if args.end is not None else detected_end
    scene.frame_step = args.step
    
    print(f"--- Rendering Range: {scene.frame_start} to {scene.frame_end} (Step: {scene.frame_step}) ---")

    # 3. GPU设置 - 在Blender内部配置GPU
    if args.gpu_id is not None:
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')
        print(f"--- GPU Selection: CUDA_VISIBLE_DEVICES={cuda_visible} (Target: Physical GPU {args.gpu_id}) ---")
        
        # 切换到Cycles渲染器，因为Cycles对GPU控制更可靠
        # EEVEE在多GPU环境下难以控制具体使用哪个GPU
        scene.render.engine = 'CYCLES'
        
        # 配置Cycles使用GPU
        scene.cycles.device = 'GPU'
        
        # 获取Cycles偏好设置
        prefs = bpy.context.preferences
        cycles_prefs = prefs.addons.get('cycles')
        
        if cycles_prefs:
            cycles_prefs = cycles_prefs.preferences
            
            # 设置计算设备类型
            # 尝试OPTIX (最快), 如果不可用则使用CUDA
            try:
                cycles_prefs.compute_device_type = 'OPTIX'
            except:
                cycles_prefs.compute_device_type = 'CUDA'
            
            # 刷新设备列表
            cycles_prefs.get_devices()
            
            # 由于CUDA_VISIBLE_DEVICES已设置，进程内只能看到一个GPU
            # 启用所有可见的GPU设备（实际上只有一个）
            enabled_devices = []
            for device in cycles_prefs.devices:
                if device.type in ('CUDA', 'OPTIX'):
                    device.use = True
                    enabled_devices.append(device.name)
                elif device.type == 'CPU':
                    device.use = False  # 禁用CPU，只用GPU
            
            if enabled_devices:
                print(f"--- Cycles GPU devices enabled: {enabled_devices} ---")
            else:
                print("--- Warning: No GPU devices found, falling back to CPU ---")
                scene.cycles.device = 'CPU'
        
        # Cycles渲染质量设置 (适合快速预览)
        scene.cycles.samples = 64  # 采样数
        scene.cycles.use_denoising = True  # 启用降噪
        scene.cycles.use_adaptive_sampling = True  # 自适应采样
        
        print(f"--- Using Cycles Engine on GPU {args.gpu_id} ---")
    else:
        # 非worker模式，使用EEVEE（更快但GPU控制有限）
        # 4. 渲染引擎设置 - 使用 Eevee (实时渲染，速度快)
        scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Blender 4.2+ 使用 EEVEE Next
        print(f"--- Using Eevee Engine (GPU Accelerated) ---")
    
    # 渲染引擎质量设置（仅当使用EEVEE时）
    if scene.render.engine == 'BLENDER_EEVEE_NEXT':
        # Eevee Next 渲染质量设置 (Blender 4.2+)
        eevee = scene.eevee
        
        # 采样设置 (Blender 4.2+ 使用新属性名)
        try:
            eevee.taa_render_samples = 64      # 旧版属性
            eevee.taa_samples = 16
        except AttributeError:
            pass  # Blender 4.2+ 可能已移除这些属性
        
        # 尝试设置通用属性 (兼容不同版本)
        # 运动模糊 (禁用以提速)
        try:
            eevee.use_motion_blur = False
        except AttributeError:
            pass
        
        # 光线追踪设置 (EEVEE Next 新增)
        try:
            eevee.use_raytracing = False       # 禁用光追以提速
        except AttributeError:
            pass
        
        # 阴影设置 (EEVEE Next 使用不同的方式)
        try:
            eevee.shadow_ray_count = 1         # 阴影光线数 (越少越快)
            eevee.shadow_step_count = 6        # 阴影步数
        except AttributeError:
            pass
        
        # 体积设置 (简单场景可禁用)
        try:
            eevee.volumetric_tile_size = '8'
            eevee.use_volumetric_shadows = False
        except AttributeError:
            pass

    # --- 环境光 ---
    if not scene.world:
        scene.world = bpy.data.worlds.new("World")
    scene.world.use_nodes = True
    bg_node = scene.world.node_tree.nodes.get('Background')
    if not bg_node:
        bg_node = scene.world.node_tree.nodes.new('ShaderNodeBackground')
        output_node = scene.world.node_tree.nodes.new('ShaderNodeOutputWorld')
        scene.world.node_tree.links.new(bg_node.outputs['Background'], output_node.inputs['Surface'])
    bg_node.inputs['Color'].default_value = (0.7, 0.7, 0.7, 1)
    
    has_light = any(obj.type == 'LIGHT' for obj in bpy.data.objects)
    if not has_light:
        bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))
        bpy.context.active_object.data.energy = 3.0
    
    output_path = os.path.join(bpy.path.abspath("//"), "hierarchy.txt")

    def write_obj_hierarchy(obj, file, level=0):
        indent = "    " * level
        file.write(f"{indent}- {obj.name} [{obj.type}]\n")
        
        # 递归遍历子物体
        for child in obj.children:
            write_obj_hierarchy(child, file, level + 1)

    with open(output_path, "w", encoding="utf-8") as f:
        # 找出所有没有父级的物体作为根节点开始遍历
        root_objects = [o for o in bpy.context.scene.objects if o.parent is None]
        for root in root_objects:
            write_obj_hierarchy(root, f)

    print("✅ 已导出层级结构树")

    # 5. 分辨率与相机
    scene.render.resolution_x = RESOLUTION[0]
    scene.render.resolution_y = RESOLUTION[1]
    
    cameras = [obj for obj in bpy.data.objects if obj.type == 'CAMERA']
    if cameras:
        scene.camera = cameras[0]
    else:
        cam_data = bpy.data.cameras.new("Camera")
        cam_obj = bpy.data.objects.new("Camera", cam_data)
        scene.collection.objects.link(cam_obj)
        k = 0.7
        cam_obj.location = (2.8342 * k, 1.9493 * k, 2.5283 * k)
        cam_obj.rotation_euler = (math.radians(54.959), 0, math.radians(125.89))
        scene.camera = cam_obj

    # 6. 输出路径与视频格式设置 (关键修改部分)
    # Worker模式使用指定的输出目录
    if args.worker_mode and args.worker_output_dir:
        output_dir = args.worker_output_dir
    else:
        output_dir = OUTPUT_DIR_BASE
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 设置完整的文件输出路径 (包括文件名和后缀)
    video_filename = args.video_name if args.video_name else "output.mp4"
    
    if args.use_ffmpeg or args.worker_mode:
        # 模式 B: 先渲染图片序列 (worker模式也使用此方式)
        if args.worker_mode:
            frames_dir = output_dir  # worker直接输出到worker目录
        else:
            frames_dir = os.path.join(output_dir, "temp_frames")
            
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            
        # Blender 会自动追加帧号 (e.g., frame_0001.png)
        scene.render.filepath = os.path.join(frames_dir, "frame_")
        scene.render.image_settings.file_format = 'PNG'
        print(f"--- Output Configured: Image Sequence at {scene.render.filepath} ---")
    else:
        scene.render.filepath = os.path.join(output_dir, video_filename)
        
        # 设置为 FFmpeg 视频输出
        scene.render.image_settings.file_format = 'FFMPEG'
        
        # 设置视频编码参数
        scene.render.ffmpeg.format = 'MPEG4'        # 容器格式 (mp4)
        scene.render.ffmpeg.codec = 'H264'          # 视频编码 (H.264)
        scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'  # 质量控制 (HIGH, MEDIUM, LOW)
        scene.render.ffmpeg.gopsize = 18            # 关键帧间隔
        
        # 如果不需要音频，可以禁用
        scene.render.ffmpeg.audio_codec = 'NONE'

        print(f"--- Output Configured: {scene.render.filepath} (H.264 MP4) ---")

    return True

def render(args):
    """执行渲染"""
    import bpy
    
    # animation=True 会自动渲染整个范围并编码为视频
    bpy.ops.render.render(animation=True)

    if args.use_ffmpeg:
        print("--- combining frames with ffmpeg ---")
        
        # Worker模式下使用worker_output_dir
        if args.worker_mode and args.worker_output_dir:
            output_base = args.worker_output_dir
        else:
            output_base = OUTPUT_DIR_BASE
            
        video_filename = args.video_name if args.video_name else "output.mp4"
        output_video_path = os.path.join(output_base, video_filename)
        
        if args.worker_mode:
            # Worker模式不合成视频，只渲染图片
            print(f"   Worker mode: frames saved to {output_base}")
            return
            
        frames_dir = os.path.join(output_base, "temp_frames")
        
        # 输入模式：frame_XXXX.png (%04d 是 Blender 默认填充)
        input_pattern = os.path.join(frames_dir, "frame_%04d.png")
        
        # 获取帧率
        fps = bpy.context.scene.render.fps
        start_frame = bpy.context.scene.frame_start

        # 构建 FFmpeg 命令
        # -y: 覆盖输出
        # -start_number: 序列起始帧
        command = [
            "ffmpeg", "-y",
            "-framerate", str(fps),
            "-start_number", str(start_frame),
            "-i", input_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            output_video_path
        ]
        
        try:
            subprocess.check_call(command)
            print(f"Success: Video saved to {output_video_path}")
        except Exception as e:
            print(f"Error running ffmpeg: {e}")

if __name__ == "__main__":
    args = get_args()
    
    # 检查是否在Blender环境中
    in_blender = is_blender_environment()
    
    # 并行渲染模式 (在Blender外部运行)
    if args.parallel and not in_blender:
        run_parallel_render(args)
    elif in_blender:
        # 在Blender内部运行 (正常模式或worker模式)
        # 1. 设置场景并准备渲染
        should_render = setup_scene(args)
        
        # 2. 直接渲染出视频
        if should_render:
            render(args)
    else:
        # 非并行模式，但也不在Blender中，启动单个Blender进程
        script_path = os.path.abspath(__file__)
        cmd = [
            "/home/jizy/tool/blender/blender", "--background", "--python", script_path,
            "--",
            "--usd_path", args.usd_path,
        ]
        if args.start is not None:
            cmd.extend(["--start", str(args.start)])
        if args.end is not None:
            cmd.extend(["--end", str(args.end)])
        cmd.extend(["--step", str(args.step)])
        cmd.extend(["--video_name", args.video_name])
        if args.use_ffmpeg:
            cmd.append("--use_ffmpeg")
        
        print(f"Starting Blender: {' '.join(cmd)}")
        subprocess.run(cmd)