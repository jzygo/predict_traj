import bpy
import sys
import os
import math
import argparse
import subprocess

# --- 全局配置 ---
OUTPUT_DIR_BASE = "/home/jizy/project/video_tokenizer/src/simulation/render/output_frames"
RESOLUTION = (1280, 720)
USE_GPU = True 

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

    return parser.parse_args(argv)

def auto_detect_frame_range():
    """(保持原逻辑不变) 根据导入的物体自动检测动画范围"""
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

    # 3. 渲染引擎设置 - 使用 Eevee (实时渲染，速度快)
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Blender 4.2+ 使用 EEVEE Next
    
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

    # 4. GPU 设置 (Eevee 默认使用 GPU，无需额外配置)
    # Eevee 自动使用 GPU 加速，比 Cycles 快很多
    print(f"--- Using Eevee Engine (GPU Accelerated) ---")

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
    if not os.path.exists(OUTPUT_DIR_BASE):
        os.makedirs(OUTPUT_DIR_BASE, exist_ok=True)
    
    # 设置完整的文件输出路径 (包括文件名和后缀)
    video_filename = args.video_name if args.video_name else "output.mp4"
    
    if args.use_ffmpeg:
        # 模式 B: 先渲染图片序列
        frames_dir = os.path.join(OUTPUT_DIR_BASE, "temp_frames")
        if not os.path.exists(frames_dir):
            os.makedirs(frames_dir, exist_ok=True)
            
        # Blender 会自动追加帧号 (e.g., frame_0001.png)
        scene.render.filepath = os.path.join(frames_dir, "frame_")
        scene.render.image_settings.file_format = 'PNG'
        print(f"--- Output Configured: Image Sequence at {scene.render.filepath} ---")
    else:
        scene.render.filepath = os.path.join(OUTPUT_DIR_BASE, video_filename)
        
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
    # animation=True 会自动渲染整个范围并编码为视频
    bpy.ops.render.render(animation=True)

    if args.use_ffmpeg:
        print("--- combining frames with ffmpeg ---")
        video_filename = args.video_name if args.video_name else "output.mp4"
        output_video_path = os.path.join(OUTPUT_DIR_BASE, video_filename)
        frames_dir = os.path.join(OUTPUT_DIR_BASE, "temp_frames")
        
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
    
    # 1. 设置场景并准备渲染
    should_render = setup_scene(args)
    
    # 2. 直接渲染出视频
    if should_render:
        render(args)