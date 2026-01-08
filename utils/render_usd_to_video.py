import bpy
import math
import os


def render_usd_to_video(usd_path, output_path, resolution=(1920, 1080), start_frame=1, end_frame=100):
    """
    加载USD文件，设置固定相机并渲染为视频。
    """

    # 1. 重置场景 (删除默认的立方体、相机、灯光)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # 2. 导入 USD 文件
    # 注意：根据Blender版本，参数可能略有不同，这里适用于 Blender 3.x/4.x
    print(f"正在导入 USD: {usd_path}")
    bpy.ops.wm.usd_import(filepath=usd_path, scale=1.0)

    # 3. 设置渲染引擎 (Eevee 较快，Cycles 质量较高)
    scene = bpy.context.scene
    scene.render.engine = 'BLENDER_EEVEE_NEXT'  # Blender 4.2+ 使用 EEVEE_NEXT，旧版使用 BLENDER_EEVEE
    # 如果想用 Cycles，取消下面注释:
    # scene.render.engine = 'CYCLES'
    # scene.cycles.device = 'GPU'

    # 4. 设置固定的相机
    # 创建相机数据和对象
    cam_data = bpy.data.cameras.new(name='FixedCamera')
    cam_obj = bpy.data.objects.new(name='FixedCamera', object_data=cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj  # 将其设为当前渲染相机

    # --- 【关键：在此处修改相机位姿】 ---
    # 位置 (X, Y, Z)
    cam_obj.location = (0, -5, 3)
    # 旋转 (X, Y, Z) - Blender使用弧度制
    # 这里的 rotation 代表相机略微向下俯视
    cam_obj.rotation_euler = (math.radians(60), 0, 0)

    # 5. 添加基础灯光 (防止场景全黑)
    # 添加一个日光 (Sun Light)
    light_data = bpy.data.lights.new(name="Sun", type='SUN')
    light_data.energy = 5.0  # 强度
    light_obj = bpy.data.objects.new(name="Sun", object_data=light_data)
    scene.collection.objects.link(light_obj)
    light_obj.location = (5, -5, 10)
    light_obj.rotation_euler = (math.radians(45), math.radians(30), 0)

    # 也可以添加一个环境光 (World Background)
    world = bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs[0].default_value = (0.5, 0.5, 0.5, 1)  # 灰色环境光
    bg_node.inputs[1].default_value = 1.0  # 强度

    # 6. 配置输出参数 (视频)
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100

    # 设置帧范围
    scene.frame_start = start_frame
    scene.frame_end = end_frame

    # 设置输出路径和格式
    scene.render.filepath = output_path
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'  # 容器格式 MP4
    scene.render.ffmpeg.codec = 'H264'  # 编码格式 H.264
    scene.render.ffmpeg.constant_rate_factor = 'MEDIUM'  # 质量

    # 7. 开始渲染
    print(f"开始渲染视频到: {output_path} ...")
    bpy.ops.render.render(animation=True)
    print("渲染完成！")


# ================= 配置区域 =================
if __name__ == "__main__":
    # 请修改这里的路径 (建议使用绝对路径)
    # Windows 示例: r"C:\Users\Name\Desktop\model.usd"
    # Linux/Mac 示例: "/home/user/model.usd"

    INPUT_USD = r"/path/to/your/file.usd"
    OUTPUT_VIDEO = r"/path/to/output/render.mp4"

    # 确保输出目录存在
    output_dir = os.path.dirname(OUTPUT_VIDEO)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 运行函数 (假设动画有100帧)
    render_usd_to_video(INPUT_USD, OUTPUT_VIDEO, start_frame=1, end_frame=100)