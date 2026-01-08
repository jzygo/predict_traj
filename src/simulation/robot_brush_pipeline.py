"""End-to-end pipeline that links the Panda arm trajectory planner with the
XPBD brush simulator.

The script:
1. Loads handwriting sample ``data_id`` from the dataset.
2. Uses the inverse-kinematics trajectory optimizer to generate Panda joint
   angles and the tracked stick-tip path.
3. Converts the predicted stick-tip path into displacement commands for the
   brush and runs the XPBD-based differentiable simulator.
4. Renders a 3D video comparing the target path and the simulated brush
   particle positions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import warp as wp
import pyrender
import trimesh

# Ensure project root is on the Python path so that intra-project imports work even
# when this script is executed directly from the simulation folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import BRUSH_UP_POSITION, MAX_GRAVITY  # noqa: E402
from control.robot_trajectory_optimizer import (
    TrajectoryOptimizerResult,
    filter_panda_arm_joints,
    optimize_trajectory,
    resize_points_space,
    undo_resize_points_space,
)  # noqa: E402
from load_data import DataLoader  # noqa: E402
from simulation.model.brush import Brush  # noqa: E402
from simulation.model.robot_model import load_panda_model  # noqa: E402
from simulation.model.robot_visual import RobotVisualizer  # noqa: E402
from simulation.xpbd_warp_diff import XPBDSimulator  # noqa: E402


def load_dataset_entry(data_folder: Path, combined_data: Path, data_id: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load handwriting sample and grayscale reference image for ``data_id``."""

    loader = DataLoader(gif_folder_path=str(data_folder), data_path=str(combined_data), max_items=1000)
    data = loader.load_data()
    if data_id not in data:
        raise KeyError(f"data_id '{data_id}' not found in dataset")
    entry = data[data_id]
    reference_image = Image.fromarray(entry["image"]).convert("L")
    reference_image_np = 1.0 - np.array(reference_image, dtype=np.float64) / 255.0
    points = np.asarray(entry["points"], dtype=np.float64)
    return points, reference_image_np


def build_brush() -> Brush:
    root_position = torch.tensor([0.0, 0.0, BRUSH_UP_POSITION], dtype=torch.float64)
    return Brush(
        radius=0.04,
        max_length=0.3,
        max_hairs=1200,
        max_particles_per_hair=30,
        thickness=0.01,
        root_position=root_position,
        tangent_vector=torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
        length_ratio=2 / 6,
    )

def tracked_tip_to_motion(tracked_tip: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert tracked tip path to simulator displacements while preserving world-space coordinates."""

    tip_world = torch.tensor(tracked_tip, dtype=torch.float64)
    tip_normalized_np = undo_resize_points_space(tip_world.cpu().detach().numpy())
    tip_normalized = torch.tensor(tip_normalized_np, dtype=torch.float64)
    displacements = tip_normalized[1:] - tip_normalized[:-1]
    return tip_world, tip_normalized, displacements


def run_simulation(
    brush: Brush,
    displacements: torch.Tensor,
    reference_image: np.ndarray,
    canvas_resolution: int,
) -> torch.Tensor:
    """Run XPBD simulation and return particle trajectories for batch=1."""

    sim_kwargs = {
        "dt": 1.0 / 60.0,
        "substeps": 1,
        "iterations": 200,
        "num_steps": int(displacements.shape[0]),
        "batch_size": 1,
        "gravity": torch.tensor([0.0, 0.0, -MAX_GRAVITY], dtype=torch.float64),
        "dis_compliance": 1e-8,
        "variable_dis_compliance": 1e-8,
        "angle_compliance": 1e-1,
        "damping": 0.2,
        "canvas_resolution": canvas_resolution,
    }
    simulator = XPBDSimulator(**sim_kwargs)
    simulator.load_brush(brush)
    simulator.load_displacements(displacements)
    simulator.set_loss(reference_image)
    simulator.step(0, sim_kwargs["num_steps"])
    wp.synchronize_device(simulator.device)
    positions = wp.to_torch(simulator.positions).detach().cpu()
    return positions


def extract_sim_tip_path(positions: np.ndarray) -> np.ndarray:
    """Extract the lowest particle per timestep to represent the brush tip."""

    tip_positions = []
    for coords in positions:
        tip_index = int(np.argmin(coords[:, 2]))
        tip_positions.append(coords[tip_index])
    return np.asarray(tip_positions, dtype=np.float64)


def _convert_simulation_to_world(points: np.ndarray) -> np.ndarray:
    """Map normalized simulation coordinates back to Panda workspace coordinates."""

    original_shape = points.shape
    points_flat = points.reshape(-1, 3)
    world_flat = resize_points_space(points_flat)
    return world_flat.reshape(original_shape)


def _align_simulation_outputs(
    particle_history_world: np.ndarray,
    simulated_tip_world: np.ndarray,
    target_tip_world: np.ndarray,
    stick_half_length: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Align simulator outputs to the Panda tip trajectory using a best-fit translation."""

    if simulated_tip_world.size == 0 or target_tip_world.size == 0:
        return particle_history_world, simulated_tip_world

    effective = min(simulated_tip_world.shape[0], target_tip_world.shape[0])
    if effective <= 0:
        offset = np.zeros(3, dtype=np.float64)
    else:
        # Anchor the simulation to start exactly at the tracked tip to keep rod/tip aligned.
        offset = target_tip_world[0] - simulated_tip_world[0]
        offset[2] = -0.05

    aligned_particles = particle_history_world + offset[None, None, :]
    aligned_tip = simulated_tip_world + offset[None, :]
    return aligned_particles, aligned_tip


def _compute_axis_limits(arrays: List[np.ndarray], margin: float = 0.05) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Compute common axis limits with an optional margin."""

    flattened = [arr.reshape(-1, 3) for arr in arrays if arr.size]
    if not flattened:
        raise ValueError("No data available to derive axis limits")
    stacked = np.concatenate(flattened, axis=0)
    mins = stacked.min(axis=0)
    maxs = stacked.max(axis=0)
    spans = np.maximum(maxs - mins, margin)
    mins -= margin * spans
    maxs += margin * spans
    return (float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1])), (float(mins[2]), float(maxs[2]))


def _make_colored_mesh(mesh: trimesh.Trimesh, color: np.ndarray | None = None) -> trimesh.Trimesh:
    result = mesh.copy()
    if color is not None:
        rgba = np.asarray(color, dtype=np.float32)
        if rgba.size == 3:
            rgba = np.concatenate([rgba, np.array([1.0], dtype=np.float32)])
        rgba = np.clip(rgba, 0.0, 1.0)
        vertex_colors = (rgba * 255).astype(np.uint8)
        result.visual.vertex_colors = np.tile(vertex_colors, (result.vertices.shape[0], 1))
    return result


def _tube_mesh(points: np.ndarray, radius: float, color: np.ndarray) -> trimesh.Trimesh | None:
    if points.shape[0] < 2:
        return None
    segments: List[trimesh.Trimesh] = []
    rgba = np.asarray(color, dtype=np.float32)
    for start, end in zip(points[:-1], points[1:]):
        direction = end - start
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        unit = direction / length
        try:
            alignment = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), unit, return_angle=False)
        except ValueError:
            alignment = np.eye(4)
        if alignment.shape == (3, 3):
            transform = np.eye(4)
            transform[:3, :3] = alignment
        else:
            transform = alignment.copy()
        transform[:3, 3] = (start + end) * 0.5

        cylinder = trimesh.creation.cylinder(radius=radius, height=length, sections=16)
        cylinder.apply_transform(transform)
        segments.append(cylinder)

    if not segments:
        return None

    merged = trimesh.util.concatenate(segments)
    return _make_colored_mesh(merged, rgba)


def _camera_pose(bounds_min: np.ndarray, bounds_max: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    center = (bounds_min + bounds_max) * 0.5
    span = bounds_max - bounds_min
    max_span = float(np.max(span))
    if max_span < 1e-3:
        max_span = 1.0
    eye = center + np.array([0.8, -1.0, 0.6]) * max_span
    forward = center - eye
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        forward = np.array([0.0, 0.0, 1.0])
        forward_norm = 1.0
    forward /= forward_norm
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
        right_norm = 1.0
    right /= right_norm
    up = np.cross(right, forward)
    up /= np.linalg.norm(up)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = -forward
    pose[:3, 3] = eye
    return pose, center


def render_simulation_video(
    particle_history_world: np.ndarray,
    target_tip_world: np.ndarray,
    simulated_tip_world: np.ndarray,
    output_path: Path,
    trajectory: TrajectoryOptimizerResult,
    fps: int = 30,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.parent / "frames_sim"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for existing in frames_dir.glob("frame_*.png"):
        existing.unlink()

    model = load_panda_model(base_xy=(0.0, 0.0))
    joint_names = filter_panda_arm_joints(model.active_joint_names("panda_hand"))
    visualizer = RobotVisualizer(model)
    target_link = "panda_hand"

    stick_half_length = 0.0
    if trajectory.tracked_tip.shape[0] and trajectory.centres.shape[0]:
        stick_half_length = float(np.linalg.norm(trajectory.tracked_tip[0] - trajectory.centres[0]))
    local_stick_a = np.concatenate([
        trajectory.grip_mid_local + trajectory.stick_dir_local * stick_half_length,
        np.array([1.0]),
    ])
    local_stick_b = np.concatenate([
        trajectory.grip_mid_local - trajectory.stick_dir_local * stick_half_length,
        np.array([1.0]),
    ])

    tracked_tip_world = trajectory.tracked_tip

    axis_arrays: List[np.ndarray] = [particle_history_world, target_tip_world, simulated_tip_world, tracked_tip_world]
    x_limits, y_limits, z_limits = _compute_axis_limits(axis_arrays)
    bounds_min = np.array([x_limits[0], y_limits[0], z_limits[0]], dtype=np.float64)
    bounds_max = np.array([x_limits[1], y_limits[1], z_limits[1]], dtype=np.float64)
    camera_pose, scene_center = _camera_pose(bounds_min, bounds_max)
    scene_extent = max(float(np.max(bounds_max - bounds_min)), 0.5)

    renderer = pyrender.OffscreenRenderer(viewport_width=1280, viewport_height=720)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(50.0))
    key_light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
    fill_light = pyrender.DirectionalLight(color=np.array([0.6, 0.7, 1.0]), intensity=1.5)

    target_tube_mesh = _tube_mesh(target_tip_world, radius=0.003, color=np.array([1.0, 0.0, 0.0, 1.0]))

    accumulated_centres: List[np.ndarray] = []

    for index, coords in enumerate(tqdm(particle_history_world, desc="Rendering simulation frames")):
        if trajectory.joint_trajectory.size == 0:
            raise RuntimeError("Trajectory has no joint configurations to render the robot")
        joint_idx = min(index, trajectory.joint_trajectory.shape[0] - 1)
        joint_values = trajectory.joint_trajectory[joint_idx]
        joint_map = {name: float(val) for name, val in zip(joint_names, joint_values)}
        transforms = model.forward_kinematics(joint_map, target_link=None)
        ee_transform = transforms[target_link]
        accumulated_centres.append(ee_transform[:3, 3].copy())
        centres_arr = np.vstack(accumulated_centres)

        stick_a = ee_transform @ local_stick_a
        stick_b = ee_transform @ local_stick_b

        scene = pyrender.Scene(bg_color=np.array([1.0, 1.0, 1.0, 0.0]), ambient_light=np.array([0.18, 0.18, 0.18, 1.0]))
        scene.add(camera, pose=camera_pose)
        scene.add(key_light, pose=camera_pose)
        light_pose = camera_pose.copy()
        light_pose[:3, 3] = scene_center + np.array([-0.6, 0.4, 0.9]) * scene_extent
        scene.add(fill_light, pose=light_pose)

        if target_tube_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(target_tube_mesh, smooth=False))

        tracked_mesh = _tube_mesh(tracked_tip_world[: index + 1], radius=0.0025, color=np.array([0.1, 0.6, 1.0, 1.0]))
        if tracked_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(tracked_mesh, smooth=False))

        simulated_mesh = _tube_mesh(simulated_tip_world[: index + 1], radius=0.0025, color=np.array([0.1, 0.8, 0.2, 1.0]))
        if simulated_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(simulated_mesh, smooth=False))

        centre_mesh = _tube_mesh(centres_arr, radius=0.002, color=np.array([0.4, 0.2, 1.0, 1.0]))
        if centre_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(centre_mesh, smooth=False))

        pen_points = np.vstack([stick_a[:3], stick_b[:3]])
        pen_mesh = _tube_mesh(pen_points, radius=0.01, color=np.array([0.55, 0.27, 0.07, 1.0]))
        if pen_mesh is not None:
            scene.add(pyrender.Mesh.from_trimesh(pen_mesh, smooth=False))

        if coords.size:
            particle_colors = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (coords.shape[0], 1))
            particle_mesh = pyrender.Mesh.from_points(coords.astype(np.float32), colors=particle_colors)
            scene.add(particle_mesh)

        for link, visuals in visualizer.visuals.items():
            link_transform = transforms.get(link)
            if link_transform is None:
                continue
            for visual in visuals:
                world_pose = link_transform @ visual.local_transform
                robot_mesh = pyrender.Mesh.from_trimesh(visual.mesh, smooth=False)
                scene.add(robot_mesh, pose=world_pose)

        plane_size = np.maximum(bounds_max - bounds_min, 0.4)
        plane_mesh = trimesh.creation.box(extents=np.array([plane_size[0], plane_size[1], 0.002]))
        plane_mesh.apply_translation(np.array([scene_center[0], scene_center[1], bounds_min[2] - 0.001]))
        plane_mesh = _make_colored_mesh(plane_mesh, np.array([0.95, 0.95, 0.95, 1.0]))
        scene.add(pyrender.Mesh.from_trimesh(plane_mesh, smooth=False))

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        frame_path = frames_dir / f"frame_{index:05d}.png"
        imageio.imwrite(frame_path, color)

    renderer.delete()

    with imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8) as writer:
        for index in range(len(particle_history_world)):
            frame_path = frames_dir / f"frame_{index:05d}.png"
            writer.append_data(imageio.imread(frame_path))


def run_pipeline(data_id: str, output_dir: Path, video_fps: int) -> Path:
    data_folder = PROJECT_ROOT / "data"
    combined_data = data_folder / "result.pkl"

    output_dir.mkdir(parents=True, exist_ok=True)
    robot_output_dir = output_dir / "robot"
    sim_output_dir = output_dir / "simulation"
    sim_output_dir.mkdir(parents=True, exist_ok=True)

    stick_half_length = 0.1

    # Step 1: compute Panda joint trajectory and tracked tip path.
    trajectory_result = optimize_trajectory(
        data_folder=str(data_folder),
        combined_data=str(combined_data),
        data_id=data_id,
        output_dir=robot_output_dir,
        stick_half_length=stick_half_length,
        visualize=False,
        render_video=False,
        show_progress=True,
        verbose=True,
    )

    # Step 2: convert tracked tip to brush displacements.
    tip_world, _, displacements = tracked_tip_to_motion(trajectory_result.tracked_tip)

    # Step 3: load dataset reference image for simulator loss setup.
    _, reference_image = load_dataset_entry(data_folder, combined_data, data_id)

    # Step 4: run XPBD simulation.
    brush = build_brush()
    particle_positions_tensor = run_simulation(
        brush,
        displacements,
        reference_image,
        reference_image.shape[0],
    )
    particle_positions = particle_positions_tensor[0].numpy()
    simulated_tip_normalized = extract_sim_tip_path(particle_positions)

    tip_world_np = tip_world.numpy()
    particle_positions_world = _convert_simulation_to_world(particle_positions)
    simulated_tip_world = _convert_simulation_to_world(simulated_tip_normalized)
    particle_positions_world, simulated_tip_world = _align_simulation_outputs(
        particle_positions_world, simulated_tip_world, tip_world_np, stick_half_length
    )

    # Step 5: render combined video.
    video_path = sim_output_dir / "brush_simulation.mp4"
    render_simulation_video(
        particle_positions_world,
        trajectory_result.reference_tip,
        simulated_tip_world,
        video_path,
        trajectory_result,
        fps=video_fps,
    )

    np.save(sim_output_dir / "simulated_tip.npy", simulated_tip_world)
    np.save(sim_output_dir / "particle_positions.npy", particle_positions_world)
    return video_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Panda trajectory + XPBD brush simulation pipeline")
    parser.add_argument("--data_id", type=str, default="19970", help="Dataset sample id to reproduce")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to store all outputs")
    parser.add_argument("--video_fps", type=int, default=30, help="Frames per second for the rendered video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        default_output = PROJECT_ROOT / "simulation" / "output" / f"run_{args.data_id}"
    else:
        default_output = Path(args.output_dir)
    video_path = run_pipeline(args.data_id, default_output, args.video_fps)
    print(f"Simulation video written to '{video_path}'")


if __name__ == "__main__":
    main()
