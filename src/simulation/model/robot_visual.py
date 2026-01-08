#!/usr/bin/python3
"""URDF-based visualizer for the Panda robot using trimesh."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

import trimesh  # type: ignore

from simulation.model.robot_model import (
	RobotModel,
	_make_transform,
	_rpy_matrix,
	load_panda_model,
)


def _parse_vector(attr: Optional[str], default: Iterable[float]) -> np.ndarray:
	if not attr:
		return np.array(list(default), dtype=float)
	values = [float(v) for v in attr.split()]
	return np.array(values, dtype=float)


def _scale_transform(scale: np.ndarray) -> np.ndarray:
	transform = np.identity(4)
	transform[0, 0] = scale[0]
	transform[1, 1] = scale[1]
	transform[2, 2] = scale[2]
	return transform


@dataclass
class VisualGeometry:
	mesh: trimesh.Trimesh
	local_transform: np.ndarray


class RobotVisualizer:
	def __init__(self, model: RobotModel):
		self.model = model
		self.urdf_root = self.model._root
		self.urdf_path = self.model.urdf_path
		self.package_root = self.urdf_path.parent
		self.visuals: Dict[str, List[VisualGeometry]] = {}
		self._collect_visuals()

	def _collect_visuals(self) -> None:
		for link in self.urdf_root.findall("link"):
			link_name = link.get("name")
			items: List[VisualGeometry] = []
			for visual in link.findall("visual"):
				origin = visual.find("origin")
				xyz_attr = origin.get("xyz") if origin is not None else None
				rpy_attr = origin.get("rpy") if origin is not None else None
				xyz = _parse_vector(xyz_attr, default=(0.0, 0.0, 0.0))
				rpy = _parse_vector(rpy_attr, default=(0.0, 0.0, 0.0))
				geometry = visual.find("geometry")
				if geometry is None:
					continue
				mesh_elem = geometry.find("mesh")
				if mesh_elem is None:
					continue
				filename = mesh_elem.get("filename")
				if not filename:
					continue
				mesh_path = self._resolve_mesh_path(filename)
				mesh = self._load_mesh(mesh_path)
				scale_attr = mesh_elem.get("scale")
				scale = _parse_vector(scale_attr, default=(1.0, 1.0, 1.0))
				local = _make_transform(_rpy_matrix(rpy), xyz) @ _scale_transform(scale)
				items.append(VisualGeometry(mesh=mesh, local_transform=local))
			if items:
				self.visuals[link_name] = items

	def _resolve_mesh_path(self, filename: str) -> Path:
		if filename.startswith("package://"):
			remainder = filename[len("package://") :]
			parts = Path(remainder)
			if len(parts.parts) < 2:
				raise ValueError(f"Cannot resolve mesh path '{filename}'")
			package_dir = self.package_root
			mesh_relative = Path(*parts.parts[1:])
			return (package_dir / mesh_relative).resolve()
		candidate = (self.urdf_path.parent / filename).resolve()
		if candidate.exists():
			return candidate
		raise FileNotFoundError(f"Mesh file not found: {filename}")

	@staticmethod
	def _load_mesh(path: Path) -> trimesh.Trimesh:
		loaded = trimesh.load(path, force="scene", process=False)
		if isinstance(loaded, trimesh.Trimesh):
			return loaded
		if isinstance(loaded, trimesh.Scene):
			geometries = [geo for geo in loaded.geometry.values()]
			if not geometries:
				raise ValueError(f"Mesh at '{path}' is empty")
			return trimesh.util.concatenate(geometries)
		raise TypeError(f"Unsupported mesh type loaded from '{path}'")

	def build_scene(self, joint_positions: Optional[Dict[str, float]] = None) -> trimesh.Scene:
		joint_positions = joint_positions or {}
		transforms = self.model.forward_kinematics(joint_positions)
		scene = trimesh.Scene()
		for link, visuals in self.visuals.items():
			link_transform = transforms.get(link)
			if link_transform is None:
				continue
			for index, visual in enumerate(visuals):
				mesh = visual.mesh.copy()
				mesh.apply_transform(link_transform @ visual.local_transform)
				node_name = f"{link}_visual_{index}"
				scene.add_geometry(mesh, node_name=node_name)
		return scene

	def show(self, joint_positions: Optional[Dict[str, float]] = None, **viewer_kwargs) -> None:
		scene = self.build_scene(joint_positions)
		scene.show(**viewer_kwargs)


def main() -> None:
	model = load_panda_model()
	visualizer = RobotVisualizer(model)
	joint_map: Dict[str, float] = {name: 0.0 for name in model.active_joint_names("panda_hand")}
	visualizer.show(joint_map)


if __name__ == "__main__":
	main()
