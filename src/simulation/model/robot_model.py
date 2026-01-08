#!/usr/bin/python3
# _*_ coding: UTF-8 _*_
#
# Copyright (C) 2022, by Leon(罗伯特祥)
#

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import xml.etree.ElementTree as ET

import numpy as np


def _to_vector(attr: Optional[str], length: int = 3) -> np.ndarray:
	values = attr.split() if attr else []
	data = np.zeros(length, dtype=float)
	if not values:
		return data
	if len(values) != length:
		raise ValueError(f"Expected {length} values, got {values}")
	return np.array([float(v) for v in values], dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
	roll, pitch, yaw = rpy
	sr, cr = math.sin(roll), math.cos(roll)
	sp, cp = math.sin(pitch), math.cos(pitch)
	sy, cy = math.sin(yaw), math.cos(yaw)
	rx = np.array([[1.0, 0.0, 0.0],
				   [0.0, cr, -sr],
				   [0.0, sr, cr]], dtype=float)
	ry = np.array([[cp, 0.0, sp],
				   [0.0, 1.0, 0.0],
				   [-sp, 0.0, cp]], dtype=float)
	rz = np.array([[cy, -sy, 0.0],
				   [sy, cy, 0.0],
				   [0.0, 0.0, 1.0]], dtype=float)
	return rz @ ry @ rx


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
	norm = np.linalg.norm(axis)
	if norm < 1e-12 or abs(angle) < 1e-12:
		return np.identity(3)
	ax = axis / norm
	x, y, z = ax
	c = math.cos(angle)
	s = math.sin(angle)
	c1 = 1.0 - c
	return np.array([[c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
					 [y * x * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
					 [z * x * c1 - y * s, z * y * c1 + x * s, c + z * z * c1]], dtype=float)


def _make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
	transform = np.identity(4)
	transform[:3, :3] = rotation
	transform[:3, 3] = translation
	return transform


def _translation_transform(translation: np.ndarray) -> np.ndarray:
	transform = np.identity(4)
	transform[:3, 3] = translation
	return transform


@dataclass
class Joint:
	name: str
	joint_type: str
	parent: str
	child: str
	origin_xyz: np.ndarray
	origin_rpy: np.ndarray
	axis: np.ndarray
	mimic: Optional[str] = None
	mimic_multiplier: float = 1.0
	mimic_offset: float = 0.0

	def origin_transform(self) -> np.ndarray:
		return _make_transform(_rpy_matrix(self.origin_rpy), self.origin_xyz)

	def motion_transform(self, position: float) -> np.ndarray:
		if self.joint_type in {"revolute", "continuous"}:
			return _make_transform(_axis_angle_matrix(self.axis, position), np.zeros(3))
		if self.joint_type == "prismatic":
			return _translation_transform(self.axis * position)
		return np.identity(4)

	def transform(self, position: float) -> np.ndarray:
		return self.origin_transform() @ self.motion_transform(position)


class RobotModel:
	def __init__(self, urdf_path: Path):
		self.urdf_path = urdf_path
		if not self.urdf_path.exists():
			raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
		self._root = ET.parse(self.urdf_path).getroot()
		self.links: Set[str] = set()
		self.joints: Dict[str, Joint] = {}
		self.children: Dict[str, List[Joint]] = {}
		self.base_link: Optional[str] = None
		self._base_transform = np.identity(4)
		self._parse()

	def _parse(self) -> None:
		self.links = {link.get("name") for link in self._root.findall("link")}
		child_links = set()
		for elem in self._root.findall("joint"):
			name = elem.get("name")
			joint_type = elem.get("type", "fixed")
			parent = elem.find("parent").get("link")
			child = elem.find("child").get("link")
			child_links.add(child)
			origin_elem = elem.find("origin")
			xyz = _to_vector(origin_elem.get("xyz") if origin_elem is not None else None)
			rpy = _to_vector(origin_elem.get("rpy") if origin_elem is not None else None)
			axis_elem = elem.find("axis")
			axis = _to_vector(axis_elem.get("xyz") if axis_elem is not None else None)
			mimic_elem = elem.find("mimic")
			mimic = mimic_elem.get("joint") if mimic_elem is not None else None
			multiplier = float(mimic_elem.get("multiplier", "1.0")) if mimic_elem is not None else 1.0
			offset = float(mimic_elem.get("offset", "0.0")) if mimic_elem is not None else 0.0
			joint = Joint(
				name=name,
				joint_type=joint_type,
				parent=parent,
				child=child,
				origin_xyz=xyz,
				origin_rpy=rpy,
				axis=axis,
				mimic=mimic,
				mimic_multiplier=multiplier,
				mimic_offset=offset,
			)
			self.joints[name] = joint
			self.children.setdefault(parent, []).append(joint)
		bases = self.links - child_links
		if not bases:
			raise ValueError("No base link detected")
		if len(bases) > 1:
			raise ValueError(f"Multiple base links detected: {bases}")
		self.base_link = bases.pop()

	def _resolve_joint_value(self, joint: Joint, joint_map: Dict[str, float], cache: Dict[str, float]) -> float:
		if joint.name in cache:
			return cache[joint.name]
		if joint.joint_type == "fixed":
			value = 0.0
		elif joint.mimic:
			ref_joint = self.joints[joint.mimic]
			ref_value = self._resolve_joint_value(ref_joint, joint_map, cache)
			value = joint.mimic_multiplier * ref_value + joint.mimic_offset
		else:
			value = joint_map.get(joint.name, 0.0)
		cache[joint.name] = value
		return value

	def forward_kinematics(self, joint_map: Dict[str, float], target_link: Optional[str] = None) -> Union[Dict[str, np.ndarray], np.ndarray]:
		if self.base_link is None:
			raise RuntimeError("Robot base link not initialized")
		transforms: Dict[str, np.ndarray] = {self.base_link: self._base_transform.copy()}
		cache: Dict[str, float] = {}

		def traverse(link: str) -> None:
			base_transform = transforms[link]
			for joint in self.children.get(link, []):
				position = self._resolve_joint_value(joint, joint_map, cache)
				transforms[joint.child] = base_transform @ joint.transform(position)
				traverse(joint.child)

		traverse(self.base_link)
		if target_link is None:
			return transforms
		if target_link not in transforms:
			raise KeyError(f"Unknown link '{target_link}'")
		return transforms[target_link]

	def get_chain(self, target_link: str) -> List[Joint]:
		if self.base_link is None:
			raise RuntimeError("Robot base link not initialized")
		stack: List[tuple[str, List[Joint]]] = [(self.base_link, [])]
		visited = set()
		while stack:
			link, path = stack.pop()
			if link == target_link:
				return path
			if link in visited:
				continue
			visited.add(link)
			for joint in self.children.get(link, []):
				stack.append((joint.child, path + [joint]))
		raise KeyError(f"Link '{target_link}' not reachable from base")

	def active_joint_names(self, target_link: Optional[str] = None) -> List[str]:
		joints = self.get_chain(target_link) if target_link else list(self.joints.values())
		names = [joint.name for joint in joints if joint.joint_type != "fixed"]
		return names

	def set_base_position(self, x: float, y: float, z: float = 0.0) -> None:
		translation = np.array([x, y, z], dtype=float)
		transform = np.identity(4)
		transform[:3, 3] = translation
		self._base_transform = transform


def load_panda_model(base_xy: Tuple[float, float] = (0.0, 0.0), base_z: float = 0.0) -> RobotModel:
	root = Path(__file__).resolve().parents[2]
	urdf_path = root / "data" / "PandaRobot" / "deps" / "Panda" / "panda.urdf"
	model = RobotModel(urdf_path)
	model.set_base_position(base_xy[0], base_xy[1], base_z)
	return model


def transform_to_position_quaternion(transform: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
	rotation = transform[:3, :3]
	trace = np.trace(rotation)
	if trace > 0:
		s = math.sqrt(trace + 1.0) * 2.0
		w = 0.25 * s
		x = (rotation[2, 1] - rotation[1, 2]) / s
		y = (rotation[0, 2] - rotation[2, 0]) / s
		z = (rotation[1, 0] - rotation[0, 1]) / s
	else:
		diag = rotation.diagonal()
		idx = int(np.argmax(diag))
		if idx == 0:
			s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
			w = (rotation[2, 1] - rotation[1, 2]) / s
			x = 0.25 * s
			y = (rotation[0, 1] + rotation[1, 0]) / s
			z = (rotation[0, 2] + rotation[2, 0]) / s
		elif idx == 1:
			s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
			w = (rotation[0, 2] - rotation[2, 0]) / s
			x = (rotation[0, 1] + rotation[1, 0]) / s
			y = 0.25 * s
			z = (rotation[1, 2] + rotation[2, 1]) / s
		else:
			s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
			w = (rotation[1, 0] - rotation[0, 1]) / s
			x = (rotation[0, 2] + rotation[2, 0]) / s
			y = (rotation[1, 2] + rotation[2, 1]) / s
			z = 0.25 * s
	quaternion = np.array([w, x, y, z], dtype=float)
	position = transform[:3, 3].copy()
	return position, quaternion


if __name__ == "__main__":
	model = load_panda_model()
	chain = model.get_chain("panda_hand")
	active_names = [joint.name for joint in chain if joint.joint_type != "fixed"]
	sample_config = {name: 0.0 for name in active_names}
	pose = model.forward_kinematics(sample_config, target_link="panda_hand")
	position, quaternion = transform_to_position_quaternion(pose)
	print("Active joints:", active_names)
	print("End effector position:", position)
	print("End effector quaternion:", quaternion)
