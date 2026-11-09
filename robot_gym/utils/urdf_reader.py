"""Resolve a robot description and read joints and candidate foot links in policy order."""

import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ressources import ROBOTS_DIR
from robot_gym import ROBOT_GYM_ROOT_DIR


class URDFReader:
    def __init__(self, robot_file_name: str):
        requested = Path(robot_file_name)
        self.robot_file_name = requested.name
        self.robot_name = requested.stem
        self.robot_file_format = requested.suffix.lower().lstrip(".")
        if self.robot_file_format not in ("urdf", "xml"):
            raise ValueError(f"Unsupported robot file format: {requested.suffix}. Expected .urdf or .xml.")

        self.relevant_paths_dict = {"project_root": Path(ROBOT_GYM_ROOT_DIR)}
        if requested.is_absolute() or requested.parent != Path("."):
            path = requested.resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            robots = Path(ROBOTS_DIR)
            matches = [(folder, path) for folder in robots.iterdir() if folder.is_dir()
                       for path in folder.rglob(requested.name) if path.is_file()]
            if not matches:
                raise FileNotFoundError(f"Robot file '{requested}' not found under {robots}")
            if len(matches) != 1:
                raise ValueError(f"Ambiguous robot filename '{requested}': {matches}; supply an explicit path")
            folder, path = matches[0]
            self.relevant_paths_dict[folder.name] = folder.resolve()
            path = path.resolve()

        self.robot_file_path_absolute = path
        self.robot_file_path_relative = Path(os.path.relpath(path, ROBOT_GYM_ROOT_DIR))
        self.relevant_paths_dict[self.robot_file_format] = path.parent
        self.root = ET.parse(path).getroot()
        joints = self.root.findall(".//joint")
        self.joint_names = [joint.get("name") for joint in joints
                            if joint.get("type", "").lower() != "fixed" and joint.get("name")]
        if self.robot_file_format == "xml":
            candidates = [body.get("name") for body in self.root.findall(".//body")
                          if "foot" in body.get("name", "").lower()]
            if not candidates:
                raise RuntimeError(f"No foot bodies found in XML/MJCF at {path}")
            left = [name for name in candidates if "Left" in name or "left" in name]
            right = [name for name in candidates if "Right" in name or "right" in name]
        else:
            candidates = self._urdf_feet(joints)
            left = [name for name in candidates if "left" in name.lower()]
            right = [name for name in candidates if "right" in name.lower()]
        self.foot_link_names = (left + right) or candidates

    @staticmethod
    def _urdf_feet(joints):
        """Use terminal links, resolving a fixed terminal joint to its parent."""
        parents = set()
        children = set()
        fixed_parents = {}
        for joint in joints:
            parent, child = joint.find("parent"), joint.find("child")
            parent_name = parent.get("link") if parent is not None else None
            child_name = child.get("link") if child is not None else None
            if parent_name:
                parents.add(parent_name)
            if child_name:
                children.add(child_name)
            if parent_name and child_name and joint.get("type", "").lower() == "fixed":
                fixed_parents[child_name] = parent_name
        return list(dict.fromkeys(fixed_parents.get(link, link) for link in sorted(children - parents)))
