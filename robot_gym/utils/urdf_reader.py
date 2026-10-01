from __future__ import annotations
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ressources import ROBOTS_DIR
from robot_gym import ROBOT_GYM_ROOT_DIR

class URDFReader():
    """
    This class solves all problems that usually appear when different people work
    on the same robotics project on different computers. Instead of hard-coding
    paths or joint names, FileFormatAndPaths automatically:

    1. finds the project root, no matter where the code is located,
    2. locates all important folders (urdf/, dodo_robot/, etc.),
    3. selects the correct robot file (URDF or XML),
    4. reads the robot file and extracts all joint names directly from it.

    As a result, everyone can run the project without manually adjusting paths or
    editing joint name lists. The class guarantees that the setup works everywhere
    and stays consistent even if file structures change.
    """

    def __init__(self, robot_file_name: str):
        # public members
        requested = Path(robot_file_name)
        self._explicit_path = requested.resolve() if requested.is_absolute() or requested.parent != Path('.') else None
        self.robot_file_name: str = requested.name
        self.robot_name: str = Path(robot_file_name).stem
        self.robot_file_format: str = ""
        self._check_robot_file_name()

        self.relevant_paths_dict: dict[str, Path] = self._get_paths()
        self.robot_file_path_absolute: Path = self.relevant_paths_dict[self.robot_file_format] / self.robot_file_name
        self.joint_names: list[str] = self._get_joint_names()
        self.foot_link_names: list[str] = self._get_foot_link_names()
        self.robot_file_path_relative: Path = self._get_relative_robot_file_path()

        # protected members

    def _check_robot_file_name(self):
        suffix = Path(self.robot_file_name).suffix.lower()

        if suffix == ".urdf":
            print(f"Loading URDF file: {self.robot_file_name}")
            self.robot_file_format = "urdf"

        elif suffix == ".xml":
            print(f"Loading XML file: {self.robot_file_name}")
            self.robot_file_format = "xml"

        else:
            raise ValueError(
                f"Unsupported robot file format '{suffix}'. Expected .urdf or .xml."
            )

    def _get_paths(self) -> dict[str, Path]:
        """
        Returns a Dict containing relevant paths, OS-independent:
        - 'project_root': project root
        - 'cwd': current working directory
        - per forldername -> absolute Path
        Throw FileNotFoundError, if required_dirs is missing.
        
        Example return:
        paths 0 = {
            'project_root': WindowsPath('.../bipedal_robot_sim'), 
            'humanoid_robot': WindowsPath('.../bipedal_robot_sim/humanoid_robot'), 
            'bipedal_robot': WindowsPath('.../bipedal_robot_sim/bipedal_robot'), 
            'urdf': WindowsPath('.../bipedal_robot_sim/bipedal_robot/urdf')
        """
        project_root = Path(ROBOT_GYM_ROOT_DIR)
        result: dict[str, Path] = {
            "project_root": project_root,
        }

        if self._explicit_path is not None:
            if not self._explicit_path.is_file():
                raise FileNotFoundError(self._explicit_path)
            result[self.robot_file_format] = self._explicit_path.parent
            return result

        robots_folder = Path(ROBOTS_DIR)

        robot_dirs = [p for p in robots_folder.iterdir() if p.is_dir()]

        matches = [(folder, path.resolve()) for folder in robot_dirs
                   for path in folder.rglob(self.robot_file_name)]
        if not matches:
            raise FileNotFoundError(
                f"Robot file '{self.robot_file_name}' not found in any subfolder of {robots_folder}"
            )
        if len(matches) != 1:
            raise ValueError(f"Ambiguous robot filename '{self.robot_file_name}': {[str(p) for _, p in matches]}; supply an explicit path")
        folder, file_path = matches[0]
        result[folder.name] = folder.resolve()
        result[self.robot_file_format] = file_path.parent

        return result


    def _extract_joints_from_urdf(self) -> list[str]:
        """Extract all non-fixed joints from URDF in the exact declared order."""

        tree = ET.parse(self.robot_file_path_absolute)
        root = tree.getroot()

        joint_names = []
        for joint in root.findall(".//joint"):
            jtype = joint.attrib.get("type", "").lower()
            if jtype != "fixed":   # remove this check if you want fixed joints too
                name = joint.attrib.get("name")
                if name:
                    joint_names.append(name)

        return joint_names
    

    def _extract_joints_from_xml(self) -> list[str]:
        """Extract all non-fixed joints from an MJCF XML in the exact declared order."""
        import xml.etree.ElementTree as ET

        tree = ET.parse(self.robot_file_path_absolute)
        root = tree.getroot()

        joint_names = []

        # In MJCF, joints are usually under <worldbody> or <body> tags,
        # but we search globally for safety
        for joint in root.findall(".//joint"):
            jtype = joint.attrib.get("type", "").lower()

            # skip fixed joints (optional)
            if jtype == "fixed":
                continue

            name = joint.attrib.get("name")
            if name:
                joint_names.append(name)

        return joint_names


    def _get_joint_names(self) -> list[str]:
        """Returns joint names exactly as they appear in the robot file."""
        file = str(self.robot_file_path_absolute)

        if file.endswith(".urdf"):
            return self._extract_joints_from_urdf()

        elif file.endswith(".xml"):
            return self._extract_joints_from_xml()  # if needed

        else:
            raise ValueError(f"Unsupported robot file format: {file}")

    
    def _get_foot_link_names(self):
        """
        Extract foot link names from the robot description file.

        XML/MJCF:
            - search all <body name="..."> and pick those that look like feet,
              e.g. contain 'FOOT' (Left_FOOT_FE, Right_FOOT_FE).
            - result is ordered left, then right if possible.

        URDF:
            - treat "feet" as end links of the kinematic chain:
              links that appear as <child link="..."> of some joint,
              but never as <parent link="...">.
            - result is ordered left, then right if possible.
        """
        if self.robot_file_path_absolute is None:
            raise RuntimeError("robot_file_path is not set before calling _get_foot_link_names().")

        path: Path = self.robot_file_path_absolute
        tree = ET.parse(path)
        root = tree.getroot()

        # --- XML / MJCF -----------------------------------------------------
        if self.robot_file_format == "xml":
            # collect all bodies
            foot_candidates: list[str] = []
            for body in root.findall(".//body"):
                name = body.get("name")
                if not name:
                    continue
                # Heuristik: everythink that looks like a foot
                if "FOOT" in name or "foot" in name.lower():
                    foot_candidates.append(name)

            if not foot_candidates:
                raise RuntimeError(
                    f"No foot bodies found in XML/MJCF at {path}. "
                    "Make sure foot bodies contain 'FOOT' in their name."
                )

            # sort Left/right if possible
            left = [n for n in foot_candidates if "Left" in n or "left" in n]
            right = [n for n in foot_candidates if "Right" in n or "right" in n]

            if left or right:
                foot_link_names = left + right
            else:
                # Fallback: simple order from file
                foot_link_names = foot_candidates

            self.foot_link_names = foot_link_names
            return foot_link_names

        # --- URDF -----------------------------------------------------------
        elif self.robot_file_format == "urdf":
            parent_links = set()
            child_links = set()
            child_to_parent = {}
            child_to_joint_type = {}

            for joint in root.findall(".//joint"):
                jtype = joint.attrib.get("type", "").lower()
                parent = joint.find("parent")
                child = joint.find("child")

                parent_name = parent.get("link") if parent is not None else None
                child_name = child.get("link") if child is not None else None

                if parent_name:
                    parent_links.add(parent_name)
                if child_name:
                    child_links.add(child_name)

                if parent_name and child_name:
                    child_to_parent[child_name] = parent_name
                    child_to_joint_type[child_name] = jtype

            leaf_links = sorted(child_links - parent_links)

            resolved_links = []
            for link in leaf_links:
                if child_to_joint_type.get(link) == "fixed":
                    resolved_links.append(child_to_parent[link])
                else:
                    resolved_links.append(link)

            # remove duplicates while preserving order
            foot_link_names = list(dict.fromkeys(resolved_links))

            left = [n for n in foot_link_names if "left" in n.lower()]
            right = [n for n in foot_link_names if "right" in n.lower()]

            if left or right:
                foot_link_names = left + right

            self.foot_link_names = foot_link_names
            return foot_link_names

        else:
            raise ValueError(
                f"Unknown robot_file_format '{self.robot_file_format}' in _get_foot_link_names()."
            )
    

    def _get_relative_robot_file_path(self) -> Path:
        """
        Return the relative robot file path
        """
        return Path(os.path.relpath(
            self.robot_file_path_absolute, self.relevant_paths_dict["project_root"]
        ))
