"""CPU integrity and saved-input parity check. No simulator or training imports."""

import hashlib
import json
from pathlib import Path
import posixpath
import sys
import xml.etree.ElementTree as ET


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_path(root, relative):
    """Also catch filename-case errors on case-insensitive Windows filesystems."""
    require(isinstance(relative, str) and ":" not in relative, f"Invalid relative path: {relative}")
    parts = relative.split("/")
    require(all(p not in ("", ".", "..") for p in parts), f"Invalid relative path: {relative}")
    path = root
    for part in parts:
        require(path.is_dir() and part in {p.name for p in path.iterdir()},
                f"Missing file/directory or wrong filename case: {relative}")
        path = path / part
    require(path.is_file(), f"Missing file: {relative}")
    return path


def references(path, relative):
    if path.suffix.lower() not in (".urdf", ".dae"):
        return []
    document = ET.parse(path).getroot()
    if path.suffix == ".urdf":
        refs = [e.attrib["filename"] for e in document.findall(".//mesh") + document.findall(".//texture")]
    else:
        refs = [e.text.strip() for e in document.findall(".//{*}image/{*}init_from") if e.text]
        refs += [e.attrib["url"] for e in document.iter()
                 if "url" in e.attrib and not e.attrib["url"].startswith("#")]
    resolved = []
    for ref in refs:
        require(":" not in ref and not ref.startswith("/"), f"Nonportable asset reference: {ref}")
        dep = posixpath.normpath(posixpath.join(posixpath.dirname(relative), ref.split("#")[0]))
        require(dep.startswith("ressources/robots/go2w/"), f"Asset outside canonical robot package: {dep}")
        resolved.append(dep)
    return sorted(set(resolved))


def check_files(package, root):
    for name in ("README.md", "policy.pt", "contract.json", "config.yaml",
                 "provenance.json", "parity_samples.npz", "check_policy.py"):
        exact_path(package, name)
    provenance = json.loads((package / "provenance.json").read_text(encoding="utf-8"))
    require(provenance["schema_version"] == 1, "Unsupported provenance schema")
    require(set(provenance["payloads"]) == {"policy.pt", "contract.json", "config.yaml", "parity_samples.npz"},
            "Missing or unexpected payload hash entries")
    for name, expected in provenance["payloads"].items():
        data = exact_path(package, name).read_bytes()
        require(not data.startswith(b"version https://git-lfs.github.com/spec/"),
                f"LFS pointer instead of real payload: {name}")
        require(len(data) == expected["bytes"], f"Size mismatch/truncated payload: {name}")
        require(hashlib.sha256(data).hexdigest() == expected["sha256"], f"SHA-256 mismatch: {name}")
    contract = json.loads((package / "contract.json").read_text(encoding="utf-8"))
    require(contract["interface_version"] == provenance["interface_version"] == 1, "Unsupported interface version")
    require(contract["files"] == {"policy": "policy.pt", "configuration_snapshot": "config.yaml",
                                 "parity": "parity_samples.npz", "provenance": "provenance.json"},
            "Unexpected public filenames")
    require(contract["checkpoint_sha256"] == provenance["source_checkpoint"]["sha256"], "Checkpoint identity mismatch")
    require(contract["config_sha256"] == provenance["payloads"]["config.yaml"]["sha256"]
            == provenance["source_config_sha256"], "Config identity mismatch")
    require(provenance["exported_actor"]["sha256"] == provenance["payloads"]["policy.pt"]["sha256"],
            "Actor identity mismatch")
    require(contract["completed_training_updates"] == provenance["source_checkpoint"]["completed_updates"],
            "Completed-update identity mismatch")
    asset = contract["robot_asset"]
    require(asset["path"] == "ressources/robots/go2w/urdf/go2w_description.urdf"
            and asset["path_base"] == "repository_root", "Unexpected robot asset path")
    assets = provenance["assets"]
    seen, pending = set(), [asset["path"]]
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        require(relative in assets, f"Unlisted asset dependency: {relative}")
        path = exact_path(root, relative)
        data, expected = path.read_bytes(), assets[relative]
        text = path.suffix.lower() in (".urdf", ".dae")
        require(expected["hash_mode"] == ("lf" if text else "exact"), f"Wrong asset hash mode: {relative}")
        canonical = data.replace(b"\r\n", b"\n") if text else data
        require(len(canonical) == expected["size_git_blob"], f"Asset size mismatch: {relative}")
        require(hashlib.sha256(canonical).hexdigest() == expected["sha256_git_blob"], f"Asset hash mismatch: {relative}")
        deps = references(path, relative)
        require(deps == expected["dependencies"], f"Asset dependency mismatch: {relative}")
        pending.extend(deps)
        seen.add(relative)
    require(seen == set(assets), "Asset manifest contains unrelated/missing dependencies")
    require(asset["source_bytes_sha256"] == contract["urdf_sha256"] == assets[asset["path"]]["sha256_source_bytes"]
            and asset["git_lf_sha256"] == assets[asset["path"]]["sha256_git_blob"], "URDF identity mismatch")
    return contract, provenance


def check_interface(contract, root, np):
    order = [f"{leg}_{joint}_joint" for leg in ("FL", "FR", "RL", "RR")
             for joint in ("hip", "thigh", "calf", "foot")]
    legs = [name for name in order if not name.endswith("_foot_joint")]
    require(contract["joint_order"] == order and contract["leg_position_observation_order"] == legs,
            "Joint/action/position-error order mismatch")
    require(contract["observation_dim"] == 56 and contract["action_dim"] == 16, "Interface dimension mismatch")
    require(contract["input"] == {"dtype": "float32", "shape": ["N", 56], "N_min": 1, "finite_required": True},
            "Invalid input schema")
    require(contract["output"] == {"dtype": "float32", "shape": ["N", 16], "finite_required": True,
                                   "normalized_bounds": [-1, 1]}, "Invalid output schema")
    require(contract["observation_normalization"] == "embedded" and contract["normalizer_epsilon"] == .01
            and contract["observation_limit"] == 100 and contract["action_bound"] == 1,
            "Clipping/normalization semantics mismatch")
    require(contract["policy_dt"] == .02 and contract["physics_dt"] == .005
            and contract["physics_decimation"] == 4 and contract["nominal_actuator_delay_ticks"] == 0,
            "Timing/delay mismatch")
    blocks = contract["observation_blocks"]
    require([b["slice"] for b in blocks] == [[0, 3], [3, 6], [6, 9], [9, 12], [12, 24], [24, 40], [40, 56]],
            "Observation slices mismatch")
    require(blocks[4]["order"] == legs and blocks[5]["order"] == blocks[6]["order"] == order,
            "Observation joint order mismatch")
    require(all(np.equal(b["scale"], 1).all() for b in blocks), "Observation physical scaling mismatch")
    require(contract["control_type"] == {n: "V" if n.endswith("_foot_joint") else "P" for n in order},
            "Named P/V mapping mismatch")
    np.testing.assert_allclose(contract["action_scale"], [.30, .35, .40, 18] * 4, rtol=0, atol=1e-6)
    np.testing.assert_allclose(contract["default_joint_angles"],
                               [0, .70, -1.33, 0] * 2 + [0, .75, -1.31, 0] * 2, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(contract["nominal_kp"], [40, 40, 40, 0] * 4)
    np.testing.assert_array_equal(contract["nominal_kd"], [1] * 16)
    require(contract["wheel_velocity_target_limit"] == 20 and contract["quaternion_convention"] == "wxyz",
            "Wheel target/quaternion convention mismatch")
    document = ET.parse(root / contract["robot_asset"]["path"]).getroot()
    joints = [j for j in document.findall("joint") if j.get("type") != "fixed"]
    require([j.get("name") for j in joints] == order, "Canonical URDF joint order mismatch")
    for i, joint in enumerate(joints):
        limit = joint.find("limit")
        require(limit is not None, f"Missing URDF limit: {order[i]}")
        np.testing.assert_allclose(contract["effort_limits"][i], float(limit.get("effort")), rtol=0, atol=2e-6)
        np.testing.assert_allclose(contract["joint_velocity_limits"][i], float(limit.get("velocity")), rtol=0, atol=1e-6)
        if joint.get("type") == "continuous":
            require(contract["hard_joint_position_limits_rad"][i] == [None, None], "Wheel position must be unbounded")
        else:
            np.testing.assert_allclose(contract["hard_joint_position_limits_rad"][i],
                                       [float(limit.get("lower")), float(limit.get("upper"))], rtol=0, atol=1e-6)


def main():
    try:
        package = Path(__file__).resolve().parent
        root = package.parents[2]
        contract, provenance = check_files(package, root)
        print(f"Integrity OK: real actor {provenance['payloads']['policy.pt']['sha256']}; "
              f"{len(provenance['assets'])} canonical asset files (XML/DAE newline equivalence only).")
        try:
            import numpy as np
            import torch
        except ImportError as error:
            raise RuntimeError("This CPU check requires compatible PyTorch and NumPy, not a simulator stack.") from error
        check_interface(contract, root, np)
        parity = provenance["parity"]
        require(parity["samples"] == 96 and parity["native_cpu_atol"] == 2e-6
                and parity["gpu_recorded_atol"] == 5e-5 and parity["rtol"] == 0, "Parity schema/tolerance mismatch")
        with np.load(package / "parity_samples.npz", allow_pickle=False) as saved:
            require(set(saved.files) == set(parity["fields"]), "Parity field mismatch")
            arrays = {k: saved[k].copy() for k in saved.files}
        for name in ("observations", "expected_clipped_actions", "recorded_gpu_raw_actions"):
            require(arrays[name].dtype == np.float32 and arrays[name].shape == (96, 56 if name == "observations" else 16),
                    f"Parity dtype/dimension mismatch: {name}")
        require(all(np.issubdtype(a.dtype, np.number) and np.isfinite(a).all() for a in arrays.values()),
                "Non-numeric/non-finite parity data")
        np.testing.assert_array_equal(arrays["case_index"], np.repeat(np.arange(3), 32))
        np.testing.assert_array_equal(arrays["sample_index"], np.tile(parity["sample_indices_per_case"], 3))
        require(np.abs(arrays["expected_clipped_actions"]).max() <= 1, "Reference actions outside bounds")
        policy = torch.jit.load(str(package / "policy.pt"), map_location="cpu").eval()
        before = {name: value.clone() for name, value in policy.named_buffers()}
        require(int(before["actor.obs_normalizer.count"]) == parity["normalizer_count"], "Normalizer count mismatch")
        require(all(torch.isfinite(v).all() for v in policy.state_dict().values()), "Non-finite model state")
        with torch.inference_mode():
            result = policy(torch.from_numpy(arrays["observations"]))
        require(result.dtype == torch.float32 and tuple(result.shape) == (96, 16), "Model output dtype/dimension mismatch")
        require(bool(torch.isfinite(result).all()) and float(result.abs().max()) <= 1, "Non-finite or unbounded model output")
        require(all(torch.equal(before[k], v) for k, v in policy.named_buffers()), "Normalizer buffers changed")
        actual = result.numpy()
        cpu_error = float(np.max(np.abs(actual - arrays["expected_clipped_actions"])))
        gpu_error = float(np.max(np.abs(actual - np.clip(arrays["recorded_gpu_raw_actions"], -1, 1))))
        require(cpu_error <= parity["native_cpu_atol"], f"Native CPU parity failed: {cpu_error} > 2e-6")
        require(gpu_error <= parity["gpu_recorded_atol"], f"Recorded GPU bounded parity failed: {gpu_error} > 5e-5")
        print(f"PASS: 96 observations; CPU max error {cpu_error:.9g} <= 2e-6; "
              f"recorded GPU bounded error {gpu_error:.9g} <= 5e-5; buffers unchanged.")
        print(f"Python {sys.version.split()[0]}, PyTorch {torch.__version__}, NumPy {np.__version__}; CPU only. "
              "This is packaging/parity validation, not Linux/PhysX or hardware qualification.")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
