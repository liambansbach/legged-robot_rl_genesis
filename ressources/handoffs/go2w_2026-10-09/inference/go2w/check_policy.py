"""Portable CPU integrity, actor parity and observation-sequence check.

Requires only PyTorch, NumPy and the Python standard library. No simulator,
training repository imports, state estimator or command/action controller.
"""
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
    require(isinstance(relative, str) and ":" not in relative, f"Invalid path: {relative}")
    parts = relative.split("/")
    require(all(part not in ("", ".", "..") for part in parts), f"Invalid path: {relative}")
    path = root
    for part in parts:
        require(path.is_dir() and part in {p.name for p in path.iterdir()},
                f"Missing file or wrong filename case: {relative}")
        path /= part
    require(path.is_file(), f"Not a file: {relative}")
    return path


def references(path, relative):
    if path.suffix.lower() not in (".urdf", ".dae"):
        return []
    document = ET.parse(path).getroot()
    if path.suffix.lower() == ".urdf":
        refs = [e.attrib["filename"] for e in document.findall(".//mesh") + document.findall(".//texture")]
    else:
        refs = [e.text.strip() for e in document.findall(".//{*}image/{*}init_from") if e.text]
        refs += [e.attrib["url"] for e in document.iter()
                 if "url" in e.attrib and not e.attrib["url"].startswith("#")]
    result = []
    for ref in refs:
        require(":" not in ref and not ref.startswith("/"), f"Nonportable asset reference: {ref}")
        target = posixpath.normpath(posixpath.join(posixpath.dirname(relative), ref.split("#")[0]))
        require(target.startswith("assets/go2w/"), f"Asset escapes the bundled robot: {target}")
        result.append(target)
    return sorted(set(result))


def check_files(package):
    required = {"README.md", "policy.pt", "contract.json", "config.yaml", "provenance.json",
                "parity_samples.npz", "check_policy.py"}
    require({p.name for p in package.iterdir() if p.is_file()} == required,
            "Missing or unexpected package root files")
    provenance = json.loads(exact_path(package, "provenance.json").read_text(encoding="utf-8"))
    contract = json.loads(exact_path(package, "contract.json").read_text(encoding="utf-8"))
    require(provenance["schema_version"] == 2 and provenance["interface_version"] == 2
            and contract["interface_version"] == 2, "Unsupported package/interface version")
    require(set(provenance["payloads"]) == {"policy.pt", "contract.json", "config.yaml", "parity_samples.npz"},
            "Unexpected payload manifest")
    require(set(provenance["support_files"]) == {"README.md", "check_policy.py"},
            "Unexpected support-file manifest")
    for name, expected in {**provenance["payloads"], **provenance["support_files"]}.items():
        data = exact_path(package, name).read_bytes()
        require(not data.startswith(b"version https://git-lfs.github.com/spec/"), f"LFS pointer: {name}")
        require(len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"],
                f"Payload size/hash mismatch: {name}")
    require(contract["checkpoint_sha256"] == provenance["source_checkpoint"]["sha256"],
            "Checkpoint identity mismatch")
    require(contract["config_sha256"] == provenance["payloads"]["config.yaml"]["sha256"],
            "Saved configuration identity mismatch")
    require(contract["actor_sha256"] == provenance["payloads"]["policy.pt"]["sha256"],
            "Actor identity mismatch")
    require(contract["completed_training_updates"] == provenance["source_checkpoint"]["completed_updates"],
            "Completed-update identity mismatch")
    asset = contract["robot_asset"]
    require(asset["path_base"] == "package_directory" and asset["path"].startswith("assets/go2w/urdf/"),
            "Robot asset must be self-contained")
    seen, pending = set(), [asset["path"]]
    while pending:
        relative = pending.pop()
        if relative in seen:
            continue
        require(relative in provenance["assets"], f"Unlisted asset dependency: {relative}")
        path = exact_path(package, relative)
        data, expected = path.read_bytes(), provenance["assets"][relative]
        canonical = data.replace(b"\r\n", b"\n") if expected["hash_mode"] == "lf" else data
        require(len(canonical) == expected["canonical_bytes"]
                and hashlib.sha256(canonical).hexdigest() == expected["canonical_sha256"],
                f"Asset size/hash mismatch: {relative}")
        deps = references(path, relative)
        require(deps == expected["dependencies"], f"Asset dependency mismatch: {relative}")
        pending.extend(deps)
        seen.add(relative)
    require(seen == set(provenance["assets"]), "Unrelated or missing asset dependencies")
    require(provenance["assets"][asset["path"]]["source_bytes_sha256"] == contract["urdf_sha256"],
            "Measured URDF identity mismatch")
    return contract, provenance


def assemble_observations(arrays, contract, np, history=None):
    """Construct physical actor inputs from real pre-action states, never world pose."""
    q = arrays["pre_quaternion_wxyz"]
    w, x, y, z = (q[:, i] for i in range(4))
    gravity = np.stack((2 * (w*y-x*z), -2 * (w*x+y*z), -(1-2*(x*x+y*y))), axis=-1)
    scales = contract["observation_scales"]
    command = arrays["commands"]
    leg_indices = contract["leg_action_indices"]
    reference = np.asarray(contract["default_joint_angles"], dtype=np.float32)
    history = arrays["previous_issued_actions"] if history is None else history
    bounds = contract["phase"]["demand"]
    lateral = bounds["lateral_start_full_m_s"]
    yaw = bounds["yaw_start_full_rad_s"]
    value = np.maximum(np.clip((np.abs(command[:, 1])-lateral[0])/(lateral[1]-lateral[0]), 0, 1),
                       np.clip((np.abs(command[:, 2])-yaw[0])/(yaw[1]-yaw[0]), 0, 1))
    activation = value*value*(3-2*value)
    angle = 2*np.pi*arrays["phase_cycles"]
    clock = activation[:, None]*np.stack((np.sin(angle), np.cos(angle)), axis=-1)
    observation = np.concatenate((arrays["pre_linear_body"]*scales["lin_vel"],
        arrays["pre_angular_body"]*scales["ang_vel"], gravity,
        command*np.asarray([scales["lin_vel"], scales["lin_vel"], scales["ang_vel"]], dtype=np.float32),
        (arrays["pre_dof_pos"][:, leg_indices]-reference[leg_indices])*scales["dof_pos"],
        arrays["pre_dof_vel"]*scales["dof_vel"], history, clock), axis=-1)
    return np.clip(observation, -contract["observation_limit"], contract["observation_limit"]).astype(np.float32)


def check_interface(contract, package, np):
    order = [f"{side}_{joint}_joint" for side in ("FL", "FR", "RL", "RR")
             for joint in ("hip", "thigh", "calf", "foot")]
    legs = [i for i, name in enumerate(order) if not name.endswith("_foot_joint")]
    require(contract["joint_order"] == order and contract["leg_action_indices"] == legs,
            "Interleaved action and leg observation orders differ")
    require(contract["observation_dim"] == 58 and contract["action_dim"] == 16
            and contract["actor_input_dim"] == 58 and not contract["privileged_inputs_exported"],
            "Actor interface or privileged-input isolation mismatch")
    require(contract["input"] == {"dtype": "float32", "shape": ["N", 58], "N_min": 1, "finite_required": True},
            "Input schema mismatch")
    require(contract["output"] == {"dtype": "float32", "shape": ["N", 16], "normalized_bounds": [-1, 1],
                                   "finite_required": True}, "Output schema mismatch")
    require(contract["config_version"] == 3 and contract["training_tracking_kernel"] == "absolute_exponential",
            "Training schema provenance mismatch")
    require(contract["observation_normalization"] == "embedded" and contract["observation_limit"] == 100
            and contract["action_bound"] == 1, "Clipping/normalization mismatch")
    require(contract["policy_dt"] == .02 and contract["physics_dt"] == .005
            and contract["physics_decimation"] == 4 and contract["nominal_actuator_delay_ticks"] == 0,
            "Nominal timing/delay mismatch")
    require([b["slice"] for b in contract["observation_blocks"]] ==
            [[0,3],[3,6],[6,9],[9,12],[12,24],[24,40],[40,56],[56,58]], "Observation slices mismatch")
    require(contract["phase"]["observation_mode"] == "command_demand"
            and contract["phase"]["period_s"] == .8, "Phase contract mismatch")
    np.testing.assert_allclose(contract["action_scale"], [.30,.35,.40,18]*4, rtol=0, atol=1e-6)
    np.testing.assert_allclose(contract["default_joint_angles"], [0,.70,-1.40,0]*4, rtol=0, atol=1e-6)
    document = ET.parse(exact_path(package, contract["robot_asset"]["path"])).getroot()
    joints = {j.get("name"): j for j in document.findall("joint") if j.get("type") != "fixed"}
    require(set(joints) == set(order), "Measured URDF action joints differ")
    for i, name in enumerate(order):
        joint, limit = joints[name], joints[name].find("limit")
        np.testing.assert_allclose(contract["effort_limits"][i], float(limit.get("effort")), rtol=0, atol=2e-6)
        np.testing.assert_allclose(contract["joint_velocity_limits"][i], float(limit.get("velocity")), rtol=0, atol=1e-6)
        expected = [None, None] if joint.get("type") == "continuous" else [float(limit.get("lower")), float(limit.get("upper"))]
        if expected[0] is None:
            require(contract["hard_joint_position_limits_rad"][i] == expected, "Continuous wheel angle was bounded")
        else:
            np.testing.assert_allclose(contract["hard_joint_position_limits_rad"][i], expected, rtol=0, atol=1e-6)


def run_checks(package):
    contract, provenance = check_files(package)
    import numpy as np
    import torch
    torch.set_num_threads(1)
    check_interface(contract, package, np)
    with np.load(package/"parity_samples.npz", allow_pickle=False) as loaded:
        arrays = {k: loaded[k].copy() for k in loaded.files}
    require(all(np.issubdtype(v.dtype, np.number) and np.isfinite(v).all() for v in arrays.values()),
            "Parity data must be finite numeric arrays")
    p = provenance["parity"]
    obs = arrays["observations"]
    require(obs.dtype == np.float32 and obs.shape == (p["samples"],58), "Real-input parity shape mismatch")
    assembled = assemble_observations(arrays, contract, np)
    assembly_error = float(np.max(np.abs(assembled-obs)))
    require(assembly_error <= p["observation_atol"], f"Physical observation assembly mismatch: {assembly_error}")
    # Physical reconstruction uses a valid recorded orientation, never a yaw placeholder.
    require(np.max(np.abs(np.linalg.norm(arrays["pre_quaternion_wxyz"], axis=1)-1)) < 1e-5,
            "Nonunit recorded quaternion")
    model = torch.jit.load(str(package/"policy.pt"), map_location="cpu").eval()
    before = {key: value.clone() for key, value in model.named_buffers()}
    require(int(before["actor.obs_normalizer.count"]) == p["normalizer_count"], "Normalizer counter mismatch")
    require(all(key.startswith("actor.") for key in model.state_dict()), "Unexpected critic/optimizer export state")
    require(all(torch.isfinite(v).all() for v in model.state_dict().values()), "Nonfinite exported state")
    with torch.inference_mode():
        result = model(torch.from_numpy(obs))
        single = torch.cat([model(torch.from_numpy(row[None])) for row in obs], dim=0)
        # Synthetic bounds probes are separate from the real-state parity evidence.
        probe = torch.from_numpy(obs[:4].copy())
        probe[:,0] = torch.tensor([101.,-101.,1000.,-1000.])
        probe_result = model(probe)
        require(torch.equal(probe_result, model(probe.clamp(-100,100))),
                "Observation clipping is not applied before the actor")
        require(torch.isfinite(probe_result).all() and probe_result.abs().max() <= 1,
                "Bounded synthetic clipping probe failed")
    require(result.dtype == torch.float32 and tuple(result.shape) == (len(obs),16)
            and torch.isfinite(result).all() and result.abs().max() <= 1, "Actor output contract failed")
    actual = result.numpy()
    cpu_error = float(np.max(np.abs(actual-arrays["expected_clipped_actions"])))
    gpu_error = float(np.max(np.abs(actual-np.clip(arrays["recorded_gpu_raw_actions"],-1,1))))
    single_error = float(np.max(np.abs(single.numpy()-actual)))
    require(cpu_error <= p["native_cpu_atol"], f"Native CPU parity failed: {cpu_error}")
    require(gpu_error <= p["gpu_recorded_atol"], f"Recorded GPU parity failed: {gpu_error}")
    require(single_error <= p["batch_single_atol"], f"Single/batched parity failed: {single_error}")
    require(np.any(np.abs(arrays["recorded_gpu_raw_actions"]) > 1),
            "Real parity samples do not exercise action clipping")
    # Full consecutive navigation sequence: real prior states, commands, clock and issued history.
    nav = arrays["navigation_mask"].astype(bool)
    ids = np.flatnonzero(nav)
    require(len(ids) == p["navigation_steps"] and len(ids) > 1, "Missing consecutive navigation evidence")
    np.testing.assert_array_equal(arrays["trace_index"][nav], np.arange(len(ids)))
    previous = arrays["previous_issued_actions"][nav]
    raw = arrays["recorded_gpu_raw_actions"][nav]
    np.testing.assert_allclose(previous[1:], np.clip(raw[:-1],-1,1), rtol=0, atol=0)
    np.testing.assert_array_equal(previous[0], np.zeros(16, dtype=np.float32))
    phase = arrays["phase_cycles"][nav]
    require(abs(float(phase[0])-contract["policy_dt"]/contract["phase"]["period_s"]) <= p["phase_atol"],
            "First actor phase does not include the native zero-action reset warmup")
    phase_error = np.abs(((phase[1:]-phase[:-1]-contract["policy_dt"]/contract["phase"]["period_s"]+.5)%1)-.5)
    require(float(phase_error.max()) <= p["phase_atol"], "Latent phase reset/advance mismatch")
    changes = np.flatnonzero(np.any(arrays["commands"][nav][1:] != arrays["commands"][nav][:-1], axis=1))+1
    require(changes.tolist() == p["navigation_command_boundaries"], "Navigation boundary evidence mismatch")
    require(np.any(np.abs(obs[nav,56:58]) > .1) and np.any(np.all(obs[nav,56:58] == 0, axis=1)),
            "Demand-conditioned clock lacks moving and zero-demand evidence")
    # The recurrent state is external issued action history only. Physical states stay recorded.
    history = previous[0:1].copy()
    sequence = []
    with torch.inference_mode():
        for index in ids:
            sample = {key: value[index:index+1] for key, value in arrays.items() if len(value) == len(obs)}
            assembled_one = assemble_observations(sample, contract, np, history)
            action = model(torch.from_numpy(assembled_one)).numpy()
            sequence.append(action[0])
            history = action.copy()
    sequence_error = float(np.max(np.abs(np.asarray(sequence)-arrays["sequence_expected_actions"][nav])))
    require(sequence_error <= p["sequence_atol"], f"Native CPU action-history sequence parity failed: {sequence_error}")
    # Automatic reset_idx has no extra warmup tick. This is an offline interface probe,
    # not a fabricated physical reset trajectory; retain the selected recorded body state.
    reset = p["automatic_reset_probe"]
    index = reset["source_parity_index"]
    reset_sample = {key:value[index:index+1].copy() for key,value in arrays.items() if len(value) == len(obs)}
    reset_sample["phase_cycles"].fill(reset["expected_phase_cycles"])
    reset_sample["previous_issued_actions"][:] = np.asarray(reset["expected_action_history"],dtype=np.float32)
    reset_obs = assemble_observations(reset_sample,contract,np)
    require(contract["reset_protocol"]["automatic_episode_first_actor_phase"] == 0
            and np.all(reset_obs[:,40:57] == 0) and reset_obs[0,57] > .1,
            "Automatic reset phase/history assembly mismatch")
    with torch.inference_mode():
        reset_action = model(torch.from_numpy(reset_obs)).numpy()[0]
    reset_error = float(np.max(np.abs(reset_action-np.asarray(reset["native_cpu_expected_actions"]))))
    require(reset_error <= p["native_cpu_atol"], "Automatic reset interface probe native CPU parity failed")
    require(provenance["native_export"]["fixed_policy_varied_critic_max_action_error"] == 0,
            "Native actor dependence on separately varied critic group")
    require(all(torch.equal(before[key], value) for key,value in model.named_buffers()),
            "Inference mutated empirical normalization buffers")
    return {"samples":len(obs), "navigation_steps":len(ids), "observation_assembly_max_error":assembly_error,
            "native_cpu_max_error":cpu_error, "recorded_gpu_max_error":gpu_error,
            "batch_single_max_error":single_error, "sequence_max_error":sequence_error,
            "automatic_reset_interface_max_error":reset_error,
            "native_actor_privileged_independence_max_error":0.,
            "normalizer_buffers_unchanged":True, "bundled_asset_count":len(provenance["assets"]),
            "python":sys.version.split()[0], "torch":torch.__version__, "numpy":np.__version__}


def main():
    try:
        result = run_checks(Path(__file__).resolve().parent)
        print("PASS: "+json.dumps(result, sort_keys=True))
        print("CPU packaging/sequence parity only; not Linux/PhysX, disturbance or hardware qualification.")
        return 0
    except Exception as error:
        print(f"FAIL: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
