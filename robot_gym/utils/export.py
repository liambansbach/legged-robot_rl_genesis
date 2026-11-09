"""Portable deterministic actor: trained normalization plus the actuator action bound."""

import json
import hashlib
import importlib.metadata
import subprocess
import shutil
from pathlib import Path
import torch


class BoundedPolicy(torch.nn.Module):
    def __init__(self, actor, observation_limit: float, action_limit: float):
        super().__init__()
        self.actor = actor
        self.observation_limit = observation_limit
        self.action_limit = action_limit

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.actor(
            observations.clamp(-self.observation_limit, self.observation_limit)
        ).clamp(-self.action_limit, self.action_limit)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def validate_onnx(model, filename, histories):
    """CPU eager/ORT parity on recorded measured inputs, including full batches."""
    import numpy as np
    import onnxruntime as ort
    session = ort.InferenceSession(str(filename), providers=["CPUExecutionProvider"])
    histories = np.asarray(histories, dtype=np.float32)
    maxima = np.zeros(2)
    with torch.no_grad():
        for start in range(0, len(histories), 256):
            inputs = histories[start:start+256]
            expected = model(torch.from_numpy(inputs))
            actual = session.run(None, {"history": inputs})
            for i, (a, b) in enumerate(zip(expected, actual)):
                np.testing.assert_allclose(a.numpy(), b, atol=2e-5, rtol=2e-5)
                maxima[i] = max(maxima[i], float(np.max(np.abs(a.numpy() - b))))
    # Batch-one is the controller use case.
    one = session.run(None, {"history": histories[:1]})
    expected = model(torch.from_numpy(histories[:1]))
    for a, b in zip(expected, one):
        np.testing.assert_allclose(a.detach().numpy(), b, atol=2e-5, rtol=2e-5)
    return dict(samples=len(histories), max_action_error=float(maxima[0]),
                max_velocity_error_m_s=float(maxima[1]), atol=2e-5, rtol=2e-5,
                runtime=ort.__version__, provider="CPUExecutionProvider")


def export_policy(runner, env, path, onnx=False, parity_histories=None):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite export: {path}; choose --export_dir")
    path.mkdir(parents=True, exist_ok=True)
    actor = runner.alg.get_policy().as_jit().cpu().eval()
    policy = BoundedPolicy(
        actor,
        float(env.cfg.normalization.clip_observations),
        float(env.cfg.normalization.clip_actions),
    ).eval()
    torch.jit.script(policy).save(str(path / "policy_1.pt"))
    contract = {
        "observation_dim": env.num_obs,
        "action_dim": env.num_actions,
        "joint_order": env.joint_names,
        "control_type": env.cfg.control.control_type,
        "action_scale": env.action_scale[0].tolist(),
        "default_joint_angles": env.default_dof_pos[0].tolist(),
        "nominal_kp": env.base_p_gains.tolist(),
        "nominal_kd": env.base_d_gains.tolist(),
        "effort_limits": env.torque_limits.tolist(),
        "joint_velocity_limits": env.dof_vel_limits.tolist(),
        "checkpoint": str(getattr(runner, "checkpoint_path", None)),
        "action_bound": env.cfg.normalization.clip_actions,
        "policy_dt": env.dt,
        "physics_dt": env.cfg.sim.dt,
        "observation_normalization": "embedded",
        "quaternion_convention": "wxyz",
        "observation_order": [
            "body_linear_velocity",
            "body_angular_velocity",
            "projected_gravity",
            "body_velocity_command",
            "joint_position_error",
            "joint_velocity",
            "previous_clipped_action",
        ],
        "observation_scales": {
            "lin_vel": env.obs_scales.lin_vel,
            "ang_vel": env.obs_scales.ang_vel,
            "dof_pos": env.obs_scales.dof_pos,
            "dof_vel": env.obs_scales.dof_vel,
        },
    }
    task_metadata = env.export_metadata()
    contract.update(task_metadata)
    source_path = getattr(runner, 'source_config_path', None)
    if source_path is not None:
        source_config = Path(source_path)
        contract['source_config_sha256'] = hashlib.sha256(source_config.read_bytes()).hexdigest()
        shutil.copyfile(source_config, path / 'training_config.yaml')
    from robot_gym.utils.velocity_estimator import enabled, deployment_policy
    if enabled(env.cfg):
        from robot_gym.utils.helpers import class_to_dict
        import numpy as np
        combined = deployment_policy(runner, env)
        spec = env.velocity_estimator_spec()
        live = set(spec["live_feature_indices"])
        sensors, controller = [], []
        slots = [""] * env.num_obs
        for slot, axis in zip(spec["velocity_indices"], "xyz"):
            slots[slot] = f"body_linear_velocity_{axis}"
        for field in spec["fields"]:
            indices = set(range(field["start"], field["stop"]))
            if indices - live:
                sensors.append(field["name"])
            if indices & live:
                controller.append(field["name"])
            for i in indices:
                slots[spec["feature_indices"][i]] = f"{field['name']}[{i - field['start']}]"
        if "observation_order" not in task_metadata:
            contract["observation_order"] = slots
        shutil.copyfile(Path(__file__).with_name('deployment_history.py'), path / 'deployment_history.py')
        torch.jit.script(combined).save(str(path / "estimated_policy.pt"))
        contract['combined_torchscript_sha256'] = hashlib.sha256((path / 'estimated_policy.pt').read_bytes()).hexdigest()
        code_files = sorted((Path(__file__).parents[1]).rglob('*.py'))
        digest = hashlib.sha256()
        for source in code_files:
            digest.update(source.relative_to(Path(__file__).parents[1]).as_posix().encode())
            digest.update(source.read_bytes())
        contract['source_python_tree_sha256'] = digest.hexdigest()
        contract.update(interface_version=3, inference_path="estimated velocity from startup",
            external_input={"name": "history", "shape": ["batch", env.cfg.velocity_estimator['history_length'],
                            len(spec['feature_indices'])], "dtype": "float32"},
            outputs=["clipped_actions", "velocity_m_s"],
            estimator={"configuration": env.cfg.velocity_estimator, "features": spec,
                "normalization": "fixed feature_mean/std embedded before MLP; initialized from parent actor statistics when available",
                "history": "Oldest to newest. Pad with the first measured frame on reset. Append once per policy boundary. Refresh only the newest live_feature_indices on controller changes.",
                "span_s": (env.cfg.velocity_estimator['history_length'] - 1) * env.dt,
                "target_timestamp": "current policy boundary even if sensor channels are delayed/held",
                "startup": "no simulator-velocity bootstrap; estimator used at first action",
                "sensor_timing_status": "simulation assumptions; hardware timestamps, bias, delay and filtering are unmeasured",
                "required_sensors": sensors,
                "controller_state": controller},
            actor_tensor_sha256=tensor_hash(runner.alg.actor.state_dict()),
            estimator_tensor_sha256=tensor_hash(runner.alg.estimator.state_dict()),
            stage_counters=runner.alg.stage_counts,
            runtime={name: importlib.metadata.version(name) for name in ('torch', 'genesis-world', 'rsl-rl-lib')},
            source_head=subprocess.run(['git', 'rev-parse', 'HEAD'], text=True, capture_output=True,
                                       stdin=subprocess.DEVNULL, check=True).stdout.strip(),
            source_dirty=bool(subprocess.run(['git', 'status', '--porcelain'], text=True, capture_output=True,
                                            stdin=subprocess.DEVNULL, check=True).stdout.strip()),
            validation_status="simulation/export screen only; no demonstrated sim2real")
        velocity = contract.setdefault('velocity', {})
        velocity.update(source='learned from measured history', target=spec['target'],
                        actor_indices=spec['velocity_indices'], scale=spec['velocity_scale'])
        (path / 'runtime_config.json').write_text(json.dumps(class_to_dict(env.cfg), indent=2), encoding='utf-8')
        if onnx:
            if parity_histories is None:
                raise ValueError("Combined ONNX export requires --parity_histories from the estimator screen")
            recorded = np.load(parity_histories, allow_pickle=False)
            history = recorded['histories']
            screen = Path(parity_histories).with_name('metrics.json')
            if screen.exists():
                summary = json.loads(screen.read_text(encoding='utf-8'))
                checkpoint_hash = hashlib.sha256(Path(runner.checkpoint_path).read_bytes()).hexdigest()
                if summary.get('definition', {}).get('checkpoint_sha256') == checkpoint_hash:
                    contract['simulation_screen'] = {key: summary[key] for key in
                        ('definition', 'comparisons', 'passed', 'qualification') if key in summary}
                    contract['simulation_screen']['metrics_sha256'] = hashlib.sha256(screen.read_bytes()).hexdigest()
                    shutil.copyfile(screen, path / 'screen_metrics.json')
            # Bind reference outputs to THIS model, not the rollout's outer run label.
            expected = [[], []]
            scripted = torch.jit.load(str(path / 'estimated_policy.pt')).eval()
            with torch.no_grad():
                for start in range(0, len(history), 256):
                    inputs = torch.from_numpy(history[start:start+256]).float()
                    eager = combined(inputs)
                    for i, (a, b) in enumerate(zip(eager, scripted(inputs))):
                        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                        expected[i].append(a.numpy())
            np.savez_compressed(path / 'parity_samples.npz', histories=history, reset=recorded['reset'],
                                reset_frames=recorded['reset_frames'],
                                expected_actions=np.concatenate(expected[0]),
                                expected_velocity=np.concatenate(expected[1]))
            torch.onnx.export(combined, (torch.from_numpy(history[:1]).float(),), str(path / 'policy.onnx'),
                input_names=['history'], output_names=['clipped_actions', 'velocity_m_s'],
                dynamic_axes={'history': {0: 'batch'}, 'clipped_actions': {0: 'batch'}, 'velocity_m_s': {0: 'batch'}},
                opset_version=17, dynamo=False)
            contract['onnx_parity'] = validate_onnx(combined, path / 'policy.onnx', history)
            contract['onnx_parity']['recording_sha256'] = hashlib.sha256(Path(parity_histories).read_bytes()).hexdigest()
            contract['onnx_sha256'] = hashlib.sha256((path / 'policy.onnx').read_bytes()).hexdigest()
    elif onnx:
        torch.onnx.export(policy, (torch.zeros(1, env.num_obs, device='cpu'),), str(path / 'policy.onnx'),
            input_names=['observations'], output_names=['clipped_actions'],
            dynamic_axes={'observations': {0: 'batch'}, 'clipped_actions': {0: 'batch'}}, opset_version=17, dynamo=False)
    if env.cfg.control.armature is not None:
        from robot_gym.utils.diagnostics import joint_dynamics, write_json, sha256
        contract["actor_sha256"] = sha256(path / "policy_1.pt")
        contract["checkpoint_sha256"] = sha256(runner.checkpoint_path)
        write_json(path / "runtime_dynamics.json", joint_dynamics(env.robot, env.joint_names))
    (path / "contract.json").write_text(json.dumps(contract, indent=2), encoding='utf-8')
    return policy
