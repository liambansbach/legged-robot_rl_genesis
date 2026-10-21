"""Bounded matched closed-loop screen, held out by whole sequence and seed.

These tolerances are proposed simulation regression screens, not hardware limits.
No environment state or target enters the inference composition.
"""

import copy
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
import genesis as gs

from .diagnostics import write_json, sha256
from .helpers import set_seed
from .replay import configure_nominal
from .task_registry import task_registry
from .velocity_estimator import SensorHistory, enabled, inference_policy, deployment_policy


TOLERANCES = dict(tracking_rmse_ratio=1.15, tracking_rmse_allowance=[.03, .03, .08],
    stop_path_ratio=1.25, stop_path_allowance_m=.03, drift_allowance_m_s=.02,
    tilt_allowance_rad=.05, height_drop_allowance_m=.02, saturation_allowance=.05,
    additional_falls=0, additional_abnormal_contact_ticks=0)


def lag_samples(estimated, truth, max_lag=10):
    """Coarse positive prediction lag at maximum centered correlation; null if static."""
    result = []
    for i in range(3):
        a, b = estimated[:, i], truth[:, i]
        if min(a.std(), b.std()) < .005:
            result.append(None)
            continue
        scores = [np.corrcoef(a[k:] if k else a, b[:-k] if k else b)[0, 1]
                  for k in range(max_lag + 1)]
        result.append(int(np.nanargmax(scores)))
    return result


def metrics(data, dt):
    error = data['estimate'] - data['target']
    actual = np.concatenate((data['velocity'][..., :2], data['angular'][..., 2:3]), axis=-1)
    tracking = actual - data['commands']
    stops = [(round(4/dt), round(6/dt)), (round(10/dt), len(actual))]
    result = []
    for j in range(actual.shape[1]):
        paths, drifts = [], []
        for first, last in stops:
            # Distance in world XY, including the transition into the first stop tick.
            xyz = data['position'][max(0, first-1):last, j, :2]
            paths.append(float(np.linalg.norm(np.diff(xyz, axis=0), axis=-1).sum()))
            drifts.append(float(np.sqrt(np.mean(data['velocity'][last-round(1/dt):last, j, :2]**2))))
        result.append(dict(
            estimator_bias_m_s=error[:, j].mean(0).tolist(),
            estimator_rmse_m_s=np.sqrt((error[:, j]**2).mean(0)).tolist(),
            estimator_p95_abs_m_s=np.quantile(abs(error[:, j]), .95, axis=0).tolist(),
            estimator_max_abs_m_s=abs(error[:, j]).max(0).tolist(),
            estimator_lag_s=[None if k is None else k*dt for k in lag_samples(data['estimate'][:, j], data['target'][:, j])],
            tracking_rmse=np.sqrt((tracking[:, j]**2).mean(0)).tolist(),
            stop_path_m=paths, final_stop_drift_rms_m_s=drifts,
            cross_axis_rms=np.sqrt(np.mean(np.where(data['commands'][:, j] == 0, actual[:, j], 0.)**2, axis=0)).tolist(),
            max_tilt_rad=float(np.linalg.norm(data['rpy'][:, j, :2], axis=1).max()),
            min_height_m=float(data['position'][:, j, 2].min()),
            saturation_fraction=float((abs(data['actions'][:, j]) >= 1.-1e-6).mean()),
            falls=int(data['fallen'][:, j].sum()),
            abnormal_contact_ticks=int((data['contacts'][:, j] > 0).sum())))
    return result


def compare(parent, estimated):
    t = TOLERANCES
    checks = []
    for a, b in zip(parent, estimated):
        checks.append(dict(
            tracking=bool(np.all(np.array(b['tracking_rmse']) <= np.array(a['tracking_rmse'])*t['tracking_rmse_ratio'] + t['tracking_rmse_allowance'])),
            stopping=bool(np.all(np.array(b['stop_path_m']) <= np.array(a['stop_path_m'])*t['stop_path_ratio'] + t['stop_path_allowance_m'])),
            drift=bool(np.all(np.array(b['final_stop_drift_rms_m_s']) <= np.array(a['final_stop_drift_rms_m_s']) + t['drift_allowance_m_s'])),
            tilt=b['max_tilt_rad'] <= a['max_tilt_rad'] + t['tilt_allowance_rad'],
            height=b['min_height_m'] >= a['min_height_m'] - t['height_drop_allowance_m'],
            saturation=b['saturation_fraction'] <= a['saturation_fraction'] + t['saturation_allowance'],
            falls=b['falls'] <= a['falls'], contacts=b['abnormal_contact_ticks'] <= a['abnormal_contact_ticks']))
    return dict(passed=all(all(row.values()) for row in checks), per_case=checks)


def evaluate_estimator(args):
    started = perf_counter()
    cfg, training, checkpoint = task_registry.resolve_replay(args)
    if not enabled(cfg):
        raise ValueError("The estimator screen requires a trained full actor+estimator checkpoint")
    if args.task != 'go2w':
        raise ValueError("This compact command panel is currently defined for Go2-W")
    out = Path(args.output)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Refusing to overwrite screen: {out}")
    out.mkdir(parents=True, exist_ok=True)
    cases = ['stand', 'forward', 'reverse', 'lateral_positive', 'lateral_negative', 'yaw_positive', 'yaw_negative', 'mixed']
    commands = [[0, 0, 0], [1., 0, 0], [-.3, 0, 0], [0, .3, 0], [0, -.3, 0], [0, 0, 1.], [0, 0, -1.], [.45, .15, -.5]]
    declaration = dict(criteria=TOLERANCES, criteria_status="proposed simulation screening tolerances",
        cases=dict(zip(cases, commands)), sequence="0-1 stand; 1-4 ramp/start/hold; 4-6 stop; 6-8 reverse half; 8-10 original half; 10-14 stop. Commands refreshed at 10 Hz.",
        seeds=[1729, 1730], training_seed=training.seed,
        conditions=['nominal', 'saved observation noise + 20 ms sensor delay + 40 ms sample holding + 30 N stop push'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    if args.comparison_checkpoint:
        declaration['comparison_checkpoint'] = str(Path(args.comparison_checkpoint).resolve())
        declaration['comparison_checkpoint_sha256'] = sha256(args.comparison_checkpoint)
    write_json(out / 'screen_definition.json', declaration)  # Before examining outcomes.
    configure_nominal(cfg, args)
    cfg.env.num_envs, cfg.env.episode_length_s = 8, 30.
    cfg.env.capture_transitions = True
    cfg.env.play_mode = True
    args.num_envs = 8
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=training, save_config=False)
    candidate_actor = copy.deepcopy(runner.alg.actor.state_dict())
    parent_actor = candidate_actor
    if args.comparison_checkpoint:
        parent_actor = torch.load(args.comparison_checkpoint, map_location=env.device, weights_only=False)['actor_state_dict']
    elif runner.alg.stage_counts['finetune_updates']:
        raise ValueError("Fine-tuned screen requires --comparison_checkpoint for the immutable truth-driven parent")
    env.command_resampling_enabled = False
    command = torch.tensor(commands, device=env.device)
    report = dict(definition=declaration, results={}, comparisons={})
    try:
        # Real native boundary/getter/reset smoke before resetting the matched trials.
        env.set_fixed_command((0., 0., 0.))
        env.reset()
        history, phase = env.sensor_history.history.clone(), env.phase.clone()
        rng = torch.cuda.get_rng_state(env.device) if env.device.type == 'cuda' else torch.get_rng_state()
        env.compute_observations()
        env.get_observations()
        env.compute_observations()
        torch.testing.assert_close(env.sensor_history.history, history, rtol=0., atol=0.)
        torch.testing.assert_close(env.phase, phase, rtol=0., atol=0.)
        after_rng = torch.cuda.get_rng_state(env.device) if env.device.type == 'cuda' else torch.get_rng_state()
        assert torch.equal(rng, after_rng), 'Getter consumed sensor RNG'
        env.reset_idx(torch.tensor([0], device=env.device))
        torch.testing.assert_close(env.sensor_history.history[1:], history[1:], rtol=0., atol=0.)
        env.step(torch.zeros(8, env.num_actions, device=env.device))
        torch.testing.assert_close(env.sensor_history.history[0], env.sensor_history.history[0, -1:].expand_as(history[0]), rtol=0., atol=0.)
        report['native_lifecycle_checks'] = 'passed: repeated getters, phase/RNG stability, selective reset and first-sample padding'
        for condition in ('nominal', 'sensor_stress'):
            sensor_cfg = copy.deepcopy(env.cfg.velocity_estimator)
            sensor_cfg.update(sensor_delay_steps=0 if condition == 'nominal' else 1,
                              sensor_hold_steps=1 if condition == 'nominal' else 2)
            env.add_noise = condition != 'nominal'
            for source in ('truth', 'estimated'):
                runner.alg.actor.load_state_dict(parent_actor if source == 'truth' else candidate_actor)
                env.sensor_history = SensorHistory(8, env.velocity_estimator_spec(), sensor_cfg, env.device)
                set_seed(1729 if condition == 'nominal' else 1730)
                env.set_fixed_command((0., 0., 0.))
                env.reset()
                env.compute_observations()
                policy = inference_policy(runner, env, source)
                data = {k: [] for k in ('histories', 'estimate', 'target', 'commands', 'velocity', 'angular',
                    'position', 'rpy', 'actions', 'fallen', 'contacts', 'reset', 'reset_frames')}
                with torch.inference_mode():
                    for tick in range(round(14/env.dt)):
                        t = tick * env.dt
                        if tick % 5 == 0:  # Navigation publishes at 10 Hz, policy at 50 Hz.
                            gain = min(1., (t-1.)/.5) if 1 <= t < 4 else -.5 if 6 <= t < 8 else .5 if 8 <= t < 10 else 0.
                            env.set_commands(command * gain)
                            env.compute_observations()
                        obs = env.get_observations()
                        data['histories'].append(obs['estimator_history'].clone())
                        data['target'].append(obs['velocity_target'].clone())
                        data['reset'].append((env.sensor_history.age <= 1).clone())
                        data['reset_frames'].append(env.sensor_history.initial_frame.clone())
                        _, estimate = runner.alg.velocity_input(obs['estimator_history'])
                        data['estimate'].append(estimate.clone())
                        data['commands'].append(env.commands.clone())
                        actions = policy(obs)
                        if condition != 'nominal' and tick == round(10.4/env.dt):
                            env.push_force[:, 0, :2] = torch.tensor([24., 18.], device=env.device)
                            env.push_steps_left[:] = round(.15/env.cfg.sim.dt)
                        env.step(actions)
                        state = env.transition_state
                        for key, field in (('velocity', 'base_lin_vel'), ('angular', 'base_ang_vel'), ('position', 'base_pos'),
                                           ('rpy', 'rpy'), ('actions', 'actions'), ('fallen', 'fallen')):
                            data[key].append(state[field].clone())
                        data['contacts'].append((state['nonfoot_contact_count'] + state['undesired_self_pair_count']).clone())
                arrays = {k: torch.stack(v).cpu().numpy() for k, v in data.items()}
                name = f'{condition}_{source}'
                report['results'][name] = dict(zip(cases, metrics(arrays, env.dt)))
                # Keep temporal arrays for analysis; flat history alone is a portable export input.
                histories = arrays.pop('histories')
                np.savez_compressed(out / f'{name}.npz', histories=histories.reshape(-1, *histories.shape[2:]), **arrays)
                print(f"Completed {name}: {len(histories)} ticks x 8 held-out episodes", flush=True)
            report['comparisons'][condition] = compare(list(report['results'][f'{condition}_truth'].values()),
                                                       list(report['results'][f'{condition}_estimated'].values()))
        model = deployment_policy(runner, env)
        sample = env.sensor_history.history[:1].cpu()
        torch.set_num_threads(1)
        with torch.no_grad():
            for _ in range(20):
                model(sample)
            start = perf_counter()
            for _ in range(1000):
                model(sample)
        report['runtime_cpu_batch1_ms'] = 1000 * (perf_counter()-start) / 1000
        report['wall_s'] = perf_counter()-started
        report['passed'] = all(v['passed'] for v in report['comparisons'].values())
        report['qualification'] = 'simulation regression screen only; source-yaw physics mismatch remains unresolved'
        write_json(out / 'metrics.json', report)
        print(f"Estimator screen passed: {report['passed']}; report {out / 'metrics.json'}", flush=True)
    finally:
        gs.destroy()
        from robot_gym.envs.base.base_task import BaseTask
        BaseTask._gs_initialized = False
    return report
