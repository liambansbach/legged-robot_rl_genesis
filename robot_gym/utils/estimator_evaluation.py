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


def stop_metrics(data, j, first, last, dt):
    """Complete pre-reset stop window, including its incoming position/heading."""
    if 'valid' in data and not data['valid'][max(0, first-1):last, j].all():
        return dict(censored=True)
    xy = data['velocity'][first:last, j, :2]
    yaw = data['angular'][first:last, j, 2]
    tail = slice(max(first, last-round(1/dt)), last)
    tail_xy = data['velocity'][tail, j, :2]
    # The legacy buffer uses Genesis's intrinsic XYZ angles (rpy=False).
    # World heading is the projected body +X axis, not its third Euler angle.
    roll, pitch, yaw_angle = data['rpy'][max(0, first-1):last, j].T
    heading = np.unwrap(np.arctan2(np.cos(roll)*np.sin(yaw_angle) + np.sin(roll)*np.sin(pitch)*np.cos(yaw_angle),
                                   np.cos(pitch)*np.cos(yaw_angle)))
    position = data['position'][max(0, first-1):last, j, :2]
    inside = (np.linalg.norm(xy, axis=1) <= .02) & (abs(yaw) <= .03)
    outside = np.flatnonzero(~inside)
    start = int(outside[-1]+1) if len(outside) else 0
    result = dict(censored=False, path_m=float(np.linalg.norm(np.diff(position, axis=0), axis=1).sum()),
        heading_change_rad=float(heading[-1]-heading[0]),
        final_yaw_rate_rms_rad_s=float(np.sqrt(np.mean(data['angular'][tail, j, 2]**2))),
        final_component_drift_rms_m_s=float(np.sqrt(np.mean(tail_xy**2))),
        final_planar_speed_rms_m_s=float(np.sqrt(np.mean(np.sum(tail_xy**2, axis=1)))),
        residual_planar_speed_m_s=float(np.linalg.norm(tail_xy.mean(0))),
        settled_after_s=(start+1)*dt if len(inside)-start >= int(np.ceil(.25/dt)) else None)
    if 'wheel_speed' in data:
        speeds = data['wheel_speed'][tail, j]
        result['final_wheel_speed_rms_rad_s'] = float(np.sqrt(np.mean(speeds**2)))
        result['final_wheel_spin_fraction_gt_1rad_s'] = float((abs(speeds) > 1.).mean())
    return result


def metrics(data, dt, stop_windows=None):
    error = data['estimate'] - data['target']
    actual = np.concatenate((data['velocity'][..., :2], data['angular'][..., 2:3]), axis=-1)
    tracking = actual - data['commands']
    stop_windows = stop_windows or [('stop', 4., 6.), ('final_stop', 10., len(actual)*dt)]
    stops = [(round(a/dt), round(b/dt)) for _, a, b in stop_windows]
    result = []
    for j in range(actual.shape[1]):
        windows = {name: stop_metrics(data, j, a, b, dt) for (name, _, _), (a, b) in zip(stop_windows, stops)}
        paths = [v.get('path_m') for v in windows.values()]
        drifts = [v.get('final_component_drift_rms_m_s') for v in windows.values()]
        valid = data.get('valid', np.ones(actual.shape[:2], dtype=bool))[:, j]
        e = error[valid, j]
        result.append(dict(
            estimator_bias_m_s=e.mean(0).tolist(),
            estimator_rmse_m_s=np.sqrt((e**2).mean(0)).tolist(),
            estimator_p95_abs_m_s=np.quantile(abs(e), .95, axis=0).tolist(),
            estimator_max_abs_m_s=abs(e).max(0).tolist(),
            estimator_lag_s=[None if k is None else k*dt for k in lag_samples(data['estimate'][valid, j], data['target'][valid, j])],
            tracking_rmse=np.sqrt((tracking[valid, j]**2).mean(0)).tolist(),
            stop_path_m=paths, final_stop_drift_rms_m_s=drifts,
            final_stop_planar_speed_rms_m_s=[v.get('final_planar_speed_rms_m_s') for v in windows.values()],
            stop_windows=windows,
            cross_axis_rms=np.sqrt(np.mean(np.where(data['commands'][valid, j] == 0, actual[valid, j], 0.)**2, axis=0)).tolist(),
            max_tilt_rad=float(np.linalg.norm(data['rpy'][valid, j, :2], axis=1).max()),
            min_height_m=float(data['position'][valid, j, 2].min()),
            saturation_fraction=float((abs(data['actions'][valid, j]) >= 1.-1e-6).mean()),
            falls=int(data['fallen'][valid, j].sum()),
            abnormal_contact_ticks=int((data['contacts'][valid, j] > 0).sum())))
        if 'wheel_speed' in data:
            zero = (data['commands'][:, j] == 0).all(axis=1) & valid
            rates = np.diff(data['actions'][:, j], axis=0) / dt
            pairs = valid[1:] & valid[:-1]
            leg_slew = np.diff(data['leg_target'][:, j], axis=0)[pairs] / dt
            wheel_slew = np.diff(data['wheel_target'][:, j], axis=0)[pairs] / dt
            result[-1].update(
                zero_command_wheel_speed_rms_rad_s=float(np.sqrt(np.mean(data['wheel_speed'][zero, j]**2))),
                action_rate_rms_per_s=float(np.sqrt(np.mean(rates[pairs]**2))),
                leg_target_slew_rms_rad_s=float(np.sqrt(np.mean(leg_slew**2))),
                wheel_target_slew_rms_rad_s2=float(np.sqrt(np.mean(wheel_slew**2))),
                roll_pitch_rms_rad=np.sqrt(np.mean(data['rpy'][valid, j, :2]**2, axis=0)).tolist(),
                roll_pitch_rate_rms_rad_s=np.sqrt(np.mean(data['angular'][valid, j, :2]**2, axis=0)).tolist(),
                vertical_velocity_rms_m_s=float(np.sqrt(np.mean(data['velocity'][valid, j, 2]**2))))
    return result


def compare(parent, estimated):
    t = TOLERANCES
    checks = []
    for a, b in zip(parent, estimated):
        checks.append(dict(
            tracking=bool(np.all(np.array(b['tracking_rmse']) <= np.array(a['tracking_rmse'])*t['tracking_rmse_ratio'] + t['tracking_rmse_allowance'])),
            stopping=bool(np.all(np.array(b['stop_path_m'], dtype=float) <= np.array(a['stop_path_m'], dtype=float)*t['stop_path_ratio'] + t['stop_path_allowance_m'])),
            drift=bool(np.all(np.array(b['final_stop_drift_rms_m_s'], dtype=float) <= np.array(a['final_stop_drift_rms_m_s'], dtype=float) + t['drift_allowance_m_s'])),
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
    three_way = bool(getattr(args, 'frozen_checkpoint', None))
    duration, seeds, push_time = (26., [2843, 2844], 20.4) if three_way else (14., [1729, 1730], 10.4)
    if three_way:
        if not args.comparison_checkpoint:
            raise ValueError("Three-way review requires --comparison_checkpoint and --frozen_checkpoint")
        cases += ['precision_forward', 'precision_lateral_positive', 'precision_lateral_negative', 'precision_yaw']
        commands += [[.08, 0, 0], [0, .04, 0], [0, -.04, 0], [0, 0, .15]]
    n = len(cases)
    declaration = dict(criteria=TOLERANCES, criteria_status="proposed simulation screening tolerances",
        cases=dict(zip(cases, commands)), sequence="0-1 stand; 1-4 ramp/start/hold; 4-6 stop; 6-8 reverse half; 8-10 original half; 10-14 stop. Commands refreshed at 10 Hz.",
        seeds=seeds, training_seed=training.seed,
        conditions=['nominal', 'saved observation noise + 20 ms sensor delay + 40 ms sample holding + 30 N stop push'],
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint))
    declaration.update(panel='three_way_review' if three_way else 'pilot',
        stop_definitions='component RMS = sqrt(mean(v_xy**2)); planar-speed RMS = sqrt(mean(vx**2+vy**2)); final windows are last 1 s',
        settling='planar <=.02 m/s and abs(body yaw rate)<=.03 rad/s continuously to window end for >=.25 s; diagnostic only',
        noise_assumptions='Synthetic uniform sensor noise, not measured IMU statistics. No extra sensor bias. Nominal physics in both conditions.',
        push_time_s=push_time, duration_s=duration)
    if three_way:
        declaration['sequence'] = '0-1.4 stand; 1.4-4.6 ramp/hold; 4.6-10.6 exact zero; 10.6-13.6 reverse half; 13.6-16 original half; 16-26 exact zero. 10 Hz commands; stress push at 20.4 s.'
        declaration['frozen_checkpoint_sha256'] = sha256(args.frozen_checkpoint)
    if args.comparison_checkpoint:
        declaration['comparison_checkpoint'] = str(Path(args.comparison_checkpoint).resolve())
        declaration['comparison_checkpoint_sha256'] = sha256(args.comparison_checkpoint)
    write_json(out / 'screen_definition.json', declaration)  # Before examining outcomes.
    configure_nominal(cfg, args)
    cfg.env.num_envs, cfg.env.episode_length_s = n, 40.
    cfg.env.capture_transitions = True
    cfg.env.capture_closed_loop = True
    cfg.env.play_mode = True
    args.num_envs = n
    env, _ = task_registry.make_env(args.task, args=args, env_cfg=cfg)
    runner, _ = task_registry.make_alg_runner(env, args=args, train_cfg=training, save_config=False)
    candidate_actor = copy.deepcopy(runner.alg.actor.state_dict())
    candidate_estimator = copy.deepcopy(runner.alg.estimator.state_dict())
    parent_actor = candidate_actor
    if args.comparison_checkpoint:
        parent_actor = torch.load(args.comparison_checkpoint, map_location=env.device, weights_only=False)['actor_state_dict']
    elif runner.alg.stage_counts['finetune_updates']:
        raise ValueError("Fine-tuned screen requires --comparison_checkpoint for the immutable truth-driven parent")
    frozen = torch.load(args.frozen_checkpoint, map_location=env.device, weights_only=False) if three_way else None
    if frozen:
        from .estimator_ppo import assert_frozen_parent
        assert_frozen_parent(torch.load(args.comparison_checkpoint, map_location=env.device, weights_only=False), frozen)
        if frozen['velocity_estimator']['spec'] != runner.alg.estimator_spec:
            raise ValueError("Frozen and fine-tuned estimator feature contracts differ")
    from .export import tensor_hash
    declaration['parameter_hashes'] = dict(parent_actor=tensor_hash(parent_actor),
        candidate_actor=tensor_hash(candidate_actor), candidate_estimator=tensor_hash(candidate_estimator),
        frozen_actor=tensor_hash(frozen['actor_state_dict']) if frozen else None,
        frozen_estimator=tensor_hash(frozen['velocity_estimator']['model']) if frozen else None)
    write_json(out / 'screen_definition.json', declaration)
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
        env.compute_observations()  # Same-tick selective reset, as used by precision_dr.
        torch.testing.assert_close(env.sensor_history.history[1:], history[1:], rtol=0., atol=0.)
        torch.testing.assert_close(env.sensor_history.history[0], env.sensor_history.history[0, -1:].expand_as(history[0]), rtol=0., atol=0.)
        assert bool(env.sensor_history.valid.all())
        report['native_lifecycle_checks'] = 'passed: repeated getters, phase/RNG stability, selective reset and first-sample padding'
        for condition in ('nominal', 'sensor_stress'):
            sensor_cfg = copy.deepcopy(env.cfg.velocity_estimator)
            sensor_cfg.update(sensor_delay_steps=0 if condition == 'nominal' else 1,
                              sensor_hold_steps=1 if condition == 'nominal' else 2)
            env.add_noise = condition != 'nominal'
            initial_state = None
            selections = [('truth', parent_actor, candidate_estimator)]
            if frozen:
                selections.append(('frozen_estimated', frozen['actor_state_dict'], frozen['velocity_estimator']['model']))
            selections.append(('estimated', candidate_actor, candidate_estimator))
            for label, actor_state, estimator_state in selections:
                source = 'truth' if label == 'truth' else 'estimated'
                runner.alg.actor.load_state_dict(actor_state)
                runner.alg.estimator.load_state_dict(estimator_state)
                env.sensor_history = SensorHistory(n, env.velocity_estimator_spec(), sensor_cfg, env.device)
                set_seed(seeds[condition != 'nominal'])
                env.set_fixed_command((0., 0., 0.))
                env.reset()
                env.compute_observations()
                state_at_start = torch.cat((env.base_pos, env.base_quat, env.dof_pos, env.dof_vel), dim=1).clone()
                if initial_state is None:
                    initial_state = state_at_start
                else:
                    torch.testing.assert_close(state_at_start, initial_state, rtol=0., atol=0.)
                policy = inference_policy(runner, env, source)
                data = {k: [] for k in ('histories', 'estimate', 'target', 'commands', 'velocity', 'angular',
                    'position', 'rpy', 'actions', 'fallen', 'contacts', 'reset', 'reset_frames',
                    'wheel_speed', 'leg_target', 'wheel_target', 'valid')}
                active = torch.ones(n, dtype=torch.bool, device=env.device)
                with torch.inference_mode():
                    for tick in range(round(duration/env.dt)):
                        t = tick * env.dt
                        if tick % 5 == 0:  # Navigation publishes at 10 Hz, policy at 50 Hz.
                            gain = min(1., (t-1.)/.5) if 1 <= t < 4 else -.5 if 6 <= t < 8 else .5 if 8 <= t < 10 else 0.
                            if three_way:
                                gain = min(1., (t-1.4)/.6) if 1.4 <= t < 4.6 else -.5 if 10.6 <= t < 13.6 else .5 if 13.6 <= t < 16 else 0.
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
                        if condition != 'nominal' and tick == round(push_time/env.dt):
                            env.push_force[:, 0, :2] = torch.tensor([24., 18.], device=env.device)
                            env.push_steps_left[:] = round(.15/env.cfg.sim.dt)
                        env.step(actions)
                        state = env.transition_state
                        data['valid'].append(active.clone())
                        active &= ~state['reset_buf'].bool()
                        data['wheel_speed'].append(state['dof_vel'][:, env.wheel_action_indices].clone())
                        targets = state['applied_actions'] * env.action_scale
                        data['leg_target'].append((targets + env.default_dof_pos)[:, env.leg_action_indices].clone())
                        data['wheel_target'].append(targets[:, env.wheel_action_indices].clamp(
                            -env.cfg.control.wheel_velocity_target_limit, env.cfg.control.wheel_velocity_target_limit).clone())
                        for key, field in (('velocity', 'base_lin_vel'), ('angular', 'base_ang_vel'), ('position', 'base_pos'),
                                           ('rpy', 'rpy'), ('actions', 'actions'), ('fallen', 'fallen')):
                            data[key].append(state[field].clone())
                        data['contacts'].append((state['nonfoot_contact_count'] + state['undesired_self_pair_count']).clone())
                arrays = {k: torch.stack(v).cpu().numpy() for k, v in data.items()}
                name = f'{condition}_{label}'
                stops = [('ordinary_stop', 4.6, 10.6), ('final_stop', 16., duration)] if three_way else None
                if three_way and condition != 'nominal':
                    stops = [('ordinary_stop', 4.6, 10.6), ('pre_push_stop', 16., push_time), ('post_push_recovery', push_time, duration)]
                report['results'][name] = dict(zip(cases, metrics(arrays, env.dt, stops)))
                for row in report['results'][name].values():
                    row['velocity_source'] = source
                # Keep temporal arrays for analysis; flat history alone is a portable export input.
                histories = arrays.pop('histories')
                np.savez_compressed(out / f'{name}.npz', histories=histories.reshape(-1, *histories.shape[2:]), **arrays)
                print(f"Completed {name}: {len(histories)} ticks x {n} held-out episodes", flush=True)
            report['comparisons'][condition] = compare(list(report['results'][f'{condition}_truth'].values()),
                                                       list(report['results'][f'{condition}_estimated'].values()))
            if frozen:
                report['comparisons'][condition+'_frozen'] = compare(list(report['results'][f'{condition}_truth'].values()),
                    list(report['results'][f'{condition}_frozen_estimated'].values()))
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
