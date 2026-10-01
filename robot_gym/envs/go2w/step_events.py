"""Training-only completed wheel steps. No actor inputs or action overrides."""

import torch
from genesis.utils.geom import inv_quat, transform_quat_by_quat
from robot_gym.utils.diagnostics import cylinder_clearance, rotate_wxyz


def step_demand(commands):
    lateral = ((commands[:, 1].abs() - 0.01) / 0.04).clamp(0, 1)
    yaw = ((commands[:, 2].abs() - 0.10) / 0.15).clamp(0, 1)
    return torch.maximum(lateral, yaw)


def lateral_high(completed_updates, cfg):
    start, end = cfg.lateral_curriculum_updates
    low, high = cfg.lateral_tail_high_range
    return low + (high - low) * min(1.0, max(0.0, (completed_updates - start) / (end - start)))


class WheelStepEvents:
    def __init__(self, count, device, cfg, geometry):
        self.cfg, self.geometry = cfg, geometry
        shape = (count, 4)
        for name in ("unloaded_time", "reload_time", "support_time", "elapsed", "duration",
                     "credit", "gate", "onset_height", "peak_actual", "peak_use", "quality",
                     "payment", "reposition"):
            setattr(self, name, torch.zeros(shape, device=device))
        for name in ("unloaded", "supported", "active", "confirmed", "support_valid",
                     "completed", "valid", "censored"):
            setattr(self, name, torch.zeros(shape, dtype=torch.bool, device=device))
        self.unloaded.fill_(True)  # Unknown initial support must first reach the reload threshold.
        self.onset_base = torch.zeros((*shape, 3), device=device)
        self.onset_quat = torch.zeros((*shape, 4), device=device)
        self.onset_quat[..., 0] = 1
        self.takeoff = torch.zeros_like(self.onset_base)
        self.touchdown = torch.zeros_like(self.takeoff)
        self.onset_commands = torch.zeros_like(self.takeoff)
        self.commands = torch.zeros((count, 3), device=device)
        self.generation = torch.zeros(count, dtype=torch.long, device=device)
        self.onset_generation = torch.zeros(shape, dtype=torch.long, device=device)
        self.censored_count = torch.zeros_like(self.onset_generation)
        self.update_count = 0

    def reset(self, ids):
        # All per-environment history, including settling and credit, is discarded.
        if len(ids) == 0:
            return
        for value in vars(self).values():
            if torch.is_tensor(value):
                value[ids] = 0
        self.unloaded[ids] = True
        self.onset_quat[ids, :, 0] = 1

    def command_changed(self, commands):
        changed = (commands != self.commands).any(dim=1)
        self.censored[:] = self.active & changed[:, None]
        self.censored_count += self.censored.long()
        self.generation += changed.long()
        self.active.masked_fill_(changed[:, None], False)
        self.confirmed.masked_fill_(changed[:, None], False)
        self.credit.masked_fill_(changed[:, None], 0)
        self.support_time.masked_fill_(changed[:, None], 0)
        self.commands.copy_(commands)

    def update(self, dt, commands, base_pos, base_quat, wheel_pos, wheel_quat, loads, failed,
               actual_clearance=None):
        self.update_count += 1
        self.command_changed(commands)
        c = self.cfg
        gate = step_demand(commands)
        self.payment.zero_()
        self.quality.zero_()
        self.completed.zero_()
        self.valid.zero_()
        self.credit[:] = (self.credit + gate[:, None] * dt).clamp(max=c["credit_cap"])
        self.credit.masked_fill_(gate[:, None] == 0, 0)

        inverse = inv_quat(base_quat)
        relative_pos = rotate_wxyz(inverse[:, None], wheel_pos - base_pos[:, None])
        relative_quat = transform_quat_by_quat(wheel_quat, inverse[:, None].expand_as(wheel_quat))
        center = relative_pos + rotate_wxyz(relative_quat, self.geometry[0])
        # The environment already refreshed this geometry for this transition.
        actual = (cylinder_clearance(wheel_pos, wheel_quat, *self.geometry)
                  if actual_clearance is None else actual_clearance)
        finite = (torch.isfinite(base_pos).all(dim=-1) & torch.isfinite(base_quat).all(dim=-1)
                  & torch.isfinite(wheel_pos).all(dim=(1, 2)) & torch.isfinite(wheel_quat).all(dim=(1, 2))
                  & torch.isfinite(loads).all(dim=1) & torch.isfinite(commands).all(dim=1))
        failed = failed | ~finite

        previous_unloaded = self.unloaded.clone()
        self.unloaded[:] = torch.where(loads <= c["unload_force"], True,
                                      torch.where(loads >= c["reload_force"], False, self.unloaded))
        start = self.unloaded & ~previous_unloaded
        eligible = start & (self.support_time >= c["prior_support"] - 1e-7) & (gate[:, None] > 0) & ~failed[:, None]
        self.unloaded_time[:] = torch.where(self.unloaded, self.unloaded_time + dt, 0)
        self.reload_time[:] = torch.where(~self.unloaded, self.reload_time + dt, 0)
        self.supported.masked_fill_(self.unloaded, False)
        self.supported |= self.reload_time >= c["reload_dwell"] - 1e-7
        self.support_time[:] = torch.where(self.supported, self.support_time + dt, 0)

        self.elapsed += self.active * dt
        self.active |= eligible
        self.elapsed.masked_fill_(eligible, 0)
        self.onset_base.copy_(torch.where(eligible[..., None], base_pos[:, None], self.onset_base))
        self.onset_quat.copy_(torch.where(eligible[..., None], base_quat[:, None], self.onset_quat))
        self.takeoff.copy_(torch.where(eligible[..., None], center, self.takeoff))
        self.onset_height.copy_(torch.where(eligible, actual, self.onset_height))
        self.onset_commands.copy_(torch.where(eligible[..., None], commands[:, None], self.onset_commands))
        self.onset_generation.copy_(torch.where(eligible, self.generation[:, None], self.onset_generation))
        self.gate.copy_(torch.where(eligible, gate[:, None], self.gate))
        self.peak_actual.masked_fill_(eligible, 0)
        self.peak_use.masked_fill_(eligible, 0)
        self.support_valid.masked_fill_(eligible, True)
        self.confirmed |= self.active & (self.unloaded_time >= c["unload_dwell"] - 1e-7)

        frozen_pos = self.onset_base + rotate_wxyz(self.onset_quat, relative_pos)
        frozen_quat = transform_quat_by_quat(relative_quat, self.onset_quat)
        limb = cylinder_clearance(frozen_pos, frozen_quat, *self.geometry) - self.onset_height
        usable = torch.minimum(actual.clamp_min(0), c["limb_factor"] * limb.clamp_min(0))
        sampling = self.active & self.unloaded
        self.peak_actual[:] = torch.where(sampling, torch.maximum(self.peak_actual, actual), self.peak_actual)
        self.peak_use[:] = torch.where(sampling, torch.maximum(self.peak_use, usable), self.peak_use)
        enough = (loads > c["unload_force"]).sum(dim=1) >= 2
        self.support_valid &= ~sampling | enough[:, None]
        first_reload = self.active & previous_unloaded & ~self.unloaded
        self.touchdown.copy_(torch.where(first_reload[..., None], center, self.touchdown))
        self.duration.copy_(torch.where(first_reload, self.elapsed, self.duration))
        # A first reload sample is retained while its dwell is confirmed.
        self.completed[:] = self.active & self.supported
        delta = rotate_wxyz(self.onset_quat, self.touchdown - self.takeoff)
        self.reposition[:] = delta[..., :2].norm(dim=-1)
        low, high = c["duration_range"]
        self.valid[:] = (self.completed & self.confirmed & self.support_valid & ~failed[:, None]
                        & (self.duration >= low - 1e-7) & (self.duration <= high + 1e-7)
                        & (self.peak_actual >= c["minimum_height"]) & (self.peak_use >= c["minimum_height"])
                        & (self.reposition >= c["minimum_reposition"]))
        if c.get("quality_profile") == "sufficient_clearance":
            u = ((self.peak_use - 0.008) / 0.017).clamp(0, 1)
            height_score = (0.15 + 0.85 * u.square()) * torch.exp(
                -(torch.relu(self.peak_actual - 0.050) / 0.020).square())
        else:
            target = c["target_base"] + c["target_gate"] * self.gate
            u = ((self.peak_use - c["minimum_height"]) / (target - c["minimum_height"])).clamp(0, 1)
            height_score = (0.15 + 0.85 * u.square()) * torch.exp(
                -(torch.relu(self.peak_actual - target - c["overshoot_band"]) / c["overshoot_band"]).square())
        self.quality[:] = torch.where(self.valid, height_score * (self.reposition / c["full_reposition"]).clamp(0, 1), 0)
        self.payment[:] = self.quality * self.credit
        # Failed, flickering, overlong and completed attempts all spend their credit.
        cancel = (self.active & ((~self.confirmed & ~self.unloaded)
                  | (self.unloaded & (self.elapsed > high + 1e-7)))) | failed[:, None]
        end = cancel | self.completed
        self.active.masked_fill_(end, False)
        self.confirmed.masked_fill_(end, False)
        self.credit.masked_fill_(end, 0)
        self.support_time.masked_fill_(cancel, 0)
        self.payment.masked_fill_(failed[:, None], 0)


def project_log_std(optimizer, distribution):
    """Project only raw Gaussian parameters, retaining native Adam moments and LR."""
    if distribution.std_type != "log":
        raise ValueError("event_step_v1 requires log std")
    def project(*_):
        with torch.no_grad():
            distribution.log_std_param.clamp_(*distribution.log_std_range)
    return optimizer.register_step_post_hook(project)


def install_event_training(runner, env):
    """Keep the native runner/PPO; count completed updates and save explicit progress."""
    runner.event_std_hook = project_log_std(runner.alg.optimizer, runner.alg.actor.distribution)
    update, save, load = runner.alg.update, runner.save, runner.load

    def updated(*args, **kwargs):
        result = update(*args, **kwargs)
        env.completed_updates += 1
        return result

    def saved(path, infos=None):
        infos = dict(infos or {})
        infos["event_step_v1"] = {"completed_updates": env.completed_updates,
                                  "lateral_high": lateral_high(env.completed_updates, env.cfg.commands)}
        return save(path, infos)

    def loaded(path, *args, **kwargs):
        from pathlib import Path
        import yaml
        cfg = yaml.safe_load(Path(path).with_name("config.yaml").read_text())
        if any(cfg[k].get("go2w_profile") != env.cfg.go2w_profile for k in ("env_cfg", "train_cfg")):
            raise ValueError(f"{env.cfg.go2w_profile} rejects historical checkpoints")
        # Check metadata before native loading can change any learning state.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        progress = (checkpoint.get("infos") or {}).get("event_step_v1")
        if not progress or not isinstance(progress.get("completed_updates"), int):
            raise ValueError("Missing event_step_v1 completed-update state")
        result = load(path, *args, **kwargs)
        env.completed_updates = progress["completed_updates"]
        if env.command_resampling_enabled:
            env._resample_commands(env.all_env_ids)
            env.compute_observations()
        return result

    runner.alg.update, runner.save, runner.load = updated, saved, loaded
