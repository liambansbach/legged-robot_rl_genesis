# Go2-W testing policy

**Experimental Go2-W policy for controlled simulation testing. Not validated for hardware. Subject to replacement.**

Selected P / CK2498 completed 2500 updates with event_step_v1, sagittal stance
weight 2.0 and original event quality (no sufficient_clearance override).
policy.pt contains the existing deterministic TorchScript actor, trained
observation normalizer and output clipping. It is not a resumable PPO checkpoint.
The actor SHA-256 is cb8454470921a5a44e8a5914a0a2dd301886ff9b7e7b6ef1dffdf5c6b661a35c.
Source identities and evidence are in [provenance.json](provenance.json).

| Interface | Value |
|---|---|
| Input | Finite float32 [N,56], N >= 1; physical scaling before the embedded normalizer |
| Output | Float32 [N,16], deterministic normalized actions clipped to [-1,1] |
| Observations | Body linear/angular velocity, projected gravity, body command, 12 leg errors, 16 joint velocities, 16 previous issued clipped actions |
| Joint/action order | [FL,FR,RL,RR] x [hip,thigh,calf,foot], interleaved; map by exact names |
| Leg targets | Nominal q + action x .30/.35/.40 rad; Kp=40 Nm/rad, Kd=1 Nm/(rad/s) |
| Wheel targets | Action x 18 rad/s, explicit cap 20 rad/s; Kp=0, Kv=1 Nm/(rad/s) |
| Timing | .02 s policy, .005 s reference physics, four substeps holding targets |
| Nominal delay | Zero; training's 0–2 tick actuator delay is not a required 40 ms delay |

[contract.json](contract.json) contains every slice, name, gain, limit and frame.
All physical observation scales happen to be 1.0 here; this does not remove the
trained normalizer. Do not normalize twice. The module clips observations at
100 before normalization and clips outputs at 1. Absolute wheel angle is omitted.
Mechanical positions, effort limits and motor velocity metadata are distinct
from action/target bounds; leg velocity metadata is not a hard velocity clamp.

The base frame is right-handed (+X forward, +Y left, +Z up). Quaternions are wxyz,
base-link to world. Linear velocity belongs to the authored base_link origin,
not its COM. If a simulator reports COM velocity, first use
v_origin = v_COM + omega_world x (p_origin - p_COM), then rotate into the body
frame. Use the target simulator's actual COM after fixed-link merging.
Sim2Sim can supply ground truth; a real-robot estimator is unvalidated.
Never substitute zero velocity.

The only required external policy history is the previous issued **clipped
normalized** action. Clear it to zero on reset; do not use a delayed action,
physical target or measured joint state in its place. The MLP has no recurrent
state. An optional actuator-delay model has separate history. Event gates,
contact trackers and the outer 10 Hz feedback controller are not in this module.

From the repository root, using the target environment's compatible PyTorch
and NumPy (do not install Genesis into IsaacLab):

    python ressources/pretrained/go2w/check_policy.py

The check verifies payload/asset integrity before loading the model, then runs
the saved 96-input parity batch on CPU without updating normalization buffers.
Windows CPU parity passed; Linux/PyTorch-version compatibility must be checked
there. Hash failures and parity failures are blockers, not reasons to widen tolerances.
Canonical XML/DAE hashes permit only CRLF-to-LF checkout differences.

Inference-only example after that check, with a correctly prepared observation
tensor called observations; this is not a robot controller. **Zero observations
are not a valid stand state.**

```python
from pathlib import Path
import torch

path = Path("ressources/pretrained/go2w/policy.pt")
policy = torch.jit.load(str(path), map_location="cpu").eval()
# observations: CPU float32 [N,56], assembled according to contract.json
with torch.inference_mode():
    actions = policy(observations)
```

Use the canonical [URDF](../../robots/go2w/urdf/go2w_description.urdf) with its
relative [mesh directory](../../robots/go2w/dae/); do not move the XML alone.
All 13 referenced meshes exist with exact case; no external texture is referenced.
config.yaml is the exact historical training snapshot for provenance. Its
machine paths and training DR settings are not runtime dependencies or nominal
inference settings.

Start the later Linux comparison with the same nominal .5 m/s straight and
[.5,+/-.1] diagonal feedback paths in the
[recorded experiment](../../../docs/go2w_diagnostics.md#frozen-feedback-candidates-and-portable-inference--29-september-2026).
The 10 Hz evaluation bounds (vx [-.25,.70], vy [-.20,.20], yaw [-.50,.50])
are optional outer-controller bounds, distinct from broader training ranges.
Neither defines an all-command deployment envelope. Do not extrapolate to strong
yaw or pure-lateral .2 transitions.

Known limits: straight front-thigh target saturation; small-command endpoint
bias; archived pure-lateral calf gap .006765 rad versus minimum .158113 rad in
feedback; opposing internal wheel moments. The maximum sampled feedback
force/effort ratio was 70.6%, not a torque-saturation frequency or hardware safety
certificate. Policy-rate samples exclude no between-sample violation. They do
not identify electrical power, continuous motor ratings or PD causality.
Linux/PhysX and estimator validation are still pending.

For replacement, update all coupled payload, contract, config, provenance and
parity files together, run the checker, then commit and push normally. Keep this
stable directory; Git history records earlier choices. Change interface_version
only if semantics change, not just weights. Record the Git commit and actor SHA
in downstream experiments. Load a fixed snapshot before a rollout; never hot-reload
files changed by a pull mid-rollout.
