# Go2-W experimental handoff policy (interface 2)

**EXPERIMENTAL HANDOFF; NOT A QUALIFIED DEFAULT.** This is the explicitly
authorized one-time handoff copy of the validated stage. It does not replace the
published pretrained default. Historical staged-selection fields remain evidence. It is a nominal flat-ground diagnostic Sim2Sim candidate with
documented failures, not a qualified navigation or hardware policy.
**Experimental Sim2Sim candidate; validated only on documented tests; not validated for hardware.**

Source: `logs/go2w/go2w_2026-10-08_23-45-24/model_1999.pt`, native label 1999,
2001 completed lineage updates. The policy was resumed from geometric CK1000;
saved training schema 3 has actor 58 / critic 65. The portable interface version
is 2, independent of training schema. `config.yaml` is the exact saved training
record, not an input recipe or a resumable checkpoint.

Nominal forward and stand improved, but sustained lateral travel failed the
retained path/heading requirements: 30-second lateral +/- cross-track was
1.2023 / 1.2350 m and heading change -18.762 / +16.558 degrees. Repeated swing-apex
action clipping remains. Any proposed open-area diagnostic use would restrict
lateral bouts to at most 2 m at 0.3 m/s; the observed first-2-m heading deviations
were still -3.5255 / +4.4717 degrees. This restriction is not a passed application
tolerance. The proposed restricted test would use an open flat area, nominal
physics without pushes/DR, ground-truth authored-base velocity, the raw actor,
forward speed at most 0.5 m/s, lateral +/-0.3 m/s in bouts at most 2 m with explicit
stops, and short yaw commands at most +/-0.8 rad/s. Lateral full 8-second stop
paths also grew from source CK1000: +0.03961 to 0.04756 m and -0.04254 to 0.06099 m;
late XY speeds were 0.001888 / 0.003868 m/s. User acceptance of these deviations
is required before replacing the published package. Complete measured selection
notes and the separate 58-second navigation screen are in contract/provenance.
No hardware, disturbance, robustness, or full command-envelope qualification is
claimed. See the preserved review, rather than treating reward or parity as
physical performance evidence.

## Contents and CPU check

- `policy.pt`: unchanged native TorchScript export, deterministic actor mean,
  embedded trained empirical normalizer, observation clip and output clip.
- `contract.json`: exact dimensions, order, physical units, scaling, control,
  phase/history/reset semantics, measured plant and evaluation limitations.
- `config.yaml`: exact source run configuration; no training state is included.
- `provenance.json`: checkpoint/export/payload/asset hashes, local Git identities,
  old published-package provenance, runtime versions and parity provenance.
- `parity_samples.npz`: real six-case and consecutive navigation inputs/states,
  original native CPU outputs and recorded native GPU proposals.
- `check_policy.py`: stdlib + NumPy + PyTorch only. Run `python check_policy.py`.
- `assets/go2w`: exact measured URDF and recursively referenced mesh dependencies,
  copied byte-for-byte with relative paths intact. No repository assets are needed.

The checker verifies payload/asset hashes, numeric interface, physical observation
assembly, batch-one/batch native parity, clipping, normalization immutability,
continuous phase and issued-action history through the real navigation sequence.
Its sequence comparison feeds each CPU actor its own prior clipped action while
holding measured physical states fixed. This tests interface assembly, not new
simulated behavior. CPU parity does not establish Linux/PhysX compatibility.

## Interface and execution

Input is finite float32 `[N,58]`, output finite float32 `[N,16]` in [-1,1].
Raw scaled inputs are body linear velocity (0:3), body angular velocity (3:6),
gravity projected into body axes (6:9), requested body vx/vy/wz (9:12), twelve
leg position errors from the fixed reference (12:24), all sixteen joint velocities
(24:40), previous **issued clipped normalized actions** before actuator delay
(40:56), and demand times phase sin/cos (56:58). Continuous wheel angles are
excluded. All listed physical scales are one in this export; consult the contract.
The policy clips observations before its embedded empirical normalization; do not
normalize twice. Inference must not update normalization counts or statistics.

Coordinates are x-forward/y-left/z-up, quaternion wxyz. The requested yaw channel
and measured angular-z are body angular-z, not Euler heading derivative under tilt.
Linear velocity refers to
the authored base-link origin, expressed in body axes. Genesis already transports
its getter to that origin; do not perform a second COM correction. A new simulator
must supply equivalent measured states. No velocity estimator, world position,
absolute heading, privileged critic state, or heading controller is included.

Joint/action order is FL/FR/RL/RR, each hip/thigh/calf/foot. Leg targets are fixed
reference plus scaled clipped/delayed action; wheel targets are scaled velocity.
Scales, P/V modes, nominal gains, armature, hard limits and force limits come from
the contract and exact measured URDF. Leg actions are neither absolute angles nor
increments, and nonzero support actions are expected. Apply control at 200 Hz,
actor at 50 Hz, holding an actor command for four physics ticks. The evaluated
nominal delay is zero; training included configured 0-2 policy-tick delay.
At policy boundary t, the actor sees the previous issued clipped action
a_issued[t-1]. Clip the new a_issued[t], enqueue it at delay-buffer index0, then
apply a_issued[t-D] for all four physics ticks. D is measured in 20 ms policy ticks.
Reset clears the complete delay buffer and issued-action history.

The explicit `BaseTask.reset()` used to start this evaluation first sets the nominal
physical reset state and clears latent phase and issued-action history, then executes one native
zero-action 20 ms warmup. The first actor phase is therefore .025 cycles, using the
real measured post-warmup inputs. The bundled first navigation row contains those
physical inputs, phase and history; the external reset-snapshot path is provenance
only and is not needed to use or check this package.

Automatic per-environment `reset_idx` during `post_physics_step` is different:
it resets that environment's phase/history to zero and supplies newly reset physical
measurements to its next actor input, without another warmup tick. The first phase
is zero; other environments continue unchanged. The checker includes a synthetic
phase/history reset probe on a real recorded physical state, compared to the native
CPU actor. It is not a recorded automatic physical reset (none occurred here).

After every actor step advance phase by .025 modulo1; do not reset it or
history on command changes. The clock is demand(command) times sin/cos, using
the smooth demand formula in the contract; zero-demand clock does not stop latent
phase. Save previous clipped **issued** action for the next actor input. Training's
random initial phase is not used by this nominal replay protocol.

The CPU artifact is actor-only. Gaussian std, critic, Adam and geometric reference
state are not exported. Commands remain external requested velocities, with no
action deadband, stand switch, correction controller or hidden world feedback.
