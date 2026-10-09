# Go2-W laptop handoff — 2026-10-09

**Experimental candidate for restricted flat/open-area diagnostic Sim2Sim.
Not fully qualified for navigation, disturbances or hardware.**

This one-time handoff distributes the validated candidate and two complete learning
checkpoints. It does **not** replace `ressources/pretrained/go2w`, establish a new
qualified default, or authorize further training. Only copied inference distribution
metadata changed; its actor, normalizer, parity samples, checker and assets are the
validated stage's exact bytes. Historical staged-selection fields remain evidence.

## Contents and identity

| Location | Purpose / identity |
| --- | --- |
| `inference/go2w/` | Standalone actor package: `policy.pt`, contract, original saved config, provenance, real parity samples, CPU checker, README and 14 bundled asset files |
| `runs/go2w_2026-10-08_23-45-24/model_1999.pt` | Selected candidate: **2001 lineage updates**, native label1999, training schema3 |
| `runs/go2w_2026-10-08_09-57-35/model_1000.pt` | Direct geometric fallback: **1001 updates**, native label1000, training schema2 |
| Each run's `config.yaml` and `preparation.json` | Byte-exact original saved settings and small initialization/lineage record |
| `lineage.json`, `manifest.json`, `validation.json` | Code/checkpoint identities, file sizes/hashes/purposes, and isolated-copy checks |
| `environment/` | Actual desktop versions and sanitized package/conda snapshots; no credentials or editable source paths |

Candidate SHA256:
`2f8f3a96ad7fdf83c2966b44d10fc76ac274d9d81130a470af881f81a5980bf6`.
Fallback SHA256:
`49339c5bde1cdfaa6d7555ecd690ba9b2e64728987fdff9c1335fef82b7bfd4a`.
Training code revisions are respectively `5b70e79de134107ef08be4bf92cc912bd334ce8b`
and `376c76e3be9f4d7933e016af759ea9c5d99a5921`; both recorded clean trees.
The compatible handoff source revision before this packaging commit is
`25e9dfdb5f5011266d236f73caa4d53304471e0f`. Pull this handoff's `testing` commit
for its current compatible loader; changing checkout is unnecessary.

The actor is **58 inputs / 16 actions** with trained normalization embedded once.
The **65-input critic is training-only**. Inference **interface2** is distinct from
candidate training **schema3**. `policy.pt` contains deterministic actor inference,
not Adam, critic or a resumable learning state. Both `model_*.pt` files retain actor,
critic, both normalizers, learned Gaussian std, Adam moments/steps/LR and iteration.
The fallback's saved Gaussian precision and schema2 remain authoritative on loading.
Saved YAMLs are provenance, not another current training parameter source.

Assets are bundled once inside the portable inference package; preserve that full
relative hierarchy. Genesis replay uses this repository's tracked measured URDF
and meshes. No desktop logs, evaluation traces, videos, caches, TensorBoard history,
ancestors or intermediate checkpoints are needed. Full learning state is preserved,
but a later continuation is not an exact simulator/RNG-state continuation.

## Laptop pull and CPU check

From the existing repository, preserve any laptop changes before switching branches:

```text
git switch testing
git pull --ff-only origin testing
python -I ressources/handoffs/go2w_2026-10-09/inference/go2w/check_policy.py
```

Activate an existing compatible laptop environment first. The checker needs only
PyTorch, NumPy and the standard library, uses CPU, and checks all coupled payloads,
asset closure, normalizer immutability and unchanged parity tolerances. It can also
run with the entire `inference/go2w` directory copied outside the repository.

For an **existing compatible CUDA Genesis environment**, this PowerShell example
loads the copied candidate into the existing replay script for ten seconds of stand:

```powershell
$run = (Resolve-Path 'ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24').Path
python -m robot_gym.scripts.play --task go2w --load_run "$run" --checkpoint 1999 --num_envs 1 --seed 1 --rl_device cuda:0 --command_vx 0 --steps 500 --no_export
```

On a POSIX shell, set the same absolute directory with
`run="$(pwd)/ressources/handoffs/go2w_2026-10-09/runs/go2w_2026-10-08_23-45-24"`
and use the identical `python -m ... --load_run "$run" ...` arguments. These are
instructions, not a replay performed during packaging. Confirm the laptop's
supported backend/GPU before running; its OS, software and VRAM are not assumed.
For the fallback, select its distinct copied directory and `--checkpoint 1000`.

`--load_run` accepts an **absolute run directory**, regardless of the default log
root. `--checkpoint` selects the explicit native label. The adjacent `config.yaml`
is restored before runtime options; no `--reference_config` is needed. Stored old
absolute source paths and the candidate's historical parent `load_run` are metadata:
the explicit resolved checkpoint takes precedence and does not recurse to ancestors.
The ordinary full-state resume resolver was checked without performing training.
No training or fine-tuning command is supplied here.

## Environment and validation limits

Desktop measured environment: Python3.11.14, PyTorch2.9.0+cu130 (CUDA build13.0),
Genesis1.4.1, RSL-RL5.5.1, NumPy2.2.6, TensorDict0.10.0, Numba0.62.1 and
Quadrants1.3.0; Windows AMD64, RTX4070Ti (12282MiB), driver617.42.
See `environment/inventory.json` for the complete measured record and the existing
repository `conda_env.yaml` for context. **Compare the laptop environment before
changing packages.** The snapshots record installed distributions, not a portable
Windows-to-Linux lock; do not install the whole list blindly.

The inventory uses the requested portable repository label `pip install -e .`,
without a desktop editable path. This revision actually imports from the repository
root and has no root `setup.py`/`pyproject.toml`; do not execute that label as an
installation command for this checkout. No environment was changed to package it.

The existing CPU checker passed in a fresh isolated copy. Both copied full checkpoints
loaded through native RSL-RL and passed the existing exact full-state comparison:
Adam80040/40040, both normalizer counts524550144/262406144, LR3e-4, labels1999/1000.
Access to original logs/evaluation/ancestors was blocked during path/state checks.
There were **zero optimizer updates and zero simulator steps**. Synthetic shape
inputs construct native models only; they are not deployment velocity placeholders.
No Linux/PhysX/hardware agreement or new policy-quality result is claimed.

## Restricted proposed test scope — limitations remain

Use nominal dynamics and simulator velocity at the authored base origin; forward
up to.5m/s, lateral±.3m/s in bouts up to2m with explicit stops, and short yaw up to
±.8rad/s. Preserve the raw actor, trained reference, demand-phase/history semantics
and P-leg/V-wheel transforms. No heading adapter, wheel deadband or action override.

**This scope is not a passed application tolerance.** Thirty-second lateral
curvature remains about1.2m; at2m, heading errors are−3.53/+4.47deg and maximum
cross-track .0122/.0459m. Apex action clipping remains. Lateral full-stop paths
increased from CK1000 .0396/.0425m to .0476/.0610m. These limitations are not solved
by distributing the package, and acceptance as a pretrained default is not granted.
Keep the frozen functional requirements. Refer to the existing
[completed review](../../../docs/go2w_training.md#completed-refinement-transfer-review-2026-10-09)
and [comparison JSON](../../../docs/go2w_refinement_transfer_2026-10-09.json), rather
than duplicating the raw evaluation history in this handoff.
