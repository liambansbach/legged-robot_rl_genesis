# Dodo walking

`model_1499.pt` is the full native checkpoint. Replay loads it with the adjacent `config.yaml`. `policy_1.pt` is the separately exported TorchScript actor. This run uses `dodo_daimao.urdf` and eight position-controlled joints.

From the repository root, set `run` to this directory's absolute path.

| Shell | Command |
| --- | --- |
| Bash | `run="$(pwd)/ressources/pretrained/dodo/dodo_walking"` |
| PowerShell | `$run = (Resolve-Path 'ressources/pretrained/dodo/dodo_walking').Path` |

```sh
python -m robot_gym.scripts.play --task dodo --load_run "$run" --checkpoint 1499 --manual_control
```

Remove `--manual_control` to sample commands or use `--command_vx 0.2` for a fixed forward command. Installation and keyboard shortcuts are in the [project README](../../../../README.md).
