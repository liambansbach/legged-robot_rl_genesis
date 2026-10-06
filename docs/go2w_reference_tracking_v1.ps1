# PREPARED ONLY. This 2000-update production command has NOT been executed.
# New actor/critic/normalizers/Gaussian/Adam; the YAML is configuration only.
# Final 2026-10-06 recipe: canonical [0, .70, -1.40, 0] per leg.
# Required vx/vy/yaw: [-.30,1.00]/[-.30,.30]/[-1.00,1.00].
# Training reserve: [-.40,1.20]/[-.50,.50]/[-1.50,1.50], one outside axis only.
# Pushes every 5-10 s: 80% 20-50 N, 15% 50-100 N, 5% 100-150 N for .10-.20 s.
Set-Location 'C:/Users/Liamb/SynologyDrive/TUM/3_Semester/dodo_alive/legged-robot_rl_genesis'
$env:NUMBA_CACHE_DIR = Join-Path (Get-Location) '.cache/numba'
$env:GS_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/genesis'
$env:QD_OFFLINE_CACHE_FILE_PATH = Join-Path (Get-Location) '.cache/quadrants'
$env:PYTHONIOENCODING = 'utf-8'
& 'C:/Users/Liamb/anaconda3/envs/genesis-gpu/python.exe' -m robot_gym.scripts.train `
  --task go2w --go2w_profile transfer_v3 `
  --go2w_fresh_recipe docs/go2w_reference_tracking_v1.yaml `
  --num_envs 4096 --max_iterations 2000 --seed 1 `
  --logger tensorboard --training_diagnostics --headless --rl_device cuda:0
exit $LASTEXITCODE
