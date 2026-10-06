# PREPARED ONLY. This 2000-update production command has NOT been executed.
# New actor/critic/normalizers/Gaussian/Adam; the YAML is configuration only.
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
