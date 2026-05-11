# Route B GPU smoke wrapper for PowerShell.
#
# Current Windows native GPU JAX has not been validated in this workspace.
# Recommended target is WSL2/Linux CUDA with a JAX build that can see GPU/CUDA.
# This script intentionally requires gpu_preflight.py --require_gpu before
# starting the 10k smoke.

$ErrorActionPreference = "Stop"

python -c "import jax, mujoco, brax, sys; print(sys.version); print('jax', jax.__version__, jax.default_backend(), jax.devices()); print('mujoco', mujoco.__version__); print('brax', getattr(brax, '__version__', 'unknown'))"

python scripts/gpu_preflight.py --impl jax --require_gpu

python -m learning.train_jax_sac_lift `
  --env_name G1JoystickFlatTerrain `
  --impl jax `
  --num_timesteps 10000 `
  --num_envs 128 `
  --num_eval_envs 32 `
  --batch_size 256 `
  --min_replay_size 1024 `
  --max_replay_size 8192 `
  --grad_updates_per_step 2 `
  --render False `
  --use_wandb False `
  --logdir ./logs/sac_lift_gpu_10k

Write-Host "Route B GPU smoke finished. Check logs and checkpoint under ./logs/sac_lift_gpu_10k"
