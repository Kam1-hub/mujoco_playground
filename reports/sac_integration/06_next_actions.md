# Next Actions

Status: updated on 2026-05-11 after Route B static implementation.

## Immediate Priority

1. Resolve the menagerie asset blocker before any env load/reset/step or training smoke is counted as validated.
2. Rerun env API checks on a host with assets or after explicitly authorized asset download.
3. Run CPU tiny smoke only on a suitable validation host or after explicit permission to simulate on this laptop.

## Validation Ladder To Run After Route B Lands

| Order | Command | Expected Status Before Running | Notes |
|---:|---|---:|---|
| 1 | `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | Confirms Route B files compile. |
| 2 | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help` | PASS | First required Route B CLI acceptance check passed. |
| 3 | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help` | READY | Previously passed; rerun to confirm PPO/Route A entry work was not disturbed. |
| 4 | `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"` | READY | Lightweight registry check; previously passed. |
| 5 | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | BLOCKED_BY_DEPENDENCY | Will block at env load until `mujoco_menagerie` assets are present or download is authorized. |
| 6 | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | BLOCKED_BY_DEPENDENCY | Same as flat terrain. |
| 7 | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --dry_run` | PASS | Dry-run avoids env load, runs one dummy SAC update, and writes a checkpoint. |
| 8 | PowerShell: `$env:JAX_PLATFORM_NAME="cpu"` then Route B `--num_timesteps 256 --num_envs 2 --num_eval_envs 2 --batch_size 8 --min_replay_size 16 --max_replay_size 128 --grad_updates_per_step 1 --render False --use_wandb False --logdir ./logs/sac_lift_cpu_tiny` | NOT VALIDATED | Requires menagerie assets and explicit permission/suitable host for simulation. |
| 9 | Route B GPU smoke `--num_timesteps 10000 --num_envs 128 --num_eval_envs 32 --batch_size 256 --min_replay_size 1024 --max_replay_size 8192 --grad_updates_per_step 2` | NOT VALIDATED | Requires WSL2/Linux CUDA JAX or another proven CUDA JAX runtime. |

## Asset Blocker Resolution Options

| Option | Status | Notes |
|---|---:|---|
| Provide `g1_env\external_deps\mujoco_menagerie` from an existing local copy | PREFERRED_IF_AVAILABLE | Avoids network and keeps validation deterministic. |
| Authorize guarded asset download with the checker/training flag | NEEDS_USER_APPROVAL | Prior reports intentionally blocked hidden network clone by default. |
| Use a separate validation host where assets already exist | ACCEPTABLE | Record exact host/runtime details in reports before counting smoke results. |

## Route B Acceptance Items To Check

- `train-g1-sac` entry point exists and does not remove or alter `train-g1-jax`, `train-g1-rsl`, or `train-g1-sac-brax`.
- CLI exposes explicit `--actor_learning_rate`, `--critic_learning_rate`, `--alpha_learning_rate`, `--policy_hidden_layer_sizes`, `--q_hidden_layer_sizes`, `--policy_obs_key`, `--value_obs_key`, `--min_replay_size`, `--max_replay_size`, `--grad_updates_per_step`, `--reward_scaling`, `--discounting`, `--tau`, `--target_entropy_coef`, `--normalize_observations`, `--deterministic_eval`, `--logdir`, and `--dry_run`.
- Actor defaults to `policy_obs_key=state`; critic defaults to `value_obs_key=privileged_state`.
- Missing `privileged_state` falls back through the configured actor key in the train loop; missing `truncation` synthesizes zeros. There is no external action-scale config in Route B because G1 action scaling is internal.
- Env action scaling remains internal; no external double-scaling is introduced.
- Reports capture full traceback for every failed command.

## Review Gate

Next review gate:

- Env load remains blocked until assets are provided or download is authorized.
- CPU tiny smoke should wait for assets and a suitable validation host.
