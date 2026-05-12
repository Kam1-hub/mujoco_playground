# G1 SAC Usage

Current workspace:

```bash
cd /home/admin/projects/mujoco_playground/g1_sac_dev
```

This guide is scoped to the SAC migration state. PPO and RSL-RL entry points
still exist, but they are not the current GPU validation target.

## Install

CPU/dev dependency sync:

```bash
uv sync
```

CUDA JAX sync using the repo-declared extra:

```bash
uv sync --frozen --extra cuda
```

Do not edit `pyproject.toml` or `uv.lock` just because JAX reports CPU. First
record the exact JAX output and decide whether the locked CUDA extra or a
separate CUDA variant is required.

## Required Assets

The G1 XMLs depend on DeepMind menagerie assets:

```bash
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
```

Do not commit `g1_env/external_deps/mujoco_menagerie`.

## Non-Training Checks

```bash
uv run --no-sync python -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

The preflight checks imports, JAX GPU visibility, menagerie commit, registry,
Route B import, and flat/rough env reset/step. It does not train.

## WSL2 GPU Environment

Set these before Python commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

`nvcc` is not required in WSL2. A passing GPU setup requires `nvidia-smi` and
JAX devices to show GPU/CUDA/ROCm:

```bash
nvidia-smi
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

If JAX still reports CPU, do not run smoke or training.

## Route B SAC Smoke Ladder

The first allowed GPU gate is:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Only after that passes may the user authorize:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

That wrapper uses the fixed 10k smoke parameters:

- `G1JoystickFlatTerrain`
- `num_timesteps=10000`
- `num_envs=128`
- `num_eval_envs=32`
- `batch_size=256`
- `min_replay_size=1024`
- `max_replay_size=8192`
- `grad_updates_per_step=2`
- `logdir=./logs/sac_lift_gpu_10k`

Do not run 1M training, domain randomization, fine-tuning, reward tuning,
`action_scale` changes, Kp changes, or PPO comparison before the 10k SAC smoke
has a clean report.

## Route B Parameters

Important CLI flags:

- `--env_name`: `G1JoystickFlatTerrain` or `G1JoystickRoughTerrain`
- `--impl`: default `jax` for SAC validation
- `--policy_obs_key`: default `state`
- `--value_obs_key`: default `privileged_state`
- `--num_envs`: first smoke keeps `128`
- `--max_replay_size`: first smoke keeps `8192`
- `--grad_updates_per_step`: update count per vectorized env step

Route B actual env steps are `num_envs * (num_timesteps // num_envs)`. A 10k
target with 128 envs records 9984 actual env steps.

## Environment Schema

Both G1 tasks expose:

- actor obs `state`: `(103,)`
- critic obs `privileged_state`: `(216,)`
- action: `(29,)`
- `state.info["truncation"]`: absent in raw runtime checks

Route B synthesizes zero truncation when missing and reports
`truncation_fraction`.

## Historical Baselines

`train-g1-jax` and `train-g1-rsl` are kept as PPO/RSL baselines. Their large
parallel-env examples are not SAC smoke parameters and should not be copied into
the current GPU migration step.
