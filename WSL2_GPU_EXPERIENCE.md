# WSL2 GPU Experience For G1 SAC

Status: added on 2026-05-12 as a project-local operating reference derived
from prior WSL2 RL/JAX experiments. Use this file as GPU operations guidance;
do not use it to override the Route B SAC design or validation ladder.

## Scope

This document applies to the G1 SAC migration workspace:

```text
/home/admin/projects/mujoco_playground/g1_sac_dev
```

Current target GPU profile from the local workstation notes:

- GPU: NVIDIA GeForce RTX 4070 SUPER
- VRAM: 12 GB class
- OS: WSL2 Ubuntu 24.04
- Python management: prefer `uv` when the repo has `pyproject.toml` and
  `uv.lock`

The following are not SAC migration defaults:

- PPO or Barkour `num_envs=2048` experiments
- PPO `num_envs=8192` defaults
- Barkour reward/action tuning
- domain randomization
- fine-tuning
- reward, `action_scale`, or Kp changes

## Required Environment Variables

Set these before any Python command that imports JAX or MuJoCo:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Notes:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` prevents JAX from reserving most of
  VRAM on first use.
- `MUJOCO_GL=egl` is required for headless WSL2 rendering and must be set
  before `import mujoco`.
- The Route B GPU smoke currently passes `--render False`; do not add training
  loop rendering to diagnose GPU preflight.

## GPU Visibility Gate

`nvcc` is not required in WSL2. JAX can use GPU through the Windows driver and
CUDA plugin packages without a CUDA toolkit inside WSL.

The required checks are:

```bash
nvidia-smi
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Expected JAX result before GPU preflight:

```text
gpu
[CudaDevice(id=0)]
```

If JAX prints CPU devices, do not run training. A common cause is installing
only CPU JAX/JAXLIB. In this repo, the locked CUDA extra is:

```bash
uv sync --frozen --extra cuda
```

That installs the project-declared `jax[cuda12]` path without editing
`pyproject.toml` or `uv.lock`. If a CUDA 13 variant is required, report that
decision first because it changes the environment strategy.

## Route B SAC Memory Notes

Route B is asymmetric SAC:

- policy obs: `state`, shape `(103,)`
- value obs: `privileged_state`, shape `(216,)`
- action size: `29`
- default first GPU smoke: `num_envs=128`, `max_replay_size=8192`

Replay is the main SAC-specific VRAM risk. A Route B transition stores roughly:

```text
policy_obs 103
value_obs 216
action 29
reward/discount/done/truncation 4
next_policy_obs 103
next_value_obs 216
total ~= 671 float32 values
```

Raw storage is about 2.6 to 2.7 KB per transition:

- `8192`: about 22 MB raw
- `100000`: about 256 MB raw
- `1000000`: about 2.5 to 2.7 GB raw

Actual VRAM also includes JAX/XLA compilation, optimizer state, network
activations, env state, and temporary buffers. Do not jump to a 1M replay or
larger run before the 10k smoke is clean.

## Validation Ladder

GPU validation remains:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Only after that passes may the user authorize:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

Do not replace the Route B 10k smoke with PPO-scale parameters. The first SAC
smoke keeps:

- `num_timesteps=10000`
- `num_envs=128`
- `num_eval_envs=32`
- `batch_size=256`
- `min_replay_size=1024`
- `max_replay_size=8192`
- `grad_updates_per_step=2`

Route B currently computes `actor_steps = num_timesteps // num_envs`, so the
actual env step count can be lower than the requested target. For 10000 steps
with 128 envs, expect 9984 actual env steps. Reports must record actual
`env_steps`, not assume it equals the requested `num_timesteps`.

## Known GPU Runtime Behaviors

- First JAX/XLA compilation can take minutes. Do not classify that alone as a
  hang.
- WSL2 may print driver passthrough warnings such as kernel mode driver version
  messages. Treat them as noise unless JAX cannot see a GPU or the command
  fails.
- Use `nvidia-smi` for VRAM, temperature, and utilization snapshots. Record
  idle VRAM before smoke and final/peak VRAM when available.

## OOM Triage

Classify OOM-like failures before changing parameters:

1. JAX preallocation not disabled.
2. CPU-only JAX or missing CUDA plugin.
3. Residual GPU processes.
4. Replay capacity too large.
5. Env batch too large.
6. Terrain/domain randomization overhead.

For SAC, prefer reducing `max_replay_size` before importing PPO-style
`num_envs=2048` assumptions. Destructive cleanup such as killing GPU processes
or restarting WSL requires explicit user approval.
