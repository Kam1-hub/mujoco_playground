# G1 SAC Integration Workspace

This repo is an isolated G1 locomotion workspace with existing PPO/RSL-RL
entry points plus a SAC integration track. The active SAC path is Route B: a
local asymmetric SAC baseline for `G1JoystickFlatTerrain` and
`G1JoystickRoughTerrain`.

Current status:

- Branch: `sac-integration`
- Route B CPU tiny smoke: passed on the Windows CPU dev host
- WSL2/GPU status in this workspace: Python env, menagerie, and CUDA JAX still
  need to be prepared and checked
- Next gate: `gpu_preflight.py --impl jax --require_gpu`
- 10k GPU smoke and all training require explicit user confirmation after
  preflight passes

## Entry Points

- PPO baseline: `train-g1-jax`
- RSL-RL baseline: `train-g1-rsl`
- SAC fallback: `train-g1-sac-brax`
- SAC main route: `train-g1-sac`

Route B defaults:

- actor obs: `obs["state"]`, shape `(103,)`
- critic obs: `obs["privileged_state"]`, shape `(216,)`
- action size: `29`
- actor action range: tanh-normalized `[-1, 1]`
- env action handling: G1 applies `action_scale` internally
- default env implementation for SAC: `impl="jax"`

Do not add external action scaling or tune reward, `action_scale`, or Kp during
GPU migration smoke.

## Setup

Use a Linux/WSL2 filesystem checkout for GPU validation:

```bash
cd /home/admin/projects/mujoco_playground/g1_sac_dev
uv sync
```

For GPU JAX, the project declares a CUDA extra:

```bash
uv sync --frozen --extra cuda
```

If JAX still reports CPU after that, stop and report the CUDA/JAX blocker
before changing `pyproject.toml` or `uv.lock`.

Menagerie assets are required but must stay untracked:

```bash
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
```

## GPU Preflight

Set WSL2 GPU variables before Python commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Check JAX GPU visibility:

```bash
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Then run the non-training preflight:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Only if preflight passes may the user authorize:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

Do not run 1M training, domain randomization, fine-tuning, or PPO comparison
before the Route B 10k GPU smoke has a clean report.

## Environment Schema

Both G1 tasks expose dict observations:

- `state`: `(103,)`, actor observation
- `privileged_state`: `(216,)`, critic observation
- action: `(29,)`
- control frequency: 50 Hz
- episode length: 1000 steps

Runtime G1 envs do not currently expose `state.info["truncation"]`; Route B
synthesizes zero truncation and reports `truncation_fraction`.

## Project Notes

Primary operating documents:

- `AGENTS.md`
- `WSL2_CODEX_HANDOFF.md`
- `WSL2_GPU_EXPERIENCE.md`
- `SAC_INTEGRATION_MASTER_PLAN.md`
- `reports/sac_integration/08_project_status_roadmap.md`

PPO/RSL-RL examples in older docs are preserved as baseline context, not as the
current SAC migration procedure.
