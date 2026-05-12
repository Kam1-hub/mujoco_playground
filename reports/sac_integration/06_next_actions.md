# Next Actions

Status: updated on 2026-05-12 after WSL2 path correction and GPU operating
notes review.

## Immediate State

- Route B `train-g1-sac` is locally runnable on CPU for tiny smoke.
- Env API readiness is validated for flat and rough G1 tasks on the CPU dev
  path after menagerie is present.
- CPU tiny smoke passed with checkpoint
  `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`.
- GPU migration prep scripts are added.
- CPU-side migration prep checks pass: compileall, checkpoint inspection, and
  `gpu_preflight.py --impl jax` without `--require_gpu`.
- Target WSL2 documentation workspace:
  `/home/admin/projects/mujoco_playground/g1_sac_dev`.
- Current target WSL2 workspace still needs local `.venv`, pinned menagerie,
  and CUDA JAX verification.
- GPU smoke is still `NOT VALIDATED`.

## Preflight Checklist

Before any Python/JAX command:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Prepare dependencies and assets:

```bash
uv sync --frozen --extra cuda
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
```

If menagerie already exists, only verify its commit. Do not commit menagerie,
`.venv`, logs, or checkpoints.

Verify runtime:

```bash
nvidia-smi
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
uv run --no-sync python -c "import mujoco, brax; print('mujoco', mujoco.__version__); print('brax', brax.__version__)"
```

`nvcc` is not required in WSL2. JAX must report GPU/CUDA/ROCm. If JAX reports
CPU, stop and report the CUDA/JAX blocker.

## Recommended Next Command On GPU Machine

Run this first. It does not train:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

If preflight passes, stop and wait for user confirmation before the Route B 10k
smoke wrapper:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

## Migration Reminders

- Keep first SAC smoke defaults: `num_envs=128`, `max_replay_size=8192`,
  `batch_size=256`, `grad_updates_per_step=2`.
- Record JAX backend/devices, MuJoCo version, Brax version, idle/final VRAM when
  available, SPS, losses, alpha, checkpoint path, and any full traceback.
- Record actual `env_steps`; Route B may record `9984` for a `10000` target with
  `128` envs.
- `gpu_preflight.py --impl jax` without `--require_gpu` is not GPU validation.

## Recommended Follow-Ups After GPU 10k Smoke

1. Update `04_smoke_results.md`, `05_known_issues.md`, and this file.
2. Add deterministic eval rollout metrics.
3. Run a longer sanity run only after GPU smoke is stable and explicitly
   authorized.
4. Compare against PPO baseline only after SAC 10k smoke has a clean report.

## Do Not Start Yet

- 10k smoke without user confirmation
- 1M training
- PPO-scale `num_envs=2048` or `8192` experiments as SAC smoke substitutes
- domain randomization
- reward, `action_scale`, or Kp tuning
- world model
- fine-tuning
- vision SAC
- real deployment
