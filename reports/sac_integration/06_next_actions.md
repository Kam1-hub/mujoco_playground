# Next Actions

Status: updated on 2026-05-12 after deterministic SAC checkpoint eval smoke.

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
- Target WSL2 workspace now has CUDA JAX via `uv sync --frozen --extra cuda`.
- JAX reports backend `gpu` and device `cuda:0`.
- Menagerie is present at commit
  `1b86ece576591213e2b666ebf59508454200ca97`.
- GPU preflight with `--require_gpu` passed.
- Route B GPU 10k smoke passed with `TRAIN_OK`.
- The existing GPU 10k checkpoint predates normalizer persistence and is not
  deterministic-eval ready.
- Future Route B checkpoints now persist `policy_normalizer` and
  `value_normalizer`; `scripts/check_sac_checkpoint.py --require_eval_ready`
  can enforce this gate.
- Schema dry-run checkpoint
  `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl` contains both
  normalizers and passes `--require_eval_ready`.
- New GPU smoke checkpoint
  `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` contains both
  normalizers and passes `--require_eval_ready`.
- Deterministic eval CLI `scripts/eval_sac_checkpoint.py` has a passing smoke:
  4 env x 200 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_smoke/eval_4x200.json`, no action/reward/obs NaN.

## Completed WSL2 GPU Validation

- GPU preflight: `PASS`
- Route B GPU 10k smoke: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- Requested timesteps: `10000`
- Actual env steps: `9984`
- Gradient steps: `142`
- Wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- Actor loss: `-1.9058758020401`
- Critic loss: `0.08382290601730347`
- Alpha: `0.04770537465810776`
- Truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics

Actual step note:

- Route B currently uses `num_envs * (num_timesteps // num_envs)`.
- With `num_timesteps=10000` and `num_envs=128`, this yields `9984` actual
  env steps.

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

## Recommended Next Step

Do not immediately run 1M. The next useful steps are report review plus the
following only after explicit user confirmation:

1. decide whether to commit the deterministic eval CLI and report updates.
2. run a slightly longer sanity check at a controlled scale only if needed.
3. compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

## Migration Reminders

- Keep first SAC smoke defaults: `num_envs=128`, `max_replay_size=8192`,
  `batch_size=256`, `grad_updates_per_step=2`.
- Record JAX backend/devices, MuJoCo version, Brax version, idle/final VRAM when
  available, SPS, losses, alpha, checkpoint path, and any full traceback.
- Record actual `env_steps`; Route B may record `9984` for a `10000` target with
  `128` envs.
- `gpu_preflight.py --impl jax` without `--require_gpu` is not GPU validation.

## Recommended Follow-Ups After Deterministic Eval Smoke

1. Commit the eval CLI/report update after review.
2. Run a longer sanity run only after GPU smoke and eval smoke are reviewed and
   explicitly authorized.
3. Keep deterministic eval scales bounded unless the user asks for a benchmark.
4. Compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

## Do Not Start Yet

- 1M training
- longer training without user confirmation
- PPO-scale `num_envs=2048` or `8192` experiments as SAC smoke substitutes
- domain randomization
- reward, `action_scale`, or Kp tuning
- world model
- fine-tuning
- vision SAC
- real deployment
