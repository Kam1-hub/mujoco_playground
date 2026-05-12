# Next Actions

Status: updated on 2026-05-12 after Route B GPU 100k sanity and bounded eval.

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
- Route B GPU 50k sanity passed with `TRAIN_OK`.
- 50k checkpoint
  `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 50k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_50k/eval_16x1000.json`, and no action/reward/obs NaN.
- Route B GPU 100k sanity passed with `TRAIN_OK`.
- 100k checkpoint
  `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 100k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_100k/eval_16x1000.json`, and no action/reward/obs NaN.

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

## Completed 50k Sanity Validation

- Route B GPU 50k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Checkpoint eval readiness: `PASS`
- Requested timesteps: `50000`
- Actual env steps: `49920`
- Gradient steps: `766`
- Wall time: `35.99766752999858`
- SPS: `1386.756515777009`
- Actor loss: `-3.6135072708129883`
- Critic loss: `0.07097882032394409`
- Alpha: `0.03992176800966263`
- Alpha loss: `1.327394962310791`
- Policy log prob: `-18.81831169128418`
- Policy Q: `2.8622469902038574`
- Q: `2.884032726287842`
- Target Q: `2.899707317352295`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error: none observed

Bounded deterministic eval after 50k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.5016322135925293` / `0.75983726978302` /
  `-6.096090316772461` / `-2.531925916671753`
- Done fraction: `1.0`
- Wall time: `74.76505397899746`
- SPS: `214.00372431342888`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

## Completed 100k Sanity Validation

- Route B GPU 100k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Checkpoint eval readiness: `PASS`
- Requested timesteps: `100000`
- Actual env steps: `99968`
- Gradient steps: `1548`
- Wall time: `56.80434615799459`
- SPS: `1759.8653406193746`
- Actor loss: `-4.994826316833496`
- Critic loss: `0.04054964333772659`
- Alpha: `0.03259027376770973`
- Alpha loss: `1.0562278032302856`
- Policy log prob: `-17.77903938293457`
- Q: `4.3698601722717285`
- Target Q: `4.456111907958984`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/checkpoint/eval error: none observed

Bounded deterministic eval after 100k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.894726037979126` / `0.9610732197761536` /
  `-7.2897186279296875` / `-2.889821767807007`
- Done fraction: `1.0`
- Wall time: `66.85148939098872`
- SPS: `239.33647770242092`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

Artifacts remain ignored under `logs`; do not commit logs, checkpoints, `.venv`,
or menagerie.

100k tooling and warning notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed again and remained non-fatal.
- The first sandboxed `uv` attempt failed before training started due a
  `snap-confine` capability issue. The identical 100k command then succeeded
  with external permission and unchanged parameters, so this is a tooling note,
  not a training failure.

Actual step note:

- Route B currently uses `num_envs * (num_timesteps // num_envs)`.
- With `num_timesteps=10000` and `num_envs=128`, this yields `9984` actual
  env steps.
- With `num_timesteps=100000` and `num_envs=128`, this yields `99968` actual
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

Do not automatically run 1M. The next useful steps are report review plus the
following only after explicit user confirmation:

1. commit the 100k sanity report update after review.
2. consider 1M only with an explicit resource budget, fresh logdir, checkpoint
   readiness gate, bounded eval command, and stop-condition plan.
3. compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

Do not jump into 1M without a separate user approval and run plan.

## Migration Reminders

- Keep first SAC smoke defaults: `num_envs=128`, `max_replay_size=8192`,
  `batch_size=256`, `grad_updates_per_step=2`.
- Record JAX backend/devices, MuJoCo version, Brax version, idle/final VRAM when
  available, SPS, losses, alpha, checkpoint path, and any full traceback.
- Record actual `env_steps`; Route B may record `9984` for a `10000` target with
  `128` envs.
- `gpu_preflight.py --impl jax` without `--require_gpu` is not GPU validation.

## Recommended Follow-Ups After Deterministic Eval Smoke

1. Commit the 100k sanity report update after review.
2. Consider 1M only after a separate resource/stop-condition plan is reviewed
   and explicitly authorized.
3. Keep deterministic eval scales bounded unless the user asks for a benchmark.
4. Do not start 1M automatically from this report update.
5. Compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

## Do Not Start Yet

- 1M training without separate user confirmation and a resource/stop-condition
  plan
- PPO-scale `num_envs=2048` or `8192` experiments as SAC smoke substitutes
- domain randomization
- reward, `action_scale`, or Kp tuning
- world model
- fine-tuning
- vision SAC
- real deployment
