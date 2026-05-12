# Next Actions

Status: updated on 2026-05-12 after both-mode eval diagnostic.

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
- Route B GPU 250k sanity passed with `TRAIN_OK`.
- 250k checkpoint
  `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 250k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_250k/eval_16x1000.json`, and no action/reward/obs NaN.
- Route B GPU 500k sanity passed with `TRAIN_OK`.
- 500k checkpoint
  `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 500k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_500k/eval_16x1000.json`, and no action/reward/obs NaN.
- Alpha/entropy diagnostic patch is synced at
  `926a14f Add SAC alpha entropy diagnostics`.
- Both-mode eval-only diagnostic passed for 100k, 250k, and 500k checkpoints
  using `--policy_mode both`, seeds `0..4`, `num_eval_envs=16`, and
  `episode_length=1000`.
- Both-mode diagnostic result: deterministic `tanh(mean)` reward degrades
  across 100k/250k/500k, while sampled stochastic reward does not show the same
  degradation.

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

## Completed 250k Sanity Validation

- Route B GPU 250k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Checkpoint eval readiness: `PASS`
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `250000`
- Actual env steps: `249984`
- Gradient steps: `3892`
- Wall time: `120.21957968600327`
- SPS: `2079.3950590488107`
- Actor loss: `-5.524118900299072`
- Critic loss: `0.02644157037138939`
- Alpha: `0.018743595108389854`
- Alpha loss: `0.5869507789611816`
- Policy log prob: `-16.657032012939453`
- Q: `5.197851181030273`
- Target Q: `5.205532073974609`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error: none observed

Bounded deterministic eval after 250k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.279743194580078` / `1.2322009801864624` /
  `-8.849853515625` / `-3.093963384628296`
- Done fraction: `1.0`
- Wall time: `68.2435936529946`
- SPS: `234.45424168833907`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

## Completed 500k Sanity Validation

- Route B GPU 500k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Checkpoint eval readiness: `PASS`
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `500000`
- Actual env steps: `499968`
- Gradient steps: `7798`
- Wall time: `223.88994164399628`
- SPS: `2233.0971919899416`
- Actor loss: `-3.510934352874756`
- Critic loss: `0.04195608198642731`
- Alpha: `0.008012857288122177`
- Alpha loss: `0.20854677259922028`
- Policy log prob: `-12.072959899902344`
- Q: `3.348696231842041`
- Target Q: `3.3187503814697266`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure: none observed

Bounded deterministic eval after 500k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.691065788269043` / `1.1929610967636108` /
  `-9.040802955627441` / `-3.7163496017456055`
- Done fraction: `1.0`
- Wall time: `68.70198891899781`
- SPS: `232.88990976468807`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

Artifacts remain ignored under `logs`; do not commit logs, checkpoints, `.venv`,
or menagerie.

## Completed Both-Mode Eval Diagnostic

- Scope: eval-only; no training.
- Checkpoints: 100k, 250k, 500k.
- Eval command class: `scripts/eval_sac_checkpoint.py --policy_mode both`.
- Seeds: `0`, `1`, `2`, `3`, `4`.
- Deterministic reward mean aggregate:
  `100k=-4.2218`, `250k=-4.4585`, `500k=-4.8476`.
- Deterministic action abs mean:
  `0.1823 -> 0.2148 -> 0.3029`.
- Stochastic reward mean aggregate:
  `100k=-6.4616`, `250k=-6.1954`, `500k=-5.9091`.
- Stochastic log-prob mean:
  `-17.9594 -> -17.2847 -> -13.4275`.
- Interpretation: deterministic `tanh(mean)` behavior degrades while sampled
  stochastic behavior does not show the same degradation. Stochastic reward is
  still lower in absolute terms at each checkpoint.
- Full details: `reports/sac_integration/10_both_mode_eval_diagnostic.md`.

250k tooling, warning, and risk notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed again and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env, checkpoint, or eval failure was
  observed.
- Alpha dropped to about `0.0187` by 250k; Q and target Q rose to about `5.2`
  while critic loss stayed low; bounded eval reward did not improve versus
  100k. This is not a failure, but 500k should watch alpha collapse, Q drift,
  critic loss, eval NaN flags, and reward trend.

500k tooling, warning, and risk notes:

- No traceback, NaN, Inf, OOM, fatal CUDA, env load/reset/step/shape, replay,
  checkpoint, or eval failure was observed.
- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed again and remained non-fatal.
- Alpha continued down from about `0.0187` at 250k to about `0.0080` at 500k.
- Q and target Q decreased from about `5.2` to about `3.3`; critic loss stayed
  finite/low.
- Bounded eval reward mean worsened from `-4.27974` to `-4.69107`; eval max
  also worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but the alpha decline and eval degradation
  block any automatic jump to 1M.

Actual step note:

- Route B currently uses `num_envs * (num_timesteps // num_envs)`.
- With `num_timesteps=10000` and `num_envs=128`, this yields `9984` actual
  env steps.
- With `num_timesteps=100000` and `num_envs=128`, this yields `99968` actual
  env steps.
- With `num_timesteps=250000` and `num_envs=128`, this yields `249984` actual
  env steps.
- With `num_timesteps=500000` and `num_envs=128`, this yields `499968` actual
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

Do not automatically run 750k or 1M. The next useful step is an actor mean /
action distribution / reward-component diagnostic design, only after explicit
user confirmation.

Recommended diagnostic questions:

1. Inspect actor mean and std/log_std statistics by checkpoint.
2. Compare deterministic action against stochastic sampled action per dimension.
3. Inspect whether deterministic action magnitude drift explains reward loss.
4. Add reward component eval only if env metrics expose stable reward terms.
5. Keep all diagnostics eval-only or dry-run unless the user explicitly
   approves a training run.

Do not jump into 750k or 1M from this report update.

## Migration Reminders

- Keep first SAC smoke defaults: `num_envs=128`, `max_replay_size=8192`,
  `batch_size=256`, `grad_updates_per_step=2`.
- Record JAX backend/devices, MuJoCo version, Brax version, idle/final VRAM when
  available, SPS, losses, alpha, checkpoint path, and any full traceback.
- Record actual `env_steps`; Route B may record `9984` for a `10000` target with
  `128` envs.
- `gpu_preflight.py --impl jax` without `--require_gpu` is not GPU validation.

## Recommended Follow-Ups After Deterministic Eval Smoke

1. Commit the both-mode eval diagnostic report update after review.
2. Draft actor mean / action distribution / reward-component diagnostics for
   user review.
3. Keep eval scales bounded unless the user asks for a benchmark.
4. Do not start 750k or 1M automatically from this report update.
5. Compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

## Do Not Start Yet

- 750k or 1M training without separate user confirmation and a
  resource/stop-condition plan
- PPO-scale `num_envs=2048` or `8192` experiments as SAC smoke substitutes
- domain randomization
- reward, `action_scale`, or Kp tuning
- world model
- fine-tuning
- vision SAC
- real deployment
