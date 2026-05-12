# Phase Summary and Risks

Date: 2026-05-12

This report freezes the current SAC Route B validation state so a new agent can
continue without relying on chat history.

## Repository State

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Current committed diagnostic baseline before this report update:
  `926a14f Add SAC alpha entropy diagnostics`
- Remote: `origin https://github.com/Kam1-hub/mujoco_playground.git`
- External menagerie commit: `1b86ece576591213e2b666ebf59508454200ca97`

Runtime artifacts are local and ignored. Do not commit `logs/`, `.venv/`,
`g1_env/external_deps/mujoco_menagerie`, or generated checkpoints.

## Validation Ladder

| Level | Status | Evidence |
| --- | --- | --- |
| CPU tiny smoke | PASS | Earlier local smoke report |
| WSL2 CUDA/JAX preflight | PASS | `scripts/gpu_preflight.py --impl jax --require_gpu` |
| Route B GPU 10k smoke | PASS | `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl` |
| Future checkpoint schema with normalizers | PASS | `c59eda0 Save SAC normalizers in checkpoints` |
| Normalizer-ready GPU 10k checkpoint | PASS | `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` |
| Deterministic eval smoke | PASS | `./logs/sac_eval_smoke/eval_4x200.json` |
| Route B GPU 50k sanity | PASS | `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` |
| 50k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 50k bounded deterministic eval | PASS | `./logs/sac_eval_50k/eval_16x1000.json` |
| 100k sanity | PASS | `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` |
| 100k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 100k bounded deterministic eval | PASS | `./logs/sac_eval_100k/eval_16x1000.json` |
| 250k sanity | PASS | `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` |
| 250k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 250k bounded deterministic eval | PASS | `./logs/sac_eval_250k/eval_16x1000.json` |
| 500k sanity | PASS | `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` |
| 500k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 500k bounded deterministic eval | PASS | `./logs/sac_eval_500k/eval_16x1000.json` |
| Both-mode eval diagnostic | PASS | `reports/sac_integration/10_both_mode_eval_diagnostic.md` |
| 1M training | NOT VALIDATED | Requires explicit user confirmation and resource/stop plan |

## Completed Outcomes

- Added a local SAC Route B implementation without modifying PPO/RSL code paths.
- Validated WSL2 CUDA/JAX operation on an RTX 4070 SUPER 12GB setup.
- Pinned the external MuJoCo Menagerie checkout and kept it untracked.
- Verified SAC training can produce checkpoints on GPU at 10k, 50k, 100k,
  250k, and 500k scales.
- Fixed future SAC checkpoint schema to save `policy_normalizer` and
  `value_normalizer`.
- Added checkpoint readiness checks for deterministic eval.
- Added a bounded deterministic SAC checkpoint eval script.
- Verified deterministic eval readiness and bounded eval on the normalizer-ready
  10k and 50k checkpoints.
- Verified 100k checkpoint readiness and bounded deterministic eval.
- Verified 250k checkpoint readiness and bounded deterministic eval.
- Verified 500k checkpoint readiness and bounded deterministic eval.
- Synced alpha/entropy diagnostics and validated `--policy_mode both` eval-only
  diagnostic on 100k, 250k, and 500k checkpoints.

## Key Metrics

### GPU 10k Smoke

- Command family: Route B SAC, `num_timesteps=10000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=8192`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- `env_steps`: `9984`
- `gradient_steps`: `142`
- `wall_time`: `56.50599093900382`
- `sps`: `176.68922947970893`
- `actor_loss`: `-1.9058758020401`
- `critic_loss`: `0.08382290601730347`
- `alpha`: `0.04770537465810776`
- `alpha_loss`: `1.585930585861206`
- `policy_log_prob`: `-18.79882049560547`
- `policy_q`: `1.0090709924697876`
- `q`: `1.0673823356628418`
- `target_q`: `1.0646085739135742`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/CUDA error observed: no

Note: the first 10k checkpoint predates the normalizer schema fix and is not a
trusted deterministic eval input.

### 10k Deterministic Eval Smoke

- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- Eval JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- `num_eval_envs`: `4`
- `episode_length`: `200`
- `eval_env_steps`: `800`
- `episode_reward_mean`: `-3.3628087043762207`
- `episode_reward_std`: `0.34718504548072815`
- `episode_reward_min`: `-3.778578281402588`
- `episode_reward_max`: `-2.876215934753418`
- `done_fraction`: `1.0`
- `wall_time`: `62.32472045900067`
- `sps`: `12.835998205981001`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 50k Sanity

- Command family: Route B SAC, `num_timesteps=50000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=50000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- `env_steps`: `49920`
- `gradient_steps`: `766`
- `wall_time`: `35.99766752999858`
- `sps`: `1386.756515777009`
- `actor_loss`: `-3.6135072708129883`
- `critic_loss`: `0.07097882032394409`
- `alpha`: `0.03992176800966263`
- `alpha_loss`: `1.327394962310791`
- `policy_log_prob`: `-18.81831169128418`
- `policy_q`: `2.8622469902038574`
- `q`: `2.884032726287842`
- `target_q`: `2.899707317352295`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error observed: no

The actual step count is `49920` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 50k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Eval JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-3.5016322135925293`
- `episode_reward_std`: `0.75983726978302`
- `episode_reward_min`: `-6.096090316772461`
- `episode_reward_max`: `-2.531925916671753`
- `done_fraction`: `1.0`
- `wall_time`: `74.76505397899746`
- `sps`: `214.00372431342888`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 100k Sanity

- Command family: Route B SAC, `num_timesteps=100000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=100000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS
- `env_steps`: `99968`
- `gradient_steps`: `1548`
- `wall_time`: `56.80434615799459`
- `sps`: `1759.8653406193746`
- `actor_loss`: `-4.994826316833496`
- `critic_loss`: `0.04054964333772659`
- `alpha`: `0.03259027376770973`
- `alpha_loss`: `1.0562278032302856`
- `policy_log_prob`: `-17.77903938293457`
- `q`: `4.3698601722717285`
- `target_q`: `4.456111907958984`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/checkpoint/eval error observed: no

The actual step count is `99968` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 100k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Eval JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-3.894726037979126`
- `episode_reward_std`: `0.9610732197761536`
- `episode_reward_min`: `-7.2897186279296875`
- `episode_reward_max`: `-2.889821767807007`
- `done_fraction`: `1.0`
- `wall_time`: `66.85148939098872`
- `sps`: `239.33647770242092`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 250k Sanity

- Command family: Route B SAC, `num_timesteps=250000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=250000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `120.21957968600327`
- `sps`: `2079.3950590488107`
- `actor_loss`: `-5.524118900299072`
- `critic_loss`: `0.02644157037138939`
- `alpha`: `0.018743595108389854`
- `alpha_loss`: `0.5869507789611816`
- `policy_log_prob`: `-16.657032012939453`
- `q`: `5.197851181030273`
- `target_q`: `5.205532073974609`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error observed: no

The actual step count is `249984` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 250k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Eval JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-4.279743194580078`
- `episode_reward_std`: `1.2322009801864624`
- `episode_reward_min`: `-8.849853515625`
- `episode_reward_max`: `-3.093963384628296`
- `done_fraction`: `1.0`
- `wall_time`: `68.2435936529946`
- `sps`: `234.45424168833907`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 500k Sanity

- Command family: Route B SAC, `num_timesteps=500000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=500000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- `env_steps`: `499968`
- `gradient_steps`: `7798`
- `wall_time`: `223.88994164399628`
- `sps`: `2233.0971919899416`
- `actor_loss`: `-3.510934352874756`
- `critic_loss`: `0.04195608198642731`
- `alpha`: `0.008012857288122177`
- `alpha_loss`: `0.20854677259922028`
- `policy_log_prob`: `-12.072959899902344`
- `q`: `3.348696231842041`
- `target_q`: `3.3187503814697266`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure observed: no

The actual step count is `499968` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 500k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Eval JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-4.691065788269043`
- `episode_reward_std`: `1.1929610967636108`
- `episode_reward_min`: `-9.040802955627441`
- `episode_reward_max`: `-3.7163496017456055`
- `done_fraction`: `1.0`
- `wall_time`: `68.70198891899781`
- `sps`: `232.88990976468807`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### Both-Mode Eval Diagnostic

- Scope: eval-only, no training.
- Script: `scripts/eval_sac_checkpoint.py --policy_mode both`
- Checkpoints: 100k, 250k, 500k.
- Seeds: `0..4`.
- Eval scale: `num_eval_envs=16`, `episode_length=1000`.
- Checkpoint readiness: PASS for all three checkpoints.
- Deterministic reward mean aggregate:
  - 100k: `-4.2218`
  - 250k: `-4.4585`
  - 500k: `-4.8476`
- Deterministic action abs mean:
  `0.1823 -> 0.2148 -> 0.3029`
- Stochastic reward mean aggregate:
  - 100k: `-6.4616`
  - 250k: `-6.1954`
  - 500k: `-5.9091`
- Stochastic log-prob mean:
  `-17.9594 -> -17.2847 -> -13.4275`
- No eval failure, traceback, OOM, CUDA fatal error, or action/reward/obs NaN
  was observed.
- Interpretation: deterministic `tanh(mean)` behavior degrades, while sampled
  stochastic behavior does not show the same degradation.

### 250k Risk Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env, checkpoint, or eval failure was
  observed.
- Alpha dropped to about `0.0187`; Q and target Q rose to about `5.2` while
  critic loss stayed low; bounded eval reward did not improve versus 100k.
  This is not a failure, but a future 500k run should watch alpha collapse,
  Q drift, critic loss, eval NaN flags, and reward trend.

### 500k Risk Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env load/reset/step/shape, replay,
  checkpoint, or eval failure was observed.
- Alpha continued down from about `0.0187` at 250k to about `0.0080` at 500k.
- Q and target Q decreased from about `5.2` at 250k to about `3.3` at 500k;
  critic loss stayed finite/low.
- Bounded eval reward mean worsened from `-4.27974` at 250k to `-4.69107` at
  500k. Eval max also worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but alpha decline and eval degradation must
  block any automatic jump to 1M.

### 100k Warning Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- The first sandboxed `uv` attempt hit a `snap-confine` capability issue before
  training started. The identical command succeeded with external permission and
  unchanged parameters, so this is tooling noise, not a training failure.

## Current Claims

It is reasonable to claim:

- SAC Route B imports, initializes, trains briefly, checkpoints, reloads, and
  runs bounded deterministic eval on the target WSL2 CUDA/JAX stack.
- The current checkpoint schema is sufficient for deterministic actor eval when
  `normalize_observations=True`.
- The 10k, 50k, 100k, 250k, and 500k GPU runs did not show NaN, Inf, OOM,
  fatal CUDA failure, env failure, checkpoint failure, or eval failure.
- Both-mode eval shows the 500k quality concern is concentrated in the
  deterministic `tanh(mean)` path; sampled stochastic eval does not show the
  same degradation.
- Runtime artifacts are ignored and have not been committed.

It is not yet reasonable to claim:

- 1M training stability.
- Any final policy quality or solved task performance.
- Tuned rewards, tuned action scale, tuned stiffness/damping, or optimized SAC
  hyperparameters.
- PPO/RSL parity or superiority.
- Robustness under domain randomization, fine-tuning, deployment, or long
  rollout evaluation.

## Remaining Risks

- SAC algorithm maturity: Route B has the core SAC pieces, but 1M-scale
  training has not been validated.
- Long training stability: 1M is still untested, so late NaN, replay drift,
  alpha instability, or target-Q drift remain possible.
- 500k exposed training-dynamics risk: alpha reached about `0.0080` and
  deterministic eval reward worsened versus 250k despite no runtime failure.
- Both-mode eval narrowed the quality issue: deterministic `tanh(mean)` reward
  degrades while sampled stochastic reward improves slightly, so the next risk
  area is actor mean / action distribution behavior.
- Eval reward is still low and should be treated as a smoke signal, not a
  performance benchmark.
- Truncation handling is currently synthesized as zero when absent. That passed
  the tested ladder but is still a modeling assumption to watch in longer runs.
- Replay, normalizer, and checkpoint interactions are now covered by schema
  checks through 500k, but not at 1M scale.
- Performance/SPS varies strongly because the first 10k run paid more compile
  and warmup cost. Use same-machine comparisons only.
- No PPO comparison has been run for the same conditions.

## Completed 100k Plan

The 100k sanity run used the 50k parameter shape and increased the horizon
conservatively:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_sanity
```

Observed checkpoint:

```text
./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl
```

Actual env steps:

```text
128 * (100000 // 128) = 99968
```

Replay memory estimate: around 256 MB raw sample storage for 100k entries, plus
JAX/device overhead and optimizer/network state. This fit the known 12GB GPU
budget during the recorded run.

Checkpoint readiness command:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --require_eval_ready
```

Bounded eval command:

```bash
uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --num_eval_envs 16 \
  --episode_length 1000 \
  --render False \
  --output_json ./logs/sac_eval_100k/eval_16x1000.json
```

Stop immediately and report if any of these occur:

- JAX backend is not GPU/CUDA.
- `nvidia-smi` is unavailable.
- Training does not print `TRAIN_OK`.
- Any NaN, Inf, OOM, CUDA, env load, obs/action shape, replay-update,
  checkpoint, or eval failure appears.
- Checkpoint readiness fails.
- Deterministic eval returns action/reward/obs NaN.
- Git status shows unignored logs, checkpoints, `.venv`, or menagerie files.

## 1M Decision And Readiness Plan

Do not automatically jump to 750k or 1M from this report update. The next
recommended step is actor mean / action distribution / reward-component
diagnostic design and user confirmation. A later 1M review should consider
whether alpha floor, target entropy, log-alpha dynamics, or deterministic mean
action drift need analysis before a longer run. Consider 1M only with:

- explicit resource budget
- fresh logdir and checkpoint path
- checkpoint readiness gate
- bounded deterministic eval command
- stop conditions for NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval failures
- no logs/checkpoints/.venv/menagerie committed
- clean ignored-artifact audit

The 1M replay buffer can require roughly 2.5-2.7 GB raw storage before overhead,
so memory pressure, checkpoint size, compile behavior, and replay update cost
must be budgeted explicitly. A 1M run should have its own user-approved command,
fresh logdir, post-run checkpoint check, bounded eval, and report update.

## Standing Prohibitions

- Do not do domain randomization.
- Do not do fine-tuning.
- Do not tune reward, `action_scale`, Kp, or other control gains.
- Do not modify PPO/RSL paths for SAC validation.
- Do not commit `logs/`, checkpoints, `.venv/`, or menagerie.
- Do not treat WSL2/JAX warnings as blockers unless the command fails.
