# Next Agent Handoff

This file is a self-contained handoff for continuing SAC Route B work after a
context reset.

## Project Goal

Add and validate a local SAC baseline for the G1 lift/joystick task while
preserving existing PPO/RSL workflows. Current work is validation-driven: move
from small smoke tests to larger sanity runs only after each previous gate is
clean.

Do not use PPO/Barkour/RSL experience to override the current SAC design,
Route B parameters, or smoke ladder. Do not tune reward, action scaling, Kp,
domain randomization, or fine-tuning as part of the current validation phase.

## Repository

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Current committed baseline before 100k report update:
  `31cc105 Add SAC phase summary and handoff`
- Remote: `origin https://github.com/Kam1-hub/mujoco_playground.git`
- Last known pushed branch: `sac-integration`

Start every new session with read-only checks:

```bash
pwd
git status --short --branch
git log --oneline -5
git remote -v
```

## Environment

Known working target:

- OS: WSL2 Ubuntu
- GPU: NVIDIA RTX 4070 SUPER, 12GB VRAM
- Python: 3.12.3 in the project uv environment
- JAX: 0.10.0
- JAX backend/device: `gpu`, `cuda:0`
- MuJoCo: 3.8.0
- Brax: 0.14.2
- `nvidia-smi`: available
- `nvcc`: missing in the checked environment, not a blocker for this stack

Required shell variables before GPU/JAX commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

`nvidia-smi` missing is a blocker. `nvcc` missing is not a blocker when JAX can
see `cuda:0`.

## External Menagerie

The project expects a local ignored checkout:

```text
g1_env/external_deps/mujoco_menagerie
```

Required commit:

```text
1b86ece576591213e2b666ebf59508454200ca97
```

Check it with:

```bash
git -C g1_env/external_deps/mujoco_menagerie rev-parse HEAD
```

Do not commit this checkout.

## Code Structure

SAC Route B code lives in local files and should not disturb PPO/RSL:

- `learning/train_jax_sac_lift.py`: CLI entry point for Route B SAC.
- `learning/sac_lift/train.py`: SAC training loop, replay insertion, updates,
  checkpoint payload.
- `learning/sac_lift/losses.py`: critic, actor, and temperature losses.
- `learning/sac_lift/networks.py`: Flax actor and twin-Q networks.
- `learning/sac_lift/replay_buffer.py`: uniform replay buffer.
- `learning/sac_lift/normalizer.py`: running observation stats.
- `learning/sac_lift/checkpoint.py`: pickle save/load helpers.
- `learning/sac_lift/evaluator.py`: helper evaluator logic.
- `scripts/gpu_preflight.py`: non-training CUDA/JAX/env/menagerie preflight.
- `scripts/check_sac_checkpoint.py`: checkpoint schema and eval readiness
  checker.
- `scripts/eval_sac_checkpoint.py`: bounded deterministic checkpoint eval.
- `g1_env/config/sac_params.py`: SAC Route B config defaults.

## Entry Points

Common commands:

```bash
uv sync --frozen --extra cuda

uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu

uv run --no-sync python -m learning.train_jax_sac_lift --help

uv run --no-sync python scripts/check_sac_checkpoint.py --help

uv run --no-sync python scripts/eval_sac_checkpoint.py --help
```

Do not run training, eval, preflight, install, or download commands unless the
user explicitly approves that phase.

## SAC Route B Design

- Environment: `G1JoystickFlatTerrain`
- Implementation: JAX/MJX via `impl=jax`
- Actor observation key: `state`
- Critic observation key: `privileged_state`
- Action size: 29
- Action scaling remains inside the environment. Do not add external action
  scaling in SAC.
- Actor: tanh-Gaussian policy.
- Critic: twin Q networks with target Q parameters.
- Entropy: learned `log_alpha`.
- Replay: local uniform replay buffer.
- Normalization: policy and value observation normalizers are saved in future
  checkpoints.
- Truncation: synthesized as zero when the env info does not provide it.
- Checkpoint schema now includes at least:
  `config`, `policy_params`, `q_params`, `target_q_params`, `log_alpha`,
  `policy_normalizer`, `value_normalizer`, and `metrics`.

## Passed Validation

Current validated ladder:

- CPU tiny smoke: PASS.
- GPU preflight: PASS.
- GPU 10k smoke: PASS.
- 10k checkpoint eval readiness with normalizer-ready checkpoint: PASS.
- Deterministic eval smoke: PASS.
- GPU 50k sanity: PASS.
- 50k checkpoint eval readiness: PASS.
- 50k bounded deterministic eval: PASS.
- GPU 100k sanity: PASS.
- 100k checkpoint eval readiness: PASS.
- 100k bounded deterministic eval: PASS.

Still not validated:

- 1M training.
- Full performance benchmark.
- PPO comparison.
- Domain randomization.
- Fine-tuning.
- Reward/action scale/Kp tuning.

## Artifacts and Logdirs

All paths below are runtime artifacts and should remain ignored:

- `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
  - First GPU 10k smoke checkpoint.
  - Old schema, missing normalizers, not trusted for deterministic eval.
- `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
  - New schema 10k checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_smoke/eval_4x200.json`
  - 4 env x 200 step deterministic eval smoke result.
- `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
  - 50k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_50k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result.
- `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
  - 100k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_100k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result after 100k.
- `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl`
  - Dry-run schema validation artifact, if still present.

Never commit:

- `logs/`
- checkpoints
- `.venv/`
- `g1_env/external_deps/mujoco_menagerie`
- generated caches

Check ignored status with:

```bash
git check-ignore -v logs .venv g1_env/external_deps/mujoco_menagerie || true
```

## Key Results

GPU 10k smoke:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- `env_steps=9984`, `gradient_steps=142`
- `wall_time=56.50599093900382`, `sps=176.68922947970893`
- No NaN/Inf/OOM/CUDA error observed.

10k deterministic eval smoke:

- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- `eval_env_steps=800`
- `episode_reward_mean=-3.3628087043762207`
- `done_fraction=1.0`
- No action/reward/obs NaN.

GPU 50k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- `env_steps=49920`, `gradient_steps=766`
- `wall_time=35.99766752999858`, `sps=1386.756515777009`
- `actor_loss=-3.6135072708129883`
- `critic_loss=0.07097882032394409`
- `alpha=0.03992176800966263`
- `policy_q=2.8622469902038574`
- `target_q=2.899707317352295`
- No NaN/Inf/OOM/CUDA/checkpoint/eval error observed.

50k bounded deterministic eval:

- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-3.5016322135925293`
- `episode_reward_std=0.75983726978302`
- `episode_reward_min=-6.096090316772461`
- `episode_reward_max=-2.531925916671753`
- `done_fraction=1.0`
- `wall_time=74.76505397899746`, `sps=214.00372431342888`
- No action/reward/obs NaN.

GPU 100k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- `env_steps=99968`, `gradient_steps=1548`
- `wall_time=56.80434615799459`, `sps=1759.8653406193746`
- `actor_loss=-4.994826316833496`
- `critic_loss=0.04054964333772659`
- `alpha=0.03259027376770973`
- `q=4.3698601722717285`
- `target_q=4.456111907958984`
- No NaN/Inf/OOM/fatal CUDA/checkpoint/eval error observed.

100k bounded deterministic eval:

- JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-3.894726037979126`
- `episode_reward_std=0.9610732197761536`
- `episode_reward_min=-7.2897186279296875`
- `episode_reward_max=-2.889821767807007`
- `done_fraction=1.0`
- `wall_time=66.85148939098872`, `sps=239.33647770242092`
- No action/reward/obs NaN.

100k tooling notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- A first sandboxed `uv` attempt hit a `snap-confine` capability issue before
  training started. The identical command succeeded with external permission and
  unchanged parameters, so this is tooling noise, not a training failure.

## Remaining Risks

- 1M stability is unknown.
- Eval rewards are low and only prove bounded eval execution, not policy quality.
- Truncation handling is still an assumption when absent from env info.
- Replay memory and normalizer behavior need 1M-scale validation.
- 1M replay can be around 2.5-2.7 GB raw before overhead.
- SPS can vary due JIT compile and warmup.
- No PPO comparison has been run.

## Recommended Next Step

1. Do read-only status checks.
2. Read this file and `reports/sac_integration/09_phase_summary_and_risks.md`.
3. If the user approves continuing, plan 1M explicitly with a resource budget,
   fresh logdir, checkpoint readiness gate, bounded eval command, and stop
   conditions.

Do not start 1M automatically. Do not modify reward, action scale, Kp, domain
randomization, fine-tuning, PPO, or RSL.

## Completed 100k Sanity Command

The 100k sanity command that passed was:

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

The readiness and bounded eval commands that passed were:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --require_eval_ready

uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --num_eval_envs 16 \
  --episode_length 1000 \
  --render False \
  --output_json ./logs/sac_eval_100k/eval_16x1000.json
```

## Suggested New Agent Prompt

```text
You are in /home/admin/projects/mujoco_playground/g1_sac_dev.

First read NEXT_AGENT_HANDOFF.md and
reports/sac_integration/09_phase_summary_and_risks.md. Then run only read-only
status checks: pwd, git status --short --branch, git log --oneline -5,
git remote -v, and git check-ignore -v logs .venv
g1_env/external_deps/mujoco_menagerie || true.

Current HEAD should be at least 31cc105 Add SAC phase summary and handoff unless
newer report commits exist. GPU 10k smoke, deterministic eval smoke, GPU 50k
sanity/eval, and GPU 100k sanity/eval have passed. 1M is not validated.

Do not run training, eval, preflight, installs, downloads, or git commits unless
explicitly asked. Do not start 1M without a separate resource/stop-condition
plan and user confirmation. Do not change reward, action_scale, Kp, domain
randomization, fine-tuning, PPO, or RSL.
```
