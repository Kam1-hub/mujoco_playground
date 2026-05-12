# G1 SAC Integration Status And Roadmap

Status: updated on 2026-05-12 after Route B GPU 50k sanity and bounded eval.

## 1. Mission

The project goal is to add a SAC baseline to the G1 pipeline without breaking the existing PPO and RSL-RL paths.

Target pipeline:

```text
G1 pipeline
├── PPO baseline: train-g1-jax
├── RSL-RL baseline: train-g1-rsl
├── SAC fallback/minimal baseline: train-g1-sac-brax
└── SAC main baseline: train-g1-sac
```

The primary route is Route B: a local asymmetric SAC implementation for G1.

Route B defaults:

- Actor observation: `obs["state"]`
- Critic observation: `obs["privileged_state"]`
- Policy obs shape: `(103,)`
- Value obs shape: `(216,)`
- Action size: `29`
- Actor distribution: tanh Gaussian
- Critic: twin Q with target Q
- Entropy: trainable alpha
- Replay: uniform replay buffer
- Checkpoint: pickle checkpoint for smoke validation
- Checkpoint schema: policy/Q/target/log-alpha, metrics, config, and
  policy/value observation normalizers
- Default env implementation: `impl="jax"`

Action scaling decision:

- The actor outputs tanh-normalized actions.
- Actions are passed directly to the G1 env.
- The G1 env applies `action_scale` internally.
- Do not add external action scaling unless the SAC objective is deliberately redesigned and documented.

Truncation decision:

- Runtime G1 env did not expose `state.info["truncation"]`.
- Route B synthesizes zero truncation when missing.
- The training metrics report `truncation_fraction`.

## 2. Hard Boundaries

Do not:

- Break or alter existing PPO/RSL-RL behavior.
- Wholesale copy LIFT forks.
- Modify `D:\mujoco_playground\template`.
- Modify `D:\mujoco_playground\LIFT-humanoid`.
- Commit menagerie assets.
- Commit `.venv`, `logs`, checkpoints, or generated caches.
- Run 1M or long training before 10k GPU smoke passes.
- Enable domain randomization for the first GPU smoke.
- Start world model, fine-tuning, vision SAC, or real deployment.
- Tune reward, `action_scale`, or Kp during migration smoke.
- Import PPO/Barkour `2048` or `8192` env assumptions into the first SAC smoke.

Failure handling rule:

- Preserve full command, stdout/stderr, traceback, failure category, diagnosis, attempted fix, and next concrete action.

Failure categories:

- dependency
- CUDA/JAX backend
- menagerie
- env load
- obs/action shape
- replay/update
- checkpoint

## 3. Route Map

### Phase 0: Workspace Manifest

Goal:

- Establish project paths, runtime, dependency visibility, git branch/commit, and validation boundaries.

Status:

- Completed.
- Reports exist under `reports/sac_integration/`.

### Phase 1: G1 Env API Readiness

Goal:

- Validate registry, config, env load, reset, step, obs schema, action size, truncation info, and domain randomizer presence.

Status:

- Completed on CPU dev with `impl="jax"`.
- Flat and rough env load/reset/step pass.
- Runtime schema:
  - obs type: dict
  - obs keys: `state`, `privileged_state`
  - `state`: `(103,)`
  - `privileged_state`: `(216,)`
  - action size: `29`
  - truncation key: absent

### Phase 2: Route A Brax SAC Fallback

Goal:

- Provide a minimal fallback SAC route using upstream Brax SAC where possible.

Status:

- Entry point exists: `train-g1-sac-brax`.
- Help command passes.
- Full runtime smoke is not the current priority.
- Route B remains the main route.

### Phase 3: Route B Asymmetric SAC Main Route

Goal:

- Implement a local LIFT-style asymmetric SAC baseline:
  - actor uses `state`
  - critic uses `privileged_state`
  - tanh Gaussian actor
  - twin Q
  - target Q
  - Polyak update
  - trainable alpha
  - replay buffer
  - obs normalization
  - dry run
  - checkpoint

Status:

- Implemented.
- Entry point exists: `train-g1-sac`.
- CLI supports required SAC parameters.
- Route B help passes.
- Dry run passes.
- CPU tiny smoke passes.

### Phase 4: Clean-Room Fallback

Goal:

- Provide a fallback if Route B gets blocked.

Status:

- Not activated.
- Route B is runnable on CPU tiny smoke, so Route C is not needed yet.

### Phase 5: Validation Ladder

Validation ladder:

1. Static compile
2. Registry import
3. Flat env API
4. Rough env API
5. Route A help
6. Route B help
7. Route B dry run
8. CPU tiny smoke
9. GPU preflight
10. GPU 10k smoke
11. 50k sanity run
12. 100k or longer sanity run

Status:

- Steps 1 through 11 are complete.
- Step 12 remains `NOT VALIDATED` and requires separate user confirmation.

### Phase 6: Reports, Commits, Migration Handoff

Goal:

- Keep complete reports, commit verified phases, and prepare a clean WSL2/GPU continuation path.

Status:

- Completed for CPU dev and GPU migration prep.
- GitHub remote pushed.
- WSL2 handoff document added.
- WSL2 clone and read-only startup handoff have been completed by the WSL2 Codex CLI session.

## 4. Current Progress

Current GitHub branch:

- `sac-integration`

Key commits:

```text
5be043c Add SAC deterministic eval smoke
c59eda0 Save SAC normalizers in checkpoints
d5c0e8d Record Route B GPU smoke results
6fa5160 Sync WSL2 GPU preflight documentation
0ffcb91 Add WSL2 Codex handoff guide
```

GitHub remote:

```text
https://github.com/Kam1-hub/mujoco_playground.git
```

Windows CPU dev state:

- Worktree was clean after the handoff commit.
- Menagerie exists locally and is ignored.
- Logs/checkpoints are ignored.
- `.venv` is ignored.
- CPU dev machine should not continue to larger training.

WSL2 target workspace state:

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration...origin/sac-integration`
- Latest committed baseline before 50k report update:
  `5be043c Add SAC deterministic eval smoke`
- Menagerie: present at `1b86ece576591213e2b666ebf59508454200ca97`
- Python env: present under ignored `.venv`
- CUDA JAX: validated, backend `gpu`, device `cuda:0`
- GPU preflight: `PASS`
- Route B GPU 10k smoke: `PASS`
- Existing GPU 10k checkpoint deterministic eval readiness: `FAIL`, because
  the checkpoint predates observation normalizer persistence.
- Future checkpoint schema: patched to save `policy_normalizer` and
  `value_normalizer`.
- Schema dry-run checkpoint:
  `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- New GPU smoke checkpoint:
  `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- Deterministic eval smoke:
  `scripts/eval_sac_checkpoint.py` passed at 4 env x 200 steps with
  `EVAL_OK`; JSON is under ignored `./logs/sac_eval_smoke`.
- Route B GPU 50k sanity: `PASS`.
- 50k checkpoint:
  `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- 50k bounded deterministic eval:
  16 env x 1000 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_50k/eval_16x1000.json`, no action/reward/obs NaN.
- Logs, checkpoints, `.venv`, and menagerie remain ignored and are not
  committed.

## 5. Completed CPU Validation

Windows CPU dev runtime:

- Python: `3.14.3`
- JAX: `0.10.0`
- JAX backend/devices: `cpu`, `cpu:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- CUDA tools: unavailable

Passing CPU commands:

- `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts`
- `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"`
- `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain`
- `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain`
- `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help`
- `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help`
- Route B dry run
- Route B CPU tiny smoke
- `scripts/check_sac_checkpoint.py`
- `scripts/gpu_preflight.py --impl jax` without `--require_gpu`

CPU tiny smoke result:

```text
status: TRAIN_OK
env_steps: 256
gradient_steps: 121
actor_loss: -1.6421515941619873
critic_loss: 0.02648034505546093
alpha: 0.04801943153142929
sps: 4.8985466406301095
checkpoint: ./logs/sac_lift_cpu_tiny/sac_lift_step_256.pkl
```

Checkpoint note:

- The CPU checkpoint is an ignored runtime artifact.
- It does not need to be migrated for GPU smoke.

## 6. Menagerie State

Required target path:

```text
g1_env/external_deps/mujoco_menagerie
```

Required source:

```text
https://github.com/deepmind/mujoco_menagerie.git
```

Required commit:

```text
1b86ece576591213e2b666ebf59508454200ca97
```

Windows CPU dev:

- Menagerie exists.
- Commit is pinned.
- Directory is ignored by git.

WSL2:

- Menagerie was missing after the initial clone.
- It must be cloned or copied before GPU preflight.
- Re-cloning inside WSL2 is preferred if network is available.

## 7. Migration Tools

Added scripts:

- `scripts/check_sac_checkpoint.py`
  - Validates Route B pickle checkpoints.
  - Reports `policy_normalizer`, `value_normalizer`, `normalize_observations`,
    and `deterministic_eval_ready`.
  - `--require_eval_ready` fails when a checkpoint cannot support trusted
    deterministic actor evaluation.
- `scripts/eval_sac_checkpoint.py`
  - Runs bounded deterministic actor eval from a Route B checkpoint.
  - Loads checkpoint config/params/normalizer, rebuilds the G1 env/network, and
    emits JSON metrics.
  - Does not train, update replay, render, or touch PPO/RSL paths.
- `scripts/gpu_preflight.py`
  - Does not train.
  - Checks JAX/MuJoCo/Brax, GPU visibility when requested, menagerie commit, registry, Route B import, and flat/rough env reset/step.
- `scripts/gpu_smoke_route_b.sh`
  - Linux/WSL2 10k GPU smoke wrapper.
- `scripts/gpu_smoke_route_b.ps1`
  - PowerShell GPU smoke wrapper with Windows-native caveat.

Added handoff:

- `WSL2_CODEX_HANDOFF.md`

Relevant reports:

- `reports/sac_integration/07_gpu_migration_prep.md`
- `reports/sac_integration/06_next_actions.md`
- `reports/sac_integration/04_smoke_results.md`
- `reports/sac_integration/03_route_b_lift_sac.md`

External local references for WSL2 agent:

- `WSL2_GPU_EXPERIENCE.md`
- `/home/admin/projects/mujoco_playground/TRAINING_NOTES.md`
- `/home/admin/projects/mujoco_playground/.md_edit/`

Use these files only as environment and operating-experience references. Do not let them override this project's SAC design or smoke ladder.

## 8. Next Immediate Objective

Current objective:

- Review and record the completed WSL2 GPU preflight and Route B GPU 10k smoke.

Completed in WSL2:

- `uv sync --frozen --extra cuda`
- menagerie checkout at `1b86ece576591213e2b666ebf59508454200ca97`
- JAX backend/device check: `gpu`, `cuda:0`
- `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu`: `PASS`
- `bash scripts/gpu_smoke_route_b.sh`: `TRAIN_OK`

## 9. GPU 10k Smoke Objective

Command wrapper:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

Underlying Route B command:

```bash
python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 10000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 8192 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_10k
```

Acceptance criteria:

- JAX sees GPU/CUDA/ROCm.
- `gpu_preflight.py --require_gpu` passes.
- Flat env load/reset/step passes.
- Rough env load/reset/step passes.
- Route B reports `TRAIN_OK`.
- Checkpoint is saved under `./logs/sac_lift_gpu_10k`.
- Actor loss, critic loss, alpha, and SPS are finite and recorded.
- No NaN is reported.
- Actual `env_steps` are recorded; Route B may record `9984` for a `10000` target with `128` envs.
- Idle and final or peak VRAM are recorded when available.

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl
env_steps: 9984
gradient_steps: 142
wall_time: 56.50599093900382
sps: 176.68922947970893
actor_loss: -1.9058758020401
critic_loss: 0.08382290601730347
alpha: 0.04770537465810776
alpha_loss: 1.585930585861206
policy_log_prob: -18.79882049560547
q: 1.0673823356628418
target_q: 1.0646085739135742
truncation_fraction: 0.0
```

No NaN was observed in reported scalar metrics.

Actual step explanation:

- Route B uses `num_envs * (num_timesteps // num_envs)`.
- `128 * (10000 // 128) = 9984`.

Warnings observed and classified as non-fatal:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- CUDA timer warmup warning: `Delay kernel timed out`.

## 10. After GPU 10k Smoke

The existing GPU 10k checkpoint is useful as a smoke artifact but not as a
trusted deterministic-eval artifact:

- `normalize_observations=True`
- `policy_normalizer` missing
- `value_normalizer` missing

This cannot be fixed retroactively for the old checkpoint. A regenerated
normalizer-ready GPU checkpoint has now passed readiness and a small deterministic
eval smoke:

```text
checkpoint: ./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl
eval: 4 env x 200 steps
status: EVAL_OK
episode_reward_mean: -3.3628087043762207
done_fraction: 1.0
action/reward/obs NaN: false/false/false
json: ./logs/sac_eval_smoke/eval_4x200.json
```

After report review, only then consider with explicit user confirmation:

- commit the 50k sanity report update
- 100k sanity run
- PPO comparison

Do not jump directly to 1M training. The 4x200 eval is a smoke, not a full
benchmark. The 50k sanity run is now validated; 100k and 1M remain
`NOT VALIDATED`.

## 11. 50k Sanity Result

Command class:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 50000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 50000 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_50k_sanity
```

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl
env_steps: 49920
gradient_steps: 766
wall_time: 35.99766752999858
sps: 1386.756515777009
actor_loss: -3.6135072708129883
critic_loss: 0.07097882032394409
alpha: 0.03992176800966263
alpha_loss: 1.327394962310791
policy_log_prob: -18.81831169128418
policy_q: 2.8622469902038574
q: 2.884032726287842
target_q: 2.899707317352295
truncation_fraction: 0.0
```

Checkpoint readiness:

- `scripts/check_sac_checkpoint.py --require_eval_ready`: `PASS`
- `policy_normalizer`: present
- `value_normalizer`: present

Bounded deterministic eval:

```text
status: EVAL_OK
json: ./logs/sac_eval_50k/eval_16x1000.json
eval_env_steps: 16000
episode_reward_mean: -3.5016322135925293
episode_reward_std: 0.75983726978302
episode_reward_min: -6.096090316772461
episode_reward_max: -2.531925916671753
done_fraction: 1.0
wall_time: 74.76505397899746
sps: 214.00372431342888
action_nan: false
reward_nan: false
obs_nan: false
truncation_present: true
truncation_fraction: 0.0
```

Post-run GPU snapshot:

- GPU: RTX 4070 SUPER
- VRAM: `1602MiB / 12282MiB`
- Temperature: `56C`
- Power: `9W / 220W`
- GPU util: `8%`
- Process table only showed `/Xwayland`.

Known warnings:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- These remain non-fatal WSL2/JAX noise unless they accompany a failed command.

No NaN, Inf, OOM, CUDA, checkpoint, or eval error was observed.

100k sanity and 1M training remain `NOT VALIDATED`. The next training scale
should be 100k only after explicit user confirmation; do not jump directly to
1M.

## 12. Current Position In One Sentence

SAC Route B is implemented; CPU tiny smoke, WSL2 GPU preflight, Route B GPU 10k smoke, normalizer-ready checkpoint validation, bounded deterministic eval smoke, 50k sanity, and 50k bounded eval have passed; 100k, 1M, full eval benchmarking, domain randomization, and fine-tuning remain `NOT VALIDATED`.
