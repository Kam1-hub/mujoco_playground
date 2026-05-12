# G1 SAC Integration Status And Roadmap

Status: updated on 2026-05-12 after WSL2 path correction and GPU operating notes review.

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
11. Longer sanity run

Status:

- Steps 1 through 8 are complete.
- Step 9 is in progress on WSL2/Linux CUDA.
- Step 10 has not started.
- Step 11 is out of scope until 10k GPU smoke passes.

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
0ffcb91 Add WSL2 Codex handoff guide
d73d45c Prepare GPU migration validation scripts
4986032 Validate G1 SAC CPU smoke
6027358 Add Route B asymmetric SAC baseline
74594b3 Add Route A Brax SAC fallback
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
- Latest commit: `0ffcb91 Add WSL2 Codex handoff guide`
- Menagerie: missing
- Python env: missing
- CUDA JAX: not checked in the read-only startup pass

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

- Complete WSL2/Linux CUDA environment preparation and run GPU preflight.

Do next in WSL2:

1. Read project handoff and local environment notes:
   - `AGENTS.md`
   - `WSL2_CODEX_HANDOFF.md`
   - `WSL2_GPU_EXPERIENCE.md`
   - `reports/sac_integration/07_gpu_migration_prep.md`
   - `reports/sac_integration/06_next_actions.md`
   - `/home/admin/projects/mujoco_playground/TRAINING_NOTES.md`
2. Rebuild Python env in WSL2.
3. Prepare menagerie at the pinned commit.
4. Verify CUDA/JAX visibility.
5. Run:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Only if preflight passes, stop and wait for user confirmation before GPU 10k smoke.

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

## 10. After GPU 10k Smoke

If GPU 10k smoke passes:

1. Update `reports/sac_integration/04_smoke_results.md`.
2. Update `reports/sac_integration/05_known_issues.md`.
3. Update `reports/sac_integration/06_next_actions.md`.
4. Record:
   - command
   - runtime versions
   - JAX backend/devices
   - menagerie commit
   - env shapes
   - checkpoint path
   - env steps
   - gradient steps
   - wall time
   - SPS
   - actor loss
   - critic loss
   - alpha
   - NaN status
5. Commit the report updates.

Only then consider:

- deterministic eval rollout metrics
- slightly longer sanity run
- PPO comparison

Do not jump directly to 1M training.

## 11. Current Position In One Sentence

SAC Route B is implemented and CPU tiny smoke has passed; GitHub migration and WSL2 handoff are complete; the project is now waiting on WSL2 Python/CUDA/JAX setup, pinned menagerie, and `gpu_preflight.py --impl jax --require_gpu` before any 10k GPU smoke is allowed.
