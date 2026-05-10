# G1 SAC Integration Master Plan

## Objective

Add a SAC baseline to the isolated G1 pipeline in this repository:

```text
G1 pipeline
  ├── PPO baseline: train-g1-jax
  ├── RSL-RL baseline: train-g1-rsl
  └── SAC baseline: train-g1-sac / train-g1-sac-brax
```

Success is not a long training curve on day one. The first target is a closed validation ladder:

1. Import works.
2. Env registry/load works.
3. Reset/step works.
4. SAC initializes.
5. Tiny CPU smoke runs without NaN.
6. 10k smoke can run in a proper JAX runtime.
7. Checkpoint and eval metrics are saved.
8. Reports document pass/fail and next action.

## Non-Goals For First Pass

- Do not implement world model pretraining.
- Do not implement LIFT fine-tuning.
- Do not implement vision SAC.
- Do not implement real deployment.
- Do not merge LIFT-humanoid forks.
- Do not start long training before smoke tests pass.

## Phase 0: Workspace Manifest

Create/update:

```text
reports/sac_integration/00_workspace_manifest.md
```

Must include:

- Project path.
- Template source path.
- LIFT report reference path.
- OS, Python, JAX, MuJoCo, Torch, CUDA/WSL2 status.
- Git branch and commit.
- Whether commands were run or only statically planned.

Validation commands:

```powershell
git status --short
git branch --show-current
git rev-parse HEAD
python --version
python -c "import importlib.util as u; print(u.find_spec('jax'))"
```

## Phase 1: G1 SAC Readiness Audit

Do not write SAC learner yet.

Create:

```text
scripts/check_g1_env_api.py
reports/sac_integration/01_g1_sac_readiness.md
```

`scripts/check_g1_env_api.py` should inspect:

- `g1_env.registry.ALL_ENVS`
- `registry.get_default_config(env_name)`
- `registry.load(env_name, config=...)`
- `env.observation_size`
- `env.action_size`
- `env.dt`
- `reset(rng)`
- `step(state, zero_action)`
- `state.obs` type and keys
- `state.obs["state"].shape`
- `state.obs["privileged_state"].shape`
- `state.info` keys and whether `truncation` exists
- metrics keys
- config action scale and where action is applied

Acceptance:

- Flat and rough env both load or failure is fully diagnosed.
- Report explicitly says whether `truncation` is native, wrapper-generated, or missing.
- Report explicitly says whether action scale should be `env_cfg.action_scale`, ones, or another value.

## Phase 2: Route A - Minimal Brax SAC

Goal: establish whether current G1 env can be consumed by an existing/upstream Brax SAC path with minimal changes.

Create:

```text
learning/sac_wrappers.py
learning/train_jax_sac_brax.py
g1_env/config/sac_params.py
reports/sac_integration/02_route_a_brax_sac.md
```

Route A defaults:

- Symmetric SAC first: actor obs = `state`, critic obs = `state`.
- Use `SelectObsWrapper` if upstream SAC cannot handle dict obs.
- Use the existing G1 wrappers if compatible.
- Do not copy LIFT learner.

Suggested command:

```bash
python -m learning.train_jax_sac_brax \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 10000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 8192 \
  --grad_updates_per_step 2 \
  --use_wandb False \
  --render False
```

Acceptance:

- `--help` works.
- Env load works.
- SAC init works or failure is classified as env/wrapper/network/replay/jit/eval/dependency.

## Phase 3: Route B - LIFT-Style Asymmetric SAC

This is the main target route.

Create:

```text
learning/sac_lift/
  __init__.py
  config.py
  train.py
  networks.py
  losses.py
  distributions.py
  replay_buffer.py
  normalizer.py
  evaluator.py
  checkpoint.py
  types.py

learning/train_jax_sac_lift.py
reports/sac_integration/03_route_b_lift_sac.md
```

### Required Algorithm Features

- Tanh Gaussian actor.
- Log std bounds.
- Twin Q.
- Target Q with Polyak update.
- Trainable entropy alpha.
- Target entropy configurable by coefficient.
- Uniform replay.
- Optional observation normalization.
- Dict obs support.
- Actor obs key default: `state`.
- Critic obs key default: `privileged_state`.
- UTD via `grad_updates_per_step`.
- Truncation masking for timeout handling.
- Deterministic evaluation.
- Checkpoint save/load enough for smoke continuation.

### Required CLI

Use explicit names:

```text
--actor_learning_rate
--critic_learning_rate
--alpha_learning_rate
--policy_hidden_layer_sizes
--q_hidden_layer_sizes
--policy_obs_key
--value_obs_key
--min_replay_size
--max_replay_size
--grad_updates_per_step
--reward_scaling
--discounting
--tau
--target_entropy_coef
--normalize_observations
--deterministic_eval
--logdir
--dry_run
```

Do not add a vague `--learning_rate` unless it explicitly maps to actor, critic, and alpha and logs that mapping.

### Action Scaling

G1 env source applies normalized action as:

```python
motor_targets = default_pose + action * env_cfg.action_scale
```

For the first SAC version, prefer the conservative design:

- Actor outputs tanh-normalized actions in `[-1, 1]`.
- Env performs `action_scale` internally.
- Use external `policy_output_scale = ones(action_size)` unless a separate policy output scale is justified.
- If implementing LIFT-style log-prob action scale correction, only apply it when the policy output is actually externally rescaled before env/Q. Do not double-scale with `env_cfg.action_scale`.

Document this decision in `03_route_b_lift_sac.md`.

### Fallbacks

- Missing `privileged_state`: fallback to `state`, log warning.
- Missing `truncation`: synthesize zeros, log warning.
- Missing action scale: use ones, log warning.

## Phase 4: Route C - Clean-Room SAC Fallback

Only start if Route A and Route B hit blockers.

Goal:

- Minimal standard SAC.
- Symmetric obs first.
- No LIFT code copy.
- Then incrementally add privileged critic, action scaling, high UTD, domain randomization.

Report:

```text
reports/sac_integration/03b_route_c_clean_sac.md
```

## Phase 5: Validation Ladder

Run in order. Do not skip earlier levels.

### Level 0: Static Checks

```bash
python -m compileall g1_env learning
python -c "import g1_env; print(g1_env.registry.ALL_ENVS)"
python -m learning.train_jax_sac_brax --help
python -m learning.train_jax_sac_lift --help
```

### Level 1: Env API

```bash
python scripts/check_g1_env_api.py --env_name G1JoystickFlatTerrain
python scripts/check_g1_env_api.py --env_name G1JoystickRoughTerrain
```

### Level 2: SAC Dry Run

```bash
python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 1 \
  --num_envs 2 \
  --num_eval_envs 2 \
  --batch_size 2 \
  --min_replay_size 2 \
  --max_replay_size 16 \
  --grad_updates_per_step 1 \
  --dry_run
```

### Level 3: CPU Tiny Smoke

PowerShell:

```powershell
$env:JAX_PLATFORM_NAME="cpu"
python -m learning.train_jax_sac_lift `
  --env_name G1JoystickFlatTerrain `
  --num_timesteps 256 `
  --num_envs 2 `
  --num_eval_envs 2 `
  --batch_size 8 `
  --min_replay_size 16 `
  --max_replay_size 128 `
  --grad_updates_per_step 1 `
  --render False `
  --use_wandb False `
  --logdir ./logs/sac_lift_cpu_tiny
```

### Level 4: GPU Smoke

Requires WSL2/Linux CUDA JAX environment unless the current environment proves otherwise.

```bash
python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 10000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 8192 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_smoke
```

### Level 5: 1M Sanity

Only after Level 4 passes.

```bash
python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --num_timesteps 1000000 \
  --num_envs 1024 \
  --num_eval_envs 128 \
  --batch_size 1024 \
  --min_replay_size 8192 \
  --max_replay_size 1000000 \
  --grad_updates_per_step 4 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_1m
```

## Phase 6: Reports

Keep these files current:

```text
reports/sac_integration/00_workspace_manifest.md
reports/sac_integration/01_g1_sac_readiness.md
reports/sac_integration/02_route_a_brax_sac.md
reports/sac_integration/03_route_b_lift_sac.md
reports/sac_integration/04_smoke_results.md
reports/sac_integration/05_known_issues.md
reports/sac_integration/06_next_actions.md
```

Each report must include:

- Files changed.
- Commands run.
- Pass/fail table.
- Full traceback if failed.
- Shapes.
- Runtime metrics if available.
- Current blocker.
- Next proposed fix.

## Acceptance Criteria For User Review

Minimum acceptable implementation:

- `train-g1-sac` entry point exists.
- `python -m learning.train_jax_sac_lift --help` works.
- `scripts/check_g1_env_api.py` produces a reportable schema.
- Tiny CPU smoke either passes or fails with a complete, actionable traceback.
- SAC code supports dict obs and asymmetric critic by design.
- No LIFT forks are imported as runtime dependencies.
- Existing `train-g1-jax` and `train-g1-rsl` entry points remain usable.

