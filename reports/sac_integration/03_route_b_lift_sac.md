# SAC Integration Status Report

## Snapshot
- Project path: `D:\mujoco_playground\g1_sac_dev`
- Git branch: `codex/route-b-lift-sac`
- Git commit: `74594b3`
- OS: Windows native, `Microsoft Windows NT 10.0.26200.0`
- Python: `.venv\Scripts\python.exe`, Python 3.14.3
- JAX: `.venv` has JAX 0.10.0, CPU backend
- MuJoCo: `.venv` has MuJoCo 3.8.0
- CUDA / WSL2: GPU not validated
- Date: 2026-05-11

## Phase Checklist
- [x] Add clean SAC package structure.
- [x] Add explicit CLI flags for actor/critic/alpha learning rates.
- [x] Add `q_hidden_layer_sizes`, not `value_hidden_layer_sizes`.
- [x] Keep actor obs default `state`.
- [x] Keep critic obs default `privileged_state`.
- [x] Add tanh Gaussian actor with log-prob correction.
- [x] Add twin Q and Polyak target update.
- [x] Add trainable entropy alpha.
- [x] Add uniform replay buffer.
- [x] Add observation normalization.
- [x] Add dry-run initialization and checkpoint save.
- [x] Run Route B help.
- [x] Run Route B dry run.
- [x] Record runtime smoke status.

## Current Route
- Route A: implemented as upstream Brax symmetric fallback
- Route B: implemented as main asymmetric SAC route
- Route C: not enabled; Route B is not blocked at static implementation level

## Files Changed
- `learning/sac_lift/__init__.py`
- `learning/sac_lift/config.py`
- `learning/sac_lift/types.py`
- `learning/sac_lift/distributions.py`
- `learning/sac_lift/networks.py`
- `learning/sac_lift/losses.py`
- `learning/sac_lift/replay_buffer.py`
- `learning/sac_lift/normalizer.py`
- `learning/sac_lift/evaluator.py`
- `learning/sac_lift/checkpoint.py`
- `learning/sac_lift/train.py`
- `learning/train_jax_sac_lift.py`
- `g1_env/config/sac_params.py`
- `pyproject.toml`
- `reports/sac_integration/03_route_b_lift_sac.md`

## Commands Run

| Command | Status | Notes |
|---|---:|---|
| Route B implementation | PASS | Static code added; validation pending. |
| `python -m learning.train_jax_sac_lift --help` | PASS | Help works with system Python because heavy runtime imports are lazy. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help` | PASS | Help works in `.venv`. |
| `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | Static compile passes. |
| `python -m compileall g1_env learning` | PASS | Static compile passes with system Python. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --dry_run --logdir ./logs/sac_lift_dry_run` | PASS | Initialized networks/replay, ran one dummy SAC update, saved checkpoint, avoided env load. |
| `python -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --dry_run` | BLOCKED_BY_DEPENDENCY | System Python lacks JAX; full traceback is in `04_smoke_results.md`. |
| `.\.venv\Scripts\python.exe -m compileall learning\sac_lift learning\train_jax_sac_lift.py` | PASS | Recompiled after stricter dry-run patch. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --logdir ./logs/sac_lift_blocked_probe` | BLOCKED_BY_DEPENDENCY | Blocked before simulation by missing `mujoco_menagerie`; full traceback below. |

## Results

| Test | Status | Notes |
|---|---:|---|
| `train-g1-sac` entry point | IMPLEMENTED | Added in `pyproject.toml`. |
| CLI learning-rate bug avoidance | PASS | Uses explicit `--actor_learning_rate`, `--critic_learning_rate`, `--alpha_learning_rate`; no ambiguous `--learning_rate`. |
| Q hidden sizes | PASS | Uses `--q_hidden_layer_sizes`; no SAC `value_hidden_layer_sizes`. |
| Actor/critic obs keys | PASS | Defaults are `state` and `privileged_state`. |
| Action scaling | PASS | Actor outputs tanh-normalized actions; no external multiplication by `env_cfg.action_scale`. G1 env applies `default_pose + action * action_scale`. |
| Truncation handling | PASS | If `state.info["truncation"]` exists, it is stored in replay. If missing, zeros are synthesized. Critic loss masks TD error by `(1 - truncation)`, matching the Brax SAC timeout convention. |
| Route B dry run | PASS | `DRY_RUN_OK`, policy obs `[103]`, value obs `[216]`, action `[29]`, replay capacity `16`, one gradient step, checkpoint saved. |
| Runtime train smoke | BLOCKED_BY_DEPENDENCY | Blocked before simulation by missing menagerie assets. |

## Shapes
- action_size: dry-run default 29.
- obs type: dict for Route B.
- obs keys: `state`, `privileged_state`.
- state shape: dry-run default `(103,)`.
- privileged_state shape: dry-run default `(216,)`, static and not runtime validated.
- replay transition shape: policy obs `(batch, 103)`, value obs `(batch, 216)`, action `(batch, 29)`.

## Runtime Metrics
- steps: dry-run initialized 0 env steps and ran 1 dummy gradient step; non-dry-run smoke blocked before env load
- wall time: dry-run completed in command wall time; detailed timing not measured
- SPS: NOT VALIDATED
- eval reward: NOT VALIDATED
- actor loss: dry-run dummy update `-0.8600597381591797`
- critic loss: dry-run dummy update `0.4279707074165344`
- alpha: dry-run dummy update `0.049787066876888275`
- NaN: NOT VALIDATED

## Errors / Blockers
- Runtime env load/reset/step remains blocked until `mujoco_menagerie` assets are available or download is explicitly authorized.
- This laptop is not intended for simulation/training, so CPU tiny smoke is expected to remain `NOT VALIDATED` unless explicitly allowed.

### Route B Blocked Probe Traceback

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\learning\sac_lift\train.py", line 184, in main
    result = train(config)
  File "D:\mujoco_playground\g1_sac_dev\learning\sac_lift\train.py", line 84, in train
    raise RuntimeError(
RuntimeError: mujoco_menagerie is missing at D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie; rerun with --allow_menagerie_download only when network/file writes are authorized.
```

## Diagnosis
- Route B is statically implemented and designed for asymmetric dict observations.
- Runtime validation should be performed on a machine with assets and suitable JAX runtime, or after explicitly authorizing/providing menagerie assets.

## Next Proposed Fix
- Continue validation ladder report updates.
- Resolve menagerie assets before env load/reset/step or CPU tiny smoke can be counted.
