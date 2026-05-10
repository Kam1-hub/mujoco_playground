# SAC Integration Status Report

## Snapshot
- Project path: `D:\mujoco_playground\g1_sac_dev`
- Git branch: `codex/route-a-brax-sac`
- Git commit: `6dcd01d`
- OS: Windows native, `Microsoft Windows NT 10.0.26200.0`
- Python: `.venv\Scripts\python.exe`, Python 3.14.3
- JAX: `.venv` has JAX 0.10.0, CPU backend
- MuJoCo: `.venv` has MuJoCo 3.8.0
- CUDA / WSL2: GPU not validated
- Date: 2026-05-11

## Phase Checklist
- [x] Check upstream Brax SAC import/API.
- [x] Implement `SelectObsWrapper` for state-only upstream SAC.
- [x] Implement minimal Route A runner.
- [x] Add `train-g1-sac-brax` entry point.
- [x] Run Route A help.
- [x] Run Route A dry run.
- [x] Record full traceback for train smoke if blocked.

## Current Route
- Route A: implemented as symmetric fallback, actor obs = `state`, critic obs = `state`.
- Route B: not started
- Route C: not started

## Files Changed
- `learning/sac_wrappers.py`
- `learning/train_jax_sac_brax.py`
- `g1_env/config/sac_params.py`
- `g1_env/config/__init__.py`
- `pyproject.toml`
- `reports/sac_integration/02_route_a_brax_sac.md`

## Commands Run

| Command | Status | Notes |
|---|---:|---|
| `.venv\Scripts\python.exe` import probe for `brax.training.agents.sac` | PASS | Brax 0.14.2 exposes upstream SAC. |
| Source audit of `.venv\Lib\site-packages\brax\training\agents\sac\train.py` | PASS | Upstream SAC rejects dict observations. |
| `python -m learning.train_jax_sac_brax --help` | PASS | Help works even with system Python because heavy imports are lazy. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help` | PASS | Help works in `.venv`. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --dry_run --num_timesteps 1 --min_replay_size 0 --max_replay_size 16 --num_envs 1 --num_eval_envs 1 --batch_size 1` | PASS | Imports SAC/JAX, builds config, prints CPU device, exits before env load. |
| `python -m compileall learning g1_env scripts` | PASS | Static compile succeeds for current tree. |
| `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --env_name G1JoystickFlatTerrain --num_timesteps 10000 --num_envs 128 --num_eval_envs 32 --batch_size 256 --min_replay_size 1024 --max_replay_size 8192 --grad_updates_per_step 2 --use_wandb False --render False` | BLOCKED_BY_DEPENDENCY | Env load blocked by missing `mujoco_menagerie`; full traceback below. |

## Results

| Test | Status | Notes |
|---|---:|---|
| Brax SAC availability | PASS | `brax.training.agents.sac.train` and `networks` are available. |
| Dict obs support | FAIL | Upstream Brax SAC raises `NotImplementedError("Dictionary observations not implemented in SAC")`. |
| Route A feasibility | PASS | Feasible only with `SelectObsWrapper(obs_key="state")`, symmetric actor/Q. |
| Action scaling | PASS | No external scaling added; env applies `action_scale`. |
| Runtime train smoke | BLOCKED_BY_DEPENDENCY | Blocked before simulation by missing `mujoco_menagerie` assets. |

## Shapes
- action_size: expected 29, runtime not validated.
- obs type: Route A converts dict obs to array.
- obs keys: raw env has `state`, `privileged_state`; Route A selects `state`.
- state shape: statically 103.
- privileged_state shape: not used by Route A.
- replay transition shape: upstream Brax SAC transition over selected state obs; runtime not validated.

## Runtime Metrics
- steps: NOT VALIDATED
- wall time: NOT VALIDATED
- SPS: NOT VALIDATED
- eval reward: NOT VALIDATED
- actor loss: NOT VALIDATED
- critic loss: NOT VALIDATED
- alpha: NOT VALIDATED
- NaN: NOT VALIDATED

## Errors / Blockers
- `registry.load` is blocked until `mujoco_menagerie` assets are available or download is explicitly allowed.
- Route A is not asymmetric; it is intentionally a fallback/minimal baseline.

### Route A Smoke Traceback

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_brax.py", line 164, in main
    registry.load(args.env_name, config=env_cfg),
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\registry.py", line 45, in load
    return locomotion.load(env_name, config, config_overrides)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 97, in load
    mjx_env.ensure_menagerie_exists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_brax.py", line 121, in _raise_missing_menagerie
    raise RuntimeError(
RuntimeError: mujoco_menagerie is missing at D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie; rerun with --allow_menagerie_download only when network/file writes are authorized.
```

## Diagnosis
- Route A should not block the main SAC path. It provides a quick upstream SAC smoke path where assets/runtime are available.

## Next Proposed Fix
- Continue Route B asymmetric SAC implementation.
