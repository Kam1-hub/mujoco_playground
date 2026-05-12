# SAC Integration Status Report

## Current Status Update

Status: superseded on 2026-05-12 by later validation reports.

This file preserves the original Phase 1 audit and tracebacks. The current
validated state is:

- Flat and rough env load/reset/step pass with `--impl jax` after menagerie is
  present.
- Runtime obs is dict with `state` shape `(103,)` and `privileged_state` shape
  `(216,)`.
- Runtime action size is `29`.
- Native `state.info["truncation"]` remains absent; Route B synthesizes zero
  truncation.
- Route B CPU tiny smoke passed; GPU smoke remains `NOT VALIDATED`.

For the active ladder, use `08_project_status_roadmap.md`,
`07_gpu_migration_prep.md`, and `06_next_actions.md`.

## Snapshot
- Project path: `D:\mujoco_playground\g1_sac_dev`
- Git branch: `codex/phase1-env-audit`
- Git commit: `2fce0c50de0bfc13028ed3127aecd7294adcd012`
- OS: Windows native, `Microsoft Windows NT 10.0.26200.0`
- System Python: `Python 3.12.11`
- `.venv` Python: `Python 3.14.3`
- System Python JAX/MuJoCo/Brax: not importable at Phase 1 start
- `.venv` JAX/MuJoCo/Brax: importable; JAX backend is CPU
- CUDA / WSL2: `nvidia-smi`/`nvcc` not found; WSL executable present but no Linux CUDA runtime validated
- Date: 2026-05-11

## Phase Checklist
- [x] Add `scripts/check_g1_env_api.py`.
- [x] Preserve full traceback when runtime imports/load/reset/step fail.
- [x] Complete static source readiness audit even when runtime checks are blocked.
- [x] Run flat env API command and record output.
- [x] Run rough env API command and record output.

## Current Route
- Route A: not started
- Route B: not started
- Route C: not started

## Files Changed
- `scripts/check_g1_env_api.py`
- `reports/sac_integration/01_g1_sac_readiness.md`

## Commands Run

| Command | Status | Notes |
|---|---:|---|
| `rg -n "ALL_ENVS\|_envs\|def load\|def get_default_config\|get_domain_randomizer" g1_env` | PASS | Static registry audit. |
| `rg -n "truncation\|privileged_state\|action_scale\|def _get_obs\|def reset\|def step\|action_size\|observation_size\|metrics\|info" g1_env\_src learning` | PASS | Static env/wrapper audit. |
| `python scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | BLOCKED_BY_DEPENDENCY | Failed at `import jax`; full traceback below. |
| `python scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | BLOCKED_BY_DEPENDENCY | Failed at `import jax`; full traceback below. |
| `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | BLOCKED_BY_DEPENDENCY | Imports/config/registry pass; `registry.load` blocked because `mujoco_menagerie` is absent and download is not allowed by default. |
| `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | BLOCKED_BY_DEPENDENCY | Same as flat env. |
| `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"` | PASS | Printed `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`. |
| `.\.venv\Scripts\python.exe -c "from g1_env import registry; c=registry.get_default_config(...); ..."` | PASS | Printed `0.02 0.002 1000 1 0.5 warp`. |
| `python -m compileall scripts` | PASS | `scripts/check_g1_env_api.py` compiled successfully. |

## Results

| Test | Status | Notes |
|---|---:|---|
| Registry source | PASS | `G1JoystickFlatTerrain` and `G1JoystickRoughTerrain` are registered in `g1_env/_src/locomotion/__init__.py`. |
| Default config source | PASS | G1 default config includes `ctrl_dt=0.02`, `sim_dt=0.002`, `episode_length=1000`, `action_repeat=1`, `action_scale=0.5`. |
| Dict observations source | PASS | `_get_obs` returns `state` and `privileged_state`. |
| Action scaling source | PASS | Env step computes `motor_targets = default_pose + action * action_scale`; SAC must not double-scale actions. |
| Native truncation source | FAIL | Raw G1 env reset/step source does not populate `state.info["truncation"]`. Training wrappers may add it via Brax `EpisodeWrapper`; Route B needs a zero fallback or wrapper-generated timeout flag. |
| Runtime flat env API | BLOCKED_BY_DEPENDENCY | System Python fails at `import jax`; `.venv` gets through imports/config but is blocked by missing `mujoco_menagerie`. |
| Runtime rough env API | BLOCKED_BY_DEPENDENCY | System Python fails at `import jax`; `.venv` gets through imports/config but is blocked by missing `mujoco_menagerie`. |

## Shapes
- action_size: statically expected `self._mjx_model.nu`; local G1 comments/source usage indicate 29, runtime not validated.
- obs type: dict, static.
- obs keys: `state`, `privileged_state`, static.
- state shape: statically `3 + 3 + 3 + 3 + 29 + 29 + 29 + 4 = 103`.
- privileged_state shape: statically likely `state(103) + gyro(3) + accelerometer(3) + gravity(3) + linvel(3) + global_angvel(3) + joint_angles(29) + joint_vel(29) + root_height(1) + actuator_force(29) + contact(2) + feet_vel(6) + feet_air_time(2) = 216`. Runtime not validated. The source comment says `4*3` for `feet_vel`, but constants/XML define two feet linear-velocity sensors.
- replay transition shape: NOT VALIDATED until SAC replay implementation.

## Runtime Metrics
- steps: NOT VALIDATED
- wall time: NOT VALIDATED
- SPS: NOT VALIDATED
- eval reward: NOT VALIDATED
- actor loss: NOT VALIDATED
- critic loss: NOT VALIDATED
- alpha: NOT VALIDATED
- NaN: NOT VALIDATED

## Static Findings
- Env registry: `registry.ALL_ENVS` delegates to locomotion envs, with flat/rough entries in `g1_env/_src/locomotion/__init__.py`.
- Config contract: `registry.get_default_config(env_name)` returns the locomotion default config.
- Env loading: `registry.load` calls `mjx_env.ensure_menagerie_exists()` before constructing the env, so runtime load is currently blocked by missing `mujoco_menagerie` assets. The checker prevents hidden network clone by default.
- Domain randomization: both G1 envs have `g1_randomize.domain_randomize` registered.
- Actor obs: `obs["state"]` is the correct default.
- Critic obs: `obs["privileged_state"]` is available statically and should be the default for Route B.
- Runtime-light registry/config probes pass under `.venv`; full `registry.load/reset/step` is not validated because the menagerie assets are absent.
- SelectObsWrapper: needed for Route A if upstream Brax SAC does not support dict observations.
- TruncationWrapper/fallback: needed for Route B because raw G1 source does not set `info["truncation"]`.
- ActionScaleAdapter: external scaling should default to identity. The G1 env already applies `env_cfg.action_scale` internally.

## Errors / Blockers
- Runtime imports are expected to fail in the current environment because `jax`, `mujoco`, and `brax` were not importable during startup probing.
- `.venv` imports JAX/MuJoCo/Brax successfully, but `registry.load` is blocked by missing assets:

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 63, in _run_stage
    value = fn()
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 176, in <lambda>
    lambda: registry.load(args.env_name, config=config),
            ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\registry.py", line 45, in load
    return locomotion.load(env_name, config, config_overrides)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 97, in load
    mjx_env.ensure_menagerie_exists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 126, in _raise_missing_menagerie
    raise RuntimeError(
RuntimeError: mujoco_menagerie is missing at D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie; rerun with --allow_menagerie_download only when network/file writes are authorized.
```

### Flat Runtime Traceback

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 63, in _run_stage
    value = fn()
            ^^^^
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 110, in <lambda>
    lambda: _import_runtime(),
            ^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 239, in _import_runtime
    import jax
ModuleNotFoundError: No module named 'jax'
```

### Rough Runtime Traceback

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 63, in _run_stage
    value = fn()
            ^^^^
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 110, in <lambda>
    lambda: _import_runtime(),
            ^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 239, in _import_runtime
    import jax
ModuleNotFoundError: No module named 'jax'
```

## Diagnosis
- Phase 1 can proceed statically. Runtime reset/step schema remains blocked by missing menagerie assets, while JAX/MuJoCo/Brax imports are available through `.venv`.

## Next Proposed Fix
- Continue Route A and Route B static implementation.
- Record env reset/step as `BLOCKED_BY_DEPENDENCY` until assets are available or a validation host provides them.
