# Smoke Results

Status: updated on 2026-05-11 after Route B static implementation.

## Snapshot

- Project path: `D:\mujoco_playground\g1_sac_dev`
- Current branch: `codex/route-b-lift-sac`
- Current commit at validation start: `74594b3`
- Asset status: `g1_env\external_deps\mujoco_menagerie` is absent.
- Runtime status: `.venv` has JAX/MuJoCo/Brax on CPU; system Python lacks JAX.
- Laptop policy: development/static validation host, not a simulation/training host.

## Validation Ladder

| Level | Command | Status | Notes |
|---|---|---:|---|
| 0 | `python -m compileall g1_env learning` | PASS | Static compile passed with system Python. |
| 0 | `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | Static compile passed with `.venv`. |
| 0 | `python -c "import g1_env; print(g1_env.registry.ALL_ENVS)"` | BLOCKED_BY_DEPENDENCY | System Python cannot import JAX; traceback below. |
| 0 | `.\.venv\Scripts\python.exe -c "import g1_env; print(g1_env.registry.ALL_ENVS)"` | PASS | Printed `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`. |
| 0 | `python -m learning.train_jax_sac_brax --help` | PASS | Route A help passed earlier. |
| 0 | `python -m learning.train_jax_sac_lift --help` | PASS | Route B help works with lazy heavy imports. |
| 0 | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help` | PASS | Route B help passed in `.venv`. |
| 1 | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | BLOCKED_BY_DEPENDENCY | Imports/config pass; `registry.load` blocked by missing menagerie assets. |
| 1 | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | BLOCKED_BY_DEPENDENCY | Same blocker as flat terrain. |
| 2 | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --dry_run --logdir ./logs/sac_lift_dry_run` | PASS | `DRY_RUN_OK`; ran one dummy SAC update and saved checkpoint under ignored `logs/`. |
| 2 | `python -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 1 --num_envs 2 --num_eval_envs 2 --batch_size 2 --min_replay_size 2 --max_replay_size 16 --grad_updates_per_step 1 --dry_run` | BLOCKED_BY_DEPENDENCY | System Python lacks JAX; use `.venv\Scripts\python.exe`. |
| 2 | Same Route B command without `--dry_run` and `--logdir ./logs/sac_lift_blocked_probe` | BLOCKED_BY_DEPENDENCY | Blocked before simulation by missing menagerie assets. |
| 3 | CPU tiny smoke with `JAX_PLATFORM_NAME=cpu` and `--num_timesteps 256` | NOT VALIDATED | Not run on this laptop; requires menagerie assets and explicit permission to simulate. |
| 4 | GPU smoke, `--num_timesteps 10000 --num_envs 128` | NOT VALIDATED | Requires WSL2/Linux CUDA JAX or another proven CUDA JAX runtime. |
| 5 | 1M sanity run | NOT VALIDATED | Only run after GPU smoke passes. |

## Shapes

- action_size: dry-run default 29.
- obs type: dict for Route B.
- obs keys: `state`, `privileged_state`.
- state shape: dry-run default `(103,)`.
- privileged_state shape: dry-run default `(216,)`, static and not runtime validated.
- replay transition shape: policy obs `(batch, 103)`, value obs `(batch, 216)`, action `(batch, 29)`. Dry-run batch size was 2.

## Runtime Metrics

- steps: Route B dry-run initialized 0 env steps and ran 1 dummy gradient step; non-dry-run smoke blocked before env load.
- wall time: not measured as a training metric.
- SPS: NOT VALIDATED
- eval reward: NOT VALIDATED
- actor loss: dry-run dummy update `-0.8600597381591797`
- critic loss: dry-run dummy update `0.4279707074165344`
- alpha: dry-run dummy update `0.049787066876888275`
- NaN: no NaN observed in dry-run output; training NaN NOT VALIDATED.

## Tracebacks

### System Python Registry Import

```text
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "D:\mujoco_playground\g1_sac_dev\g1_env\__init__.py", line 11, in <module>
    from g1_env._src import locomotion
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 20, in <module>
    import jax
ModuleNotFoundError: No module named 'jax'
```

### Env API Asset Blocker

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

### Route B Non-Dry-Run Asset Blocker

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\learning\sac_lift\train.py", line 184, in main
    result = train(config)
  File "D:\mujoco_playground\g1_sac_dev\learning\sac_lift\train.py", line 84, in train
    raise RuntimeError(
RuntimeError: mujoco_menagerie is missing at D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie; rerun with --allow_menagerie_download only when network/file writes are authorized.
```

### System Python Route B Dry Run

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_lift.py", line 135, in <module>
    run()
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_lift.py", line 131, in run
    sys.exit(main())
             ^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_lift.py", line 124, in main
    config = _build_config(args)
             ^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\learning\train_jax_sac_lift.py", line 69, in _build_config
    from g1_env.config import sac_params
  File "D:\mujoco_playground\g1_sac_dev\g1_env\__init__.py", line 11, in <module>
    from g1_env._src import locomotion
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 20, in <module>
    import jax
ModuleNotFoundError: No module named 'jax'
```

## Diagnosis

The static/CLI acceptance gate is now passing for Route B. Runtime env validation is blocked by missing menagerie assets, not by Route B import/CLI structure.
