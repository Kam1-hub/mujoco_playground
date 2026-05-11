# Smoke Results

Status: updated on 2026-05-11 after menagerie install, env API validation, and CPU tiny SAC smoke.

## Snapshot

- Project path: `D:\mujoco_playground\g1_sac_dev`
- Branch: `sac-integration`
- Validation start commit: `6027358 Add Route B asymmetric SAC baseline`
- Runtime: `.\.venv\Scripts\python.exe`
- JAX: `0.10.0`, backend `cpu`, devices `['cpu:0']`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- Torch: not importable in `.venv`
- CUDA: `nvidia-smi` and `nvcc` not found
- WSL2: `wsl -l -v` returned exit 1; no usable WSL2/Linux CUDA runtime validated
- Training actually run: yes, CPU tiny smoke completed 256 env steps

## Asset Status

- Initial local search under `D:\mujoco_playground\mujoco_playground` and `D:\mujoco_playground` found no existing `mujoco_menagerie` directory.
- User authorized network download.
- Download target: `D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie`
- Source: `https://github.com/deepmind/mujoco_menagerie.git`
- Checked out commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Size probe: 2176 files, 1517813291 bytes
- Git ignore: `.gitignore:6:mujoco_menagerie` ignores `g1_env/external_deps/mujoco_menagerie`
- Asset commit status: not tracked by project git

## Validation Table

| Step | Command | Status | Result |
|---|---|---:|---|
| Phase A | `git status --short` | PASS | Clean before this round. |
| Phase A | `git branch --show-current` | PASS | `sac-integration`. |
| Phase A | `git log --oneline -5` | PASS | HEAD was `6027358`. |
| Phase A | `.venv runtime import probe` | PASS | Python 3.14.3; JAX/MuJoCo/Brax specs present. |
| Phase A | `Test-Path g1_env\external_deps\mujoco_menagerie` | PASS | Initially `False`; now `True`. |
| Phase A | `git check-ignore -v g1_env/external_deps/mujoco_menagerie` | PASS | Ignored by `.gitignore`. |
| Phase B | Local menagerie search under `D:\mujoco_playground` | PASS | No local copy found. |
| Phase B | `git clone https://github.com/deepmind/mujoco_menagerie.git ...` | PASS | Downloaded after explicit user authorization. |
| Phase B | Menagerie checkout `1b86ece...` | PASS | Effective asset commit matches reference. |
| Phase C | `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | Static compile passed. Command also traverses ignored menagerie because it lives under `g1_env`. |
| Phase C | `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"` | PASS | `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`. |
| Phase C | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | PASS | Load/reset/step passed with `--impl jax` default. |
| Phase C | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | PASS | Load/reset/step passed with `--impl jax` default. |
| Phase C | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help` | PASS | Route A entry point help passed. |
| Phase C | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help` | PASS | Route B entry point help passed, including `--impl`. |
| Phase C | Route B dry-run command | PASS | `DRY_RUN_OK`; one dummy SAC update; checkpoint saved. |
| Probe | 16-step non-dry-run before wider JIT | PASS | `TRAIN_OK`; 16 env steps; 7 gradient steps; checkpoint saved. |
| Probe | 16-step non-dry-run after wider JIT | PASS | `TRAIN_OK`; 16 env steps; 7 gradient steps; wall time improved. |
| Phase D | CPU tiny smoke command with `JAX_PLATFORM_NAME=cpu` | PASS | `TRAIN_OK`; 256 env steps; 121 gradient steps; checkpoint saved. |
| Phase D | Earlier interrupted CPU tiny process | INCONCLUSIVE | User interrupted foreground tool; background process later ended without captured exit code or checkpoint. Superseded by later passing run. |
| Phase D | Background `Start-Process` wrapper | FAIL_NON_BLOCKING | PowerShell environment conflict; direct foreground run succeeded. |
| Phase GPU | `nvidia-smi` | NOT AVAILABLE | Command not found. |
| Phase GPU | `nvcc --version` | NOT AVAILABLE | Command not found. |
| Phase GPU | GPU smoke 10k command | NOT VALIDATED | Requires WSL2/Linux CUDA or available CUDA JAX runtime. |

## Env API Schema

Both `G1JoystickFlatTerrain` and `G1JoystickRoughTerrain`:

- registry: `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`
- default config reports `impl="warp"`, but checker passes `config_overrides={"impl": "jax"}` by default for CPU validation
- env type: `Joystick`
- action_size: 29
- dt: 0.02
- sim_dt: 0.002
- n_substeps: 10
- obs type: dict
- obs keys: `state`, `privileged_state`
- `state` shape/dtype: `(103,)`, `float32`
- `privileged_state` shape/dtype: `(216,)`, `float32`
- reset reward/done shape: scalar `float32`
- step reward/done shape: scalar `float32`
- `state.info["truncation"]`: absent
- SAC behavior: missing truncation is synthesized as zeros; CPU tiny observed `truncation_fraction=0.0`

Flat XML:

- `D:/mujoco_playground/g1_sac_dev/g1_env/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml`

Rough XML:

- `D:/mujoco_playground/g1_sac_dev/g1_env/_src/locomotion/g1/xmls/scene_mjx_feetonly_rough_terrain.xml`

## SAC Metrics

Dry run:

- status: `DRY_RUN_OK`
- policy obs shape: `[103]`
- value obs shape: `[216]`
- action shape: `[29]`
- replay capacity: 16
- gradient steps: 1
- actor loss: `-0.8600597381591797`
- critic loss: `0.4279707074165344`
- alpha: `0.049787066876888275`
- checkpoint: `./logs/sac_lift_dry_run\sac_lift_step_0.pkl`

CPU tiny smoke:

- status: `TRAIN_OK`
- env steps: 256
- gradient steps: 121
- wall time: `52.260398599995824`
- SPS: `4.8985466406301095`
- actor loss: `-1.6421515941619873`
- critic loss: `0.02648034505546093`
- alpha loss: `1.5696232318878174`
- alpha: `0.04801943153142929`
- policy log prob: `-17.302722930908203`
- policy Q: `0.8112847805023193`
- Q: `0.8451158404350281`
- target Q: `0.8984131813049316`
- truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics
- checkpoint: `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`

## Tracebacks And Failures

### Initial Env API Failure Before `--impl jax` Patch

Failure category: dependency/runtime backend.

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
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 103, in load
    return _envs[env_name](config=config, config_overrides=config_overrides)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\g1\joystick.py", line 119, in __init__
    super().__init__(
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\g1\base.py", line 63, in __init__
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
                      ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 543, in put_model
    impl, device = _resolve_impl_and_device(impl, device)
                   ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 152, in _resolve_impl_and_device
    device = _resolve_device(impl)
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 100, in _resolve_device
    cuda_gpus = [d for d in jax.devices('cuda')]
                            ~~~~~~~~~~~^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 1010, in devices
    return get_backend(backend).devices()
           ~~~~~~~~~~~^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 944, in get_backend
    return _get_backend_uncached(platform)
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 932, in _get_backend_uncached
    raise RuntimeError(
RuntimeError: Unknown backend cuda. Available backends are ['cpu']
```

Fix applied:

- `scripts/check_g1_env_api.py` now defaults to `--impl jax`.
- `learning.train_jax_sac_lift` now exposes `--impl`.
- `g1_env.config.sac_params.lift_sac_config` now defaults Route B to `impl="jax"`.
- Route B env loading passes `config_overrides={"impl": config.impl}`.

### Interrupted CPU Tiny Attempt

Failure category: inconclusive/user interruption.

```text
The foreground CPU tiny smoke tool call was interrupted by the user after 509.2s.
Process inspection then found two Python processes that had continued running.
They exited during a bounded wait, but no stdout/stderr, exit code, traceback, or
`logs\sac_lift_cpu_tiny` checkpoint was captured from that interrupted attempt.
```

Resolution:

- Added broader JIT in the SAC training loop.
- Reran the exact CPU tiny command successfully; result supersedes this inconclusive attempt.

### Background Wrapper Failure

Failure category: validation harness, non-blocking.

```text
Start-Process : Added item. Key in dictionary: "Path" Key being added: "PATH"
At line:3 char:6
+ $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentLi ...
+      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [Start-Process], ArgumentException
    + FullyQualifiedErrorId : System.ArgumentException,Microsoft.PowerShell.Commands.StartProcessCommand
```

Resolution:

- Did not use background wrapper.
- Ran the exact foreground CPU tiny command and captured passing output.

### GPU Runtime Probes

Failure category: dependency/platform.

```text
nvidia-smi : The term 'nvidia-smi' is not recognized as the name of a cmdlet,
function, script file, or operable program.
```

```text
nvcc : The term 'nvcc' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

Resolution:

- GPU smoke remains `NOT VALIDATED`.
- Next GPU command should run only in WSL2/Linux CUDA or another environment where CUDA JAX is actually visible.
