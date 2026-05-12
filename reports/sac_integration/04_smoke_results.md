# Smoke Results

Status: updated on 2026-05-12 after Route B GPU 50k sanity and bounded eval.

## 2026-05-12 WSL2 GPU Result

Workspace:

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Latest committed baseline before 50k report update:
  `5be043c Add SAC deterministic eval smoke`

Runtime:

- Python: `3.12.3`
- JAX: `0.10.0`
- JAX backend/devices: `gpu`, `cuda:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB class
- Driver: `591.74`
- CUDA reported by `nvidia-smi`: `13.1`
- `nvcc`: not installed; not a blocker for WSL2 JAX CUDA plugin runtime

Asset state:

- Menagerie path: `g1_env/external_deps/mujoco_menagerie`
- Menagerie commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Menagerie, `.venv`, `logs`, and checkpoints remain ignored runtime artifacts.

GPU preflight:

- Command class: `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu`
- Status: `PASS`
- Stages: imports PASS, flat env PASS, rough env PASS
- Runtime summary: `has_gpu=true`, `jax_backend=gpu`, `jax_devices=["cuda:0"]`
- Flat/Rough env schema: action size `29`, `state (103,)`, `privileged_state (216,)`, no native truncation key

Route B GPU 10k smoke:

- Command wrapper: `bash scripts/gpu_smoke_route_b.sh`
- Status: `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- Env steps: `9984`
- Gradient steps: `142`
- Wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- Actor loss: `-1.9058758020401`
- Critic loss: `0.08382290601730347`
- Alpha loss: `1.585930585861206`
- Alpha: `0.04770537465810776`
- Policy log prob: `-18.79882049560547`
- Policy Q: `1.0090709924697876`
- Q: `1.0673823356628418`
- Target Q: `1.0646085739135742`
- Truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics

Route B deterministic eval smoke:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- Eval command scale: `num_eval_envs=4`, `episode_length=200`
- Status: `EVAL_OK`
- Eval env steps: `800`
- Episode reward mean/std/min/max:
  `-3.3628087043762207` / `0.34718504548072815` /
  `-3.778578281402588` / `-2.876215934753418`
- Done fraction: `1.0`
- Wall time: `62.32472045900067`
- SPS: `12.835998205981001`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- Scope note: this is a small deterministic eval smoke, not a full benchmark.

Route B GPU 50k sanity:

- Command class: `uv run --no-sync python -m learning.train_jax_sac_lift`
- Status: `TRAIN_OK`
- Logdir: `./logs/sac_lift_gpu_50k_sanity`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- Requested timesteps: `50000`
- Actual env steps: `49920`
- Gradient steps: `766`
- Wall time: `35.99766752999858`
- SPS: `1386.756515777009`
- Actor loss: `-3.6135072708129883`
- Critic loss: `0.07097882032394409`
- Alpha loss: `1.327394962310791`
- Alpha: `0.03992176800966263`
- Policy log prob: `-18.81831169128418`
- Policy Q: `2.8622469902038574`
- Q: `2.884032726287842`
- Target Q: `2.899707317352295`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error: none observed

Route B 50k bounded deterministic eval:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- Status: `EVAL_OK`
- Eval command scale: `num_eval_envs=16`, `episode_length=1000`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.5016322135925293` / `0.75983726978302` /
  `-6.096090316772461` / `-2.531925916671753`
- Done fraction: `1.0`
- Wall time: `74.76505397899746`
- SPS: `214.00372431342888`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- Scope note: this is a bounded eval attached to 50k sanity, not a full
  benchmark.

Step count note:

- The command requested `num_timesteps=10000` with `num_envs=128`.
- Route B currently computes actual env steps as `num_envs * (num_timesteps // num_envs)`.
- Therefore this smoke records `128 * (10000 // 128) = 9984` actual env steps.

Warnings observed:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- CUDA timer warmup warning: `Delay kernel timed out: measured time has sub-optimal accuracy`.
- These warnings did not fail preflight, 10k smoke, 50k sanity, or bounded eval;
  they remain non-fatal WSL2/JAX noise unless accompanied by a failed command.

Post-smoke `nvidia-smi` summary:

- Time: 2026-05-12 09:12:51
- VRAM: `1517MiB / 12282MiB`
- GPU util: `10%`
- Temperature: `56C`
- Process table only showed `/Xwayland`.

Post-50k `nvidia-smi` summary:

- Time: 2026-05-12 10:58:20
- GPU: RTX 4070 SUPER
- VRAM: `1602MiB / 12282MiB`
- GPU util: `8%`
- Temperature: `56C`
- Power: `9W / 220W`
- Process table only showed `/Xwayland`.

Not run:

- 50k sanity: `PASS`
- 100k sanity and 1M training: `NOT VALIDATED`
- deterministic eval smoke and 50k bounded eval: `PASS`; full eval benchmark:
  `NOT VALIDATED`
- PPO comparison: `NOT VALIDATED`
- domain randomization, fine-tuning, reward/action_scale/Kp tuning: not run
- No logs, checkpoints, `.venv`, or menagerie assets are committed.

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
| Phase GPU | WSL2 `nvidia-smi` | PASS | RTX 4070 SUPER visible in WSL2; post-smoke VRAM `1517MiB / 12282MiB`. |
| Phase GPU | `nvcc --version` | NOT AVAILABLE_NON_BLOCKING | `nvcc` not installed; WSL2 uses JAX CUDA plugin/runtime wheels. |
| Phase GPU | `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu` | PASS | JAX backend `gpu`, device `cuda:0`; flat/rough env reset/step passed. |
| Phase GPU | Route B GPU 10k smoke wrapper | PASS | `TRAIN_OK`; 9984 actual env steps; checkpoint saved under `./logs/sac_lift_gpu_10k`. |
| Phase Eval | `scripts/eval_sac_checkpoint.py --num_eval_envs 4 --episode_length 200` | PASS | `EVAL_OK`; 800 eval env steps; JSON saved under `./logs/sac_eval_smoke`; no action/reward/obs NaN. |
| Phase Sanity | Route B GPU 50k sanity command | PASS | `TRAIN_OK`; 49920 actual env steps; checkpoint saved under `./logs/sac_lift_gpu_50k_sanity`; no observed NaN/Inf/OOM/CUDA error. |
| Phase Sanity Eval | `scripts/eval_sac_checkpoint.py --num_eval_envs 16 --episode_length 1000` | PASS | `EVAL_OK`; 16000 eval env steps; JSON saved under `./logs/sac_eval_50k`; no action/reward/obs NaN. |

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

GPU 10k smoke:

- status: `TRAIN_OK`
- requested timesteps: `10000`
- actual env steps: `9984`
- gradient steps: `142`
- wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- actor loss: `-1.9058758020401`
- critic loss: `0.08382290601730347`
- alpha loss: `1.585930585861206`
- alpha: `0.04770537465810776`
- policy log prob: `-18.79882049560547`
- policy Q: `1.0090709924697876`
- Q: `1.0673823356628418`
- target Q: `1.0646085739135742`
- truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics
- checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- actual step explanation: `128 * (10000 // 128) = 9984`

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

- Windows native GPU smoke remained unavailable.
- WSL2/Linux CUDA later superseded this blocker; GPU preflight, GPU 10k smoke,
  and the small deterministic eval smoke now pass in the WSL2 workspace.
