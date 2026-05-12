# GPU Migration Prep

Status: prepared on 2026-05-11 for moving Route B SAC validation from CPU dev to WSL2/Linux CUDA.

## 2026-05-12 WSL2 GPU Validation Result

Target workspace:

```text
/home/admin/projects/mujoco_playground/g1_sac_dev
```

CUDA/JAX setup:

- `uv sync --frozen --extra cuda`: `PASS`
- Python: `3.12.3`
- JAX: `0.10.0`
- JAX backend/devices: `gpu`, `cuda:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB class
- Driver: `591.74`
- CUDA reported by `nvidia-smi`: `13.1`
- `nvcc`: not installed; not a blocker in WSL2 plugin-based runtime

Assets:

- Menagerie path: `g1_env/external_deps/mujoco_menagerie`
- Menagerie commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Menagerie, `.venv`, `logs`, and checkpoints remain ignored.

Preflight:

- Command: `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu`
- Status: `PASS`
- Stages: imports PASS, flat env PASS, rough env PASS
- Runtime summary: `has_gpu=true`, `jax_backend=gpu`, `jax_devices=["cuda:0"]`

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

Step count note:

- The smoke requested `num_timesteps=10000`.
- Route B currently records actual steps as `num_envs * (num_timesteps // num_envs)`.
- With `num_envs=128`, the actual count is `9984`.

Warnings observed:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- CUDA timer warmup warning: `Delay kernel timed out: measured time has sub-optimal accuracy`.
- These were non-fatal for preflight and smoke.

Remaining validation status:

- 1M and longer training: `NOT VALIDATED`
- deterministic eval rollout: `NOT VALIDATED`
- PPO comparison: `NOT VALIDATED`
- domain randomization and fine-tuning: not run

## 2026-05-12 WSL2 Documentation Update

Target WSL2 documentation workspace:

```text
/home/admin/projects/mujoco_playground/g1_sac_dev
```

The local helper paths from earlier prompts that pointed at a Windows-share
scratch location are stale for this session. Use the project-local references
instead:

- `WSL2_GPU_EXPERIENCE.md`
- `/home/admin/projects/mujoco_playground/TRAINING_NOTES.md`
- `/home/admin/projects/mujoco_playground/.md_edit/`

These files are operating references only. They do not override Route B SAC
design, action handling, or the preflight-first validation ladder.

## Repository State

- Project path: `D:\mujoco_playground\g1_sac_dev`
- Branch: `sac-integration`
- Base commit at prep start: `4986032 Validate G1 SAC CPU smoke`
- Git status at prep start: clean tracked tree
- GPU smoke status: `NOT VALIDATED`
- CPU dev policy: do not expand local training; use this machine for code, checks, and migration packaging

## CPU Dev Runtime

- Python: `3.14.3`
- JAX: `0.10.0`
- JAX backend: `cpu`
- JAX devices: `[CpuDevice(id=0)]`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- CUDA tools: `nvidia-smi` and `nvcc` not available in prior probes
- WSL2/Linux CUDA: not validated on this host

## Menagerie State

- Path: `D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie`
- Source URL: `https://github.com/deepmind/mujoco_menagerie.git`
- Required commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Current commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Git ignore: `.gitignore:6:mujoco_menagerie`
- File count: 2176
- Size: 1517813291 bytes
- Submission rule: do not commit menagerie assets

## CPU Tiny Smoke Recap

- Command class: Route B `learning.train_jax_sac_lift` CPU tiny smoke
- Env: `G1JoystickFlatTerrain`
- Impl: `jax`
- Env steps: 256
- Gradient steps: 121
- Checkpoint: `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`
- Actor loss: `-1.6421515941619873`
- Critic loss: `0.02648034505546093`
- Alpha: `0.04801943153142929`
- SPS: `4.8985466406301095`
- NaN: no NaN observed in reported scalar metrics

## Added Migration Tools

- `scripts/check_sac_checkpoint.py`
  - Reads Route B pickle checkpoint.
  - Prints config keys, metrics, and required parameter presence.
  - Fails if checkpoint or key fields are missing.
- `scripts/gpu_preflight.py`
  - Does not run training.
  - Checks JAX/MuJoCo/Brax, menagerie path and commit, registry, train CLI import, and flat/rough env load/reset/step.
  - Supports `--impl`, default `jax`.
  - Supports `--require_gpu` to fail unless JAX sees GPU/CUDA/ROCm.
- `scripts/gpu_smoke_route_b.sh`
  - Linux/WSL bash wrapper for preflight plus 10k Route B smoke.
- `scripts/gpu_smoke_route_b.ps1`
  - PowerShell wrapper for Windows CUDA or outer orchestration.
  - Comments explicitly state Windows native GPU JAX is not validated here and WSL2/Linux CUDA is recommended.

## CPU-Side Validation Results

These commands were run on the CPU dev machine and do not perform GPU smoke or long training.

| Command | Status | Notes |
|---|---:|---|
| `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | New scripts compile. The command also traverses ignored menagerie because it is under `g1_env`. |
| `.\.venv\Scripts\python.exe scripts\check_sac_checkpoint.py --checkpoint .\logs\sac_lift_cpu_tiny\sac_lift_step_256.pkl` | PASS | Read config, metrics, policy/Q/target/log_alpha fields. |
| `.\.venv\Scripts\python.exe scripts\gpu_preflight.py --impl jax` | PASS | JAX CPU backend accepted because `--require_gpu` was not set; flat/rough env load/reset/step passed. |

Checkpoint inspection result:

- top-level keys: `config`, `log_alpha`, `metrics`, `policy_params`, `q_params`, `target_q_params`
- env_name: `G1JoystickFlatTerrain`
- impl: `jax`
- policy_obs_key: `state`
- value_obs_key: `privileged_state`
- metrics status: `TRAIN_OK`
- env_steps: 256
- gradient_steps: 121
- actor_loss: `-1.6421515941619873`
- critic_loss: `0.02648034505546093`
- alpha: `0.04801943153142929`
- sps: `4.8985466406301095`

Preflight result:

- status: `PASS`
- require_gpu: `false`
- JAX backend: `cpu`
- JAX devices: `cpu:0`
- has_gpu: `false`
- menagerie commit match: `true`
- registry envs: `G1JoystickFlatTerrain`, `G1JoystickRoughTerrain`
- Route B CLI module import: `learning.train_jax_sac_lift`, `main=True`, `run=True`
- flat env: action_size 29, obs keys `state` and `privileged_state`, shapes `(103,)` and `(216,)`, no truncation key
- rough env: action_size 29, obs keys `state` and `privileged_state`, shapes `(103,)` and `(216,)`, no truncation key

## Target Machine Requirements

- Linux or WSL2 with CUDA.
- JAX must report GPU/CUDA/ROCm backend or devices.
- `nvcc` is not required in WSL2; `nvidia-smi` and JAX GPU devices are the gate.
- MuJoCo, Brax, and JAX versions should be recorded in the migration report.
- `g1_env/external_deps/mujoco_menagerie` must exist.
- Menagerie must be checked out to `1b86ece576591213e2b666ebf59508454200ca97`.
- Project `.venv` from this Windows CPU machine should not be copied.
- Assets, logs, checkpoints, and `.venv` must remain untracked.

## WSL2 GPU Operational Lessons

Set before Python imports:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Use the repo CUDA extra first:

```bash
uv sync --frozen --extra cuda
```

If JAX still reports CPU, do not train and do not change `pyproject.toml` or
`uv.lock` without a separate decision.

Route B replay budget:

- transition payload: about 671 float32 values
- raw size: about 2.6 to 2.7 KB per transition
- `max_replay_size=8192`: about 22 MB raw
- `max_replay_size=1000000`: about 2.5 to 2.7 GB raw before JAX/XLA overhead

OOM categories to record:

- XLA preallocation
- CPU-only JAX or missing CUDA plugin
- residual GPU processes
- replay capacity
- env batch size
- terrain/domain randomization overhead

Do not use destructive cleanup such as killing GPU processes or restarting WSL
without explicit user approval.

## Migration Procedure

1. Clone or copy `g1_sac_dev` to the target machine.
2. Rebuild the Python environment on the target; do not copy `.venv`.
3. Clone or copy menagerie to `g1_env/external_deps/mujoco_menagerie`.
4. In menagerie, checkout `1b86ece576591213e2b666ebf59508454200ca97`.
5. Run:

```bash
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

6. Stop for user confirmation. Only then run:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

## GPU 10k Smoke Acceptance

- JAX backend/devices show GPU/CUDA/ROCm.
- `scripts/gpu_preflight.py --impl jax --require_gpu` exits 0.
- Flat env load/reset/step passes.
- Rough env load/reset/step passes.
- Route B 10k command reports `TRAIN_OK`.
- Checkpoint is saved under `./logs/sac_lift_gpu_10k`.
- Actor loss, critic loss, and alpha are finite.
- No NaN is reported in metrics.
- SPS is recorded.
- idle and final or peak GPU VRAM are recorded when available.
- actual `env_steps` are recorded; do not assume they equal requested `num_timesteps`.

## Failure Classification

Use these categories in the smoke report:

- dependency
- CUDA/JAX backend
- menagerie
- env load
- obs/action shape
- replay/update
- checkpoint

For every failed command, preserve:

- full command
- exit code
- full traceback/stdout/stderr
- failure category
- diagnosis
- next concrete fix or command

## Prohibited During Migration

- Do not run 1M training before 10k smoke passes.
- Do not enable domain randomization in this migration smoke.
- Do not add world model work.
- Do not merge changes back into `D:\mujoco_playground\template`.
- Do not submit menagerie assets.
- Do not submit logs, checkpoints, or `.venv`.
