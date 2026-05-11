# GPU Migration Prep

Status: prepared on 2026-05-11 for moving Route B SAC validation from CPU dev to WSL2/Linux CUDA.

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
- MuJoCo, Brax, and JAX versions should be recorded in the migration report.
- `g1_env/external_deps/mujoco_menagerie` must exist.
- Menagerie must be checked out to `1b86ece576591213e2b666ebf59508454200ca97`.
- Project `.venv` from this Windows CPU machine should not be copied.
- Assets, logs, checkpoints, and `.venv` must remain untracked.

## Migration Procedure

1. Clone or copy `g1_sac_dev` to the target machine.
2. Rebuild the Python environment on the target; do not copy `.venv`.
3. Clone or copy menagerie to `g1_env/external_deps/mujoco_menagerie`.
4. In menagerie, checkout `1b86ece576591213e2b666ebf59508454200ca97`.
5. Run:

```bash
python scripts/gpu_preflight.py --impl jax --require_gpu
```

6. Run:

```bash
bash scripts/gpu_smoke_route_b.sh
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
