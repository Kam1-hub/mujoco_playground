# WSL2 Codex Handoff For G1 SAC Integration

This is the continuation document for starting a new Codex CLI session inside WSL2/Linux CUDA. Read this file before running GPU validation.

## Workspace Identity

- Windows source workspace: `D:\mujoco_playground\g1_sac_dev`
- Expected WSL2 target workspace: a Linux filesystem path such as `~/mujoco_playground/g1_sac_dev`
- Main branch: `sac-integration`
- Latest completed work before this handoff: `d73d45c Prepare GPU migration validation scripts`
- Current mission: validate the Route B asymmetric SAC baseline on a GPU-capable WSL2/Linux machine.
- Do not continue large training on the Windows CPU dev machine.

## Mandatory First Reads For A New Codex CLI Session

Codex CLI should read `AGENTS.md` automatically. After that, read these files in order:

1. `WSL2_CODEX_HANDOFF.md`
2. `reports/sac_integration/07_gpu_migration_prep.md`
3. `reports/sac_integration/06_next_actions.md`
4. `reports/sac_integration/04_smoke_results.md`
5. `reports/sac_integration/03_route_b_lift_sac.md`
6. `AGENT_MEMORY.md`
7. `SAC_INTEGRATION_MASTER_PLAN.md`

Use local command output and repo files as the source of truth. Do not rely on memory.

## Current Implementation State

- Existing PPO entry point: `train-g1-jax`
- Existing RSL-RL entry point: `train-g1-rsl`
- SAC fallback entry point: `train-g1-sac-brax`
- SAC main entry point: `train-g1-sac`
- Route B main module: `learning.train_jax_sac_lift`
- Route B train loop: `learning/sac_lift/train.py`
- Route B defaults:
  - `impl="jax"`
  - actor obs key: `state`
  - critic obs key: `privileged_state`
  - action size: 29
  - policy obs shape: `(103,)`
  - value obs shape: `(216,)`
- Action handling:
  - Actor outputs tanh-normalized actions.
  - Actions are passed directly to the env.
  - G1 env applies `action_scale` internally.
  - Do not add an external action scale multiplier unless the action objective is deliberately redesigned and documented.
- Truncation handling:
  - Runtime env did not expose `state.info["truncation"]`.
  - Route B synthesizes zero truncation when missing.
  - Loss/reporting records `truncation_fraction`.

## Completed Validation On Windows CPU Dev

Runtime:

- Python: `3.14.3`
- JAX: `0.10.0`
- JAX backend/devices: CPU only, `cpu:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- CUDA: not available on this host

Passing checks:

- `compileall g1_env learning scripts`
- `registry.ALL_ENVS`
- flat env load/reset/step with `--impl jax`
- rough env load/reset/step with `--impl jax`
- Route A help
- Route B help
- Route B dry run
- Route B CPU tiny smoke

CPU tiny smoke result:

- Env: `G1JoystickFlatTerrain`
- Env steps: 256
- Gradient steps: 121
- Actor loss: `-1.6421515941619873`
- Critic loss: `0.02648034505546093`
- Alpha: `0.04801943153142929`
- SPS: `4.8985466406301095`
- Checkpoint path on Windows dev: `./logs/sac_lift_cpu_tiny/sac_lift_step_256.pkl`
- Checkpoint is an ignored runtime artifact and does not need to be migrated.

## Menagerie Asset State

Do not commit `mujoco_menagerie`.

Required target path inside the repo:

```text
g1_env/external_deps/mujoco_menagerie
```

Required source and commit:

```text
https://github.com/deepmind/mujoco_menagerie.git
1b86ece576591213e2b666ebf59508454200ca97
```

On Windows dev, the asset directory exists and is ignored. On WSL2/Linux, either copy it intentionally or clone it again. Re-cloning on WSL2 is preferred if network is available.

## Files Added For Migration

- `scripts/check_sac_checkpoint.py`
  - Reads a Route B SAC pickle checkpoint.
  - Fails if required keys/metrics are missing.
- `scripts/gpu_preflight.py`
  - Does not run training.
  - Checks JAX/MuJoCo/Brax, GPU visibility when requested, menagerie commit, registry, flat/rough env reset/step, and Route B CLI import.
- `scripts/gpu_smoke_route_b.sh`
  - Linux/WSL2 wrapper for GPU preflight plus Route B 10k smoke.
- `scripts/gpu_smoke_route_b.ps1`
  - PowerShell wrapper for Windows CUDA or orchestration.
  - WSL2/Linux CUDA remains the recommended path.
- `reports/sac_integration/07_gpu_migration_prep.md`
  - Full migration prep report.

## Hard Constraints For The WSL2 Continuation

- Do not modify `D:\mujoco_playground\template`.
- Do not modify `D:\mujoco_playground\LIFT-humanoid`.
- Do not wholesale copy LIFT forks.
- Do not submit `g1_env/external_deps/mujoco_menagerie`.
- Do not submit `.venv`, `logs`, checkpoints, or generated caches.
- Do not run 1M training before 10k GPU smoke passes.
- Do not enable domain randomization for the first GPU smoke.
- Do not do world model, fine-tuning, vision SAC, or real deployment.
- Preserve full stdout/stderr/traceback for failures.
- Classify failures as dependency, CUDA/JAX backend, menagerie, env load, obs/action shape, replay/update, or checkpoint.

## WSL2 Migration Procedure

Recommended approach: clone from the Windows repo into the WSL2 Linux filesystem, then rebuild environment and assets.

From WSL2:

```bash
mkdir -p ~/mujoco_playground
cd ~/mujoco_playground
git clone /mnt/d/mujoco_playground/g1_sac_dev g1_sac_dev
cd g1_sac_dev
git checkout sac-integration
git log --oneline -5
```

If the Windows repo has a remote instead, using the remote is also fine:

```bash
git clone <your-remote-url> g1_sac_dev
cd g1_sac_dev
git checkout sac-integration
```

Do not copy the Windows `.venv`.

Create a fresh Linux Python environment. If this repo's `uv` setup is available:

```bash
uv sync
```

Otherwise create a venv and install the repo requirements according to the project environment files available on the target machine.

Clone menagerie inside the WSL2 repo:

```bash
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
```

Verify that ignored runtime artifacts stay ignored:

```bash
git check-ignore -v g1_env/external_deps/mujoco_menagerie logs .venv
git status --short
```

## WSL2 GPU Validation Ladder

First verify CUDA/JAX visibility:

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
nvidia-smi
```

Then run preflight:

```bash
python scripts/gpu_preflight.py --impl jax --require_gpu
```

Only if preflight passes, run the 10k Route B smoke:

```bash
bash scripts/gpu_smoke_route_b.sh
```

The wrapper runs:

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

## GPU Smoke Acceptance Criteria

- JAX backend/devices show GPU/CUDA/ROCm.
- `python scripts/gpu_preflight.py --impl jax --require_gpu` exits 0.
- Flat env load/reset/step passes.
- Rough env load/reset/step passes.
- Route B 10k smoke reports `TRAIN_OK`.
- Checkpoint is saved under `./logs/sac_lift_gpu_10k`.
- Actor loss, critic loss, alpha, and SPS are recorded.
- No NaN in reported scalar metrics.

## Recommended Prompt For WSL2 Codex CLI

Paste this into the new WSL2 Codex CLI session after opening the repo:

```text
You are continuing the G1 SAC integration in this WSL2/Linux CUDA workspace.
First read AGENTS.md and WSL2_CODEX_HANDOFF.md, then reports/sac_integration/07_gpu_migration_prep.md and 06_next_actions.md.
Do not run 1M training. Do not modify template or LIFT-humanoid. Do not commit menagerie/logs/.venv/checkpoints.
Goal: run GPU preflight, then if and only if it passes, run Route B 10k GPU smoke via scripts/gpu_smoke_route_b.sh.
Record all commands, pass/fail, metrics, checkpoint path, and full tracebacks in reports/sac_integration/.
```
