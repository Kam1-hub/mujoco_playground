# Agent Memory: G1 SAC Integration

## User Goal

The user wants a new isolated project at `D:\mujoco_playground\g1_sac_dev` where Codex can implement and validate SAC for the local G1 MuJoCo Playground pipeline. The intended downstream agent should work autonomously, use parallelism, self-plan, self-validate, and keep phase reports for user review.

## Current Workspace State

- New project path: `D:\mujoco_playground\g1_sac_dev`
- Source template path: `D:\mujoco_playground\template`
- LIFT reference repo path: `D:\mujoco_playground\LIFT-humanoid`
- LIFT audit report copy: `external_references/LIFT-humanoid-reports/`
- Git baseline commit: `f4d23d5 baseline g1 template before sac integration`
- Current branch: `sac-integration`

The original `template` was not a git repo. This workspace was initialized as a new repo from that template, then switched to `sac-integration`.

## Verified Local G1 Pipeline Facts

- `pyproject.toml` defines package `g1-training-env` and includes `g1_env` and `learning` packages (`pyproject.toml:6`, `pyproject.toml:78`, `pyproject.toml:79`).
- Existing entry points are PPO/RSL only: `train-g1-jax` and `train-g1-rsl` (`pyproject.toml:81`, `pyproject.toml:82`, `pyproject.toml:83`).
- Dependencies include Brax, JAX, MuJoCo/MJX, Warp, and Orbax (`pyproject.toml:24`, `pyproject.toml:26`, `pyproject.toml:29`, `pyproject.toml:33`, `pyproject.toml:35`, `pyproject.toml:36`).
- Registry exposes G1 envs through `g1_env._src.registry` (`g1_env/_src/registry.py:30`, `g1_env/_src/registry.py:31`, `g1_env/_src/registry.py:39`, `g1_env/_src/registry.py:45`).
- Locomotion registry includes exactly the two known env names at source level: `G1JoystickFlatTerrain`, `G1JoystickRoughTerrain` (`g1_env/_src/locomotion/__init__.py:29`, `g1_env/_src/locomotion/__init__.py:35`).
- G1 default config sets `ctrl_dt=0.02`, `sim_dt=0.002`, `episode_length=1000`, `action_repeat=1`, `action_scale=0.5` (`g1_env/_src/locomotion/g1/joystick.py:32`, `g1_env/_src/locomotion/g1/joystick.py:38`).
- Env action application is `motor_targets = default_pose + action * action_scale` (`g1_env/_src/locomotion/g1/joystick.py:363`).
- `action_size` is `self._mjx_model.nu`; expected G1 action dimension is 29 by local usage docs and source comments (`g1_env/_src/locomotion/g1/base.py:110`, `g1_env/_src/locomotion/g1/base.py:112`, `USAGE.md:82`, `USAGE.md:83`).
- Observation dict is returned with keys `state` and `privileged_state` (`g1_env/_src/locomotion/g1/joystick.py:528`, `g1_env/_src/locomotion/g1/joystick.py:530`).
- Source composition gives actor state dimensions: 3 + 3 + 3 + 3 + 29 + 29 + 29 + phase, matching the usage note of about 101 dims (`g1_env/_src/locomotion/g1/joystick.py:496`, `g1_env/_src/locomotion/g1/joystick.py:505`, `USAGE.md:82`).
- PPO config already uses asymmetric actor/critic keys for G1: `policy_obs_key="state"`, `value_obs_key="privileged_state"` (`g1_env/config/locomotion_params.py:59`, `g1_env/config/locomotion_params.py:64`).
- RSL wrapper detects dict obs and uses `state` / `privileged_state` (`g1_env/_src/wrapper_torch.py:133`, `g1_env/_src/wrapper_torch.py:145`, `g1_env/_src/wrapper_torch.py:164`, `g1_env/_src/wrapper_torch.py:166`).
- Domain randomizer exists for both G1 envs (`g1_env/_src/locomotion/__init__.py:43`, `g1_env/_src/locomotion/__init__.py:46`) and returns `(model, in_axes)` (`g1_env/_src/locomotion/g1/randomize.py:83`, `g1_env/_src/locomotion/g1/randomize.py:100`).

## LIFT Reference Findings To Use

Copied reports are in `external_references/LIFT-humanoid-reports/`.

Use these ideas:

- High-throughput SAC with `grad_updates_per_step` / UTD.
- Actor uses `obs["state"]`; critic uses `obs["privileged_state"]`.
- Tanh Gaussian actor with log-prob correction.
- Action scaling must be paired with log-prob correction.
- Uniform replay buffer, target Q Polyak update, trainable entropy alpha.
- Truncation/timeout handling should be explicit.
- Deterministic evaluation and checkpoint protocol.

Do not inherit these LIFT traps:

- Ambiguous `--learning_rate` that does not update actor/critic/alpha learning rates.
- `--value_hidden_layer_sizes` writing the wrong key instead of `q_hidden_layer_sizes`.
- CLI default `value_obs_key=state` overriding asymmetric critic config.
- `robot_config` being `None` while losses dereference `policy_output_scale`.
- Whole local forks shadowing upstream packages.
- World-model file-wait logic and single-device fine-tune assumptions.

## Environment Risk

Current observed host during preparation:

- Windows, Python 3.12.11.
- JAX/Torch/MuJoCo were not importable in the preparation environment.
- JAX native Windows GPU should not be assumed. Official JAX docs indicate Windows x86_64 CPU support and no native Windows NVIDIA GPU support, while WSL2 NVIDIA GPU is experimental.

Official references:

- JAX installation supported platforms: https://docs.jax.dev/en/latest/installation.html
- JAX build/developer CUDA note: https://docs.jax.dev/en/latest/developer.html

