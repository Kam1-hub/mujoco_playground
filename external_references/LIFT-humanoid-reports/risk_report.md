# Risk Report

## Installation Risk: High

- README assumes Ubuntu 22.04, Python 3.10, and H800/4090 GPUs (`README.md:82`, `README.md:83`, `README.md:84`).
- Current audit machine is Windows with Python 3.12 and no installed JAX/Torch/MuJoCo in the observed Python env.
- The repo expects multiple editable installs: `mujoco_playground`, `brax_env`, and top-level package (`README.md:86`, `README.md:90`, `README.md:94`).

## Dependency Risk: High

- Requirements pin Torch CUDA 12.4 and JAX CUDA 12 (`requirements.txt:16`, `requirements.txt:19`, `requirements.txt:20`, `requirements.txt:33`).
- `mujoco_playground` comments out `brax>=0.12.1` (`mujoco_playground/pyproject.toml:26`).
- `brax_env` uses legacy `gym` (`brax_env/setup.py:42`, `brax_env/setup.py:44`) while top-level requirements include Gymnasium (`requirements.txt:1`).
- Vendored PPO imports `brax.training.*`, which may not exist in modern/local layouts (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:30`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:35`).

## Fork / Shadowing Risk: High

- Local Playground fork packages `mujoco_playground` (`mujoco_playground/pyproject.toml:66`, `mujoco_playground/pyproject.toml:67`).
- Local `brax_env` uses `find_packages()` and contains `brax` code (`brax_env/setup.py:31`, `brax_env/brax_env_utils.py:20`).
- Installing both into a main research environment can shadow upstream packages.

## License Risk: Medium

- Code packages declare Apache 2.0 (`setup.py:15`, `mujoco_playground/README.md:3`, `brax_env/setup.py:30`).
- UNKNOWN: binary robot assets, meshes, deployment model files, and external data/checkpoint redistribution terms were not audited.

## Version Risk: High

- Top-level package says Python `>=3.8` (`setup.py:37`), but local Playground requires Python `>=3.10` (`mujoco_playground/pyproject.toml:12`).
- Requirements pin `jax[cuda12]==0.4.35` (`requirements.txt:19`) while Playground requires `mujoco-mjx>=3.2.7` (`mujoco_playground/pyproject.toml:32`).
- Current audit Python is 3.12.11 and lacks all core runtime packages; run compatibility is UNKNOWN.

## Implementation Transparency Risk: Medium/High

- SAC is adapted local code, not simply upstream Brax SAC (`policy_pretrain/train.py:15`, `policy_pretrain/train.py:139`).
- Action scaling affects policy output, Q input, and log-prob correction (`policy_pretrain/sac_networks.py:55`, `policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:49`).
- Hardcoded dict obs keys are embedded in learner initialization (`policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`, `policy_pretrain/train.py:282`).

## MuJoCo Playground Compatibility Risk: Medium

- The top-level runner is compatible with the included local fork (`train_in_mujoco_playground.py:220`, `train_in_mujoco_playground.py:317`).
- Compatibility with current upstream MuJoCo Playground is UNKNOWN because local fork modifications are present (`mujoco_playground/README.md:3`, `mujoco_playground/README.md:4`).
- Vision is not implemented in the top-level SAC runner (`train_in_mujoco_playground.py:218`, `train_in_mujoco_playground.py:219`).

## Performance Risk: Medium

- SAC training uses device-resident pmap/scan patterns (`policy_pretrain/train.py:520`, `policy_pretrain/train.py:449`).
- Evaluation metrics, rendering, and checkpointing bring data to host (`brax_env/brax_env_utils.py:132`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:315`, `train_in_mujoco_playground.py:414`).
- Fine-tune env rollouts require one local device (`world_model/finetune_wm_ac.py:1450`).

## Algorithm Coverage Risk: Medium

- TD3/REDQ/DroQ/CrossQ/DDPG were not found.
- PPO exists as config/vendored code, not active LIFT top-level baseline (`mujoco_playground/mujoco_playground/config/locomotion_params.py:133`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:154`).
- Optuna CMA-ES is HPO only, not population-based policy optimization (`train_in_mujoco_playground_optuna.py:376`, `train_in_mujoco_playground_optuna.py:380`).

