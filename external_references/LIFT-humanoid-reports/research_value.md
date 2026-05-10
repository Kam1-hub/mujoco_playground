# Research Value

## Main Value For Robotics RL / Embodied AI

This repo is valuable as a high-throughput humanoid SAC and world-model fine-tuning reference, especially for locomotion tasks with asymmetric actor-critic observations.

Key reasons:

- It targets MuJoCo Playground/MJX envs through a JAX/Brax-style learner (`train_in_mujoco_playground.py:220`, `train_in_mujoco_playground.py:317`, `policy_pretrain/train.py:520`).
- It uses actor `state` and critic `privileged_state` by default (`lift_configs.py:28`, `lift_configs.py:29`, `policy_pretrain/sac_networks.py:151`, `policy_pretrain/sac_networks.py:164`).
- It emphasizes high-UTD SAC, with locomotion configs using `grad_updates_per_step` up to 16/17/19 (`lift_configs.py:56`, `lift_configs.py:88`, `lift_configs.py:107`).
- It provides a complete SAC -> replay data -> world model -> model-based fine-tune pipeline (`README.md:103`, `README.md:104`, `README.md:105`, `train_wm_from_file.py:167`, `finetune.py:900`).

## Candidate Baselines

| Baseline | Usefulness | Evidence |
|---|---|---|
| High-throughput SAC locomotion | Strong | SAC train entry (`policy_pretrain/train.py:139`), tuned config (`lift_configs.py:6`), UTD scan (`policy_pretrain/train.py:440`). |
| Asymmetric SAC | Strong | Actor/critic obs key split (`lift_configs.py:28`, `lift_configs.py:29`, `policy_pretrain/sac_networks.py:116`, `policy_pretrain/sac_networks.py:117`). |
| Domain randomized SAC | Medium/strong | CLI flag and registry randomizer (`train_in_mujoco_playground.py:92`, `train_in_mujoco_playground.py:290`, `mujoco_playground/mujoco_playground/_src/registry.py:66`). |
| WM fine-tuned SAC | Research baseline, keep isolated | WM config and fine-tune train path (`lift_configs.py:116`, `lift_configs.py:143`, `world_model/finetune_wm_ac.py:54`). |
| PPO | Config reference only | Playground PPO configs exist (`mujoco_playground/mujoco_playground/config/locomotion_params.py:133`), but LIFT top-level PPO runner was not found. |

## Ablation Ideas

- UTD ablation: compare `grad_updates_per_step` 1, 2, 4, 8, 16, 19 using existing SAC scan structure (`policy_pretrain/train.py:440`, `policy_pretrain/train.py:449`).
- Critic information ablation: `state` critic vs `privileged_state` critic by changing `value_obs_key` (`lift_configs.py:29`, `policy_pretrain/sac_networks.py:164`).
- Domain randomization ablation: same seed/config with and without registry randomizer (`train_in_mujoco_playground.py:290`, `mujoco_playground/mujoco_playground/_src/wrapper.py:158`).
- Action scaling/log-prob ablation: compare objective with and without action-scale correction; current code corrects log-prob (`policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:49`).
- WM rollout horizon ablation: horizon schedule from 1 to 20 (`lift_configs.py:233`, `lift_configs.py:237`).
- Real/model ratio ablation in fine-tuning: `real_ratio=0.06` (`lift_configs.py:210`).
- Probabilistic vs deterministic world model: `model_probabilistic=True` (`lift_configs.py:130`, `world_model/wm_networks.py:95`).

## Suitable For Reproduction

Good candidates:

- SAC pretraining on low-dimensional T1/G1 locomotion tasks using `train_in_mujoco_playground.py`.
- Buffer collection and WM pretraining after SAC is stable.
- Fine-tune experiments only after hardcoded env assumptions are patched or accepted.

Evidence:

- README provides SAC pretrain commands (`README.md:121`, `README.md:128`).
- WM pretrain requires a `buffer_data` directory (`train_wm_from_file.py:167`, `train_wm_from_file.py:171`).
- Fine-tune requires pretrain policy and WM paths (`finetune.py:168`, `finetune.py:169`, `finetune.py:881`, `finetune.py:891`).

## Suitable For Secondary Innovation

- Generalized asymmetric observation adapters for MuJoCo Playground.
- High-throughput SAC hyperparameter study across locomotion/manipulation.
- WM uncertainty and hallucinated rollout scheduling.
- Sim2real protocol analysis: deterministic real-env actions plus stochastic model exploration is described by the README (`README.md:105`, `README.md:106`, `README.md:107`).

## Complementarity With MuJoCo Playground + Brax Learner Pipeline

This repo complements an upstream MuJoCo Playground + Brax learner pipeline by providing:

- A tuned high-UTD SAC variant for humanoid locomotion.
- Concrete asymmetric critic conventions.
- Replay buffer export and WM fine-tuning protocol.
- Domain randomization examples for humanoid tasks.

It does not replace upstream learners cleanly because:

- Local package forks can shadow upstream modules (`mujoco_playground/pyproject.toml:67`, `brax_env/setup.py:31`).
- PPO is not wired as an active top-level LIFT baseline.
- Vision is not implemented in the active SAC path.
- Fine-tuning is single-device constrained and env-specific.

