# Repo Card

## Classification

This repository is a full research pipeline for humanoid locomotion RL, with local simulator/environment forks and custom JAX/Brax-style learners. It is not a clean standalone algorithm library and should not be merged wholesale.

| Role | Assessment | Evidence |
|---|---|---|
| Simulator | Partial | The local Playground fork exposes `MjxEnv` with `mj_model`, `mjx_model`, reset/step/render methods (`mujoco_playground/mujoco_playground/_src/mjx_env.py:210`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:245`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:302`). |
| Environment library | Yes | Registry combines DM Control, locomotion, and manipulation suites (`mujoco_playground/mujoco_playground/_src/registry.py:32`, `mujoco_playground/mujoco_playground/_src/registry.py:34`, `mujoco_playground/mujoco_playground/_src/registry.py:35`, `mujoco_playground/mujoco_playground/_src/registry.py:36`). |
| Learner | Yes | Main SAC learner is `policy_pretrain/train.py:139`; model-based fine-tuning learner is `world_model/finetune_wm_ac.py:54`. |
| Algorithm library | Partial | SAC and WM fine-tuning are wired; PPO/MAPPO/IRL/DIAYN exist mainly as vendored experimental code (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:154`, `brax_env/brax/v1/experimental/composer/training/mappo.py:169`). |
| Experiment manager | Partial | Top-level scripts create logdirs, policy dirs, video dirs, W&B/TB hooks, Optuna studies (`train_in_mujoco_playground.py:341`, `train_in_mujoco_playground.py:344`, `train_in_mujoco_playground_optuna.py:410`, `train_in_mujoco_playground_optuna.py:419`). |
| Deployment tool | Partial | Torch/RSL wrapper bridges JAX envs to Torch via DLPack (`mujoco_playground/mujoco_playground/_src/wrapper_torch.py:35`, `mujoco_playground/mujoco_playground/_src/wrapper_torch.py:42`, `mujoco_playground/mujoco_playground/_src/wrapper_torch.py:75`). |
| Full research pipeline | Yes | README describes SAC policy pretraining, WM pretraining from SAC data, and fine-tuning/sim2sim (`README.md:103`, `README.md:104`, `README.md:105`, `README.md:106`, `README.md:107`). |

## Backend / Platform

| Backend / Platform | Status | Evidence |
|---|---|---|
| JAX | Primary | Top-level package depends on JAX/JAXlib (`setup.py:21`, `setup.py:22`); main scripts set JAX/XLA flags (`train_in_mujoco_playground.py:16`, `train_in_mujoco_playground.py:23`, `train_in_mujoco_playground.py:24`). |
| PyTorch | Secondary | Requirements pin Torch packages (`requirements.txt:16`, `requirements.txt:17`, `requirements.txt:18`); Torch bridge is used for RSL-style wrapper (`mujoco_playground/mujoco_playground/_src/wrapper_torch.py:27`, `mujoco_playground/mujoco_playground/_src/wrapper_torch.py:30`). |
| TensorFlow | Not found | UNKNOWN: no TensorFlow learner/backend path found in scanned source. |
| CUDA | Expected, not validated | Requirements pin `jax[cuda12]` and CUDA packages (`requirements.txt:19`, `requirements.txt:20`, `requirements.txt:33`); current scan environment has no `nvidia-smi` or `nvcc`. |
| Warp / MuJoCo Warp | Not found | UNKNOWN: no Warp backend path found in scanned training code. |
| C++ | Not a primary backend | MuJoCo is used through Python/JAX APIs; no custom C++ learner path found. |
| MuJoCo / MJX | Primary | Registry imports `mujoco.mjx`; env step uses `mjx.step` inside `jax.lax.scan` (`mujoco_playground/mujoco_playground/_src/registry.py:20`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:155`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:166`). |
| Brax-style env/learner | Primary | Training wraps envs via Brax-style wrappers and uses pmap/scans (`policy_pretrain/train.py:223`, `policy_pretrain/train.py:520`). |
| Gymnasium | Partial/risky | `requirements.txt` lists Gymnasium (`requirements.txt:1`), but local wrapper imports legacy `gym` (`brax_env/brax/envs/wrappers/gym.py:20`, `brax_env/brax/envs/wrappers/gym.py:27`). |
| dm_control | Env suite present | Registry includes `dm_control_suite` (`mujoco_playground/mujoco_playground/_src/registry.py:22`, `mujoco_playground/mujoco_playground/_src/registry.py:34`). |

## Device Residency And Parallelism

The main SAC path is mostly device-resident: replay storage, training state, and update epochs are JAX data structures and `pmap`/`scan` based (`policy_pretrain/train.py:299`, `policy_pretrain/train.py:440`, `policy_pretrain/train.py:449`, `policy_pretrain/train.py:520`). CPU/host round trips still exist for evaluation aggregation, checkpoint writes, and rendering (`brax_env/brax_env_utils.py:131`, `brax_env/brax_env_utils.py:132`, `train_in_mujoco_playground.py:414`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:315`).

Parallel support:

- `vmap`: MuJoCo Playground training wrapper uses VmapWrapper or domain-randomization vmap wrapper (`mujoco_playground/mujoco_playground/_src/wrapper.py:155`, `mujoco_playground/mujoco_playground/_src/wrapper.py:158`).
- `scan`: MJX physics stepping and learner updates use `jax.lax.scan` (`mujoco_playground/mujoco_playground/_src/mjx_env.py:166`, `policy_pretrain/train.py:449`).
- `pmap`: SAC training pmap exists (`policy_pretrain/train.py:520`).
- Fine-tuning has a single-local-device assertion for env rollouts (`world_model/finetune_wm_ac.py:1450`).

## Observation / Critic / Vision Support

| Feature | Status | Evidence |
|---|---|---|
| Dict observations | Supported but schema-specific | `Observation` can be a mapping (`mujoco_playground/mujoco_playground/_src/mjx_env.py:111`), but SAC dummy obs hardcodes `state`, `privileged_state`, `wm_state` (`policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`, `policy_pretrain/train.py:282`). |
| `privileged_state` / asymmetric actor-critic | Supported in SAC | Config sets actor key `state` and critic key `privileged_state` (`lift_configs.py:28`, `lift_configs.py:29`); networks select separate keys (`policy_pretrain/sac_networks.py:151`, `policy_pretrain/sac_networks.py:164`). |
| Vision | Not implemented in top-level SAC | `_VISION` raises `ValueError("not implement")` (`train_in_mujoco_playground.py:218`, `train_in_mujoco_playground.py:219`). |

## Integration Posture

Best use for your MuJoCo Playground + JAX/Brax pipeline:

- Directly reference SAC config and high-throughput update structure.
- Cherry-pick action scaling/log-prob correction only with the matching `robot_config.policy_output_scale` contract.
- Keep `mujoco_playground/` and `brax_env/` isolated unless you intentionally want local forks to shadow upstream packages.
- Keep world-model fine-tuning isolated until hardcoded env names and single-device assumptions are patched.

