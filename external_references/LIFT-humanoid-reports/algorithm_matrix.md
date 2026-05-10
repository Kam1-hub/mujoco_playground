# Algorithm Matrix

Status legend:

- Wired: connected to a top-level script.
- Source-only: implementation exists but is not used by the main LIFT scripts.
- Config-only: config exists but no local top-level runner was found.
- Not found: no implementation path found by source scan.
- NOT VALIDATED: not run in this audit environment.

| Algorithm / Family | README claimed | Source exists | Wired top-level | Status | Entry / config | Evidence |
|---|---:|---:|---:|---|---|---|
| SAC | Yes | Yes | Yes | Wired, NOT VALIDATED | `train_in_mujoco_playground.py`; `policy_pretrain/train.py`; `lift_configs.py` | README says policy training uses SAC (`README.md:103`, `README.md:121`); script imports SAC modules (`train_in_mujoco_playground.py:36`, `train_in_mujoco_playground.py:37`); learner entry is `policy_pretrain/train.py:139`; config is `lift_configs.py:6`. |
| SAC Optuna HPO | Yes | Yes | Yes | Wired, NOT VALIDATED | `train_in_mujoco_playground_optuna.py` | Optuna sampler mutates SAC params (`train_in_mujoco_playground_optuna.py:222`, `train_in_mujoco_playground_optuna.py:285`, `train_in_mujoco_playground_optuna.py:419`). |
| World model pretraining | Yes | Yes | Yes | Wired, NOT VALIDATED | `train_wm_from_file.py`; `world_model/pretrain_wm.py` | README describes WM pretraining (`README.md:201`); script requires SAC buffer data (`train_wm_from_file.py:167`, `train_wm_from_file.py:171`); train function is `world_model/pretrain_wm.py:123`. |
| Model-based RL / LIFT fine-tuning | Yes | Yes | Yes | Wired, NOT VALIDATED | `finetune.py`; `world_model/finetune_wm_ac.py` | Fine-tune config begins at `lift_configs.py:143`; top-level script builds model env and calls WM+AC train (`finetune.py:268`, `finetune.py:900`); learner train starts at `world_model/finetune_wm_ac.py:54`. |
| PPO | Config/reference | Yes | No active LIFT runner found | Config-only plus vendored source | `mujoco_playground/.../config/*`; `brax_env/.../ppo.py` | Playground locomotion PPO config exists (`mujoco_playground/mujoco_playground/config/locomotion_params.py:133`); vendored PPO train exists (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:154`) but imports external `brax.training.*` (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:30`). |
| RSL-RL PPO | Config/wrapper only | Partial | No | Config-only/wrapper | `wrapper_torch.py`; Playground configs | RSL wrapper class exists (`mujoco_playground/mujoco_playground/_src/wrapper_torch.py:75`), but `rsl_rl` import is optional and may be missing (`mujoco_playground/mujoco_playground/_src/wrapper_torch.py:23`, `mujoco_playground/mujoco_playground/_src/wrapper_torch.py:26`). |
| MAPPO | Not central | Yes | No | Vendored unused | `brax_env/brax/v1/experimental/composer/training/mappo.py` | MAPPO train function exists (`brax_env/brax/v1/experimental/composer/training/mappo.py:169`) and PPO loss wrapper exists (`brax_env/brax/v1/experimental/composer/training/mappo.py:71`). |
| Imitation / IRL: GAIL, GAIL2, AIRL, FAIRL, MLE | Not central | Yes | No | Vendored unused | `brax_env/brax/v1/experimental/braxlines/irl_smm/` | IRL train entry exists (`brax_env/brax/v1/experimental/braxlines/irl_smm/train.py:39`); reward types are implemented in utility code (`brax_env/brax/v1/experimental/braxlines/irl_smm/utils.py:160`). |
| Unsupervised skill / VGCRL: DIAYN, GCRL, CDIAYN | Not central | Yes | No | Vendored unused | `brax_env/brax/v1/experimental/braxlines/vgcrl/` | Train entry exists (`brax_env/brax/v1/experimental/braxlines/vgcrl/train.py:36`); `algo_name` defaults to DIAYN (`brax_env/brax/v1/experimental/braxlines/vgcrl/train.py:56`); discriminator factory supports several methods (`brax_env/brax/v1/experimental/braxlines/vgcrl/utils.py:330`). |
| Offline RL | Partial | Replay-data based WM pretrain only | Yes for WM | Not a general offline RL algorithm | `train_wm_from_file.py` | Offline data path is SAC replay buffer data, not CQL/IQL/AWAC-style offline RL (`train_wm_from_file.py:167`, `world_model/pretrain_wm.py:499`). |
| TD3 / FastTD3 | Fork note only | UNKNOWN / not found | No | Not found | UNKNOWN | Fork README mentions FastTD3 modifications for custom tasks (`mujoco_playground/README.md:4`), but no TD3 learner/loss source was found in scanned files. |
| REDQ | No | No | No | Not found | UNKNOWN | No REDQ implementation path found. |
| DroQ | No | No | No | Not found | UNKNOWN | No DroQ implementation path found. |
| CrossQ | No | No | No | Not found | UNKNOWN | No CrossQ implementation path found. |
| DDPG | No | No | No | Not found | UNKNOWN | No DDPG implementation path found. |
| Evolutionary / population-based policy optimization | No | No | No | Not found | UNKNOWN | Optuna can use CMA-ES for hyperparameter search (`train_in_mujoco_playground_optuna.py:376`, `train_in_mujoco_playground_optuna.py:380`), but it does not optimize policies directly. |

## SAC Details

| Detail | Implementation | Evidence |
|---|---|---|
| Policy distribution | Default `tanh_normal`; `normal` is also accepted. | `policy_pretrain/sac_networks.py:112`, `policy_pretrain/sac_networks.py:125`, `policy_pretrain/sac_networks.py:130` |
| Tanh correction | Distribution log-prob includes tanh bijector Jacobian. | `lift_utils/distribution.py:148`, `lift_utils/distribution.py:163`, `lift_utils/distribution.py:176` |
| Action scaling | Actor output and Q action are multiplied by `robot_config.policy_output_scale`; log-prob is adjusted by action-scale log determinant. | `policy_pretrain/sac_networks.py:55`, `policy_pretrain/sac_networks.py:65`, `policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:49`, `policy_pretrain/losses.py:95` |
| Log std bounds | `NormalTanhDistribution` bounds log std from `-5.0` to `2.0`. | `lift_utils/distribution.py:151`, `lift_utils/distribution.py:170`, `lift_utils/distribution.py:176` |
| Twin Q | Q network defaults to two critics. | `policy_pretrain/base_networks.py:267`, `policy_pretrain/base_networks.py:305` |
| Target update tau | Polyak target update each update step. | `policy_pretrain/train.py:374`, `policy_pretrain/train.py:377` |
| Entropy alpha | Trainable log alpha with separate optimizer. | `policy_pretrain/train.py:104`, `policy_pretrain/train.py:322`, `policy_pretrain/losses.py:51`, `policy_pretrain/losses.py:68` |
| Target entropy | `-target_entropy_coef * action_size`, optionally scheduled by `entropy_rate`. | `policy_pretrain/train.py:673`, `policy_pretrain/train.py:676` |
| Replay buffer | FIFO uniform sampling queue with replacement. | `policy_pretrain/train.py:299`, `lift_utils/replay_buffers.py:177`, `lift_utils/replay_buffers.py:183`, `lift_utils/replay_buffers.py:197` |
| Min/max replay | `min_replay_size` and `max_replay_size` are train params; config defaults are 8192 and 1,000,000. | `policy_pretrain/train.py:160`, `policy_pretrain/train.py:161`, `lift_configs.py:22`, `lift_configs.py:23` |
| UTD | `grad_updates_per_step`; sampled batch reshaped then scanned. | `policy_pretrain/train.py:162`, `policy_pretrain/train.py:440`, `policy_pretrain/train.py:449` |
| Reward scaling | Critic target uses `reward * reward_scaling`. | `policy_pretrain/losses.py:105` |
| Observation normalization | Optional running stats; pixels keys removed before normalizer state. | `policy_pretrain/train.py:130`, `policy_pretrain/train.py:250`, `lift_utils/running_statistics.py:201` |
| Dict obs | Hardcoded keys `state`, `privileged_state`, optional `wm_state`. | `policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`, `policy_pretrain/train.py:282` |
| Asymmetric critic | Actor uses `policy_obs_key`, critic uses `value_obs_key`. | `policy_pretrain/sac_networks.py:116`, `policy_pretrain/sac_networks.py:117`, `policy_pretrain/sac_networks.py:151`, `policy_pretrain/sac_networks.py:164` |
| Timeout / truncation | Actor stores `truncation`; critic masks q-error by `1 - truncation`. | `policy_pretrain/train.py:405`, `policy_pretrain/losses.py:111`, `policy_pretrain/losses.py:112` |
| Deterministic eval | Main script passes `deterministic_eval=True`. | `train_in_mujoco_playground.py:325`, `train_in_mujoco_playground.py:326` |

## PPO Details

PPO is not the active LIFT top-level baseline, but vendored Braxlines PPO and MuJoCo Playground PPO configs are useful references.

| Detail | Implementation | Evidence |
|---|---|---|
| Policy distribution | `NormalTanhDistribution` default. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:173`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:176`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:518` |
| Log std design | Delegated to vendored Brax distribution/model code; exact local log_std design UNKNOWN from PPO file alone. | PPO takes `parametric_action_distribution_fn` as an injected dependency (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:173`). |
| Advantage normalization | UNKNOWN: no explicit normalization in local PPO loss wrapper. | `advantages` from `compute_gae` are used directly in surrogate losses (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:105`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:115`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:117`). |
| GAE | Uses Brax PPO `compute_gae`. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:105`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:112` |
| Value clipping | Not found. | Local PPO loss uses MSE `vs - baseline`; no value clip path visible (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:121`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:123`). |
| Entropy cost | `entropy_cost * -entropy`. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:125`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:127` |
| KL handling | Not found. | No KL stop/target line found in local PPO loss; UNKNOWN beyond source scan. |
| Reward scaling | Applied before GAE. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:96` |
| Observation normalization | Normalizer created and updated from rollout observations. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:248`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:376` |
| Timeout bootstrap | `termination = dones * (1 - truncation)` and bootstrap value from final state. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:85`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:98` |
| Rollout shape | Unroll length scanned; data reshaped to `[T+1, batch, ...]` style before minibatching. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:309`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:314`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:371`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:373` |
| Minibatch / epochs | Permutes batch dimension, splits into `num_minibatches`, scans `num_update_epochs`. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:340`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:347`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:354`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:386` |
| Evaluation | JIT eval rollout and progress callback. | `brax_env/brax/v1/experimental/braxlines/training/ppo.py:279`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:284`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:427`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:451` |

