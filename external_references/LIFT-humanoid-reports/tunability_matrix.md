# Tunability Matrix

Categories:

- A: CLI / config tunable.
- B: wrapper-level small change.
- C: local module change.
- D: learner core change.
- E: not recommended in current repo shape.

| Knob | Category | Current surface | Evidence | Integration note |
|---|---:|---|---|---|
| `env_name` | A | CLI + registry | `train_in_mujoco_playground.py:66`, `train_in_mujoco_playground.py:220` | Direct for registered Playground envs. |
| Domain randomization | A/B | CLI boolean plus registry randomizer | `train_in_mujoco_playground.py:92`, `train_in_mujoco_playground.py:290`, `mujoco_playground/mujoco_playground/_src/registry.py:66` | A if randomizer exists; B/C to add new randomizer. |
| Timesteps/evals | A | CLI/config | `lift_configs.py:12`, `lift_configs.py:13`, `train_in_mujoco_playground.py:173`, `train_in_mujoco_playground.py:177` | Direct. |
| Reward scaling | A | CLI/config/loss arg | `train_in_mujoco_playground.py:103`, `policy_pretrain/losses.py:105` | Important baseline knob. |
| Discount | A | CLI/config/loss arg | `train_in_mujoco_playground.py:112`, `policy_pretrain/losses.py:106` | Direct. |
| Num envs / batch size | A | CLI/config | `train_in_mujoco_playground.py:114`, `train_in_mujoco_playground.py:118`, `policy_pretrain/train.py:299` | Direct, hardware-dependent. |
| UTD / `grad_updates_per_step` | A | CLI/config | `train_in_mujoco_playground.py:109`, `policy_pretrain/train.py:440`, `policy_pretrain/train.py:449` | Direct; central high-throughput SAC parameter. |
| Replay size | A | CLI/config | `train_in_mujoco_playground.py:137`, `lift_configs.py:22`, `policy_pretrain/train.py:299` | Direct, but CLI default `1000` is too small for real runs. |
| Min replay size | A | Config/train arg | `lift_configs.py:23`, `policy_pretrain/train.py:160` | Add CLI if needed. |
| Tau | A | Config/train arg/sweep | `lift_configs.py:24`, `policy_pretrain/train.py:159`, `train_in_mujoco_playground_optuna.py:230` | Direct via config. |
| Actor/critic/alpha LR | A/C | Config works; main CLI bug | `lift_configs.py:37`, `lift_configs.py:38`, `lift_configs.py:39`, `train_in_mujoco_playground.py:192` | Patch main CLI to map `learning_rate` or expose three flags. |
| Policy hidden layers | A | CLI/config | `train_in_mujoco_playground.py:120`, `train_in_mujoco_playground.py:200`, `policy_pretrain/sac_networks.py:107` | Direct. |
| Q hidden layers | A/C | Config works; main CLI writes wrong key | `lift_configs.py:27`, `train_in_mujoco_playground.py:205`, `policy_pretrain/sac_networks.py:108` | Patch `value_hidden_layer_sizes` to `q_hidden_layer_sizes`. |
| Actor/critic obs keys | A | CLI/config/network factory | `train_in_mujoco_playground.py:130`, `train_in_mujoco_playground.py:133`, `policy_pretrain/sac_networks.py:116`, `policy_pretrain/sac_networks.py:117` | Good for asymmetric critic. |
| Dict obs schema | B/C | Hardcoded dummy obs keys | `policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`, `policy_pretrain/train.py:282` | B if wrapper maps keys; C for general dict support. |
| Policy distribution | C | Factory arg only | `policy_pretrain/sac_networks.py:112`, `policy_pretrain/sac_networks.py:125` | Local config exposure is easy; new distribution needs C. |
| Number of critics | C/D | Q network defaults to 2 critics | `policy_pretrain/base_networks.py:267`, `policy_pretrain/base_networks.py:305` | C to expose count; D for REDQ target sampling. |
| Prioritized replay / HER | D | Uniform queue only | `lift_utils/replay_buffers.py:177`, `lift_utils/replay_buffers.py:183` | Requires replay and learner core changes. |
| Timeout handling | D | Critic masks truncation | `policy_pretrain/losses.py:111`, `policy_pretrain/losses.py:112` | Keep unless doing deliberate algorithm study. |
| Evaluation determinism | A | Train arg/main script constant | `policy_pretrain/train.py:163`, `train_in_mujoco_playground.py:326` | Add CLI if needed. |
| Save replay data | A | CLI flag and pickle writer | `train_in_mujoco_playground.py:95`, `policy_pretrain/train.py:544` | Useful for WM, storage-heavy. |
| Vision SAC | E/D | Raises not implemented | `train_in_mujoco_playground.py:218`, `train_in_mujoco_playground.py:219` | Do not rely on this path. |
| Gymnasium API | B/C | Legacy Gym wrapper | `requirements.txt:1`, `brax_env/brax/envs/wrappers/gym.py:20`, `brax_env/brax/envs/wrappers/gym.py:27` | Add Gymnasium adapter separately. |
| WM ensemble/hidden/loss horizon | A | Config | `lift_configs.py:123`, `lift_configs.py:126`, `lift_configs.py:129` | Good ablation knobs. |
| Fine-tune real/model buffer mix | A | Config | `lift_configs.py:205`, `lift_configs.py:206`, `lift_configs.py:210` | Keep isolated initially. |
| Multi-GPU fine-tune env rollouts | E | Single-device assertion | `world_model/finetune_wm_ac.py:1450` | Refactor before integration. |
| Hardcoded fine-tune env names | C | Dict literals | `finetune.py:251`, `finetune.py:258`, `train_wm_from_file.py:142`, `train_wm_from_file.py:158` | Replace with registry/adapter. |
| Fork package shadowing | E/C | Local packages install `mujoco_playground`/`brax` code | `mujoco_playground/pyproject.toml:66`, `mujoco_playground/pyproject.toml:67`, `brax_env/setup.py:31` | Keep isolated unless intended. |

