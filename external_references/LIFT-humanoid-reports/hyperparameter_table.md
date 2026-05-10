# Hyperparameter Table

Legend:

- Config default: value from `lift_configs.py` or Playground config.
- CLI default: value from `flags.DEFINE_*`.
- Train default: value from train function signature.
- Sweep: Optuna search space if present.
- Tunability: A/B/C/D/E follows `tunability_matrix.md`.

## SAC Pretraining

| Parameter | Default | Source file | CLI override | Runtime use | Sweep search space | Tunability | Recommended for my pipeline? |
|---|---|---|---|---|---|---:|---|
| `env_name` | CLI default `LeapCubeReorient` | `train_in_mujoco_playground.py:66`, `train_in_mujoco_playground.py:68` | `--env_name` | Registry load (`train_in_mujoco_playground.py:220`) | Not swept | A | Yes, but use locomotion/manipulation task explicitly. |
| `num_timesteps` | Config `5_000_000`; low/high dim overrides to `5_000_000_000` | `lift_configs.py:12`, `lift_configs.py:69`, `lift_configs.py:78`, `lift_configs.py:97` | CLI default `1_000_000`; applied if present (`train_in_mujoco_playground.py:99`, `train_in_mujoco_playground.py:173`) | Epoch schedule (`policy_pretrain/train.py:203`, `policy_pretrain/train.py:524`) | Optuna default if absent `80_000_000` (`train_in_mujoco_playground_optuna.py:275`) | A | Yes. |
| `num_evals` | Config `1000` | `lift_configs.py:13` | CLI default `5`; applied if present (`train_in_mujoco_playground.py:102`, `train_in_mujoco_playground.py:177`) | Evaluator cadence (`policy_pretrain/train.py:155`) | Optuna default if absent `20` (`train_in_mujoco_playground_optuna.py:277`) | A | Yes. |
| `reward_scaling` | Config `1.0`; env overrides `3/16/32` | `lift_configs.py:14`, `lift_configs.py:60`, `lift_configs.py:92`, `lift_configs.py:111` | CLI default `0.1`; applied if present (`train_in_mujoco_playground.py:103`, `train_in_mujoco_playground.py:179`) | Critic target (`policy_pretrain/losses.py:105`) | `2**round(k)`, `k in [1,3]` (`train_in_mujoco_playground_optuna.py:255`, `train_in_mujoco_playground_optuna.py:256`) | A | Yes, high priority. |
| `episode_length` | From env config | `lift_configs.py:15` | CLI default `1000`; applied if present (`train_in_mujoco_playground.py:104`, `train_in_mujoco_playground.py:181`) | Wrappers/eval (`policy_pretrain/train.py:142`, `policy_pretrain/train.py:223`) | Not swept | A | Yes. |
| `normalize_observations` | `True` | `lift_configs.py:16` | CLI default `True` (`train_in_mujoco_playground.py:105`) | Running stats (`policy_pretrain/train.py:250`) | Not swept | A | Yes. |
| `action_repeat` | `1` | `lift_configs.py:17` | CLI default `1` (`train_in_mujoco_playground.py:108`) | Env step accounting/wrapper (`policy_pretrain/train.py:146`, `policy_pretrain/train.py:203`) | Not swept | A | Yes. |
| `discounting` | `0.9870596636685084`; env overrides around `0.982` | `lift_configs.py:18`, `lift_configs.py:49`, `lift_configs.py:81`, `lift_configs.py:100` | CLI default `0.99`; applied (`train_in_mujoco_playground.py:112`, `train_in_mujoco_playground.py:189`) | Critic target (`policy_pretrain/losses.py:106`) | `[0.975, 0.995]` (`train_in_mujoco_playground_optuna.py:226`) | A | Yes. |
| `num_envs` | Base `1000`; locomotion overrides `4096` | `lift_configs.py:19`, `lift_configs.py:58`, `lift_configs.py:90`, `lift_configs.py:109` | CLI default `1024`; applied (`train_in_mujoco_playground.py:114`, `train_in_mujoco_playground.py:193`) | Vmap/pmap batch (`policy_pretrain/train.py:147`, `policy_pretrain/train.py:203`) | `2**round(k)`, `k in [10,13]` (`train_in_mujoco_playground_optuna.py:247`, `train_in_mujoco_playground_optuna.py:248`) | A | Yes. |
| `num_eval_envs` | Low/high dim `1024` | `lift_configs.py:71`, `lift_configs.py:80`, `lift_configs.py:99` | CLI default `128`; applied (`train_in_mujoco_playground.py:115`, `train_in_mujoco_playground.py:195`) | Evaluator (`policy_pretrain/train.py:148`, `brax_env/brax_env_utils.py:107`) | Not swept | A | Yes. |
| `batch_size` | Base `1024`; high dim `16384` | `lift_configs.py:20`, `lift_configs.py:91`, `lift_configs.py:110` | CLI default `256`; applied (`train_in_mujoco_playground.py:118`, `train_in_mujoco_playground.py:197`) | Replay sample/update (`policy_pretrain/train.py:154`, `policy_pretrain/train.py:303`) | `2**round(k)`, `k in [4,13]` (`train_in_mujoco_playground_optuna.py:251`, `train_in_mujoco_playground_optuna.py:252`) | A | Yes. |
| `grad_updates_per_step` | Base `9`; env overrides `16/17/19` | `lift_configs.py:21`, `lift_configs.py:56`, `lift_configs.py:88`, `lift_configs.py:107` | CLI default `8`; applied (`train_in_mujoco_playground.py:109`, `train_in_mujoco_playground.py:187`) | UTD scan (`policy_pretrain/train.py:440`, `policy_pretrain/train.py:449`) | Integer `2..20` (`train_in_mujoco_playground_optuna.py:239`, `train_in_mujoco_playground_optuna.py:240`) | A | Yes, core high-throughput SAC knob. |
| `max_replay_size` | `1_000_000` | `lift_configs.py:22` | CLI default `1000`; applied (`train_in_mujoco_playground.py:137`, `train_in_mujoco_playground.py:216`) | Replay queue capacity (`policy_pretrain/train.py:161`, `policy_pretrain/train.py:299`) | `10**round(k)`, `k in [4,5]` despite comment mismatch (`train_in_mujoco_playground_optuna.py:243`, `train_in_mujoco_playground_optuna.py:244`) | A | Yes, but fix CLI default for real runs. |
| `min_replay_size` | `8192` | `lift_configs.py:23` | No main CLI | Prefill assertion/use (`policy_pretrain/train.py:160`, `policy_pretrain/train.py:459`) | Not swept | A | Yes. |
| `tau` | `0.024184784277809342`; env overrides | `lift_configs.py:24`, `lift_configs.py:53`, `lift_configs.py:85`, `lift_configs.py:104` | No main CLI | Target update (`policy_pretrain/train.py:374`) | `[1e-3, 5e-2]` log (`train_in_mujoco_playground_optuna.py:230`) | A | Yes. |
| `policy_hidden_layer_sizes` | `(512, 256, 128)` | `lift_configs.py:26` | CLI list default `[64,64,64]`; applied (`train_in_mujoco_playground.py:120`, `train_in_mujoco_playground.py:200`) | Policy network (`policy_pretrain/sac_networks.py:107`, `policy_pretrain/sac_networks.py:140`) | Not swept | A | Yes. |
| `q_hidden_layer_sizes` | `(1024, 512, 256)` | `lift_configs.py:27` | Main CLI has bug: writes `value_hidden_layer_sizes` (`train_in_mujoco_playground.py:125`, `train_in_mujoco_playground.py:205`) | Q network (`policy_pretrain/sac_networks.py:108`, `policy_pretrain/sac_networks.py:157`) | Not swept | A/C | Yes; patch CLI first. |
| `policy_obs_key` | `state` | `lift_configs.py:28` | CLI default `state` (`train_in_mujoco_playground.py:130`) | Actor input key (`policy_pretrain/sac_networks.py:151`) | Not swept | A | Yes. |
| `value_obs_key` | `privileged_state` | `lift_configs.py:29` | CLI default `state`; applied if present (`train_in_mujoco_playground.py:133`, `train_in_mujoco_playground.py:210`) | Critic input key (`policy_pretrain/sac_networks.py:164`) | Not swept | A | Yes; keep default config, beware CLI default. |
| `activation` | `swish` | `lift_configs.py:30` | No CLI | Network activation lookup (`policy_pretrain/sac_networks.py:96`, `policy_pretrain/sac_networks.py:138`) | Not swept | A/C | Yes. |
| `q_network_layer_norm` | `False` | `lift_configs.py:31` | No CLI | Q network factory (`policy_pretrain/sac_networks.py:111`, `policy_pretrain/sac_networks.py:163`) | Not swept | A/C | Maybe. |
| `target_entropy_coef` | `0.5`; env overrides | `lift_configs.py:34`, `lift_configs.py:54`, `lift_configs.py:86`, `lift_configs.py:105` | No main CLI | Target entropy (`policy_pretrain/train.py:673`) | `[0.01, 1.0]` (`train_in_mujoco_playground_optuna.py:231`) | A | Yes. |
| `int_log_alpha` | `-3.348060147490869`; env overrides | `lift_configs.py:35`, `lift_configs.py:55`, `lift_configs.py:87`, `lift_configs.py:106` | CLI `--log_alpha` default `1e-3`, applied to `int_log_alpha` (`train_in_mujoco_playground.py:135`, `train_in_mujoco_playground.py:213`) | Alpha init (`policy_pretrain/train.py:104`, `policy_pretrain/train.py:165`) | log alpha from sampled alpha `[1e-5,1]` (`train_in_mujoco_playground_optuna.py:234`, `train_in_mujoco_playground_optuna.py:235`) | A | Yes, but rename CLI. |
| `actor_learning_rate` | `1.034e-4`; env overrides | `lift_configs.py:37`, `lift_configs.py:51`, `lift_configs.py:83`, `lift_configs.py:102` | Main `--learning_rate` does not map to this (`train_in_mujoco_playground.py:191`, `train_in_mujoco_playground.py:192`) | Actor optimizer (`policy_pretrain/train.py:150`, `policy_pretrain/train.py:96`) | `[1e-4,2e-3]` log (`train_in_mujoco_playground_optuna.py:228`) | A/C | Yes; patch CLI. |
| `critic_learning_rate` | `1.001e-4`; env overrides | `lift_configs.py:38`, `lift_configs.py:52`, `lift_configs.py:84`, `lift_configs.py:103` | Main `--learning_rate` bug | Critic optimizer (`policy_pretrain/train.py:151`, `policy_pretrain/train.py:101`) | `[1e-4,2e-3]` log (`train_in_mujoco_playground_optuna.py:229`) | A/C | Yes. |
| `alpha_learning_rate` | `9.969e-3`; env overrides | `lift_configs.py:39`, `lift_configs.py:50`, `lift_configs.py:82`, `lift_configs.py:101` | Main `--learning_rate` bug | Alpha optimizer (`policy_pretrain/train.py:149`, `policy_pretrain/train.py:105`) | `[1e-4,2e-3]` log (`train_in_mujoco_playground_optuna.py:227`) | A/C | Yes, inspect high default. |

## SAC Train Function Defaults

These are lower-level defaults if config/CLI do not override:

| Parameter | Train default | Source file | Runtime use | Recommended? |
|---|---:|---|---|---|
| `action_repeat` | `1` | `policy_pretrain/train.py:146` | Env wrapper and step accounting | Yes |
| `num_envs` | `1` | `policy_pretrain/train.py:147` | Vectorized actors | Override for throughput |
| `num_eval_envs` | `1024` | `policy_pretrain/train.py:148` | Evaluator | Yes |
| `alpha_learning_rate` | `1e-4` | `policy_pretrain/train.py:149` | Alpha optimizer | Use config/sweep |
| `actor_learning_rate` | `1e-4` | `policy_pretrain/train.py:150` | Actor optimizer | Use config/sweep |
| `critic_learning_rate` | `1e-4` | `policy_pretrain/train.py:151` | Critic optimizer | Use config/sweep |
| `discounting` | `0.9` | `policy_pretrain/train.py:152` | Bellman target | Override |
| `normalize_observations` | `False` | `policy_pretrain/train.py:156` | Running stats | Override to True |
| `reward_scaling` | `1.0` | `policy_pretrain/train.py:158` | Critic target | Tune |
| `tau` | `0.005` | `policy_pretrain/train.py:159` | Target update | Tune |
| `grad_updates_per_step` | `1` | `policy_pretrain/train.py:162` | UTD | Override for high-throughput SAC |

## World Model And Fine-Tuning

| Parameter | Default | Source file | CLI override | Runtime use | Tunability | Recommended? |
|---|---|---|---|---|---:|---|
| `wm.max_replay_size` | SAC max replay size | `lift_configs.py:121` | No | WM replay buffer (`world_model/pretrain_wm.py:244`) | A | Use isolated first. |
| `model_training_max_epochs` | pretrain `1`; fine-tune `1000` | `lift_configs.py:122`, `lift_configs.py:195` | No | WM training loops (`world_model/pretrain_wm.py:524`) | A | Tune for WM. |
| `ensemble_size` | `1` | `lift_configs.py:123` | No | Ensemble model (`world_model/wm_networks.py:44`) | A | Increase for uncertainty ablations. |
| `num_elites` | `1` | `lift_configs.py:124` | No | Elite selection in WM inference/training | A | Useful for ablation. |
| `wm_learning_rate` | `1e-3` | `lift_configs.py:125` | No | Model optimizer | A | Tune. |
| `hidden_size` | `400` | `lift_configs.py:126` | No | WM network | A | Tune. |
| `model_training_batch_size` | `200` | `lift_configs.py:127` | No | WM batches | A | Tune. |
| `model_loss_horizon` | pretrain `1`; fine-tune `4` | `lift_configs.py:129`, `lift_configs.py:203` | No | WM loss rollout horizon | A | Yes for ablation. |
| `model_probabilistic` | `True` | `lift_configs.py:130` | No | Probabilistic mean/logvar head (`world_model/wm_networks.py:95`) | A | Yes. |
| `ssrl_dynamics_fn` / `dynamics_fn` | `contact_integrate_only` | `lift_configs.py:136`, `lift_configs.py:171` | No | Env dynamics fn (`finetune.py:309`) | A/C | Keep isolated. |
| `model_rollouts_per_hallucination_update` | `400` | `lift_configs.py:180` | No | Model rollout generation | A | Yes for MBPO-style tuning. |
| `sac_grad_updates_per_hallucination_update` | `20` | `lift_configs.py:181` | No | Fine-tune SAC UTD | A | Yes. |
| `real_ratio` | `0.06` | `lift_configs.py:210` | No | Env/model buffer mixing (`world_model/finetune_wm_ac.py:740`) | A | Yes. |
| `sac_tau` | `0.001` | `lift_configs.py:212` | No | Fine-tune target update | A | Tune. |
| `deterministic_in_env` | `True` | `lift_configs.py:214` | No | Env data collection | A | Relevant for sim2real protocol. |
| `deterministic_eval` | `True` | `lift_configs.py:215` | No | Evaluation policy | A | Yes. |
| model horizon schedule | `1 -> 20`, epochs `0 -> 10` | `lift_configs.py:233`, `lift_configs.py:237` | No | `update_model_horizon` path (`world_model/finetune_wm_ac.py:1466`) | A | Yes for ablation. |
| HUPTS schedule | `10 -> 1000`, epochs `0 -> 4` | `lift_configs.py:239`, `lift_configs.py:243` | No | Hallucination updates schedule (`world_model/finetune_wm_ac.py:1467`) | A | Yes for ablation. |

