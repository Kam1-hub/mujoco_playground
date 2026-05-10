# Integration Plan

## Principle

Do not merge this repository wholesale. Treat it as a reference pipeline. Extract only reviewed configs, wrappers, implementation details, and experiment protocol pieces.

## Directly Usable

| Item | Use | Evidence | Notes |
|---|---|---|---|
| SAC hyperparameter profiles | Baseline config reference | `lift_configs.py:6`, `lift_configs.py:24`, `lift_configs.py:37`, `lift_configs.py:39` | Extract values, not whole config object. |
| Asymmetric actor-critic convention | Actor `state`, critic `privileged_state` | `lift_configs.py:28`, `lift_configs.py:29`, `policy_pretrain/sac_networks.py:151`, `policy_pretrain/sac_networks.py:164` | Strong fit for your pipeline. |
| High-UTD SAC structure | Throughput baseline | `lift_configs.py:21`, `policy_pretrain/train.py:440`, `policy_pretrain/train.py:449` | Useful for high-throughput SAC comparison. |
| MuJoCo Playground wrappers | Env wrapping pattern | `mujoco_playground/mujoco_playground/_src/wrapper.py:124`, `mujoco_playground/mujoco_playground/_src/wrapper.py:159`, `mujoco_playground/mujoco_playground/_src/wrapper.py:160` | Compare with upstream wrappers before copying. |

## Extract As Config

- SAC defaults and env-specific locomotion overrides from `lift_configs.py:6` through `lift_configs.py:113`.
- World-model/fine-tune schedules from `lift_configs.py:116` through `lift_configs.py:248`.
- MuJoCo Playground PPO configs only as references, not as active LIFT code (`mujoco_playground/mujoco_playground/config/locomotion_params.py:133`, `mujoco_playground/mujoco_playground/config/manipulation_params.py:22`).

## Extract As Wrapper

Candidate wrappers:

- Dict observation adapter to provide `state`, `privileged_state`, `wm_state`.
- Action scale adapter to provide `robot_config.policy_output_scale`.
- Domain randomization adapter matching `(mjx.Model, rng) -> (model, in_axes)`.

Evidence for contracts:

- Dict obs expected by learner (`policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`, `policy_pretrain/train.py:282`).
- Action scaling expected by loss (`policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:46`).
- Randomization function type in registry (`mujoco_playground/mujoco_playground/_src/registry.py:27`, `mujoco_playground/mujoco_playground/_src/registry.py:28`).

## Extract As Algorithm Reference

Good references:

- SAC loss with action-scale log-prob correction (`policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:49`, `policy_pretrain/losses.py:105`).
- Uniform replay queue and UTD scan (`lift_utils/replay_buffers.py:177`, `policy_pretrain/train.py:440`, `policy_pretrain/train.py:449`).
- Deterministic evaluation and checkpoint protocol (`train_in_mujoco_playground.py:325`, `train_in_mujoco_playground.py:414`).

Use carefully:

- Fine-tuning/WM implementation, because it assumes specific env methods and has single-device rollout constraints (`finetune.py:309`, `world_model/finetune_wm_ac.py:1450`).
- PPO/MAPPO/IRL vendored code, because it is not the active pipeline and imports old/external Brax training modules (`brax_env/brax/v1/experimental/braxlines/training/ppo.py:30`, `brax_env/brax/v1/experimental/braxlines/training/ppo.py:154`).

## Keep Isolated

- `mujoco_playground/` local fork: it is a fork with custom task modifications (`mujoco_playground/README.md:3`, `mujoco_playground/README.md:4`).
- `brax_env/` local package: it installs packages with `find_packages()` and includes Brax-style imports (`brax_env/setup.py:31`, `brax_env/brax_env_utils.py:20`).
- Torch/RSL wrapper path until dependency and API assumptions are validated (`mujoco_playground/mujoco_playground/_src/wrapper_torch.py:23`, `mujoco_playground/mujoco_playground/_src/wrapper_torch.py:75`).

## Do Not Merge

- Vendored Brax v1 experimental algorithms into the main learner.
- Binary assets/meshes without license review.
- Hardcoded fine-tune env dictionaries.
- Vision path, because it is explicitly unimplemented in the top-level SAC runner.

## Required Patches Before Integration

1. Fix SAC CLI learning-rate mapping.
   - Evidence: `train_in_mujoco_playground.py:191`, `train_in_mujoco_playground.py:192`, `policy_pretrain/train.py:149`, `policy_pretrain/train.py:151`.

2. Fix Q hidden layer CLI key.
   - Evidence: `train_in_mujoco_playground.py:205`, `policy_pretrain/sac_networks.py:108`.

3. Add identity/default action scale for non-G1/T1 envs.
   - Evidence: resolver can return `None` (`train_in_mujoco_playground.py:162`); loss dereferences `robot_config.policy_output_scale` (`policy_pretrain/losses.py:43`).

4. Replace hardcoded env dictionaries with registry/factory interface.
   - Evidence: `finetune.py:251`, `train_wm_from_file.py:142`.

5. Add file-wait timeout to WM pretrain.
   - Evidence: `world_model/pretrain_wm.py:502`, `world_model/pretrain_wm.py:507`.

6. Decide upstream vs fork package ownership before installing into the main env.
   - Evidence: `mujoco_playground/pyproject.toml:66`, `brax_env/setup.py:31`.

## Rollback Plan

1. Keep external repo in a separate virtual environment and separate Git workspace.
2. Integrate configs first as plain copied values.
3. Add wrappers behind feature flags.
4. Add algorithm changes behind a learner flag, with original Brax SAC/PPO baseline unchanged.
5. For any regression, disable wrapper/learner flag and fall back to upstream MuJoCo Playground + existing Brax learner.
6. Keep checkpoints and logs segregated by experiment name and repo commit SHA.

