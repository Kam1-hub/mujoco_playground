# API Contract For MuJoCo Playground Integration

This is the minimum contract needed to connect the useful parts of this repo to a MuJoCo Playground + JAX/Brax-style pipeline.

## Env Loading

Required:

- `registry.get_default_config(env_name)` returns an `ml_collections.ConfigDict`.
- `registry.load(env_name, config=env_cfg)` returns an `mjx_env.MjxEnv`.
- `registry.get_domain_randomizer(env_name)` returns either `None` or a function compatible with randomized vmap.

Evidence:

- Config dispatch: `mujoco_playground/mujoco_playground/_src/registry.py:40`, `mujoco_playground/mujoco_playground/_src/registry.py:48`.
- Env load dispatch: `mujoco_playground/mujoco_playground/_src/registry.py:51`, `mujoco_playground/mujoco_playground/_src/registry.py:63`.
- Domain randomizer dispatch: `mujoco_playground/mujoco_playground/_src/registry.py:66`, `mujoco_playground/mujoco_playground/_src/registry.py:73`.

## Observation Schema

Required for SAC pretraining:

- `state`: actor observation.
- `privileged_state`: critic observation for asymmetric actor-critic.
- Optional `wm_state`: required by world-model/fine-tune paths.

Evidence:

- MJX env observation type supports mappings (`mujoco_playground/mujoco_playground/_src/mjx_env.py:111`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:112`).
- SAC dummy observation hardcodes `state` and `privileged_state` (`policy_pretrain/train.py:273`, `policy_pretrain/train.py:277`).
- WM/fine-tune also expects `wm_state` (`policy_pretrain/train.py:282`, `finetune.py:270`, `finetune.py:272`).

## Action Schema

Required:

- Env exposes `action_size`.
- Policy outputs normalized/tanh actions.
- `robot_config.policy_output_scale` must exist if using local SAC losses/inference.

Evidence:

- SAC networks require `action_size` (`policy_pretrain/sac_networks.py:103`, `policy_pretrain/sac_networks.py:105`).
- Policy inference multiplies action by `robot_config.policy_output_scale` (`policy_pretrain/sac_networks.py:55`, `policy_pretrain/sac_networks.py:65`).
- SAC loss scales actions for Q and adjusts log-prob (`policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:46`, `policy_pretrain/losses.py:49`).

## Reward / Done / Truncation

Required:

- Env step returns `reward` and `done`.
- `state.info["truncation"]` should exist for timeout handling.

Evidence:

- `mjx_env.State` carries reward, done, metrics, and info (`mujoco_playground/mujoco_playground/_src/mjx_env.py:173`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:178`).
- Actor step stores requested extra fields including `truncation` (`brax_env/brax_env_utils.py:47`, `policy_pretrain/train.py:405`).
- SAC critic masks truncations (`policy_pretrain/losses.py:111`, `policy_pretrain/losses.py:112`).

## Randomization Function

Required when `--domain_randomization` is enabled:

- Function signature compatible with `Callable[[mjx.Model, jax.Array], Tuple[mjx.Model, mjx.Model]]`.
- Returns randomized model and `in_axes` for vmap-style vectorization.

Evidence:

- Type alias in registry (`mujoco_playground/mujoco_playground/_src/registry.py:27`, `mujoco_playground/mujoco_playground/_src/registry.py:28`).
- Training wrapper uses `BraxDomainRandomizationVmapWrapper` when randomizer is non-None (`mujoco_playground/mujoco_playground/_src/wrapper.py:155`, `mujoco_playground/mujoco_playground/_src/wrapper.py:158`).
- T1 sim-finetune randomizer returns `model, in_axes` (`mujoco_playground/mujoco_playground/_src/locomotion/t1_12dof_sim_finetune/randomize.py:115`, `mujoco_playground/mujoco_playground/_src/locomotion/t1_12dof_sim_finetune/randomize.py:138`).

## Robot Config Requirements

Required for SAC path:

- `policy_output_scale`.

Required for WM/fine-tune:

- Additional robot-specific methods/fields used by low-level control/dynamics are environment-specific and should be treated as UNKNOWN until audited.

Evidence:

- Main runner resolves robot config from env name (`train_in_mujoco_playground.py:154`, `train_in_mujoco_playground.py:161`).
- SAC losses require `robot_config.policy_output_scale` (`policy_pretrain/losses.py:43`, `policy_pretrain/losses.py:46`).

## Checkpoint Format

SAC policy checkpoints:

- Saved as dill pickle files named `policy{num_steps}.pkl`.
- Object is local SAC training state passed to progress callback.

Evidence:

- Policy directory creation: `train_in_mujoco_playground.py:344`, `train_in_mujoco_playground.py:345`.
- Policy pickle save: `train_in_mujoco_playground.py:414`, `train_in_mujoco_playground.py:415`.

WM checkpoints:

- Saved as dill pickle files named `wm_state{num_steps}.pkl`.

Evidence:

- WM state path and pickle save: `train_wm_from_file.py:176`, `train_wm_from_file.py:191`, `train_wm_from_file.py:192`.

Replay buffer data:

- Saved as pickle shards when `--save_buffer_data` is enabled.

Evidence:

- Buffer data path creation: `train_in_mujoco_playground.py:346`, `train_in_mujoco_playground.py:348`.
- Atomic buffer pickle writing in learner: `policy_pretrain/train.py:544`, `policy_pretrain/train.py:554`.

## Evaluation Function

Required:

- A deterministic policy maker for eval.
- Batched eval env with `EvalWrapper`.

Evidence:

- Main script passes deterministic eval (`train_in_mujoco_playground.py:325`, `train_in_mujoco_playground.py:326`).
- Evaluator wraps eval env and JITs eval unroll (`brax_env/brax_env_utils.py:103`, `brax_env/brax_env_utils.py:116`, `brax_env/brax_env_utils.py:127`).

## Rendering Function

Rendering is host-side MuJoCo rendering, not device-resident.

Evidence:

- Main script JITs reset/step for render rollout but calls env render and writes video (`train_in_mujoco_playground.py:335`, `train_in_mujoco_playground.py:407`, `train_in_mujoco_playground.py:410`).
- Renderer constructs `mujoco.MjData` and `mujoco.Renderer` on host (`mujoco_playground/mujoco_playground/_src/mjx_env.py:315`, `mujoco_playground/mujoco_playground/_src/mjx_env.py:323`).

