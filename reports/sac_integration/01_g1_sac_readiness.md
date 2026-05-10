# G1 SAC Readiness

Status: TODO for next agent.

This report must be produced after implementing and running `scripts/check_g1_env_api.py`.

## Required Answers

- `g1_env.registry.ALL_ENVS`
- `registry.get_default_config(env_name)`
- `registry.load(env_name, config=...)`
- `env.observation_size`
- `env.action_size`
- reset obs type and keys
- step obs type and keys
- `state` shape
- `privileged_state` shape
- action range expected by env
- action scale source
- whether `state.info["truncation"]` exists
- metrics keys
- whether current wrappers are sufficient for SAC

## Initial Static Evidence

- Env names are registered in `g1_env/_src/locomotion/__init__.py:29` through `g1_env/_src/locomotion/__init__.py:35`.
- `action_scale=0.5` is in `g1_env/_src/locomotion/g1/joystick.py:38`.
- The env step applies actions through `motor_targets = default_pose + action * action_scale` (`g1_env/_src/locomotion/g1/joystick.py:363`).
- `_get_obs` returns `state` and `privileged_state` (`g1_env/_src/locomotion/g1/joystick.py:528`, `g1_env/_src/locomotion/g1/joystick.py:530`).

## Runtime Results

NOT VALIDATED.

