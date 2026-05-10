# GPT Plan Verification And Feasibility

## Verdict

The GPT-proposed plan is broadly feasible and mostly consistent with the local `template` source and the LIFT audit reports. It needs two important refinements:

1. Route A using upstream/Brax SAC may be blocked or require adaptation because this template currently only wires Brax PPO in `learning/train_jax_ppo.py`; the next agent must inspect the installed Brax version before assuming a compatible SAC learner exists.
2. G1 action scaling should not blindly copy LIFT's external `policy_output_scale` design. G1 already applies `env_cfg.action_scale` inside the env step, so the SAC implementation must avoid double-scaling.

## Verified Claims

| Claim | Status | Evidence |
|---|---:|---|
| Template contains `g1_env`, `learning/train_jax_ppo.py`, `learning/train_rsl_rl.py`, wrappers, config, XML/assets | Verified | `pyproject.toml:78`, `pyproject.toml:79`; `learning/train_jax_ppo.py:36`, `learning/train_jax_ppo.py:39`; `g1_env/_src/wrapper.py:1`; `g1_env/_src/wrapper_torch.py:1`. |
| Existing envs are `G1JoystickFlatTerrain` and `G1JoystickRoughTerrain` | Verified | `g1_env/_src/locomotion/__init__.py:29`, `g1_env/_src/locomotion/__init__.py:35`; usage note says the same (`USAGE.md:28`). |
| Existing entry points are PPO/RSL | Verified | `pyproject.toml:81`, `pyproject.toml:82`, `pyproject.toml:83`. |
| G1 obs supports `state` and `privileged_state` | Verified statically | `_get_obs` returns dict keys (`g1_env/_src/locomotion/g1/joystick.py:528`, `g1_env/_src/locomotion/g1/joystick.py:530`). |
| G1 action dimension is 29 | Strongly supported, runtime still to confirm | `action_size` is model.nu (`g1_env/_src/locomotion/g1/base.py:111`, `g1_env/_src/locomotion/g1/base.py:112`); local usage states action dim 29 (`USAGE.md:82`, `USAGE.md:83`); code uses 29 controlled joints in obs/randomization comments (`g1_env/_src/locomotion/g1/joystick.py:501`, `g1_env/_src/locomotion/g1/randomize.py:34`). |
| G1 has action scale | Verified | Default config has `action_scale=0.5` (`g1_env/_src/locomotion/g1/joystick.py:38`); step applies it to motor targets (`g1_env/_src/locomotion/g1/joystick.py:363`). |
| Existing PPO config already uses asymmetric critic | Verified | G1 PPO config sets `policy_obs_key="state"` and `value_obs_key="privileged_state"` (`g1_env/config/locomotion_params.py:59`, `g1_env/config/locomotion_params.py:64`). |
| Domain randomization exists | Verified | Randomizer registered for both envs (`g1_env/_src/locomotion/__init__.py:43`, `g1_env/_src/locomotion/__init__.py:46`) and returns `(model, in_axes)` (`g1_env/_src/locomotion/g1/randomize.py:83`, `g1_env/_src/locomotion/g1/randomize.py:100`). |
| LIFT should be reference-only, not wholesale merged | Verified | LIFT reports copied to `external_references/LIFT-humanoid-reports/`; see `integration_plan.md` and `bug_traps.md` there. |
| Windows native GPU JAX should not be default success standard | Verified from official docs | JAX docs: https://docs.jax.dev/en/latest/installation.html and https://docs.jax.dev/en/latest/developer.html. |

## Unknowns To Resolve In Phase 1

- Whether `state.info["truncation"]` exists on raw env reset/step or only after wrapper.
- Exact runtime `observation_size` values and shapes in the installed runtime.
- Exact installed Brax SAC API availability.
- Whether native Windows CPU smoke can run once dependencies are installed.
- Whether WSL2/Linux CUDA is available for GPU smoke.

## Feasibility Assessment

| Route | Feasibility | Reason |
|---|---:|---|
| Phase 0/1 audit | High | Source structure is compact and registry/wrappers are already present. |
| Route A Brax SAC | Medium | Depends on installed Brax version and SAC API; template only currently imports Brax PPO (`learning/train_jax_ppo.py:27`, `learning/train_jax_ppo.py:29`). |
| Route B LIFT-style SAC | High but more work | G1 obs/action layout matches asymmetric SAC; implementation can be clean-room while referencing LIFT report details. |
| Route C clean-room SAC | High as fallback | Standard SAC with symmetric obs first is straightforward, then add privileged critic. |
| World model/fine-tune | Out of scope | LIFT reports flag hardcoded env assumptions and single-device fine-tune traps. |

## Recommended Adjustment To GPT Plan

Proceed with the plan, but make Phase 1 mandatory before writing SAC code and make the first SAC implementation conservative about action scaling:

- Actor outputs normalized actions.
- G1 env applies `action_scale`.
- Do not externally multiply action by `action_scale` unless the loss and env path are updated together and documented.

