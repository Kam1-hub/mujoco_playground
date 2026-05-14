# Terminal Render Diagnostics

Date: 2026-05-15

## Context

- Commit under test: `5b3f393 Add SAC terminal render diagnostics`.
- Scope: deterministic short render terminal diagnostic sweep over existing
  failed 100k checkpoints.
- Output directory: `./logs/sac_render_terminal_diag_100k/`.
- Render outputs: `9` matching `.json` and `.mp4` pairs.
- File naming: `render_<variant>_<command>_seed0_det.*`.
- Variants: `push_disable`, `phase_freeze`, and `feet_air_time_mask`.
- Commands: `fwd0.5`, `fwd1.0`, and `stand`.
- Checkpoint readiness: PASS for all three checkpoints.
- Runtime: JAX GPU.
- The repository stayed clean during the sweep.
- No training, eval benchmark, preflight, code change, or report change was
  performed during the sweep itself.

## Sweep Result

| variant | command | done step/frame | reason | torso_up_z | root_h | torso_ang_xy | feet_contact | pelvis_local_linvel |
|---|---|---:|---|---:|---:|---:|---|---|
| push_disable | fwd0.5 | 51 / 50 | fall | -0.0356 | -0.1275 | 7.2921 | [0,0] | [0.926,-2.053,1.939] |
| push_disable | fwd1.0 | 51 / 50 | fall | -0.0533 | -0.1337 | 7.2625 | [0,0] | [0.888,-2.046,1.976] |
| push_disable | stand | 52 / 51 | fall | -0.1073 | -0.1687 | 7.4798 | [0,1] | [1.007,-1.901,2.121] |
| phase_freeze | fwd0.5 | 51 / 50 | fall | -0.0498 | -0.1728 | 7.2386 | [0,1] | [0.998,-2.205,1.811] |
| phase_freeze | fwd1.0 | 51 / 50 | fall | -0.0501 | -0.1727 | 7.1651 | [0,1] | [0.988,-2.220,1.808] |
| phase_freeze | stand | 51 / 50 | fall | -0.1161 | -0.2228 | 8.0283 | [0,1] | [1.186,-1.922,2.019] |
| feet_air_time_mask | fwd0.5 | 52 / 51 | fall | -0.0832 | -0.1923 | 6.9256 | [0,1] | [0.818,-1.954,2.197] |
| feet_air_time_mask | fwd1.0 | 52 / 51 | fall | -0.1059 | -0.2027 | 6.9432 | [0,1] | [0.777,-1.916,2.267] |
| feet_air_time_mask | stand | 52 / 51 | fall | -0.0672 | -0.1870 | 6.9997 | [0,1] | [0.895,-1.935,2.158] |

## Interpretation

- All 9 deterministic renders terminate via `fall`.
- No case terminates from illegal contact or qpos/qvel NaN.
- Survival time is effectively identical across variants: `51` to `52`
  control steps, about `1.0s` at `ctrl_dt=0.02`.
- Terminal state consistently has negative torso-up z, negative root height,
  high torso XY angular velocity around `6.9` to `8.0`, and large local
  velocity.
- The render JSON supports early torso/base stability failure.
- Do not overclaim forward/backward fall direction from the JSON alone; that
  requires detailed video inspection.
- No variant meaningfully improves survival.

## Consequence

The short render sweep reinforces the eval-only termination/contact conclusion:
the 100k variants are failing through early torso/base instability rather than
illegal contact, numerical instability, or an isolated reward accounting issue.

This argues against any immediate 250k, 5M, or 10M continuation. The next
target should be early-fall stabilization or curriculum design before longer
training.

Candidate design areas:

- reset disturbance and warmup;
- command warmup/curriculum;
- base-height, alive, orientation, and angular-velocity stabilizers;
- action-rate smoothing.

Do not tune reward weights, `action_scale`, or Kp from this report alone.

## Warnings

- Known non-fatal WSL2 CUDA/JAX warnings.
- Sandbox `snap-confine` reruns were required for some `uv` commands.
- No NaN, OOM, fatal CUDA, checkpoint failure, or render failure was observed.
