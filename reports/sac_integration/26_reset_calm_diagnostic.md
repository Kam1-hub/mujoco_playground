# Reset Calm Diagnostic

Date: 2026-05-15

## Context

- Commit under test: `2dcf287 Add SAC reset disturbance scale overrides`.
- Scope: isolated fresh env1024/R3/UTD-preserving 100k diagnostic run.
- Command override under test:
  `--env_reset_joint_noise_scale 0.0 --env_reset_root_qvel_scale 0.0`.
- Isolation: no push-disable, phase-freeze, feet-air-time command mask,
  feet-slip mode, or feet-slip scale override was combined with this run.
- Purpose: test whether removing reset joint/root velocity disturbance delays
  the early fall identified by the termination and terminal-render diagnostics.
- No code, report, reward, `action_scale`, Kp, PPO/RSL, domain randomization,
  fine-tuning, or checkpoint schema change was made during the run.

## Training

- Status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_reset_calm/sac_lift_step_99328.pkl`.
- Checkpoint readiness: PASS.

| metric | value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 77.4890 |
| sps | 1281.8339 |
| actor_loss | -6.0832 |
| critic_loss | 0.1122 |
| alpha | 0.04278 |
| log_alpha | -3.15161 |
| alpha_effective | 0.04278 |
| q | 5.2516 |
| target_q | 5.2811 |
| reward_mean | -0.06960 |
| done_fraction | 0.00391 |
| discount_mean | 0.99609 |

## Actor Metrics

| metric | value |
|---|---:|
| actor_policy_mean_abs_mean | 0.12152 |
| actor_policy_mean_abs_max | 1.31191 |
| actor_log_std mean/min/max | -0.14480 / -0.45237 / 0.08262 |
| actor_policy_std_mean | 0.86669 |
| sampled_action_abs_mean | 0.52066 |
| sampled_action_saturation_fraction_095 | 0.03650 |
| deterministic_action_abs_mean | 0.11827 |
| deterministic_action_saturation_fraction_095 | 0.0 |

## Eval Summary

All eval JSON sanity checks passed: top-level and per-mode `EVAL_OK`, NaN
flags false, diagnostics present, and env overrides include both reset scales
at `0.0`.

| command / mode | reward mean | tracking lin / ang | termination | first done | reason | terminal torso_up_z / root_h / ang_xy |
|---|---:|---:|---:|---:|---|---:|
| fwd0.5 det | -2.0443 | 32.8000 / 42.3541 | -100 | 67.5 | fall 4/4 | -0.0376 / -0.0732 / 3.7162 |
| fwd0.5 stoch | -6.4242 | 20.4621 / 4.6537 | -100 | 63.0 | fall 4/4 | -0.0488 / -0.1044 / 5.7189 |
| fwd1.0 det | -2.3095 | 16.2483 / 44.3136 | -100 | 67.5 | fall 4/4 | -0.0324 / -0.0760 / 3.6232 |
| fwd1.0 stoch | -6.7478 | 7.5346 / 4.3731 | -100 | 62.75 | fall 4/4 | -0.0333 / -0.1270 / 4.3064 |
| stand det | -4.4102 | 30.1104 / 41.6209 | -100 | 67.5 | fall 4/4 | -0.0363 / -0.0687 / 3.8399 |
| stand stoch | -11.9690 | 25.2253 / 5.1955 | -100 | 66.0 | fall 4/4 | -0.0370 / -0.0731 / 6.0462 |

## Render Summary

All deterministic renders returned `RENDER_OK`, used `--stop_on_done True`,
and wrote ignored video plus JSON artifacts.

| command | artifact | first done | reason | terminal torso_up_z / root_h / ang_xy |
|---|---|---:|---|---:|
| fwd0.5 | `logs/sac_render_reset_calm_100k_smoke/render_fwd0p5_seed0_det.mp4` / `.json` | 68 | fall | -0.0705 / -0.1075 / 3.7879 |
| fwd1.0 | `logs/sac_render_reset_calm_100k_smoke/render_fwd1p0_seed0_det.mp4` / `.json` | 68 | fall | -0.0654 / -0.1109 / 3.7555 |
| stand | `logs/sac_render_reset_calm_100k_smoke/render_stand_seed0_det.mp4` / `.json` | 68 | fall | -0.0748 / -0.1018 / 3.9088 |

## Interpretation

- Reset-calm is a meaningful positive signal: previous push-disable,
  phase-freeze, and feet-air-time terminal renders fell at step `51-52`;
  reset-calm deterministic renders fell at step `68`.
- Eval first done also moved later, mostly to `63-68`.
- Training `done_fraction` improved to `0.00391`, and `reward_mean` improved
  to `-0.06960`.
- The gate still fails: all fixed-command eval and render cases still
  terminate by fall, `reward/termination` remains `-100`, torso-up z crosses
  below `0`, and root height is still negative at terminal state.
- This does not justify 250k, 5M, or 10M.
- Reset disturbance contributes to early fall, but it is not the whole blocker.

## Next Target

The next design target should be a controlled base-stability/action-smoothness
review or default-off ablation design. Candidate areas:

- alive signal;
- base-height support;
- vertical velocity (`lin_vel_z`) support;
- stronger orientation / angular-velocity stabilization;
- action-rate smoothing.

Do not run 250k yet. Do not tune `action_scale`, Kp, domain randomization,
fine-tuning, PPO, or RSL from this result.

## Warnings

- Known non-fatal WSL2 CUDA/JAX warnings.
- Sandbox `snap-confine` reruns were required for some `uv` commands.
- No NaN, OOM, fatal CUDA, checkpoint failure, eval failure, or render failure
  was observed.
