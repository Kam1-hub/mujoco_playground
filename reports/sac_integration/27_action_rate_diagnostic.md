# Reset-Calm Action-Rate Diagnostic

Date: 2026-05-15

## Context

- Commit under test: `5d61e3c Add SAC action rate reward scale override`.
- Scope: fresh env1024/R3/UTD-preserving 100k diagnostic run.
- Variant: reset-calm baseline plus
  `--env_reward_action_rate_scale -0.01`.
- Reset-calm baseline retained:
  `--env_reset_joint_noise_scale 0.0 --env_reset_root_qvel_scale 0.0`.
- Isolation: no push-disable, zero-command phase-freeze, feet-air-time
  command-mask, feet-slip mode, or feet-slip scale override was combined with
  this run.
- Purpose: test whether small action-rate smoothing improves early torso fall
  beyond reset-calm.
- No training beyond this 100k diagnostic, no 250k/5M/10M, no code/report
  changes during the run, and no reward, `action_scale`, Kp, PPO/RSL, domain
  randomization, fine-tuning, or checkpoint schema change was made.

## Training

- Status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_reset_calm_action_rate_m0p01/sac_lift_step_99328.pkl`.
- Checkpoint readiness: PASS.

| metric | value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 76.891 |
| sps | 1291.80 |
| actor_loss | -6.0360 |
| critic_loss | 0.1037 |
| alpha | 0.04278 |
| log_alpha | -3.1516 |
| alpha_loss | 1.1314 |
| q | 5.2058 |
| target_q | 5.2693 |
| reward_mean | -0.07426 |
| done_fraction | 0.003906 |
| discount_mean | 0.996094 |

## Actor Metrics

| metric | value |
|---|---:|
| actor_policy_mean_abs_mean | 0.1190 |
| actor_policy_mean_abs_max | 1.2029 |
| actor_log_std mean/min/max | -0.1419 / -0.5583 / 0.0611 |
| actor_policy_std_mean | 0.8691 |
| sampled_action_abs_mean | 0.5206 |
| sampled_action_saturation_fraction_095 | 0.03637 |
| deterministic_action_abs_mean | 0.1164 |
| deterministic_action_saturation_fraction_095 | 0.0 |

## Eval Summary

All fixed-command eval smokes completed with `EVAL_OK`, NaN flags false, and
termination diagnostics present.

| command/mode | reward | action_rate | track_lin | track_ang | termination | first_done | reason | root_h | up_z | torso_ang_xy |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| fwd0.5 det | -2.2099 | -0.0165 | 31.6174 | 40.0881 | -100 | 68.0 | fall 4/4 | -0.0956 | -0.0388 | 4.0454 |
| fwd0.5 stoch | -6.4904 | -11.8008 | 20.8587 | 4.1166 | -100 | 62.0 | fall 4/4 | -0.1011 | -0.0355 | 5.6132 |
| fwd1.0 det | -2.5499 | -0.0167 | 13.4108 | 40.7461 | -100 | 68.5 | fall 4/4 | -0.1226 | -0.0538 | 3.8079 |
| fwd1.0 stoch | -6.8279 | -12.0762 | 8.8962 | 4.8678 | -100 | 63.5 | fall 4/4 | -0.0388 | -0.0407 | 6.0984 |
| stand det | -4.8924 | -0.0164 | 29.5168 | 39.3451 | -100 | 68.0 | fall 4/4 | -0.1011 | -0.0513 | 4.2377 |
| stand stoch | -12.2471 | -12.0383 | 23.8466 | 5.0292 | -100 | 63.5 | fall 4/4 | -0.1214 | -0.0999 | 9.4084 |

## Render Summary

All deterministic renders returned `RENDER_OK`, used `--stop_on_done True`,
and wrote ignored video plus JSON artifacts.

| command | artifact | first_done | root_h | up_z | torso_ang_xy | total_reward |
|---|---|---:|---:|---:|---:|---:|
| fwd0.5 | `logs/sac_render_reset_calm_action_rate_100k_smoke/render_fwd0p5_deterministic_seed0.mp4` | 68 | -0.0976 | -0.0422 | 4.1292 | -2.2553 |
| fwd1.0 | `logs/sac_render_reset_calm_action_rate_100k_smoke/render_fwd1p0_deterministic_seed0.mp4` | 68 | -0.0892 | -0.0251 | 3.8114 | -2.5255 |
| stand | `logs/sac_render_reset_calm_action_rate_100k_smoke/render_stand_deterministic_seed0.mp4` | 68 | -0.1035 | -0.0586 | 4.3298 | -5.0056 |

## Interpretation

- Gate failed.
- Fall timing did not materially improve versus reset-calm baseline:
  deterministic eval first done moved only from roughly `67.5` to
  `68` / `68.5`, and deterministic render remained `68` for all commands.
- Deterministic reward worsened slightly:
  `fwd0.5 -2.0443 -> -2.2099`,
  `fwd1.0 -2.3095 -> -2.5499`, and
  `stand -4.4102 -> -4.8924`.
- Deterministic `fwd1.0` tracking weakened:
  `tracking_lin_vel 16.2483 -> 13.4108`.
- The action-rate penalty is tiny on deterministic trajectories but large on
  stochastic trajectories.
- Small action-rate smoothing did not fix early torso/base fall.
- This does not justify 250k, 5M, or 10M.

## Next Target

The next target should be explicit base/upright stability design around:

- orientation;
- `ang_vel_xy`;
- `base_height`;
- `alive`;
- possibly `lin_vel_z`.

Do not tune `action_scale`, Kp, PPO/RSL, domain randomization, fine-tuning, or
long-run scale from this result.

## Warnings

- Known non-fatal WSL2 CUDA/JAX warnings.
- Sandbox `snap-confine` reruns were required for some `uv` commands.
- No NaN, OOM, fatal CUDA, checkpoint failure, eval failure, or render failure
  was observed.
