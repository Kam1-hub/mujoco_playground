# Zero-Command Phase-Freeze Diagnostic

Date: 2026-05-15

## Context

- Commit under test: `297f046 Add SAC zero-command phase freeze override`.
- Scope: fresh env1024/R3/UTD-preserving 100k diagnostic run with
  `--env_zero_command_phase_freeze True`.
- Isolation: this test did not combine phase freeze with push-disable.
- Purpose: test whether freezing gait phase on zero command improves stand and
  fixed-command stability.
- No 250k, 5M, 10M, render, code, reward, `action_scale`, Kp, PPO/RSL, or
  checkpoint schema change was made for this report update.

## Training Result

- Status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_phase_freeze/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.

| Metric | Value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 77.1315 |
| sps | 1287.7740 |
| actor_loss | -5.6766 |
| critic_loss | 0.1277 |
| alpha | 0.04279 |
| log_alpha | -3.15137 |
| alpha_effective | 0.04279 |
| q | 4.8994 |
| target_q | 4.8731 |
| reward_mean | -0.12105 |
| done_fraction | 0.01953 |
| discount_mean | 0.98047 |

## Actor Metrics

| Metric | Value |
|---|---:|
| actor_policy_mean_abs_mean | 0.13041 |
| actor_policy_mean_abs_max | 1.38412 |
| actor_log_std mean/min/max | -0.14604 / -0.65995 / 0.06970 |
| actor_policy_std_mean | 0.86613 |
| sampled_action_abs_mean | 0.52296 |
| sampled_action_saturation_fraction_095 | 0.03704 |
| deterministic_action_abs_mean | 0.12426 |
| deterministic_action_saturation_fraction_095 | 0.0 |

## Eval Smokes

- JSON sanity: PASS for `fwd0.5`, `fwd1.0`, and `stand`.
- Eval env overrides confirmed:
  `{"impl":"jax", "zero_command_phase_freeze": true}`.
- All evals returned `EVAL_OK`.
- All action/reward/obs NaN flags were false.

| command/mode | reward mean | tracking_lin_vel | tracking_ang_vel | termination | orientation | ang_vel_xy | feet_phase | feet_air_time | feet_slip | stand_still | action_abs | sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd0.5 det | -3.6287 | 9.9911 | 25.8440 | -100.0 | -37.7401 | -48.2413 | 24.4461 | -1.2200 | -16.6528 | 0.0 | 0.1111 | 0.0 |
| fwd0.5 stoch | -6.4417 | 9.2433 | 3.6729 | -100.0 | -54.7383 | -136.7063 | 19.5672 | -2.2300 | -9.6859 | 0.0 | 0.5202 | 0.0395 |
| fwd1.0 det | -3.7182 | 5.4434 | 24.9592 | -100.0 | -37.1584 | -47.3144 | 24.5605 | -1.2100 | -16.8150 | 0.0 | 0.1119 | 0.0 |
| fwd1.0 stoch | -6.4997 | 3.5212 | 3.9904 | -100.0 | -54.7134 | -133.9037 | 19.7433 | -2.3300 | -9.9311 | 0.0 | 0.5202 | 0.0395 |
| stand det | -6.3323 | 7.6393 | 24.9028 | -100.0 | -36.0732 | -47.1275 | 30.3374 | -1.3700 | -16.5943 | -140.0983 | 0.1080 | 0.0 |
| stand stoch | -10.9066 | 10.4273 | 3.4705 | -100.0 | -54.0306 | -137.5564 | 24.0718 | -2.3100 | -9.9143 | -230.4297 | 0.5205 | 0.0398 |

## Interpretation

- This is runtime-valid, but the gate failed.
- Forward tracking is similar to the push-disable gate and not sufficient.
- `fwd0.5` and `fwd1.0` still terminate with `termination=-100`.
- Stand did not improve. `stand_still` is strongly negative and termination is
  still saturated.
- This weakens the zero-command phase-freeze hypothesis as a standalone fix.
- Do not run 250k, 5M, or 10M from this result.
- The next target should be an isolated default-off `feet_air_time` command-mask
  design/patch.

## Warnings

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `snap-confine` required escalated reruns.
- No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or eval failure
  was observed.
