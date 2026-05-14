# Feet-Air-Time Command-Mask Diagnostic

Date: 2026-05-15

## Context

- Commit under test: `d353fe6 Add SAC feet air time command mask override`.
- Scope: fresh env1024/R3/UTD-preserving 100k diagnostic run with
  `--env_feet_air_time_command_mask True`.
- Isolation: this test did not combine the command mask with push-disable or
  zero-command phase-freeze.
- Purpose: test whether removing `feet_air_time` reward on zero-command samples
  improves stand and fixed-command stability.
- No 250k, 5M, 10M, render, code, reward, `action_scale`, Kp, PPO/RSL, or
  checkpoint schema change was made for this report update.

## Training Result

- Status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_air_time_mask/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.

| Metric | Value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 77.6672 |
| sps | 1278.8924 |
| actor_loss | -5.8315 |
| critic_loss | 0.1118 |
| alpha | 0.04279 |
| log_alpha | -3.15146 |
| alpha_effective | 0.04279 |
| q | 5.0715 |
| target_q | 5.0114 |
| reward_mean | -0.1165 |
| done_fraction | 0.01953 |
| discount_mean | 0.98047 |

## Actor Metrics

| Metric | Value |
|---|---:|
| actor_policy_mean_abs_mean | 0.12316 |
| actor_policy_mean_abs_max | 1.11455 |
| actor_log_std mean/min/max | -0.14503 / -0.54318 / 0.06541 |
| actor_policy_std_mean | 0.86674 |
| sampled_action_abs_mean | 0.52025 |
| sampled_action_saturation_fraction_095 | 0.03718 |
| deterministic_action_abs_mean | 0.11887 |
| deterministic_action_saturation_fraction_095 | 0.0 |

## Eval Smokes

- JSON sanity: PASS for `fwd0.5`, `fwd1.0`, and `stand`.
- Eval env overrides confirmed:
  `{"feet_air_time_command_mask": true, "impl":"jax"}`.
- All evals returned `EVAL_OK`.
- All action/reward/obs NaN flags were false.

| command/mode | reward mean | tracking_lin_vel | tracking_ang_vel | termination | orientation | ang_vel_xy | feet_phase | feet_air_time | feet_slip | stand_still | action_abs | sat |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fwd0.5 det | -3.6505 | 10.2344 | 20.3590 | -100.0 | -37.7849 | -46.8961 | 25.5653 | -1.4700 | -16.7857 | 0.0 | 0.0999 | 0.0 |
| fwd0.5 stoch | -6.4587 | 10.6768 | 3.5048 | -100.0 | -57.9756 | -140.3279 | 20.7195 | -2.1400 | -9.5508 | 0.0 | 0.5190 | 0.0383 |
| fwd1.0 det | -3.8335 | 4.1466 | 20.5456 | -100.0 | -39.2348 | -49.2255 | 25.7549 | -1.3400 | -16.1351 | 0.0 | 0.1000 | 0.0 |
| fwd1.0 stoch | -6.5072 | 4.2529 | 2.8092 | -100.0 | -57.1087 | -139.6565 | 21.3788 | -2.3800 | -9.4494 | 0.0 | 0.5192 | 0.0383 |
| stand det | -6.5074 | 8.6266 | 20.5424 | -100.0 | -37.9999 | -46.8852 | 25.6276 | 0.0 | -16.3734 | -143.5669 | 0.1002 | 0.0 |
| stand stoch | -10.8848 | 11.1323 | 2.7537 | -100.0 | -56.5973 | -136.7390 | 20.7464 | 0.0 | -9.5355 | -229.1816 | 0.5189 | 0.0388 |

## Interpretation

- This is runtime-valid and the mask works: stand eval reports
  `reward/feet_air_time=0.0`.
- The gate failed: `reward/termination` remains saturated at `-100` for
  `fwd0.5`, `fwd1.0`, and `stand`.
- `fwd0.5` tracking is similar or slightly higher than the push-disable and
  phase-freeze gates, but `fwd1.0` is worse than both.
- Stand did not improve. `stand_still` remains strongly negative.
- This weakens the `feet_air_time` zero-command conflict as a standalone fix.
- Do not run 250k, 5M, or 10M from this result.
- The next target should be component-level termination/contact analysis.

## Warnings

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `snap-confine` required escalated reruns.
- No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or eval failure
  was observed.
