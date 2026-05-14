# Push Disable Diagnostic

Date: 2026-05-15

## Context

- Commit under test: `cea95a7 Add SAC push disable env override`.
- Fresh env1024/R3/UTD-preserving 100k run with `--env_push_enable False`.
- No 250k, 5M, or 10M run was executed for this diagnostic.
- No code or report changes were made during the run; artifacts were written
  only under ignored `logs/`.
- Purpose: test whether disabling default push perturbations improves early
  fixed-forward tracking and termination behavior.

## Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_push_disable/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS

| Metric | Value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 76.5224 |
| sps | 1298.0258 |
| actor_loss | -5.8302 |
| critic_loss | 0.0771 |
| alpha | 0.04279 |
| log_alpha | -3.15139 |
| alpha_effective | 0.04279 |
| q | 5.0656 |
| target_q | 5.0483 |
| reward_mean | -0.10694 |
| done_fraction | 0.015625 |
| discount_mean | 0.984375 |

Actor metrics:

| Metric | Value |
|---|---:|
| actor_policy_mean_abs_mean | 0.13060 |
| actor_policy_mean_abs_max | 1.09176 |
| actor_log_std_mean | -0.14840 |
| actor_log_std_min | -0.54374 |
| actor_log_std_max | 0.07622 |
| actor_policy_std_mean | 0.86383 |
| sampled_action_abs_mean | 0.52134 |
| sampled_action_saturation_fraction_095 | 0.03704 |
| deterministic_action_abs_mean | 0.12580 |
| deterministic_action_saturation_fraction_095 | 0.0 |

## Fixed-Command Eval Smokes

Both `fwd0.5` and `fwd1.0` JSON sanity checks passed. Eval inherited the
checkpoint override:

```json
{"impl": "jax", "push_config.enable": false}
```

### fwd0.5 `[0.5, 0, 0]`

| Mode | Reward Mean | Reward Std | Reward Min | Reward Max | tracking_lin_vel | termination | orientation | ang_vel_xy | feet_phase | feet_slip | Action Abs | Saturation | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic | -3.6396 | 0.5483 | -4.2221 | -2.7382 | 9.9375 | -100.0 | -37.7836 | -46.9416 | 25.0424 | -16.5546 | 0.1080 | 0.0 | false |
| stochastic | -6.1141 | 0.2205 | -6.4565 | -5.8536 | 9.3145 | -100.0 | -51.0247 | -129.8872 | 20.5887 | -9.5412 | 0.5187 | 0.0383 | false |

### fwd1.0 `[1.0, 0, 0]`

| Mode | Reward Mean | Reward Std | Reward Min | Reward Max | tracking_lin_vel | termination | orientation | ang_vel_xy | feet_phase | feet_slip | Action Abs | Saturation | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic | -3.7192 | 0.3622 | -4.1736 | -3.1601 | 5.2440 | -100.0 | -36.9127 | -46.5780 | 25.1010 | -16.6252 | 0.1086 | 0.0 | false |
| stochastic | -5.6779 | 0.8665 | -6.3842 | -4.2354 | 3.6313 | -100.0 | -41.2300 | -116.2568 | 20.2056 | -9.1818 | 0.5183 | 0.0380 | false |

## Interpretation

- Push-disable gives a real signal and improves deterministic fixed-forward
  tracking relative to prior 100k gates, especially for `fwd1.0`.
- However, `reward/termination` is still saturated at `-100.0` on `fwd0.5`
  and `fwd1.0`, for both deterministic and stochastic modes.
- Therefore push-disable alone does not pass the fixed-forward stability gate.
- This result does not justify 250k, 5M, or 10M training.
- The result supports moving from perturbation ablation to phase /
  `feet_air_time` prior ablation design.

## Warnings

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- `uv` sandbox `snap-confine` issue; rerun externally with unchanged
  parameters.
- CLI parse issue from passing `--fixed_command` without a value; rerun
  correctly with `--fixed_command True`.
- No traceback after rerun, NaN, OOM, fatal CUDA error, checkpoint failure, or
  eval failure was observed.

## Next Action

- Do not run 250k, 5M, or 10M from this result.
- Next design target: phase freeze / `feet_air_time` command-mask prior
  ablation.
- Keep reward weights, `action_scale`, and Kp unchanged.
