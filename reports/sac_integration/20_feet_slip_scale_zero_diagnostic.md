# Feet Slip Scale Zero Diagnostic

Date: 2026-05-15

## Context

- Earlier feet-slip ablation controls were committed in
  `51c91c8 Add SAC feet slip ablation controls`.
- The `foot_velocity` 100k result was recorded in
  `815b5ae Record SAC foot velocity feet slip diagnostic`.
- Eval override inheritance was committed in
  `442d297 Apply SAC eval env overrides from checkpoint`.
- The eval inheritance patch fixes an important mismatch: checkpoint env
  overrides such as `env_feet_slip_scale` and `env_feet_slip_mode` are now
  applied to eval env config.
- This report records a fresh env1024 R3 100k diagnostic with
  `--env_feet_slip_scale 0.0`.
- No 250k, 5M, 10M, preflight, code change, reward/action scale/Kp change,
  PPO/RSL change, or generated artifact commit was made for this report.

## Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_slip_scale_0/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS

| Metric | Value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1312 |
| wall_time | 73.8268 |
| sps | 1345.4198 |
| actor_loss | -5.78947 |
| critic_loss | 0.20850 |
| q | 4.95647 |
| target_q | 4.94161 |
| reward_mean | -0.12537 |
| done_fraction | 0.01953125 |
| discount_mean | 0.98046875 |
| alpha | 0.043785 |
| log_alpha | -3.12846 |
| alpha_effective | 0.043785 |

Actor metrics:

| Metric | Value |
|---|---:|
| actor_policy_mean_abs_mean | 0.14590 |
| actor_policy_mean_abs_max | 1.29871 |
| actor_log_std_mean | -0.14949 |
| actor_log_std_min | -0.56515 |
| actor_log_std_max | 0.07885 |
| actor_policy_std_mean | 0.86332 |
| sampled_action_abs_mean | 0.52383 |
| deterministic_action_abs_mean | 0.13972 |

## Eval Override Caveat

The initial eval before `442d297` did not inherit checkpoint env overrides, so
`reward/feet_slip` appeared nonzero despite training with
`--env_feet_slip_scale 0.0`. Treat that pre-override eval as superseded for
reward-component interpretation.

After `442d297`, inherited eval applies
`reward_config.scales.feet_slip: 0.0`, and `reward/feet_slip` is correctly
zero in the eval outputs.

## Inherited Fixed-Command Smoke

Eval settings:

- checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_slip_scale_0/sac_lift_step_99328.pkl`
- `policy_mode=both`
- `num_eval_envs=4`
- `episode_length=200`
- action diagnostics and reward components enabled
- fixed command enabled

JSON outputs:

- `./logs/sac_eval_feet_slip_100k_smoke/eval_fwd0p5_seed0_both_4x200_scale0_inherited.json`
- `./logs/sac_eval_feet_slip_100k_smoke/eval_fwd1p0_seed0_both_4x200_scale0_inherited.json`

Both outputs returned `EVAL_OK`, inherited
`reward_config.scales.feet_slip: 0.0`, and had action/reward/obs NaN flags
false.

| Command | Mode | Reward Mean | tracking_lin_vel | termination | reward/feet_slip |
|---|---|---:|---:|---:|---:|
| `[0.5,0,0]` | deterministic | -3.4454 | n/a | n/a | 0.0 |
| `[0.5,0,0]` | stochastic | -5.6693 | n/a | n/a | 0.0 |
| `[1.0,0,0]` | deterministic | -3.4952 | 2.4967 | -100 | 0.0 |
| `[1.0,0,0]` | stochastic | -5.7831 | 3.0799 | -100 | 0.0 |

## Interpretation

- `--env_feet_slip_scale 0.0` successfully removes the feet-slip penalty in
  inherited eval.
- The eval inheritance patch is necessary for reward-component interpretation
  whenever a checkpoint was trained with env overrides.
- Removing the feet-slip penalty improves score accounting, but fixed-forward
  behavior is still weak.
- The `[1.0,0,0]` smoke still has low `tracking_lin_vel` and
  `termination=-100`.
- Therefore `feet_slip` is not the sole blocker.
- This result does not justify 250k, 5M, or 10M training.

## Next Action

Prefer a default-off push-disable diagnostic as the next short gate if feasible.
If push-disable is not the next step, design the phase / `feet_air_time` prior
ablation before touching broader reward terms. Do not tune reward,
`action_scale`, or Kp from this result alone.
