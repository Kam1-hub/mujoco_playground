# Env1024 1M R3 Results

## Context

- Checkpoint: `CHECKPOINT AK - 1024 Env 1M R3 Run`
- Scope: bounded high-parallel 1M R3 validation.
- Runtime stability: PASS.
- This is the first successful bounded `1024`-env `1M` R3 run.
- This is not a policy-quality breakthrough: reward regressed versus R3 250k
  and actor mean/std drift continued.
- No 3M or 10M run was executed.
- No code, report, reward, `action_scale`, Kp, PPO/RSL, domain randomization,
  or fine-tuning change was made during the run.

## Runtime And Artifacts

- Checkpoint:
  `./logs/sac_lift_gpu_1m_env1024_r3_b256_g16_replay1m/sac_lift_step_999424.pkl`
- Eval JSON directory:
  `./logs/sac_eval_env1024_1m_r3_multiseed/`
- Checkpoint exists: yes, about `15M`.
- Checkpoint readiness: PASS.
- `policy_normalizer`: present.
- `value_normalizer`: present.
- `deterministic_eval_ready`: true.
- Five eval JSONs exist; all deterministic/stochastic evals returned
  `EVAL_OK`.
- All action/reward/obs NaN flags were false.

GPU memory:

| Phase | Memory |
|---|---:|
| Pre-run | 853MiB / 12282MiB |
| During 1M train | 9838MiB / 12282MiB |
| Post-train | 853MiB / 12282MiB |
| Eval sample | about 1130MiB / 12282MiB |

## Training Metrics

| Metric | Value |
|---|---:|
| env_steps | 999424 |
| gradient_steps | 15376 |
| wall_time | 311.3788s |
| sps | 3209.6723 |
| actor_loss | -3.8164 |
| critic_loss | 0.0830 |
| alpha | 0.01231 |
| log_alpha | -4.3973 |
| alpha_loss | 0.2878 |
| alpha_log_prob | -16.1272 |
| alpha_error_log_prob_plus_target | -23.3772 |
| alpha_error_neg_log_prob_minus_target | 23.3772 |
| alpha_grad_proxy_exp | 0.2878 |
| q | 3.6304 |
| target_q | 3.6343 |
| reward_mean | -0.1171 |
| done_fraction | 0.01953 |
| discount_mean | 0.98047 |

## Actor Drift Metrics

| Metric | Final | Interval |
|---|---:|---:|
| actor_policy_mean_abs_mean | 0.2299 | 0.2083 |
| actor_policy_mean_abs_max | 2.7383 | 1.9955 |
| actor_log_std_mean | -0.2881 | -0.2291 |
| actor_log_std_min | -1.0246 | -0.8269 |
| actor_log_std_max | 0.0095 | 0.0580 |
| actor_policy_std_mean | 0.7602 | 0.8010 |
| sampled_action_abs_mean | 0.5073 | 0.5103 |
| sampled_action_saturation_fraction_095 | 0.02883 | 0.03361 |
| deterministic_action_abs_mean | 0.2046 | 0.1923 |
| deterministic_action_saturation_fraction_095 | 0.00202 | 0.00046 |

## Regularization Metrics

| Metric | Final | Interval |
|---|---:|---:|
| deterministic_action_l2 | 0.08415 | 0.06840 |
| actor_mean_l2 | 0.12882 | 0.09238 |
| actor_regularization_loss | 0.04852 | 0.03882 |
| deterministic_action_l2_coef | 0.5 | 0.5 |
| actor_mean_l2_coef | 0.05 | 0.05 |

The regularization loss did not dominate actor loss magnitude.

## Deterministic Eval

| Seed | Reward Mean | Std | Min | Max | Action Abs | Mean Abs | Log Std | Std Mean | Sat 0.95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | -4.9282 | 1.5782 | -10.0484 | -3.4129 | 0.2159 | 0.2400 | -0.3064 | 0.7452 | 0.00128 |
| 1 | -4.6388 | 0.9993 | -6.9311 | -3.4377 | 0.2045 | 0.2253 | -0.2887 | 0.7579 | 0.00069 |
| 2 | -4.5887 | 1.3219 | -9.0646 | -3.3700 | 0.1879 | 0.2076 | -0.2603 | 0.7781 | 0.00098 |
| 3 | -4.8827 | 1.6431 | -10.5762 | -3.2844 | 0.2056 | 0.2284 | -0.2830 | 0.7622 | 0.00128 |
| 4 | -5.9070 | 2.7112 | -14.5249 | -3.5745 | 0.2076 | 0.2304 | -0.2951 | 0.7534 | 0.00111 |

Deterministic aggregate:

| Metric | Value |
|---|---:|
| reward mean avg/stdev | -4.9891 / 0.5340 |
| reward min/max avg | -10.2290 / -3.4159 |
| action_abs avg | 0.2043 |
| policy_mean_abs avg | 0.2263 |
| policy_log_std_mean avg | -0.2867 |
| policy_std_mean avg | 0.7594 |
| saturation avg | 0.00107 |

## Stochastic Eval

| Metric | Value |
|---|---:|
| reward mean avg/stdev | -6.7546 / 0.5407 |
| reward min/max avg | -12.4739 / -4.6233 |
| action_log_prob avg | -16.0527 |
| action_abs avg | 0.4948 |
| policy_mean_abs avg | 0.2364 |
| policy_log_std_mean avg | -0.3242 |
| policy_std_mean avg | 0.7341 |
| saturation avg | 0.0267 |

## Reward Components

Deterministic positives:

- `reward/feet_phase`: 37.66
- `reward/tracking_lin_vel`: 16.56
- `reward/tracking_ang_vel`: 9.75

Deterministic negatives:

- `reward/termination`: -100
- `reward/ang_vel_xy`: -76.86
- `reward/orientation`: -48.11
- `reward/stand_still`: -26.38
- `reward/joint_deviation_hip`: -25.02

Stochastic notable negative:

- `reward/ang_vel_xy`: -146.63

## Comparisons

Versus R3 250k:

- actor mean abs: `0.1630 -> 0.2299`
- deterministic action abs: `0.1556 -> 0.2046`
- log_std: `-0.1709 -> -0.2881`
- std: `0.8444 -> 0.7602`
- critic_loss: `0.0536 -> 0.0830`, still finite
- q/target_q: `8.38/8.41 -> 3.63/3.63`, no Q explosion
- deterministic reward: `-3.7941 -> -4.9891`, worse
- stochastic reward: `-5.9802 -> -6.7546`, worse

Versus env1024 65k capacity benchmark:

- SPS: `953.36 -> 3209.67`
- critic_loss: `0.1217 -> 0.0830`
- q/target_q: `3.497/3.491 -> 3.630/3.634`
- actor mean abs: `0.1120 -> 0.2299`
- deterministic action abs: `0.1099 -> 0.2046`

## Warnings

- Known non-fatal WSL2 CUDA driver version warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `uv` snap-confine issue on first attempts; reruns used the same
  parameters externally.
- No traceback, OOM, fatal CUDA/XLA, checkpoint failure, eval failure, or NaN
  was observed.

## Interpretation

Runtime stability supports `1024 envs + replay1M + R3 + UTD~4` on the 12GB
RTX 4070 SUPER. The run reached `999424` env steps, passed checkpoint
readiness, completed 5-seed both-mode eval, and used about `9838MiB` at peak
during training.

Policy quality does not yet justify declaring stable SAC integration. Reward
regressed versus R3 250k and actor mean/std drift continued: actor mean abs and
deterministic action abs increased while log_std/std decreased. This remains a
training-dynamics issue rather than a runtime/backend/checkpoint failure.

Next decision should be a review between:

1. a bounded 3M continuation with the same stable high-parallel setup to test
   whether longer horizon recovers gait learning;
2. diagnostic or regularization adjustment before 3M because 1M reward
   worsened.

Do not jump directly to 10M.
