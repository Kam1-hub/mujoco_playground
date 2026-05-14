# Env1024 3M R3 Results

## Context

- Checkpoint: `CHECKPOINT AM - 1024 Env 3M R3 Run`
- Scope: bounded high-parallel 3M R3 validation.
- Runtime stability: PASS.
- Policy result: mixed. Deterministic eval recovered strongly, but stochastic
  eval degraded and alpha/std collapsed severely.
- No 10M run was executed.
- No code, report, reward, `action_scale`, Kp, PPO/RSL, domain randomization,
  or fine-tuning change was made during the run.

This is the first strong deterministic-policy improvement signal after the
1024-env high-parallel move, but it is not enough to declare stable SAC
integration because entropy-related collapse remains severe.

## Runtime And Artifacts

- Checkpoint:
  `./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl`
- Eval JSON directory:
  `./logs/sac_eval_env1024_3m_r3_multiseed/`
- Checkpoint exists: yes, `15,441,267` bytes.
- Checkpoint readiness: PASS.
- Five eval JSONs exist; all deterministic/stochastic evals returned
  `EVAL_OK`.
- All action/reward/obs NaN flags were false.

GPU memory:

| Phase | Memory |
|---|---:|
| Pre-run | 853MiB |
| During 3M train | 9838MiB / 12282MiB |
| Post-train | 853MiB |

## Training Metrics

| Metric | Value |
|---|---:|
| env_steps | 2999296 |
| gradient_steps | 46624 |
| wall_time | 914.5063s |
| sps | 3279.6889 |
| actor_loss | -1.5595 |
| critic_loss | 0.01014 |
| alpha | 0.000766 |
| log_alpha | -7.1739 |
| alpha_loss | 0.00995 |
| alpha_log_prob | -5.7375 |
| alpha_error_log_prob_plus_target | -12.9875 |
| alpha_error_neg_log_prob_minus_target | 12.9875 |
| alpha_grad_proxy_exp | 0.00995 |
| q | 1.6108 |
| target_q | 1.5903 |
| reward_mean | -0.02250 |
| done_fraction | 0.0 |
| discount_mean | 1.0 |

## Actor Drift Metrics

| Metric | Final | Interval |
|---|---:|---:|
| actor_policy_mean_abs_mean | 0.2071 | 0.2472 |
| actor_policy_mean_abs_max | 3.4852 | 2.9895 |
| actor_log_std_mean | -0.9573 | -0.4745 |
| actor_log_std_min | -2.6765 | - |
| actor_log_std_max | -0.1389 | - |
| actor_policy_std_mean | 0.4106 | 0.6539 |
| sampled_action_abs_mean | 0.3641 | 0.4712 |
| sampled_action_saturation_fraction_095 | 0.00458 | 0.02388 |
| deterministic_action_abs_mean | 0.1845 | 0.2164 |
| deterministic_action_saturation_fraction_095 | 0.00337 | 0.00430 |

## Regularization Metrics

| Metric | Final | Interval |
|---|---:|---:|
| deterministic_action_l2 | 0.06880 | 0.09112 |
| actor_mean_l2 | 0.11501 | 0.15248 |
| actor_regularization_loss | 0.04015 | 0.05318 |
| deterministic_action_l2_coef | 0.5 | 0.5 |
| actor_mean_l2_coef | 0.05 | 0.05 |

The regularization loss did not dominate actor loss magnitude.

## Deterministic Eval

| Metric | Value |
|---|---:|
| reward mean avg/stdev | -2.3314 / 1.9412 |
| reward min/max avg | -18.3913 / 9.0293 |
| action_abs avg | 0.1572 |
| policy_mean_abs avg | 0.1739 |
| policy_log_std_mean avg | -0.8741 |
| policy_std_mean avg | 0.4443 |
| saturation avg | 0.00255 |

## Stochastic Eval

| Metric | Value |
|---|---:|
| reward mean avg/stdev | -10.9904 / 2.5724 |
| reward min/max avg | -35.7932 / -3.1370 |
| action_log_prob avg | -6.3099 |
| action_abs avg | 0.3616 |
| policy_mean_abs avg | 0.1970 |
| policy_log_std_mean avg | -0.9378 |
| policy_std_mean avg | 0.4188 |
| saturation avg | 0.00463 |

## Comparisons

Versus 1M R3:

- Deterministic reward improved: `-4.9891 -> -2.3314`.
- Stochastic reward worsened: `-6.7546 -> -10.9904`.
- Actor mean abs improved: `0.2299 -> 0.2071`.
- Deterministic action abs improved: `0.2046 -> 0.1845`.
- Alpha collapsed: `0.01231 -> 0.000766`.
- Log std collapsed: `-0.2881 -> -0.9573`.
- Std collapsed: `0.7602 -> 0.4106`.

Versus R3 250k:

- Deterministic reward improved: `-3.7941 -> -2.3314`.
- Stochastic reward worsened: `-5.9802 -> -10.9904`.

## Warnings

- Known non-fatal WSL2 CUDA driver version warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `uv` snap-confine issue on first attempts; reruns used the same
  parameters externally.
- No traceback, OOM, fatal CUDA/XLA, checkpoint failure, eval failure, or NaN
  was observed.

## Interpretation

Runtime stability supports `1024 envs + replay1M + R3 + UTD~4` on the 12GB
RTX 4070 SUPER through 3M env steps. The run reached `2,999,296` env steps,
passed checkpoint readiness, completed 5-seed both-mode eval, and used about
`9838MiB / 12282MiB` during training.

Policy quality is mixed. The deterministic deployment/eval path improved
strongly and is the first clear positive long-horizon signal in this SAC
sequence. However, stochastic policy behavior degraded sharply and alpha/std
collapsed. This means the result is not a clean stable-SAC declaration.

Next decision should review entropy/alpha handling before any 10M run:

1. whether alpha/log_std need a floor or alternate entropy handling;
2. whether to run an entropy ablation before longer training;
3. whether to add a render helper for deterministic visual inspection;
4. whether the deterministic-policy improvement justifies a carefully gated
   longer run after the entropy review.

Do not jump directly to 10M from this result.
