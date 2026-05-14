# High-Parallel SAC Capacity Results

## Context

- Checkpoint: `CHECKPOINT AI - High Parallel Capacity Benchmark`
- Scope: capacity benchmark only; no extra eval, code change, report change, or
  commit was performed during the benchmark run.
- Baseline: R3 actor-regularization settings with coordinated off-policy update
  ratios.
- Goal: test 512, 1024, and 2048 parallel envs on the 12GB RTX 4070 SUPER
  while preserving approximate sampled update-to-data ratio.

All three runs produced `TRAIN_OK`, wrote checkpoints under ignored `logs/`,
and passed checkpoint readiness. No NaN, Inf, OOM, fatal CUDA, checkpoint
failure, or unignored artifact was reported.

## Checkpoints

| Case | Status | Checkpoint | Readiness |
|---|---|---|---|
| env512 | `TRAIN_OK` | `./logs/sac_capacity_env512_65k_r3_b256_g8_replay262k/sac_lift_step_65536.pkl` | PASS |
| env1024 | `TRAIN_OK` | `./logs/sac_capacity_env1024_65k_r3_b256_g16_replay262k/sac_lift_step_65536.pkl` | PASS |
| env2048 | `TRAIN_OK` | `./logs/sac_capacity_env2048_65k_r3_b256_g32_replay262k/sac_lift_step_65536.pkl` | PASS |

## Capacity Metrics

| envs | ckpt size | UTD | grad_updates | wall_time | SPS | actor_loss | critic_loss | alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 14.726 MiB | 4.0 | 776 | 84.768s | 773.12 | -4.0305 | 0.1166 | 0.04608 | 3.1799 | 3.1973 |
| 1024 | 14.726 MiB | 4.0 | 784 | 68.742s | 953.36 | -4.3618 | 0.1217 | 0.04605 | 3.4970 | 3.4905 |
| 2048 | 14.726 MiB | 4.0 | 800 | 79.317s | 826.26 | -4.6280 | 0.0706 | 0.04597 | 3.7519 | 3.7731 |

## Training State

| envs | env_steps | log_alpha | alpha_loss | reward_mean | done_frac | discount_mean |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 65536 | -3.0774 | 1.2250 | -0.1489 | 0.03125 | 0.96875 |
| 1024 | 65536 | -3.0781 | 1.2194 | -0.1163 | 0.015625 | 0.984375 |
| 2048 | 65536 | -3.0797 | 1.2206 | -0.0931 | 0.003906 | 0.996094 |

## Actor And Action Metrics

| envs | mean_abs | mean_abs_max | log_std mean/min/max | std | sampled_abs | sampled_sat | det_abs | det_sat |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 512 | 0.1081 | 0.6758 | -0.1388 / -0.4862 / 0.1194 | 0.8727 | 0.5173 | 0.0381 | 0.1063 | 0.0 |
| 1024 | 0.1120 | 0.7641 | -0.1333 / -0.4376 / 0.1948 | 0.8775 | 0.5195 | 0.0411 | 0.1099 | 0.0 |
| 2048 | 0.1103 | 1.0103 | -0.1368 / -0.4474 / 0.1750 | 0.8743 | 0.5187 | 0.0392 | 0.1078 | 0.0 |

## Regularization And Config

| envs | det_l2 | mean_l2 | reg_loss | batch | gups | min_replay | max_replay | target_entropy_coef | alpha_lr | det_l2_coef | mean_l2_coef |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.01979 | 0.02098 | 0.01094 | 256 | 8 | 16384 | 262144 | 0.25 | 0.0001 | 0.5 | 0.05 |
| 1024 | 0.02104 | 0.02243 | 0.01164 | 256 | 16 | 16384 | 262144 | 0.25 | 0.0001 | 0.5 | 0.05 |
| 2048 | 0.02200 | 0.02407 | 0.01221 | 256 | 32 | 16384 | 262144 | 0.25 | 0.0001 | 0.5 | 0.05 |

## Interpretation

- Sample UTD was preserved at about `4.0` by scaling
  `grad_updates_per_step` to `8 / 16 / 32` for `512 / 1024 / 2048` envs.
- `1024` envs was the best bounded 1M candidate: it was fastest at
  `953.36` SPS, passed readiness, reported no NaN/Inf, and kept actor and
  regularization metrics stable.
- `2048` envs is feasible, but it was slower than `1024` envs in this test, so
  it should not replace `1024` yet.
- `512` envs is stable but slower.
- The selected follow-up was a bounded 1024-env 1M run using R3 settings,
  `grad_updates_per_step=16`, and `max_replay_size=1000000`; see below.

## Follow-Up

The bounded 1024-env 1M R3 follow-up has now completed. It reached `999424`
env steps, passed checkpoint readiness, and passed 5-seed both-mode eval with
no NaN flags. Runtime stability supports the high-parallel configuration, but
policy quality did not improve versus R3 250k. See
`reports/sac_integration/15_env1024_1m_r3_results.md` for details before
planning any 3M or 10M run.
