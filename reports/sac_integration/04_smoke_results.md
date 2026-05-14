# Smoke Results

Status: updated on 2026-05-14 after the bounded 1024-env 1M R3 run.

## 2026-05-14 Bounded 1024-Env 1M R3 Run

- Scope: bounded high-parallel 1M validation with R3 settings; no 3M/10M,
  code change, or report change was performed during the run.
- Runtime stability: PASS; this is the first successful bounded `1024`-env
  `1M` R3 run.
- Policy quality: not a breakthrough. Reward regressed versus R3 250k and
  actor mean/std drift continued.
- Checkpoint:
  `./logs/sac_lift_gpu_1m_env1024_r3_b256_g16_replay1m/sac_lift_step_999424.pkl`
- Eval JSONs: `./logs/sac_eval_env1024_1m_r3_multiseed/`
- Checkpoint exists at about `15M`; readiness PASS; `policy_normalizer` and
  `value_normalizer` present; `deterministic_eval_ready=true`.
- Five eval JSONs: all `EVAL_OK`; all action/reward/obs NaN flags false.
- GPU memory: pre-run `853MiB / 12282MiB`; during train
  `9838MiB / 12282MiB`; post-train `853MiB / 12282MiB`; eval sample about
  `1130MiB / 12282MiB`.
- No traceback, OOM, fatal CUDA/XLA, checkpoint failure, eval failure, or NaN
  was observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 999424 | 15376 | 311.3788s | 3209.6723 | -3.8164 | 0.0830 | 0.01231 | -4.3973 | 3.6304 | 3.6343 |

Actor and regularization summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.2299 | 0.2083 |
| deterministic action abs | 0.2046 | 0.1923 |
| log_std mean | -0.2881 | -0.2291 |
| std mean | 0.7602 | 0.8010 |
| sampled action abs | 0.5073 | 0.5103 |
| deterministic action saturation 0.95 | 0.00202 | 0.00046 |
| deterministic action L2 | 0.08415 | 0.06840 |
| actor mean L2 | 0.12882 | 0.09238 |
| actor regularization loss | 0.04852 | 0.03882 |

Eval aggregate:

| Mode | Reward Avg | Reward SD | Reward Min Avg | Reward Max Avg | Action Abs | Mean Abs | Log Std | Std Mean | Sat 0.95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.9891 | 0.5340 | -10.2290 | -3.4159 | 0.2043 | 0.2263 | -0.2867 | 0.7594 | 0.00107 |
| stochastic | -6.7546 | 0.5407 | -12.4739 | -4.6233 | 0.4948 | 0.2364 | -0.3242 | 0.7341 | 0.0267 |

Interpretation: runtime stability supports `1024 envs + replay1M + R3 +
UTD~4`. Policy quality remains unresolved: versus R3 250k, actor mean abs
rose `0.1630 -> 0.2299`, deterministic action abs rose `0.1556 -> 0.2046`,
log_std narrowed `-0.1709 -> -0.2881`, deterministic reward worsened
`-3.7941 -> -4.9891`, and stochastic reward worsened `-5.9802 -> -6.7546`.
Do not jump directly to 10M; next step should be a decision review between a
bounded 3M continuation and further diagnostic/regularization adjustment.

## 2026-05-14 High-Parallel Capacity Benchmark

- Scope: capacity benchmark only; no extra eval, code change, or commit during
  the benchmark run.
- Cases: `512`, `1024`, and `2048` envs with R3 settings and UTD-preserving
  `grad_updates_per_step=8/16/32`.
- All three cases: `TRAIN_OK`, checkpoint exists, checkpoint readiness PASS.
- Checkpoints:
  - `./logs/sac_capacity_env512_65k_r3_b256_g8_replay262k/sac_lift_step_65536.pkl`
  - `./logs/sac_capacity_env1024_65k_r3_b256_g16_replay262k/sac_lift_step_65536.pkl`
  - `./logs/sac_capacity_env2048_65k_r3_b256_g32_replay262k/sac_lift_step_65536.pkl`
- No NaN, Inf, OOM, fatal CUDA, checkpoint failure, or unignored artifact was
  reported.

Summary:

| envs | UTD | gradient_steps | wall_time | SPS | critic_loss | alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 4.0 | 776 | 84.768s | 773.12 | 0.1166 | 0.04608 | 3.1799 | 3.1973 |
| 1024 | 4.0 | 784 | 68.742s | 953.36 | 0.1217 | 0.04605 | 3.4970 | 3.4905 |
| 2048 | 4.0 | 800 | 79.317s | 826.26 | 0.0706 | 0.04597 | 3.7519 | 3.7731 |

Interpretation: `1024` envs was selected for the bounded 1M follow-up. It was
fastest at `953.36` SPS and retained finite stable metrics with readiness PASS.
`2048` envs is feasible but slower in this test; `512` envs is stable but
slower. Full detail is in
`reports/sac_integration/14_high_parallel_capacity_results.md`.

## 2026-05-14 R3 250k Actor-Regularization Extension

- Scope: bounded R3 250k extension using A4 alpha settings and the strongest
  100k regularization coefficients; no 500k, 750k, 1M, or code change was
  executed.
- Parameters: `target_entropy_coef=0.25`, `alpha_learning_rate=1e-4`,
  `deterministic_action_l2_coef=0.5`, `actor_mean_l2_coef=0.05`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_actor_reg_te0p25_alr1e4_l2_0p5_mean_0p05_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS; `policy_normalizer` and `value_normalizer`
  present; `deterministic_eval_ready=true`.
- 4 env x 200 action diagnostic eval: PASS / `EVAL_OK`, JSON
  `./logs/sac_eval_actor_reg_250k/eval_R3_seed0_4x200_actiondiag.json`.
- 5-seed 16 env x 1000 eval: PASS, five JSONs in
  `./logs/sac_eval_actor_reg_250k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, replay, checkpoint, or eval failure was
  observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 249984 | 3892 | 145.9190 | 1713.1695 | -9.0065 | 0.05362 | 0.034497 | -3.36687 | 8.3828 | 8.4127 |

Actor and regularization summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.163048 | 0.143159 |
| deterministic action abs | 0.155556 | 0.136790 |
| log_std mean | -0.170873 | -0.152856 |
| std mean | 0.844429 | 0.861275 |
| sampled action abs | 0.526266 | 0.519440 |
| deterministic action saturation 0.95 | 0.000269 | 0.000039 |
| deterministic action L2 | 0.043000 | 0.036152 |
| actor mean L2 | 0.052856 | 0.044359 |
| actor regularization loss | 0.024143 | 0.020294 |

Eval summary:

| Eval | Mode | Reward Avg/Mean | Reward SD | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---|
| 4x200 seed0 | deterministic | -3.27499 | 0.28916 | 0.11736 | 0.0 | false |
| 4x200 seed0 | stochastic | -5.86309 | 0.16949 | 0.51463 | 0.03552 | false |
| 5-seed 16x1000 | deterministic | -3.7941 | 0.2941 | 0.1335 | 0.0 | false |
| 5-seed 16x1000 | stochastic | -5.9802 | 0.3177 | 0.5147 | 0.0360 | false |

Interpretation: R3 retained actor mean / deterministic action drift control at
250k. Versus R3 100k, train mean abs rose only `0.1506 -> 0.1630`,
deterministic action abs rose `0.1430 -> 0.1556`, deterministic 5-seed reward
improved `-4.1681 -> -3.7941`, stochastic 5-seed reward improved
`-6.3983 -> -5.9802`, and critic loss improved `0.1684 -> 0.0536`. Versus A4
250k, R3 has lower action magnitude and better deterministic/stochastic eval.
This supports R3 as the current stability candidate, but it is not a long-run
policy-quality claim.

## 2026-05-14 High-Parallel Capacity Plan

- User noted that `num_envs=128` is likely too conservative and that G1 may
  need multi-million or 10M-scale experience before useful gait quality appears.
- The 10k through 500k ladder should be treated as runtime/diagnostic evidence,
  not policy-quality evidence.
- Current training loop means increasing `num_envs` without increasing
  `grad_updates_per_step` lowers the sample update-to-data ratio.
- With `batch_size=256`, preserving the 128-env baseline sampled UTD of about
  `4` implies:
  - `512 envs`: `grad_updates_per_step=8`
  - `1024 envs`: `grad_updates_per_step=16`
  - `2048 envs`: `grad_updates_per_step=32`
- This plan was executed in the high-parallel capacity benchmark above.

## 2026-05-14 Fresh 100k Actor-Regularization R2/R3 Sweep

- Scope: fresh 100k actor-regularization coefficient sweep using A4 alpha
  settings only; no 250k, 500k, 750k, or 1M run was executed.
- A4 alpha settings: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- R2: `deterministic_action_l2_coef=0.1`,
  `actor_mean_l2_coef=0.01`.
- R3: `deterministic_action_l2_coef=0.5`,
  `actor_mean_l2_coef=0.05`.
- Both variants: `TRAIN_OK`, checkpoint readiness PASS, 4 env x 200 action
  diagnostic eval PASS, and 5-seed 16 env x 1000 eval PASS.
- Multi-seed outputs: `./logs/sac_eval_actor_reg_100k_multiseed/`, `15`
  total JSON files including R1/R2/R3.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.
- Runtime artifacts remain under ignored `logs/` and are not committed.

Training and actor summary:

| Variant | actor_loss | critic_loss | alpha | log_alpha | q | target_q | reward_mean | sps | mean_abs | det_abs | log_std | std | reg_loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | -5.9766 | 0.1383 | 0.042841 | -3.1503 | 5.2078 | 5.1507 | -0.1561 | 1396.73 | 0.2012 | 0.1887 | -0.1513 | 0.8618 | 0.00700 |
| R3 | -6.0585 | 0.1684 | 0.042818 | -3.1508 | 5.3317 | 5.2599 | -0.1340 | 1414.64 | 0.1506 | 0.1430 | -0.1548 | 0.8589 | 0.02259 |

5-seed eval aggregate:

| Variant | Mode | Reward Avg | Reward SD | Action Abs | Policy Mean Abs | Log Std | Std | OK |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R2 | deterministic | -4.2422 | 0.3848 | 0.1614 | 0.1720 | -0.0980 | 0.9084 | true |
| R2 | stochastic | -6.4680 | 0.4782 | 0.5247 | 0.1943 | -0.1527 | 0.8605 | true |
| R3 | deterministic | -4.1681 | 0.4120 | 0.1348 | 0.1438 | -0.1014 | 0.9058 | true |
| R3 | stochastic | -6.3983 | 0.4580 | 0.5198 | 0.1483 | -0.1556 | 0.8582 | true |

Interpretation: R2 improves over A4 100k and R1 on drift metrics and eval
reward. R3 improves more strongly and has the best 5-seed deterministic and
stochastic rewards among R2/R3/A4 100k/R1. R3 is now the stronger 100k
regularization candidate, with critic loss `0.1684` as a watch item; R2 remains
a conservative backup. Next step is a decision review before any bounded R3
250k extension. Do not run 500k, 750k, or 1M from this result.

## 2026-05-14 Fresh 100k Actor-Regularization R1

- Scope: fresh 100k regularization ablation using A4 alpha settings plus
  `deterministic_action_l2_coef=0.01` and `actor_mean_l2_coef=0.001`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; normalizers
  present.
- 4 env x 200 seed 0 action diagnostic eval: PASS / `EVAL_OK`, JSON
  `./logs/sac_eval_actor_reg_100k/eval_R1_seed0_4x200_actiondiag.json`.
- 5-seed 16 env x 1000 eval: PASS, five JSONs in
  `./logs/sac_eval_actor_reg_100k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made during the run.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99968 | 1548 | 71.0452 | 1407.1039 | -5.7620 | 0.06915 | 0.042852 | -3.149996 | 4.9464 | 4.9560 |

Actor drift and regularization summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.23127 | 0.16341 |
| deterministic action abs | 0.21378 | 0.15691 |
| log_std mean | -0.15577 | -0.13456 |
| std mean | 0.85809 | 0.87862 |
| deterministic action L2 | 0.07827 | 0.04244 |
| actor mean L2 | 0.10289 | 0.04958 |
| actor regularization loss | 0.000886 | 0.000474 |

Eval summary:

| Eval | Mode | Reward Avg/Mean | Reward SD | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---|
| 4x200 seed0 | deterministic | -4.1646 | 0.4782 | 0.1652 | 0.0 | false |
| 4x200 seed0 | stochastic | -6.3677 | 0.4307 | 0.5296 | 0.04487 | false |
| 5-seed 16x1000 | deterministic | -4.3587 | 0.3786 | 0.17245 | 0.00000043 | false |
| 5-seed 16x1000 | stochastic | -6.6210 | 0.4996 | 0.52716 | 0.04406 | false |

Interpretation: R1 is runtime clean, but it does not reduce train-time actor
mean or deterministic action magnitude versus the A4 100k baseline. The final
regularization contribution `0.000886` is tiny relative to actor loss, so this
coefficient set appears too weak. Do not extend R1 to 250k as-is; next action
should be a decision review or stronger bounded coefficient sweep, not
250k/750k/1M.

## 2026-05-14 Fresh 750k A4 Alpha/Entropy Bridge

- Scope: bounded fresh 750k A4 bridge only; not 1M.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; normalizers
  present.
- 4 env x 200 seed 0 eval: PASS / `EVAL_OK`, JSON
  `./logs/sac_eval_alpha_ablate_750k/eval_A4_seed0_4x200_actiondiag.json`
- 5-seed 16 env x 1000 eval: PASS, five JSONs in
  `./logs/sac_eval_alpha_ablate_750k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 749952 | 11704 | 397.2631 | 1887.7966 | -6.5525 | 0.1441 | 0.017246 | -4.06016 | 6.1746 | 6.2776 |

Actor drift summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.37190 | 0.27588 |
| deterministic action abs | 0.31850 | 0.24939 |
| log_std mean | -0.24975 | -0.19484 |
| std mean | 0.78393 | 0.82621 |

Eval summary:

| Eval | Mode | Reward Avg/Mean | Reward SD | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---|
| 4x200 seed0 | deterministic | -5.1279 | 0.5785 | 0.3075 | 0.00151 | false |
| 4x200 seed0 | stochastic | -5.9461 | 0.2064 | 0.5247 | 0.04280 | false |
| 5-seed 16x1000 | deterministic | -5.7314 | 0.3095 | 0.3027 | 0.00201 | false |
| 5-seed 16x1000 | stochastic | -6.3802 | 0.4577 | 0.5274 | 0.04521 | false |

Interpretation: A4 750k is runtime stable but not a clean stability
improvement. Versus A4 500k, alpha declined `0.02435 -> 0.01725`, actor mean
abs rose `0.29323 -> 0.37190`, deterministic action abs rose
`0.26540 -> 0.31850`, log_std narrowed `-0.22279 -> -0.24975`, critic loss
rose `0.0803 -> 0.1441`, deterministic 5-seed eval worsened
`-4.4075 -> -5.7314`, and stochastic 5-seed eval worsened
`-6.0434 -> -6.3802`. Q/target_q moved down from A4 500k
`8.47/8.41 -> 6.17/6.28`, but the drift/eval degradation blocks any
automatic 1M or longer run.

## 2026-05-14 Fresh 500k A4 Alpha/Entropy Extension

- Scope: bounded fresh 500k A4 extension only; not 750k or 1M.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; normalizers
  present.
- 4 env x 200 seed 0 eval: PASS / `EVAL_OK`, JSON
  `./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json`
- 5-seed 16 env x 1000 eval: PASS, five JSONs in
  `./logs/sac_eval_alpha_ablate_500k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 499968 | 7798 | 271.9703 | 1838.3185 | -8.8341 | 0.0803 | 0.02435 | -3.71524 | 8.47294 | 8.41017 |

Actor drift summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.29323 | 0.23339 |
| deterministic action abs | 0.26540 | 0.21651 |
| log_std mean | -0.22279 | -0.17587 |
| std mean | 0.80245 | 0.84132 |

Eval summary:

| Eval | Mode | Reward Avg/Mean | Reward SD | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---|
| 4x200 seed0 | deterministic | -4.3446 | 0.4644 | 0.2304 | 0.0 | false |
| 4x200 seed0 | stochastic | -6.2873 | 0.6175 | 0.5148 | 0.0342 | false |
| 5-seed 16x1000 | deterministic | -4.4075 | 0.3493 | 0.2289 | 0.0000004 | false |
| 5-seed 16x1000 | stochastic | -6.0434 | 0.5364 | 0.5152 | 0.0366 | false |

Interpretation: A4 500k mitigates the old fresh 500k deterministic drift
pattern on alpha (`0.02435` vs old `0.00801`), actor mean/action magnitude,
log_std/std, and deterministic eval reward. It still drifts relative to A4
250k, and Q/target_q plus critic loss remain watch items. This is a bounded
diagnostic PASS, not authorization to run 750k or 1M.

## 2026-05-14 Fresh 250k A4 Alpha/Entropy Extension

- Scope: bounded fresh 250k A4 extension only; not 500k, 750k, or 1M.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; normalizers
  present.
- 4 env x 200 seed 0 eval: PASS / `EVAL_OK`, JSON
  `./logs/sac_eval_alpha_ablate_250k/eval_A4_seed0_4x200_actiondiag.json`
- 5-seed 16 env x 1000 eval: PASS, five JSONs in
  `./logs/sac_eval_alpha_ablate_250k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training summary:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 249984 | 3892 | 148.8921 | 1678.9605 | -9.3541 | 0.05245 | 0.03455 | -3.36536 | 8.7442 | 8.7170 |

Actor drift summary:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.22572 | 0.19623 |
| deterministic action abs | 0.21220 | 0.18468 |
| log_std mean | -0.17093 | -0.15217 |
| std mean | 0.84424 | 0.86181 |

Eval summary:

| Eval | Mode | Reward Avg/Mean | Reward SD | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---|
| 4x200 seed0 | deterministic | -4.0879 | n/a | 0.1739 | 0.0 | false |
| 4x200 seed0 | stochastic | -5.9671 | n/a | 0.5213 | 0.03970 | false |
| 5-seed 16x1000 | deterministic | -4.1561 | 0.4168 | 0.1861 | 0.000006 | false |
| 5-seed 16x1000 | stochastic | -6.1400 | 0.4045 | 0.5213 | 0.04050 | false |

Interpretation: A4 clearly mitigates 250k actor mean / deterministic action
drift versus the fresh 250k baseline: alpha is higher by `+0.01578`, actor
mean abs is lower by `-0.07330`, deterministic action abs is lower by
`-0.05875`, log_std is less negative by `+0.03434`, and std is higher by
`+0.02802`. A4 does not eliminate drift relative to A4 100k, and Q/target_q
are higher than earlier baselines, so they are watch items. This is promising
drift-control evidence that justified the subsequent bounded A4 500k decision
review. It did not authorize 750k or 1M.

## 2026-05-14 Alpha/Entropy Ablation Multi-Seed Eval-Only Diagnostic

- Scope: eval-only follow-up for fresh 100k A1/A3/A4 ablation checkpoints.
- JSON directory: `./logs/sac_eval_alpha_ablate_multiseed/`
- JSON count: `15`
- Checkpoint readiness: A1/A3/A4 PASS; all normalizer/eval-ready checks passed.
- Eval setup: seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`,
  `--policy_mode both`, `--action_diagnostics`, `--reward_components`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No training, code change, report change, traceback, OOM, fatal CUDA,
  checkpoint failure, or eval failure occurred before this report update.

Deterministic aggregate:

| Variant | Reward Avg | Reward SD | Action Abs | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | -4.3262 | 0.4001 | 0.1826 | 0.1961 | -0.1065 | 0.9006 | true |
| A3 | -4.5518 | 0.4394 | 0.1906 | 0.2076 | -0.1107 | 0.8972 | true |
| A4 | -4.3436 | 0.3791 | 0.1774 | 0.1920 | -0.1058 | 0.9015 | true |

Stochastic aggregate:

| Variant | Reward Avg | Reward SD | Action Abs | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | -6.3990 | 0.3916 | 0.5259 | 0.2054 | -0.1534 | 0.8599 | true |
| A3 | -6.4888 | 0.4957 | 0.5285 | 0.2291 | -0.1599 | 0.8548 | true |
| A4 | -6.4935 | 0.5123 | 0.5266 | 0.2067 | -0.1540 | 0.8595 | true |

Interpretation: A4 remains the best drift-control candidate because it has the
lowest deterministic action magnitude and actor mean magnitude. It is not
strictly best on deterministic reward: A1 is slightly better
(`-4.3262` vs `-4.3436`), a small gap relative to seed variance. A4 stochastic
reward is worse than A1 by about `0.0945` and essentially tied with A3. This
result supported later bounded A4 extensions, but it did not authorize 750k or
1M.

## 2026-05-14 Fresh 100k Alpha/Entropy Ablation Diagnostics

- Scope: fresh 100k ablation training plus small action diagnostic eval.
- Variants:
  - A1: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.5`.
  - A3: `alpha_learning_rate=3e-4`, `target_entropy_coef=0.25`.
  - A4: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`.
- A4 was run because A1 and A3 both passed runtime, checkpoint, and eval gates.
- No 250k, 500k, 750k, or 1M run was executed in this ablation phase.
- No reward, `action_scale`, Kp, PPO, RSL, domain-randomization, or
  fine-tuning changes were made.
- Runtime artifacts are under ignored `logs/` and are not committed.

Gate summary:

| Variant | Train | Checkpoint readiness | Eval | JSON |
|---|---:|---:|---:|---|
| A1 | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/alr1e4_seed0_4x200_actiondiag.json` |
| A3 | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/te0p25_seed0_4x200_actiondiag.json` |
| A4 | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/te0p25_alr1e4_seed0_4x200_actiondiag.json` |

Training summary:

| Variant | Checkpoint | env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | `./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 72.58993083500536 | 1377.1607005277926 | -5.957967758178711 | 0.051446348428726196 | 0.04285280779004097 | -3.149984121322632 | 5.1932373046875 | 5.197819709777832 |
| A3 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 70.95217173699348 | 1408.9491209735313 | -4.622934341430664 | 0.05104774236679077 | 0.032598935067653656 | -3.423475742340088 | 4.00905704498291 | 4.069530963897705 |
| A4 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 71.56172043700644 | 1396.947968683882 | -5.977431297302246 | 0.04582885652780533 | 0.04284820705652237 | -3.1500914096832275 | 5.179529666900635 | 5.181567192077637 |

Actor drift comparison against fresh 100k baseline
(`alpha=0.032585`, actor mean abs `0.238568`, deterministic action abs
`0.218864`, log_std mean `-0.157116`, std mean `0.857329`):

| Variant | actor mean abs | deterministic action abs | log_std mean | std mean | Interpretation |
|---|---:|---:|---:|---:|---|
| A1 | 0.21597573161125183 | 0.20185625553131104 | -0.15477555990219116 | 0.8587937355041504 | Better than baseline on drift metrics |
| A3 | 0.24479639530181885 | 0.22468040883541107 | -0.15980812907218933 | 0.8549157381057739 | Slightly worse than baseline |
| A4 | 0.2069278359413147 | 0.19313891232013702 | -0.15182653069496155 | 0.8613420128822327 | Best 100k drift candidate |

Small eval summary, seed 0, 4 env x 200 steps:

| Variant | Mode | Reward Mean | Reward SD | Reward Min | Reward Max | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | deterministic | -4.416665077209473 | 0.3470674157142639 | -4.957084655761719 | -4.006556987762451 | 0.18505924940109253 | 0.0 | false |
| A1 | stochastic | -6.050605773925781 | 0.42214730381965637 | -6.710614204406738 | -5.661291599273682 | 0.528386652469635 | 0.04306034743785858 | false |
| A3 | deterministic | -4.555072784423828 | 0.6614199280738831 | -5.202037334442139 | -3.4488778114318848 | 0.18952174484729767 | 0.000043103449570480734 | false |
| A3 | stochastic | -6.32877779006958 | 0.5614966750144958 | -7.13820743560791 | -5.716131210327148 | 0.532009482383728 | 0.04543103650212288 | false |
| A4 | deterministic | -4.366635799407959 | 0.3521541357040405 | -4.8208231925964355 | -3.8527913093566895 | 0.17405447363853455 | 0.0 | false |
| A4 | stochastic | -6.495170593261719 | 0.6918737888336182 | -7.607295036315918 | -5.806713104248047 | 0.5309661030769348 | 0.046120692044496536 | false |

Interpretation: A4 is the best current 100k candidate because it preserved
higher alpha, reduced actor mean and deterministic action magnitude, kept
log_std/std healthier than baseline, and had the best deterministic 4x200 reward
among A1/A3/A4. Caveat: A4 stochastic 4x200 reward was worse than A1/A3, so
this is a diagnostic signal, not a final policy-quality benchmark.

Warnings: known non-fatal WSL2 CUDA driver version warning, known non-fatal JAX
cast overflow warning, and sandbox `snap-confine` execution noise. No traceback,
NaN, OOM, fatal CUDA error, checkpoint failure, or eval failure was observed.

## 2026-05-13 Fresh 250k Actor Drift Diagnostic

- Scope: fresh diagnostic training run plus small eval-only action diagnostic.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
- Eval JSON:
  `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`
- Eval status: `EVAL_OK`
- JSON sanity: PASS
- No fresh 500k, 750k, or 1M run was executed.
- No code or report changes were made during the diagnostic run.

Training metrics:

- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `140.85426465800265`
- `sps`: `1774.7705446261555`
- `actor_loss`: `-5.850378513336182`
- `critic_loss`: `0.03159185126423836`
- `alpha`: `0.01876842975616455`
- `log_alpha`: `-3.975579023361206`
- `alpha_loss`: `0.587762713432312`
- `alpha_log_prob`: `-16.816566467285156`
- `q`: `5.547477722167969`
- `target_q`: `5.520053863525391`
- `reward_mean`: `-0.12411123514175415`
- `done_fraction`: `0.01953125`
- `discount_mean`: `0.98046875`

Actor drift metrics:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.2990209758281708 | 0.233315885204818 |
| actor policy mean abs max | 1.9897916316986084 | 1.739523811275398 |
| actor log_std mean | -0.20526975393295288 | -0.16304866696410225 |
| actor log_std min | -0.7766156792640686 | -0.6616444636331555 |
| actor log_std max | 0.07799282670021057 | 0.1512706000589979 |
| actor policy std mean | 0.8162157535552979 | 0.8529925538726603 |
| sampled action abs mean | 0.535484254360199 | 0.5281302615586679 |
| sampled action saturation 0.95 | 0.04431573301553726 | 0.04565783552844594 |
| deterministic action abs mean | 0.2709442377090454 | 0.21555377979248855 |
| deterministic action saturation 0.95 | 0.0005387931014411151 | 0.00023288404075520971 |

Fresh 100k comparison:

- Actor mean abs increased: final `0.238568 -> 0.299021`, interval
  `0.170754 -> 0.233316`.
- Deterministic action abs increased: final `0.218864 -> 0.270944`, interval
  `0.163467 -> 0.215554`.
- Log_std/std continued downward: final log_std `-0.157116 -> -0.205270`,
  final std `0.857329 -> 0.816216`.
- Alpha declined: `0.032585 -> 0.018768`.
- Interpretation: drift amplifies in absolute level by fresh 250k.

Small action diagnostic eval:

| Mode | Reward Mean | Reward SD | Reward Min | Reward Max | Action Abs | Sat 0.95 | Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.01785135269165 | 0.34408432245254517 | -4.546696662902832 | -3.588685989379883 | 0.18537074327468872 | 0.0 | 0.19577054679393768 | -0.13236531615257263 | 0.8772842288017273 |
| stochastic | -5.697283744812012 | 0.1435307413339615 | -5.888537883758545 | -5.491363525390625 | 0.5233082175254822 | 0.04060344770550728 | 0.269175261259079 | -0.20079341530799866 | 0.8196967244148254 |

Reward component highlights:

- Deterministic negatives: `reward/termination -100`,
  `reward/orientation -47.33`, `reward/ang_vel_xy -44.81`,
  `reward/joint_deviation_hip -31.37`, `reward/feet_slip -14.83`.
- Deterministic positives: `reward/feet_phase 25.82`,
  `reward/tracking_ang_vel 20.20`, `reward/tracking_lin_vel 2.90`.
- Stochastic negatives: `reward/ang_vel_xy -108.26`,
  `reward/termination -100`, `reward/orientation -46.87`,
  `reward/joint_deviation_hip -35.09`, `reward/feet_slip -10.07`.
- Stochastic positives: `reward/feet_phase 27.63`,
  `reward/tracking_lin_vel 4.17`, `reward/tracking_ang_vel 3.64`.

Warnings: known non-fatal WSL2 CUDA driver version warning, known non-fatal JAX
cast overflow warning, and sandbox `snap-confine` capability errors on first
`uv` attempts. The same commands were rerun externally with unchanged
parameters. No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or
eval failure was observed.

## 2026-05-13 Fresh 100k Actor Drift Diagnostic

- Scope: fresh diagnostic training run plus small eval-only action diagnostic.
- Code baseline: `202c6a9 Add SAC actor drift train diagnostics`.
- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS
- Eval JSON:
  `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`
- Eval status: `EVAL_OK`
- JSON sanity: PASS
- No 250k, 750k, or 1M run was executed.
- No code or report changes were made during the diagnostic run.

Training metrics:

- `env_steps`: `99968`
- `gradient_steps`: `1548`
- `wall_time`: `68.07941276300699`
- `sps`: `1468.4027952473857`
- `actor_loss`: `-4.796189308166504`
- `critic_loss`: `0.04373161494731903`
- `alpha`: `0.03258506953716278`
- `log_alpha`: `-3.423901081085205`
- `alpha_loss`: `1.0434787273406982`
- `alpha_log_prob`: `-17.523212432861328`
- `alpha_error_log_prob_plus_target`: `-32.02321243286133`
- `alpha_error_neg_log_prob_minus_target`: `32.02321243286133`
- `alpha_grad_proxy_exp`: `1.0434786081314087`
- `q`: `4.1935601234436035`
- `target_q`: `4.180259704589844`
- `reward_mean`: `-0.1281944066286087`
- `done_fraction`: `0.01953125`
- `discount_mean`: `0.98046875`

Actor drift metrics:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.23856812715530396 | 0.1707537253543696 |
| actor policy mean abs max | 1.7372020483016968 | 1.1534156603329557 |
| actor log_std mean | -0.15711648762226105 | -0.13639042302196033 |
| actor log_std min | -0.7160005569458008 | -0.5989941709725431 |
| actor log_std max | 0.13092592358589172 | 0.24042806924544563 |
| actor policy std mean | 0.8573285341262817 | 0.8771794435281778 |
| sampled action abs mean | 0.5348999500274658 | 0.5254770112669129 |
| sampled action saturation 0.95 | 0.046336207538843155 | 0.04516821260051441 |
| deterministic action abs mean | 0.218863844871521 | 0.16346676852698475 |
| deterministic action saturation 0.95 | 0.0 | 3.74161874120682e-06 |

Interpretation: fresh 100k already shows actor mean drift forming. Final actor
mean magnitude and deterministic action magnitude are higher than their interval
averages, while final log_std/std are lower. This supports the hypothesis that
deterministic path degradation starts early and is coupled with reduced
std/entropy pressure. This is not a runtime failure.

Small action diagnostic eval:

| Mode | Reward Mean | Reward SD | Reward Min | Reward Max | Action Abs | Sat 0.95 | Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.315369606018066 | 0.4851875901222229 | -4.684144496917725 | -3.4863786697387695 | 0.17660009860992432 | 0.0 | 0.18980830907821655 | -0.10445457696914673 | 0.9032111763954163 |
| stochastic | -6.333320140838623 | 0.6809744238853455 | -7.328940391540527 | -5.642662525177002 | 0.5322253704071045 | 0.04625000059604645 | 0.22738231718540192 | -0.15493662655353546 | 0.8592240214347839 |

Reward component highlights:

- Deterministic main negatives: `reward/termination -100`,
  `reward/ang_vel_xy -59.17`, `reward/joint_deviation_hip -36.96`,
  `reward/orientation -33.44`, `reward/feet_slip -16.88`.
- Deterministic positives: `reward/feet_phase 23.39`,
  `reward/tracking_ang_vel 17.95`, `reward/tracking_lin_vel 1.09`.
- Stochastic main negatives: `reward/termination -100`,
  `reward/ang_vel_xy -130.07`, `reward/orientation -41.22`,
  `reward/joint_deviation_hip -40.42`, `reward/feet_slip -12.29`.
- Stochastic positives: `reward/feet_phase 22.56`,
  `reward/tracking_ang_vel 4.79`, `reward/tracking_lin_vel 2.78`.

Warnings: known non-fatal WSL2 CUDA driver version warning, known non-fatal JAX
cast overflow warning, and sandbox `snap-confine` capability errors on first
`uv` attempts. The same commands were rerun externally with unchanged
parameters. No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or
eval failure was observed.

## 2026-05-13 Full Action Diagnostic Eval

- Scope: eval-only; no training and no SAC code modification.
- Script class: `scripts/eval_sac_checkpoint.py --policy_mode both --action_diagnostics --reward_components`.
- Checkpoints: 100k, 250k, and 500k sanity checkpoints.
- Checkpoint readiness: PASS for all three checkpoints.
- Seeds: `0..4`.
- Eval scale: `num_eval_envs=16`, `episode_length=1000`.
- Output directory: `./logs/sac_eval_action_diag_full/`.
- JSON outputs present: `15`.
- Status: all deterministic and stochastic evals returned `EVAL_OK`.
- NaN: all action/reward/obs NaN flags were false.

Deterministic aggregate:

| Scale | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | Det Sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k | -4.2130 | 0.3669 | 0.1823 | 0.1950 | -0.1077 | 0.8996 | 0.0000004 |
| 250k | -4.4792 | 0.3633 | 0.2147 | 0.2288 | -0.1415 | 0.8692 | 0.0000 |
| 500k | -4.8204 | 0.3151 | 0.3029 | 0.3468 | -0.2909 | 0.7519 | 0.00113 |

Stochastic aggregate:

| Scale | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | Sto Sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k | -6.4741 | 0.5045 | 0.5274 | 0.2240 | -0.1562 | 0.8578 | 0.0445 |
| 250k | -6.2374 | 0.4256 | 0.5236 | 0.2712 | -0.1939 | 0.8254 | 0.0415 |
| 500k | -5.8911 | 0.5841 | 0.5221 | 0.3954 | -0.3288 | 0.7256 | 0.0413 |

Interpretation: deterministic eval degradation is not a runtime failure and not
stochastic policy collapse. The main risk is deterministic deployment/eval
degradation driven by actor mean/action magnitude drift. Full details are in
`reports/sac_integration/10_action_distribution_diagnostics.md`.

## 2026-05-12 WSL2 GPU Result

Workspace:

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Latest committed diagnostic baseline before this report update:
  `926a14f Add SAC alpha entropy diagnostics`

Runtime:

- Python: `3.12.3`
- JAX: `0.10.0`
- JAX backend/devices: `gpu`, `cuda:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- GPU: NVIDIA GeForce RTX 4070 SUPER, 12 GB class
- Driver: `591.74`
- CUDA reported by `nvidia-smi`: `13.1`
- `nvcc`: not installed; not a blocker for WSL2 JAX CUDA plugin runtime

Asset state:

- Menagerie path: `g1_env/external_deps/mujoco_menagerie`
- Menagerie commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Menagerie, `.venv`, `logs`, and checkpoints remain ignored runtime artifacts.

GPU preflight:

- Command class: `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu`
- Status: `PASS`
- Stages: imports PASS, flat env PASS, rough env PASS
- Runtime summary: `has_gpu=true`, `jax_backend=gpu`, `jax_devices=["cuda:0"]`
- Flat/Rough env schema: action size `29`, `state (103,)`, `privileged_state (216,)`, no native truncation key

Route B GPU 10k smoke:

- Command wrapper: `bash scripts/gpu_smoke_route_b.sh`
- Status: `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- Env steps: `9984`
- Gradient steps: `142`
- Wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- Actor loss: `-1.9058758020401`
- Critic loss: `0.08382290601730347`
- Alpha loss: `1.585930585861206`
- Alpha: `0.04770537465810776`
- Policy log prob: `-18.79882049560547`
- Policy Q: `1.0090709924697876`
- Q: `1.0673823356628418`
- Target Q: `1.0646085739135742`
- Truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics

Route B deterministic eval smoke:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- Eval command scale: `num_eval_envs=4`, `episode_length=200`
- Status: `EVAL_OK`
- Eval env steps: `800`
- Episode reward mean/std/min/max:
  `-3.3628087043762207` / `0.34718504548072815` /
  `-3.778578281402588` / `-2.876215934753418`
- Done fraction: `1.0`
- Wall time: `62.32472045900067`
- SPS: `12.835998205981001`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- Scope note: this is a small deterministic eval smoke, not a full benchmark.

Route B GPU 50k sanity:

- Command class: `uv run --no-sync python -m learning.train_jax_sac_lift`
- Status: `TRAIN_OK`
- Logdir: `./logs/sac_lift_gpu_50k_sanity`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- Requested timesteps: `50000`
- Actual env steps: `49920`
- Gradient steps: `766`
- Wall time: `35.99766752999858`
- SPS: `1386.756515777009`
- Actor loss: `-3.6135072708129883`
- Critic loss: `0.07097882032394409`
- Alpha loss: `1.327394962310791`
- Alpha: `0.03992176800966263`
- Policy log prob: `-18.81831169128418`
- Policy Q: `2.8622469902038574`
- Q: `2.884032726287842`
- Target Q: `2.899707317352295`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error: none observed

Route B 50k bounded deterministic eval:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- Status: `EVAL_OK`
- Eval command scale: `num_eval_envs=16`, `episode_length=1000`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.5016322135925293` / `0.75983726978302` /
  `-6.096090316772461` / `-2.531925916671753`
- Done fraction: `1.0`
- Wall time: `74.76505397899746`
- SPS: `214.00372431342888`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- Scope note: this is a bounded eval attached to 50k sanity, not a full
  benchmark.

Route B GPU 100k sanity:

- Command class: `uv run --no-sync python -m learning.train_jax_sac_lift`
- Status: `TRAIN_OK`
- Logdir: `./logs/sac_lift_gpu_100k_sanity`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- Requested timesteps: `100000`
- Actual env steps: `99968`
- Gradient steps: `1548`
- Wall time: `56.80434615799459`
- SPS: `1759.8653406193746`
- Actor loss: `-4.994826316833496`
- Critic loss: `0.04054964333772659`
- Alpha loss: `1.0562278032302856`
- Alpha: `0.03259027376770973`
- Policy log prob: `-17.77903938293457`
- Q: `4.3698601722717285`
- Target Q: `4.456111907958984`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/checkpoint/eval error: none observed

Route B 100k bounded deterministic eval:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- Status: `EVAL_OK`
- Eval command scale: `num_eval_envs=16`, `episode_length=1000`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.894726037979126` / `0.9610732197761536` /
  `-7.2897186279296875` / `-2.889821767807007`
- Done fraction: `1.0`
- Wall time: `66.85148939098872`
- SPS: `239.33647770242092`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- Scope note: this is a bounded eval attached to 100k sanity, not a full
  benchmark.

Route B GPU 250k sanity:

- Command class: `uv run --no-sync python -m learning.train_jax_sac_lift`
- Status: `TRAIN_OK`
- Logdir: `./logs/sac_lift_gpu_250k_sanity`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `250000`
- Actual env steps: `249984`
- Gradient steps: `3892`
- Wall time: `120.21957968600327`
- SPS: `2079.3950590488107`
- Actor loss: `-5.524118900299072`
- Critic loss: `0.02644157037138939`
- Alpha loss: `0.5869507789611816`
- Alpha: `0.018743595108389854`
- Policy log prob: `-16.657032012939453`
- Q: `5.197851181030273`
- Target Q: `5.205532073974609`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error: none observed

Route B 250k bounded deterministic eval:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- Status: `EVAL_OK`
- Eval command scale: `num_eval_envs=16`, `episode_length=1000`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.279743194580078` / `1.2322009801864624` /
  `-8.849853515625` / `-3.093963384628296`
- Done fraction: `1.0`
- Wall time: `68.2435936529946`
- SPS: `234.45424168833907`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- Scope note: this is a bounded eval attached to 250k sanity, not a full
  benchmark.

Route B GPU 500k sanity:

- Command class: `uv run --no-sync python -m learning.train_jax_sac_lift`
- Status: `TRAIN_OK`
- Logdir: `./logs/sac_lift_gpu_500k_sanity`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Checkpoint readiness: `--require_eval_ready` PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `500000`
- Actual env steps: `499968`
- Gradient steps: `7798`
- Wall time: `223.88994164399628`
- SPS: `2233.0971919899416`
- Actor loss: `-3.510934352874756`
- Critic loss: `0.04195608198642731`
- Alpha loss: `0.20854677259922028`
- Alpha: `0.008012857288122177`
- Policy log prob: `-12.072959899902344`
- Q: `3.348696231842041`
- Target Q: `3.3187503814697266`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure: none observed

Route B 500k bounded deterministic eval:

- Script: `scripts/eval_sac_checkpoint.py`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- Status: `EVAL_OK`
- Eval command scale: `num_eval_envs=16`, `episode_length=1000`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.691065788269043` / `1.1929610967636108` /
  `-9.040802955627441` / `-3.7163496017456055`
- Done fraction: `1.0`
- Wall time: `68.70198891899781`
- SPS: `232.88990976468807`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`
- Scope note: this is a bounded eval attached to 500k sanity, not a full
  benchmark.

Both-mode eval diagnostic:

- Script: `scripts/eval_sac_checkpoint.py --policy_mode both`
- Checkpoints: 100k, 250k, and 500k sanity checkpoints
- Eval scale: seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`
- JSON root: `./logs/sac_eval_bothmode`
- Checkpoint readiness: PASS for all three checkpoints
- Status: all deterministic and stochastic evals returned `EVAL_OK`
- NaN: no action/reward/obs NaN in either mode
- Deterministic aggregate reward mean:
  - 100k: `-4.2218`
  - 250k: `-4.4585`
  - 500k: `-4.8476`
- Deterministic action abs mean increased:
  `0.1823 -> 0.2148 -> 0.3029`
- Stochastic aggregate reward mean:
  - 100k: `-6.4616`
  - 250k: `-6.1954`
  - 500k: `-5.9091`
- Stochastic log-prob mean became less negative:
  `-17.9594 -> -17.2847 -> -13.4275`
- Interpretation: deterministic `tanh(mean)` behavior degrades, while sampled
  stochastic behavior does not show the same degradation. This does not
  authorize 750k or 1M.
- Full diagnostic report:
  `reports/sac_integration/10_both_mode_eval_diagnostic.md`

Step count note:

- The command requested `num_timesteps=10000` with `num_envs=128`.
- Route B currently computes actual env steps as `num_envs * (num_timesteps // num_envs)`.
- Therefore this smoke records `128 * (10000 // 128) = 9984` actual env steps.
- The 100k sanity records `128 * (100000 // 128) = 99968` actual env steps.
- The 250k sanity records `128 * (250000 // 128) = 249984` actual env steps.
- The 500k sanity records `128 * (500000 // 128) = 499968` actual env steps.

Warnings observed:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- CUDA timer warmup warning: `Delay kernel timed out: measured time has sub-optimal accuracy`.
- These warnings did not fail preflight, 10k smoke, 50k sanity, 100k sanity,
  250k sanity, 500k sanity, bounded A4 750k bridge, or bounded eval; they
  remain non-fatal WSL2/JAX noise unless accompanied by a failed command.
- A first sandboxed `uv` attempt hit a `snap-confine` capability issue. The
  same command succeeded with external permission and unchanged parameters, so
  this is recorded as tooling noise, not a training failure.

Post-smoke `nvidia-smi` summary:

- Time: 2026-05-12 09:12:51
- VRAM: `1517MiB / 12282MiB`
- GPU util: `10%`
- Temperature: `56C`
- Process table only showed `/Xwayland`.

Post-50k `nvidia-smi` summary:

- Time: 2026-05-12 10:58:20
- GPU: RTX 4070 SUPER
- VRAM: `1602MiB / 12282MiB`
- GPU util: `8%`
- Temperature: `56C`
- Power: `9W / 220W`
- Process table only showed `/Xwayland`.

Post-100k `nvidia-smi` summary:

- Time: 2026-05-12 12:24:43
- GPU: RTX 4070 SUPER
- VRAM: `1508MiB / 12282MiB`
- GPU util: `14%`
- Temperature: `55C`
- Power: `9W / 220W`
- Process table only showed `/Xwayland`.

Post-250k `nvidia-smi` summary:

- Time: 2026-05-12 13:15:10
- GPU: RTX 4070 SUPER
- VRAM: `1508MiB / 12282MiB`
- GPU util: `11%`
- Temperature: `57C`
- Power: `9W / 220W`
- Process table only showed `/Xwayland`.

Post-500k `nvidia-smi` summary:

- GPU: RTX 4070 SUPER
- VRAM: `1548MiB / 12282MiB`
- GPU util: `8%`
- Temperature: `37C`
- Power: `8W / 220W`
- Process table only showed `/Xwayland`.

Current validation state:

- 50k sanity: `PASS`
- 100k sanity: `PASS`
- 250k sanity: `PASS`
- 500k sanity: `PASS`
- bounded A4 750k bridge: `PASS_RUNTIME_UNCLEAN_TREND`
- both-mode eval diagnostic for 100k/250k/500k: `PASS`
- bounded 1024-env 1M R3: `PASS_RUNTIME_UNCLEAN_TREND`
- deterministic eval smoke, 50k bounded eval, 100k bounded eval, 250k bounded
  eval, and 500k bounded eval: `PASS`; full eval benchmark: `NOT VALIDATED`
- Risk note: alpha continued down from about `0.0187` at 250k to about
  `0.0080` at 500k. Q and target Q decreased from about `5.2` to about `3.3`,
  critic loss stayed finite/low, and bounded deterministic eval reward
  worsened. Both-mode eval narrows this to a deterministic `tanh(mean)` issue:
  stochastic sampled eval did not show the same reward degradation. This was
  not a runtime failure.
- A4 750k remained runtime stable but worsened actor drift and eval quality
  versus A4 500k: alpha `0.02435 -> 0.01725`, actor mean abs
  `0.29323 -> 0.37190`, deterministic 5-seed reward `-4.4075 -> -5.7314`,
  stochastic 5-seed reward `-6.0434 -> -6.3802`, and critic loss
  `0.0803 -> 0.1441`. This blocks any automatic jump to 1M or longer runs.
- PPO comparison: `NOT VALIDATED`
- domain randomization, fine-tuning, reward/action_scale/Kp tuning: not run
- No logs, checkpoints, `.venv`, or menagerie assets are committed.

## Snapshot

- Project path: `D:\mujoco_playground\g1_sac_dev`
- Branch: `sac-integration`
- Validation start commit: `6027358 Add Route B asymmetric SAC baseline`
- Runtime: `.\.venv\Scripts\python.exe`
- JAX: `0.10.0`, backend `cpu`, devices `['cpu:0']`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- Torch: not importable in `.venv`
- CUDA: `nvidia-smi` and `nvcc` not found
- WSL2: `wsl -l -v` returned exit 1; no usable WSL2/Linux CUDA runtime validated
- Training actually run: yes, CPU tiny smoke completed 256 env steps

## Asset Status

- Initial local search under `D:\mujoco_playground\mujoco_playground` and `D:\mujoco_playground` found no existing `mujoco_menagerie` directory.
- User authorized network download.
- Download target: `D:\mujoco_playground\g1_sac_dev\g1_env\external_deps\mujoco_menagerie`
- Source: `https://github.com/deepmind/mujoco_menagerie.git`
- Checked out commit: `1b86ece576591213e2b666ebf59508454200ca97`
- Size probe: 2176 files, 1517813291 bytes
- Git ignore: `.gitignore:6:mujoco_menagerie` ignores `g1_env/external_deps/mujoco_menagerie`
- Asset commit status: not tracked by project git

## Validation Table

| Step | Command | Status | Result |
|---|---|---:|---|
| Phase A | `git status --short` | PASS | Clean before this round. |
| Phase A | `git branch --show-current` | PASS | `sac-integration`. |
| Phase A | `git log --oneline -5` | PASS | HEAD was `6027358`. |
| Phase A | `.venv runtime import probe` | PASS | Python 3.14.3; JAX/MuJoCo/Brax specs present. |
| Phase A | `Test-Path g1_env\external_deps\mujoco_menagerie` | PASS | Initially `False`; now `True`. |
| Phase A | `git check-ignore -v g1_env/external_deps/mujoco_menagerie` | PASS | Ignored by `.gitignore`. |
| Phase B | Local menagerie search under `D:\mujoco_playground` | PASS | No local copy found. |
| Phase B | `git clone https://github.com/deepmind/mujoco_menagerie.git ...` | PASS | Downloaded after explicit user authorization. |
| Phase B | Menagerie checkout `1b86ece...` | PASS | Effective asset commit matches reference. |
| Phase C | `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts` | PASS | Static compile passed. Command also traverses ignored menagerie because it lives under `g1_env`. |
| Phase C | `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"` | PASS | `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`. |
| Phase C | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain` | PASS | Load/reset/step passed with `--impl jax` default. |
| Phase C | `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain` | PASS | Load/reset/step passed with `--impl jax` default. |
| Phase C | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help` | PASS | Route A entry point help passed. |
| Phase C | `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help` | PASS | Route B entry point help passed, including `--impl`. |
| Phase C | Route B dry-run command | PASS | `DRY_RUN_OK`; one dummy SAC update; checkpoint saved. |
| Probe | 16-step non-dry-run before wider JIT | PASS | `TRAIN_OK`; 16 env steps; 7 gradient steps; checkpoint saved. |
| Probe | 16-step non-dry-run after wider JIT | PASS | `TRAIN_OK`; 16 env steps; 7 gradient steps; wall time improved. |
| Phase D | CPU tiny smoke command with `JAX_PLATFORM_NAME=cpu` | PASS | `TRAIN_OK`; 256 env steps; 121 gradient steps; checkpoint saved. |
| Phase D | Earlier interrupted CPU tiny process | INCONCLUSIVE | User interrupted foreground tool; background process later ended without captured exit code or checkpoint. Superseded by later passing run. |
| Phase D | Background `Start-Process` wrapper | FAIL_NON_BLOCKING | PowerShell environment conflict; direct foreground run succeeded. |
| Phase GPU | `nvidia-smi` | NOT AVAILABLE | Command not found. |
| Phase GPU | `nvcc --version` | NOT AVAILABLE | Command not found. |
| Phase GPU | WSL2 `nvidia-smi` | PASS | RTX 4070 SUPER visible in WSL2; post-smoke VRAM `1517MiB / 12282MiB`. |
| Phase GPU | `nvcc --version` | NOT AVAILABLE_NON_BLOCKING | `nvcc` not installed; WSL2 uses JAX CUDA plugin/runtime wheels. |
| Phase GPU | `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu` | PASS | JAX backend `gpu`, device `cuda:0`; flat/rough env reset/step passed. |
| Phase GPU | Route B GPU 10k smoke wrapper | PASS | `TRAIN_OK`; 9984 actual env steps; checkpoint saved under `./logs/sac_lift_gpu_10k`. |
| Phase Eval | `scripts/eval_sac_checkpoint.py --num_eval_envs 4 --episode_length 200` | PASS | `EVAL_OK`; 800 eval env steps; JSON saved under `./logs/sac_eval_smoke`; no action/reward/obs NaN. |
| Phase Sanity | Route B GPU 50k sanity command | PASS | `TRAIN_OK`; 49920 actual env steps; checkpoint saved under `./logs/sac_lift_gpu_50k_sanity`; no observed NaN/Inf/OOM/CUDA error. |
| Phase Sanity Eval | `scripts/eval_sac_checkpoint.py --num_eval_envs 16 --episode_length 1000` | PASS | `EVAL_OK`; 16000 eval env steps; JSON saved under `./logs/sac_eval_50k`; no action/reward/obs NaN. |

## Env API Schema

Both `G1JoystickFlatTerrain` and `G1JoystickRoughTerrain`:

- registry: `('G1JoystickFlatTerrain', 'G1JoystickRoughTerrain')`
- default config reports `impl="warp"`, but checker passes `config_overrides={"impl": "jax"}` by default for CPU validation
- env type: `Joystick`
- action_size: 29
- dt: 0.02
- sim_dt: 0.002
- n_substeps: 10
- obs type: dict
- obs keys: `state`, `privileged_state`
- `state` shape/dtype: `(103,)`, `float32`
- `privileged_state` shape/dtype: `(216,)`, `float32`
- reset reward/done shape: scalar `float32`
- step reward/done shape: scalar `float32`
- `state.info["truncation"]`: absent
- SAC behavior: missing truncation is synthesized as zeros; CPU tiny observed `truncation_fraction=0.0`

Flat XML:

- `D:/mujoco_playground/g1_sac_dev/g1_env/_src/locomotion/g1/xmls/scene_mjx_feetonly_flat_terrain.xml`

Rough XML:

- `D:/mujoco_playground/g1_sac_dev/g1_env/_src/locomotion/g1/xmls/scene_mjx_feetonly_rough_terrain.xml`

## SAC Metrics

Dry run:

- status: `DRY_RUN_OK`
- policy obs shape: `[103]`
- value obs shape: `[216]`
- action shape: `[29]`
- replay capacity: 16
- gradient steps: 1
- actor loss: `-0.8600597381591797`
- critic loss: `0.4279707074165344`
- alpha: `0.049787066876888275`
- checkpoint: `./logs/sac_lift_dry_run\sac_lift_step_0.pkl`

CPU tiny smoke:

- status: `TRAIN_OK`
- env steps: 256
- gradient steps: 121
- wall time: `52.260398599995824`
- SPS: `4.8985466406301095`
- actor loss: `-1.6421515941619873`
- critic loss: `0.02648034505546093`
- alpha loss: `1.5696232318878174`
- alpha: `0.04801943153142929`
- policy log prob: `-17.302722930908203`
- policy Q: `0.8112847805023193`
- Q: `0.8451158404350281`
- target Q: `0.8984131813049316`
- truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics
- checkpoint: `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`

GPU 10k smoke:

- status: `TRAIN_OK`
- requested timesteps: `10000`
- actual env steps: `9984`
- gradient steps: `142`
- wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- actor loss: `-1.9058758020401`
- critic loss: `0.08382290601730347`
- alpha loss: `1.585930585861206`
- alpha: `0.04770537465810776`
- policy log prob: `-18.79882049560547`
- policy Q: `1.0090709924697876`
- Q: `1.0673823356628418`
- target Q: `1.0646085739135742`
- truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics
- checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- actual step explanation: `128 * (10000 // 128) = 9984`

## Tracebacks And Failures

### Initial Env API Failure Before `--impl jax` Patch

Failure category: dependency/runtime backend.

```text
Traceback (most recent call last):
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 63, in _run_stage
    value = fn()
  File "D:\mujoco_playground\g1_sac_dev\scripts\check_g1_env_api.py", line 176, in <lambda>
    lambda: registry.load(args.env_name, config=config),
            ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\registry.py", line 45, in load
    return locomotion.load(env_name, config, config_overrides)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\__init__.py", line 103, in load
    return _envs[env_name](config=config, config_overrides=config_overrides)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\g1\joystick.py", line 119, in __init__
    super().__init__(
  File "D:\mujoco_playground\g1_sac_dev\g1_env\_src\locomotion\g1\base.py", line 63, in __init__
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
                      ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 543, in put_model
    impl, device = _resolve_impl_and_device(impl, device)
                   ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 152, in _resolve_impl_and_device
    device = _resolve_device(impl)
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\mujoco\mjx\_src\io.py", line 100, in _resolve_device
    cuda_gpus = [d for d in jax.devices('cuda')]
                            ~~~~~~~~~~~^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 1010, in devices
    return get_backend(backend).devices()
           ~~~~~~~~~~~^^^^^^^^^
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 944, in get_backend
    return _get_backend_uncached(platform)
  File "D:\mujoco_playground\g1_sac_dev\.venv\Lib\site-packages\jax\_src\xla_bridge.py", line 932, in _get_backend_uncached
    raise RuntimeError(
RuntimeError: Unknown backend cuda. Available backends are ['cpu']
```

Fix applied:

- `scripts/check_g1_env_api.py` now defaults to `--impl jax`.
- `learning.train_jax_sac_lift` now exposes `--impl`.
- `g1_env.config.sac_params.lift_sac_config` now defaults Route B to `impl="jax"`.
- Route B env loading passes `config_overrides={"impl": config.impl}`.

### Interrupted CPU Tiny Attempt

Failure category: inconclusive/user interruption.

```text
The foreground CPU tiny smoke tool call was interrupted by the user after 509.2s.
Process inspection then found two Python processes that had continued running.
They exited during a bounded wait, but no stdout/stderr, exit code, traceback, or
`logs\sac_lift_cpu_tiny` checkpoint was captured from that interrupted attempt.
```

Resolution:

- Added broader JIT in the SAC training loop.
- Reran the exact CPU tiny command successfully; result supersedes this inconclusive attempt.

### Background Wrapper Failure

Failure category: validation harness, non-blocking.

```text
Start-Process : Added item. Key in dictionary: "Path" Key being added: "PATH"
At line:3 char:6
+ $p = Start-Process -FilePath ".\.venv\Scripts\python.exe" -ArgumentLi ...
+      ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    + CategoryInfo          : NotSpecified: (:) [Start-Process], ArgumentException
    + FullyQualifiedErrorId : System.ArgumentException,Microsoft.PowerShell.Commands.StartProcessCommand
```

Resolution:

- Did not use background wrapper.
- Ran the exact foreground CPU tiny command and captured passing output.

### GPU Runtime Probes

Failure category: dependency/platform.

```text
nvidia-smi : The term 'nvidia-smi' is not recognized as the name of a cmdlet,
function, script file, or operable program.
```

```text
nvcc : The term 'nvcc' is not recognized as the name of a cmdlet, function,
script file, or operable program.
```

Resolution:

- Windows native GPU smoke remained unavailable.
- WSL2/Linux CUDA later superseded this blocker; GPU preflight, GPU 10k smoke,
  and the small deterministic eval smoke now pass in the WSL2 workspace.
