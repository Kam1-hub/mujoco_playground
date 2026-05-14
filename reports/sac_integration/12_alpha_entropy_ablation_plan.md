# Alpha Entropy Ablation Plan

Status: results recorded after fresh 100k A1/A3/A4 ablations, multi-seed
eval-only diagnostics, the bounded fresh 250k and 500k A4 extensions, the
bounded fresh 750k A4 bridge, the fresh 100k actor-regularization R1/R2/R3
ablations, and the bounded R3 250k actor-regularization extension.

This report records the controlled diagnostic plan, the validated fresh 100k
A1/A3/A4 alpha/entropy ablation results, the follow-up multi-seed eval-only
diagnostics, the bounded fresh 250k and 500k A4 extensions, the bounded fresh
750k A4 bridge, and the default-off actor-regularization ablations through
R3 250k.

## Context

- Current branch: `sac-integration`
- Current focus: stabilize deterministic SAC actor behavior before any 1M or
  longer run.
- Fresh 100k and fresh 250k actor drift diagnostics passed runtime gates.
- Both diagnostics show actor mean / deterministic action magnitude rising
  while alpha, log_std, and policy std decline.
- Full action diagnostics show deterministic `tanh(mean)` reward degrades,
  while sampled stochastic reward does not show the same degradation.
- No reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes are authorized by this plan.

## Fresh 100k Ablation Results

Execution context:

- User explicitly approved fresh 100k alpha/entropy ablation training.
- A1 and A3 were run first; A4 was run because both passed
  runtime/checkpoint/eval gates.
- No 250k, 500k, 750k, or 1M run was executed in this ablation phase.
- No reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- Runtime artifacts are under ignored `logs/` and are not committed.

Gate summary:

| ID | Parameters | Train | Readiness | Eval | JSON |
|---|---|---:|---:|---:|---|
| A1 | `alpha_learning_rate=1e-4`, `target_entropy_coef=0.5` | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/alr1e4_seed0_4x200_actiondiag.json` |
| A3 | `alpha_learning_rate=3e-4`, `target_entropy_coef=0.25` | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/te0p25_seed0_4x200_actiondiag.json` |
| A4 | `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25` | TRAIN_OK | PASS | EVAL_OK | `logs/sac_eval_alpha_ablate/te0p25_alr1e4_seed0_4x200_actiondiag.json` |

Training metrics:

| ID | Checkpoint | env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | `./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 72.58993083500536 | 1377.1607005277926 | -5.957967758178711 | 0.051446348428726196 | 0.04285280779004097 | -3.149984121322632 | 5.1932373046875 | 5.197819709777832 |
| A3 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 70.95217173699348 | 1408.9491209735313 | -4.622934341430664 | 0.05104774236679077 | 0.032598935067653656 | -3.423475742340088 | 4.00905704498291 | 4.069530963897705 |
| A4 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_99968.pkl` | 99968 | 1548 | 71.56172043700644 | 1396.947968683882 | -5.977431297302246 | 0.04582885652780533 | 0.04284820705652237 | -3.1500914096832275 | 5.179529666900635 | 5.181567192077637 |

Additional training diagnostics:

| ID | alpha_loss | alpha_log_prob | reward_mean | done_fraction | discount_mean |
|---|---:|---:|---:|---:|---:|
| A1 | 1.3956815004348755 | -18.069194793701172 | -0.09942552447319031 | 0.0078125 | 0.9921875 |
| A3 | 0.8089544773101807 | -17.565364837646484 | -0.11961531639099121 | 0.01171875 | 0.98828125 |
| A4 | 1.0875182151794434 | -18.130718231201172 | -0.11797440052032471 | 0.015625 | 0.984375 |

Fresh 100k baseline reference:

| alpha | actor mean abs | deterministic action abs | log_std mean | std mean |
|---:|---:|---:|---:|---:|
| 0.032585 | 0.238568 | 0.218864 | -0.157116 | 0.857329 |

Actor drift comparison:

| ID | actor mean abs | det action abs | log_std mean | std mean | interval actor mean abs | interval det action abs | interval log_std mean | interval std mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A1 | 0.21597573161125183 | 0.20185625553131104 | -0.15477555990219116 | 0.8587937355041504 | 0.16891714930534363 | 0.1619931809479029 | -0.13590599107495882 | 0.877461152406318 |
| A3 | 0.24479639530181885 | 0.22468040883541107 | -0.15980812907218933 | 0.8549157381057739 | 0.17016767629588297 | 0.16267807873625317 | -0.13607482575914925 | 0.8774873124151575 |
| A4 | 0.2069278359413147 | 0.19313891232013702 | -0.15182653069496155 | 0.8613420128822327 | 0.16146480113036873 | 0.1549823469017904 | -0.13399408028511575 | 0.8791330905308711 |

Small eval summary, seed 0, 4 env x 200 steps:

| ID | Mode | reward_mean | reward_std | reward_min | reward_max | action_abs | saturation 0.95 | NaN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | deterministic | -4.416665077209473 | 0.3470674157142639 | -4.957084655761719 | -4.006556987762451 | 0.18505924940109253 | 0.0 | false |
| A1 | stochastic | -6.050605773925781 | 0.42214730381965637 | -6.710614204406738 | -5.661291599273682 | 0.528386652469635 | 0.04306034743785858 | false |
| A3 | deterministic | -4.555072784423828 | 0.6614199280738831 | -5.202037334442139 | -3.4488778114318848 | 0.18952174484729767 | 0.000043103449570480734 | false |
| A3 | stochastic | -6.32877779006958 | 0.5614966750144958 | -7.13820743560791 | -5.716131210327148 | 0.532009482383728 | 0.04543103650212288 | false |
| A4 | deterministic | -4.366635799407959 | 0.3521541357040405 | -4.8208231925964355 | -3.8527913093566895 | 0.17405447363853455 | 0.0 | false |
| A4 | stochastic | -6.495170593261719 | 0.6918737888336182 | -7.607295036315918 | -5.806713104248047 | 0.5309661030769348 | 0.046120692044496536 | false |

Interpretation:

- A1 improves the main 100k drift metrics versus the fresh 100k baseline:
  alpha is higher, actor mean abs is lower, deterministic action abs is lower,
  and log_std/std are slightly healthier.
- A3 does not improve the drift metrics: alpha is near baseline, but actor
  mean and deterministic action magnitude are worse and std is slightly lower.
- A4 is the best current 100k candidate. It combines higher alpha with lower
  actor mean and deterministic action magnitude, healthier log_std/std, and the
  best deterministic 4x200 reward among A1/A3/A4.
- Caveat: A4 stochastic 4x200 reward is worse than A1/A3, so this is a
  diagnostic signal, not a final policy-quality benchmark.
- Top recurring saturated dims were `7`, `8`, `1`, `3`, `9`, `17`, `19`, and
  `20`, with dim `7` strongest across variants.
- Deterministic negatives are still dominated by `reward/termination`,
  `reward/ang_vel_xy`, `reward/joint_deviation_hip`, `reward/orientation`, and
  `reward/feet_slip`; positives remain `reward/feet_phase`,
  `reward/tracking_ang_vel`, and `reward/tracking_lin_vel`.

Warnings:

- Known non-fatal WSL2 CUDA driver version format warning.
- Known non-fatal JAX cast overflow warning.
- Sandboxed `uv` hit the known `snap-confine` issue; commands were executed
  externally with unchanged parameters.
- No traceback, OOM, fatal CUDA, env, checkpoint, eval failure, or NaN was
  observed.

Next action:

- Do not run 1M automatically.
- Fresh 250k and fresh 500k A4 extensions have now completed; use the results
  below for the next bounded decision.
- Bounded 750k bridge has now completed; do not run 1M automatically.
- Recommended next step is a decision review before any longer A4 extension.

## Multi-Seed 100k Ablation Eval-Only Results

Execution context:

- `CHECKPOINT ALPHA-MULTISEED-EVAL` completed after the A1/A3/A4 100k
  ablation training report.
- Scope: eval-only. No training, code change, or report change happened before
  this report update.
- JSON directory: `./logs/sac_eval_alpha_ablate_multiseed/`.
- JSON count: `15`.
- Checkpoint readiness: A1/A3/A4 PASS; all normalizer and eval-readiness
  checks passed.
- Eval setup: seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`,
  `--policy_mode both`, `--action_diagnostics`, and `--reward_components`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, checkpoint failure, or eval failure was
  observed.

Deterministic aggregate table:

| Variant | Reward Avg | Reward SD | Min Avg | Max Avg | Action Abs | Sat 0.95 | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | -4.3262 | 0.4001 | -8.0999 | -3.1584 | 0.1826 | 0.000000 | 0.1961 | -0.1065 | 0.9006 | true |
| A3 | -4.5518 | 0.4394 | -8.2729 | -3.1404 | 0.1906 | 0.000063 | 0.2076 | -0.1107 | 0.8972 | true |
| A4 | -4.3436 | 0.3791 | -7.9928 | -2.9098 | 0.1774 | 0.000025 | 0.1920 | -0.1058 | 0.9015 | true |

Stochastic aggregate table:

| Variant | Reward Avg | Reward SD | Min Avg | Max Avg | Action Abs | Sat 0.95 | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | -6.3990 | 0.3916 | -10.1843 | -4.9255 | 0.5259 | 0.0433 | 0.2054 | -0.1534 | 0.8599 | true |
| A3 | -6.4888 | 0.4957 | -10.1385 | -5.1995 | 0.5285 | 0.0451 | 0.2291 | -0.1599 | 0.8548 | true |
| A4 | -6.4935 | 0.5123 | -10.1824 | -5.1044 | 0.5266 | 0.0440 | 0.2067 | -0.1540 | 0.8595 | true |

Interpretation:

- A4 remains best on deterministic action magnitude and actor mean drift
  metrics.
- A4 is not strictly best on deterministic reward: A1 is slightly better
  (`-4.3262` vs `-4.3436`), and this gap is tiny relative to seed variance.
- A4 stochastic reward is worse than A1 by about `0.0945` and essentially tied
  with A3.
- A3 remains the weakest candidate for drift/reward.
- A4 is still the best drift-control candidate, but it should be described
  with the reward caveat.
- The next decision should not be 1M. Use the bounded A4 extension evidence
  below before any longer-run decision.

## Fresh 250k A4 Extension Results

Execution context:

- `CHECKPOINT A4-250K-EXTENSION` completed after the fresh 100k A4
  ablation and multi-seed eval-only diagnostics.
- Scope: bounded fresh 250k A4 extension only. This was not a 500k, 750k, or
  1M run.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- Checkpoint:
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; policy/value
  normalizers present.
- Small eval: `./logs/sac_eval_alpha_ablate_250k/eval_A4_seed0_4x200_actiondiag.json`
  returned `EVAL_OK`.
- Multi-seed eval: `./logs/sac_eval_alpha_ablate_250k_multiseed/` contains
  five 16 env x 1000 JSON outputs.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training metrics:

| Metric | Value |
|---|---:|
| env_steps | 249984 |
| gradient_steps | 3892 |
| wall_time | 148.8921 |
| sps | 1678.9605 |
| actor_loss | -9.3541 |
| critic_loss | 0.05245 |
| alpha | 0.03455 |
| log_alpha | -3.36536 |
| alpha_loss | 0.87640 |
| alpha_log_prob | -18.1164 |
| alpha_error_log_prob_plus_target | -25.3664 |
| alpha_grad_proxy_exp | 0.87640 |
| q | 8.7442 |
| target_q | 8.7170 |
| reward_mean | -0.13106 |
| done_fraction | 0.02344 |
| discount_mean | 0.97656 |

Actor drift metrics:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.22572 | 0.19623 |
| actor mean abs max | 1.93320 | n/a |
| log_std mean | -0.17093 | -0.15217 |
| log_std min | -0.66452 | n/a |
| log_std max | 0.03127 | n/a |
| std mean | 0.84424 | 0.86181 |
| sampled action abs | 0.52981 | n/a |
| sampled saturation 0.95 | 0.04297 | n/a |
| deterministic action abs | 0.21220 | 0.18468 |
| deterministic saturation 0.95 | 0.000269 | n/a |

Comparison:

- Versus fresh 250k baseline, A4 improves the main drift metrics: alpha is
  higher by `+0.01578`, actor mean abs is lower by `-0.07330`,
  deterministic action abs is lower by `-0.05875`, log_std is less negative by
  `+0.03434`, and std is higher by `+0.02802`.
- Versus A4 100k, drift still increases moderately: alpha
  `0.04285 -> 0.03455`, actor mean abs `0.20693 -> 0.22572`,
  deterministic action abs `0.19314 -> 0.21220`, and log_std
  `-0.15183 -> -0.17093`.

Small eval, seed 0, 4 env x 200 steps:

| Mode | reward_mean | action_abs | saturation 0.95 | NaN |
|---|---:|---:|---:|---|
| deterministic | -4.0879 | 0.1739 | 0.0 | false |
| stochastic | -5.9671 | 0.5213 | 0.03970 | false |

Multi-seed 16 env x 1000 aggregate:

| Mode | reward avg | reward stdev | min avg | max avg | action abs | sat 0.95 | policy mean abs | log_std | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.1561 | 0.4168 | -7.7058 | -2.9907 | 0.1861 | 0.000006 | 0.1960 | -0.1183 | 0.8894 |
| stochastic | -6.1400 | 0.4045 | -9.9431 | -4.8325 | 0.5213 | 0.04050 | 0.2254 | -0.1733 | 0.8422 |

Interpretation:

- A4 clearly mitigates 250k actor mean / deterministic action drift versus the
  fresh 250k baseline.
- A4 does not eliminate drift relative to A4 100k.
- Q/target_q are higher than earlier baselines and should be treated as a
  watch item.
- This result justified a bounded A4 500k decision review, which has now run;
  see the result below.

## Fresh 500k A4 Extension Results

Execution context:

- `CHECKPOINT A4-500K-EXTENSION` completed after the bounded fresh 250k A4
  alpha/entropy extension and decision review.
- Scope: bounded fresh 500k A4 extension only. This was not a 750k or 1M run.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- Checkpoint:
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; policy/value
  normalizers present.
- Small eval: `./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json`
  returned `EVAL_OK`.
- Multi-seed eval: `./logs/sac_eval_alpha_ablate_500k_multiseed/` contains
  five 16 env x 1000 JSON outputs.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training metrics:

| Metric | Value |
|---|---:|
| env_steps | 499968 |
| gradient_steps | 7798 |
| wall_time | 271.9703 |
| sps | 1838.3185 |
| actor_loss | -8.8341 |
| critic_loss | 0.0803 |
| alpha | 0.02435 |
| log_alpha | -3.71524 |
| alpha_loss | 0.58360 |
| alpha_log_prob | -16.71769 |
| alpha_error_log_prob_plus_target | -23.96769 |
| alpha_error_neg_log_prob_minus_target | 23.96769 |
| alpha_grad_proxy_exp | 0.58360 |
| q | 8.47294 |
| target_q | 8.41017 |
| reward_mean | -0.13071 |
| done_fraction | 0.02344 |
| discount_mean | 0.97656 |

Actor drift metrics:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.29323 | 0.23339 |
| actor policy mean abs max | 2.35011 | 1.84830 |
| actor log_std mean | -0.22279 | -0.17587 |
| actor log_std min | -0.57048 | -0.60559 |
| actor log_std max | 0.10298 | 0.09101 |
| actor policy std mean | 0.80245 | 0.84132 |
| sampled action abs mean | 0.51804 | 0.52402 |
| sampled saturation 0.95 | 0.03933 | 0.04252 |
| deterministic action abs mean | 0.26540 | 0.21651 |
| deterministic saturation 0.95 | 0.000539 | 0.000159 |

Small eval, seed 0, 4 env x 200 steps:

| Mode | reward_mean | reward_std | reward_min | reward_max | action_abs | saturation 0.95 | NaN |
|---|---:|---:|---:|---:|---:|---:|---|
| deterministic | -4.3446 | 0.4644 | -5.0165 | -3.7633 | 0.2304 | 0.0 | false |
| stochastic | -6.2873 | 0.6175 | -7.3428 | -5.7918 | 0.5148 | 0.0342 | false |

Multi-seed 16 env x 1000 aggregate:

| Mode | Reward Avg | Reward SD | Reward Min Avg | Reward Max Avg | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.4075 | 0.3493 | -8.9045 | -3.1749 | 0.2289 | 0.0000004 | 0.2477 | -0.1909 | 0.8277 |
| stochastic | -6.0434 | 0.5364 | -10.5803 | -4.4890 | 0.5152 | 0.0366 | 0.2704 | -0.2277 | 0.7984 |

Comparison:

- Versus the old fresh 500k baseline, A4 500k mitigates drift on alpha
  (`0.02435` vs `0.00801`), actor mean / action magnitude, log_std/std, and
  deterministic eval reward (`-4.4075` 5-seed avg vs old 500k deterministic
  around `-4.69` to `-4.82`).
- Versus A4 250k, drift continues but remains controlled: alpha
  `0.03455 -> 0.02435`, actor mean abs `0.22572 -> 0.29323`,
  deterministic action abs `0.21220 -> 0.26540`, log_std
  `-0.17093 -> -0.22279`, and std `0.84424 -> 0.80245`.
- Q/target_q are slightly lower than A4 250k (`8.74/8.72 -> 8.47/8.41`) but
  much higher than the old fresh 500k baseline (`3.35/3.32`). Critic loss rose
  to `0.0803`. This is finite and not a failure, but it is the main watch item.

Interpretation:

- A4 continues mitigating the old 500k deterministic drift pattern and passes
  runtime/checkpoint/eval gates.
- Q/target_q and critic loss required a decision review before any longer run.
  The bounded 750k bridge below has now completed and supersedes this as the
  latest A4 long-horizon evidence.

## Fresh 750k A4 Bridge Results

Execution context:

- `CHECKPOINT A4-750K-BRIDGE` completed after the bounded fresh 500k A4
  extension and decision review.
- Scope: bounded fresh 750k A4 bridge only. This was not a 1M run.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- Checkpoint:
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`
- Checkpoint readiness: PASS; `deterministic_eval_ready=true`; policy/value
  normalizers present.
- Small eval: `./logs/sac_eval_alpha_ablate_750k/eval_A4_seed0_4x200_actiondiag.json`
  returned `EVAL_OK`.
- Multi-seed eval: `./logs/sac_eval_alpha_ablate_750k_multiseed/` contains
  five 16 env x 1000 JSON outputs.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.

Training metrics:

| Metric | Value |
|---|---:|
| env_steps | 749952 |
| gradient_steps | 11704 |
| wall_time | 397.2631 |
| sps | 1887.7966 |
| actor_loss | -6.5525 |
| critic_loss | 0.1441 |
| alpha | 0.017246 |
| log_alpha | -4.06016 |
| alpha_loss | 0.37864 |
| alpha_log_prob | -14.7051 |
| alpha_error_log_prob_plus_target | -21.9551 |
| alpha_error_neg_log_prob_minus_target | 21.9551 |
| alpha_grad_proxy_exp | 0.37864 |
| q | 6.1746 |
| target_q | 6.2776 |
| reward_mean | -0.0920 |
| done_fraction | 0.0078125 |
| discount_mean | 0.9921875 |

Actor drift metrics:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.37190 | 0.27588 |
| actor mean abs max | 2.98192 | 2.04548 |
| log_std mean | -0.24975 | -0.19484 |
| log_std min | -0.67336 | -0.60936 |
| log_std max | 0.04948 | 0.08209 |
| std mean | 0.78393 | 0.82621 |
| sampled action abs | 0.53627 | 0.52738 |
| sampled saturation 0.95 | 0.04943 | 0.04476 |
| deterministic action abs | 0.31850 | 0.24939 |
| deterministic saturation 0.95 | 0.00310 | 0.00069 |

Small eval, seed 0, 4 env x 200 steps:

| Mode | reward_mean | reward_std | reward_min | reward_max | action_abs | saturation 0.95 | NaN |
|---|---:|---:|---:|---:|---:|---:|---|
| deterministic | -5.1279 | 0.5785 | -5.9269 | -4.5109 | 0.3075 | 0.00151 | false |
| stochastic | -5.9461 | 0.2064 | -6.3028 | -5.8096 | 0.5247 | 0.04280 | false |

Multi-seed 16 env x 1000 aggregate:

| Mode | reward avg | reward stdev | min avg | max avg | action abs | sat 0.95 | policy mean abs | log_std | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -5.7314 | 0.3095 | -11.2225 | -4.3237 | 0.3027 | 0.00201 | 0.3493 | -0.2376 | 0.7924 |
| stochastic | -6.3802 | 0.4577 | -10.9567 | -4.3543 | 0.5274 | 0.04521 | 0.3551 | -0.2611 | 0.7751 |

Comparisons:

- Versus A4 500k, alpha declined `0.02435 -> 0.01725`, actor mean abs rose
  `0.29323 -> 0.37190`, deterministic action abs rose
  `0.26540 -> 0.31850`, log_std narrowed `-0.22279 -> -0.24975`, and std
  fell `0.80245 -> 0.78393`.
- Q/target_q moved down versus A4 500k (`8.47/8.41 -> 6.17/6.28`), but
  critic loss worsened `0.0803 -> 0.1441`.
- Deterministic 5-seed eval worsened sharply `-4.4075 -> -5.7314`.
- Stochastic 5-seed eval also worsened `-6.0434 -> -6.3802`.
- Versus the old fresh 500k baseline, alpha is still better than old `0.00801`,
  but deterministic reward is worse than the old degraded range around
  `-4.69` to `-4.82`.

Interpretation:

- A4 750k is runtime stable, but it is not a clean stability improvement.
- It should block any automatic 1M or longer run.
- Next action should be a decision review/design pass, not immediate longer
  training.

## Current Alpha And Entropy Mechanics

| Item | Value |
|---|---:|
| action dim | 29 |
| `target_entropy_coef` | 0.5 |
| target entropy formula | `-target_entropy_coef * action_dim` |
| current target entropy | -14.5 |
| `init_log_alpha` | -3.0 |
| initial alpha | about 0.0498 |
| `alpha_learning_rate` | 3e-4 |
| alpha floor / clip | none |
| actor `log_std` clamp | `[-5.0, 2.0]` |

Current actor loss:

```text
mean(alpha * log_prob - min_q)
```

Current alpha loss:

```text
mean(alpha * stop_gradient(-log_prob - target_entropy))
```

With the current negative target entropy, the observed
`alpha_grad_proxy_exp` stayed positive in fresh diagnostics, which explains why
gradient descent keeps pushing `log_alpha` and alpha downward.

## Existing Evidence

| Run | alpha | log_alpha | actor mean abs | deterministic action abs | log_std mean | std mean |
|---|---:|---:|---:|---:|---:|---:|
| fresh 100k final | 0.032585 | -3.423901 | 0.238568 | 0.218864 | -0.157116 | 0.857329 |
| fresh 250k final | 0.018768 | -3.975579 | 0.299021 | 0.270944 | -0.205270 | 0.816216 |

Interpretation:

- actor mean and deterministic action magnitude increase from fresh 100k to
  fresh 250k;
- log_std, std, and alpha decrease over the same interval;
- the drift amplifies in absolute level by fresh 250k;
- this is not a runtime failure, checkpoint failure, or stochastic policy
  collapse.

## Action Mapping Support

`4ffd301 Add SAC action diagnostic mapping tools` added no-training helpers:

- `scripts/inspect_g1_action_mapping.py`
- `scripts/summarize_sac_action_diag.py`

These tools map action dimensions to actuator/joint names and join existing
action diagnostic JSON with the mapping. They do not train or evaluate a
policy. They write optional JSON only under ignored `logs/`.

The current 500k deterministic action diagnostic implicates mainly:

- right ankle roll / pitch;
- waist pitch / roll;
- right knee;
- hip roll;
- one right wrist dimension.

This supports interpreting future ablation results without changing SAC math.

## Fresh 100k Ablation Matrix

Use the existing fresh 100k actor diagnostic as the baseline:

```text
./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl
```

Run only 100k diagnostics first. Do not run fresh 250k, fresh 500k, 750k, or
1M until a 100k variant shows a useful signal.

| ID | `target_entropy_coef` | `alpha_learning_rate` | Purpose |
|---|---:|---:|---|
| A1 | 0.5 | 1e-4 | Test slower alpha decay |
| A3 | 0.25 | 3e-4 | Test lower target entropy magnitude under current formula |
| A4 | 0.25 | 1e-4 | Combined conservative candidate, only if A1 or A3 passes runtime gates |
| A2 optional | 0.5 | 3e-5 | Stronger slow-alpha test if A1 is promising but insufficient |

Do not start with `target_entropy_coef > 0.5`; with the current formula this
would make target entropy more negative and likely increase downward alpha
pressure.

## Commands

Required environment:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

A1:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --alpha_learning_rate 1e-4 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1
```

A3:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --target_entropy_coef 0.25 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1
```

A4, only after A1 or A3 passes runtime gates:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --alpha_learning_rate 1e-4 \
  --target_entropy_coef 0.25 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1
```

Expected checkpoint for each:

```text
<logdir>/sac_lift_step_99968.pkl
```

## Validation Gates

For each variant:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint <logdir>/sac_lift_step_99968.pkl \
  --require_eval_ready

uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint <logdir>/sac_lift_step_99968.pkl \
  --seed 0 \
  --num_eval_envs 4 \
  --episode_length 200 \
  --render False \
  --policy_mode both \
  --action_diagnostics \
  --reward_components \
  --top_k_actions 8 \
  --output_json ./logs/sac_eval_alpha_ablate/<variant>_seed0_4x200_actiondiag.json
```

If a variant improves actor drift at 100k, run 5-seed eval before planning a
fresh 250k extension:

```bash
for seed in 0 1 2 3 4; do
  uv run --no-sync python scripts/eval_sac_checkpoint.py \
    --checkpoint <logdir>/sac_lift_step_99968.pkl \
    --seed "$seed" \
    --num_eval_envs 16 \
    --episode_length 1000 \
    --render False \
    --policy_mode both \
    --action_diagnostics \
    --reward_components \
    --top_k_actions 8 \
    --output_json "./logs/sac_eval_alpha_ablate/<variant>_seed_${seed}_16x1000_actiondiag.json"
done
```

## Success Signals

Compare each ablation against fresh 100k baseline:

- alpha remains higher than `0.032585`, or declines more slowly;
- `actor_policy_mean_abs_mean` is lower than `0.238568`;
- `deterministic_action_abs_mean` is lower than `0.218864`;
- `actor_log_std_mean` is less negative than `-0.157116`;
- `actor_policy_std_mean` stays near or above `0.857329`;
- deterministic small eval reward does not worsen;
- stochastic sampled eval remains healthy;
- q, target_q, and critic loss remain finite and stable;
- deterministic saturation does not increase.

## Fresh 100k Actor-Regularization R1 Result

Execution context:

- `CHECKPOINT ACTOR-REG-100K-R1` completed after the default-off actor
  regularization patch.
- Scope: fresh 100k ablation using A4 alpha settings plus
  `deterministic_action_l2_coef=0.01` and `actor_mean_l2_coef=0.001`.
- This was not a 250k, 500k, 750k, or 1M run.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made during the run.
- Runtime artifacts are under ignored `logs/` and are not committed.

Gate summary:

| Gate | Result | Evidence |
|---|---:|---|
| Train | TRAIN_OK | `./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl` |
| Checkpoint readiness | PASS | `deterministic_eval_ready=true`, normalizers present |
| 4x200 action diagnostic eval | PASS | `./logs/sac_eval_actor_reg_100k/eval_R1_seed0_4x200_actiondiag.json` |
| 5-seed eval-only | PASS | five JSONs in `./logs/sac_eval_actor_reg_100k_multiseed/` |

Training metrics:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99968 | 1548 | 71.0452 | 1407.1039 | -5.7620 | 0.06915 | 0.042852 | -3.149996 | 4.9464 | 4.9560 |

Alpha and reward diagnostics:

| alpha_loss | alpha_log_prob | alpha_error_log_prob_plus_target | alpha_error_neg_log_prob_minus_target | alpha_grad_proxy_exp | reward_mean | done_fraction | discount_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.074637 | -17.8277 | -25.0777 | 25.0777 | 1.074637 | -0.12368 | 0.015625 | 0.984375 |

Actor and regularization metrics:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor mean abs | 0.23127 | 0.16341 |
| actor mean abs max | 1.86644 | 1.18543 |
| log_std mean | -0.15577 | -0.13456 |
| log_std min | -0.63851 | -0.59243 |
| log_std max | 0.09566 | 0.23248 |
| std mean | 0.85809 | 0.87862 |
| sampled action abs | 0.53080 | 0.52514 |
| sampled saturation 0.95 | 0.04755 | 0.04483 |
| deterministic action abs | 0.21378 | 0.15691 |
| deterministic saturation 0.95 | 0.000135 | 0.0000045 |
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

Interpretation:

- R1 is runtime clean: all train, checkpoint, and eval gates passed; all
  action/reward/obs NaN flags were false.
- Compared with the A4 100k baseline, alpha is unchanged, but train actor mean
  abs worsened `0.20693 -> 0.23127` and train deterministic action abs
  worsened `0.19314 -> 0.21378`.
- Deterministic 5-seed reward is comparable/slightly worse
  `-4.3436 -> -4.3587`, while stochastic 5-seed reward worsened
  `-6.4935 -> -6.6210`.
- The regularization contribution `0.000886` is tiny relative to actor loss, so
  R1 coefficients appear too weak for train-time drift control.
- Do not extend R1 to 250k as-is. Next action should be a decision review or
  stronger bounded coefficient sweep, not 250k, 750k, or 1M.

## Fresh 100k Actor-Regularization R2/R3 Coefficient Sweep

Execution context:

- `CHECKPOINT ACTOR-REG-100K-R2R3` completed after R1 showed too-weak
  regularization.
- Scope: fresh 100k coefficient sweep only; no 250k, 500k, 750k, or 1M run was
  executed.
- Both variants used A4 alpha settings: `target_entropy_coef=0.25` and
  `alpha_learning_rate=1e-4`.
- R2: `deterministic_action_l2_coef=0.1`, `actor_mean_l2_coef=0.01`.
- R3: `deterministic_action_l2_coef=0.5`, `actor_mean_l2_coef=0.05`.
- Both variants passed training, checkpoint readiness, 4x200 action diagnostic
  eval, and 5-seed eval-only gates.
- Multi-seed JSON directory: `./logs/sac_eval_actor_reg_100k_multiseed/`,
  with `15` total JSON outputs including R1/R2/R3.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No traceback, OOM, fatal CUDA, env, checkpoint, or eval failure was observed.
- Runtime artifacts remain under ignored `logs/` and are not committed.

Training metrics:

| Variant | actor_loss | critic_loss | alpha | log_alpha | q | target_q | reward_mean | sps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | -5.9766 | 0.1383 | 0.042841 | -3.1503 | 5.2078 | 5.1507 | -0.1561 | 1396.73 |
| R3 | -6.0585 | 0.1684 | 0.042818 | -3.1508 | 5.3317 | 5.2599 | -0.1340 | 1414.64 |

Actor and regularization metrics:

| Variant | mean_abs | det_abs | log_std | std | reg_loss | det_l2 | mean_l2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| R2 | 0.2012 | 0.1887 | -0.1513 | 0.8618 | 0.00700 | 0.06218 | 0.07867 |
| R3 | 0.1506 | 0.1430 | -0.1548 | 0.8589 | 0.02259 | 0.04015 | 0.05033 |

5-seed eval aggregates:

| Variant | Mode | Reward Avg | Reward SD | Action Abs | Policy Mean Abs | Log Std | Std | OK |
|---|---|---:|---:|---:|---:|---:|---:|---|
| R2 | deterministic | -4.2422 | 0.3848 | 0.1614 | 0.1720 | -0.0980 | 0.9084 | true |
| R2 | stochastic | -6.4680 | 0.4782 | 0.5247 | 0.1943 | -0.1527 | 0.8605 | true |
| R3 | deterministic | -4.1681 | 0.4120 | 0.1348 | 0.1438 | -0.1014 | 0.9058 | true |
| R3 | stochastic | -6.3983 | 0.4580 | 0.5198 | 0.1483 | -0.1556 | 0.8582 | true |

Interpretation:

- R2 improves over A4 100k and R1 on drift metrics and eval reward.
- R3 improves more strongly and has the best 5-seed deterministic and
  stochastic rewards among R2/R3/A4 100k/R1.
- R3 has higher critic loss than R2 (`0.1684` vs `0.1383`), so it is the
  stronger candidate with a critic-loss watch item.
- R2 remains a conservative backup.
- Next step should be a decision review before any 250k extension. If
  proceeding, a bounded R3 250k diagnostic is the likely candidate. Do not run
  500k, 750k, or 1M from this result.

## R3 250k Actor-Regularization Extension Results

Execution context:

- Scope: bounded R3 250k extension using A4 alpha settings and stronger
  actor-regularization coefficients.
- Parameters: `target_entropy_coef=0.25`, `alpha_learning_rate=1e-4`,
  `deterministic_action_l2_coef=0.5`, `actor_mean_l2_coef=0.05`.
- This was not a 500k, 750k, 1M, or policy-quality benchmark run.
- No code, reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.
- Runtime artifacts are under ignored `logs/` and are not committed.

Gate summary:

| Gate | Result | Evidence |
|---|---:|---|
| Train | TRAIN_OK | `./logs/sac_lift_gpu_250k_actor_reg_te0p25_alr1e4_l2_0p5_mean_0p05_s1/sac_lift_step_249984.pkl` |
| Checkpoint readiness | PASS | `policy_normalizer` and `value_normalizer` present; `deterministic_eval_ready=true` |
| 4x200 action diagnostic eval | PASS | `./logs/sac_eval_actor_reg_250k/eval_R3_seed0_4x200_actiondiag.json` |
| 5-seed eval-only | PASS | five JSONs in `./logs/sac_eval_actor_reg_250k_multiseed/` |

Training metrics:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | alpha | log_alpha | q | target_q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 249984 | 3892 | 145.9190 | 1713.1695 | -9.0065 | 0.05362 | 0.034497 | -3.36687 | 8.3828 | 8.4127 |

Actor and regularization metrics:

| Metric | Final | Interval |
|---|---:|---:|
| actor mean abs | 0.163048 | 0.143159 |
| deterministic action abs | 0.155556 | 0.136790 |
| log_std mean | -0.170873 | -0.152856 |
| std mean | 0.844429 | 0.861275 |
| sampled action abs | 0.526266 | 0.519440 |
| deterministic action L2 | 0.043000 | 0.036152 |
| actor mean L2 | 0.052856 | 0.044359 |
| actor regularization loss | 0.024143 | 0.020294 |

Eval aggregates:

| Mode | Reward Avg | Reward SD | Action Abs | Policy Mean Abs | LogStd Mean | Std Mean | Sat 0.95 | OK |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic | -3.7941 | 0.2941 | 0.1335 | 0.1393 | -0.1140 | 0.8935 | 0.0 | true |
| stochastic | -5.9802 | 0.3177 | 0.5147 | 0.1546 | -0.1698 | 0.8452 | 0.0360 | true |

Interpretation:

- R3 retained drift control at 250k.
- Versus R3 100k, train mean abs moved only `0.1506 -> 0.1630`,
  deterministic action abs moved `0.1430 -> 0.1556`, deterministic eval
  improved `-4.1681 -> -3.7941`, stochastic eval improved
  `-6.3983 -> -5.9802`, and critic loss improved `0.1684 -> 0.0536`.
- Versus A4 250k, R3 has lower action magnitude and better eval.
- This supports using R3 settings in the next capacity benchmark, but it is not
  a final policy-quality claim.

## Stop Conditions

Stop and report immediately if any of these occur:

- JAX backend is not GPU/CUDA;
- `nvidia-smi` is unavailable;
- `TRAIN_OK` is missing;
- checkpoint is missing;
- checkpoint `--require_eval_ready` fails;
- eval status is not `EVAL_OK`;
- `action_nan`, `reward_nan`, or `obs_nan` is true;
- actor loss, critic loss, alpha, log_alpha, q, target_q, log_prob, reward, or
  log_std contains NaN/Inf;
- alpha collapses faster than baseline while actor mean/action magnitude rises;
- q or target_q grows in scale with critic loss degradation;
- logs, checkpoints, `.venv`, or menagerie appear as unignored git changes.

## Execution Notes

The user approved fresh 100k alpha/entropy ablation training. A1 and A3 were
run first; A4 was run only after A1 and A3 passed runtime/checkpoint/eval gates.

Sandboxed `uv` execution can still hit the known `snap-confine` issue. In this
phase, commands were executed externally with unchanged parameters when needed.
This is tooling noise, not a SAC technical blocker.

## Reporting After Execution

The validated ablation results have been recorded here and summarized in:

- `reports/sac_integration/12_alpha_entropy_ablation_plan.md`;
- `NEXT_AGENT_HANDOFF.md`;
- `reports/sac_integration/README.md`;
- `reports/sac_integration/04_smoke_results.md`;
- `reports/sac_integration/05_known_issues.md`;
- `reports/sac_integration/06_next_actions.md`;
- `reports/sac_integration/08_project_status_roadmap.md`;
- `reports/sac_integration/09_phase_summary_and_risks.md`;
- `reports/sac_integration/11_actor_drift_train_diagnostic.md`.

Use the SSH push workflow from the workspace `AGENTS.md`; do not retry the
known-failing HTTPS push path first.
