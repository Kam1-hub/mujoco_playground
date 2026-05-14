# Actor Drift Train Diagnostic

## Context

- Scope: fresh 100k diagnostic training run plus a small action diagnostic eval.
- HEAD during run: `202c6a9 Add SAC actor drift train diagnostics`.
- Purpose: observe train-time actor mean, log_std, std, sampled action, and
  deterministic `tanh(mean)` action drift.
- No 250k, 750k, or 1M run was executed.
- No code changes were made during the run.
- No reward, `action_scale`, Kp, PPO, or RSL changes were made.

## Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS
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

## Actor Final Metrics

- `actor_policy_mean_abs_mean`: `0.23856812715530396`
- `actor_policy_mean_abs_max`: `1.7372020483016968`
- `actor_log_std_mean`: `-0.15711648762226105`
- `actor_log_std_min`: `-0.7160005569458008`
- `actor_log_std_max`: `0.13092592358589172`
- `actor_policy_std_mean`: `0.8573285341262817`
- `sampled_action_abs_mean`: `0.5348999500274658`
- `sampled_action_saturation_fraction_095`: `0.046336207538843155`
- `deterministic_action_abs_mean`: `0.218863844871521`
- `deterministic_action_saturation_fraction_095`: `0.0`

## Actor Interval Metrics

- `interval/actor_policy_mean_abs_mean`: `0.1707537253543696`
- `interval/actor_policy_mean_abs_max`: `1.1534156603329557`
- `interval/actor_log_std_mean`: `-0.13639042302196033`
- `interval/actor_log_std_min`: `-0.5989941709725431`
- `interval/actor_log_std_max`: `0.24042806924544563`
- `interval/actor_policy_std_mean`: `0.8771794435281778`
- `interval/sampled_action_abs_mean`: `0.5254770112669129`
- `interval/sampled_action_saturation_fraction_095`: `0.04516821260051441`
- `interval/deterministic_action_abs_mean`: `0.16346676852698475`
- `interval/deterministic_action_saturation_fraction_095`:
  `3.74161874120682e-06`

## Interpretation

- Fresh 100k already shows actor mean drift forming.
- Final actor mean absolute magnitude is higher than its interval average:
  `0.23856812715530396` vs `0.1707537253543696`.
- Final deterministic action absolute magnitude is higher than its interval
  average: `0.218863844871521` vs `0.16346676852698475`.
- Final log_std mean is lower than its interval average:
  `-0.15711648762226105` vs `-0.13639042302196033`.
- Alpha is already down to about `0.0326`.
- This supports the hypothesis that deterministic path degradation starts
  early and is coupled with reduced std / entropy pressure.
- This is not a runtime failure.

## Small Action Diagnostic Eval

- Script class: `scripts/eval_sac_checkpoint.py --policy_mode both
  --action_diagnostics --reward_components`
- Checkpoint:
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`
- JSON:
  `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`
- Eval scale: `seed=0`, `num_eval_envs=4`, `episode_length=200`
- Status: `EVAL_OK`
- JSON sanity check: PASS

Deterministic summary:

- Episode reward mean/std/min/max:
  `-4.315369606018066` / `0.4851875901222229` /
  `-4.684144496917725` / `-3.4863786697387695`
- `action_abs_mean`: `0.17660009860992432`
- `action_saturation_fraction_095`: `0.0`
- `policy_mean_abs_mean`: `0.18980830907821655`
- `policy_log_std_mean`: `-0.10445457696914673`
- `policy_log_std_min`: `-0.6363540291786194`
- `policy_log_std_max`: `0.08436376601457596`
- `policy_std_mean`: `0.9032111763954163`

Stochastic summary:

- Episode reward mean/std/min/max:
  `-6.333320140838623` / `0.6809744238853455` /
  `-7.328940391540527` / `-5.642662525177002`
- `action_abs_mean`: `0.5322253704071045`
- `action_saturation_fraction_095`: `0.04625000059604645`
- `policy_mean_abs_mean`: `0.22738231718540192`
- `policy_log_std_mean`: `-0.15493662655353546`
- `policy_log_std_min`: `-0.6728885769844055`
- `policy_log_std_max`: `0.12309007346630096`
- `policy_std_mean`: `0.8592240214347839`

## Reward Components

Deterministic main negatives:

- `reward/termination`: `-100`
- `reward/ang_vel_xy`: `-59.17`
- `reward/joint_deviation_hip`: `-36.96`
- `reward/orientation`: `-33.44`
- `reward/feet_slip`: `-16.88`

Deterministic positives:

- `reward/feet_phase`: `23.39`
- `reward/tracking_ang_vel`: `17.95`
- `reward/tracking_lin_vel`: `1.09`

Stochastic main negatives:

- `reward/termination`: `-100`
- `reward/ang_vel_xy`: `-130.07`
- `reward/orientation`: `-41.22`
- `reward/joint_deviation_hip`: `-40.42`
- `reward/feet_slip`: `-12.29`

Stochastic positives:

- `reward/feet_phase`: `22.56`
- `reward/tracking_ang_vel`: `4.79`
- `reward/tracking_lin_vel`: `2.78`

## Warnings

- Known non-fatal WSL2 CUDA driver version format warning.
- Known non-fatal JAX cast overflow warning.
- First sandboxed `uv` attempts hit `snap-confine` capability errors; the same
  commands were rerun externally with unchanged parameters.
- No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or eval failure
  was observed.

## Fresh 250k Diagnostic

### Context

- Scope: fresh 250k diagnostic training run plus a small action diagnostic eval.
- Purpose: compare against the fresh 100k actor drift trajectory.
- No fresh 500k, 750k, or 1M run was executed.
- No code changes were made during the run.
- No reward, `action_scale`, Kp, PPO, or RSL changes were made.

### Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
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
- `alpha_error_log_prob_plus_target`: `-31.316566467285156`
- `alpha_error_neg_log_prob_minus_target`: `31.316566467285156`
- `alpha_grad_proxy_exp`: `0.5877627730369568`
- `q`: `5.547477722167969`
- `target_q`: `5.520053863525391`
- `reward_mean`: `-0.12411123514175415`
- `done_fraction`: `0.01953125`
- `discount_mean`: `0.98046875`

### Actor Final Metrics

- `actor_policy_mean_abs_mean`: `0.2990209758281708`
- `actor_policy_mean_abs_max`: `1.9897916316986084`
- `actor_log_std_mean`: `-0.20526975393295288`
- `actor_log_std_min`: `-0.7766156792640686`
- `actor_log_std_max`: `0.07799282670021057`
- `actor_policy_std_mean`: `0.8162157535552979`
- `sampled_action_abs_mean`: `0.535484254360199`
- `sampled_action_saturation_fraction_095`: `0.04431573301553726`
- `deterministic_action_abs_mean`: `0.2709442377090454`
- `deterministic_action_saturation_fraction_095`: `0.0005387931014411151`

### Actor Interval Metrics

- `interval/actor_policy_mean_abs_mean`: `0.233315885204818`
- `interval/actor_policy_mean_abs_max`: `1.739523811275398`
- `interval/actor_log_std_mean`: `-0.16304866696410225`
- `interval/actor_log_std_min`: `-0.6616444636331555`
- `interval/actor_log_std_max`: `0.1512706000589979`
- `interval/actor_policy_std_mean`: `0.8529925538726603`
- `interval/sampled_action_abs_mean`: `0.5281302615586679`
- `interval/sampled_action_saturation_fraction_095`: `0.04565783552844594`
- `interval/deterministic_action_abs_mean`: `0.21555377979248855`
- `interval/deterministic_action_saturation_fraction_095`:
  `0.00023288404075520971`

### Comparison With Fresh 100k

- Actor mean abs increased:
  - final: `0.238568 -> 0.299021`
  - interval: `0.170754 -> 0.233316`
- Deterministic action abs increased:
  - final: `0.218864 -> 0.270944`
  - interval: `0.163467 -> 0.215554`
- The final-vs-interval gap stayed similar, but the whole mean/action magnitude
  level moved upward.
- Log_std/std continued downward:
  - final log_std: `-0.157116 -> -0.205270`
  - final std: `0.857329 -> 0.816216`
- Alpha declined: `0.032585 -> 0.018768`.
- Interpretation: drift amplifies in absolute level by fresh 250k.

### Small Action Diagnostic Eval

- Script class: `scripts/eval_sac_checkpoint.py --policy_mode both
  --action_diagnostics --reward_components`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`
- JSON:
  `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`
- Eval scale: `seed=0`, `num_eval_envs=4`, `episode_length=200`
- Status: `EVAL_OK`
- JSON sanity check: PASS

Deterministic summary:

- Episode reward mean/std/min/max:
  `-4.01785135269165` / `0.34408432245254517` /
  `-4.546696662902832` / `-3.588685989379883`
- `action_abs_mean`: `0.18537074327468872`
- `action_saturation_fraction_095`: `0.0`
- `policy_mean_abs_mean`: `0.19577054679393768`
- `policy_log_std_mean`: `-0.13236531615257263`
- `policy_log_std_min`: `-0.5227047204971313`
- `policy_log_std_max`: `0.0420893058180809`
- `policy_std_mean`: `0.8772842288017273`

Stochastic summary:

- Episode reward mean/std/min/max:
  `-5.697283744812012` / `0.1435307413339615` /
  `-5.888537883758545` / `-5.491363525390625`
- `action_abs_mean`: `0.5233082175254822`
- `action_saturation_fraction_095`: `0.04060344770550728`
- `policy_mean_abs_mean`: `0.269175261259079`
- `policy_log_std_mean`: `-0.20079341530799866`
- `policy_log_std_min`: `-0.6182204484939575`
- `policy_log_std_max`: `0.08598750829696655`
- `policy_std_mean`: `0.8196967244148254`

### Reward Components

Deterministic main negatives:

- `reward/termination`: `-100`
- `reward/orientation`: `-47.33`
- `reward/ang_vel_xy`: `-44.81`
- `reward/joint_deviation_hip`: `-31.37`
- `reward/feet_slip`: `-14.83`

Deterministic positives:

- `reward/feet_phase`: `25.82`
- `reward/tracking_ang_vel`: `20.20`
- `reward/tracking_lin_vel`: `2.90`

Stochastic main negatives:

- `reward/ang_vel_xy`: `-108.26`
- `reward/termination`: `-100`
- `reward/orientation`: `-46.87`
- `reward/joint_deviation_hip`: `-35.09`
- `reward/feet_slip`: `-10.07`

Stochastic positives:

- `reward/feet_phase`: `27.63`
- `reward/tracking_lin_vel`: `4.17`
- `reward/tracking_ang_vel`: `3.64`

### Warnings

- Known non-fatal WSL2 CUDA driver version format warning.
- Known non-fatal JAX cast overflow warning.
- First sandboxed `uv` attempts hit `snap-confine` capability errors; the same
  commands were rerun externally with unchanged parameters.
- No traceback, NaN, OOM, fatal CUDA error, checkpoint failure, or eval failure
  was observed.

## Next Action

- Fresh 100k supports the early actor mean drift hypothesis, and fresh 250k
  shows the drift amplifies in absolute level.
- Fresh 100k alpha/entropy ablation A1/A3/A4 has since run; see below.
- Do not run fresh 500k, 750k, or 1M automatically.
- Next step should review the A4 100k signal before any longer extension.
- Do not tune reward, `action_scale`, or Kp yet.

## Fresh 100k Alpha/Entropy Ablation

### Context

- Scope: fresh 100k alpha/entropy ablation training plus small action
  diagnostic eval.
- Purpose: test whether slower alpha decay and/or smaller target entropy
  magnitude improves the early actor mean / deterministic action drift signal.
- Variants:
  - A1: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.5`.
  - A3: `alpha_learning_rate=3e-4`, `target_entropy_coef=0.25`.
  - A4: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`.
- A4 was run because A1 and A3 passed runtime, checkpoint, and eval gates.
- No 250k, 500k, 750k, or 1M run was executed in this ablation phase.
- No reward, `action_scale`, Kp, PPO, or RSL changes were made.

### Gate Summary

| Variant | Checkpoint | Train | Readiness | Eval |
|---|---|---:|---:|---:|
| A1 | `./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1/sac_lift_step_99968.pkl` | TRAIN_OK | PASS | EVAL_OK |
| A3 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1/sac_lift_step_99968.pkl` | TRAIN_OK | PASS | EVAL_OK |
| A4 | `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_99968.pkl` | TRAIN_OK | PASS | EVAL_OK |

### Training And Drift Metrics

| Variant | alpha | log_alpha | actor mean abs | det action abs | log_std mean | std mean | reward_mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fresh 100k baseline | 0.032585 | -3.423901 | 0.238568 | 0.218864 | -0.157116 | 0.857329 | -0.128194 |
| A1 | 0.04285280779004097 | -3.149984121322632 | 0.21597573161125183 | 0.20185625553131104 | -0.15477555990219116 | 0.8587937355041504 | -0.09942552447319031 |
| A3 | 0.032598935067653656 | -3.423475742340088 | 0.24479639530181885 | 0.22468040883541107 | -0.15980812907218933 | 0.8549157381057739 | -0.11961531639099121 |
| A4 | 0.04284820705652237 | -3.1500914096832275 | 0.2069278359413147 | 0.19313891232013702 | -0.15182653069496155 | 0.8613420128822327 | -0.11797440052032471 |

Interval metrics:

| Variant | interval actor mean abs | interval det action abs | interval log_std mean | interval std mean |
|---|---:|---:|---:|---:|
| A1 | 0.16891714930534363 | 0.1619931809479029 | -0.13590599107495882 | 0.877461152406318 |
| A3 | 0.17016767629588297 | 0.16267807873625317 | -0.13607482575914925 | 0.8774873124151575 |
| A4 | 0.16146480113036873 | 0.1549823469017904 | -0.13399408028511575 | 0.8791330905308711 |

### Small Action Diagnostic Eval

| Variant | Mode | Reward Mean | Reward SD | Reward Min | Reward Max | Action Abs | Sat 0.95 | NaN |
|---|---|---:|---:|---:|---:|---:|---:|---|
| A1 | deterministic | -4.416665077209473 | 0.3470674157142639 | -4.957084655761719 | -4.006556987762451 | 0.18505924940109253 | 0.0 | false |
| A1 | stochastic | -6.050605773925781 | 0.42214730381965637 | -6.710614204406738 | -5.661291599273682 | 0.528386652469635 | 0.04306034743785858 | false |
| A3 | deterministic | -4.555072784423828 | 0.6614199280738831 | -5.202037334442139 | -3.4488778114318848 | 0.18952174484729767 | 0.000043103449570480734 | false |
| A3 | stochastic | -6.32877779006958 | 0.5614966750144958 | -7.13820743560791 | -5.716131210327148 | 0.532009482383728 | 0.04543103650212288 | false |
| A4 | deterministic | -4.366635799407959 | 0.3521541357040405 | -4.8208231925964355 | -3.8527913093566895 | 0.17405447363853455 | 0.0 | false |
| A4 | stochastic | -6.495170593261719 | 0.6918737888336182 | -7.607295036315918 | -5.806713104248047 | 0.5309661030769348 | 0.046120692044496536 | false |

### Interpretation

- A1 improves the main 100k drift metrics versus the fresh 100k baseline.
- A3 does not improve the drift metrics: alpha is near baseline, but actor mean
  and deterministic action magnitude are worse and std is slightly lower.
- A4 is the best current 100k candidate. It combines higher alpha with lower
  actor mean and deterministic action magnitude, healthier log_std/std, and
  the best deterministic 4x200 reward among A1/A3/A4.
- A4 stochastic 4x200 reward was worse than A1/A3, so this remains a diagnostic
  signal rather than a final policy-quality benchmark.
- No traceback, NaN, OOM, fatal CUDA, checkpoint failure, or eval failure was
  observed.

### Updated Next Action

- Do not run 750k or 1M.
- Fresh 250k and fresh 500k A4 extensions have completed; see below.
- Do not run 750k or 1M automatically.
- Recommended next step: decision review on bounded A4 continuation versus
  further targeted design or holding.

## Fresh 100k Alpha/Entropy Multi-Seed Eval Follow-Up

### Context

- Scope: eval-only follow-up for A1/A3/A4 fresh 100k ablation checkpoints.
- JSON directory: `./logs/sac_eval_alpha_ablate_multiseed/`.
- JSON count: `15`.
- Eval setup: seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`,
  `--policy_mode both`, `--action_diagnostics`, and `--reward_components`.
- Checkpoint readiness: A1/A3/A4 PASS.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
- No training or code changes occurred in this follow-up.

### Deterministic Aggregate

| Variant | Reward Avg | Reward SD | Min Avg | Max Avg | Action Abs | Sat 0.95 | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | -4.3262 | 0.4001 | -8.0999 | -3.1584 | 0.1826 | 0.000000 | 0.1961 | -0.1065 | 0.9006 | true |
| A3 | -4.5518 | 0.4394 | -8.2729 | -3.1404 | 0.1906 | 0.000063 | 0.2076 | -0.1107 | 0.8972 | true |
| A4 | -4.3436 | 0.3791 | -7.9928 | -2.9098 | 0.1774 | 0.000025 | 0.1920 | -0.1058 | 0.9015 | true |

### Stochastic Aggregate

| Variant | Reward Avg | Reward SD | Min Avg | Max Avg | Action Abs | Sat 0.95 | Mean Abs | Log Std Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A1 | -6.3990 | 0.3916 | -10.1843 | -4.9255 | 0.5259 | 0.0433 | 0.2054 | -0.1534 | 0.8599 | true |
| A3 | -6.4888 | 0.4957 | -10.1385 | -5.1995 | 0.5285 | 0.0451 | 0.2291 | -0.1599 | 0.8548 | true |
| A4 | -6.4935 | 0.5123 | -10.1824 | -5.1044 | 0.5266 | 0.0440 | 0.2067 | -0.1540 | 0.8595 | true |

### Interpretation

- A4 remains best on deterministic action magnitude and actor mean drift
  metrics.
- A4 is not strictly best on deterministic reward: A1 is slightly better
  (`-4.3262` vs `-4.3436`), and this gap is tiny relative to seed variance.
- A4 stochastic reward is worse than A1 by about `0.0945` and essentially tied
  with A3.
- A3 remains the weakest candidate for drift/reward.
- A4 is still the best drift-control candidate, but it should be described
  with the reward caveat.
- Fresh 250k and fresh 500k A4 extensions have completed; see below. Do not
  run 750k or 1M automatically.

## Fresh 250k A4 Alpha/Entropy Extension

### Context

- Scope: bounded fresh 250k A4 extension using the current best
  drift-control alpha/entropy candidate.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- This was not a fresh 500k, 750k, or 1M run.
- No code changes were made during the run.
- No reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.

### Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
- `deterministic_eval_ready`: `true`
- `policy_normalizer` / `value_normalizer`: present
- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `148.8921`
- `sps`: `1678.9605`
- `actor_loss`: `-9.3541`
- `critic_loss`: `0.05245`
- `alpha`: `0.03455`
- `log_alpha`: `-3.36536`
- `alpha_loss`: `0.87640`
- `alpha_log_prob`: `-18.1164`
- `alpha_error_log_prob_plus_target`: `-25.3664`
- `alpha_grad_proxy_exp`: `0.87640`
- `q`: `8.7442`
- `target_q`: `8.7170`
- `reward_mean`: `-0.13106`
- `done_fraction`: `0.02344`
- `discount_mean`: `0.97656`

### Actor Drift Metrics

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.22572 | 0.19623 |
| actor policy mean abs max | 1.93320 | n/a |
| actor log_std mean | -0.17093 | -0.15217 |
| actor log_std min | -0.66452 | n/a |
| actor log_std max | 0.03127 | n/a |
| actor policy std mean | 0.84424 | 0.86181 |
| sampled action abs mean | 0.52981 | n/a |
| sampled action saturation 0.95 | 0.04297 | n/a |
| deterministic action abs mean | 0.21220 | 0.18468 |
| deterministic action saturation 0.95 | 0.000269 | n/a |

### Comparison

Versus fresh 250k baseline:

- Alpha is higher by `+0.01578`.
- Actor mean abs is lower by `-0.07330`.
- Deterministic action abs is lower by `-0.05875`.
- Log_std is less negative by `+0.03434`.
- Std is higher by `+0.02802`.

Versus A4 100k:

- Alpha declined moderately: `0.04285 -> 0.03455`.
- Actor mean abs increased: `0.20693 -> 0.22572`.
- Deterministic action abs increased: `0.19314 -> 0.21220`.
- Log_std decreased: `-0.15183 -> -0.17093`.

Interpretation: A4 clearly mitigates 250k actor mean / deterministic action
drift versus the fresh 250k baseline, but it does not eliminate drift relative
to A4 100k.

### Eval Summary

- Small eval JSON:
  `./logs/sac_eval_alpha_ablate_250k/eval_A4_seed0_4x200_actiondiag.json`
- Multi-seed JSON directory:
  `./logs/sac_eval_alpha_ablate_250k_multiseed/`
- Small eval status: PASS / `EVAL_OK`
- Multi-seed eval status: PASS; five 16 env x 1000 JSON outputs.
- All action/reward/obs NaN flags were false.

Small eval, seed 0, 4 env x 200 steps:

| Mode | Reward Mean | Action Abs | Sat 0.95 | NaN |
|---|---:|---:|---:|---|
| deterministic | -4.0879 | 0.1739 | 0.0 | false |
| stochastic | -5.9671 | 0.5213 | 0.03970 | false |

Multi-seed 16 env x 1000 aggregate:

| Mode | Reward Avg | Reward SD | Reward Min Avg | Reward Max Avg | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.1561 | 0.4168 | -7.7058 | -2.9907 | 0.1861 | 0.000006 | 0.1960 | -0.1183 | 0.8894 |
| stochastic | -6.1400 | 0.4045 | -9.9431 | -4.8325 | 0.5213 | 0.04050 | 0.2254 | -0.1733 | 0.8422 |

### Interpretation And Next Action

- A4 is promising drift-control evidence at 250k.
- Q/target_q are higher than earlier baselines and should be treated as a
  watch item.
- This result justified a bounded A4 500k decision review, which has now run.

## Fresh 500k A4 Alpha/Entropy Extension

### Context

- Scope: bounded fresh 500k A4 extension using the current best
  drift-control alpha/entropy candidate.
- Parameters: `target_entropy_coef=0.25`,
  `alpha_learning_rate=1e-4`.
- This was not a 750k or 1M run.
- No code changes were made during the run.
- No reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes were made.

### Training Result

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS
- `deterministic_eval_ready`: `true`
- `policy_normalizer` / `value_normalizer`: present
- `env_steps`: `499968`
- `gradient_steps`: `7798`
- `wall_time`: `271.9703`
- `sps`: `1838.3185`
- `actor_loss`: `-8.8341`
- `critic_loss`: `0.0803`
- `alpha`: `0.02435`
- `log_alpha`: `-3.71524`
- `alpha_loss`: `0.58360`
- `alpha_log_prob`: `-16.71769`
- `alpha_error_log_prob_plus_target`: `-23.96769`
- `alpha_error_neg_log_prob_minus_target`: `23.96769`
- `alpha_grad_proxy_exp`: `0.58360`
- `q`: `8.47294`
- `target_q`: `8.41017`
- `reward_mean`: `-0.13071`
- `done_fraction`: `0.02344`
- `discount_mean`: `0.97656`

### Actor Drift Metrics

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.29323 | 0.23339 |
| actor policy mean abs max | 2.35011 | 1.84830 |
| actor log_std mean | -0.22279 | -0.17587 |
| actor log_std min | -0.57048 | -0.60559 |
| actor log_std max | 0.10298 | 0.09101 |
| actor policy std mean | 0.80245 | 0.84132 |
| sampled action abs mean | 0.51804 | 0.52402 |
| sampled action saturation 0.95 | 0.03933 | 0.04252 |
| deterministic action abs mean | 0.26540 | 0.21651 |
| deterministic action saturation 0.95 | 0.000539 | 0.000159 |

### Eval Summary

- Small eval JSON:
  `./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json`
- Multi-seed JSON directory:
  `./logs/sac_eval_alpha_ablate_500k_multiseed/`
- Small eval status: PASS / `EVAL_OK`
- Multi-seed eval status: PASS; five 16 env x 1000 JSON outputs.
- All action/reward/obs NaN flags were false.

Small eval, seed 0, 4 env x 200 steps:

| Mode | Reward Mean | Reward SD | Reward Min | Reward Max | Action Abs | Sat 0.95 | NaN |
|---|---:|---:|---:|---:|---:|---:|---|
| deterministic | -4.3446 | 0.4644 | -5.0165 | -3.7633 | 0.2304 | 0.0 | false |
| stochastic | -6.2873 | 0.6175 | -7.3428 | -5.7918 | 0.5148 | 0.0342 | false |

Multi-seed 16 env x 1000 aggregate:

| Mode | Reward Avg | Reward SD | Reward Min Avg | Reward Max Avg | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.4075 | 0.3493 | -8.9045 | -3.1749 | 0.2289 | 0.0000004 | 0.2477 | -0.1909 | 0.8277 |
| stochastic | -6.0434 | 0.5364 | -10.5803 | -4.4890 | 0.5152 | 0.0366 | 0.2704 | -0.2277 | 0.7984 |

### Interpretation And Next Action

- A4 500k mitigates the old fresh 500k deterministic drift pattern on alpha,
  actor mean / action magnitude, log_std/std, and deterministic eval reward.
- A4 500k still drifts relative to A4 250k: alpha `0.03455 -> 0.02435`,
  actor mean abs `0.22572 -> 0.29323`, deterministic action abs
  `0.21220 -> 0.26540`, and std `0.84424 -> 0.80245`.
- Q/target_q are slightly lower than A4 250k but much higher than the old
  fresh 500k baseline; critic loss rose to `0.0803`. This is the main watch
  item.
- This result does not authorize a jump to 750k or 1M. Next action should be a
  decision review before any longer training. Do not tune reward,
  `action_scale`, or Kp yet.
