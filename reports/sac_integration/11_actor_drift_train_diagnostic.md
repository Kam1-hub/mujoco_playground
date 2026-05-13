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

## Next Action

- Fresh 100k supports the early actor mean drift hypothesis.
- The next recommended run is fresh 250k diagnostic, only after user
  confirmation.
- Do not run 750k or 1M yet.
- Do not tune reward, `action_scale`, or Kp yet.
