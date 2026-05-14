# Alpha Entropy And Fixed-Command Eval Gate

## Context

- Base commit: `fd8a014 Add SAC fixed-command render support`.
- Scope: eval-only fixed joystick command support plus alpha/entropy sign audit.
- No training was run for this report update.
- No render or full benchmark was run for this report update.
- No SAC train/loss/reward/action_scale/Kp/env/PPO/RSL/checkpoint schema
  changes were made.
- Runtime artifacts remain under ignored `logs/` and must not be committed.

## Fixed-Command Forward Eval Gate

Gate name: fixed-command forward eval gate on the 3M R3 checkpoint.

- Checkpoint:
  `./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl`
- Scope: eval-only, fixed forward commands, no training, no render, and no code
  changes during the gate.
- Output directory: `./logs/sac_eval_fixedcmd_3m_gate/` under ignored `logs/`;
  these JSON artifacts must not be committed.
- Commands:
  - `fwd0.5`: `[0.5, 0.0, 0.0]`, seeds `0..4`.
  - `fwd1.0`: `[1.0, 0.0, 0.0]`, seeds `0..4`.
- Eval settings for both commands: `num_eval_envs=16`,
  `episode_length=1000`, `policy_mode=both`, action diagnostics enabled, and
  reward components enabled.
- Result: all evals returned `EVAL_OK`; all action/reward/obs NaN flags were
  false.

Forward `0.5` aggregate:

| Mode | Reward Avg | Reward Stdev | Reward Min | Reward Max | Done Avg | Action Abs | Sat 0.95 | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic | 0.0192 | 0.6026 | -0.9842 | 0.6049 | 1.0 | 0.1630 | 0.00327 | false |
| stochastic | -9.9480 | 0.8755 | -10.8305 | -8.5759 | 1.0 | 0.3614 | 0.00471 | false |

Forward `0.5` key reward components:

| Mode | tracking_lin_vel | tracking_ang_vel | feet_phase | ang_vel_xy | orientation | feet_slip | termination |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 183.95 | 220.87 | 380.34 | -130.85 | -49.64 | -115.53 | -43.75 |
| stochastic | 115.39 | 83.96 | 354.06 | -287.69 | -42.68 | -124.65 | -56.25 |

Forward `1.0` aggregate:

| Mode | Reward Avg | Reward Stdev | Reward Min | Reward Max | Done Avg | Action Abs | Sat 0.95 | NaN |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| deterministic | -3.2302 | 0.3845 | -3.5655 | -2.6177 | 1.0 | 0.1646 | 0.00348 | false |
| stochastic | -11.5115 | 0.6616 | -12.3932 | -10.8398 | 1.0 | 0.3608 | 0.00475 | false |

Forward `1.0` key reward components:

| Mode | tracking_lin_vel | tracking_ang_vel | feet_phase | ang_vel_xy | orientation | feet_slip | termination |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | 25.94 | 212.79 | 363.20 | -125.97 | -45.45 | -109.40 | -56.25 |
| stochastic | 23.01 | 80.69 | 346.02 | -276.85 | -42.63 | -122.04 | -62.50 |

Interpretation:

- This is not a runtime failure.
- `fwd0.5` deterministic is near break-even but noisy; it is not a robust
  solved gait.
- `fwd1.0` deterministic is weak. The main signal is that deterministic
  `tracking_lin_vel` collapses from `183.95` at `fwd0.5` to `25.94` at
  `fwd1.0`.
- Stochastic fixed-forward eval remains poor for both commands, consistent with
  the existing entropy/std collapse diagnosis.
- This gate blocks any direct 5M/10M continuation.
- The main route should be targeted ablation before longer training: alpha
  floor, fixed alpha, standard log-alpha update, or another controlled
  entropy/temperature diagnostic. A fixed-command `fwd1.0` render/video review
  is also useful, but should not replace the alpha/entropy ablation decision.

Warnings:

- Sandbox `snap-confine` blocked some `uv` attempts; the same commands were
  rerun with approved external execution.
- The known non-fatal WSL2 CUDA driver version warning was observed.
- The known non-fatal JAX cast overflow warning was observed.
- No traceback, OOM, fatal CUDA error, non-`EVAL_OK` result, or NaN flag was
  observed in successful gate outputs.

## Fixed-Command Eval Support

`scripts/eval_sac_checkpoint.py` now supports:

- `--fixed_command`
- `--command_x`
- `--command_y`
- `--command_yaw`

When `--fixed_command true` is passed, eval forces the requested joystick
command throughout the rollout. The override updates both:

- `state.info["command"]`
- the command slice `9:12` in `obs["state"]` and `obs["privileged_state"]`

This matches the fixed-command render helper semantics. Default eval behavior
is unchanged when `--fixed_command false`; the output JSON only gains additive
`fixed_command` and `command` fields.

## Validation

Static checks:

- `git diff --check`: PASS.
- `uv run --no-sync python scripts/eval_sac_checkpoint.py --help`: PASS.
- Help output includes `--fixed_command`, `--command_x`, `--command_y`, and
  `--command_yaw`.

Beauvoir fixed-command smoke:

- Checkpoint:
  `./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl`
- JSON:
  `./logs/sac_eval_fixedcmd_smoke_3m/eval_cmd_x0p5_seed0_both_4x200.json`
- Policy mode: `both`
- Fixed command: `[0.5, 0.0, 0.0]`
- Status: `EVAL_OK`
- Deterministic reward mean: `0.5099`
- Stochastic reward mean: `-2.3354`
- Action/reward/obs NaN flags: false.

Default compatibility smoke:

- JSON:
  `./logs/sac_eval_fixedcmd_smoke_3m/eval_default_compat_seed0_4x100.json`
- Policy mode: `deterministic`
- `fixed_command`: false
- Status: `EVAL_OK`
- Eval env steps: `400`
- Reward mean/std/min/max:
  `-0.3196667730808258 / 0.2676604986190796 /
  -0.7161723375320435 / -0.011976183392107487`
- Action abs mean: `0.17856687307357788`
- Action saturation fraction 0.95: `0.0013793103862553835`
- Action/reward/obs NaN flags: false.

## Alpha Sign Audit

Noether's alpha sign audit result is `SIGN_OK_BUT_COLLAPSE_RISK`.

Code facts:

- `action_dim = 29`.
- `target_entropy = -target_entropy_coef * action_dim`.
- R3 uses `target_entropy_coef=0.25`, so `target_entropy = -7.25`.
- Current alpha loss:

```text
L = mean(exp(log_alpha) * stop_gradient(-log_prob - target_entropy))
dL/dlog_alpha = alpha * mean(-log_prob - target_entropy)
```

Gradient descent interpretation:

- positive gradient decreases `log_alpha`, so `alpha` decreases;
- negative gradient increases `log_alpha`, so `alpha` increases.

Audit conclusion:

- No direct sign bug was found relative to Brax-style SAC.
- The sign matches the common log-alpha equivalent form.
- The key difference is gradient magnitude: the current `exp(log_alpha)` form
  scales the temperature update by `alpha`.
- In the observed log-probability range, `-log_prob - target_entropy` remains
  positive, so gradient descent keeps pushing `alpha` downward.
- As `alpha` approaches zero, the current update also weakens, which makes
  low-alpha collapse plausible without requiring a runtime failure.

3M R3 example:

- report `log_prob`: `-5.7375`
- R3 target entropy: `-7.25`
- bracket: `-(-5.7375) - (-7.25) = 12.9875`
- alpha: `0.000766`
- gradient proxy: about `0.00995`

## Fresh 100k Fixed-Alpha Diagnostic

Gate name: fresh env1024 R3 100k fixed-alpha diagnostic, first variant.

- Variant: `fixed_alpha=0.03`.
- Scope: short diagnostic training, checkpoint readiness, and small
  fixed-command smoke already completed before this report update.
- No 250k, 5M, 10M, render, code change, reward, `action_scale`, Kp, PPO/RSL,
  or checkpoint schema change was made for this report update.
- Training status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p03/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.

Training metrics:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | q | target_q | reward_mean | done_fraction | discount_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99328 | 1312 | 46.3008s | 2145.2764 | -3.45169 | 0.114333 | 2.87063 | 2.87375 | -0.134369 | 0.0234375 | 0.976563 |

Alpha metrics:

| alpha_raw | log_alpha_raw | alpha_effective | log_alpha_effective | fixed_alpha | alpha_loss_type_id | alpha_loss | alpha_log_prob | alpha_grad_proxy_exp | alpha_grad_proxy_log |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0497871 | -3.0 | 0.03 | -3.50656 | 0.03 | 0 | 1.29717 | -18.8043 | 1.29717 | 26.0543 |

Interpretation: the fixed-alpha mechanism worked. Effective alpha stayed fixed
at `0.03`, and raw `log_alpha` stayed at init `-3.0`.

Actor drift metrics:

| Metric | Final | Interval |
|---|---:|---:|
| actor policy mean abs mean | 0.147356 | 0.125020 |
| actor policy mean abs max | 1.25299 | 0.826388 |
| actor log_std mean | -0.157015 | -0.138324 |
| actor log_std min | -0.602285 | -0.608109 |
| actor log_std max | 0.086500 | 0.235712 |
| actor policy std mean | 0.856957 | 0.875592 |
| sampled action abs mean | 0.523311 | 0.519919 |
| sampled action saturation 0.95 | 0.0387931 | 0.0417752 |
| deterministic action abs mean | 0.141470 | 0.122110 |
| deterministic action saturation 0.95 | 0.0 | 0.0 |

Small fixed-command smoke:

- `fwd0.5` JSON:
  `./logs/sac_eval_fixed_alpha_100k_smoke/eval_fwd0p5_seed0_both_4x200.json`
- `fwd1.0` JSON:
  `./logs/sac_eval_fixed_alpha_100k_smoke/eval_fwd1p0_seed0_both_4x200.json`
- Both outputs returned `EVAL_OK`, `policy_mode=both`, `fixed_command=true`,
  action diagnostics and reward components present, and action/reward/obs NaN
  flags false.

| Command | Mode | Reward | Action Abs | tracking_lin_vel | tracking_ang_vel | ang_vel_xy | orientation | termination |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `[0.5,0,0]` | deterministic | -3.9345 | 0.1268 | 8.0171 | 21.7543 | -56.3528 | -35.5030 | -100 |
| `[0.5,0,0]` | stochastic | -5.9812 | 0.5193 | 6.3323 | 4.3008 | -122.0122 | -46.7654 | -100 |
| `[1.0,0,0]` | deterministic | -3.9779 | 0.1255 | 1.8457 | 21.7867 | -53.9996 | -34.8214 | -100 |
| `[1.0,0,0]` | stochastic | -5.9239 | 0.5188 | 2.7226 | 3.8782 | -118.0140 | -44.9634 | -100 |

Interpretation:

- This is not an infrastructure failure: train, checkpoint readiness, and smoke
  eval all passed.
- `fixed_alpha=0.03` preserves alpha but does not solve forward tracking at
  100k.
- Both `fwd0.5` and `fwd1.0` smokes have negative reward, low
  `tracking_lin_vel`, and `termination=-100`.
- Do not run 250k, 5M, or 10M from this result.
- Next short diagnostic should be `fixed_alpha=0.05` 100k first, with
  `alpha_floor=0.03` as the next alternative.

Warnings:

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `snap-confine` blocked some `uv` attempts; same commands were rerun
  externally unchanged.
- No traceback, OOM, fatal CUDA error, NaN, checkpoint readiness failure, or
  eval failure was observed.

## Fresh 100k Fixed-Alpha 0.05 Diagnostic

Gate name: fresh env1024 R3 100k fixed-alpha diagnostic, second variant.

- Variant: `fixed_alpha=0.05`.
- Scope: short diagnostic training, checkpoint readiness, and small
  fixed-command smoke already completed before this report update.
- No 250k, 5M, 10M, render, code change, reward, `action_scale`, Kp, PPO/RSL,
  or checkpoint schema change was made for this report update.
- Training status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p05/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.

Training metrics:

| env_steps | gradient_steps | wall_time | sps | actor_loss | critic_loss | q | target_q | reward_mean | done_fraction | discount_mean |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 99328 | 1312 | 46.3739s | 2141.8925 | -6.27606 | 0.344604 | 5.32498 | 5.24534 | -0.175746 | 0.0429688 | 0.957031 |

Alpha metrics:

| alpha_raw | log_alpha_raw | alpha_effective | log_alpha_effective | fixed_alpha | alpha_floor | alpha_floor_active | alpha_loss_type_id | alpha_loss | alpha_log_prob | alpha_grad_proxy_exp | alpha_grad_proxy_log |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0497871 | -3.0 | 0.05 | -2.99573 | 0.05 | 0.0 | 0.0 | 0.0 | 1.29048 | -18.6699 | 1.29048 | 25.9199 |

Interpretation: the fixed-alpha mechanism worked. Effective alpha stayed fixed
at `0.05`, raw `log_alpha` stayed at init `-3.0`, and `alpha_floor` remained
inactive.

Actor drift metrics:

| Metric | Final | Interval |
|---|---:|---:|
| actor policy mean abs mean | 0.157220 | 0.122486 |
| actor policy mean abs max | 1.51309 | 0.760698 |
| actor log_std mean | -0.153414 | -0.133005 |
| actor log_std min | -0.640887 | -0.577784 |
| actor log_std max | 0.065457 | 0.239801 |
| actor policy std mean | 0.859991 | 0.880084 |
| sampled action abs mean | 0.524748 | 0.521151 |
| sampled action saturation 0.95 | 0.0409483 | 0.0425800 |
| deterministic action abs mean | 0.149726 | 0.119732 |
| deterministic action saturation 0.95 | 0.0 | 0.0 |
| alpha effective | 0.05 | 0.05 |
| alpha raw | 0.0497871 | 0.0497871 |

Small fixed-command smoke:

- `fwd0.5` JSON:
  `./logs/sac_eval_fixed_alpha_100k_smoke/eval_fwd0p5_seed0_both_4x200_alpha0p05.json`
- `fwd1.0` JSON:
  `./logs/sac_eval_fixed_alpha_100k_smoke/eval_fwd1p0_seed0_both_4x200_alpha0p05.json`
- Both outputs returned `EVAL_OK`, `policy_mode=both`, `fixed_command=true`,
  and action/reward/obs NaN flags false.

| Command | Mode | Reward | Reward Std | Reward Min | Reward Max | Action Abs | Sat 0.95 | tracking_lin_vel | tracking_ang_vel | feet_phase | ang_vel_xy | orientation | feet_slip | termination |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `[0.5,0,0]` | deterministic | -3.7355 | 0.6304 | -4.4104 | -2.7027 | 0.1138 | 0.0 | 8.7119 | 24.1327 | 22.3450 | -52.3156 | -37.8373 | -16.4873 | -100 |
| `[0.5,0,0]` | stochastic | -5.9704 |  |  |  | 0.5190 | 0.0380 | 7.5620 | 4.0102 | 19.9924 | -124.6087 | -47.3861 | -10.2421 | -100 |
| `[1.0,0,0]` | deterministic | -3.8258 | 0.4952 | -4.3698 | -3.0269 | 0.1117 | 0.0 | 2.6894 | 24.6694 | 22.4319 | -51.3789 | -37.9635 | -16.5967 | -100 |
| `[1.0,0,0]` | stochastic | -6.1849 |  |  |  | 0.5186 | 0.0381 | 3.2869 | 3.9803 | 20.1518 | -126.3973 | -48.2956 | -11.0013 | -100 |

Interpretation:

- This is not an infrastructure failure: train, checkpoint readiness, and smoke
  eval all passed.
- `fixed_alpha=0.05` preserves alpha, but it does not solve forward tracking at
  100k.
- `critic_loss=0.3446` crosses the previous `0.3` watch threshold.
- `fixed_alpha=0.05` did not improve the fixed-forward smoke versus
  `fixed_alpha=0.03`; both `fwd0.5` and `fwd1.0` still terminate in the 4x200
  smoke, and `tracking_lin_vel` remains low.
- Do not continue to 250k, 5M, or 10M from fixed alpha alone.
- The next route should shift toward reward/prior targeted audit or ablation;
  `alpha_floor=0.03` remains a secondary alpha/entropy diagnostic, but do not
  keep increasing fixed alpha blindly.

Warnings:

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- Sandbox `snap-confine` blocked some `uv` attempts; same commands were rerun
  externally unchanged.
- No traceback, OOM, fatal CUDA error, NaN, checkpoint readiness failure, or
  eval failure was observed.

## Interpretation

- Fixed-command eval support closes the gap between render-only command
  inspection and bounded vectorized eval.
- The `[0.5, 0.0, 0.0]` smoke shows deterministic fixed-forward eval can be
  positive on the 3M R3 checkpoint while stochastic fixed-forward eval remains
  negative.
- This supports the existing interpretation: deterministic mean-policy behavior
  contains useful command-following signal, but stochastic policy quality and
  entropy health remain unresolved.
- The alpha sign audit does not justify a sign-bug patch by itself. It does
  justify treating alpha floor, fixed alpha, standard log-alpha update, and
  target-entropy sweeps as controlled ablations before longer 5M/10M runs.
- Both fixed-alpha `0.03` and `0.05` verified the mechanism but failed to
  produce useful 100k fixed-forward tracking. Higher fixed alpha alone is not a
  credible next route.

## Next Action

Do not run 5M or 10M yet.

The fixed-forward gate has now covered `[0.5, 0.0, 0.0]` and
`[1.0, 0.0, 0.0]`. The results are not strong enough to justify longer
training. The next route should be targeted alpha/entropy ablation or a
fixed-command `fwd1.0` render/video review, with alpha-floor, fixed-alpha, and
standard log-alpha update variants as the main ablation candidates. Remaining
fixed-command coverage for `[0.0, 0.3, 0.0]`, `[0.0, 0.0, 0.5]`, and
`[0.0, 0.0, 0.0]` is still useful, but should not be used to justify 5M/10M
without resolving the forward tracking weakness and stochastic collapse.

The first fixed-alpha gate (`fixed_alpha=0.03`, fresh env1024 R3 100k) preserved
effective alpha but produced weak fixed-command smoke. Do not extend it to 250k
or longer from this result. If continuing alpha/entropy diagnostics, run another
short 100k gate with `fixed_alpha=0.05` first; use `alpha_floor=0.03` as the
next alternative.

The second fixed-alpha gate (`fixed_alpha=0.05`, fresh env1024 R3 100k) also
preserved effective alpha but produced weak fixed-command smoke and a critic
loss watch item. Do not extend fixed-alpha variants to 250k, 5M, or 10M from
these results. The next main route should be reward/prior targeted audit or
ablation. Treat `alpha_floor=0.03` as a secondary diagnostic rather than the
primary path, and do not keep increasing fixed alpha blindly.
