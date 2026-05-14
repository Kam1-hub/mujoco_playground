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
