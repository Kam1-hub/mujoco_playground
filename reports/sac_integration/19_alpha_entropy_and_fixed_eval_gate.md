# Alpha Entropy And Fixed-Command Eval Gate

## Context

- Base commit: `fd8a014 Add SAC fixed-command render support`.
- Scope: eval-only fixed joystick command support plus alpha/entropy sign audit.
- No training was run for this report update.
- No render or full benchmark was run for this report update.
- No SAC train/loss/reward/action_scale/Kp/env/PPO/RSL/checkpoint schema
  changes were made.
- Runtime artifacts remain under ignored `logs/` and must not be committed.

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

Before longer training, run full fixed-command eval-only coverage across:

- `[0.5, 0.0, 0.0]`
- `[1.0, 0.0, 0.0]`
- `[0.0, 0.3, 0.0]`
- `[0.0, 0.0, 0.5]`
- `[0.0, 0.0, 0.0]`

Each command should use bounded deterministic/stochastic eval, action
diagnostics, reward components, and NaN checks. Then decide whether the next
step is an alpha-floor/fixed-alpha/log-alpha ablation or a gated longer run.
