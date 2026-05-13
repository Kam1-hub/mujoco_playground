# Action Distribution Diagnostics

Status: recorded on 2026-05-13 after CHECKPOINT W.

## Context

- Scope: eval-only full action diagnostic.
- Head before report update: `8ff4f1d Add SAC action distribution eval diagnostics`.
- Output directory: `./logs/sac_eval_action_diag_full/`.
- JSON output count: `15`.
- Checkpoints: 100k, 250k, and 500k sanity checkpoints.
- Policy modes: deterministic and stochastic through `--policy_mode both`.
- Seeds: `0`, `1`, `2`, `3`, `4`.
- Eval scale: `num_eval_envs=16`, `episode_length=1000`.
- Checkpoint readiness: PASS for all three checkpoints.
- Eval status: all deterministic and stochastic runs returned `EVAL_OK`.
- NaN flags: all `action_nan=false`, `reward_nan=false`, and `obs_nan=false`.
- No training was run for this diagnostic.
- No SAC code was modified by this diagnostic run.

Runtime artifacts remain ignored under `logs/`; do not commit generated JSON
files or checkpoints.

## Deterministic Aggregates

| Scale | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | Det Sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k | -4.2130 | 0.3669 | 0.1823 | 0.1950 | -0.1077 | 0.8996 | 0.0000004 |
| 250k | -4.4792 | 0.3633 | 0.2147 | 0.2288 | -0.1415 | 0.8692 | 0.0000 |
| 500k | -4.8204 | 0.3151 | 0.3029 | 0.3468 | -0.2909 | 0.7519 | 0.00113 |

Top saturated action dimensions:

- 100k: `1`, `7`, `0`, `6`, `2`, `8`, `22`, `9`
- 250k: `9`, `3`, `14`, `6`, `11`, `12`, `25`, `10`
- 500k: `11`, `10`, `14`, `27`, `9`, `13`, `1`, `7`

## Stochastic Aggregates

| Scale | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | Sto Sat |
|---|---:|---:|---:|---:|---:|---:|---:|
| 100k | -6.4741 | 0.5045 | 0.5274 | 0.2240 | -0.1562 | 0.8578 | 0.0445 |
| 250k | -6.2374 | 0.4256 | 0.5236 | 0.2712 | -0.1939 | 0.8254 | 0.0415 |
| 500k | -5.8911 | 0.5841 | 0.5221 | 0.3954 | -0.3288 | 0.7256 | 0.0413 |

Top saturated action dimensions:

- 100k: `7`, `8`, `6`, `1`, `14`, `0`, `9`, `12`
- 250k: `3`, `9`, `14`, `7`, `6`, `25`, `13`, `11`
- 500k: `14`, `11`, `10`, `13`, `9`, `7`, `25`, `18`

## Reward Component Trend

Deterministic degradation aligns mainly with these components:

| Component | 100k | 250k | 500k |
|---|---:|---:|---:|
| `reward/ang_vel_xy` | -50.82 | -62.66 | -73.31 |
| `reward/stand_still` | -18.72 | -21.72 | -28.33 |
| `reward/orientation` | -34.80 | -43.27 | -40.37 |

Positive components partially offset the deterministic degradation:

| Component | 100k | 250k | 500k |
|---|---:|---:|---:|
| `reward/feet_phase` | 27.44 | 29.75 | 34.04 |
| `reward/tracking_lin_vel` | 7.68 | 10.60 | 14.19 |

Stochastic differs from deterministic:

| Component | 100k | 250k | 500k |
|---|---:|---:|---:|
| `reward/ang_vel_xy` | -131.89 | -123.32 | -119.50 |
| `reward/orientation` | -43.91 | -45.99 | -36.25 |

Stochastic reward improves overall despite lower entropy / narrower policy
standard deviation.

## Interpretation

This is not a runtime failure. It is not stochastic policy collapse.

The key risk is deterministic deployment/eval degradation driven by actor
mean/action magnitude drift:

- deterministic action abs rises from `0.1823` to `0.2147` to `0.3029`;
- policy mean abs rises from `0.1950` to `0.2288` to `0.3468`;
- policy log_std mean falls from `-0.1077` to `-0.1415` to `-0.2909`;
- policy std mean narrows from `0.8996` to `0.8692` to `0.7519`;
- deterministic saturation remains low but becomes nonzero at 500k.

Entropy/alpha dynamics remain likely upstream because alpha/log_std/std
decrease while policy mean/action magnitude increase. The current evidence
suggests a deterministic eval/deployment path issue tied to actor mean drift and
specific reward components, especially angular velocity, stand-still, and
orientation.

## Next Actions

Do not run 750k or 1M yet.

Recommended next diagnostic/design work:

- inspect actor mean drift by action dimension and reward components;
- compare deterministic vs stochastic eval behavior in a targeted report;
- review target entropy, alpha loss, and log_std dynamics;
- consider instrumentation or controlled ablation before longer runs;
- do not tune reward, `action_scale`, or Kp yet.
