# Action Joint Mapping Diagnostic

Status: recorded after `4ffd301 Add SAC action diagnostic mapping tools`.

This report records a no-training, no-eval interpretability diagnostic. It
uses existing full action diagnostic JSON under ignored `logs/` and joins the
top action dimensions to G1 actuator/joint names.

## Context

- Scope: action dimension to joint mapping plus existing action diagnostic
  summary.
- No training was run.
- No eval was run.
- No SAC algorithm, reward, `action_scale`, Kp, PPO, or RSL behavior changed.
- Runtime artifacts:
  - `./logs/sac_action_mapping/g1_flat_action_mapping.json`
  - `./logs/sac_action_mapping/action_diag_joint_summary.json`
- Runtime artifacts remain ignored and are not committed.

## Tools

Added helper scripts:

- `scripts/inspect_g1_action_mapping.py`
- `scripts/summarize_sac_action_diag.py`

Validation performed:

```bash
python3 -m compileall scripts
python3 scripts/inspect_g1_action_mapping.py --help
python3 scripts/summarize_sac_action_diag.py --help
.venv/bin/python scripts/inspect_g1_action_mapping.py \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --output_json ./logs/sac_action_mapping/g1_flat_action_mapping.json
.venv/bin/python scripts/summarize_sac_action_diag.py \
  --mapping_json ./logs/sac_action_mapping/g1_flat_action_mapping.json \
  --input_dir ./logs/sac_eval_action_diag_full \
  --output_json ./logs/sac_action_mapping/action_diag_joint_summary.json
```

The mapping command produced a non-fatal JAX CUDA plugin initialization warning
but completed successfully. It only needs MuJoCo model metadata for this use.

## Deterministic Summary

| Scale | Reward Avg | Policy Mean Abs | Policy Std Mean | Top mapped action dimensions |
|---|---:|---:|---:|---|
| 100k | -4.2130 | 0.1950 | 0.8996 | `1 left_hip_roll_joint`, `7 right_hip_roll_joint`, `0 left_hip_pitch_joint`, `6 right_hip_pitch_joint`, `2 left_hip_yaw_joint` |
| 250k | -4.4792 | 0.2288 | 0.8692 | `9 right_knee_joint`, `3 left_knee_joint`, `14 waist_pitch_joint`, `6 right_hip_pitch_joint`, `11 right_ankle_roll_joint` |
| 500k | -4.8204 | 0.3468 | 0.7519 | `11 right_ankle_roll_joint`, `10 right_ankle_pitch_joint`, `14 waist_pitch_joint`, `27 right_wrist_pitch_joint`, `9 right_knee_joint` |

Interpretation:

- Deterministic policy mean magnitude increases while std narrows.
- Top mapped deterministic action dimensions shift from hip-dominant at 100k
  toward knee, ankle, and waist dimensions by 250k/500k.
- 500k deterministic degradation is especially associated with right ankle
  roll/pitch, waist pitch, and right knee action dimensions.

## Stochastic Summary

| Scale | Reward Avg | Policy Mean Abs | Policy Std Mean | Top mapped action dimensions |
|---|---:|---:|---:|---|
| 100k | -6.4741 | 0.2240 | 0.8578 | `7 right_hip_roll_joint`, `8 right_hip_yaw_joint`, `6 right_hip_pitch_joint`, `1 left_hip_roll_joint`, `14 waist_pitch_joint` |
| 250k | -6.2374 | 0.2712 | 0.8254 | `3 left_knee_joint`, `9 right_knee_joint`, `14 waist_pitch_joint`, `7 right_hip_roll_joint`, `6 right_hip_pitch_joint` |
| 500k | -5.8911 | 0.3954 | 0.7256 | `14 waist_pitch_joint`, `11 right_ankle_roll_joint`, `10 right_ankle_pitch_joint`, `13 waist_roll_joint`, `9 right_knee_joint` |

Interpretation:

- Stochastic reward improves despite policy std narrowing.
- Stochastic top dimensions overlap the deterministic 500k ankle/waist/knee
  set, but sampled actions remain higher magnitude and do not show the same
  reward degradation trend.
- This reinforces that the current issue is the deterministic deployment/eval
  path, not a simple stochastic policy collapse.

## Reward Component Alignment

Existing full action diagnostics already linked deterministic degradation to:

- `reward/ang_vel_xy`
- `reward/stand_still`
- `reward/orientation`

Mapped top dimensions suggest these reward changes plausibly relate to ankle,
waist, knee, and hip roll behavior. This is interpretive evidence only; it does
not prove causality.

## Implication For Next Work

This diagnostic supports the alpha/entropy ablation plan:

- if an ablation reduces actor mean magnitude and deterministic action
  magnitude, check whether the mapped top dimensions move away from the
  500k ankle/waist/knee concentration;
- if reward improves without shifting these dimensions, alpha/entropy may be
  affecting action distribution more globally;
- if dimensions remain concentrated and reward stays poor, consider a later
  targeted deterministic actor or action-regularization ablation.

Do not run fresh 500k, 750k, or 1M from this diagnostic alone.
