# AngVelXY Reward Scale Diagnostic

Date: 2026-05-15

## Context

- Checkpoint: AngVelXY 100k Gate.
- Commit under test: `d2b76bf Add SAC angular velocity reward override`.
- Scope: fresh env1024/R3/UTD-preserving 100k diagnostic run.
- Variant: reset-calm baseline plus `--env_reward_ang_vel_xy_scale -0.5`.
- Reset-calm baseline retained:
  `--env_reset_joint_noise_scale 0.0 --env_reset_root_qvel_scale 0.0`.
- Isolation: no action-rate override, push-disable, zero-command phase-freeze,
  feet-air-time command mask, feet-slip mode, or feet-slip scale override was
  combined with this run.
- Purpose: test whether stronger torso XY angular-velocity penalty improves
  fixed-command early fall beyond reset-calm.
- No 250k, 5M, or 10M run was authorized or executed for this gate.

## Training

- Status: `TRAIN_OK`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_reset_calm_angvelxy_m0p5/sac_lift_step_99328.pkl`.
- Checkpoint readiness: PASS.

| metric | value |
|---|---:|
| env_steps | 99328 |
| gradient_steps | 1552 |
| wall_time | 75.75s |
| sps | 1311.23 |
| actor_loss | -5.3756 |
| critic_loss | 0.08565 |
| alpha | 0.042784 |
| log_alpha | -3.15159 |
| reward_mean | -0.1733 |
| done_fraction | 0.0078125 |

## Eval Summary

Fixed-command eval completed in both deterministic and stochastic modes for
`fwd0.5`, `fwd1.0`, and `stand`. All evals returned `EVAL_OK`, and no
action/reward/obs NaN flags were observed.

| command | mode | reward | first_done | reason | root_h | up_z | torso_ang_xy |
|---|---|---:|---:|---|---:|---:|---:|
| fwd0.5 | deterministic | -3.5869 | 68.5 | fall 4/4 | -0.1829 | -0.0416 | 4.5852 |
| fwd1.0 | deterministic | -3.8951 | 68.75 | fall 4/4 | -0.2017 | -0.0420 | 4.4262 |
| stand | deterministic | -6.2890 | 68.5 | fall 4/4 | -0.1798 | -0.0580 | 4.6635 |

Stochastic mode also fell `4/4` in all three commands and was generally
earlier and worse than deterministic mode.

## Render Summary

All deterministic render smokes returned `RENDER_OK`, used stop-on-done
behavior, and wrote ignored video plus JSON artifacts.

| command | status | first_done | reason | artifact status |
|---|---|---:|---|---|
| fwd0.5 | `RENDER_OK` | 68 | fall | MP4 and JSON present |
| fwd1.0 | `RENDER_OK` | 69 | fall | MP4 and JSON present |
| stand | `RENDER_OK` | 68 | fall | MP4 and JSON present |

Artifacts:

- Eval JSONs: `./logs/sac_eval_reset_calm_angvelxy_100k_fixed/` (`3` JSONs).
- Render artifacts:
  `./logs/sac_render_reset_calm_angvelxy_100k_smoke/`
  (`3` MP4 files plus `3` JSON files).

## Comparison To Reset-Calm

- Reset-calm baseline eval first done was around `67.5`; render first done was
  around `68`.
- This run was effectively unchanged: eval `68.5 / 68.75 / 68.5`, render
  `68 / 69 / 68`.
- Deterministic rewards worsened materially:
  - fwd0.5: `-2.0443 -> -3.5869`
  - fwd1.0: `-2.3095 -> -3.8951`
  - stand: `-4.4102 -> -6.2890`
- Stronger `ang_vel_xy` penalty increased the negative angular-velocity
  component but did not prevent fall.

## Conclusion

- Gate result: FAIL for fixed-command early-fall improvement.
- This is a negative gate, not a runtime failure.
- `ang_vel_xy=-0.5` does not justify 250k, 5M, or 10M.
- The project is paused/stopped per user instruction after recording this
  result.
- If this work is revisited later, do not continue this route automatically.
  Consider base-height target audit, alive/survival incentive, or command
  warmup as design topics only.

## Warnings

- Known non-fatal WSL2 CUDA driver warning.
- Known non-fatal JAX cast overflow warning.
- Initial `uv` sandbox `snap-confine` failure; rerun externally with unchanged
  parameters.
- No NaN, OOM, fatal CUDA, checkpoint failure, eval failure, or render failure
  was observed.
