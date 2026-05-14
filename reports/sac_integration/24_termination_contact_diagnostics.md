# Termination Contact Diagnostics

Date: 2026-05-15

## Context

- Commit under test: `fe69d8d Add SAC termination eval diagnostics`.
- Scope: eval-only termination/contact diagnostic sweep over existing 100k
  push-disable, zero-command phase-freeze, and feet-air-time command-mask
  checkpoints.
- Output directory: `./logs/sac_eval_termination_diag_100k/`.
- JSON count: `9`.
- Variants: `push_disable`, `phase_freeze`, `feet_air_time_mask`.
- Commands: `fwd0.5`, `fwd1.0`, and `stand`.
- Modes: deterministic and stochastic.
- All evals returned `EVAL_OK`.
- All action/reward/obs NaN flags were false.
- No training, preflight, code change, or report change was performed during
  the sweep itself.

## Sweep Result

| variant | command | mode | reward | tracking_lin | termination | first_done_mean | first_done_range | fall | contact_any | torso_up_z_mean | root_height_mean | torso_ang_vel_xy_mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| push_disable | fwd0.5 | det | -3.6472 | 9.9450 | -100 | 54.25 | 48-68 | 4 | 0 | -0.0561 | -0.2618 | 5.9652 |
| push_disable | fwd0.5 | stoch | -5.9446 | 9.1982 | -100 | 51.25 | 46-59 | 4 | 0 | -0.0113 | -0.1012 | 6.1006 |
| push_disable | fwd1.0 | det | -3.7750 | 5.2704 | -100 | 54.50 | 49-68 | 4 | 0 | -0.0711 | -0.2646 | 6.0401 |
| push_disable | fwd1.0 | stoch | -6.1786 | 3.8591 | -100 | 51.75 | 47-58 | 4 | 0 | -0.0535 | -0.1109 | 5.8003 |
| push_disable | stand | det | -6.4732 | 7.9120 | -100 | 54.25 | 48-68 | 4 | 0 | -0.0643 | -0.2717 | 5.9043 |
| push_disable | stand | stoch | -10.4980 | 10.2343 | -100 | 52.00 | 47-60 | 4 | 0 | -0.0847 | -0.1195 | 6.8909 |
| phase_freeze | fwd0.5 | det | -3.5786 | 9.9990 | -100 | 53.00 | 47-67 | 4 | 0 | -0.0368 | -0.2302 | 5.8633 |
| phase_freeze | fwd0.5 | stoch | -6.4501 | 9.2514 | -100 | 52.00 | 47-57 | 4 | 0 | -0.0615 | -0.1047 | 5.9190 |
| phase_freeze | fwd1.0 | det | -3.7812 | 5.4509 | -100 | 53.50 | 48-67 | 4 | 0 | -0.0776 | -0.2476 | 5.9809 |
| phase_freeze | fwd1.0 | stoch | -6.5442 | 3.6362 | -100 | 52.75 | 47-58 | 4 | 0 | -0.0681 | -0.1065 | 7.4550 |
| phase_freeze | stand | det | -6.3403 | 7.6392 | -100 | 53.00 | 47-67 | 4 | 0 | -0.0427 | -0.2140 | 5.7336 |
| phase_freeze | stand | stoch | -11.0311 | 10.7181 | -100 | 53.00 | 47-59 | 4 | 0 | -0.0522 | -0.0884 | 6.7242 |
| feet_air_time_mask | fwd0.5 | det | -3.6332 | 10.2312 | -100 | 55.00 | 49-68 | 4 | 0 | -0.0567 | -0.2109 | 5.9815 |
| feet_air_time_mask | fwd0.5 | stoch | -6.4714 | 10.5826 | -100 | 54.00 | 48-59 | 4 | 0 | -0.0984 | -0.1210 | 6.6367 |
| feet_air_time_mask | fwd1.0 | det | -3.7544 | 4.2751 | -100 | 55.25 | 50-68 | 4 | 0 | -0.0561 | -0.2008 | 5.9953 |
| feet_air_time_mask | fwd1.0 | stoch | -5.9671 | 4.0323 | -100 | 50.50 | 41-59 | 3 | 1 | 0.0961 | 0.0558 | 4.8977 |
| feet_air_time_mask | stand | det | -6.4800 | 8.6753 | -100 | 55.00 | 49-68 | 4 | 0 | -0.0511 | -0.2167 | 5.9376 |
| feet_air_time_mask | stand | stoch | -10.9260 | 11.4141 | -100 | 53.25 | 48-58 | 4 | 0 | -0.0609 | -0.0995 | 7.3160 |

## Interpretation

- The sweep is not a runtime failure: all evals completed with `EVAL_OK`, no
  NaN flags, no OOM, no fatal CUDA error, and no eval failure.
- `reward/termination=-100` appears in all 18 variant/command/mode cases, and
  `no_done_count=0` everywhere.
- First termination happens early and consistently, mostly around `51` to `55`
  steps.
- The failures are fall-dominated. All cases report fall counts of `4` except
  `feet_air_time_mask fwd1.0 stochastic`, which reports fall `3` plus one
  `right_foot_left_foot` contact.
- Illegal contact is not the dominant failure mode: `contact_any=0` in 17 of
  18 cases.
- Numerical instability is not the failure mode: qpos/qvel NaN counts are
  always `0`.
- Terminal state statistics consistently show torso upright failure:
  torso-up z is negative or near the fall threshold, root height is often
  negative, and torso angular velocity in XY is high, with means around
  `5.7` to `7.5`.
- None of push-disable, zero-command phase-freeze, or feet-air-time command
  mask meaningfully delays first termination at 100k.

## Consequence

The root 100k blocker is early torso fall/upright instability, not illegal
foot contact, not qpos/qvel NaN, and not an isolated zero-command
`feet_air_time` reward conflict.

This does not justify 250k, 5M, or 10M training. The next target should be
torso fall, orientation, base-stability, and terminal-state diagnostics or a
controlled stabilization/curriculum design before any longer run.

Do not tune reward weights, `action_scale`, or Kp from this report alone.

## Warnings

- Known non-fatal WSL2 CUDA/JAX warnings.
- Sandbox `snap-confine` reruns were required for some `uv` commands.
- No NaN, OOM, fatal CUDA, checkpoint failure, or eval failure was observed.
