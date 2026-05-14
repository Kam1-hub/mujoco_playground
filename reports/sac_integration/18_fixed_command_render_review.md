# Fixed-Command Render Review

## Context

- Checkpoint: fixed-command render helper increment after
  `0f628c5 Add SAC checkpoint render helper`.
- Code change: `scripts/render_sac_checkpoint.py` now supports
  `--fixed_command`, `--command_x`, `--command_y`, and `--command_yaw`.
- Purpose: distinguish joystick-command behavior from videos that use the
  random command sampled by env reset/resampling.
- No training was run for this report update.
- No eval benchmark was run for this report update.
- No SAC training math, loss math, reward, `action_scale`, Kp, env behavior,
  PPO/RSL path, or checkpoint schema was changed.
- Runtime artifacts remain under ignored `logs/` and must not be committed.

## Implementation Summary

When `--fixed_command true` is passed, the render helper forces the requested
3D joystick command throughout the rollout. The override updates both:

- `state.info["command"]`
- the command slice `9:12` in `obs["state"]` and `obs["privileged_state"]`

Default behavior is unchanged when `--fixed_command false`; the helper keeps
using the command sampled by env reset/resampling.

## Validation

Static checks:

- `git diff --check`: PASS.
- `uv run --no-sync python scripts/render_sac_checkpoint.py --help`: PASS.
- Help output includes `--fixed_command`, `--command_x`, `--command_y`, and
  `--command_yaw`.

Review result:

- Huygens review found no blockers.
- Fixed mode updates both the env info command and policy-observation command
  slice after reset and after each step.
- The hard-coded obs command slice `9:12` matches the G1 joystick observation
  layout: linvel 3, gyro 3, gravity 3, command 3.
- Additive JSON fields `fixed_command` and `command` are expected and do not
  change old rollout behavior.

## Fixed-Command Smoke Artifacts

All three smokes used the 3M R3 checkpoint:

```text
./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl
```

All renders used deterministic policy mode, seed `0`, `episode_length=600`,
`640x480`, `fps=30`, and `render_every=1`.

| Command | Status | Done | Frames | Total Reward | MP4 | JSON |
|---|---|---:|---:|---:|---|---|
| `[0.5, 0.0, 0.0]` | `RENDER_OK` | false | 600 | 5.6711320877075195 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0p5_y0_yaw0_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0p5_y0_yaw0_seed0_det_600.json` |
| `[0.0, 0.0, 0.0]` | `RENDER_OK` | false | 600 | -17.35877227783203 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0_seed0_det_600.json` |
| `[0.0, 0.0, 0.5]` | `RENDER_OK` | false | 600 | 6.60101842880249 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0p5_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0p5_seed0_det_600.json` |

## Caveats

- Render videos are diagnostic visual evidence, not a policy-quality benchmark.
- Render reward should not be compared numerically with vectorized eval reward.
- Fixed command values are not range-checked by the CLI; out-of-distribution
  commands remain the caller's responsibility.
- The MP4/JSON files under `logs/` are local ignored artifacts and should not
  be committed.

## Next Action

Use the fixed-command MP4s for command-specific visual inspection before
deciding on longer SAC runs. The next SAC stabilization step should remain a
bounded decision review; do not jump blindly to 750k, 1M, 5M, or 10M from this
render helper update alone.

## Follow-Up Gate

Fixed-command eval support has now been added to
`scripts/eval_sac_checkpoint.py` with the same command override semantics as
this render helper. The 3M R3 `[0.5, 0.0, 0.0]` eval smoke returned `EVAL_OK`,
deterministic reward mean `0.5099`, stochastic reward mean `-2.3354`, and no
action/reward/obs NaN flags. The related alpha sign audit found no direct SAC
temperature sign bug, but confirmed persistent downward alpha pressure remains
a risk. Details are recorded in
`reports/sac_integration/19_alpha_entropy_and_fixed_eval_gate.md`.
