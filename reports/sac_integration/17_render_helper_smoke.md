# Render Helper Smoke

## Context

- Checkpoint: `CHECKPOINT AP - SAC Render Helper Smoke`
- Code change: added eval-only render helper `scripts/render_sac_checkpoint.py`.
- Purpose: visually inspect the deterministic 3M R3 policy before deciding on
  any 5M, 10M, or entropy/alpha ablation path.
- No training was run.
- No SAC loss math, reward, `action_scale`, Kp, env behavior, PPO/RSL path, or
  checkpoint schema was changed.

## Validation

Static validation:

- `uv run --no-sync python -m compileall scripts learning/sac_lift`: PASS.
- `uv run --no-sync python scripts/render_sac_checkpoint.py --help`: PASS.
- Fixed-command help flags: `--fixed_command`, `--command_x`, `--command_y`,
  and `--command_yaw`: PASS.

Render smoke:

```bash
uv run --no-sync python scripts/render_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl \
  --seed 0 \
  --episode_length 300 \
  --width 640 \
  --height 480 \
  --fps 30 \
  --policy_mode deterministic \
  --output ./logs/sac_render_3m_r3/render_seed0_det.mp4
```

Result:

| Field | Value |
|---|---:|
| status | `RENDER_OK` |
| output | `./logs/sac_render_3m_r3/render_seed0_det.mp4` |
| type | MP4 |
| frames | 300 |
| duration | 10s |
| resolution | 640x480 |
| fps | 30 |
| size | 1,552,965 bytes |
| total rollout reward | 4.084568977355957 |
| done | false |

The MP4 is under ignored `logs/` and must not be committed.

## Fixed-Command Render Support

The render helper now supports explicit joystick commands for visual
diagnostics:

- `--fixed_command`
- `--command_x`
- `--command_y`
- `--command_yaw`

This is intended to separate command-following inspection from videos that use
the random command sampled by env reset/resampling. Default behavior is
unchanged when `--fixed_command false`.

Implementation review found no blockers:

- Fixed mode updates `state.info["command"]`.
- Fixed mode updates command slice `9:12` in both `obs["state"]` and
  `obs["privileged_state"]` after reset and after each step.
- The command slice matches the G1 joystick observation layout.
- No SAC training/loss/reward/action_scale/Kp/env/PPO/RSL/checkpoint schema
  changes were made.

Fixed-command 3M R3 smoke artifacts:

| Command | Status | Done | Frames | MP4 | JSON |
|---|---|---:|---:|---|---|
| `[0.5, 0.0, 0.0]` | `RENDER_OK` | false | 600 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0p5_y0_yaw0_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0p5_y0_yaw0_seed0_det_600.json` |
| `[0.0, 0.0, 0.0]` | `RENDER_OK` | false | 600 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0_seed0_det_600.json` |
| `[0.0, 0.0, 0.5]` | `RENDER_OK` | false | 600 | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0p5_seed0_det_600.mp4` | `./logs/sac_render_3m_r3_fixedcmd/render_cmd_x0_y0_yaw0p5_seed0_det_600.json` |

These files are under ignored `logs/` and must not be committed. Fixed command
values are not range-checked by the CLI, so out-of-distribution command values
remain caller responsibility.

## Review Notes

The code review found no blockers.

- The helper is eval-only and does not enter a training path.
- It loads checkpoints through `learning.sac_lift.checkpoint.load`.
- It checks checkpoint eval readiness requirements before rendering.
- It reconstructs the actor network from checkpoint/config metadata.
- The deterministic action path calls
  `networks.sample_action(..., deterministic=True)`, which resolves to
  `tanh(mean)`.
- It applies `policy_normalizer` to the selected policy observation key.
- It uses a raw single env from `registry.load`, while vectorized eval uses the
  training wrapper. This is acceptable for visual inspection, but render reward
  should not be treated as numerically identical to vectorized eval reward.

## Visual Sanity

Main-agent frame inspection found the video nonblank, with the humanoid upright
in sampled frames and no obvious fall. This is only a quick sanity check; the
MP4 still needs human/video inspection before any 5M or 10M decision.

## Warnings

- Known non-fatal WSL2 CUDA driver version warning.
- Known non-fatal JAX cast overflow warning.
- Initial sandbox `uv` attempt hit the known snap-confine capability issue; the
  exact same render command was rerun externally unchanged.
- No traceback was observed in the successful render run.

## Interpretation

The render helper closes the immediate tooling gap: the 3M deterministic policy
can now be rendered from a checkpoint. This does not by itself prove gait
quality or stable SAC integration. It provides the required visual-inspection
artifact for deciding whether the 3M deterministic reward recovery represents
meaningful motion or reward exploitation.

Next action: inspect the fixed-command MP4s under
`./logs/sac_render_3m_r3_fixedcmd/` visually. After inspection, choose between
entropy/alpha ablation design, a bounded longer run, or further render/reward
diagnostics. Do not jump directly to 10M.
