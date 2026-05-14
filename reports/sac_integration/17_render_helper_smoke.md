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

Next action: inspect
`./logs/sac_render_3m_r3/render_seed0_det.mp4` visually. After inspection,
choose between entropy/alpha ablation design, a bounded longer run, or further
render/reward diagnostics. Do not jump directly to 10M.
