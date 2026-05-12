# Known Issues

Status: updated on 2026-05-12 after WSL2 GPU preflight and Route B GPU 10k smoke.

## Open

| Issue | Category | Status | Evidence | Next action |
|---|---|---:|---|---|
| WSL2 GPU env vars not enforced by wrappers | runtime configuration | OPEN | `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MUJOCO_GL=egl` are operational requirements but not enforced in all entry points. | Export them before GPU checks; consider wrapper hardening after preflight is stable. |
| First JIT latency can look like a hang | validation noise | OPEN_NON_BLOCKING | WSL2/JAX first compile may take minutes. | Record wall time and wait through first compile before classifying a failure. |
| Route B actual env steps may be lower than requested | reporting | OPEN_NON_BLOCKING | Training loop uses `num_timesteps // num_envs`; `10000` with `128` envs yields `9984`. | Record actual `env_steps` and checkpoint filename; do not assume target equals actual. |
| Replay buffer scale can become the SAC VRAM bottleneck | memory | OPEN | Route B stores about 671 float32 values per transition; 1M raw replay is about 2.5-2.7 GB before JAX/XLA overhead. | Keep first smoke at `max_replay_size=8192`; increase only after measured GPU smoke. |
| 1M and longer runs not validated | validation scope | OPEN | GPU 10k smoke passed, but no longer sanity run has been executed. | Next step should be deterministic eval or a user-approved longer sanity run; do not jump straight to 1M. |
| PPO default env count is not a SAC smoke setting | scope | OPEN_NON_BLOCKING | Older PPO docs mention large env counts; Route B smoke is `num_envs=128`. | Do not import PPO/Barkour `2048` or `8192` env assumptions into SAC migration smoke. |
| Default G1 config reports `impl="warp"` | runtime backend | MITIGATED | CPU-only JAX cannot satisfy Warp's CUDA backend probe. | Route B and env checker now default to `impl="jax"` and expose `--impl`. Use `--impl warp` only where CUDA JAX/Warp is validated. |
| `state.info["truncation"]` absent | truncation | MITIGATED | Flat/rough env API both omit `truncation` in reset/step info keys. | SAC synthesizes zero truncation and reports `truncation_fraction`; revisit if env adds timeout metadata. |
| Route A runtime smoke not rerun | route coverage | OPEN | Route A help passes; this round focused on Route B and latest allowed write set did not include Route A file. | If Route A is needed on CPU, add the same explicit `--impl` override path and run a tiny Brax SAC smoke. |
| `compileall g1_env ...` traverses ignored menagerie assets | validation noise | OPEN | Menagerie is under `g1_env\external_deps`, so compileall lists that tree. | Accept as noisy but passing, or narrow future compile command if report policy allows. |
| Git global ignore permission warning | tooling | OPEN_NON_BLOCKING | `git status` and `git check-ignore` warn about `C:\Users\Kam1/.config/git/ignore` permission denied. | Optional local machine permission fix; project status checks still work. |
| PowerShell `Start-Process` wrapper failed | validation harness | CLOSED_NON_BLOCKING | Environment block had both `Path` and `PATH`; direct foreground run passed. | Use direct foreground command or a cleaner wrapper if long monitoring is needed. |

## Closed This Round

| Issue | Category | Resolution |
|---|---|---|
| Missing `g1_env\external_deps\mujoco_menagerie` | asset | User authorized download; cloned DeepMind menagerie and checked out `1b86ece576591213e2b666ebf59508454200ca97`; directory is ignored by project git. |
| Env load failed on CPU due default `impl="warp"` | runtime backend | Added `--impl` support and Route B/checker default `impl="jax"`; flat/rough env API now pass. |
| CPU tiny smoke not validated | validation | Exact CPU tiny command now passes with 256 env steps, 121 gradient steps, finite losses, and checkpoint. |
| GPU preflight not validated in target WSL2 workspace | CUDA/JAX backend | `uv sync --frozen --extra cuda` installed CUDA JAX plugin/runtime packages; JAX reports backend `gpu`, device `cuda:0`; `gpu_preflight.py --impl jax --require_gpu` passed. |
| Route B GPU 10k smoke not validated | validation | User authorized one controlled 10k smoke; wrapper reported `TRAIN_OK`, checkpoint `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`, finite scalar metrics, and no observed NaN. |
| CPU-only JAX after base sync | dependency | Base `uv sync` remains CPU-only, but the target WSL2 environment now uses the locked CUDA extra. Keep using `uv sync --frozen --extra cuda` for GPU validation. |

## Current Acceptance Position

- Minimum success is met: menagerie is present and flat/rough env API load/reset/step pass.
- Ideal local success is met: CPU tiny smoke passes and writes a checkpoint with finite actor/critic/alpha metrics.
- GPU preflight is validated in WSL2 with JAX backend `gpu` and device `cuda:0`.
- Route B GPU 10k smoke is validated with `TRAIN_OK` and checkpoint
  `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`.
- 1M, longer sanity runs, deterministic eval, and PPO comparison are still
  `NOT VALIDATED`.
- No LIFT fork wholesale copy was made.
- No PPO/RSL behavior changes were made.
- No world model, fine-tuning, vision, domain randomization, or deployment work was added.
