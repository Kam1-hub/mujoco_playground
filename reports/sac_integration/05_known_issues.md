# Known Issues

Status: updated on 2026-05-11 after CPU tiny SAC smoke.

## Open

| Issue | Category | Status | Evidence | Next action |
|---|---|---:|---|---|
| GPU smoke not validated | dependency/platform | OPEN | JAX sees only `cpu:0`; `nvidia-smi` and `nvcc` are not found; WSL command returned exit 1. | Run the 10k smoke only in WSL2/Linux CUDA or another CUDA JAX runtime. |
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

## Current Acceptance Position

- Minimum success is met: menagerie is present and flat/rough env API load/reset/step pass.
- Ideal local success is met: CPU tiny smoke passes and writes a checkpoint with finite actor/critic/alpha metrics.
- GPU training is not validated on this Windows host.
- No LIFT fork wholesale copy was made.
- No PPO/RSL behavior changes were made.
- No world model, fine-tuning, vision, domain randomization, or deployment work was added.
