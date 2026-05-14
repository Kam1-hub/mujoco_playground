# Known Issues

Status: updated on 2026-05-14 after the fresh 100k actor-regularization R2/R3
coefficient sweep.

## Open

| Issue | Category | Status | Evidence | Next action |
|---|---|---:|---|---|
| WSL2 GPU env vars not enforced by wrappers | runtime configuration | OPEN | `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `MUJOCO_GL=egl` are operational requirements but not enforced in all entry points. | Export them before GPU checks; consider wrapper hardening after preflight is stable. |
| First JIT latency can look like a hang | validation noise | OPEN_NON_BLOCKING | WSL2/JAX first compile may take minutes. | Record wall time and wait through first compile before classifying a failure. |
| Route B actual env steps may be lower than requested | reporting | OPEN_NON_BLOCKING | Training loop uses `num_timesteps // num_envs`; `10000` with `128` envs yields `9984`. | Record actual `env_steps` and checkpoint filename; do not assume target equals actual. |
| Replay buffer scale can become the SAC VRAM bottleneck | memory | OPEN | Route B stores about 671 float32 values per transition; 1M raw replay is about 2.5-2.7 GB before JAX/XLA overhead. | Keep first smoke at `max_replay_size=8192`; increase only after measured GPU smoke. |
| 1M run not validated | validation scope | OPEN | GPU 10k smoke, 4x200 deterministic eval smoke, 50k/100k/250k/500k sanity/eval, and bounded A4 750k bridge gates passed; 1M has not been executed. A4 750k worsened actor drift and eval quality versus A4 500k. | Do not jump straight to 1M; first do a decision review/design pass. |
| 500k metrics require follow-up diagnostics | training dynamics | OPEN_NON_BLOCKING | At 500k, alpha dropped to about `0.0080`; bounded deterministic eval reward worsened from `-4.27974` at 250k to `-4.69107`; critic loss stayed finite/low and Q/target Q decreased to about `3.3`. | Treat 500k as sanity PASS, not a failure. Before any 750k/1M run, diagnose actor mean/action distribution/reward components and keep watching alpha/log-alpha dynamics. |
| Deterministic `tanh(mean)` path degrades while stochastic sampled eval does not | policy diagnostics | OPEN | Both-mode eval across 100k/250k/500k showed deterministic reward mean `-4.2218 -> -4.4585 -> -4.8476`, while stochastic reward mean improved `-6.4616 -> -6.1954 -> -5.9091`. Deterministic action magnitude increased `0.1823 -> 0.2148 -> 0.3029`. | Do not run 1M automatically. Continue actor mean / action distribution / reward-component diagnostics or design review. |
| Actor mean/action magnitude drift explains deterministic eval risk | policy diagnostics | OPEN | Full action diagnostic found deterministic action abs `0.1823 -> 0.2147 -> 0.3029`, policy mean abs `0.1950 -> 0.2288 -> 0.3468`, policy std mean `0.8996 -> 0.8692 -> 0.7519`, and deterministic reward avg `-4.2130 -> -4.4792 -> -4.8204`. | Keep 750k/1M paused. Review actor mean drift by dimension, target entropy / alpha / log_std dynamics, and reward component sensitivity before longer runs. |
| Actor mean drift appears by fresh 100k | policy diagnostics | OPEN | Fresh 100k diagnostic showed final actor mean abs `0.238568` vs interval avg `0.170754`, final deterministic action abs `0.218864` vs interval avg `0.163467`, final log_std mean `-0.157116` vs interval avg `-0.136390`, and alpha about `0.0326`. | Fresh 250k confirmed amplification; A4 extensions now mitigate but do not eliminate drift. |
| Actor mean drift amplifies by fresh 250k | policy diagnostics | OPEN | Fresh 250k diagnostic showed actor mean abs increasing from fresh 100k `0.238568 -> 0.299021`, deterministic action abs `0.218864 -> 0.270944`, final log_std `-0.157116 -> -0.205270`, and alpha `0.032585 -> 0.018768`. | Do not run 1M automatically. Use A4 750k evidence in a decision review before any longer run. |
| A4 alpha/entropy ablation mitigates but does not eliminate drift | policy diagnostics | OPEN | Fresh 100k A1/A3/A4 ablations all passed runtime, checkpoint readiness, and eval gates. Bounded fresh 250k, 500k, and 750k A4 extensions also passed. At 500k, A4 mitigated the old fresh 500k alpha collapse and deterministic drift pattern; at 750k, runtime remained clean but actor drift and eval quality worsened. | Do not jump to 1M. Run a decision review before any longer A4 extension or further diagnostic. |
| A4 750k bridge is runtime stable but not a clean stability improvement | policy diagnostics | OPEN | Bounded fresh 750k A4 bridge passed training, checkpoint readiness, 4x200 eval, and 5-seed eval with no NaN/OOM/fatal CUDA/checkpoint/eval failure. However alpha declined `0.02435 -> 0.01725`, actor mean abs rose `0.29323 -> 0.37190`, deterministic action abs rose `0.26540 -> 0.31850`, deterministic eval worsened `-4.4075 -> -5.7314`, stochastic eval worsened `-6.0434 -> -6.3802`, and critic loss rose `0.0803 -> 0.1441`. | Block any automatic 1M or longer run. Do a decision review/design pass before any further training. |
| Actor regularization R1 is too weak for drift control | policy diagnostics | OPEN | Fresh 100k R1 with A4 alpha settings plus `deterministic_action_l2_coef=0.01` and `actor_mean_l2_coef=0.001` passed train/checkpoint/eval gates, but train actor mean abs worsened versus A4 100k (`0.20693 -> 0.23127`), train deterministic action abs worsened (`0.19314 -> 0.21378`), stochastic 5-seed reward worsened (`-6.4935 -> -6.6210`), and the final regularization contribution was only `0.000886` versus actor loss magnitude `5.7620`. | Do not extend R1 to 250k as-is. Run a decision review or stronger bounded coefficient sweep before any further training. |
| Actor regularization R3 improves 100k drift but has a critic-loss watch item | policy diagnostics | OPEN_NON_BLOCKING | Fresh 100k R2/R3 coefficient sweep passed train/checkpoint/eval gates. R3 had the best 5-seed deterministic and stochastic rewards among R2/R3/A4 100k/R1 and reduced train actor mean abs to `0.1506` and deterministic action abs to `0.1430`, but critic loss was higher than R2 (`0.1684` vs `0.1383`). | Treat R3 as the likely bounded 250k candidate only after decision review. Keep R2 as conservative backup and watch critic loss/Q if extending. |
| A4 250k Q/target_q are higher than earlier baselines | critic diagnostics | OPEN_NON_BLOCKING | Fresh 250k A4 extension finished with `q=8.7442` and `target_q=8.7170`, higher than the fresh 250k actor drift baseline (`q=5.5475`, `target_q=5.5201`) and earlier sanity baselines. Critic loss stayed finite at `0.05245` and no NaN/Inf/checkpoint/eval failure was observed. | Treat as a watch item during any next bounded A4 extension. Do not classify the 250k A4 result as a runtime failure. |
| A4 500k Q/target_q and critic loss remain watch items | critic diagnostics | OPEN_NON_BLOCKING | Bounded fresh 500k A4 extension passed runtime/checkpoint/eval gates with `q=8.47294`, `target_q=8.41017`, and `critic_loss=0.0803`. Q/target_q were slightly lower than A4 250k but much higher than the old fresh 500k baseline (`3.35/3.32`). | Treat as the main watch item before any longer A4 run. Do not classify the 500k A4 result as a failure, but do not auto-run 750k/1M. |
| Deterministic reward degradation is component-specific | reward diagnostics | OPEN | Full action diagnostic links deterministic degradation mainly to `reward/ang_vel_xy` `-50.82 -> -62.66 -> -73.31`, `reward/stand_still` `-18.72 -> -21.72 -> -28.33`, and `reward/orientation` `-34.80 -> -43.27 -> -40.37`; positive `feet_phase` and `tracking_lin_vel` partially offset it. | Diagnose affected components before reward tuning. Do not change reward/action_scale/Kp in the current validation phase. |
| Existing GPU 10k checkpoint is not deterministic-eval ready | checkpoint/eval | OPEN_NON_BLOCKING | `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl` has `normalize_observations=True` but lacks `policy_normalizer` and `value_normalizer`. | Do not use the old checkpoint for trusted deterministic eval; use normalizer-ready 10k, 50k, or 100k checkpoints instead. |
| Sandboxed `uv` may hit `snap-confine` capability restrictions | tooling | OPEN_NON_BLOCKING | The first sandboxed 100k `uv` attempt failed before training started with a `snap-confine` capability error; the identical command then succeeded with external permission and unchanged parameters. | Treat as tooling noise unless it prevents a command from starting; do not classify it as a training failure. |
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
| Future checkpoint schema did not persist observation normalizers | checkpoint schema | Route B checkpoint payloads now save `policy_normalizer` and `value_normalizer`; `scripts/check_sac_checkpoint.py` reports `deterministic_eval_ready` and can enforce it with `--require_eval_ready`; a dry-run schema checkpoint passed the readiness gate. |
| Deterministic eval CLI not available | eval tooling | Added `scripts/eval_sac_checkpoint.py`; 4 env x 200 step eval smoke passed with `EVAL_OK`, no action/reward/obs NaN, and JSON output under ignored `logs`. |
| Route B GPU 50k sanity not validated | validation | User authorized one controlled 50k sanity run; `TRAIN_OK`, checkpoint `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`, `--require_eval_ready` PASS, and no observed NaN/Inf/OOM/CUDA/checkpoint/eval error. |
| Route B 50k bounded deterministic eval not validated | eval validation | `scripts/eval_sac_checkpoint.py` passed with 16 env x 1000 steps; JSON `./logs/sac_eval_50k/eval_16x1000.json`; no action/reward/obs NaN. |
| Route B GPU 100k sanity not validated | validation | User authorized one controlled 100k sanity run; `TRAIN_OK`, checkpoint `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`, `--require_eval_ready` PASS, and no observed NaN/Inf/OOM/fatal CUDA/checkpoint/eval error. |
| Route B 100k bounded deterministic eval not validated | eval validation | `scripts/eval_sac_checkpoint.py` passed with 16 env x 1000 steps; JSON `./logs/sac_eval_100k/eval_16x1000.json`; no action/reward/obs NaN. |
| Route B GPU 250k sanity not validated | validation | User authorized one controlled 250k sanity run; `TRAIN_OK`, checkpoint `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`, `--require_eval_ready` PASS, and no observed NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error. |
| Route B 250k bounded deterministic eval not validated | eval validation | `scripts/eval_sac_checkpoint.py` passed with 16 env x 1000 steps; JSON `./logs/sac_eval_250k/eval_16x1000.json`; no action/reward/obs NaN. |
| Route B GPU 500k sanity not validated | validation | User authorized one controlled 500k sanity run; `TRAIN_OK`, checkpoint `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`, `--require_eval_ready` PASS, and no observed NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error. |
| Route B 500k bounded deterministic eval not validated | eval validation | `scripts/eval_sac_checkpoint.py` passed with 16 env x 1000 steps; JSON `./logs/sac_eval_500k/eval_16x1000.json`; no action/reward/obs NaN. |

## Current Acceptance Position

- Minimum success is met: menagerie is present and flat/rough env API load/reset/step pass.
- Ideal local success is met: CPU tiny smoke passes and writes a checkpoint with finite actor/critic/alpha metrics.
- GPU preflight is validated in WSL2 with JAX backend `gpu` and device `cuda:0`.
- Route B GPU 10k smoke is validated with `TRAIN_OK` and checkpoint
  `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`.
- The existing GPU 10k checkpoint is not a trusted deterministic-eval artifact
  because it predates normalizer persistence.
- Future SAC checkpoints include policy/value observation normalizers for eval
  readiness checks.
- Deterministic eval smoke is validated at 4 env x 200 steps; this is not a
  full benchmark.
- 50k sanity and its 16 env x 1000 bounded deterministic eval are validated.
- 100k sanity and its 16 env x 1000 bounded deterministic eval are validated.
- 250k sanity and its 16 env x 1000 bounded deterministic eval are validated.
- 500k sanity and its 16 env x 1000 bounded deterministic eval are validated.
- Both-mode eval-only diagnostic for 100k/250k/500k is validated. It shows the
  previous reward-degradation conclusion was incomplete: deterministic
  `tanh(mean)` reward degrades, but sampled stochastic reward does not show the
  same degradation.
- Fresh 100k and fresh 250k train-time actor drift diagnostics are validated.
  Fresh 250k shows actor mean and deterministic action magnitude increase in
  absolute level while log_std/std and alpha continue downward.
- Fresh 100k alpha/entropy ablation A1/A3/A4 is validated at the runtime,
  checkpoint readiness, small eval, and multi-seed eval-only gate level. A4 is
  still the best current 100k drift-control candidate, with the caveat that A1
  is slightly better on deterministic reward and A4 is slightly worse on
  stochastic reward.
- Bounded fresh 250k and 500k A4 alpha/entropy extensions are validated.
  A4 500k mitigates the old 500k deterministic drift pattern but leaves
  Q/target_q and critic loss as watch items.
- Fresh 100k actor-regularization R2/R3 coefficient sweep is validated. R3 is
  the best current 100k regularization candidate, but its higher critic loss
  makes a decision review necessary before any bounded 250k extension.
- 1M, full eval benchmark, and PPO comparison are still `NOT VALIDATED`.
- The 500k sanity PASS exposed a watch item: alpha declined to about `0.0080`
  and deterministic reward worsened versus 250k. Both-mode eval suggests the
  next question is actor mean / action distribution behavior, not a longer run.
- Logs, checkpoints, `.venv`, and menagerie assets remain ignored and are not
  committed.
- No LIFT fork wholesale copy was made.
- No PPO/RSL behavior changes were made.
- No world model, fine-tuning, vision, domain randomization, or deployment work was added.
