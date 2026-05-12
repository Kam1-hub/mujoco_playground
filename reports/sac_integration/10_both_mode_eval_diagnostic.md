# Both-Mode Eval Diagnostic

## Context

- HEAD under test: `926a14f Add SAC alpha entropy diagnostics`.
- Scope: eval-only diagnostic.
- No training was run.
- No code or report changes were made before this report update.
- Compared checkpoints: 100k, 250k, and 500k.
- Each checkpoint was evaluated with `scripts/eval_sac_checkpoint.py --policy_mode both`.
- Seeds: `0`, `1`, `2`, `3`, `4`.
- Eval scale: `num_eval_envs=16`, `episode_length=1000`.

## Checkpoint Readiness

| Scale | Checkpoint | Readiness |
|---|---|---:|
| 100k | `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` | PASS |
| 250k | `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` | PASS |
| 500k | `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` | PASS |

## Deterministic Aggregate

| Scale | Reward Avg | Reward Stdev | Min Avg | Max Avg | Action Abs Avg | Sat 0.95 Avg | All OK |
|---|---:|---:|---:|---:|---:|---:|---|
| 100k | -4.2218 | 0.3790 | -7.7808 | -3.0968 | 0.1823 | 0.0000 | true |
| 250k | -4.4585 | 0.4026 | -8.1472 | -3.3968 | 0.2148 | 0.0000 | true |
| 500k | -4.8476 | 0.2879 | -10.2433 | -3.2822 | 0.3029 | 0.0011 | true |

## Stochastic Aggregate

| Scale | Reward Avg | Reward Stdev | Min Avg | Max Avg | Log Prob Avg | Action Abs Avg | Sat 0.95 Avg | All OK |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 100k | -6.4616 | 0.5185 | -10.1315 | -4.9407 | -17.9594 | 0.5273 | 0.0445 | true |
| 250k | -6.1954 | 0.4392 | -10.0901 | -4.7913 | -17.2847 | 0.5236 | 0.0415 | true |
| 500k | -5.9091 | 0.5319 | -10.1368 | -4.4472 | -13.4275 | 0.5222 | 0.0414 | true |

## Interpretation

- Deterministic reward degrades monotonically:
  - 100k: `-4.2218`
  - 250k: `-4.4585`
  - 500k: `-4.8476`
- Deterministic action magnitude increases: `0.1823 -> 0.2148 -> 0.3029`.
- Deterministic saturation remains low but becomes nonzero at 500k.
- Stochastic reward does not degrade:
  - 100k: `-6.4616`
  - 250k: `-6.1954`
  - 500k: `-5.9091`
- Stochastic policy is still lower reward in absolute terms than deterministic at each checkpoint.
- Stochastic reward improves slightly over training.
- Stochastic log-prob becomes much less negative: `-17.9594 -> -17.2847 -> -13.4275`.
- This is consistent with decreasing entropy / lower stochasticity.
- The earlier conclusion "policy quality simply degrades" was incomplete.
- Updated conclusion: deterministic `tanh(mean)` behavior degrades, while sampled stochastic behavior does not show the same degradation.

## Risk And Implication

- This does not authorize 750k or 1M.
- The next diagnostic target is actor mean / action distribution / reward components.
- Deterministic eval may be the relevant deployment path, so deterministic degradation still matters.
- Stochastic sampled eval being better over time suggests the distribution still contains useful behavior.
- Need to inspect why the `tanh(mean)` path drifts toward worse reward.

## Warnings

- No traceback.
- No OOM.
- No CUDA fatal error.
- No eval failure.
- No action/reward/obs NaN.
- Known non-fatal WSL2 CUDA driver version warning observed.
- Known non-fatal JAX cast overflow warning observed.
- Eval commands were rerun externally when the sandbox hit a `snap-confine` issue; parameters were unchanged.

## Next Action

- Do not run 750k or 1M.
- Design actor mean / action distribution / reward-component diagnostic.
- Possible future diagnostic:
  - inspect actor mean and std/log_std statistics
  - compare deterministic action against stochastic sampled action
  - add reward component eval if env metrics expose reward terms
  - inspect whether deterministic mean action magnitude drift explains reward loss
  - inspect action saturation and per-dimension action means/stds
