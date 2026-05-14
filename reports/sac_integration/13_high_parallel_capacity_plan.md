# High-Parallel SAC Capacity Plan

Execution update: this plan has now been run. See
`reports/sac_integration/14_high_parallel_capacity_results.md` for detailed
CHECKPOINT AI results. All 512/1024/2048 cases returned `TRAIN_OK` and
readiness PASS; 1024 envs was fastest and is the next bounded 1M candidate.

## Context

- User flagged that `num_envs=128` is likely too conservative for G1 SAC.
- G1 has a much larger action/control space than the early sanity ladder can
  evaluate for policy quality.
- Prior 10k, 50k, 100k, 250k, and 500k runs should be treated as runtime,
  checkpoint, eval, and diagnostic evidence, not final policy-quality evidence.
- The next useful step at planning time was a high-parallel capacity benchmark,
  not another 128-env quality run and not a direct 10M run.

## Current Implementation Relationship

The current Route B training loop uses this relationship:

```text
actor_steps = num_timesteps // num_envs
actual_env_steps = actor_steps * num_envs
```

Each actor step:

1. steps `num_envs` parallel environments once;
2. inserts `num_envs` transitions into replay;
3. after `min_replay_size`, runs `grad_updates_per_step` optimizer updates.

The approximate sampled update-to-data ratio is:

```text
sample UTD ~= grad_updates_per_step * batch_size / num_envs
```

The historical 128-env baseline used `batch_size=256` and
`grad_updates_per_step=2`, giving:

```text
2 * 256 / 128 = 4 sampled training items per new transition
```

If `num_envs` rises from `128` to `1024` while `grad_updates_per_step` remains
`2`, the sampled update-to-data ratio drops by 8x. High-parallel tests must
therefore coordinate `num_envs`, `batch_size`, and `grad_updates_per_step`
instead of changing env count alone.

## UTD-Preserving Capacity Settings

With `batch_size=256`, keeping sample UTD near the 128-env baseline of `4`
requires:

| num_envs | grad_updates_per_step | Approx sample UTD |
|---:|---:|---:|
| 512 | 8 | 4 |
| 1024 | 16 | 4 |
| 2048 | 32 | 4 |

## Replay Memory Estimate

Route B replay stores roughly 671 float32-like values per transition, or about
`2684` raw bytes per transition.

| Replay size | Raw decimal | Raw binary |
|---:|---:|---:|
| 250k | 671 MB | 640 MiB |
| 500k | 1.34 GB | 1.25 GiB |
| 1M | 2.68 GB | 2.50 GiB |
| 2M | 5.37 GB | 5.00 GiB |
| 5M | 13.42 GB | 12.50 GiB |
| 10M | 26.84 GB | 25.00 GiB |

On a 12GB RTX 4070 SUPER, `5M` and `10M` replay caps are unrealistic for this
implementation. Raw replay is only one part of the memory footprint; XLA temp
buffers, vectorized env state, optimizer state, network params, normalizers,
sampled batches, and checkpoint serialization overhead also matter.

Long runs should decouple `num_timesteps` from `max_replay_size`. Multi-million
training should start with a replay cap around `1M`, and consider `2M` only
after explicit replay stress evidence.

## Immediate Capacity Benchmark Plan

Use R3 actor-regularization settings because R3 retained drift control at 250k:

```text
target_entropy_coef = 0.25
alpha_learning_rate = 1e-4
deterministic_action_l2_coef = 0.5
actor_mean_l2_coef = 0.05
num_timesteps = 65536
batch_size = 256
min_replay_size = 16384
max_replay_size = 262144
```

Expected checkpoint name for each run:

```text
sac_lift_step_65536.pkl
```

Capacity ladder:

| Case | num_envs | grad_updates_per_step | logdir |
|---|---:|---:|---|
| env512 | 512 | 8 | `./logs/sac_capacity_env512_65k_r3_b256_g8_replay262k` |
| env1024 | 1024 | 16 | `./logs/sac_capacity_env1024_65k_r3_b256_g16_replay262k` |
| env2048 | 2048 | 32 | `./logs/sac_capacity_env2048_65k_r3_b256_g32_replay262k` |

Approximate gradient update counts after warmup:

| Case | Actual env steps | Approx gradient steps |
|---|---:|---:|
| env512 | 65536 | 776 |
| env1024 | 65536 | 784 |
| env2048 | 65536 | 800 |

## Metrics To Compare

For each capacity run, record:

- `TRAIN_OK`
- wall time and SPS
- post-run or sampled GPU memory from `nvidia-smi`
- `env_steps` and `gradient_steps`
- actor and critic losses
- alpha and log_alpha
- q and target_q
- reward_mean, done_fraction, and discount_mean
- actor mean abs, deterministic action abs, log_std, and std
- sampled action abs and saturation
- regularization loss, deterministic action L2, and actor mean L2
- checkpoint existence and `--require_eval_ready` result

## Stop Conditions

Stop the capacity ladder if any of these occur:

- JAX backend is not GPU/CUDA.
- `nvidia-smi` is unavailable.
- OOM or fatal CUDA/XLA compile/runtime error occurs.
- `TRAIN_OK` is missing.
- checkpoint is missing or readiness fails.
- NaN/Inf appears in actor loss, critic loss, alpha, q, target_q, reward,
  log_std, or actor drift metrics.
- replay insert, sample, or update fails.
- critic loss or Q scale materially worsens with reward collapse.
- regularization loss dominates actor loss magnitude.
- 1024 or 2048 gives poor SPS relative to lower env counts or shows unstable
  memory pressure.
- `logs`, checkpoints, `.venv`, or menagerie appear as unignored git changes.

## Longer-Run Ladder After Capacity

After capacity testing:

- Prefer `1024` envs as the default only if it is stable and materially better
  than 512.
- Use `2048` only if it is clearly faster and not memory-fragile.
- Keep sample UTD near the historical baseline of about `4` unless a separate
  lower-UTD ablation is planned.
- If update overhead dominates, test `batch_size=512` with about half the
  update count as a second pass.

Candidate long-run ladder after capacity:

| Scale | Suggested num_envs | Actual env steps | Replay cap |
|---|---:|---:|---:|
| 1M | 1024 | 999424 | 1M |
| 3M | 1024 | 2999296 | 1M |
| 10M | 1024 | 9999360 | 1M first |

Do not use `max_replay_size=num_timesteps` for huge runs by default.

## Current Recommendation

The 512, 1024, and 2048 env capacity benchmark has been completed. `1024` envs
is the current best next bounded 1M candidate because it was stable and fastest
at `953.36` SPS. `2048` envs is feasible but slower in this benchmark. Do not
jump directly to 3M or 10M; first run a bounded 1024-env 1M validation with R3
settings, `grad_updates_per_step=16`, and a replay cap decoupled from future
multi-million runs.
