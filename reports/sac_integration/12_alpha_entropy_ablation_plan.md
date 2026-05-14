# Alpha Entropy Ablation Plan

Status: plan recorded after `4ffd301 Add SAC action diagnostic mapping tools`.

This report records the next controlled diagnostic plan. It is not a training
result. No alpha/entropy ablation training has been executed in this phase.

## Context

- Current branch: `sac-integration`
- Current focus: stabilize deterministic SAC actor behavior before any fresh
  500k, 750k, or 1M run.
- Fresh 100k and fresh 250k actor drift diagnostics passed runtime gates.
- Both diagnostics show actor mean / deterministic action magnitude rising
  while alpha, log_std, and policy std decline.
- Full action diagnostics show deterministic `tanh(mean)` reward degrades,
  while sampled stochastic reward does not show the same degradation.
- No reward, `action_scale`, Kp, PPO, RSL, domain randomization, or
  fine-tuning changes are authorized by this plan.

## Current Alpha And Entropy Mechanics

| Item | Value |
|---|---:|
| action dim | 29 |
| `target_entropy_coef` | 0.5 |
| target entropy formula | `-target_entropy_coef * action_dim` |
| current target entropy | -14.5 |
| `init_log_alpha` | -3.0 |
| initial alpha | about 0.0498 |
| `alpha_learning_rate` | 3e-4 |
| alpha floor / clip | none |
| actor `log_std` clamp | `[-5.0, 2.0]` |

Current actor loss:

```text
mean(alpha * log_prob - min_q)
```

Current alpha loss:

```text
mean(alpha * stop_gradient(-log_prob - target_entropy))
```

With the current negative target entropy, the observed
`alpha_grad_proxy_exp` stayed positive in fresh diagnostics, which explains why
gradient descent keeps pushing `log_alpha` and alpha downward.

## Existing Evidence

| Run | alpha | log_alpha | actor mean abs | deterministic action abs | log_std mean | std mean |
|---|---:|---:|---:|---:|---:|---:|
| fresh 100k final | 0.032585 | -3.423901 | 0.238568 | 0.218864 | -0.157116 | 0.857329 |
| fresh 250k final | 0.018768 | -3.975579 | 0.299021 | 0.270944 | -0.205270 | 0.816216 |

Interpretation:

- actor mean and deterministic action magnitude increase from fresh 100k to
  fresh 250k;
- log_std, std, and alpha decrease over the same interval;
- the drift amplifies in absolute level by fresh 250k;
- this is not a runtime failure, checkpoint failure, or stochastic policy
  collapse.

## Action Mapping Support

`4ffd301 Add SAC action diagnostic mapping tools` added no-training helpers:

- `scripts/inspect_g1_action_mapping.py`
- `scripts/summarize_sac_action_diag.py`

These tools map action dimensions to actuator/joint names and join existing
action diagnostic JSON with the mapping. They do not train or evaluate a
policy. They write optional JSON only under ignored `logs/`.

The current 500k deterministic action diagnostic implicates mainly:

- right ankle roll / pitch;
- waist pitch / roll;
- right knee;
- hip roll;
- one right wrist dimension.

This supports interpreting future ablation results without changing SAC math.

## Fresh 100k Ablation Matrix

Use the existing fresh 100k actor diagnostic as the baseline:

```text
./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl
```

Run only 100k diagnostics first. Do not run fresh 250k, fresh 500k, 750k, or
1M until a 100k variant shows a useful signal.

| ID | `target_entropy_coef` | `alpha_learning_rate` | Purpose |
|---|---:|---:|---|
| A1 | 0.5 | 1e-4 | Test slower alpha decay |
| A3 | 0.25 | 3e-4 | Test lower target entropy magnitude under current formula |
| A4 | 0.25 | 1e-4 | Combined conservative candidate, only if A1 or A3 passes runtime gates |
| A2 optional | 0.5 | 3e-5 | Stronger slow-alpha test if A1 is promising but insufficient |

Do not start with `target_entropy_coef > 0.5`; with the current formula this
would make target entropy more negative and likely increase downward alpha
pressure.

## Commands

Required environment:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

A1:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --alpha_learning_rate 1e-4 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1
```

A3:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --target_entropy_coef 0.25 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1
```

A4, only after A1 or A3 passes runtime gates:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 100000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 100000 \
  --grad_updates_per_step 2 \
  --alpha_learning_rate 1e-4 \
  --target_entropy_coef 0.25 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1
```

Expected checkpoint for each:

```text
<logdir>/sac_lift_step_99968.pkl
```

## Validation Gates

For each variant:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint <logdir>/sac_lift_step_99968.pkl \
  --require_eval_ready

uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint <logdir>/sac_lift_step_99968.pkl \
  --seed 0 \
  --num_eval_envs 4 \
  --episode_length 200 \
  --render False \
  --policy_mode both \
  --action_diagnostics \
  --reward_components \
  --top_k_actions 8 \
  --output_json ./logs/sac_eval_alpha_ablate/<variant>_seed0_4x200_actiondiag.json
```

If a variant improves actor drift at 100k, run 5-seed eval before planning a
fresh 250k extension:

```bash
for seed in 0 1 2 3 4; do
  uv run --no-sync python scripts/eval_sac_checkpoint.py \
    --checkpoint <logdir>/sac_lift_step_99968.pkl \
    --seed "$seed" \
    --num_eval_envs 16 \
    --episode_length 1000 \
    --render False \
    --policy_mode both \
    --action_diagnostics \
    --reward_components \
    --top_k_actions 8 \
    --output_json "./logs/sac_eval_alpha_ablate/<variant>_seed_${seed}_16x1000_actiondiag.json"
done
```

## Success Signals

Compare each ablation against fresh 100k baseline:

- alpha remains higher than `0.032585`, or declines more slowly;
- `actor_policy_mean_abs_mean` is lower than `0.238568`;
- `deterministic_action_abs_mean` is lower than `0.218864`;
- `actor_log_std_mean` is less negative than `-0.157116`;
- `actor_policy_std_mean` stays near or above `0.857329`;
- deterministic small eval reward does not worsen;
- stochastic sampled eval remains healthy;
- q, target_q, and critic loss remain finite and stable;
- deterministic saturation does not increase.

## Stop Conditions

Stop and report immediately if any of these occur:

- JAX backend is not GPU/CUDA;
- `nvidia-smi` is unavailable;
- `TRAIN_OK` is missing;
- checkpoint is missing;
- checkpoint `--require_eval_ready` fails;
- eval status is not `EVAL_OK`;
- `action_nan`, `reward_nan`, or `obs_nan` is true;
- actor loss, critic loss, alpha, log_alpha, q, target_q, log_prob, reward, or
  log_std contains NaN/Inf;
- alpha collapses faster than baseline while actor mean/action magnitude rises;
- q or target_q grows in scale with critic loss degradation;
- logs, checkpoints, `.venv`, or menagerie appear as unignored git changes.

## Execution Approval State

The first A1 training attempt was blocked in the sandbox by the known
`snap-confine` issue. An external execution request was then rejected by the
execution reviewer because it requires explicit user approval for a new 100k
training run after prior read-only instructions.

Therefore this plan is ready, but training is intentionally not executed here.
The next turn must include explicit user approval for:

```text
fresh 100k alpha/entropy ablation training: A1 and A3, then A4 if runtime gates pass
```

This approval requirement is an execution-policy gate, not a SAC technical
blocker.

## Reporting After Execution

After validated ablation results exist, add or update:

- `reports/sac_integration/12_alpha_entropy_ablation_plan.md` or a result
  follow-up section;
- `NEXT_AGENT_HANDOFF.md`;
- `reports/sac_integration/README.md`;
- `reports/sac_integration/04_smoke_results.md`;
- `reports/sac_integration/05_known_issues.md`;
- `reports/sac_integration/06_next_actions.md`;
- `reports/sac_integration/08_project_status_roadmap.md`;
- `reports/sac_integration/09_phase_summary_and_risks.md`;
- `reports/sac_integration/11_actor_drift_train_diagnostic.md`.

Use the SSH push workflow from the workspace `AGENTS.md`; do not retry the
known-failing HTTPS push path first.
