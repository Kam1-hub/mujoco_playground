# Phase Summary and Risks

Date: 2026-05-14

This report freezes the current SAC Route B validation state so a new agent can
continue without relying on chat history.

## Repository State

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Current actor-regularization diagnostic code baseline before this report
  update: `208eef2 Add SAC actor regularization diagnostics`
- Remote: `origin https://github.com/Kam1-hub/mujoco_playground.git`
- External menagerie commit: `1b86ece576591213e2b666ebf59508454200ca97`

Runtime artifacts are local and ignored. Do not commit `logs/`, `.venv/`,
`g1_env/external_deps/mujoco_menagerie`, or generated checkpoints.

## Validation Ladder

| Level | Status | Evidence |
| --- | --- | --- |
| CPU tiny smoke | PASS | Earlier local smoke report |
| WSL2 CUDA/JAX preflight | PASS | `scripts/gpu_preflight.py --impl jax --require_gpu` |
| Route B GPU 10k smoke | PASS | `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl` |
| Future checkpoint schema with normalizers | PASS | `c59eda0 Save SAC normalizers in checkpoints` |
| Normalizer-ready GPU 10k checkpoint | PASS | `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` |
| Deterministic eval smoke | PASS | `./logs/sac_eval_smoke/eval_4x200.json` |
| Route B GPU 50k sanity | PASS | `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` |
| 50k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 50k bounded deterministic eval | PASS | `./logs/sac_eval_50k/eval_16x1000.json` |
| 100k sanity | PASS | `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` |
| 100k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 100k bounded deterministic eval | PASS | `./logs/sac_eval_100k/eval_16x1000.json` |
| 250k sanity | PASS | `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` |
| 250k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 250k bounded deterministic eval | PASS | `./logs/sac_eval_250k/eval_16x1000.json` |
| 500k sanity | PASS | `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` |
| 500k checkpoint eval readiness | PASS | `scripts/check_sac_checkpoint.py --require_eval_ready` |
| 500k bounded deterministic eval | PASS | `./logs/sac_eval_500k/eval_16x1000.json` |
| Both-mode eval diagnostic | PASS | `reports/sac_integration/10_both_mode_eval_diagnostic.md` |
| Full action distribution diagnostic | PASS | `reports/sac_integration/10_action_distribution_diagnostics.md` |
| Fresh 100k actor drift train diagnostic | PASS | `reports/sac_integration/11_actor_drift_train_diagnostic.md` |
| Fresh 250k actor drift train diagnostic | PASS | `reports/sac_integration/11_actor_drift_train_diagnostic.md` |
| Fresh 100k alpha/entropy ablation A1/A3/A4 | PASS | `reports/sac_integration/12_alpha_entropy_ablation_plan.md` |
| Fresh 100k alpha/entropy ablation multi-seed eval-only | PASS | `./logs/sac_eval_alpha_ablate_multiseed/`, `15` JSON outputs |
| Bounded fresh 250k A4 alpha/entropy extension | PASS | `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl` |
| Bounded fresh 500k A4 alpha/entropy extension | PASS | `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl` |
| Bounded fresh 750k A4 alpha/entropy bridge | PASS_RUNTIME_UNCLEAN_TREND | `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl` |
| Fresh 100k actor-regularization R1 | PASS_RUNTIME_WEAK_EFFECT | `./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl` |
| 1M training | NOT VALIDATED | Requires explicit user confirmation and resource/stop plan |

## Completed Outcomes

- Added a local SAC Route B implementation without modifying PPO/RSL code paths.
- Validated WSL2 CUDA/JAX operation on an RTX 4070 SUPER 12GB setup.
- Pinned the external MuJoCo Menagerie checkout and kept it untracked.
- Verified SAC training can produce checkpoints on GPU at 10k, 50k, 100k,
  250k, and 500k scales.
- Fixed future SAC checkpoint schema to save `policy_normalizer` and
  `value_normalizer`.
- Added checkpoint readiness checks for deterministic eval.
- Added a bounded deterministic SAC checkpoint eval script.
- Verified deterministic eval readiness and bounded eval on the normalizer-ready
  10k and 50k checkpoints.
- Verified 100k checkpoint readiness and bounded deterministic eval.
- Verified 250k checkpoint readiness and bounded deterministic eval.
- Verified 500k checkpoint readiness and bounded deterministic eval.
- Synced alpha/entropy diagnostics and validated `--policy_mode both` eval-only
  diagnostic on 100k, 250k, and 500k checkpoints.
- Validated full action distribution / reward-component eval diagnostic on
  100k, 250k, and 500k checkpoints. All 15 JSON outputs are present under
  ignored `./logs/sac_eval_action_diag_full/`; all evals returned `EVAL_OK` and
  all action/reward/obs NaN flags were false.
- Committed train-time actor drift instrumentation and validated a fresh 100k
  diagnostic run. The run produced checkpoint
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`, passed
  checkpoint readiness, and passed a 4 env x 200 action diagnostic eval.
- Validated a fresh 250k actor drift diagnostic run. The run produced
  checkpoint `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`,
  passed checkpoint readiness, and passed a 4 env x 200 action diagnostic eval.
- Added no-training action mapping tools.
- Validated fresh 100k alpha/entropy ablation A1/A3/A4. All three variants
  returned `TRAIN_OK`, passed checkpoint readiness, and passed small both-mode
  action diagnostic eval. A4 is the best current 100k drift candidate.
- Validated multi-seed eval-only follow-up for A1/A3/A4. All `15` JSON outputs
  under ignored `./logs/sac_eval_alpha_ablate_multiseed/` returned `EVAL_OK`
  with action/reward/obs NaN flags false. A4 remained best on deterministic
  action magnitude and actor mean drift metrics, while A1 slightly edged
  deterministic reward and A4 retained a stochastic reward caveat.
- Validated a bounded fresh 250k A4 alpha/entropy extension. The run produced
  checkpoint
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`,
  passed checkpoint readiness, passed 4 env x 200 action diagnostic eval, and
  passed 5-seed 16 env x 1000 eval. A4 mitigated fresh 250k drift versus the
  baseline but did not eliminate drift relative to A4 100k.
- Validated a bounded fresh 500k A4 alpha/entropy extension. The run produced
  checkpoint
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`,
  passed checkpoint readiness, passed 4 env x 200 action diagnostic eval, and
  passed 5-seed 16 env x 1000 eval. A4 mitigated the old fresh 500k
  deterministic drift pattern but did not eliminate A4's own 250k-to-500k
  drift.
- Validated a bounded fresh 750k A4 alpha/entropy bridge. The run produced
  checkpoint
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`,
  passed checkpoint readiness, passed 4 env x 200 action diagnostic eval, and
  passed 5-seed 16 env x 1000 eval. Runtime remained stable, but actor drift,
  deterministic eval, stochastic eval, and critic loss worsened versus A4
  500k, so the result blocks any automatic 1M or longer run.
- Validated fresh 100k actor-regularization R1 with A4 alpha settings,
  `deterministic_action_l2_coef=0.01`, and `actor_mean_l2_coef=0.001`. The run
  produced checkpoint
  `./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl`,
  passed checkpoint readiness, passed 4 env x 200 action diagnostic eval, and
  passed 5-seed 16 env x 1000 eval. R1 was runtime-clean but too weak to
  reduce train-time actor mean / deterministic action drift versus A4 100k.

### Full Action Diagnostic Summary

Deterministic aggregate:

| Scale | Reward Avg | Action Abs | Mean Abs | LogStd Mean | Std Mean | Det Sat |
|---|---:|---:|---:|---:|---:|---:|
| 100k | -4.2130 | 0.1823 | 0.1950 | -0.1077 | 0.8996 | 0.0000004 |
| 250k | -4.4792 | 0.2147 | 0.2288 | -0.1415 | 0.8692 | 0.0000 |
| 500k | -4.8204 | 0.3029 | 0.3468 | -0.2909 | 0.7519 | 0.00113 |

Stochastic aggregate:

| Scale | Reward Avg | Action Abs | Mean Abs | LogStd Mean | Std Mean | Sto Sat |
|---|---:|---:|---:|---:|---:|---:|
| 100k | -6.4741 | 0.5274 | 0.2240 | -0.1562 | 0.8578 | 0.0445 |
| 250k | -6.2374 | 0.5236 | 0.2712 | -0.1939 | 0.8254 | 0.0415 |
| 500k | -5.8911 | 0.5221 | 0.3954 | -0.3288 | 0.7256 | 0.0413 |

Interpretation:

- This is not a runtime failure.
- This is not stochastic policy collapse.
- The key risk is deterministic deployment/eval degradation driven by actor
  mean/action magnitude drift.
- Entropy/alpha dynamics remain likely upstream because alpha/log_std/std
  decrease while policy mean/action magnitude increase.
- Deterministic degradation aligns mainly with `reward/ang_vel_xy`,
  `reward/stand_still`, and `reward/orientation`.
- 1M should remain paused until actor mean drift and the A4 750k degradation
  are understood.

### Fresh 100k Actor Drift Diagnostic Summary

Training:

- Checkpoint:
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS
- `env_steps`: `99968`
- `gradient_steps`: `1548`
- `wall_time`: `68.07941276300699`
- `sps`: `1468.4027952473857`
- `alpha`: `0.03258506953716278`
- `log_alpha`: `-3.423901081085205`
- `q`: `4.1935601234436035`
- `target_q`: `4.180259704589844`

Actor drift evidence:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.23856812715530396 | 0.1707537253543696 |
| actor policy mean abs max | 1.7372020483016968 | 1.1534156603329557 |
| actor log_std mean | -0.15711648762226105 | -0.13639042302196033 |
| actor policy std mean | 0.8573285341262817 | 0.8771794435281778 |
| deterministic action abs mean | 0.218863844871521 | 0.16346676852698475 |
| sampled action abs mean | 0.5348999500274658 | 0.5254770112669129 |

Small action diagnostic eval:

- JSON:
  `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`
- Status: `EVAL_OK`
- Deterministic reward mean: `-4.315369606018066`
- Stochastic reward mean: `-6.333320140838623`
- NaN flags: `action_nan=false`, `reward_nan=false`, `obs_nan=false`

Interpretation: fresh 100k already shows actor mean drift forming. Final actor
mean magnitude and deterministic action magnitude are higher than interval
averages, while final log_std/std are lower. This supports the early-drift
hypothesis and is not a runtime failure.

### Fresh 250k Actor Drift Diagnostic Summary

Training:

- Checkpoint:
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `140.85426465800265`
- `sps`: `1774.7705446261555`
- `alpha`: `0.01876842975616455`
- `log_alpha`: `-3.975579023361206`
- `q`: `5.547477722167969`
- `target_q`: `5.520053863525391`

Actor drift evidence:

| Metric | Final | Interval Avg |
|---|---:|---:|
| actor policy mean abs mean | 0.2990209758281708 | 0.233315885204818 |
| actor policy mean abs max | 1.9897916316986084 | 1.739523811275398 |
| actor log_std mean | -0.20526975393295288 | -0.16304866696410225 |
| actor policy std mean | 0.8162157535552979 | 0.8529925538726603 |
| deterministic action abs mean | 0.2709442377090454 | 0.21555377979248855 |
| sampled action abs mean | 0.535484254360199 | 0.5281302615586679 |

Fresh 100k comparison:

- Actor mean abs final: `0.238568 -> 0.299021`.
- Deterministic action abs final: `0.218864 -> 0.270944`.
- Final log_std: `-0.157116 -> -0.205270`.
- Final std: `0.857329 -> 0.816216`.
- Alpha: `0.032585 -> 0.018768`.

Small action diagnostic eval:

- JSON:
  `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`
- Status: `EVAL_OK`
- Deterministic reward mean: `-4.01785135269165`
- Stochastic reward mean: `-5.697283744812012`
- NaN flags: `action_nan=false`, `reward_nan=false`, `obs_nan=false`

Interpretation: drift amplifies in absolute level by fresh 250k. This is not a
runtime failure.

### Fresh 100k Alpha/Entropy Ablation Summary

All three ablation variants passed runtime, checkpoint readiness, and small
eval gates:

| Variant | Parameters | alpha | actor mean abs | det action abs | log_std mean | std mean | det reward mean |
|---|---|---:|---:|---:|---:|---:|---:|
| A1 | `alpha_lr=1e-4`, `target_entropy_coef=0.5` | 0.04285280779004097 | 0.21597573161125183 | 0.20185625553131104 | -0.15477555990219116 | 0.8587937355041504 | -4.416665077209473 |
| A3 | `alpha_lr=3e-4`, `target_entropy_coef=0.25` | 0.032598935067653656 | 0.24479639530181885 | 0.22468040883541107 | -0.15980812907218933 | 0.8549157381057739 | -4.555072784423828 |
| A4 | `alpha_lr=1e-4`, `target_entropy_coef=0.25` | 0.04284820705652237 | 0.2069278359413147 | 0.19313891232013702 | -0.15182653069496155 | 0.8613420128822327 | -4.366635799407959 |

Fresh 100k baseline reference: alpha `0.032585`, actor mean abs `0.238568`,
deterministic action abs `0.218864`, log_std mean `-0.157116`, std mean
`0.857329`.

Interpretation:

- A1 improves the main drift metrics versus the fresh 100k baseline.
- A3 does not improve drift metrics.
- A4 is the best current 100k candidate: higher alpha, lower actor mean and
  deterministic action magnitude, healthier log_std/std, and best deterministic
  4x200 reward among A1/A3/A4.
- A4 stochastic 4x200 reward was worse than A1/A3, so this remains a diagnostic
  signal, not a final policy-quality benchmark.

### Fresh 100k Alpha/Entropy Multi-Seed Eval Summary

Eval-only setup: A1/A3/A4 checkpoints, seeds `0..4`, `num_eval_envs=16`,
`episode_length=1000`, `--policy_mode both`, `--action_diagnostics`, and
`--reward_components`. JSON directory:
`./logs/sac_eval_alpha_ablate_multiseed/`; JSON count: `15`.

Deterministic aggregate:

| Variant | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | -4.3262 | 0.4001 | 0.1826 | 0.1961 | -0.1065 | 0.9006 | true |
| A3 | -4.5518 | 0.4394 | 0.1906 | 0.2076 | -0.1107 | 0.8972 | true |
| A4 | -4.3436 | 0.3791 | 0.1774 | 0.1920 | -0.1058 | 0.9015 | true |

Stochastic aggregate:

| Variant | Reward Avg | Reward SD | Action Abs | Mean Abs | LogStd Mean | Std Mean | OK |
|---|---:|---:|---:|---:|---:|---:|---|
| A1 | -6.3990 | 0.3916 | 0.5259 | 0.2054 | -0.1534 | 0.8599 | true |
| A3 | -6.4888 | 0.4957 | 0.5285 | 0.2291 | -0.1599 | 0.8548 | true |
| A4 | -6.4935 | 0.5123 | 0.5266 | 0.2067 | -0.1540 | 0.8595 | true |

Interpretation: A4 remains best on deterministic action magnitude and actor
mean drift metrics. It is not strictly best on deterministic reward: A1 is
slightly better (`-4.3262` vs `-4.3436`), and the gap is small relative to seed
variance. A4 stochastic reward is worse than A1 by about `0.0945` and
essentially tied with A3.

### Bounded Fresh 250k A4 Extension Summary

The bounded fresh 250k A4 extension passed runtime, checkpoint, and eval gates:

- Checkpoint:
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS, `deterministic_eval_ready=true`, normalizers
  present.
- 4 env x 200 seed 0 eval: PASS,
  `./logs/sac_eval_alpha_ablate_250k/eval_A4_seed0_4x200_actiondiag.json`.
- 5-seed 16 env x 1000 eval: PASS, five JSON outputs in
  `./logs/sac_eval_alpha_ablate_250k_multiseed/`.

Training and drift metrics:

| Metric | Value |
|---|---:|
| env_steps | 249984 |
| gradient_steps | 3892 |
| wall_time | 148.8921 |
| sps | 1678.9605 |
| alpha | 0.03455 |
| log_alpha | -3.36536 |
| actor mean abs | 0.22572 |
| deterministic action abs | 0.21220 |
| log_std mean | -0.17093 |
| std mean | 0.84424 |
| q | 8.7442 |
| target_q | 8.7170 |

Eval aggregate:

| Mode | Reward Avg | Reward SD | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.1561 | 0.4168 | 0.1861 | 0.000006 | 0.1960 | -0.1183 | 0.8894 |
| stochastic | -6.1400 | 0.4045 | 0.5213 | 0.04050 | 0.2254 | -0.1733 | 0.8422 |

Interpretation:

- Versus fresh 250k baseline, A4 improves alpha by `+0.01578`, actor mean abs
  by `-0.07330`, deterministic action abs by `-0.05875`, log_std by
  `+0.03434`, and std by `+0.02802`.
- Versus A4 100k, drift still increases moderately: alpha
  `0.04285 -> 0.03455`, actor mean abs `0.20693 -> 0.22572`,
  deterministic action abs `0.19314 -> 0.21220`, and log_std
  `-0.15183 -> -0.17093`.
- Q/target_q are higher than earlier baselines and should be watched.
- This is promising drift-control evidence, and it justified a bounded A4 500k
  decision review that has now completed.

### Bounded Fresh 500k A4 Extension Summary

The bounded fresh 500k A4 extension passed runtime, checkpoint, and eval gates:

- Checkpoint:
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS, `deterministic_eval_ready=true`, normalizers
  present.
- 4 env x 200 seed 0 eval: PASS,
  `./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json`.
- 5-seed 16 env x 1000 eval: PASS, five JSON outputs in
  `./logs/sac_eval_alpha_ablate_500k_multiseed/`.

Training and drift metrics:

| Metric | Value |
|---|---:|
| env_steps | 499968 |
| gradient_steps | 7798 |
| wall_time | 271.9703 |
| sps | 1838.3185 |
| alpha | 0.02435 |
| log_alpha | -3.71524 |
| actor mean abs | 0.29323 |
| deterministic action abs | 0.26540 |
| log_std mean | -0.22279 |
| std mean | 0.80245 |
| q | 8.47294 |
| target_q | 8.41017 |
| critic_loss | 0.0803 |

Eval aggregate:

| Mode | Reward Avg | Reward SD | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -4.4075 | 0.3493 | 0.2289 | 0.0000004 | 0.2477 | -0.1909 | 0.8277 |
| stochastic | -6.0434 | 0.5364 | 0.5152 | 0.0366 | 0.2704 | -0.2277 | 0.7984 |

Interpretation:

- Versus the old fresh 500k baseline, A4 improves alpha (`0.02435` vs
  `0.00801`), actor mean / action magnitude, log_std/std, and deterministic
  eval reward.
- Versus A4 250k, drift continues: alpha `0.03455 -> 0.02435`, actor mean abs
  `0.22572 -> 0.29323`, deterministic action abs `0.21220 -> 0.26540`, and
  std `0.84424 -> 0.80245`.
- Q/target_q are slightly lower than A4 250k but much higher than the old fresh
  500k baseline. Critic loss rose to `0.0803`; this is finite and not a failure,
  but it is the main watch item.

### Bounded Fresh 750k A4 Bridge Summary

The bounded fresh 750k A4 bridge passed runtime, checkpoint, and eval gates:

- Checkpoint:
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`
- Checkpoint readiness: PASS, `deterministic_eval_ready=true`, normalizers
  present.
- 4 env x 200 seed 0 eval: PASS,
  `./logs/sac_eval_alpha_ablate_750k/eval_A4_seed0_4x200_actiondiag.json`.
- 5-seed 16 env x 1000 eval: PASS, five JSON outputs in
  `./logs/sac_eval_alpha_ablate_750k_multiseed/`.
- All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.

Training and drift metrics:

| Metric | Value |
|---|---:|
| env_steps | 749952 |
| gradient_steps | 11704 |
| wall_time | 397.2631 |
| sps | 1887.7966 |
| alpha | 0.017246 |
| log_alpha | -4.06016 |
| actor mean abs | 0.37190 |
| deterministic action abs | 0.31850 |
| log_std mean | -0.24975 |
| std mean | 0.78393 |
| q | 6.1746 |
| target_q | 6.2776 |
| critic_loss | 0.1441 |

Eval aggregate:

| Mode | Reward Avg | Reward SD | Action Abs | Sat 0.95 | Policy Mean Abs | LogStd Mean | Std Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| deterministic | -5.7314 | 0.3095 | 0.3027 | 0.00201 | 0.3493 | -0.2376 | 0.7924 |
| stochastic | -6.3802 | 0.4577 | 0.5274 | 0.04521 | 0.3551 | -0.2611 | 0.7751 |

Interpretation:

- A4 750k is runtime stable: training, checkpoint readiness, 4x200 eval, and
  5-seed eval all passed.
- It is not a clean stability improvement. Versus A4 500k, alpha declined
  `0.02435 -> 0.01725`, actor mean abs rose `0.29323 -> 0.37190`,
  deterministic action abs rose `0.26540 -> 0.31850`, log_std narrowed
  `-0.22279 -> -0.24975`, and std fell `0.80245 -> 0.78393`.
- Q/target_q moved down versus A4 500k (`8.47/8.41 -> 6.17/6.28`), but
  critic loss worsened `0.0803 -> 0.1441`.
- Deterministic 5-seed eval worsened sharply
  `-4.4075 -> -5.7314`; stochastic 5-seed eval also worsened
  `-6.0434 -> -6.3802`.
- This result blocks any automatic 1M or longer run.

## Key Metrics

### GPU 10k Smoke

- Command family: Route B SAC, `num_timesteps=10000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=8192`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- `env_steps`: `9984`
- `gradient_steps`: `142`
- `wall_time`: `56.50599093900382`
- `sps`: `176.68922947970893`
- `actor_loss`: `-1.9058758020401`
- `critic_loss`: `0.08382290601730347`
- `alpha`: `0.04770537465810776`
- `alpha_loss`: `1.585930585861206`
- `policy_log_prob`: `-18.79882049560547`
- `policy_q`: `1.0090709924697876`
- `q`: `1.0673823356628418`
- `target_q`: `1.0646085739135742`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/CUDA error observed: no

Note: the first 10k checkpoint predates the normalizer schema fix and is not a
trusted deterministic eval input.

### 10k Deterministic Eval Smoke

- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- Eval JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- `num_eval_envs`: `4`
- `episode_length`: `200`
- `eval_env_steps`: `800`
- `episode_reward_mean`: `-3.3628087043762207`
- `episode_reward_std`: `0.34718504548072815`
- `episode_reward_min`: `-3.778578281402588`
- `episode_reward_max`: `-2.876215934753418`
- `done_fraction`: `1.0`
- `wall_time`: `62.32472045900067`
- `sps`: `12.835998205981001`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 50k Sanity

- Command family: Route B SAC, `num_timesteps=50000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=50000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- `env_steps`: `49920`
- `gradient_steps`: `766`
- `wall_time`: `35.99766752999858`
- `sps`: `1386.756515777009`
- `actor_loss`: `-3.6135072708129883`
- `critic_loss`: `0.07097882032394409`
- `alpha`: `0.03992176800966263`
- `alpha_loss`: `1.327394962310791`
- `policy_log_prob`: `-18.81831169128418`
- `policy_q`: `2.8622469902038574`
- `q`: `2.884032726287842`
- `target_q`: `2.899707317352295`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error observed: no

The actual step count is `49920` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 50k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Eval JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-3.5016322135925293`
- `episode_reward_std`: `0.75983726978302`
- `episode_reward_min`: `-6.096090316772461`
- `episode_reward_max`: `-2.531925916671753`
- `done_fraction`: `1.0`
- `wall_time`: `74.76505397899746`
- `sps`: `214.00372431342888`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 100k Sanity

- Command family: Route B SAC, `num_timesteps=100000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=100000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Checkpoint readiness: PASS
- `env_steps`: `99968`
- `gradient_steps`: `1548`
- `wall_time`: `56.80434615799459`
- `sps`: `1759.8653406193746`
- `actor_loss`: `-4.994826316833496`
- `critic_loss`: `0.04054964333772659`
- `alpha`: `0.03259027376770973`
- `alpha_loss`: `1.0562278032302856`
- `policy_log_prob`: `-17.77903938293457`
- `q`: `4.3698601722717285`
- `target_q`: `4.456111907958984`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/checkpoint/eval error observed: no

The actual step count is `99968` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 100k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Eval JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-3.894726037979126`
- `episode_reward_std`: `0.9610732197761536`
- `episode_reward_min`: `-7.2897186279296875`
- `episode_reward_max`: `-2.889821767807007`
- `done_fraction`: `1.0`
- `wall_time`: `66.85148939098872`
- `sps`: `239.33647770242092`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 250k Sanity

- Command family: Route B SAC, `num_timesteps=250000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=250000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Checkpoint readiness: PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `120.21957968600327`
- `sps`: `2079.3950590488107`
- `actor_loss`: `-5.524118900299072`
- `critic_loss`: `0.02644157037138939`
- `alpha`: `0.018743595108389854`
- `alpha_loss`: `0.5869507789611816`
- `policy_log_prob`: `-16.657032012939453`
- `q`: `5.197851181030273`
- `target_q`: `5.205532073974609`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error observed: no

The actual step count is `249984` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 250k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Eval JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-4.279743194580078`
- `episode_reward_std`: `1.2322009801864624`
- `episode_reward_min`: `-8.849853515625`
- `episode_reward_max`: `-3.093963384628296`
- `done_fraction`: `1.0`
- `wall_time`: `68.2435936529946`
- `sps`: `234.45424168833907`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### GPU 500k Sanity

- Command family: Route B SAC, `num_timesteps=500000`, `num_envs=128`,
  `batch_size=256`, `max_replay_size=500000`, `grad_updates_per_step=2`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Checkpoint readiness: PASS
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- `env_steps`: `499968`
- `gradient_steps`: `7798`
- `wall_time`: `223.88994164399628`
- `sps`: `2233.0971919899416`
- `actor_loss`: `-3.510934352874756`
- `critic_loss`: `0.04195608198642731`
- `alpha`: `0.008012857288122177`
- `alpha_loss`: `0.20854677259922028`
- `policy_log_prob`: `-12.072959899902344`
- `q`: `3.348696231842041`
- `target_q`: `3.3187503814697266`
- `truncation_fraction`: `0.0`
- NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure observed: no

The actual step count is `499968` because Route B uses
`num_envs * (num_timesteps // num_envs)`.

### 500k Bounded Deterministic Eval

- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Eval JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- Status: `EVAL_OK`
- `num_eval_envs`: `16`
- `episode_length`: `1000`
- `eval_env_steps`: `16000`
- `episode_reward_mean`: `-4.691065788269043`
- `episode_reward_std`: `1.1929610967636108`
- `episode_reward_min`: `-9.040802955627441`
- `episode_reward_max`: `-3.7163496017456055`
- `done_fraction`: `1.0`
- `wall_time`: `68.70198891899781`
- `sps`: `232.88990976468807`
- `action_nan`: `false`
- `reward_nan`: `false`
- `obs_nan`: `false`
- `truncation_present`: `true`
- `truncation_fraction`: `0.0`

### Both-Mode Eval Diagnostic

- Scope: eval-only, no training.
- Script: `scripts/eval_sac_checkpoint.py --policy_mode both`
- Checkpoints: 100k, 250k, 500k.
- Seeds: `0..4`.
- Eval scale: `num_eval_envs=16`, `episode_length=1000`.
- Checkpoint readiness: PASS for all three checkpoints.
- Deterministic reward mean aggregate:
  - 100k: `-4.2218`
  - 250k: `-4.4585`
  - 500k: `-4.8476`
- Deterministic action abs mean:
  `0.1823 -> 0.2148 -> 0.3029`
- Stochastic reward mean aggregate:
  - 100k: `-6.4616`
  - 250k: `-6.1954`
  - 500k: `-5.9091`
- Stochastic log-prob mean:
  `-17.9594 -> -17.2847 -> -13.4275`
- No eval failure, traceback, OOM, CUDA fatal error, or action/reward/obs NaN
  was observed.
- Interpretation: deterministic `tanh(mean)` behavior degrades, while sampled
  stochastic behavior does not show the same degradation.

### 250k Risk Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env, checkpoint, or eval failure was
  observed.
- Alpha dropped to about `0.0187`; Q and target Q rose to about `5.2` while
  critic loss stayed low; bounded eval reward did not improve versus 100k.
  This is not a failure, but a future 500k run should watch alpha collapse,
  Q drift, critic loss, eval NaN flags, and reward trend.

### 500k Risk Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env load/reset/step/shape, replay,
  checkpoint, or eval failure was observed.
- Alpha continued down from about `0.0187` at 250k to about `0.0080` at 500k.
- Q and target Q decreased from about `5.2` at 250k to about `3.3` at 500k;
  critic loss stayed finite/low.
- Bounded eval reward mean worsened from `-4.27974` at 250k to `-4.69107` at
  500k. Eval max also worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but alpha decline and eval degradation must
  block any automatic jump to 1M.

### 100k Warning Notes

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- The first sandboxed `uv` attempt hit a `snap-confine` capability issue before
  training started. The identical command succeeded with external permission and
  unchanged parameters, so this is tooling noise, not a training failure.

## Current Claims

It is reasonable to claim:

- SAC Route B imports, initializes, trains briefly, checkpoints, reloads, and
  runs bounded deterministic eval on the target WSL2 CUDA/JAX stack.
- The current checkpoint schema is sufficient for deterministic actor eval when
  `normalize_observations=True`.
- The 10k, 50k, 100k, 250k, and 500k GPU runs did not show NaN, Inf, OOM,
  fatal CUDA failure, env failure, checkpoint failure, or eval failure.
- Both-mode eval shows the 500k quality concern is concentrated in the
  deterministic `tanh(mean)` path; sampled stochastic eval does not show the
  same degradation.
- Fresh 100k and fresh 250k train-time diagnostics show actor mean drift is
  already visible by 100k and amplifies in absolute level by 250k.
- Bounded A4 250k and 500k alpha/entropy extensions mitigate the corresponding
  baseline drift patterns without eliminating A4's own longer-horizon drift.
- Bounded A4 750k is runtime stable but not a clean improvement. It passed
  training, checkpoint, and eval gates, but actor drift and eval quality
  worsened versus A4 500k.
- Runtime artifacts are ignored and have not been committed.

It is not yet reasonable to claim:

- 1M training stability.
- Any final policy quality or solved task performance.
- Tuned rewards, tuned action scale, tuned stiffness/damping, or optimized SAC
  hyperparameters.
- PPO/RSL parity or superiority.
- Robustness under domain randomization, fine-tuning, deployment, or long
  rollout evaluation.

## Remaining Risks

- SAC algorithm maturity: Route B has the core SAC pieces, but 1M-scale
  training has not been validated.
- Long training stability: 1M is still untested, so late NaN, replay drift,
  alpha instability, or target-Q drift remain possible.
- 500k exposed training-dynamics risk: alpha reached about `0.0080` and
  deterministic eval reward worsened versus 250k despite no runtime failure.
- Both-mode eval narrowed the quality issue: deterministic `tanh(mean)` reward
  degrades while sampled stochastic reward improves slightly, so the next risk
  area is actor mean / action distribution behavior.
- Fresh 100k/250k train-time diagnostics support that the actor mean /
  deterministic action drift starts early and grows in absolute level.
- Fresh 100k alpha/entropy ablation plus multi-seed eval suggests A4 can
  reduce early actor mean and deterministic action magnitude while preserving
  healthier std. This is useful but not sufficient for 750k/1M because A1
  slightly edges deterministic reward and A4 remains weaker on stochastic
  reward.
- Bounded fresh 250k and 500k A4 extensions mitigated the corresponding
  baselines, but did not eliminate A4's own 100k-to-500k drift. Q/target_q and
  critic loss remain watch items before any longer extension.
- Bounded fresh 750k A4 bridge passed runtime/checkpoint/eval gates, but actor
  mean abs rose to `0.37190`, deterministic action abs rose to `0.31850`,
  critic loss rose to `0.1441`, deterministic 5-seed eval worsened to
  `-5.7314`, and stochastic 5-seed eval worsened to `-6.3802`. This blocks any
  automatic 1M or longer run.
- Fresh 100k actor-regularization R1 passed runtime/checkpoint/eval gates, but
  actor mean abs worsened versus A4 100k (`0.20693 -> 0.23127`),
  deterministic action abs worsened (`0.19314 -> 0.21378`), stochastic
  5-seed reward worsened (`-6.4935 -> -6.6210`), and the final regularization
  contribution was only `0.000886`. Do not extend R1 to 250k as-is.
- Eval reward is still low and should be treated as a smoke signal, not a
  performance benchmark.
- Truncation handling is currently synthesized as zero when absent. That passed
  the tested ladder but is still a modeling assumption to watch in longer runs.
- Replay, normalizer, and checkpoint interactions are now covered by schema
  checks through 500k, but not at 1M scale.
- Performance/SPS varies strongly because the first 10k run paid more compile
  and warmup cost. Use same-machine comparisons only.
- No PPO comparison has been run for the same conditions.

## Completed 100k Plan

The 100k sanity run used the 50k parameter shape and increased the horizon
conservatively:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

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
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_sanity
```

Observed checkpoint:

```text
./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl
```

Actual env steps:

```text
128 * (100000 // 128) = 99968
```

Replay memory estimate: around 256 MB raw sample storage for 100k entries, plus
JAX/device overhead and optimizer/network state. This fit the known 12GB GPU
budget during the recorded run.

Checkpoint readiness command:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --require_eval_ready
```

Bounded eval command:

```bash
uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --num_eval_envs 16 \
  --episode_length 1000 \
  --render False \
  --output_json ./logs/sac_eval_100k/eval_16x1000.json
```

Stop immediately and report if any of these occur:

- JAX backend is not GPU/CUDA.
- `nvidia-smi` is unavailable.
- Training does not print `TRAIN_OK`.
- Any NaN, Inf, OOM, CUDA, env load, obs/action shape, replay-update,
  checkpoint, or eval failure appears.
- Checkpoint readiness fails.
- Deterministic eval returns action/reward/obs NaN.
- Git status shows unignored logs, checkpoints, `.venv`, or menagerie files.

## 1M Decision And Readiness Plan

Do not automatically jump to 1M or any longer run from this report update. The
next recommended step is a decision review/design pass using the bounded A4
750k evidence and fresh 100k R1 result. The 750k bridge was runtime stable but
not a clean improvement: actor drift, deterministic eval, stochastic eval, and
critic loss worsened versus A4 500k. R1 showed the first actor-regularization
coefficients are too weak for train-time drift control. A later 1M review
should consider whether alpha floor, target entropy, log-alpha dynamics, critic
scale, stronger actor regularization, or deterministic mean action drift need
more analysis before a longer run. Consider 1M only with:

- explicit resource budget
- fresh logdir and checkpoint path
- checkpoint readiness gate
- bounded deterministic eval command
- stop conditions for NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval failures
- no logs/checkpoints/.venv/menagerie committed
- clean ignored-artifact audit

The 1M replay buffer can require roughly 2.5-2.7 GB raw storage before overhead,
so memory pressure, checkpoint size, compile behavior, and replay update cost
must be budgeted explicitly. A 1M run should have its own user-approved command,
fresh logdir, post-run checkpoint check, bounded eval, and report update.

## Standing Prohibitions

- Do not do domain randomization.
- Do not do fine-tuning.
- Do not tune reward, `action_scale`, Kp, or other control gains.
- Do not modify PPO/RSL paths for SAC validation.
- Do not commit `logs/`, checkpoints, `.venv/`, or menagerie.
- Do not treat WSL2/JAX warnings as blockers unless the command fails.
