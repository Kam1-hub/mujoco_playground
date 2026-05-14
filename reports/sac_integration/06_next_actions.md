# Next Actions

Status: updated on 2026-05-14 after the fixed-command 3M render helper smoke.

## Immediate State

- Route B `train-g1-sac` is locally runnable on CPU for tiny smoke.
- Env API readiness is validated for flat and rough G1 tasks on the CPU dev
  path after menagerie is present.
- CPU tiny smoke passed with checkpoint
  `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`.
- GPU migration prep scripts are added.
- CPU-side migration prep checks pass: compileall, checkpoint inspection, and
  `gpu_preflight.py --impl jax` without `--require_gpu`.
- Target WSL2 documentation workspace:
  `/home/admin/projects/mujoco_playground/g1_sac_dev`.
- Target WSL2 workspace now has CUDA JAX via `uv sync --frozen --extra cuda`.
- JAX reports backend `gpu` and device `cuda:0`.
- Menagerie is present at commit
  `1b86ece576591213e2b666ebf59508454200ca97`.
- GPU preflight with `--require_gpu` passed.
- Route B GPU 10k smoke passed with `TRAIN_OK`.
- The existing GPU 10k checkpoint predates normalizer persistence and is not
  deterministic-eval ready.
- Future Route B checkpoints now persist `policy_normalizer` and
  `value_normalizer`; `scripts/check_sac_checkpoint.py --require_eval_ready`
  can enforce this gate.
- Schema dry-run checkpoint
  `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl` contains both
  normalizers and passes `--require_eval_ready`.
- New GPU smoke checkpoint
  `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` contains both
  normalizers and passes `--require_eval_ready`.
- Deterministic eval CLI `scripts/eval_sac_checkpoint.py` has a passing smoke:
  4 env x 200 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_smoke/eval_4x200.json`, no action/reward/obs NaN.
- Route B GPU 50k sanity passed with `TRAIN_OK`.
- 50k checkpoint
  `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 50k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_50k/eval_16x1000.json`, and no action/reward/obs NaN.
- Route B GPU 100k sanity passed with `TRAIN_OK`.
- 100k checkpoint
  `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 100k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_100k/eval_16x1000.json`, and no action/reward/obs NaN.
- Route B GPU 250k sanity passed with `TRAIN_OK`.
- 250k checkpoint
  `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 250k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_250k/eval_16x1000.json`, and no action/reward/obs NaN.
- Route B GPU 500k sanity passed with `TRAIN_OK`.
- 500k checkpoint
  `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` exists under
  ignored `logs` and passes `--require_eval_ready`.
- 500k bounded deterministic eval passed with 16 env x 1000 steps, `EVAL_OK`,
  JSON `./logs/sac_eval_500k/eval_16x1000.json`, and no action/reward/obs NaN.
- Alpha/entropy diagnostic patch is synced at
  `926a14f Add SAC alpha entropy diagnostics`.
- Both-mode eval-only diagnostic passed for 100k, 250k, and 500k checkpoints
  using `--policy_mode both`, seeds `0..4`, `num_eval_envs=16`, and
  `episode_length=1000`.
- Both-mode diagnostic result: deterministic `tanh(mean)` reward degrades
  across 100k/250k/500k, while sampled stochastic reward does not show the same
  degradation.
- Full action diagnostic eval passed for 100k, 250k, and 500k checkpoints with
  `--policy_mode both --action_diagnostics --reward_components`.
- Full action diagnostic output: `./logs/sac_eval_action_diag_full/`, `15`
  ignored JSON files.
- Full action diagnostic result: deterministic action magnitude and actor mean
  magnitude increase with scale, policy std narrows, and deterministic reward
  degradation aligns mainly with angular-velocity, stand-still, and orientation
  reward components.
- Train-time actor drift instrumentation is committed at
  `202c6a9 Add SAC actor drift train diagnostics`.
- Fresh 100k actor drift diagnostic passed with `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`, checkpoint
  readiness PASS, and 4 env x 200 action diagnostic eval `EVAL_OK`.
- Fresh 100k result: final actor mean magnitude and deterministic action
  magnitude are already above their interval averages, while final log_std/std
  are lower than interval averages.
- Fresh 250k actor drift diagnostic passed with `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`, checkpoint
  readiness PASS, and 4 env x 200 action diagnostic eval `EVAL_OK`.
- Fresh 250k result: actor mean abs and deterministic action abs increased
  versus fresh 100k, while log_std/std and alpha continued downward.
- No-training action mapping tools are committed:
  `scripts/inspect_g1_action_mapping.py` and
  `scripts/summarize_sac_action_diag.py`.
- Action joint mapping diagnostic is recorded in
  `reports/sac_integration/13_action_joint_mapping_diagnostic.md`; it maps the
  500k deterministic top action dimensions mainly to right ankle roll/pitch,
  waist pitch, right knee, and hip roll.
- Fresh 100k alpha/entropy ablation A1/A3/A4 passed runtime, checkpoint
  readiness, and small eval gates. Results are recorded in
  `reports/sac_integration/12_alpha_entropy_ablation_plan.md`.
- A4 is the best current 100k candidate: `alpha=0.042848`, actor mean abs
  `0.206928`, deterministic action abs `0.193139`, log_std mean `-0.151827`,
  and std mean `0.861342`; it had the best deterministic 4x200 reward among
  A1/A3/A4.
- Caveat: A4 stochastic 4x200 reward was worse than A1/A3, so this remains a
  diagnostic signal rather than a final policy-quality benchmark.
- Fresh 100k A1/A3/A4 multi-seed eval-only follow-up also passed:
  `./logs/sac_eval_alpha_ablate_multiseed/` contains `15` ignored JSON outputs,
  all `EVAL_OK`, all action/reward/obs NaN flags false.
- Multi-seed result: A4 remains best on deterministic action/mean drift
  metrics, but A1 is slightly better on deterministic reward
  (`-4.3262` vs A4 `-4.3436`) and A4 stochastic reward is worse than A1 by
  about `0.0945`.
- Bounded fresh 250k A4 alpha/entropy extension passed with `TRAIN_OK`,
  checkpoint
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`,
  checkpoint readiness PASS, 4 env x 200 eval PASS, and 5-seed 16 env x 1000
  eval PASS.
- A4 250k mitigates drift versus the fresh 250k baseline: alpha higher by
  `+0.01578`, actor mean abs lower by `-0.07330`, deterministic action abs
  lower by `-0.05875`, log_std less negative by `+0.03434`, and std higher by
  `+0.02802`.
- A4 250k does not eliminate drift relative to A4 100k, and Q/target_q are
  higher than earlier baselines (`q=8.7442`, `target_q=8.7170`).
- Bounded fresh 500k A4 alpha/entropy extension passed with `TRAIN_OK`,
  checkpoint
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`,
  checkpoint readiness PASS, 4 env x 200 eval PASS, and 5-seed 16 env x 1000
  eval PASS.
- A4 500k mitigates the old fresh 500k deterministic drift pattern: alpha
  stayed at `0.02435` instead of the old 500k `0.00801`, deterministic eval
  reward averaged `-4.4075` instead of the old degraded `-4.69` to `-4.82`
  range, and actor/log_std metrics were healthier.
- A4 500k still drifts relative to A4 250k, and Q/target_q plus critic loss
  remain watch items (`q=8.47294`, `target_q=8.41017`, `critic_loss=0.0803`).
- Bounded fresh 750k A4 alpha/entropy bridge passed with `TRAIN_OK`,
  checkpoint
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`,
  checkpoint readiness PASS, 4 env x 200 eval PASS, and 5-seed 16 env x 1000
  eval PASS.
- A4 750k is runtime stable but not a clean stability improvement: alpha
  declined `0.02435 -> 0.01725`, actor mean abs rose
  `0.29323 -> 0.37190`, deterministic action abs rose
  `0.26540 -> 0.31850`, critic loss rose `0.0803 -> 0.1441`,
  deterministic 5-seed reward worsened `-4.4075 -> -5.7314`, and stochastic
  5-seed reward worsened `-6.0434 -> -6.3802`.
- Default-off actor regularization is committed and fresh 100k R1 passed
  runtime/checkpoint/eval gates with
  `deterministic_action_l2_coef=0.01` and `actor_mean_l2_coef=0.001`.
- R1 did not reduce train-time drift versus the A4 100k baseline: actor mean
  abs worsened `0.20693 -> 0.23127`, deterministic action abs worsened
  `0.19314 -> 0.21378`, and stochastic 5-seed reward worsened
  `-6.4935 -> -6.6210`.
- R1 regularization contribution was tiny (`0.000886`) relative to actor loss,
  so this coefficient set appears too weak and should not be extended to 250k
  as-is.
- Fresh 100k actor-regularization R2/R3 coefficient sweep passed with
  `TRAIN_OK`, checkpoint readiness PASS, 4 env x 200 eval PASS, and 5-seed
  eval PASS.
- R2 used `deterministic_action_l2_coef=0.1` and
  `actor_mean_l2_coef=0.01`; R3 used `deterministic_action_l2_coef=0.5` and
  `actor_mean_l2_coef=0.05`. Both used A4 alpha settings.
- R3 is the best current 100k regularization candidate: train actor mean abs
  `0.1506`, deterministic action abs `0.1430`, deterministic 5-seed reward
  `-4.1681`, and stochastic 5-seed reward `-6.3983`.
- R3 has a critic-loss watch item (`0.1684` vs R2 `0.1383`); R2 remains a
  conservative backup.
- Bounded R3 250k has passed with `TRAIN_OK`, checkpoint readiness PASS, 4 env
  x 200 action diagnostic eval PASS, and 5-seed eval PASS.
- R3 retained drift control at 250k: train actor mean abs
  `0.1506 -> 0.1630`, deterministic action abs `0.1430 -> 0.1556`,
  deterministic 5-seed reward `-4.1681 -> -3.7941`, stochastic 5-seed reward
  `-6.3983 -> -5.9802`, and critic loss `0.1684 -> 0.0536`.
- User flagged the previous 128-env ladder as likely too conservative for G1
  policy-quality conclusions. Treat 10k through 500k as runtime/diagnostic
  gates, not final learning-quality evidence.
- High-parallel capacity results are recorded in
  `reports/sac_integration/14_high_parallel_capacity_results.md`: 512, 1024,
  and 2048 envs all passed, and 1024 envs was fastest.
- Bounded 1024-env 1M R3 run passed runtime/checkpoint/eval gates. Checkpoint:
  `./logs/sac_lift_gpu_1m_env1024_r3_b256_g16_replay1m/sac_lift_step_999424.pkl`.
- 1M result: runtime stability supports `1024 envs + replay1M + R3 + UTD~4`
  on the 12GB GPU, but policy quality is not solved. Versus R3 250k,
  deterministic reward worsened `-3.7941 -> -4.9891`, stochastic reward
  worsened `-5.9802 -> -6.7546`, actor mean abs rose `0.1630 -> 0.2299`, and
  log_std narrowed `-0.1709 -> -0.2881`.
- Bounded 1024-env 3M R3 run passed runtime/checkpoint/eval gates. Checkpoint:
  `./logs/sac_lift_gpu_3m_env1024_r3_b256_g16_replay1m/sac_lift_step_2999296.pkl`.
- 3M result: runtime stability remains clean and deterministic policy eval
  recovered strongly. Deterministic 5-seed reward improved from 1M
  `-4.9891` to `-2.3314`, and from R3 250k `-3.7941` to `-2.3314`.
  However stochastic reward worsened to `-10.9904`, alpha collapsed to
  `0.000766`, log_std to `-0.9573`, and std to `0.4106`.
- Eval-only render helper `scripts/render_sac_checkpoint.py` is available.
  The deterministic 3M checkpoint smoke rendered
  `./logs/sac_render_3m_r3/render_seed0_det.mp4` with `RENDER_OK`, `300`
  frames, `10s`, H.264 MP4, and `done=false`. Main sampled-frame inspection
  found a nonblank upright humanoid with no obvious fall.
- Fixed-command render support is available in
  `scripts/render_sac_checkpoint.py` through `--fixed_command`,
  `--command_x`, `--command_y`, and `--command_yaw`. Three 3M R3 fixed-command
  smokes have local ignored artifacts under `./logs/sac_render_3m_r3_fixedcmd/`:
  forward `[0.5, 0.0, 0.0]`, stand `[0.0, 0.0, 0.0]`, and yaw
  `[0.0, 0.0, 0.5]`. All reported `RENDER_OK`, `done=false`, and `600`
  frames.

## Current Recommendation

- Do not jump directly to 10M.
- Do not declare stable SAC integration from the 3M result.
- Next step should use the fixed-command MP4s under
  `./logs/sac_render_3m_r3_fixedcmd/` for command-specific visual inspection,
  followed by an entropy/alpha decision review before any longer run:
  - decide whether alpha/log_std need a floor or another entropy-handling
    ablation;
  - decide whether to run a bounded entropy ablation before longer training;
  - only then decide whether the deterministic improvement justifies a
    carefully gated longer run.

## Completed WSL2 GPU Validation

- GPU preflight: `PASS`
- Route B GPU 10k smoke: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- Requested timesteps: `10000`
- Actual env steps: `9984`
- Gradient steps: `142`
- Wall time: `56.50599093900382`
- SPS: `176.68922947970893`
- Actor loss: `-1.9058758020401`
- Critic loss: `0.08382290601730347`
- Alpha: `0.04770537465810776`
- Truncation fraction: `0.0`
- NaN: no NaN observed in reported scalar metrics

## Completed 50k Sanity Validation

- Route B GPU 50k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- Checkpoint eval readiness: `PASS`
- Requested timesteps: `50000`
- Actual env steps: `49920`
- Gradient steps: `766`
- Wall time: `35.99766752999858`
- SPS: `1386.756515777009`
- Actor loss: `-3.6135072708129883`
- Critic loss: `0.07097882032394409`
- Alpha: `0.03992176800966263`
- Alpha loss: `1.327394962310791`
- Policy log prob: `-18.81831169128418`
- Policy Q: `2.8622469902038574`
- Q: `2.884032726287842`
- Target Q: `2.899707317352295`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/CUDA/checkpoint/eval error: none observed

Bounded deterministic eval after 50k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.5016322135925293` / `0.75983726978302` /
  `-6.096090316772461` / `-2.531925916671753`
- Done fraction: `1.0`
- Wall time: `74.76505397899746`
- SPS: `214.00372431342888`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

## Completed 100k Sanity Validation

- Route B GPU 100k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- Checkpoint eval readiness: `PASS`
- Requested timesteps: `100000`
- Actual env steps: `99968`
- Gradient steps: `1548`
- Wall time: `56.80434615799459`
- SPS: `1759.8653406193746`
- Actor loss: `-4.994826316833496`
- Critic loss: `0.04054964333772659`
- Alpha: `0.03259027376770973`
- Alpha loss: `1.0562278032302856`
- Policy log prob: `-17.77903938293457`
- Q: `4.3698601722717285`
- Target Q: `4.456111907958984`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/checkpoint/eval error: none observed

Bounded deterministic eval after 100k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-3.894726037979126` / `0.9610732197761536` /
  `-7.2897186279296875` / `-2.889821767807007`
- Done fraction: `1.0`
- Wall time: `66.85148939098872`
- SPS: `239.33647770242092`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

## Completed 250k Sanity Validation

- Route B GPU 250k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- Checkpoint eval readiness: `PASS`
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `250000`
- Actual env steps: `249984`
- Gradient steps: `3892`
- Wall time: `120.21957968600327`
- SPS: `2079.3950590488107`
- Actor loss: `-5.524118900299072`
- Critic loss: `0.02644157037138939`
- Alpha: `0.018743595108389854`
- Alpha loss: `0.5869507789611816`
- Policy log prob: `-16.657032012939453`
- Q: `5.197851181030273`
- Target Q: `5.205532073974609`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error: none observed

Bounded deterministic eval after 250k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.279743194580078` / `1.2322009801864624` /
  `-8.849853515625` / `-3.093963384628296`
- Done fraction: `1.0`
- Wall time: `68.2435936529946`
- SPS: `234.45424168833907`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

## Completed 500k Sanity Validation

- Route B GPU 500k sanity: `PASS`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- Checkpoint eval readiness: `PASS`
- `policy_normalizer` / `value_normalizer`: present
- `deterministic_eval_ready`: `true`
- Requested timesteps: `500000`
- Actual env steps: `499968`
- Gradient steps: `7798`
- Wall time: `223.88994164399628`
- SPS: `2233.0971919899416`
- Actor loss: `-3.510934352874756`
- Critic loss: `0.04195608198642731`
- Alpha: `0.008012857288122177`
- Alpha loss: `0.20854677259922028`
- Policy log prob: `-12.072959899902344`
- Q: `3.348696231842041`
- Target Q: `3.3187503814697266`
- Truncation fraction: `0.0`
- NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure: none observed

Bounded deterministic eval after 500k:

- Status: `EVAL_OK`
- JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- Eval env steps: `16000`
- Episode reward mean/std/min/max:
  `-4.691065788269043` / `1.1929610967636108` /
  `-9.040802955627441` / `-3.7163496017456055`
- Done fraction: `1.0`
- Wall time: `68.70198891899781`
- SPS: `232.88990976468807`
- NaN: `action_nan=false`, `reward_nan=false`, `obs_nan=false`
- Truncation: `truncation_present=true`, `truncation_fraction=0.0`

Artifacts remain ignored under `logs`; do not commit logs, checkpoints, `.venv`,
or menagerie.

## Historical Post-Capacity Recommendation

This section is retained as historical context. The 1024-env 1M and 3M R3
runs have now both been executed and recorded. Do not run 10M training yet.

Fresh 100k alpha/entropy ablation A1/A3/A4, the multi-seed eval-only
follow-up, the bounded fresh 250k, 500k, and 750k A4 extensions, the fresh 100k
actor-regularization R1 ablation, the fresh 100k R2/R3 coefficient sweep, and
the bounded R3 250k extension are complete. The 512/1024/2048 high-parallel
capacity benchmark is also complete. All three capacity runs were `TRAIN_OK`
and readiness PASS with sampled UTD preserved at about `4.0`; `1024` envs was
fastest at `953.36` SPS, while `2048` envs was feasible but slower.

1. Keep `1024 envs`, `batch_size=256`, `grad_updates_per_step=16`, and
   `max_replay_size=1000000` as the proven high-parallel runtime baseline.
2. Keep `2048` as feasible but not selected unless a later batch/update-ratio
   pass makes it faster and non-fragile.
3. Treat the completed 3M result as mixed: deterministic recovery is strong,
   but entropy collapse and stochastic degradation block automatic 10M.
4. Do not use `max_replay_size=num_timesteps` for 5M or 10M runs on 12GB VRAM.
5. Do not run 10M directly from this report update.
6. Do not tune reward, `action_scale`, or Kp yet.

## Completed Both-Mode Eval Diagnostic

- Scope: eval-only; no training.
- Checkpoints: 100k, 250k, 500k.
- Eval command class: `scripts/eval_sac_checkpoint.py --policy_mode both`.
- Seeds: `0`, `1`, `2`, `3`, `4`.
- Deterministic reward mean aggregate:
  `100k=-4.2218`, `250k=-4.4585`, `500k=-4.8476`.
- Deterministic action abs mean:
  `0.1823 -> 0.2148 -> 0.3029`.
- Stochastic reward mean aggregate:
  `100k=-6.4616`, `250k=-6.1954`, `500k=-5.9091`.
- Stochastic log-prob mean:
  `-17.9594 -> -17.2847 -> -13.4275`.
- Interpretation: deterministic `tanh(mean)` behavior degrades while sampled
  stochastic behavior does not show the same degradation. Stochastic reward is
  still lower in absolute terms at each checkpoint.
- Full details: `reports/sac_integration/10_both_mode_eval_diagnostic.md`.

250k tooling, warning, and risk notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed again and remained non-fatal.
- No traceback, NaN, Inf, OOM, fatal CUDA, env, checkpoint, or eval failure was
  observed.
- Alpha dropped to about `0.0187` by 250k; Q and target Q rose to about `5.2`
  while critic loss stayed low; bounded eval reward did not improve versus
  100k. This is not a failure, but 500k should watch alpha collapse, Q drift,
  critic loss, eval NaN flags, and reward trend.

500k tooling, warning, and risk notes:

- No traceback, NaN, Inf, OOM, fatal CUDA, env load/reset/step/shape, replay,
  checkpoint, or eval failure was observed.
- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed again and remained non-fatal.
- Alpha continued down from about `0.0187` at 250k to about `0.0080` at 500k.
- Q and target Q decreased from about `5.2` to about `3.3`; critic loss stayed
  finite/low.
- Bounded eval reward mean worsened from `-4.27974` to `-4.69107`; eval max
  also worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but the alpha decline and eval degradation
  block any automatic jump to 1M.

Actual step note:

- Route B currently uses `num_envs * (num_timesteps // num_envs)`.
- With `num_timesteps=10000` and `num_envs=128`, this yields `9984` actual
  env steps.
- With `num_timesteps=100000` and `num_envs=128`, this yields `99968` actual
  env steps.
- With `num_timesteps=250000` and `num_envs=128`, this yields `249984` actual
  env steps.
- With `num_timesteps=500000` and `num_envs=128`, this yields `499968` actual
  env steps.

## Preflight Checklist

Before any Python/JAX command:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

Prepare dependencies and assets:

```bash
uv sync --frozen --extra cuda
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
```

If menagerie already exists, only verify its commit. Do not commit menagerie,
`.venv`, logs, or checkpoints.

Verify runtime:

```bash
nvidia-smi
uv run --no-sync python -c "import jax; print(jax.default_backend()); print(jax.devices())"
uv run --no-sync python -c "import mujoco, brax; print('mujoco', mujoco.__version__); print('brax', brax.__version__)"
```

`nvcc` is not required in WSL2. JAX must report GPU/CUDA/ROCm. If JAX reports
CPU, stop and report the CUDA/JAX blocker.

## Recommended Next Step

The bounded 1024-env 3M R3 result has now been executed and recorded. Do not
automatically run 10M. The next useful step is a decision review using the 3M
evidence:

1. Entropy path: review alpha/log_std handling because alpha collapsed to
   `0.000766`, log_std to `-0.9573`, and stochastic reward worsened to
   `-10.9904`.
2. Inspection path: plan a render helper for deterministic visual inspection,
   because deterministic eval improved strongly to `-2.3314`.
3. Training path: only after that review, decide whether a carefully gated
   longer run is justified.

Any 10M plan must restate memory stop conditions, keep replay cap decoupled
from `num_timesteps`, and must not change reward/action scale/Kp.

## Migration Reminders

- Keep first SAC smoke defaults only for smoke reproduction. For capacity
  testing, use R3 settings and coordinated env/update ratios from
  `reports/sac_integration/13_high_parallel_capacity_plan.md`.
- Record JAX backend/devices, MuJoCo version, Brax version, idle/final VRAM when
  available, SPS, losses, alpha, checkpoint path, and any full traceback.
- Record actual `env_steps`; Route B may record `9984` for a `10000` target with
  `128` envs.
- `gpu_preflight.py --impl jax` without `--require_gpu` is not GPU validation.

## Recommended Follow-Ups After Deterministic Eval Smoke

1. Run a decision review before any longer A4 bridge.
2. Keep eval scales bounded unless the user asks for a benchmark.
3. Do not start 1M automatically from this report update.
4. Compare against PPO baseline only after SAC smoke plus eval have clean
   reports.

## Do Not Start Yet

- 1M or longer training before capacity results and a resource/stop-condition
  plan
- PPO-scale `num_envs=8192` experiments as SAC smoke substitutes
- domain randomization
- reward, `action_scale`, or Kp tuning
- world model
- fine-tuning
- vision SAC
- real deployment
