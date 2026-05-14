# G1 SAC Integration Status And Roadmap

Status: updated on 2026-05-15 after the eval-only 100k termination/contact diagnostic sweep.

## 1. Mission

The project goal is to add a SAC baseline to the G1 pipeline without breaking the existing PPO and RSL-RL paths.

Target pipeline:

```text
G1 pipeline
├── PPO baseline: train-g1-jax
├── RSL-RL baseline: train-g1-rsl
├── SAC fallback/minimal baseline: train-g1-sac-brax
└── SAC main baseline: train-g1-sac
```

The primary route is Route B: a local asymmetric SAC implementation for G1.

Route B defaults:

- Actor observation: `obs["state"]`
- Critic observation: `obs["privileged_state"]`
- Policy obs shape: `(103,)`
- Value obs shape: `(216,)`
- Action size: `29`
- Actor distribution: tanh Gaussian
- Critic: twin Q with target Q
- Entropy: trainable alpha
- Replay: uniform replay buffer
- Checkpoint: pickle checkpoint for smoke validation
- Checkpoint schema: policy/Q/target/log-alpha, metrics, config, and
  policy/value observation normalizers
- Default env implementation: `impl="jax"`

Action scaling decision:

- The actor outputs tanh-normalized actions.
- Actions are passed directly to the G1 env.
- The G1 env applies `action_scale` internally.
- Do not add external action scaling unless the SAC objective is deliberately redesigned and documented.

Truncation decision:

- Runtime G1 env did not expose `state.info["truncation"]`.
- Route B synthesizes zero truncation when missing.
- The training metrics report `truncation_fraction`.

## 2. Hard Boundaries

Do not:

- Break or alter existing PPO/RSL-RL behavior.
- Wholesale copy LIFT forks.
- Modify `D:\mujoco_playground\template`.
- Modify `D:\mujoco_playground\LIFT-humanoid`.
- Commit menagerie assets.
- Commit `.venv`, `logs`, checkpoints, or generated caches.
- Run 1M or long training before 10k GPU smoke passes.
- Enable domain randomization for the first GPU smoke.
- Start world model, fine-tuning, vision SAC, or real deployment.
- Tune reward, `action_scale`, or Kp during migration smoke.
- Import PPO/Barkour `2048` or `8192` env assumptions into the first SAC smoke.

Failure handling rule:

- Preserve full command, stdout/stderr, traceback, failure category, diagnosis, attempted fix, and next concrete action.

Failure categories:

- dependency
- CUDA/JAX backend
- menagerie
- env load
- obs/action shape
- replay/update
- checkpoint

## 3. Route Map

### Phase 0: Workspace Manifest

Goal:

- Establish project paths, runtime, dependency visibility, git branch/commit, and validation boundaries.

Status:

- Completed.
- Reports exist under `reports/sac_integration/`.

### Phase 1: G1 Env API Readiness

Goal:

- Validate registry, config, env load, reset, step, obs schema, action size, truncation info, and domain randomizer presence.

Status:

- Completed on CPU dev with `impl="jax"`.
- Flat and rough env load/reset/step pass.
- Runtime schema:
  - obs type: dict
  - obs keys: `state`, `privileged_state`
  - `state`: `(103,)`
  - `privileged_state`: `(216,)`
  - action size: `29`
  - truncation key: absent

### Phase 2: Route A Brax SAC Fallback

Goal:

- Provide a minimal fallback SAC route using upstream Brax SAC where possible.

Status:

- Entry point exists: `train-g1-sac-brax`.
- Help command passes.
- Full runtime smoke is not the current priority.
- Route B remains the main route.

### Phase 3: Route B Asymmetric SAC Main Route

Goal:

- Implement a local LIFT-style asymmetric SAC baseline:
  - actor uses `state`
  - critic uses `privileged_state`
  - tanh Gaussian actor
  - twin Q
  - target Q
  - Polyak update
  - trainable alpha
  - replay buffer
  - obs normalization
  - dry run
  - checkpoint

Status:

- Implemented.
- Entry point exists: `train-g1-sac`.
- CLI supports required SAC parameters.
- Route B help passes.
- Dry run passes.
- CPU tiny smoke passes.

### Phase 4: Clean-Room Fallback

Goal:

- Provide a fallback if Route B gets blocked.

Status:

- Not activated.
- Route B is runnable on CPU tiny smoke, so Route C is not needed yet.

### Phase 5: Validation Ladder

Validation ladder:

1. Static compile
2. Registry import
3. Flat env API
4. Rough env API
5. Route A help
6. Route B help
7. Route B dry run
8. CPU tiny smoke
9. GPU preflight
10. GPU 10k smoke
11. 50k sanity run
12. 100k or longer sanity run
13. 250k sanity run
14. 500k sanity run
15. Both-mode deterministic/stochastic eval diagnostic
16. Full action distribution / reward-component eval diagnostic
17. Fresh 100k train-time actor drift diagnostic
18. Fresh 250k train-time actor drift diagnostic
19. Fresh 100k alpha/entropy ablation diagnostics
20. Fresh 100k alpha/entropy ablation multi-seed eval-only diagnostic
21. Bounded fresh 250k A4 alpha/entropy extension
22. Bounded fresh 500k A4 alpha/entropy extension
23. Bounded fresh 750k A4 alpha/entropy bridge
24. Fresh 100k actor regularization R1
25. Fresh 100k actor regularization R2/R3 coefficient sweep
26. Bounded R3 250k actor regularization extension
27. High-parallel 512/1024/2048 capacity benchmark
28. Bounded 1024-env 1M run
29. Bounded 1024-env 3M run
30. Deterministic checkpoint render helper smoke
31. Fixed-command checkpoint render helper smoke
32. Fixed-command eval helper smoke and alpha sign audit
33. Fixed-command forward eval gate
34. Fresh env1024 R3 100k `fixed_alpha=0.03` diagnostic
35. Fresh env1024 R3 100k `fixed_alpha=0.05` diagnostic
36. Fresh env1024 R3 100k `--env_feet_slip_scale 0.0` diagnostic
37. Fresh env1024 R3 100k `--env_push_enable False` diagnostic
38. Fresh env1024 R3 100k `--env_zero_command_phase_freeze True` diagnostic
39. Fresh env1024 R3 100k `--env_feet_air_time_command_mask True` diagnostic
40. Eval-only 100k termination/contact diagnostic sweep
41. 10M-scale training

Status:

- Steps 1 through 33 are complete.
- Step 29 is runtime/checkpoint/eval PASS with the first strong deterministic
  policy improvement signal, but not a clean stable-SAC declaration because
  stochastic eval degraded and alpha/std collapsed.
- Step 30 produced a deterministic 3M MP4 render smoke for visual inspection.
- Step 31 added fixed-command render support and produced three 600-frame
  fixed-command 3M R3 smokes for forward, stand, and yaw commands.
- Step 32 added fixed-command eval support and recorded the alpha sign audit.
  The `[0.5,0,0]` 3M R3 eval smoke returned `EVAL_OK`; the alpha sign audit
  found no direct sign bug but confirmed persistent downward alpha pressure is
  still a stability risk.
- Step 33 ran fixed-forward eval on the 3M R3 checkpoint for `[0.5,0,0]` and
  `[1,0,0]`, seeds `0..4`, `policy_mode=both`, action diagnostics, and reward
  components. All evals returned `EVAL_OK` and NaN flags false. The gate blocks
  direct 5M/10M because `fwd0.5` deterministic is only near break-even/noisy,
  `fwd1.0` deterministic is weak, and stochastic fixed-forward eval remains
  poor.
- Step 34 passed runtime/checkpoint gates with weak fixed-command smoke:
  fresh env1024 R3 100k `fixed_alpha=0.03` returned `TRAIN_OK`, checkpoint
  readiness PASS, and effective alpha stayed fixed at `0.03`, but `fwd0.5` and
  `fwd1.0` small fixed-command smokes had negative rewards, low
  `tracking_lin_vel`, and `termination=-100`.
- Step 35 passed runtime/checkpoint gates with weak fixed-command smoke:
  fresh env1024 R3 100k `fixed_alpha=0.05` returned `TRAIN_OK`, checkpoint
  readiness PASS, and effective alpha stayed fixed at `0.05`, but fixed-forward
  tracking stayed weak and `critic_loss=0.3446` crossed the previous watch
  threshold.
- A follow-up reward/prior gate passed runtime/checkpoint gates with weak
  fixed-command smoke: fresh env1024 R3 100k
  `--env_feet_slip_mode foot_velocity` returned `TRAIN_OK`, checkpoint
  readiness PASS, and finite metrics, but `fwd0.5` and `fwd1.0`
  fixed-command smokes still had negative deterministic rewards, low
  `tracking_lin_vel`, and `termination=-100`.
- Step 36 passed runtime/checkpoint gates and confirmed eval override
  inheritance: fresh env1024 R3 100k `--env_feet_slip_scale 0.0` returned
  `TRAIN_OK`, checkpoint readiness PASS, and inherited fixed-command eval
  reported `reward/feet_slip=0.0`, but fixed-command smoke remained weak.
- Step 37 passed runtime/checkpoint gates with a meaningful but incomplete
  fixed-command signal: fresh env1024 R3 100k `--env_push_enable False`
  returned `TRAIN_OK`, checkpoint readiness PASS, and inherited fixed-command
  eval with `push_config.enable=false` returned `EVAL_OK` with NaN flags false.
  Deterministic tracking improved versus prior 100k gates, but
  `reward/termination=-100` remained saturated for both `fwd0.5` and `fwd1.0`
  in deterministic and stochastic modes.
- Step 38 passed runtime/checkpoint gates but failed the stability gate:
  fresh env1024 R3 100k `--env_zero_command_phase_freeze True` returned
  `TRAIN_OK`, checkpoint readiness PASS, and inherited fixed-command/stand eval
  with `zero_command_phase_freeze=true` returned `EVAL_OK` with NaN flags
  false. Forward tracking was similar to push-disable, but
  `reward/termination=-100` remained saturated for `fwd0.5`, `fwd1.0`, and
  stand; stand did not improve.
- Step 39 passed runtime/checkpoint gates but failed the stability gate:
  fresh env1024 R3 100k `--env_feet_air_time_command_mask True` returned
  `TRAIN_OK`, checkpoint readiness PASS, and inherited fixed-command/stand eval
  with `feet_air_time_command_mask=true` returned `EVAL_OK` with NaN flags
  false. Stand eval correctly reported `reward/feet_air_time=0.0`, but
  `reward/termination=-100` remained saturated for `fwd0.5`, `fwd1.0`, and
  stand; `fwd1.0` tracking was worse than push-disable and phase-freeze, and
  stand did not improve.
- Step 40 passed as an eval-only diagnostic and identified the current 100k
  failure mode. The sweep covered push-disable, phase-freeze, and
  feet-air-time-mask checkpoints with fixed `fwd0.5`, `fwd1.0`, and stand
  commands in deterministic and stochastic modes. All `9` JSON outputs under
  `./logs/sac_eval_termination_diag_100k/` returned `EVAL_OK`; all NaN flags
  were false; `reward/termination=-100` appeared in all 18 mode cases; first
  done happened around `51-55` steps; and failure was fall-dominated rather
  than contact-dominated or numerical.
- Step 41 remains `NOT VALIDATED` and requires reward/prior targeted audit,
  resource plan, and stop conditions.

### Phase 6: Reports, Commits, Migration Handoff

Goal:

- Keep complete reports, commit verified phases, and prepare a clean WSL2/GPU continuation path.

Status:

- Completed for CPU dev and GPU migration prep.
- GitHub remote pushed.
- WSL2 handoff document added.
- WSL2 clone and read-only startup handoff have been completed by the WSL2 Codex CLI session.

## 4. Current Progress

Current GitHub branch:

- `sac-integration`

Key commits:

```text
208eef2 Add SAC actor regularization diagnostics
202c6a9 Add SAC actor drift train diagnostics
8ff4f1d Add SAC action distribution eval diagnostics
9cb5112 Record SAC both-mode eval diagnostics
926a14f Add SAC alpha entropy diagnostics
87bca63 Record SAC 500k sanity results
99da67d Record SAC 250k sanity results
ee3f766 Record SAC 100k sanity results
31cc105 Add SAC phase summary and handoff
a64eaf6 Record SAC 50k sanity results
5be043c Add SAC deterministic eval smoke
c59eda0 Save SAC normalizers in checkpoints
```

GitHub remote:

```text
https://github.com/Kam1-hub/mujoco_playground.git
```

Windows CPU dev state:

- Worktree was clean after the handoff commit.
- Menagerie exists locally and is ignored.
- Logs/checkpoints are ignored.
- `.venv` is ignored.
- CPU dev machine should not continue to larger training.

WSL2 target workspace state:

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration...origin/sac-integration`
- Latest recorded validation state before this report update: bounded
  `1024`-env 3M R3 runtime/checkpoint/eval PASS, with deterministic recovery,
  deterministic and fixed-command render helper smokes, fixed-command eval
  helper smoke, alpha sign audit, and fixed-command forward eval gate. The
  forward gate artifacts are under ignored `./logs/sac_eval_fixedcmd_3m_gate/`.
  `fwd0.5` deterministic averaged `0.0192`, `fwd1.0` deterministic averaged
  `-3.2302`, and stochastic fixed-forward eval remained poor. Later short
  100k gates with `fixed_alpha=0.03`, `fixed_alpha=0.05`, and
  `--env_feet_slip_mode foot_velocity` all passed runtime/checkpoint checks
  but did not solve fixed-forward tracking. The follow-up
  `--env_feet_slip_scale 0.0` gate also passed and confirmed
  `reward/feet_slip=0.0` in inherited eval after `442d297`, but `fwd1.0`
  tracking remained weak and `termination=-100` remained present.
  The latest termination/contact sweep now indicates those 100k failures are
  early torso fall/upright failures, not illegal contact or NaN.
- Menagerie: present at `1b86ece576591213e2b666ebf59508454200ca97`
- Python env: present under ignored `.venv`
- CUDA JAX: validated, backend `gpu`, device `cuda:0`
- GPU preflight: `PASS`
- Route B GPU 10k smoke: `PASS`
- Existing GPU 10k checkpoint deterministic eval readiness: `FAIL`, because
  the checkpoint predates observation normalizer persistence.
- Future checkpoint schema: patched to save `policy_normalizer` and
  `value_normalizer`.
- Schema dry-run checkpoint:
  `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- New GPU smoke checkpoint:
  `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- Deterministic eval smoke:
  `scripts/eval_sac_checkpoint.py` passed at 4 env x 200 steps with
  `EVAL_OK`; JSON is under ignored `./logs/sac_eval_smoke`.
- Route B GPU 50k sanity: `PASS`.
- 50k checkpoint:
  `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- 50k bounded deterministic eval:
  16 env x 1000 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_50k/eval_16x1000.json`, no action/reward/obs NaN.
- Route B GPU 100k sanity: `PASS`.
- 100k checkpoint:
  `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- 100k bounded deterministic eval:
  16 env x 1000 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_100k/eval_16x1000.json`, no action/reward/obs NaN.
- Route B GPU 250k sanity: `PASS`.
- 250k checkpoint:
  `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- 250k bounded deterministic eval:
  16 env x 1000 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_250k/eval_16x1000.json`, no action/reward/obs NaN.
- Route B GPU 500k sanity: `PASS`.
- 500k checkpoint:
  `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl` passed
  `scripts/check_sac_checkpoint.py --require_eval_ready`.
- 500k bounded deterministic eval:
  16 env x 1000 steps, `EVAL_OK`, JSON
  `./logs/sac_eval_500k/eval_16x1000.json`, no action/reward/obs NaN.
- Both-mode eval diagnostic:
  `scripts/eval_sac_checkpoint.py --policy_mode both` passed for 100k, 250k,
  and 500k checkpoints with seeds `0..4`. Deterministic `tanh(mean)` reward
  degraded from `-4.2218` to `-4.8476`, while sampled stochastic reward did
  not show the same degradation and improved from `-6.4616` to `-5.9091`.
- Full action distribution / reward-component diagnostic:
  `scripts/eval_sac_checkpoint.py --policy_mode both --action_diagnostics --reward_components`
  passed for 100k, 250k, and 500k checkpoints with seeds `0..4`.
  All 15 JSON outputs are present under ignored
  `./logs/sac_eval_action_diag_full/`. Deterministic degradation is tied to
  actor mean/action magnitude drift and component-specific penalties, not
  stochastic policy collapse.
- Fresh 100k train-time actor drift diagnostic:
  `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`, checkpoint
  readiness PASS, and 4 env x 200 action diagnostic eval `EVAL_OK`.
  Final actor mean abs `0.238568` exceeded interval avg `0.170754`, final
  deterministic action abs `0.218864` exceeded interval avg `0.163467`, and
  final log_std mean `-0.157116` was below interval avg `-0.136390`.
- Fresh 250k train-time actor drift diagnostic:
  `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`, checkpoint
  readiness PASS, and 4 env x 200 action diagnostic eval `EVAL_OK`.
  Actor mean abs increased from fresh 100k `0.238568 -> 0.299021`,
  deterministic action abs increased `0.218864 -> 0.270944`, final log_std
  moved down `-0.157116 -> -0.205270`, and alpha declined
  `0.032585 -> 0.018768`.
- Fresh 100k alpha/entropy ablation diagnostics:
  A1, A3, and A4 all returned `TRAIN_OK`, passed checkpoint readiness, and
  passed 4 env x 200 both-mode action diagnostic eval. A4 is the best current
  100k drift candidate with alpha `0.042848`, actor mean abs `0.206928`,
  deterministic action abs `0.193139`, log_std mean `-0.151827`, std mean
  `0.861342`, and the best deterministic 4x200 reward among A1/A3/A4.
  A4 stochastic 4x200 reward was worse than A1/A3, so this is a diagnostic
  signal rather than final policy-quality evidence.
- Fresh 100k alpha/entropy ablation multi-seed eval-only diagnostic:
  A1, A3, and A4 all passed checkpoint readiness and 5-seed both-mode eval
  with `15` JSON outputs in `./logs/sac_eval_alpha_ablate_multiseed/`.
  All evals returned `EVAL_OK` and all action/reward/obs NaN flags were false.
  A4 remained best on deterministic action magnitude (`0.1774`) and actor
  mean magnitude (`0.1920`), but A1 slightly edged deterministic reward
  (`-4.3262` vs A4 `-4.3436`), and A4 stochastic reward was worse than A1 by
  about `0.0945`.
- Bounded fresh 250k A4 alpha/entropy extension:
  `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`,
  checkpoint readiness PASS, 4 env x 200 eval PASS, and 5-seed 16 env x 1000
  eval PASS. Versus fresh 250k baseline, A4 improved alpha by `+0.01578`,
  actor mean abs by `-0.07330`, deterministic action abs by `-0.05875`,
  log_std by `+0.03434`, and std by `+0.02802`. A4 still drifted moderately
  from its own 100k result, and Q/target_q rose to `8.7442` / `8.7170`, so
  these remain watch items.
- Latest diagnostic report:
  `reports/sac_integration/12_alpha_entropy_ablation_plan.md`.
- Logs, checkpoints, `.venv`, and menagerie remain ignored and are not
  committed.

## 5. Completed CPU Validation

Windows CPU dev runtime:

- Python: `3.14.3`
- JAX: `0.10.0`
- JAX backend/devices: `cpu`, `cpu:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- CUDA tools: unavailable

Passing CPU commands:

- `.\.venv\Scripts\python.exe -m compileall g1_env learning scripts`
- `.\.venv\Scripts\python.exe -c "import g1_env; from g1_env import registry; print(registry.ALL_ENVS)"`
- `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickFlatTerrain`
- `.\.venv\Scripts\python.exe scripts\check_g1_env_api.py --env_name G1JoystickRoughTerrain`
- `.\.venv\Scripts\python.exe -m learning.train_jax_sac_brax --help`
- `.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --help`
- Route B dry run
- Route B CPU tiny smoke
- `scripts/check_sac_checkpoint.py`
- `scripts/gpu_preflight.py --impl jax` without `--require_gpu`

CPU tiny smoke result:

```text
status: TRAIN_OK
env_steps: 256
gradient_steps: 121
actor_loss: -1.6421515941619873
critic_loss: 0.02648034505546093
alpha: 0.04801943153142929
sps: 4.8985466406301095
checkpoint: ./logs/sac_lift_cpu_tiny/sac_lift_step_256.pkl
```

Checkpoint note:

- The CPU checkpoint is an ignored runtime artifact.
- It does not need to be migrated for GPU smoke.

## 6. Menagerie State

Required target path:

```text
g1_env/external_deps/mujoco_menagerie
```

Required source:

```text
https://github.com/deepmind/mujoco_menagerie.git
```

Required commit:

```text
1b86ece576591213e2b666ebf59508454200ca97
```

Windows CPU dev:

- Menagerie exists.
- Commit is pinned.
- Directory is ignored by git.

WSL2:

- Menagerie was missing after the initial clone.
- It must be cloned or copied before GPU preflight.
- Re-cloning inside WSL2 is preferred if network is available.

## 7. Migration Tools

Added scripts:

- `scripts/check_sac_checkpoint.py`
  - Validates Route B pickle checkpoints.
  - Reports `policy_normalizer`, `value_normalizer`, `normalize_observations`,
    and `deterministic_eval_ready`.
  - `--require_eval_ready` fails when a checkpoint cannot support trusted
    deterministic actor evaluation.
- `scripts/eval_sac_checkpoint.py`
  - Runs bounded actor eval from a Route B checkpoint.
  - Loads checkpoint config/params/normalizer, rebuilds the G1 env/network, and
    emits JSON metrics.
  - Supports `--policy_mode deterministic|stochastic|both`.
  - Does not train, update replay, render, or touch PPO/RSL paths.
- `scripts/gpu_preflight.py`
  - Does not train.
  - Checks JAX/MuJoCo/Brax, GPU visibility when requested, menagerie commit, registry, Route B import, and flat/rough env reset/step.
- `scripts/gpu_smoke_route_b.sh`
  - Linux/WSL2 10k GPU smoke wrapper.
- `scripts/gpu_smoke_route_b.ps1`
  - PowerShell GPU smoke wrapper with Windows-native caveat.

Added handoff:

- `WSL2_CODEX_HANDOFF.md`

Relevant reports:

- `reports/sac_integration/07_gpu_migration_prep.md`
- `reports/sac_integration/06_next_actions.md`
- `reports/sac_integration/04_smoke_results.md`
- `reports/sac_integration/03_route_b_lift_sac.md`

External local references for WSL2 agent:

- `WSL2_GPU_EXPERIENCE.md`
- `/home/admin/projects/mujoco_playground/TRAINING_NOTES.md`
- `/home/admin/projects/mujoco_playground/.md_edit/`

Use these files only as environment and operating-experience references. Do not let them override this project's SAC design or smoke ladder.

## 8. Next Immediate Objective

Current objective:

- Review and record the completed both-mode eval diagnostic.

Completed in WSL2:

- `uv sync --frozen --extra cuda`
- menagerie checkout at `1b86ece576591213e2b666ebf59508454200ca97`
- JAX backend/device check: `gpu`, `cuda:0`
- `uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu`: `PASS`
- `bash scripts/gpu_smoke_route_b.sh`: `TRAIN_OK`
- 50k sanity and bounded eval: `PASS`
- 100k sanity and bounded eval: `PASS`
- 250k sanity and bounded eval: `PASS`
- 500k sanity and bounded eval: `PASS`
- both-mode deterministic/stochastic eval diagnostic: `PASS`

## 9. GPU 10k Smoke Objective

Command wrapper:

```bash
uv run bash scripts/gpu_smoke_route_b.sh
```

Underlying Route B command:

```bash
python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 10000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 8192 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_10k
```

Acceptance criteria:

- JAX sees GPU/CUDA/ROCm.
- `gpu_preflight.py --require_gpu` passes.
- Flat env load/reset/step passes.
- Rough env load/reset/step passes.
- Route B reports `TRAIN_OK`.
- Checkpoint is saved under `./logs/sac_lift_gpu_10k`.
- Actor loss, critic loss, alpha, and SPS are finite and recorded.
- No NaN is reported.
- Actual `env_steps` are recorded; Route B may record `9984` for a `10000` target with `128` envs.
- Idle and final or peak VRAM are recorded when available.

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl
env_steps: 9984
gradient_steps: 142
wall_time: 56.50599093900382
sps: 176.68922947970893
actor_loss: -1.9058758020401
critic_loss: 0.08382290601730347
alpha: 0.04770537465810776
alpha_loss: 1.585930585861206
policy_log_prob: -18.79882049560547
q: 1.0673823356628418
target_q: 1.0646085739135742
truncation_fraction: 0.0
```

No NaN was observed in reported scalar metrics.

Actual step explanation:

- Route B uses `num_envs * (num_timesteps // num_envs)`.
- `128 * (10000 // 128) = 9984`.

Warnings observed and classified as non-fatal:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- CUDA timer warmup warning: `Delay kernel timed out`.

## 10. After GPU 10k Smoke

The existing GPU 10k checkpoint is useful as a smoke artifact but not as a
trusted deterministic-eval artifact:

- `normalize_observations=True`
- `policy_normalizer` missing
- `value_normalizer` missing

This cannot be fixed retroactively for the old checkpoint. A regenerated
normalizer-ready GPU checkpoint has now passed readiness and a small deterministic
eval smoke:

```text
checkpoint: ./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl
eval: 4 env x 200 steps
status: EVAL_OK
episode_reward_mean: -3.3628087043762207
done_fraction: 1.0
action/reward/obs NaN: false/false/false
json: ./logs/sac_eval_smoke/eval_4x200.json
```

After report review, only then consider:

- decision review before any bounded 3M continuation;
- diagnostic or regularization adjustment if prioritizing the 1M reward
  regression;
- PPO comparison after SAC policy-quality gates become meaningful.

Do not jump directly to 10M training. The 4x200 eval is a smoke, not a full
benchmark. The 50k, 100k, 250k, and 500k sanity runs are runtime/diagnostic
gates. The bounded 1024-env 1M R3 run now validates the high-parallel runtime
path, but reward regressed versus R3 250k and actor mean/std drift continued.

## 11. 50k Sanity Result

Command class:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 50000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 50000 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_50k_sanity
```

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl
env_steps: 49920
gradient_steps: 766
wall_time: 35.99766752999858
sps: 1386.756515777009
actor_loss: -3.6135072708129883
critic_loss: 0.07097882032394409
alpha: 0.03992176800966263
alpha_loss: 1.327394962310791
policy_log_prob: -18.81831169128418
policy_q: 2.8622469902038574
q: 2.884032726287842
target_q: 2.899707317352295
truncation_fraction: 0.0
```

Checkpoint readiness:

- `scripts/check_sac_checkpoint.py --require_eval_ready`: `PASS`
- `policy_normalizer`: present
- `value_normalizer`: present

Bounded deterministic eval:

```text
status: EVAL_OK
json: ./logs/sac_eval_50k/eval_16x1000.json
eval_env_steps: 16000
episode_reward_mean: -3.5016322135925293
episode_reward_std: 0.75983726978302
episode_reward_min: -6.096090316772461
episode_reward_max: -2.531925916671753
done_fraction: 1.0
wall_time: 74.76505397899746
sps: 214.00372431342888
action_nan: false
reward_nan: false
obs_nan: false
truncation_present: true
truncation_fraction: 0.0
```

Post-run GPU snapshot:

- GPU: RTX 4070 SUPER
- VRAM: `1602MiB / 12282MiB`
- Temperature: `56C`
- Power: `9W / 220W`
- GPU util: `8%`
- Process table only showed `/Xwayland`.

Known warnings:

- WSL2 CUDA driver passthrough warning: `Could not get kernel mode driver version`.
- JAX cast warning: `RuntimeWarning: overflow encountered in cast`.
- These remain non-fatal WSL2/JAX noise unless they accompany a failed command.

No NaN, Inf, OOM, CUDA, checkpoint, or eval error was observed.

## 12. 100k Sanity Result

Command class:

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
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_100k_sanity
```

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl
env_steps: 99968
gradient_steps: 1548
wall_time: 56.80434615799459
sps: 1759.8653406193746
actor_loss: -4.994826316833496
critic_loss: 0.04054964333772659
alpha: 0.03259027376770973
alpha_loss: 1.0562278032302856
policy_log_prob: -17.77903938293457
q: 4.3698601722717285
target_q: 4.456111907958984
truncation_fraction: 0.0
```

Checkpoint readiness:

- `scripts/check_sac_checkpoint.py --require_eval_ready`: `PASS`
- `policy_normalizer`: present
- `value_normalizer`: present

Bounded deterministic eval:

```text
status: EVAL_OK
json: ./logs/sac_eval_100k/eval_16x1000.json
eval_env_steps: 16000
episode_reward_mean: -3.894726037979126
episode_reward_std: 0.9610732197761536
episode_reward_min: -7.2897186279296875
episode_reward_max: -2.889821767807007
done_fraction: 1.0
wall_time: 66.85148939098872
sps: 239.33647770242092
action_nan: false
reward_nan: false
obs_nan: false
truncation_present: true
truncation_fraction: 0.0
```

Post-run GPU snapshot:

- GPU: RTX 4070 SUPER
- VRAM: `1508MiB / 12282MiB`
- Temperature: `55C`
- Power: `9W / 220W`
- GPU util: `14%`
- Process table only showed `/Xwayland`.

Known warnings and tooling notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- The first sandboxed `uv` attempt hit a `snap-confine` capability issue before
  training started. The identical command succeeded with external permission and
  unchanged parameters, so this is tooling noise, not a training failure.

No NaN, Inf, OOM, fatal CUDA, checkpoint, or eval error was observed.

## 13. 250k Sanity Result

Command class:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 250000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 250000 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_250k_sanity
```

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl
env_steps: 249984
gradient_steps: 3892
wall_time: 120.21957968600327
sps: 2079.3950590488107
actor_loss: -5.524118900299072
critic_loss: 0.02644157037138939
alpha: 0.018743595108389854
alpha_loss: 0.5869507789611816
policy_log_prob: -16.657032012939453
q: 5.197851181030273
target_q: 5.205532073974609
truncation_fraction: 0.0
```

Checkpoint readiness:

- `scripts/check_sac_checkpoint.py --require_eval_ready`: `PASS`
- `policy_normalizer`: present
- `value_normalizer`: present
- `deterministic_eval_ready`: `true`

Bounded deterministic eval:

```text
status: EVAL_OK
json: ./logs/sac_eval_250k/eval_16x1000.json
eval_env_steps: 16000
episode_reward_mean: -4.279743194580078
episode_reward_std: 1.2322009801864624
episode_reward_min: -8.849853515625
episode_reward_max: -3.093963384628296
done_fraction: 1.0
wall_time: 68.2435936529946
sps: 234.45424168833907
action_nan: false
reward_nan: false
obs_nan: false
truncation_present: true
truncation_fraction: 0.0
```

Known warnings and risk notes:

- No traceback, NaN, Inf, OOM, fatal CUDA, env, checkpoint, or eval failure was
  observed.
- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- Alpha dropped to about `0.0187`; Q and target Q rose to about `5.2` while
  critic loss stayed low; bounded eval reward did not improve versus 100k.
  This is not a failure, but 500k should watch alpha collapse, Q drift, critic
  loss, eval NaN flags, and reward trend.

## 14. 500k Sanity Result

Command class:

```bash
uv run --no-sync python -m learning.train_jax_sac_lift \
  --env_name G1JoystickFlatTerrain \
  --impl jax \
  --num_timesteps 500000 \
  --num_envs 128 \
  --num_eval_envs 32 \
  --batch_size 256 \
  --min_replay_size 1024 \
  --max_replay_size 500000 \
  --grad_updates_per_step 2 \
  --render False \
  --use_wandb False \
  --logdir ./logs/sac_lift_gpu_500k_sanity
```

Observed result:

```text
status: TRAIN_OK
checkpoint: ./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl
env_steps: 499968
gradient_steps: 7798
wall_time: 223.88994164399628
sps: 2233.0971919899416
actor_loss: -3.510934352874756
critic_loss: 0.04195608198642731
alpha: 0.008012857288122177
alpha_loss: 0.20854677259922028
policy_log_prob: -12.072959899902344
q: 3.348696231842041
target_q: 3.3187503814697266
truncation_fraction: 0.0
```

Checkpoint readiness:

- `scripts/check_sac_checkpoint.py --require_eval_ready`: `PASS`
- `policy_normalizer`: present
- `value_normalizer`: present
- `deterministic_eval_ready`: `true`

Bounded deterministic eval:

```text
status: EVAL_OK
json: ./logs/sac_eval_500k/eval_16x1000.json
eval_env_steps: 16000
episode_reward_mean: -4.691065788269043
episode_reward_std: 1.1929610967636108
episode_reward_min: -9.040802955627441
episode_reward_max: -3.7163496017456055
done_fraction: 1.0
wall_time: 68.70198891899781
sps: 232.88990976468807
action_nan: false
reward_nan: false
obs_nan: false
truncation_present: true
truncation_fraction: 0.0
```

Known warnings and risk notes:

- No traceback, NaN, Inf, OOM, fatal CUDA, env load/reset/step/shape, replay,
  checkpoint, or eval failure was observed.
- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- Alpha continued down from `0.0187436` at 250k to `0.0080129` at 500k.
- Q and target Q decreased from about `5.2` to about `3.3`; critic loss stayed
  finite/low.
- Bounded eval reward mean worsened from `-4.27974` to `-4.69107`; eval max
  also worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but alpha decline and eval degradation must
  block any automatic jump to 1M.

This older 500k signal has now been superseded by bounded high-parallel R3
work. The 1024-env 1M R3 run is runtime/checkpoint/eval PASS but still shows
policy-quality regression versus R3 250k, so a decision review is required
before any 3M continuation or regularization adjustment.

## 15. Fresh 100k Actor Drift Diagnostic

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl
checkpoint readiness: PASS
env_steps: 99968
gradient_steps: 1548
wall_time: 68.07941276300699
sps: 1468.4027952473857
alpha: 0.03258506953716278
log_alpha: -3.423901081085205
q: 4.1935601234436035
target_q: 4.180259704589844
```

Actor drift evidence:

```text
actor_policy_mean_abs_mean final / interval: 0.23856812715530396 / 0.1707537253543696
deterministic_action_abs_mean final / interval: 0.218863844871521 / 0.16346676852698475
actor_log_std_mean final / interval: -0.15711648762226105 / -0.13639042302196033
actor_policy_std_mean final / interval: 0.8573285341262817 / 0.8771794435281778
```

Small action diagnostic eval:

```text
json: ./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json
status: EVAL_OK
deterministic reward mean: -4.315369606018066
stochastic reward mean: -6.333320140838623
action/reward/obs NaN: false/false/false
```

Interpretation: fresh 100k supports the early actor mean drift hypothesis. This
is not a runtime failure.

## 16. Fresh 250k Actor Drift Diagnostic

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl
checkpoint readiness: PASS
env_steps: 249984
gradient_steps: 3892
wall_time: 140.85426465800265
sps: 1774.7705446261555
alpha: 0.01876842975616455
log_alpha: -3.975579023361206
q: 5.547477722167969
target_q: 5.520053863525391
```

Actor drift evidence:

```text
actor_policy_mean_abs_mean final / interval: 0.2990209758281708 / 0.233315885204818
deterministic_action_abs_mean final / interval: 0.2709442377090454 / 0.21555377979248855
actor_log_std_mean final / interval: -0.20526975393295288 / -0.16304866696410225
actor_policy_std_mean final / interval: 0.8162157535552979 / 0.8529925538726603
```

Fresh 100k comparison:

```text
actor mean abs final: 0.238568 -> 0.299021
deterministic action abs final: 0.218864 -> 0.270944
log_std final: -0.157116 -> -0.205270
alpha: 0.032585 -> 0.018768
```

Small action diagnostic eval:

```text
json: ./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json
status: EVAL_OK
deterministic reward mean: -4.01785135269165
stochastic reward mean: -5.697283744812012
action/reward/obs NaN: false/false/false
```

Interpretation: drift amplifies in absolute level by fresh 250k. This is not a
runtime failure. The A4 500k and 750k extensions have since completed; do not
run 1M automatically.

## 17. Bounded Fresh 500k A4 Alpha/Entropy Extension

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl
checkpoint readiness: PASS
env_steps: 499968
gradient_steps: 7798
wall_time: 271.9703
sps: 1838.3185
alpha: 0.02435
log_alpha: -3.71524
critic_loss: 0.0803
q: 8.47294
target_q: 8.41017
```

Actor drift evidence:

```text
actor_policy_mean_abs_mean final / interval: 0.29323 / 0.23339
deterministic_action_abs_mean final / interval: 0.26540 / 0.21651
actor_log_std_mean final / interval: -0.22279 / -0.17587
actor_policy_std_mean final / interval: 0.80245 / 0.84132
```

Eval evidence:

```text
4x200 JSON: ./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json
5-seed JSON dir: ./logs/sac_eval_alpha_ablate_500k_multiseed/
deterministic 5-seed reward avg: -4.4075
stochastic 5-seed reward avg: -6.0434
action/reward/obs NaN: false
```

Interpretation: A4 500k mitigates the old 500k deterministic drift pattern and
passes runtime/checkpoint/eval gates. The bounded A4 750k bridge has since run
and should be used for the current longer-run decision.

## 18. Bounded Fresh 750k A4 Alpha/Entropy Bridge

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl
checkpoint readiness: PASS
env_steps: 749952
gradient_steps: 11704
wall_time: 397.2631
sps: 1887.7966
alpha: 0.017246
log_alpha: -4.06016
critic_loss: 0.1441
q: 6.1746
target_q: 6.2776
```

Actor drift evidence:

```text
actor_policy_mean_abs_mean final / interval: 0.37190 / 0.27588
deterministic_action_abs_mean final / interval: 0.31850 / 0.24939
actor_log_std_mean final / interval: -0.24975 / -0.19484
actor_policy_std_mean final / interval: 0.78393 / 0.82621
```

Eval evidence:

```text
4x200 JSON: ./logs/sac_eval_alpha_ablate_750k/eval_A4_seed0_4x200_actiondiag.json
5-seed JSON dir: ./logs/sac_eval_alpha_ablate_750k_multiseed/
deterministic 5-seed reward avg: -5.7314
stochastic 5-seed reward avg: -6.3802
action/reward/obs NaN: false
```

Interpretation: A4 750k is runtime stable but not a clean stability
improvement. Actor mean/action drift increased, entropy/std narrowed, critic
loss rose, and deterministic plus stochastic 5-seed eval rewards worsened
versus A4 500k. This blocks any automatic 1M or longer run.

## 19. Fresh 100k Actor-Regularization R1

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl
checkpoint readiness: PASS
env_steps: 99968
gradient_steps: 1548
wall_time: 71.0452
sps: 1407.1039
alpha: 0.042852
log_alpha: -3.149996
critic_loss: 0.06915
q: 4.9464
target_q: 4.9560
```

Regularization:

```text
deterministic_action_l2_coef: 0.01
actor_mean_l2_coef: 0.001
actor_regularization_loss final / interval: 0.000886 / 0.000474
```

Eval evidence:

```text
4x200 JSON: ./logs/sac_eval_actor_reg_100k/eval_R1_seed0_4x200_actiondiag.json
5-seed JSON dir: ./logs/sac_eval_actor_reg_100k_multiseed/
deterministic 5-seed reward avg: -4.3587
stochastic 5-seed reward avg: -6.6210
action/reward/obs NaN: false
```

Interpretation: R1 is runtime clean, but it does not reduce train-time actor
mean or deterministic action magnitude versus A4 100k. The regularization
contribution is tiny relative to actor loss, so R1 should not be extended to
250k as-is.

## 20. Fresh 100k Actor-Regularization R2/R3 Coefficient Sweep

Status: `TRAIN_OK` for both R2 and R3.

```text
scope: fresh 100k coefficient sweep only
alpha settings: target_entropy_coef=0.25, alpha_learning_rate=1e-4
R2 coefs: deterministic_action_l2_coef=0.1, actor_mean_l2_coef=0.01
R3 coefs: deterministic_action_l2_coef=0.5, actor_mean_l2_coef=0.05
checkpoint readiness: PASS for both
4x200 action diagnostic eval: PASS for both
5-seed eval-only: PASS for both
multiseed JSON dir: ./logs/sac_eval_actor_reg_100k_multiseed/
multiseed JSON count: 15 total including R1/R2/R3
action/reward/obs NaN: false
```

Training summary:

| Variant | actor_loss | critic_loss | alpha | q | target_q | mean_abs | det_abs | log_std | std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | -5.9766 | 0.1383 | 0.042841 | 5.2078 | 5.1507 | 0.2012 | 0.1887 | -0.1513 | 0.8618 |
| R3 | -6.0585 | 0.1684 | 0.042818 | 5.3317 | 5.2599 | 0.1506 | 0.1430 | -0.1548 | 0.8589 |

5-seed eval summary:

| Variant | deterministic reward | stochastic reward | deterministic action abs | stochastic action abs |
|---|---:|---:|---:|---:|
| R2 | -4.2422 | -6.4680 | 0.1614 | 0.5247 |
| R3 | -4.1681 | -6.3983 | 0.1348 | 0.5198 |

Interpretation: R2 improves over A4 100k and R1 on drift metrics and eval
reward. R3 improves more strongly and has the best 5-seed deterministic and
stochastic rewards among R2/R3/A4 100k/R1. R3 is the stronger 100k candidate,
but its higher critic loss versus R2 is a watch item. Next step is a decision
review before any bounded R3 250k extension; do not run 500k, 750k, or 1M from
this result.

## 21. Bounded R3 250k Actor-Regularization Extension

Status: `TRAIN_OK`

```text
checkpoint: ./logs/sac_lift_gpu_250k_actor_reg_te0p25_alr1e4_l2_0p5_mean_0p05_s1/sac_lift_step_249984.pkl
checkpoint readiness: PASS
env_steps: 249984
gradient_steps: 3892
wall_time: 145.91901159299596
sps: 1713.1694991004113
alpha: 0.03449748829007149
log_alpha: -3.366868734359741
critic_loss: 0.053617656230926514
q: 8.382803916931152
target_q: 8.41272258758545
```

Actor and regularization evidence:

```text
actor_policy_mean_abs_mean final / interval: 0.1630484462 / 0.1431587681
deterministic_action_abs_mean final / interval: 0.155556202 / 0.136790473
actor_log_std_mean final / interval: -0.170873329 / -0.152855622
actor_policy_std_mean final / interval: 0.844428778 / 0.861274787
actor_regularization_loss final / interval: 0.024142943 / 0.020294007
```

Eval evidence:

```text
4x200 JSON: ./logs/sac_eval_actor_reg_250k/eval_R3_seed0_4x200_actiondiag.json
5-seed JSON dir: ./logs/sac_eval_actor_reg_250k_multiseed/
deterministic 5-seed reward avg: -3.7941
stochastic 5-seed reward avg: -5.9802
action/reward/obs NaN: false
```

Interpretation: R3 retains drift control at 250k. Versus R3 100k, actor mean
and deterministic action magnitude rose only moderately, deterministic and
stochastic eval improved, and critic loss improved from `0.1684` to `0.0536`.
Versus A4 250k, R3 has lower action magnitude and better eval. This supports
R3 as the current stability candidate, but it is not a final policy-quality
claim.

## 22. High-Parallel Capacity Plan

The 128-env ladder is now classified as runtime/diagnostic evidence, not a
policy-quality benchmark for G1. The high-parallel capacity benchmark was
planned to coordinate off-policy update pressure:

| num_envs | batch_size | grad_updates_per_step | Replay cap | Logdir |
|---:|---:|---:|---:|---|
| 512 | 256 | 8 | 262144 | `./logs/sac_capacity_env512_65k_r3_b256_g8_replay262k` |
| 1024 | 256 | 16 | 262144 | `./logs/sac_capacity_env1024_65k_r3_b256_g16_replay262k` |
| 2048 | 256 | 32 | 262144 | `./logs/sac_capacity_env2048_65k_r3_b256_g32_replay262k` |

This preserves approximate sampled UTD near the historical 128-env baseline:

```text
sample UTD ~= grad_updates_per_step * batch_size / num_envs ~= 4
```

For multi-million runs, do not use `max_replay_size=num_timesteps` by default.
At about 2684 raw bytes per transition, 5M replay is already about 13.42GB
decimal / 12.50GiB raw and 10M replay is about 26.84GB / 25.00GiB before
overhead. Long runs should begin with about a 1M replay cap, increasing only
after stress evidence.

## 23. High-Parallel Capacity Results

The 512/1024/2048 env capacity benchmark is complete. All three cases returned
`TRAIN_OK`, wrote checkpoints, and passed checkpoint readiness. Sample UTD was
kept near `4.0` by scaling `grad_updates_per_step=8/16/32`.

| num_envs | grad_updates_per_step | SPS | critic_loss | alpha | q | target_q | Result |
|---:|---:|---:|---:|---:|---:|---:|---|
| 512 | 8 | 773.12 | 0.1166 | 0.04608 | 3.1799 | 3.1973 | PASS |
| 1024 | 16 | 953.36 | 0.1217 | 0.04605 | 3.4970 | 3.4905 | PASS |
| 2048 | 32 | 826.26 | 0.0706 | 0.04597 | 3.7519 | 3.7731 | PASS |

`1024` envs was selected for the bounded 1M follow-up. `2048` is feasible but
slower in this benchmark; `512` is stable but slower. Full details are recorded
in `reports/sac_integration/14_high_parallel_capacity_results.md`.

## 24. Bounded 1024-Env 1M R3 Result

The bounded 1024-env 1M R3 run passed runtime/checkpoint/eval gates:

- checkpoint:
  `./logs/sac_lift_gpu_1m_env1024_r3_b256_g16_replay1m/sac_lift_step_999424.pkl`
- actual env steps: `999424`
- gradient steps: `15376`
- SPS: `3209.6723`
- checkpoint readiness: PASS
- five both-mode eval JSONs under `./logs/sac_eval_env1024_1m_r3_multiseed/`
  all returned `EVAL_OK` with no action/reward/obs NaN flags
- peak training memory: about `9838MiB / 12282MiB`

Interpretation: runtime stability supports `1024 envs + replay1M + R3 +
UTD~4`, but this is not a policy-quality breakthrough. Versus R3 250k,
deterministic reward worsened `-3.7941 -> -4.9891`, stochastic reward worsened
`-5.9802 -> -6.7546`, actor mean abs rose `0.1630 -> 0.2299`, deterministic
action abs rose `0.1556 -> 0.2046`, and log_std narrowed
`-0.1709 -> -0.2881`.

## 25. Fixed-Command Eval And Alpha Sign Gate

Fixed-command eval support is now available in `scripts/eval_sac_checkpoint.py`
through `--fixed_command`, `--command_x`, `--command_y`, and `--command_yaw`.
The 3M R3 `[0.5,0,0]` smoke returned `EVAL_OK`, deterministic reward mean
`0.5099`, stochastic reward mean `-2.3354`, and no action/reward/obs NaN
flags. A default deterministic compatibility smoke without fixed command also
returned `EVAL_OK`.

Alpha sign audit result: `SIGN_OK_BUT_COLLAPSE_RISK`. No direct sign bug was
found versus Brax-style SAC, but persistent downward alpha pressure remains a
likely risk because the observed log-probability range keeps
`-log_prob - target_entropy` positive and there is no alpha floor.

## 26. Fresh Env1024 R3 100k Fixed-Alpha Diagnostic

Status: `TRAIN_OK`.

- Variant: `fixed_alpha=0.03`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p03/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.
- `env_steps`: `99328`.
- `gradient_steps`: `1312`.
- `wall_time`: `46.3008s`.
- `sps`: `2145.2764`.
- `actor_loss`: `-3.45169`.
- `critic_loss`: `0.114333`.
- `q / target_q`: `2.87063 / 2.87375`.
- `reward_mean`: `-0.134369`.
- `done_fraction`: `0.0234375`.
- `discount_mean`: `0.976563`.

Alpha behavior:

- `alpha_raw=0.0497871`.
- `log_alpha_raw=-3.0`.
- `alpha_effective=0.03`.
- `log_alpha_effective=-3.50656`.
- `alpha_loss=1.29717`.
- `alpha_log_prob=-18.8043`.
- `alpha_grad_proxy_exp=1.29717`.
- `alpha_grad_proxy_log=26.0543`.

Fixed-alpha mechanics worked: effective alpha stayed fixed at `0.03`, and raw
`log_alpha` stayed at init `-3.0`.

Small fixed-command smoke was weak:

- `fwd0.5` deterministic reward `-3.9345`, action abs `0.1268`,
  `tracking_lin_vel=8.0171`, `termination=-100`.
- `fwd1.0` deterministic reward `-3.9779`, action abs `0.1255`,
  `tracking_lin_vel=1.8457`, `termination=-100`.
- Stochastic smokes were also negative (`-5.9812` and `-5.9239`) with sampled
  action magnitude around `0.519`.

Interpretation: runtime/checkpoint are fine, but `fixed_alpha=0.03` does not
solve forward tracking at 100k. Do not run 250k, 5M, or 10M from this result.
Next short diagnostic should be fresh env1024 R3 100k `fixed_alpha=0.05`, with
`alpha_floor=0.03` as the next alternative.

## 27. Fresh Env1024 R3 100k Fixed-Alpha 0.05 Diagnostic

Status: `TRAIN_OK`.

- Variant: `fixed_alpha=0.05`.
- Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p05/sac_lift_step_99328.pkl`
- Checkpoint readiness: PASS.
- `env_steps`: `99328`.
- `gradient_steps`: `1312`.
- `wall_time`: `46.3739s`.
- `sps`: `2141.8925`.
- `actor_loss`: `-6.27606`.
- `critic_loss`: `0.344604`.
- `q / target_q`: `5.32498 / 5.24534`.
- `reward_mean`: `-0.175746`.
- `done_fraction`: `0.0429688`.
- `discount_mean`: `0.957031`.

Alpha behavior:

- `alpha_raw=0.0497871`.
- `log_alpha_raw=-3.0`.
- `alpha_effective=0.05`.
- `log_alpha_effective=-2.99573`.
- `alpha_loss=1.29048`.
- `alpha_log_prob=-18.6699`.
- `alpha_grad_proxy_exp=1.29048`.
- `alpha_grad_proxy_log=25.9199`.

Fixed-alpha mechanics worked: effective alpha stayed fixed at `0.05`, raw
`log_alpha` stayed at init `-3.0`, and `alpha_floor` stayed inactive.

Small fixed-command smoke was still weak:

- `fwd0.5` deterministic reward `-3.7355`, action abs `0.1138`,
  `tracking_lin_vel=8.7119`, `termination=-100`.
- `fwd1.0` deterministic reward `-3.8258`, action abs `0.1117`,
  `tracking_lin_vel=2.6894`, `termination=-100`.
- Stochastic smokes were also negative (`-5.9704` and `-6.1849`) with sampled
  action magnitude around `0.519`.

Interpretation: runtime/checkpoint are fine, but `fixed_alpha=0.05` does not
solve forward tracking at 100k and does not improve the fixed-forward smoke
versus `fixed_alpha=0.03`. The critic loss watch item is stronger
(`0.3446`). Do not extend fixed-alpha variants to 250k, 5M, or 10M from this
result. Shift the main route toward reward/prior targeted audit or ablation;
keep `alpha_floor=0.03` only as a secondary diagnostic.

## 28. Current Position In One Sentence

SAC Route B is implemented; CPU tiny smoke, WSL2 GPU preflight, Route B GPU 10k smoke, normalizer-ready checkpoint validation, bounded deterministic eval smoke, 50k sanity/eval, 100k sanity/eval, 250k sanity/eval, 500k sanity/eval, both-mode eval diagnostic, full action diagnostic, fresh 100k/250k train-time actor drift diagnostics, fresh 100k alpha/entropy ablation diagnostics, the A1/A3/A4 multi-seed eval-only ablation diagnostic, bounded fresh 250k/500k/750k A4 extensions, fresh 100k actor-regularization R1, fresh 100k actor-regularization R2/R3, bounded R3 250k, the 512/1024/2048 high-parallel capacity benchmark, bounded 1024-env 1M R3, bounded 1024-env 3M R3, fixed-command render support, fixed-command eval support, alpha sign audit, fixed-command forward eval gate, fresh env1024 R3 100k `fixed_alpha=0.03`/`0.05` diagnostics, fresh env1024 R3 100k `foot_velocity` feet-slip diagnostic, fresh env1024 R3 100k `feet_slip_scale=0` diagnostic, fresh env1024 R3 100k push-disable diagnostic, fresh env1024 R3 100k zero-command phase-freeze diagnostic, and fresh env1024 R3 100k feet-air-time command-mask diagnostic have passed their bounded runtime gates; 3M provides the first strong deterministic-policy improvement signal but also severe alpha/std collapse, stochastic degradation, and weak `[1,0,0]` forward tracking; fixed-alpha, feet-slip, push-disable, phase-freeze, and feet-air-time command-mask short gates did not solve 100k fixed-forward/stand termination; 5M/10M, full eval benchmarking, domain randomization, and fine-tuning remain `NOT VALIDATED`.
