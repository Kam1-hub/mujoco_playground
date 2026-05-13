# G1 SAC Migration Handoff

Date: 2026-05-13

This is the canonical handoff for moving the G1 SAC integration project to a
new machine. It is intended for a new agent or engineer starting from a clean
clone where `mujoco_menagerie` has not been cloned yet.

The document is self-contained on purpose. Runtime logs and checkpoints are not
tracked by git, so this handoff separates repository state from optional local
artifacts that may or may not exist after migration.

## 1. Source Of Truth

- Repository: `https://github.com/Kam1-hub/mujoco_playground.git`
- Branch: `sac-integration`
- Latest project-state commit before this migration handoff:
  `d094a4a Record SAC fresh 250k actor diagnostic`
- Full commit:
  `d094a4a37660bdcde35ee96be89552dee6ca703c`
- Current branch head may be newer than `d094a4a` because this handoff document
  is committed after the fresh 250k actor diagnostic. Treat `d094a4a` as the
  validated project-state baseline, not as the maximum expected HEAD.
- Handoff file:
  `MIGRATION_HANDOFF.md` in the repository root.
- Canonical WSL2 path used during validation:
  `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Historical Windows path:
  `D:\mujoco_playground\g1_sac_dev`

If a local clone is behind `origin/sac-integration`, do not trust its reports as
the latest source. Fetch and fast-forward first.

Recommended initial checks:

```bash
git fetch origin sac-integration
git checkout sac-integration
git pull --ff-only origin sac-integration
git status --short --branch
git log --oneline -8
git rev-parse HEAD
git rev-parse origin/sac-integration
```

Expected project-state sequence should include these commits below any newer
handoff-only commits:

```text
d094a4a Record SAC fresh 250k actor diagnostic
4ca55f8 Record SAC fresh actor drift diagnostic
202c6a9 Add SAC actor drift train diagnostics
2f60809 Record SAC action diagnostic findings
8ff4f1d Add SAC action distribution eval diagnostics
9cb5112 Record SAC both-mode eval diagnostics
926a14f Add SAC alpha entropy diagnostics
87bca63 Record SAC 500k sanity results
```

## 2. Project Mission

The project goal is to add a local SAC baseline for the G1 MuJoCo Playground
pipeline while preserving the existing PPO and RSL-RL workflows.

Target pipeline:

```text
G1 pipeline
  - PPO baseline: train-g1-jax
  - RSL-RL baseline: train-g1-rsl
  - SAC fallback/minimal baseline: train-g1-sac-brax
  - SAC main baseline: train-g1-sac
```

Current main route:

- Route B LIFT-style asymmetric SAC.
- Actor observation: `obs["state"]`.
- Critic/value observation: `obs["privileged_state"]`.
- Main environment: `G1JoystickFlatTerrain`.
- Rough terrain env readiness has also been checked.

Out of scope for the current phase:

- world model
- vision SAC
- fine-tuning
- real deployment
- domain randomization
- reward tuning
- `action_scale` tuning
- Kp/control-gain tuning
- PPO/RSL behavior changes

## 3. Repository Layout

Primary implementation paths:

```text
g1_env/                         G1 env package and configs
learning/train_jax_sac_lift.py  Route B train-g1-sac CLI entry point
learning/train_jax_sac_brax.py  Route A Brax fallback entry point
learning/sac_lift/              Route B SAC implementation
scripts/                        validation, checkpoint, eval, GPU scripts
reports/sac_integration/        integration reports and validation records
```

Important Route B files:

```text
learning/sac_lift/config.py
learning/sac_lift/train.py
learning/sac_lift/losses.py
learning/sac_lift/networks.py
learning/sac_lift/distributions.py
learning/sac_lift/replay_buffer.py
learning/sac_lift/normalizer.py
learning/sac_lift/checkpoint.py
scripts/check_sac_checkpoint.py
scripts/eval_sac_checkpoint.py
g1_env/config/sac_params.py
```

Important reports:

```text
reports/sac_integration/03_route_b_lift_sac.md
reports/sac_integration/04_smoke_results.md
reports/sac_integration/05_known_issues.md
reports/sac_integration/06_next_actions.md
reports/sac_integration/09_phase_summary_and_risks.md
reports/sac_integration/10_both_mode_eval_diagnostic.md
reports/sac_integration/10_action_distribution_diagnostics.md
reports/sac_integration/11_actor_drift_train_diagnostic.md
```

Read these first on a new machine:

```text
AGENTS.md
MIGRATION_HANDOFF.md
NEXT_AGENT_HANDOFF.md
reports/sac_integration/11_actor_drift_train_diagnostic.md
reports/sac_integration/10_action_distribution_diagnostics.md
reports/sac_integration/10_both_mode_eval_diagnostic.md
reports/sac_integration/09_phase_summary_and_risks.md
reports/sac_integration/06_next_actions.md
reports/sac_integration/05_known_issues.md
reports/sac_integration/04_smoke_results.md
```

## 4. New Machine Migration Steps

The new machine does not have `mujoco_menagerie` yet. Clone it separately; do
not vendor it into this repository.

### 4.1 Clone Repository

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Kam1-hub/mujoco_playground.git
cd mujoco_playground/g1_sac_dev
git checkout sac-integration
git pull --ff-only origin sac-integration
```

Verify:

```bash
git status --short --branch
git log --oneline -5
git rev-parse HEAD
```

Expected HEAD should be `d094a4a` or newer. If `MIGRATION_HANDOFF.md` exists in
the repository root, the clone is on a post-handoff commit and this is expected.

### 4.2 Recreate Python Environment

Do not copy `.venv` from another machine.

```bash
uv sync --frozen --extra cuda
```

Expected previously validated stack:

- Python: `3.12.3`
- JAX: `0.10.0`
- JAX backend: `gpu`
- JAX device: `cuda:0`
- MuJoCo: `3.8.0`
- Brax: `0.14.2`
- GPU used in validation: NVIDIA RTX 4070 SUPER, 12GB
- `nvcc`: absent in the validation environment; not a blocker when JAX sees
  CUDA through the JAX CUDA plugin.

Runtime check:

```bash
uv run --no-sync python -c "import jax, mujoco, brax; print('jax', jax.__version__, jax.default_backend(), jax.devices()); print('mujoco', mujoco.__version__); print('brax', brax.__version__)"
```

### 4.3 Clone MuJoCo Menagerie

Required ignored path:

```text
g1_env/external_deps/mujoco_menagerie
```

Commands:

```bash
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97
git -C g1_env/external_deps/mujoco_menagerie rev-parse HEAD
```

Required menagerie commit:

```text
1b86ece576591213e2b666ebf59508454200ca97
```

Check ignore rules:

```bash
git check-ignore -v logs .venv g1_env/external_deps/mujoco_menagerie || true
```

### 4.4 Required Environment Variables

Set these before GPU/JAX commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

### 4.5 Preflight

Only after the Python env and menagerie are ready:

```bash
nvidia-smi
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Stop if JAX reports CPU only. GPU validation requires JAX to report a GPU/CUDA
backend.

## 5. Files And Artifacts Not To Commit

Do not commit:

```text
.venv/
logs/
checkpoints
g1_env/external_deps/mujoco_menagerie/
generated caches
research_agent_packet/
research_agent_return/
local migration export folders
```

Runtime logs/checkpoints are optional local artifacts. They are useful for
continuing diagnostics, but they are not repository state. A new machine may not
have them unless they are copied separately.

## 6. SAC Route B Design

Route B is the current main SAC baseline.

Key design choices:

- `train-g1-sac` entry point calls `learning.train_jax_sac_lift:run`.
- Actor obs key defaults to `state`.
- Critic/value obs key defaults to `privileged_state`.
- Runtime observation shapes:
  - `state`: `(103,)`
  - `privileged_state`: `(216,)`
- Action size: `29`.
- Actor: tanh Gaussian policy.
- Critic: twin Q networks.
- Target critic: Polyak update with `tau`.
- Entropy temperature: trainable `log_alpha`.
- Replay: uniform replay buffer.
- Observation normalization: enabled.
- Checkpoints include normalizers.
- G1 env applies action scale internally as `default_pose + action *
  action_scale`; SAC does not apply an additional external action scale.
- Native `state.info["truncation"]` is absent in G1 env checks; Route B
  synthesizes zero truncation and records `truncation_fraction`.

Current Route B config defaults in `g1_env/config/sac_params.py`:

```text
actor_learning_rate = 1e-4
critic_learning_rate = 1e-4
alpha_learning_rate = 3e-4
discounting = 0.99
reward_scaling = 1.0
tau = 0.005
target_entropy_coef = 0.5
init_log_alpha = -3.0
num_envs = 128
num_eval_envs = 32
batch_size = 256
min_replay_size = 1024
grad_updates_per_step = 2
policy_hidden_layer_sizes = (512, 256, 128)
q_hidden_layer_sizes = (1024, 512, 256)
activation = swish
policy_obs_key = state
value_obs_key = privileged_state
```

Parameter origin:

- Network shape and asymmetric SAC structure were derived from the local
  LIFT-humanoid report references.
- `target_entropy_coef=0.5` and the exp-alpha loss family match Brax-style SAC
  behavior inspected during research.
- Throughput parameters were kept conservative for G1 smoke validation rather
  than copied wholesale from LIFT high-throughput settings.

## 7. Validation Ladder

Validated:

- CPU tiny smoke: PASS.
- WSL2 CUDA/JAX preflight: PASS.
- GPU 10k sanity: PASS.
- GPU 50k sanity: PASS.
- GPU 100k sanity: PASS.
- GPU 250k sanity: PASS.
- GPU 500k sanity: PASS.
- Checkpoint readiness: PASS for normalizer-ready checkpoints.
- Bounded deterministic eval: PASS.
- Both-mode deterministic/stochastic eval diagnostic: PASS.
- Action distribution and reward-component eval diagnostic: PASS.
- Fresh 100k train-time actor drift diagnostic: PASS.
- Fresh 250k train-time actor drift diagnostic: PASS.

Not validated:

- 750k / 1M status: NOT VALIDATED.
- fresh 500k actor drift diagnostic.
- 750k training.
- 1M training.
- PPO comparison.
- full performance benchmark.
- domain randomization.
- fine-tuning.
- reward/action scale/Kp tuning.

This project has moved past "can SAC run?" and is now in the stage of deciding
how to handle deterministic actor mean drift.

## 8. Key Historical Milestones

Important commits:

```text
d094a4a Record SAC fresh 250k actor diagnostic
4ca55f8 Record SAC fresh actor drift diagnostic
202c6a9 Add SAC actor drift train diagnostics
2f60809 Record SAC action diagnostic findings
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
d5c0e8d Record Route B GPU smoke results
6fa5160 Sync WSL2 GPU preflight documentation
d73d45c Prepare GPU migration validation scripts
```

High-level progression:

1. Implemented Route A Brax fallback and Route B SAC baseline.
2. Validated CPU tiny smoke.
3. Migrated to WSL2/CUDA and passed GPU preflight.
4. Passed 10k, 50k, 100k, 250k, and 500k GPU sanity runs.
5. Added checkpoint normalizer persistence.
6. Added deterministic eval.
7. Added alpha/entropy diagnostics after observing alpha decline.
8. Added deterministic/stochastic both-mode eval.
9. Added action distribution and reward-component diagnostics.
10. Added train-time actor drift diagnostics.
11. Ran fresh 100k and fresh 250k actor drift diagnostic runs.

## 9. Key Results And Metrics

### GPU 100k Sanity

- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- `TRAIN_OK`
- `env_steps`: `99968`
- `gradient_steps`: `1548`
- `wall_time`: `56.80434615799459`
- `sps`: `1759.8653406193746`
- `actor_loss`: `-4.994826316833496`
- `critic_loss`: `0.04054964333772659`
- `alpha`: `0.03259027376770973`
- `q`: `4.3698601722717285`
- `target_q`: `4.456111907958984`
- Checkpoint readiness: PASS.
- Bounded eval reward mean: `-3.894726037979126`.

### GPU 250k Sanity

- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- `TRAIN_OK`
- `env_steps`: `249984`
- `gradient_steps`: `3892`
- `wall_time`: `120.21957968600327`
- `sps`: `2079.3950590488107`
- `actor_loss`: `-5.524118900299072`
- `critic_loss`: `0.02644157037138939`
- `alpha`: `0.018743595108389854`
- `q`: `5.197851181030273`
- `target_q`: `5.205532073974609`
- Checkpoint readiness: PASS.
- Bounded eval reward mean: `-4.279743194580078`.

### GPU 500k Sanity

- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- `TRAIN_OK`
- `env_steps`: `499968`
- `gradient_steps`: `7798`
- `wall_time`: `223.88994164399628`
- `sps`: `2233.0971919899416`
- `actor_loss`: `-3.510934352874756`
- `critic_loss`: `0.04195608198642731`
- `alpha`: `0.008012857288122177`
- `q`: `3.348696231842041`
- `target_q`: `3.3187503814697266`
- Checkpoint readiness: PASS.
- Bounded eval reward mean: `-4.691065788269043`.

### Both-Mode Eval Diagnostic

Scope:

- checkpoints: 100k, 250k, 500k.
- seeds: `0..4`.
- eval scale: `num_eval_envs=16`, `episode_length=1000`.
- command class: `scripts/eval_sac_checkpoint.py --policy_mode both`.

Deterministic `tanh(mean)` path:

```text
reward mean aggregate: -4.2218 -> -4.4585 -> -4.8476
action abs mean:       0.1823 -> 0.2148 -> 0.3029
```

Stochastic sampled path:

```text
reward mean aggregate: -6.4616 -> -6.1954 -> -5.9091
log-prob mean:         -17.9594 -> -17.2847 -> -13.4275
```

Conclusion:

- deterministic `tanh(mean)` behavior degrades.
- stochastic sampled behavior does not show the same degradation.
- stochastic reward remains worse in absolute terms at each checkpoint, but its
  trend improves.

### Action Distribution Diagnostic

Scope:

- checkpoints: 100k, 250k, 500k.
- seeds: `0..4`.
- command class:
  `scripts/eval_sac_checkpoint.py --policy_mode both --action_diagnostics --reward_components`.

Deterministic aggregates:

```text
reward avg:   -4.2130 -> -4.4792 -> -4.8204
action abs:    0.1823 ->  0.2147 ->  0.3029
mean abs:      0.1950 ->  0.2288 ->  0.3468
log_std mean: -0.1077 -> -0.1415 -> -0.2909
std mean:      0.8996 ->  0.8692 ->  0.7519
```

Stochastic aggregates:

```text
reward avg:   -6.4741 -> -6.2374 -> -5.8911
action abs:    0.5274 ->  0.5236 ->  0.5221
mean abs:      0.2240 ->  0.2712 ->  0.3954
log_std mean: -0.1562 -> -0.1939 -> -0.3288
std mean:      0.8578 ->  0.8254 ->  0.7256
```

Reward components implicated in deterministic degradation:

```text
reward/ang_vel_xy:  -50.82 -> -62.66 -> -73.31
reward/stand_still: -18.72 -> -21.72 -> -28.33
reward/orientation: -34.80 -> -43.27 -> -40.37
```

Conclusion:

- deterministic degradation is tied to actor mean/action magnitude drift.
- policy std narrows as training progresses.
- reward degradation is concentrated in angular velocity, stand-still, and
  orientation terms.

### Fresh 100k Actor Drift Diagnostic

Scope:

- fresh diagnostic training run with train-time actor drift metrics.
- no 250k/750k/1M in that checkpoint.
- checkpoint:
  `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`.
- small eval:
  `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`.

Training metrics:

```text
alpha:                                0.03258506953716278
log_alpha:                           -3.423901081085205
actor_policy_mean_abs_mean final:     0.23856812715530396
actor_policy_mean_abs_mean interval:  0.1707537253543696
deterministic_action_abs_mean final:  0.218863844871521
deterministic_action_abs_mean interval: 0.16346676852698475
actor_log_std_mean final:            -0.15711648762226105
actor_log_std_mean interval:         -0.13639042302196033
actor_policy_std_mean final:          0.8573285341262817
actor_policy_std_mean interval:       0.8771794435281778
```

Conclusion:

- actor mean drift is already visible by fresh 100k.
- final mean/action magnitudes exceed interval averages.
- final log_std/std are lower than interval averages.

### Fresh 250k Actor Drift Diagnostic

Scope:

- fresh diagnostic training run to compare with fresh 100k.
- no fresh 500k/750k/1M in that checkpoint.
- checkpoint:
  `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`.
- small eval:
  `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`.

Training metrics:

```text
alpha:                                0.01876842975616455
log_alpha:                           -3.975579023361206
actor_policy_mean_abs_mean final:     0.2990209758281708
actor_policy_mean_abs_mean interval:  0.233315885204818
deterministic_action_abs_mean final:  0.2709442377090454
deterministic_action_abs_mean interval: 0.21555377979248855
actor_log_std_mean final:            -0.20526975393295288
actor_log_std_mean interval:         -0.16304866696410225
actor_policy_std_mean final:          0.8162157535552979
actor_policy_std_mean interval:       0.8529925538726603
```

Comparison with fresh 100k:

```text
alpha:                         0.032585 -> 0.018768
actor mean abs final:          0.238568 -> 0.299021
deterministic action abs final: 0.218864 -> 0.270944
log_std mean final:           -0.157116 -> -0.205270
std mean final:                0.857329 -> 0.816216
```

Conclusion:

- actor mean/action magnitude drift amplifies in absolute level by fresh 250k.
- std/log_std and alpha continue downward.
- this is not a runtime failure.

## 10. Current Technical Diagnosis

The project is not currently blocked by integration or runtime issues.

What is working:

- env load/reset/step.
- SAC init.
- replay.
- updates.
- checkpoint save/load.
- normalizer persistence.
- deterministic eval.
- stochastic eval.
- action and reward diagnostics.
- WSL2/CUDA execution.

What is not resolved:

- deterministic `tanh(mean)` path degrades as training progresses.
- actor mean and deterministic action magnitude grow.
- log_std/std and alpha decline.
- stochastic sampled policy does not collapse in the same way.
- reward degradation is concentrated in angular velocity, orientation, and
  stand-still components.

Interpretation:

SAC trains a stochastic policy distribution through sampled actions. The
deterministic deployment path `tanh(mean)` is not directly optimized as a
separate objective. As alpha decreases and policy std narrows, actor mean/action
magnitude drift appears to harm the deterministic path, especially torso
stability-related rewards.

This is not yet proven to require hyperparameter changes. It does mean that
running 750k or 1M without a decision review would be under-justified.

## 11. Current Open Decision

Do not automatically run fresh 500k, 750k, or 1M.

Next step should be a decision review comparing these routes:

1. **Fresh 500k diagnostic**
   - Completes fresh 100k/250k/500k actor drift trajectory.
   - Costs GPU time and may only confirm the known drift.

2. **Alpha/entropy ablation design**
   - Reviews `target_entropy_coef`, `alpha_learning_rate`, `init_log_alpha`,
     and possibly alpha objective variants.
   - Must be designed as an ablation, not slipped into the baseline.

3. **Deterministic actor / eval-policy design**
   - Directly targets deterministic `tanh(mean)` deployment path.
   - Would be an algorithmic change and must not be mixed into the baseline
     without explicit decision.

4. **Action/reward component targeted analysis**
   - Further explains which action dims/joints and reward terms drive the
     drift.
   - Diagnosis only, not a fix by itself.

## 12. Stop Conditions

Stop and ask before:

- installing dependencies or downloading assets without authorization.
- writing outside the project workspace.
- committing `.venv`, logs, checkpoints, or menagerie.
- modifying reward, `action_scale`, Kp, PPO, RSL, env XML/assets, or env core
  behavior.
- running fresh 500k, 750k, or 1M without an explicit user-approved plan.
- treating deterministic drift as solved without evidence.

Stop during any future training/eval if:

- JAX backend is not GPU/CUDA for GPU validation.
- `nvidia-smi` is unavailable when GPU validation is required.
- `TRAIN_OK` is missing.
- checkpoint is missing.
- `--require_eval_ready` fails.
- eval is not `EVAL_OK`.
- `action_nan`, `reward_nan`, or `obs_nan` is true.
- actor/critic/alpha/Q/target-Q metrics contain NaN/Inf.
- OOM or fatal CUDA error occurs.
- env load/reset/step fails.
- replay/update/checkpoint save/load fails.
- logs/checkpoints/.venv/menagerie appear as unignored git files.

## 13. Startup Prompt For New Agent

Copy this into the new agent after cloning the repo and preparing the workspace:

```text
You are in the migrated G1 SAC workspace. Work in the repository root.

First, do only read-only startup checks. Do not train, eval, preflight, install,
download, modify files, or commit yet.

Run:

pwd
git fetch origin sac-integration
git status --short --branch
git log --oneline -8
git rev-parse HEAD
git rev-parse origin/sac-integration
git check-ignore -v logs .venv g1_env/external_deps/mujoco_menagerie || true
test -d g1_env/external_deps/mujoco_menagerie && echo MENAGERIE_PRESENT || echo MENAGERIE_MISSING
test -d .venv && echo VENV_PRESENT || echo VENV_MISSING

Expected branch is sac-integration. Expected project-state commit is at least:
d094a4a Record SAC fresh 250k actor diagnostic
HEAD may be newer because the migration handoff itself is committed after
d094a4a. Confirm MIGRATION_HANDOFF.md exists in the repository root.

Then read:

1. AGENTS.md
2. MIGRATION_HANDOFF.md
3. NEXT_AGENT_HANDOFF.md
4. reports/sac_integration/11_actor_drift_train_diagnostic.md
5. reports/sac_integration/10_action_distribution_diagnostics.md
6. reports/sac_integration/10_both_mode_eval_diagnostic.md
7. reports/sac_integration/09_phase_summary_and_risks.md
8. reports/sac_integration/06_next_actions.md
9. reports/sac_integration/05_known_issues.md
10. reports/sac_integration/04_smoke_results.md

If mujoco_menagerie is missing, report that it must be cloned to:
g1_env/external_deps/mujoco_menagerie
and checked out to:
1b86ece576591213e2b666ebf59508454200ca97

Output a startup report with:
- current path
- branch and commit
- git status
- menagerie status
- Python/JAX/MuJoCo/Brax status if environment exists
- summary of validated ladder
- current unresolved issue
- next recommended action

The next recommended action is decision review only:
compare fresh 500k diagnostic, alpha/entropy ablation design, deterministic
actor/eval-policy design, and action/reward component targeted analysis.

Do not run fresh 500k, 750k, or 1M. Do not tune reward/action_scale/Kp. Do not
modify PPO/RSL. Do not commit logs/checkpoints/.venv/menagerie.
```

## 14. New Machine Setup Checklist

Use this if the new machine has no environment yet:

```bash
# Clone project.
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/Kam1-hub/mujoco_playground.git
cd mujoco_playground/g1_sac_dev
git checkout sac-integration
git pull --ff-only origin sac-integration

# Build Python env.
uv sync --frozen --extra cuda

# Clone menagerie.
mkdir -p g1_env/external_deps
git clone https://github.com/deepmind/mujoco_menagerie.git g1_env/external_deps/mujoco_menagerie
git -C g1_env/external_deps/mujoco_menagerie checkout 1b86ece576591213e2b666ebf59508454200ca97

# Runtime env vars.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"

# Verify.
git status --short --branch
git check-ignore -v logs .venv g1_env/external_deps/mujoco_menagerie || true
nvidia-smi
uv run --no-sync python -c "import jax, mujoco, brax; print('jax', jax.__version__, jax.default_backend(), jax.devices()); print('mujoco', mujoco.__version__); print('brax', brax.__version__)"
uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu
```

Do not run training until the startup report is reviewed.
