# Next Agent Handoff

This file is a self-contained handoff for continuing SAC Route B work after a
context reset.

## Project Goal

Add and validate a local SAC baseline for the G1 lift/joystick task while
preserving existing PPO/RSL workflows. Current work is validation-driven: move
from small smoke tests to larger sanity runs only after each previous gate is
clean.

Do not use PPO/Barkour/RSL experience to override the current SAC design,
Route B parameters, or smoke ladder. Do not tune reward, action scaling, Kp,
domain randomization, or fine-tuning as part of the current validation phase.

## Repository

- Path: `/home/admin/projects/mujoco_playground/g1_sac_dev`
- Branch: `sac-integration`
- Latest local diagnostic report baseline before this update:
  `f454fef Record SAC action joint mapping diagnostic`
- Remote: `origin https://github.com/Kam1-hub/mujoco_playground.git`
- Last known pushed branch: `sac-integration`

Start every new session with read-only checks:

```bash
pwd
git status --short --branch
git log --oneline -5
git remote -v
```

## Environment

Known working target:

- OS: WSL2 Ubuntu
- GPU: NVIDIA RTX 4070 SUPER, 12GB VRAM
- Python: 3.12.3 in the project uv environment
- JAX: 0.10.0
- JAX backend/device: `gpu`, `cuda:0`
- MuJoCo: 3.8.0
- Brax: 0.14.2
- `nvidia-smi`: available
- `nvcc`: missing in the checked environment, not a blocker for this stack

Required shell variables before GPU/JAX commands:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl
export JAX_COMPILATION_CACHE_DIR="$HOME/.cache/jax"
```

`nvidia-smi` missing is a blocker. `nvcc` missing is not a blocker when JAX can
see `cuda:0`.

## External Menagerie

The project expects a local ignored checkout:

```text
g1_env/external_deps/mujoco_menagerie
```

Required commit:

```text
1b86ece576591213e2b666ebf59508454200ca97
```

Check it with:

```bash
git -C g1_env/external_deps/mujoco_menagerie rev-parse HEAD
```

Do not commit this checkout.

## Code Structure

SAC Route B code lives in local files and should not disturb PPO/RSL:

- `learning/train_jax_sac_lift.py`: CLI entry point for Route B SAC.
- `learning/sac_lift/train.py`: SAC training loop, replay insertion, updates,
  checkpoint payload.
- `learning/sac_lift/losses.py`: critic, actor, and temperature losses.
- `learning/sac_lift/networks.py`: Flax actor and twin-Q networks.
- `learning/sac_lift/replay_buffer.py`: uniform replay buffer.
- `learning/sac_lift/normalizer.py`: running observation stats.
- `learning/sac_lift/checkpoint.py`: pickle save/load helpers.
- `learning/sac_lift/evaluator.py`: helper evaluator logic.
- `scripts/gpu_preflight.py`: non-training CUDA/JAX/env/menagerie preflight.
- `scripts/check_sac_checkpoint.py`: checkpoint schema and eval readiness
  checker.
- `scripts/eval_sac_checkpoint.py`: bounded checkpoint eval with
  `--policy_mode deterministic|stochastic|both`, optional
  `--action_diagnostics`, optional `--reward_components`, and
  `--top_k_actions`.
- `scripts/inspect_g1_action_mapping.py`: no-training action dimension to
  actuator/joint mapping helper.
- `scripts/summarize_sac_action_diag.py`: no-training summary helper that joins
  action diagnostic JSON with the action mapping.
- `g1_env/config/sac_params.py`: SAC Route B config defaults.

## Entry Points

Common commands:

```bash
uv sync --frozen --extra cuda

uv run --no-sync python scripts/gpu_preflight.py --impl jax --require_gpu

uv run --no-sync python -m learning.train_jax_sac_lift --help

uv run --no-sync python scripts/check_sac_checkpoint.py --help

uv run --no-sync python scripts/eval_sac_checkpoint.py --help
```

Do not run training, eval, preflight, install, or download commands unless the
user explicitly approves that phase.

## SAC Route B Design

- Environment: `G1JoystickFlatTerrain`
- Implementation: JAX/MJX via `impl=jax`
- Actor observation key: `state`
- Critic observation key: `privileged_state`
- Action size: 29
- Action scaling remains inside the environment. Do not add external action
  scaling in SAC.
- Actor: tanh-Gaussian policy.
- Critic: twin Q networks with target Q parameters.
- Entropy: learned `log_alpha`.
- Replay: local uniform replay buffer.
- Normalization: policy and value observation normalizers are saved in future
  checkpoints.
- Truncation: synthesized as zero when the env info does not provide it.
- Checkpoint schema now includes at least:
  `config`, `policy_params`, `q_params`, `target_q_params`, `log_alpha`,
  `policy_normalizer`, `value_normalizer`, and `metrics`.

## Passed Validation

Current validated ladder:

- CPU tiny smoke: PASS.
- GPU preflight: PASS.
- GPU 10k smoke: PASS.
- 10k checkpoint eval readiness with normalizer-ready checkpoint: PASS.
- Deterministic eval smoke: PASS.
- GPU 50k sanity: PASS.
- 50k checkpoint eval readiness: PASS.
- 50k bounded deterministic eval: PASS.
- GPU 100k sanity: PASS.
- 100k checkpoint eval readiness: PASS.
- 100k bounded deterministic eval: PASS.
- GPU 250k sanity: PASS.
- 250k checkpoint eval readiness: PASS.
- 250k bounded deterministic eval: PASS.
- GPU 500k sanity: PASS.
- 500k checkpoint eval readiness: PASS.
- 500k bounded deterministic eval: PASS.
- 100k/250k/500k both-mode deterministic/stochastic eval diagnostic: PASS.
- 100k/250k/500k full action distribution / reward-component eval diagnostic:
  PASS.
- Fresh 100k train-time actor drift diagnostic: PASS.
- Fresh 250k train-time actor drift diagnostic: PASS.
- Fresh 100k alpha/entropy ablation A1/A3/A4: PASS.
- Action joint mapping diagnostic: PASS.

Still not validated:

- 750k training.
- 1M training.
- Full performance benchmark.
- PPO comparison.
- Domain randomization.
- Fine-tuning.
- Reward/action scale/Kp tuning.

## Artifacts and Logdirs

All paths below are runtime artifacts and should remain ignored:

- `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
  - First GPU 10k smoke checkpoint.
  - Old schema, missing normalizers, not trusted for deterministic eval.
- `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
  - New schema 10k checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_smoke/eval_4x200.json`
  - 4 env x 200 step deterministic eval smoke result.
- `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
  - 50k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_50k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result.
- `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
  - 100k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_100k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result after 100k.
- `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
  - 250k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_250k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result after 250k.
- `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
  - 500k sanity checkpoint with normalizers.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_500k/eval_16x1000.json`
  - 16 env x 1000 step bounded deterministic eval result after 500k.
- `./logs/sac_eval_bothmode/`
  - 100k/250k/500k eval-only diagnostic JSONs using
    `--policy_mode both`, seeds `0..4`, `num_eval_envs=16`, and
    `episode_length=1000`.
- `./logs/sac_eval_action_diag_full/`
  - 100k/250k/500k eval-only diagnostic JSONs using `--policy_mode both`,
    `--action_diagnostics`, and `--reward_components`.
  - Seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`.
  - Contains `15` JSON outputs. All evals returned `EVAL_OK`; all
    action/reward/obs NaN flags were false.
- `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`
  - Fresh 100k train-time actor drift diagnostic checkpoint.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`
  - 4 env x 200 action diagnostic eval from the fresh 100k actor drift
    checkpoint.
  - Status `EVAL_OK`; JSON sanity PASS.
- `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`
  - Fresh 250k train-time actor drift diagnostic checkpoint.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`
  - 4 env x 200 action diagnostic eval from the fresh 250k actor drift
    checkpoint.
  - Status `EVAL_OK`; JSON sanity PASS.
- `./logs/sac_lift_gpu_100k_alpha_ablate_alr1e4_s1/sac_lift_step_99968.pkl`
  - Fresh 100k A1 alpha/entropy ablation checkpoint.
  - `alpha_learning_rate=1e-4`, `target_entropy_coef=0.5`.
  - Checkpoint readiness PASS.
- `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_s1/sac_lift_step_99968.pkl`
  - Fresh 100k A3 alpha/entropy ablation checkpoint.
  - `alpha_learning_rate=3e-4`, `target_entropy_coef=0.25`.
  - Checkpoint readiness PASS.
- `./logs/sac_lift_gpu_100k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_99968.pkl`
  - Fresh 100k A4 alpha/entropy ablation checkpoint.
  - `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`.
  - Checkpoint readiness PASS.
- `./logs/sac_eval_alpha_ablate/`
  - Small 4 env x 200 both-mode action diagnostic eval JSONs for A1/A3/A4.
  - All returned `EVAL_OK`; action/reward/obs NaN flags were false.
- `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl`
  - Dry-run schema validation artifact, if still present.

Never commit:

- `logs/`
- checkpoints
- `.venv/`
- `g1_env/external_deps/mujoco_menagerie`
- generated caches

Check ignored status with:

```bash
git check-ignore -v logs .venv g1_env/external_deps/mujoco_menagerie || true
```

## Key Results

Full action distribution diagnostic:

- Scope: eval-only; no training and no SAC code change during the diagnostic.
- Output directory: `./logs/sac_eval_action_diag_full/`
- JSON count: `15`
- Checkpoints: 100k, 250k, 500k
- Checkpoint readiness: PASS for all three checkpoints
- Deterministic reward average: `-4.2130 -> -4.4792 -> -4.8204`
- Deterministic action abs: `0.1823 -> 0.2147 -> 0.3029`
- Deterministic policy mean abs: `0.1950 -> 0.2288 -> 0.3468`
- Deterministic policy std mean: `0.8996 -> 0.8692 -> 0.7519`
- Stochastic reward average: `-6.4741 -> -6.2374 -> -5.8911`
- Stochastic policy std mean: `0.8578 -> 0.8254 -> 0.7256`
- Deterministic degradation aligns mainly with `reward/ang_vel_xy`,
  `reward/stand_still`, and `reward/orientation`.
- Conclusion: not a runtime failure and not stochastic policy collapse. The key
  risk is deterministic deployment/eval degradation driven by actor mean/action
  magnitude drift, with entropy/alpha dynamics likely upstream.

Fresh 100k actor drift diagnostic:

- Code baseline: `202c6a9 Add SAC actor drift train diagnostics`.
- Scope: diagnostic training plus small action diagnostic eval; no 250k/750k/1M.
- Checkpoint: `./logs/sac_lift_gpu_100k_actor_diag/sac_lift_step_99968.pkl`.
- Checkpoint readiness: PASS.
- Eval JSON:
  `./logs/sac_eval_actor_diag_100k/eval_both_seed0_4x200_actiondiag.json`.
- Eval status: `EVAL_OK`.
- Final actor mean abs `0.238568` exceeded interval avg `0.170754`.
- Final deterministic action abs `0.218864` exceeded interval avg `0.163467`.
- Final log_std mean `-0.157116` was below interval avg `-0.136390`.
- Alpha was about `0.0326`.
- Interpretation: actor mean drift is already forming by 100k. This is not a
  runtime failure.

Fresh 250k actor drift diagnostic:

- Scope: diagnostic training plus small action diagnostic eval; no fresh
  500k/750k/1M.
- Checkpoint: `./logs/sac_lift_gpu_250k_actor_diag/sac_lift_step_249984.pkl`.
- Checkpoint readiness: PASS.
- Eval JSON:
  `./logs/sac_eval_actor_diag_250k/eval_both_seed0_4x200_actiondiag.json`.
- Eval status: `EVAL_OK`.
- Actor mean abs final increased `0.238568 -> 0.299021`.
- Deterministic action abs final increased `0.218864 -> 0.270944`.
- Final log_std moved down `-0.157116 -> -0.205270`.
- Alpha declined `0.032585 -> 0.018768`.
- Interpretation: actor mean / deterministic action drift amplifies in
  absolute level by fresh 250k. This is not a runtime failure.

Fresh 100k alpha/entropy ablation:

- A1: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.5`,
  `TRAIN_OK`, checkpoint readiness PASS, eval `EVAL_OK`.
- A3: `alpha_learning_rate=3e-4`, `target_entropy_coef=0.25`,
  `TRAIN_OK`, checkpoint readiness PASS, eval `EVAL_OK`.
- A4: `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`,
  `TRAIN_OK`, checkpoint readiness PASS, eval `EVAL_OK`.
- A4 was run because A1 and A3 both passed runtime/checkpoint/eval gates.
- A4 is the best current 100k candidate: alpha `0.042848`, actor mean abs
  `0.206928`, deterministic action abs `0.193139`, log_std mean `-0.151827`,
  std mean `0.861342`, and best deterministic 4x200 reward among A1/A3/A4.
- Caveat: A4 stochastic 4x200 reward was worse than A1/A3, so this remains a
  diagnostic signal rather than final policy-quality evidence.

Next recommended step: bounded A4 follow-up decision. Prefer multi-seed 100k
ablation eval and/or a fresh 250k A4 extension only after confirmation. Do not
run fresh 500k, 750k, or 1M automatically.

GPU 10k smoke:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_10k/sac_lift_step_9984.pkl`
- `env_steps=9984`, `gradient_steps=142`
- `wall_time=56.50599093900382`, `sps=176.68922947970893`
- No NaN/Inf/OOM/CUDA error observed.

10k deterministic eval smoke:

- Checkpoint: `./logs/sac_lift_gpu_10k_normalizer/sac_lift_step_9984.pkl`
- JSON: `./logs/sac_eval_smoke/eval_4x200.json`
- `eval_env_steps=800`
- `episode_reward_mean=-3.3628087043762207`
- `done_fraction=1.0`
- No action/reward/obs NaN.

GPU 50k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_50k_sanity/sac_lift_step_49920.pkl`
- `env_steps=49920`, `gradient_steps=766`
- `wall_time=35.99766752999858`, `sps=1386.756515777009`
- `actor_loss=-3.6135072708129883`
- `critic_loss=0.07097882032394409`
- `alpha=0.03992176800966263`
- `policy_q=2.8622469902038574`
- `target_q=2.899707317352295`
- No NaN/Inf/OOM/CUDA/checkpoint/eval error observed.

50k bounded deterministic eval:

- JSON: `./logs/sac_eval_50k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-3.5016322135925293`
- `episode_reward_std=0.75983726978302`
- `episode_reward_min=-6.096090316772461`
- `episode_reward_max=-2.531925916671753`
- `done_fraction=1.0`
- `wall_time=74.76505397899746`, `sps=214.00372431342888`
- No action/reward/obs NaN.

GPU 100k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl`
- `env_steps=99968`, `gradient_steps=1548`
- `wall_time=56.80434615799459`, `sps=1759.8653406193746`
- `actor_loss=-4.994826316833496`
- `critic_loss=0.04054964333772659`
- `alpha=0.03259027376770973`
- `q=4.3698601722717285`
- `target_q=4.456111907958984`
- No NaN/Inf/OOM/fatal CUDA/checkpoint/eval error observed.

100k bounded deterministic eval:

- JSON: `./logs/sac_eval_100k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-3.894726037979126`
- `episode_reward_std=0.9610732197761536`
- `episode_reward_min=-7.2897186279296875`
- `episode_reward_max=-2.889821767807007`
- `done_fraction=1.0`
- `wall_time=66.85148939098872`, `sps=239.33647770242092`
- No action/reward/obs NaN.

GPU 250k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_250k_sanity/sac_lift_step_249984.pkl`
- `env_steps=249984`, `gradient_steps=3892`
- `wall_time=120.21957968600327`, `sps=2079.3950590488107`
- `actor_loss=-5.524118900299072`
- `critic_loss=0.02644157037138939`
- `alpha=0.018743595108389854`
- `q=5.197851181030273`
- `target_q=5.205532073974609`
- Checkpoint readiness PASS with `policy_normalizer` and `value_normalizer`
  present.
- No NaN/Inf/OOM/fatal CUDA/env/checkpoint/eval error observed.

250k bounded deterministic eval:

- JSON: `./logs/sac_eval_250k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-4.279743194580078`
- `episode_reward_std=1.2322009801864624`
- `episode_reward_min=-8.849853515625`
- `episode_reward_max=-3.093963384628296`
- `done_fraction=1.0`
- `wall_time=68.2435936529946`, `sps=234.45424168833907`
- No action/reward/obs NaN.

GPU 500k sanity:

- `TRAIN_OK`
- Checkpoint: `./logs/sac_lift_gpu_500k_sanity/sac_lift_step_499968.pkl`
- `env_steps=499968`, `gradient_steps=7798`
- `wall_time=223.88994164399628`, `sps=2233.0971919899416`
- `actor_loss=-3.510934352874756`
- `critic_loss=0.04195608198642731`
- `alpha=0.008012857288122177`
- `q=3.348696231842041`
- `target_q=3.3187503814697266`
- Checkpoint readiness PASS with `policy_normalizer` and `value_normalizer`
  present.
- No NaN/Inf/OOM/fatal CUDA/env load/reset/step/shape/replay/checkpoint/eval
  failure observed.

500k bounded deterministic eval:

- JSON: `./logs/sac_eval_500k/eval_16x1000.json`
- `eval_env_steps=16000`
- `episode_reward_mean=-4.691065788269043`
- `episode_reward_std=1.1929610967636108`
- `episode_reward_min=-9.040802955627441`
- `episode_reward_max=-3.7163496017456055`
- `done_fraction=1.0`
- `wall_time=68.70198891899781`, `sps=232.88990976468807`
- No action/reward/obs NaN.

Both-mode eval diagnostic:

- Script: `scripts/eval_sac_checkpoint.py --policy_mode both`
- Checkpoints: 100k, 250k, 500k.
- Seeds: `0..4`; `num_eval_envs=16`; `episode_length=1000`.
- Deterministic reward mean aggregate:
  `100k=-4.2218`, `250k=-4.4585`, `500k=-4.8476`.
- Deterministic action abs mean:
  `0.1823 -> 0.2148 -> 0.3029`.
- Stochastic reward mean aggregate:
  `100k=-6.4616`, `250k=-6.1954`, `500k=-5.9091`.
- Stochastic log-prob mean:
  `-17.9594 -> -17.2847 -> -13.4275`.
- Core conclusion: deterministic `tanh(mean)` behavior degrades while sampled
  stochastic behavior does not show the same degradation. This narrows the next
  diagnostic target to actor mean / action distribution / reward components.

500k risk notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- Alpha continued down from about `0.0187` at 250k to about `0.0080` at 500k.
- Q and target Q decreased from about `5.2` to about `3.3`; critic loss stayed
  finite/low.
- Bounded eval reward mean worsened from `-4.27974` to `-4.69107`, and eval
  max worsened from `-3.09396` to `-3.71635`.
- This is not a runtime failure, but alpha decline and eval degradation block
  any automatic jump to 1M.

100k tooling notes:

- WSL2 CUDA driver version format warning and JAX cast overflow warning were
  observed and remained non-fatal.
- A first sandboxed `uv` attempt hit a `snap-confine` capability issue before
  training started. The identical command succeeded with external permission and
  unchanged parameters, so this is tooling noise, not a training failure.

## Remaining Risks

- 1M stability is unknown.
- Eval rewards are low and only prove bounded eval execution, not policy quality.
- Truncation handling is still an assumption when absent from env info.
- Replay memory and normalizer behavior need 1M-scale validation.
- 500k exposed alpha-decline and deterministic eval-degradation risk despite a
  clean runtime sanity PASS.
- Both-mode eval suggests deterministic actor mean behavior is the immediate
  issue; stochastic sampled behavior does not degrade in the same way.
- Full action diagnostic confirms the issue is tied to deterministic actor
  mean/action magnitude drift and specific reward components, mainly
  `reward/ang_vel_xy`, `reward/stand_still`, and `reward/orientation`.
- Fresh 100k train-time diagnostics show the mean/action drift signal is already
  visible by 100k, so the next diagnostic should test whether the final-vs-
  interval gap widens by fresh 250k.
- Fresh 250k train-time diagnostics show the drift amplifies in absolute level,
  while log_std/std and alpha continue downward.
- Fresh 100k alpha/entropy ablation A1/A3/A4 has run. A4 is the best current
  100k drift candidate, but it still needs multi-seed or fresh 250k
  confirmation before longer training because its stochastic 4x200 reward was
  worse than A1/A3.
- Action joint mapping now links the 500k deterministic top action dimensions
  mainly to right ankle roll/pitch, waist pitch, right knee, and hip roll. See
  `reports/sac_integration/13_action_joint_mapping_diagnostic.md`.
- 1M replay can be around 2.5-2.7 GB raw before overhead.
- SPS can vary due JIT compile and warmup.
- No PPO comparison has been run.

## Recommended Next Step

1. Do read-only status checks.
2. Read this file and `reports/sac_integration/09_phase_summary_and_risks.md`.
3. Review `reports/sac_integration/12_alpha_entropy_ablation_plan.md`.
4. If the user explicitly approves another bounded diagnostic, run multi-seed
   100k ablation eval for A4 or plan a fresh 250k A4 extension.
5. Do not draft or execute fresh 500k/750k/1M until A4 is confirmed.

Do not start fresh 500k, 750k, or 1M automatically. Do not modify reward,
action scale, Kp, domain randomization, fine-tuning, PPO, or RSL.

## Completed 100k Sanity Command

The 100k sanity command that passed was:

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

The readiness and bounded eval commands that passed were:

```bash
uv run --no-sync python scripts/check_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --require_eval_ready

uv run --no-sync python scripts/eval_sac_checkpoint.py \
  --checkpoint ./logs/sac_lift_gpu_100k_sanity/sac_lift_step_99968.pkl \
  --num_eval_envs 16 \
  --episode_length 1000 \
  --render False \
  --output_json ./logs/sac_eval_100k/eval_16x1000.json
```

## Suggested New Agent Prompt

```text
You are in /home/admin/projects/mujoco_playground/g1_sac_dev.

First read NEXT_AGENT_HANDOFF.md and
reports/sac_integration/09_phase_summary_and_risks.md. Then run only read-only
status checks: pwd, git status --short --branch, git log --oneline -5,
git remote -v, and git check-ignore -v logs .venv
g1_env/external_deps/mujoco_menagerie || true.

Current HEAD should be at least f454fef Record SAC action joint mapping diagnostic
unless newer report commits exist. GPU 10k smoke, deterministic
eval smoke, GPU 50k sanity/eval, GPU 100k sanity/eval, GPU 250k sanity/eval,
GPU 500k sanity/eval, 100k/250k/500k both-mode eval diagnostic, and full action
distribution / reward-component diagnostic have passed. Fresh 100k and fresh
250k actor drift diagnostics have also passed. Fresh 100k alpha/entropy
ablation A1/A3/A4 has passed runtime/checkpoint/eval gates, and A4 is the best
current 100k drift candidate. 750k and 1M are not validated.

Do not run training, eval, preflight, installs, downloads, or git commits unless
explicitly asked. Next recommended work is multi-seed 100k A4 ablation eval
and/or a fresh 250k A4 extension after confirmation. Do not start fresh 500k,
750k, or 1M without a separate resource/stop-condition plan and user
confirmation. Do not change reward, action_scale, Kp, domain randomization,
fine-tuning, PPO, or RSL.
```
