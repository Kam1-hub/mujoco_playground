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
- Latest recorded diagnostic state: short render terminal diagnostic sweep
  after the eval-only 100k termination/contact diagnostic sweep, fresh env1024
  R3 100k feet-air-time command-mask diagnostic, zero-command phase-freeze,
  push-disable,
  `feet_slip_scale=0`,
  `foot_velocity`, `fixed_alpha=0.03`, and `fixed_alpha=0.05` gates,
  fixed-command forward eval gate, fixed-command eval support, alpha sign audit,
  fixed-command 3M render helper smoke, deterministic 3M render helper smoke,
  bounded 1024-env 3M R3 run, high-parallel 512/1024/2048 capacity benchmark,
  and 1M follow-up; use `git log --oneline -5` for the exact commit hash.
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
  `--action_diagnostics`, optional `--reward_components`, `--top_k_actions`,
  and optional fixed joystick command support through `--fixed_command`,
  `--command_x`, `--command_y`, and `--command_yaw`.
- `scripts/render_sac_checkpoint.py`: eval-only checkpoint render/export
  helper for MP4/GIF/PNG-frame visual inspection. It supports default
  reset-sampled joystick commands and optional fixed joystick commands with
  `--fixed_command`, `--command_x`, `--command_y`, and `--command_yaw`, plus
  optional terminal render diagnostics through `--termination_diagnostics`.
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
- Fresh 100k alpha/entropy ablation A1/A3/A4 multi-seed eval-only diagnostic:
  PASS.
- Bounded fresh 250k A4 alpha/entropy extension: PASS.
- Bounded fresh 500k A4 alpha/entropy extension: PASS.
- Bounded fresh 750k A4 alpha/entropy bridge: PASS runtime/checkpoint/eval,
  but not a clean stability improvement.
- Fresh 100k actor-regularization R1: PASS runtime/checkpoint/eval, but too
  weak to control train-time drift.
- Fresh 100k actor-regularization R2/R3 coefficient sweep: PASS
  runtime/checkpoint/eval; R3 is the best current 100k regularization
  candidate, with critic loss as a watch item.
- Bounded R3 250k actor-regularization extension: PASS
  runtime/checkpoint/eval; R3 retained drift control at 250k and improved
  deterministic/stochastic eval versus R3 100k and A4 250k.
- High-parallel 512/1024/2048 capacity benchmark: PASS. All three cases
  returned `TRAIN_OK`, wrote checkpoints, and passed readiness. `1024` envs was
  fastest at `953.36` SPS and was selected for the bounded 1M follow-up.
- Bounded 1024-env 1M R3 run: PASS runtime/checkpoint/eval. It is the first
  successful `1024`-env 1M run and shows the setup fits in 12GB VRAM, but it is
  not a policy-quality breakthrough because reward regressed versus R3 250k
  and actor mean/std drift continued.
- Bounded 1024-env 3M R3 run: PASS runtime/checkpoint/eval. It is the first
  strong deterministic-policy improvement signal: deterministic 5-seed reward
  improved versus 1M from `-4.9891` to `-2.3314` and versus R3 250k from
  `-3.7941` to `-2.3314`. However stochastic reward worsened to `-10.9904`
  and alpha/std collapsed (`alpha=0.000766`, `log_std=-0.9573`,
  `std=0.4106`), so SAC stability is not yet declared.
- Deterministic 3M render helper smoke: PASS. `scripts/render_sac_checkpoint.py`
  rendered
  `./logs/sac_render_3m_r3/render_seed0_det.mp4` from the 3M R3 checkpoint:
  H.264 MP4, `640x480`, `30fps`, `300` frames, `10s`, `1,552,965` bytes,
  `RENDER_OK`, total rollout reward `4.084568977355957`, and `done=false`.
  Main sampled-frame inspection found a nonblank upright humanoid with no
  obvious fall.
- Fixed-command 3M render helper smoke: PASS. The helper now supports
  `--fixed_command`, `--command_x`, `--command_y`, and `--command_yaw`.
  Three deterministic 3M R3 fixed-command smokes under
  `./logs/sac_render_3m_r3_fixedcmd/` all returned `RENDER_OK`, `done=false`,
  and `600` frames: forward `[0.5, 0.0, 0.0]`, stand `[0.0, 0.0, 0.0]`, and
  yaw `[0.0, 0.0, 0.5]`.
- Fixed-command 3M eval helper smoke: PASS. `scripts/eval_sac_checkpoint.py`
  now supports fixed joystick commands with the same command override semantics
  as the render helper. The `[0.5, 0.0, 0.0]` 3M R3 smoke returned `EVAL_OK`
  with deterministic reward mean `0.5099`, stochastic reward mean `-2.3354`,
  `policy_mode=both`, and no action/reward/obs NaN flags. A default
  deterministic compatibility smoke without `--fixed_command` also returned
  `EVAL_OK`, `fixed_command=false`, and no NaN flags.
- Alpha sign audit: PASS with caveat. No direct sign bug was found relative to
  Brax-style SAC: `target_entropy = -target_entropy_coef * action_dim`, R3
  target entropy is `-7.25`, and the current alpha-loss sign matches the common
  log-alpha equivalent. The unresolved risk is persistent downward alpha
  pressure in the observed log-probability range plus no alpha floor; the
  `exp(log_alpha)` form also weakens updates as alpha approaches zero.
- Fixed-command forward eval gate: PASS runtime/eval with mixed policy result.
  The 3M R3 checkpoint was evaluated on `[0.5,0,0]` and `[1.0,0,0]`, seeds
  `0..4`, `num_eval_envs=16`, `episode_length=1000`, `policy_mode=both`,
  action diagnostics, and reward components. All evals returned `EVAL_OK`; all
  action/reward/obs NaN flags were false. `fwd0.5` deterministic was near
  break-even but noisy (`0.0192` reward avg, `0.6026` stdev), `fwd1.0`
  deterministic was weak (`-3.2302` reward avg), deterministic
  `tracking_lin_vel` collapsed from `183.95` at `fwd0.5` to `25.94` at
  `fwd1.0`, and stochastic fixed-forward eval remained poor (`-9.9480` and
  `-11.5115`). This is not a runtime failure, but it blocks direct 5M/10M.
- Fresh env1024 R3 100k fixed-alpha diagnostic: PASS runtime/checkpoint with
  weak fixed-command smoke. `fixed_alpha=0.03` kept effective alpha fixed at
  `0.03` while raw `log_alpha` stayed at `-3.0`. The run returned `TRAIN_OK`,
  checkpoint
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p03/sac_lift_step_99328.pkl`,
  and readiness PASS. However small fixed-command smoke was weak:
  `fwd0.5` deterministic reward `-3.9345`, `fwd1.0` deterministic reward
  `-3.9779`, low `tracking_lin_vel`, and `termination=-100` in both smokes.
  This is not infrastructure failure, but it does not justify 250k/5M/10M from
  `fixed_alpha=0.03`.
- Fresh env1024 R3 100k `fixed_alpha=0.05` diagnostic: PASS
  runtime/checkpoint with weak fixed-command smoke. Effective alpha stayed
  fixed at `0.05`, raw `log_alpha` stayed at `-3.0`, and `alpha_floor` stayed
  inactive. The run returned `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_100k_env1024_r3_fixed_alpha_0p05/sac_lift_step_99328.pkl`,
  and readiness PASS. However `critic_loss=0.3446` crossed the prior watch
  threshold, `fwd0.5` deterministic reward was `-3.7355`, `fwd1.0`
  deterministic reward was `-3.8258`, `tracking_lin_vel` remained low, and
  `termination=-100` remained present. Do not extend fixed-alpha variants to
  250k/5M/10M from these results.
- Fresh env1024 R3 100k feet-slip and push-disable diagnostics: PASS
  runtime/checkpoint with weak fixed-command smoke. `foot_velocity`,
  `feet_slip_scale=0`, and `--env_push_enable False` all returned
  `TRAIN_OK` and readiness PASS. `feet_slip_scale=0` confirmed inherited eval
  with `reward/feet_slip=0.0`; push-disable improved deterministic
  `tracking_lin_vel` (`fwd0.5=9.9375`, `fwd1.0=5.2440`), but
  `reward/termination=-100` remained saturated.
- Fresh env1024 R3 100k zero-command phase-freeze diagnostic: PASS
  runtime/checkpoint with failed stability gate. `--env_zero_command_phase_freeze True`
  returned `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_100k_env1024_r3_phase_freeze/sac_lift_step_99328.pkl`,
  readiness PASS, and fixed-command/stand eval `EVAL_OK` with
  `zero_command_phase_freeze=true` inherited. Forward tracking was similar to
  push-disable (`fwd0.5=9.9911`, `fwd1.0=5.4434` deterministic
  `tracking_lin_vel`), but `reward/termination=-100` remained saturated for
  `fwd0.5`, `fwd1.0`, and stand; stand did not improve
  (`stand_still=-140.0983` deterministic). Do not extend this variant to
  250k/5M/10M.
- Fresh env1024 R3 100k feet-air-time command-mask diagnostic: PASS
  runtime/checkpoint with failed stability gate.
  `--env_feet_air_time_command_mask True` returned `TRAIN_OK`, checkpoint
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_air_time_mask/sac_lift_step_99328.pkl`,
  readiness PASS, and fixed-command/stand eval `EVAL_OK` with
  `feet_air_time_command_mask=true` inherited. The mask works: stand eval
  reports `reward/feet_air_time=0.0`. However `reward/termination=-100`
  remained saturated for `fwd0.5`, `fwd1.0`, and stand; `fwd1.0` tracking was
  weaker than push-disable and phase-freeze, and stand did not improve
  (`stand_still=-143.5669` deterministic). Do not extend this variant to
  250k/5M/10M.
- Eval-only 100k termination/contact diagnostic sweep: PASS diagnostic,
  failure mode identified. The sweep used existing 100k push-disable,
  phase-freeze, and feet-air-time-mask checkpoints under fixed `fwd0.5`,
  `fwd1.0`, and stand commands, deterministic and stochastic modes. All `9`
  JSON outputs under `./logs/sac_eval_termination_diag_100k/` returned
  `EVAL_OK`, all action/reward/obs NaN flags were false, and
  `reward/termination=-100` appeared in all 18 mode cases. First done happened
  early, mostly around `51-55` steps. qpos/qvel NaN counts were always `0`;
  illegal contact appeared only once (`feet_air_time_mask fwd1.0 stochastic`,
  `right_foot_left_foot=1`) while fall remained dominant. Terminal torso-up z
  was negative or near threshold with high torso angular velocity, pointing to
  early torso fall/upright instability rather than contact or numerical
  failure.
- Short render terminal diagnostic sweep: PASS diagnostic, failure mode
  reinforced. The deterministic sweep used the same existing 100k
  push-disable, phase-freeze, and feet-air-time-mask checkpoints under fixed
  `fwd0.5`, `fwd1.0`, and stand commands. It produced `9` matching `.json` and
  `.mp4` pairs under `./logs/sac_render_terminal_diag_100k/`. All cases
  terminated via fall at step `51-52`; contact and NaN reasons were absent.
  Terminal states consistently had negative torso-up z, negative root height,
  high torso XY angular velocity around `6.9-8.0`, and large local velocity.
  No variant meaningfully improved survival.
- Action joint mapping diagnostic: PASS.

Still not validated:

- 5M and 10M training.
- Full performance benchmark.
- Human/video inspection of the 3M deterministic and fixed-command renders.
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
- `./logs/sac_eval_alpha_ablate_multiseed/`
  - Eval-only multi-seed diagnostic JSONs for A1/A3/A4.
  - Seeds `0..4`, `num_eval_envs=16`, `episode_length=1000`,
    `--policy_mode both`, `--action_diagnostics`, and `--reward_components`.
  - Contains `15` JSON outputs. All evals returned `EVAL_OK`; all
    action/reward/obs NaN flags were false.
- `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
  - Bounded fresh 250k A4 alpha/entropy extension checkpoint.
  - `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`.
  - Checkpoint readiness PASS; `deterministic_eval_ready=true`; normalizers
    present.
- `./logs/sac_eval_alpha_ablate_250k/eval_A4_seed0_4x200_actiondiag.json`
  - Small 4 env x 200 both-mode action diagnostic eval from the A4 250k
    extension checkpoint.
  - Status `EVAL_OK`; action/reward/obs NaN flags false.
- `./logs/sac_eval_alpha_ablate_250k_multiseed/`
  - Five 16 env x 1000 both-mode eval JSONs from the A4 250k extension
    checkpoint.
  - All returned `EVAL_OK`; all action/reward/obs NaN flags false.
- `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
  - Bounded fresh 500k A4 alpha/entropy extension checkpoint.
  - `alpha_learning_rate=1e-4`, `target_entropy_coef=0.25`.
  - Checkpoint readiness PASS; `deterministic_eval_ready=true`; normalizers
    present.
- `./logs/sac_eval_alpha_ablate_500k/eval_A4_seed0_4x200_actiondiag.json`
  - Small 4 env x 200 both-mode action diagnostic eval from the A4 500k
    extension checkpoint.
  - Status `EVAL_OK`; action/reward/obs NaN flags false.
- `./logs/sac_eval_alpha_ablate_500k_multiseed/`
  - Five 16 env x 1000 both-mode eval JSONs from the A4 500k extension
    checkpoint.
  - All returned `EVAL_OK`; all action/reward/obs NaN flags false.
- `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`
  - Bounded fresh 750k A4 alpha/entropy bridge checkpoint.
  - `target_entropy_coef=0.25`, `alpha_learning_rate=1e-4`.
  - Checkpoint readiness PASS; normalizers present.
- `./logs/sac_eval_alpha_ablate_750k/eval_A4_seed0_4x200_actiondiag.json`
  - Small 4 env x 200 both-mode action diagnostic eval from the A4 750k
    bridge checkpoint.
  - Status `EVAL_OK`; action/reward/obs NaN flags false.
- `./logs/sac_eval_alpha_ablate_750k_multiseed/`
  - Five 16 env x 1000 both-mode eval JSONs from the A4 750k bridge
    checkpoint.
  - All returned `EVAL_OK`; all action/reward/obs NaN flags false.
- `./logs/sac_lift_gpu_100k_actor_reg_te0p25_alr1e4_l2_0p01_mean_0p001_s1/sac_lift_step_99968.pkl`
  - Fresh 100k actor-regularization R1 checkpoint.
  - `target_entropy_coef=0.25`, `alpha_learning_rate=1e-4`,
    `deterministic_action_l2_coef=0.01`, and `actor_mean_l2_coef=0.001`.
  - Checkpoint readiness PASS; normalizers present.
- `./logs/sac_eval_actor_reg_100k/eval_R1_seed0_4x200_actiondiag.json`
  - Small 4 env x 200 both-mode action diagnostic eval from the R1 checkpoint.
  - Status `EVAL_OK`; action/reward/obs NaN flags false.
- `./logs/sac_eval_actor_reg_100k_multiseed/`
  - Fifteen 16 env x 1000 both-mode eval JSONs from the R1/R2/R3
    actor-regularization checkpoints.
  - All returned `EVAL_OK`; all action/reward/obs NaN flags false.
- `./logs/sac_lift_gpu_250k_actor_reg_te0p25_alr1e4_l2_0p5_mean_0p05_s1/sac_lift_step_249984.pkl`
  - Bounded R3 250k actor-regularization checkpoint.
  - `target_entropy_coef=0.25`, `alpha_learning_rate=1e-4`,
    `deterministic_action_l2_coef=0.5`, and `actor_mean_l2_coef=0.05`.
  - Checkpoint readiness PASS; normalizers present.
- `./logs/sac_eval_actor_reg_250k/eval_R3_seed0_4x200_actiondiag.json`
  - Small 4 env x 200 both-mode action diagnostic eval from the R3 250k
    checkpoint.
  - Status `EVAL_OK`; action/reward/obs NaN flags false.
- `./logs/sac_eval_actor_reg_250k_multiseed/`
  - Five 16 env x 1000 both-mode eval JSONs from the R3 250k checkpoint.
  - All returned `EVAL_OK`; all action/reward/obs NaN flags false.
- `./logs/sac_capacity_env512_65k_r3_b256_g8_replay262k/sac_lift_step_65536.pkl`
  - High-parallel 512-env capacity checkpoint.
  - `TRAIN_OK`; checkpoint readiness PASS.
- `./logs/sac_capacity_env1024_65k_r3_b256_g16_replay262k/sac_lift_step_65536.pkl`
  - High-parallel 1024-env capacity checkpoint.
  - `TRAIN_OK`; checkpoint readiness PASS; fastest capacity case at
    `953.36` SPS.
- `./logs/sac_capacity_env2048_65k_r3_b256_g32_replay262k/sac_lift_step_65536.pkl`
  - High-parallel 2048-env capacity checkpoint.
  - `TRAIN_OK`; checkpoint readiness PASS; feasible but slower than 1024.
- `./logs/sac_lift_schema_dry_run/sac_lift_step_0.pkl`
  - Dry-run schema validation artifact, if still present.
- `./logs/sac_eval_termination_diag_100k/`
  - Eval-only termination/contact sweep over 100k push-disable, phase-freeze,
    and feet-air-time-mask checkpoints.
  - Contains `9` JSON outputs.
  - All evals returned `EVAL_OK`; all action/reward/obs NaN flags were false.
  - The result is fall-dominated early termination, not contact-dominated or
    numerical failure.

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

Bounded fresh 250k A4 extension:

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_250k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_249984.pkl`
- Checkpoint readiness PASS; `deterministic_eval_ready=true`; normalizers
  present.
- Training metrics: `env_steps=249984`, `gradient_steps=3892`,
  `wall_time=148.8921`, `sps=1678.9605`, `actor_loss=-9.3541`,
  `critic_loss=0.05245`, `alpha=0.03455`, `log_alpha=-3.36536`,
  `q=8.7442`, `target_q=8.7170`.
- Drift metrics: final actor mean abs `0.22572`, deterministic action abs
  `0.21220`, log_std mean `-0.17093`, std mean `0.84424`.
- Versus fresh 250k baseline, A4 improved alpha by `+0.01578`, actor mean abs
  by `-0.07330`, deterministic action abs by `-0.05875`, log_std by
  `+0.03434`, and std by `+0.02802`.
- Versus A4 100k, drift still increased moderately: actor mean abs
  `0.20693 -> 0.22572`, deterministic action abs `0.19314 -> 0.21220`, and
  log_std `-0.15183 -> -0.17093`.
- 5-seed deterministic eval aggregate: reward avg `-4.1561`, action abs
  `0.1861`, policy mean abs `0.1960`, log_std `-0.1183`, std `0.8894`.
- 5-seed stochastic eval aggregate: reward avg `-6.1400`, action abs `0.5213`,
  policy mean abs `0.2254`, log_std `-0.1733`, std `0.8422`.

Bounded fresh 500k A4 extension:

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_500k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_499968.pkl`
- Checkpoint readiness PASS; `deterministic_eval_ready=true`; normalizers
  present.
- Training metrics: `env_steps=499968`, `gradient_steps=7798`,
  `wall_time=271.9703`, `sps=1838.3185`, `actor_loss=-8.8341`,
  `critic_loss=0.0803`, `alpha=0.02435`, `log_alpha=-3.71524`,
  `q=8.47294`, `target_q=8.41017`.
- Drift metrics: final actor mean abs `0.29323`, deterministic action abs
  `0.26540`, log_std mean `-0.22279`, std mean `0.80245`.
- Versus the old fresh 500k baseline, A4 mitigates alpha collapse and
  deterministic drift: alpha `0.02435` vs old `0.00801`, and 5-seed
  deterministic reward avg `-4.4075` vs the old degraded `-4.69` to `-4.82`
  range.
- Versus A4 250k, drift still continues: actor mean abs `0.22572 -> 0.29323`,
  deterministic action abs `0.21220 -> 0.26540`, and std
  `0.84424 -> 0.80245`.
- 5-seed deterministic eval aggregate: reward avg `-4.4075`, action abs
  `0.2289`, policy mean abs `0.2477`, log_std `-0.1909`, std `0.8277`.
- 5-seed stochastic eval aggregate: reward avg `-6.0434`, action abs `0.5152`,
  policy mean abs `0.2704`, log_std `-0.2277`, std `0.7984`.

Bounded fresh 750k A4 bridge:

- Status: `TRAIN_OK`
- Checkpoint:
  `./logs/sac_lift_gpu_750k_alpha_ablate_te0p25_alr1e4_s1/sac_lift_step_749952.pkl`
- Checkpoint readiness PASS; `deterministic_eval_ready=true`; normalizers
  present.
- Training metrics: `env_steps=749952`, `gradient_steps=11704`,
  `wall_time=397.2631`, `sps=1887.7966`, `actor_loss=-6.5525`,
  `critic_loss=0.1441`, `alpha=0.017246`, `log_alpha=-4.06016`,
  `q=6.1746`, `target_q=6.2776`.
- Drift metrics: final actor mean abs `0.37190`, deterministic action abs
  `0.31850`, log_std mean `-0.24975`, std mean `0.78393`.
- Versus A4 500k, alpha declined `0.02435 -> 0.01725`, actor mean abs rose
  `0.29323 -> 0.37190`, deterministic action abs rose
  `0.26540 -> 0.31850`, log_std narrowed `-0.22279 -> -0.24975`, and std
  fell `0.80245 -> 0.78393`.
- Q/target_q moved down from A4 500k (`8.47/8.41 -> 6.17/6.28`), but critic
  loss worsened `0.0803 -> 0.1441`.
- 5-seed deterministic eval aggregate: reward avg `-5.7314`, action abs
  `0.3027`, policy mean abs `0.3493`, log_std `-0.2376`, std `0.7924`.
- 5-seed stochastic eval aggregate: reward avg `-6.3802`, action abs `0.5274`,
  policy mean abs `0.3551`, log_std `-0.2611`, std `0.7751`.
- Conclusion: runtime stable but not a clean stability improvement. This blocks
  any automatic 1M or longer run.

Next recommended step: decision review/design pass using A4 750k evidence. Do
not run 1M automatically.

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
  100k drift-control candidate after multi-seed eval-only follow-up. A4 has
  the lowest deterministic action magnitude and actor mean magnitude, but A1
  slightly edges deterministic reward and A4 remains weaker on stochastic
  reward.
- Bounded fresh 250k and 500k A4 extensions have run and passed. They mitigate
  the corresponding baselines but do not eliminate A4's own 100k-to-500k drift.
  Q/target_q and critic loss should be watched.
- Bounded fresh 750k A4 bridge has run and passed runtime/checkpoint/eval
  gates, but it is not a clean stability improvement. Actor mean/action drift,
  deterministic eval reward, stochastic eval reward, and critic loss worsened
  versus A4 500k.
- Fresh 100k actor-regularization R1 has run and passed runtime/checkpoint/eval
  gates. It did not reduce train-time actor mean or deterministic action
  magnitude versus A4 100k: actor mean abs worsened `0.20693 -> 0.23127`,
  deterministic action abs worsened `0.19314 -> 0.21378`, stochastic 5-seed
  reward worsened `-6.4935 -> -6.6210`, and the final regularization
  contribution was only `0.000886`.
- Fresh 100k actor-regularization R2/R3 coefficient sweep has run and passed
  runtime/checkpoint/eval gates. R2 improved over A4 100k and R1. R3 improved
  more strongly: actor mean abs `0.1506`, deterministic action abs `0.1430`,
  deterministic 5-seed reward `-4.1681`, and stochastic 5-seed reward
  `-6.3983`. R3 is the strongest current 100k regularization candidate, but
  critic loss `0.1684` is higher than R2 `0.1383`, so watch critic loss/Q if
  extending.
- Bounded R3 250k has run and passed runtime/checkpoint/eval gates. R3 retained
  drift control: train actor mean abs `0.1506 -> 0.1630`, deterministic action
  abs `0.1430 -> 0.1556`, deterministic 5-seed reward `-4.1681 -> -3.7941`,
  stochastic 5-seed reward `-6.3983 -> -5.9802`, and critic loss improved
  `0.1684 -> 0.0536`.
- The earlier 128-env ladder is now runtime/diagnostic evidence, not a
  policy-quality conclusion for G1. The 512/1024/2048 env capacity benchmark
  has passed while preserving approximate sampled update-to-data ratio. `1024`
  envs was selected for the bounded 1M follow-up.
- Bounded 1024-env 1M R3 has run and passed runtime/checkpoint/eval gates. It
  reached `999424` env steps with `3209.6723` SPS and peaked around
  `9838MiB / 12282MiB`, but deterministic reward worsened versus R3 250k
  `-3.7941 -> -4.9891` and stochastic reward worsened
  `-5.9802 -> -6.7546`, so it is not a policy-quality breakthrough.
- Bounded 1024-env 3M R3 has run and passed runtime/checkpoint/eval gates. It
  reached `2999296` env steps with `3279.6889` SPS and similar peak training
  memory around `9838MiB / 12282MiB`. Deterministic reward improved strongly
  to `-2.3314`, but stochastic reward worsened to `-10.9904` and entropy
  collapsed (`alpha=0.000766`, `log_std=-0.9573`, `std=0.4106`).
- Action joint mapping now links the 500k deterministic top action dimensions
  mainly to right ankle roll/pitch, waist pitch, right knee, and hip roll. See
  `reports/sac_integration/13_action_joint_mapping_diagnostic.md`.
- 1M replay can be around 2.5-2.7 GB raw before overhead. 5M/10M replay caps
  are too large for 12GB VRAM in this raw layout, so long runs should decouple
  `num_timesteps` from `max_replay_size`.
- SPS can vary due JIT compile and warmup.
- No PPO comparison has been run.
- Fresh env1024 R3 100k `--env_feet_slip_mode foot_velocity` has run and
  passed runtime/checkpoint gates. Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_slip_foot_velocity/sac_lift_step_99328.pkl`.
  The alternate feet-slip branch works, but fixed-command smoke remains weak:
  `fwd0.5` deterministic reward `-3.8426`, `fwd1.0` deterministic reward
  `-3.8702`, low `tracking_lin_vel`, and `termination=-100`.
- Eval override inheritance is committed at
  `442d297 Apply SAC eval env overrides from checkpoint`. Eval now applies
  checkpoint overrides such as `env_feet_slip_scale` and
  `env_feet_slip_mode`; check `env_overrides` before interpreting
  reward-component diagnostics for ablation checkpoints.
- Fresh env1024 R3 100k `--env_feet_slip_scale 0.0` has run and passed
  runtime/checkpoint gates. Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_feet_slip_scale_0/sac_lift_step_99328.pkl`.
  Inherited fixed-command eval confirms `reward/feet_slip=0.0` and NaN flags
  false, but fixed-forward behavior remains weak: `fwd1.0` deterministic
  reward `-3.4952`, `tracking_lin_vel=2.4967`, and `termination=-100`.
- Fresh env1024 R3 100k `--env_push_enable False` has run and passed
  runtime/checkpoint gates. Checkpoint:
  `./logs/sac_lift_gpu_100k_env1024_r3_push_disable/sac_lift_step_99328.pkl`.
  Inherited fixed-command eval confirms `push_config.enable=false` and NaN
  flags false. Deterministic tracking improves versus prior 100k gates
  (`fwd0.5 tracking_lin_vel=9.9375`, `fwd1.0 tracking_lin_vel=5.2440`), but
  `reward/termination=-100` remains saturated for both fixed-forward commands
  and both policy modes.

## Recommended Next Step

1. Do read-only status checks.
2. Read this file and `reports/sac_integration/09_phase_summary_and_risks.md`.
3. Review `reports/sac_integration/19_alpha_entropy_and_fixed_eval_gate.md`.
4. Plan an isolated default-off `feet_air_time` command-mask prior ablation
   before any longer run.
5. Do not draft or execute 10M automatically.

Do not start long training automatically. Do not modify reward, action scale,
Kp, domain randomization, fine-tuning, PPO, or RSL without an explicit targeted
audit/ablation plan.

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

Current HEAD should include the report commit for the bounded 1024-env 3M R3
run unless newer report commits exist. GPU 10k smoke, deterministic eval smoke, GPU 50k
sanity/eval, GPU 100k sanity/eval, GPU 250k sanity/eval, GPU 500k sanity/eval,
100k/250k/500k both-mode eval diagnostic, and full action distribution /
reward-component diagnostic have passed. Fresh 100k and fresh 250k actor drift
diagnostics have also passed. Fresh 100k alpha/entropy A1/A3/A4, A4 250k,
A4 500k, and A4 750k have passed runtime gates, but A4 750k worsened drift and
eval quality. Fresh 100k R1 was too weak; fresh 100k R2/R3 found R3 as the
strongest candidate; bounded R3 250k has now passed and retained drift control.
The high-parallel 512/1024/2048 capacity benchmark has passed, and 1024 envs is
now validated at bounded 1M and 3M runtime/checkpoint/eval scale. The 3M
result is the first strong deterministic-policy improvement signal, but
stochastic eval degraded and alpha/std collapsed.
Fixed-command 3M render support has also been added and smoke-validated for
forward, stand, and yaw commands, with artifacts under ignored
`./logs/sac_render_3m_r3_fixedcmd/`.
Fixed-command eval support has also been added to
`scripts/eval_sac_checkpoint.py` and smoke-validated on the 3M R3 checkpoint:
`[0.5,0,0]` returned `EVAL_OK` with deterministic reward mean `0.5099`,
stochastic reward mean `-2.3354`, and no action/reward/obs NaN flags. Alpha
sign audit found `SIGN_OK_BUT_COLLAPSE_RISK`: no direct sign bug, but current
target entropy and observed log-probability ranges keep downward pressure on
alpha and there is still no alpha floor.
Fixed-command forward eval gate has also been run for `[0.5,0,0]` and
`[1.0,0,0]`, seeds `0..4`, with both deterministic and stochastic policy
modes. All evals were `EVAL_OK` with NaN flags false. `fwd0.5` deterministic
was near break-even but noisy (`0.0192` reward avg); `fwd1.0` deterministic was
weak (`-3.2302` reward avg) and deterministic `tracking_lin_vel` collapsed from
`183.95` to `25.94`. Stochastic remained poor for both commands.

Do not run training, eval, preflight, installs, downloads, or git commits unless
explicitly asked. The latest short gates show `fixed_alpha=0.03` and
`fixed_alpha=0.05` preserve alpha but do not solve 100k fixed-forward
tracking; `0.05` also introduced a critic-loss watch item. The `foot_velocity`
feet-slip gate also passed runtime/checkpoint checks but did not solve
fixed-command smoke. The `--env_feet_slip_scale 0.0` gate also passed and,
after eval override inheritance in `442d297`, correctly reports
`reward/feet_slip=0.0`; however `fwd1.0` still has low tracking and
`termination=-100`. The default-off push-disable gate passed runtime/checkpoint
checks and improved deterministic tracking somewhat, but `reward/termination`
remains saturated for both fixed-forward commands and both policy modes. The
zero-command phase-freeze gate also passed runtime/checkpoint checks and
inherited eval, but it did not remove termination saturation and did not
improve stand. The feet-air-time command-mask gate also passed runtime,
checkpoint, and inherited eval checks; it correctly zeroed
`reward/feet_air_time` on stand, but termination remained saturated and stand
did not improve. The termination/contact sweep and terminal render sweep now
show the common failure is early torso/base fall around `51-52` steps, with
negative torso-up z, negative root height, high torso XY angular velocity, and
no contact/NaN root cause. Next recommended work is early-fall
stabilization/curriculum design: reset disturbance/warmup, command warmup,
base-height/alive/orientation/angular-velocity stabilizers, and action-rate
smoothing. `alpha_floor=0.03` is only a secondary diagnostic. Do not run
fixed-alpha, foot-velocity, feet-slip-scale-zero, push-disable, phase-freeze,
or feet-air-time-mask 250k, 5M, or 10M from these results. Detailed video
inspection may still help distinguish fall direction or posture collapse, but
it should not justify 5M/10M without resolving early fall, forward tracking
weakness, and stochastic collapse. Do not change reward, action_scale, Kp,
domain randomization, fine-tuning, PPO, or RSL without a targeted
audit/ablation plan.
```
