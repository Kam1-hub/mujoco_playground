# Prompt For The Next Codex Agent

You are working in `D:\mujoco_playground\g1_sac_dev`, an isolated G1 MuJoCo Playground pipeline copied from `D:\mujoco_playground\template`.

The user wants a closed-loop engineering run: plan, implement, validate, document, and continue until the SAC integration is ready for review.

You may use parallel tool calls aggressively for file reads/searches. If subagents/delegation are available in your environment, you are explicitly permitted to use parallel agents for independent subtasks such as env API exploration, learner implementation, and verification. Keep write sets disjoint when delegating.

## Read First

1. `AGENTS.md`
2. `AGENT_MEMORY.md`
3. `SAC_INTEGRATION_MASTER_PLAN.md`
4. `external_references/LIFT-humanoid-reports/bug_traps.md`
5. `external_references/LIFT-humanoid-reports/integration_plan.md`
6. `external_references/LIFT-humanoid-reports/hyperparameter_table.md`

## Mission

Add SAC to the existing G1 pipeline without disturbing PPO/RSL-RL.

Implement:

- `scripts/check_g1_env_api.py`
- `learning/sac_wrappers.py`
- `learning/train_jax_sac_brax.py` if upstream Brax SAC is feasible
- `learning/sac_lift/` and `learning/train_jax_sac_lift.py`
- `g1_env/config/sac_params.py`
- new entry points:
  - `train-g1-sac-brax`
  - `train-g1-sac`

Do not implement world model/fine-tuning/vision/real deployment in this phase.

## Required SAC Behavior

- Actor obs key default: `state`.
- Critic obs key default: `privileged_state`.
- If `privileged_state` is missing, fallback to `state` and log a warning.
- If `truncation` is missing, synthesize zeros and log a warning.
- Use explicit actor/critic/alpha learning-rate CLI flags.
- Use `q_hidden_layer_sizes`, not `value_hidden_layer_sizes`, for SAC critic.
- Support `grad_updates_per_step`.
- Support deterministic eval.
- Save checkpoints under the provided logdir.

## Validation Order

1. Write/update `reports/sac_integration/00_workspace_manifest.md`.
2. Implement and run `scripts/check_g1_env_api.py`.
3. Write `reports/sac_integration/01_g1_sac_readiness.md`.
4. Implement Route A if feasible.
5. Implement Route B as the main route.
6. Run validation ladder from `SAC_INTEGRATION_MASTER_PLAN.md`.
7. Update reports after each phase.

## Reporting Format

Every phase report must include:

```markdown
# SAC Integration Status Report

## Snapshot
- Project path:
- Git branch:
- Git commit:
- OS:
- Python:
- JAX:
- MuJoCo:
- CUDA / WSL2:
- Date:

## Current Route
- Route A:
- Route B:
- Route C:

## Files Changed

## Commands Run

## Results
| Test | Status | Notes |

## Shapes
- action_size:
- obs type:
- obs keys:
- state shape:
- privileged_state shape:
- replay transition shape:

## Runtime Metrics
- steps:
- wall time:
- SPS:
- eval reward:
- actor loss:
- critic loss:
- alpha:
- NaN:

## Errors / Blockers

## Diagnosis

## Next Proposed Fix
```

## Stop Conditions

Stop and ask for review only if:

- dependencies are missing and installation requires user approval;
- env import/load cannot proceed after documenting traceback;
- SAC design choice is ambiguous and affects algorithm correctness;
- smoke run fails after a clear patch attempt and traceback is captured.

Otherwise, continue through implementation and verification.

