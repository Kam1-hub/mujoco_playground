# Agent Instructions For G1 SAC Dev

This repository is an isolated SAC-integration workspace copied from `D:\mujoco_playground\template`.

## Mission

Implement a G1 SAC baseline without disturbing the existing PPO/RSL-RL paths.

Primary target:

- MuJoCo Playground-style G1 env package: `g1_env/`
- JAX / Brax learner style
- G1 locomotion tasks:
  - `G1JoystickFlatTerrain`
  - `G1JoystickRoughTerrain`
- SAC first; world model, vision, real deployment, and fine-tuning are out of scope for the first implementation pass.

## Operating Rules

- Do not wholesale copy `LIFT-humanoid/mujoco_playground`, `LIFT-humanoid/brax_env`, or `LIFT-humanoid/world_model`.
- Do not alter the existing PPO/RSL behavior except for adding new entry points in `pyproject.toml`.
- Keep all phase reports in `reports/sac_integration/`.
- Preserve full tracebacks in reports when a command fails.
- Use small smoke tests before longer training.
- Mark any unrun command as `NOT VALIDATED`.
- On Windows native, treat GPU JAX training as unsupported unless a WSL2/Linux CUDA environment is actually detected.

## Parallel / Closed-Loop Workflow

- Use parallel file reads/searches whenever possible.
- Before implementing a phase, write or update a short checklist in the relevant phase report.
- After implementing, run the matching validation commands and record:
  - commands
  - pass/fail
  - shapes
  - metrics
  - full traceback if failed
  - diagnosis
  - next action
- If subagents are available and the user prompt explicitly permits parallel/delegated agent work, split independent tasks:
  - Explorer A: env API and wrappers
  - Explorer B: Brax SAC route
  - Worker C: LIFT-style SAC modules
  - Verifier D: smoke commands and report audit
- Keep implementation and verification in the same phase loop until the acceptance criteria pass or a blocker is documented.

## Reference Context

Read these before coding:

- `WSL2_CODEX_HANDOFF.md`
- `AGENT_MEMORY.md`
- `SAC_INTEGRATION_MASTER_PLAN.md`
- `AGENT_EXECUTION_PROMPT.md`
- `external_references/LIFT-humanoid-reports/`
