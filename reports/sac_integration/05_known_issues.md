# Known Issues

Status: updated on 2026-05-11 after Route B static implementation.

## Current Blockers

| Issue | Status | Evidence | Impact | Next Action |
|---|---:|---|---|---|
| Missing `mujoco_menagerie` assets | BLOCKED_BY_DEPENDENCY | `Test-Path g1_env\external_deps\mujoco_menagerie` returned `False`; Phase 1 and Route A tracebacks both stop at `registry.load`/`ensure_menagerie_exists()`. | Env load/reset/step, Route A smoke, Route B CPU smoke, and GPU smoke cannot be validated. | Provide assets locally or authorize the guarded download path, then rerun Level 1 before training smoke. |
| Route B env runtime smoke | BLOCKED_BY_DEPENDENCY | `learning.train_jax_sac_lift` non-dry-run probe stops before env load because `mujoco_menagerie` is absent. | Env load/reset/step and CPU tiny smoke cannot be validated. | Provide assets locally or authorize guarded download, then rerun env API before training smoke. |
| System Python lacks runtime deps | BLOCKED_BY_DEPENDENCY | Phase 1 reports `ModuleNotFoundError: No module named 'jax'` for system Python. | Commands using bare `python` may fail depending on PATH. | Prefer `.\.venv\Scripts\python.exe` for validation unless the runtime is intentionally changed. |
| Windows native GPU JAX unsupported as success target | NOT VALIDATED | Workspace reports detected CPU JAX in `.venv`; GPU/WSL2 CUDA was not validated. | Level 4 GPU smoke should not be expected to pass on native Windows. | Run Level 4 only in WSL2/Linux CUDA JAX or another proven CUDA JAX environment. |
| Raw G1 env does not set `state.info["truncation"]` in static audit | MITIGATED | Phase 1 static audit marks native truncation source as FAIL; Route B synthesizes zeros when missing and warns. | Timeout handling still needs runtime validation once assets exist. | Rerun env/API and smoke tests with assets to confirm wrapper-generated truncation appears under training wrappers. |
| G1 action scaling is internal to env step | MITIGATED | Phase 1 static audit: env computes `default_pose + action * action_scale`; Route B does not externally rescale actions. | Runtime action path still needs smoke validation once assets exist. | Keep external action scale as identity unless a future change explicitly changes env/Q action units and log-prob correction together. |

## Known Non-Blockers

- Route A upstream Brax SAC availability is already established in `02_route_a_brax_sac.md`.
- Route B entry point now exists and `python -m learning.train_jax_sac_lift --help` passes.
- Route B dry-run now passes, initializes network/replay shapes, and writes a checkpoint under `logs/`.
- Upstream Brax SAC dict observation limitation is known and Route A uses a selected `state` observation fallback.
- Static registry/config checks found both target env names: `G1JoystickFlatTerrain` and `G1JoystickRoughTerrain`.
- `.venv` has JAX/MuJoCo/Brax on CPU per prior reports, so dependency work should focus first on assets and Route B implementation.

## Pending Tracebacks

- Route B help/dry-run tracebacks: none; both pass.
- CPU tiny smoke traceback: NOT VALIDATED, pending menagerie assets and explicit permission to simulate on a suitable host.
- GPU smoke traceback: NOT VALIDATED, pending CUDA-capable JAX runtime.

## LIFT Traps To Avoid

See copied report:

```text
external_references/LIFT-humanoid-reports/bug_traps.md
```

Do not inherit these issues in Route B:

- Ambiguous `--learning_rate` that does not clearly map to actor, critic, and alpha rates.
- `--value_hidden_layer_sizes` accidentally writing SAC critic hidden sizes; use `--q_hidden_layer_sizes`.
- CLI defaults that override asymmetric critic config by setting `value_obs_key=state`.
- Loss code dereferencing missing robot/action scale config.
- Importing or shadowing whole LIFT forks at runtime.
