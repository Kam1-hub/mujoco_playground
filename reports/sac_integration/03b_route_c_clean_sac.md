# Route C: Clean-Room SAC Fallback

Status: NOT ENABLED.

## Reason

Route C was reserved for the case where Route A and Route B both blocked mainline progress. That condition did not occur:

- Route A is implemented as a minimal upstream Brax SAC fallback using `SelectObsWrapper(obs_key="state")`.
- Route B is implemented as the main asymmetric SAC route with actor `state` and critic `privileged_state`.
- Route B `--help` and dry-run pass.

## Current Blocker

The remaining blocker is not algorithm structure. Env load/reset/step and training smoke are blocked because `g1_env\external_deps\mujoco_menagerie` is absent and hidden network clone is disabled by default.

## Next Action

Do not implement Route C unless Route B fails after assets/runtime are available and a full traceback points to a Route B design or implementation issue that cannot be fixed directly.
