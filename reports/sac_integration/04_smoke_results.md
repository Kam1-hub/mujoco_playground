# Smoke Results

Status: TODO for next agent.

## Validation Ladder

| Level | Command | Status | Notes |
|---|---|---:|---|
| 0 | `python -m compileall g1_env learning` | NOT RUN | |
| 0 | `python -c "import g1_env; print(g1_env.registry.ALL_ENVS)"` | NOT RUN | |
| 1 | `python scripts/check_g1_env_api.py --env_name G1JoystickFlatTerrain` | NOT RUN | |
| 2 | SAC dry run | NOT RUN | |
| 3 | CPU tiny smoke | NOT RUN | |
| 4 | GPU smoke | NOT RUN | Requires WSL2/Linux CUDA unless proven otherwise. |

## Runtime Metrics

NOT VALIDATED.

