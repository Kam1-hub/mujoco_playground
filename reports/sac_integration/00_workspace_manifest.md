# Workspace Manifest

## Paths

| Field | Value |
|---|---|
| Project path | `D:\mujoco_playground\g1_sac_dev` |
| Source template | `D:\mujoco_playground\template` |
| LIFT repo reference | `D:\mujoco_playground\LIFT-humanoid` |
| LIFT report copy | `external_references/LIFT-humanoid-reports/` |

## Git

| Field | Value |
|---|---|
| Initial baseline commit | `f4d23d5` |
| Baseline commit message | `baseline g1 template before sac integration` |
| Working branch | `sac-integration` |
| Repo origin | None yet |

## Observed Preparation Environment

| Field | Value |
|---|---|
| OS | Windows 11 / `Microsoft Windows NT 10.0.26200.0` observed from preparation host |
| Shell | PowerShell 5.1 |
| Python | `C:\msys64\ucrt64\bin\python.exe`, Python `3.12.11` observed from preparation host |
| JAX | Not importable in preparation environment |
| Torch | Not importable in preparation environment |
| MuJoCo | Not importable in preparation environment |
| CUDA | `nvidia-smi` and `nvcc` not found in preparation environment |
| Runtime validation | Not run yet in this new project |

## Initial Source Facts

- Package name is `g1-training-env` (`pyproject.toml:6`).
- Existing scripts are `train-g1-jax` and `train-g1-rsl` (`pyproject.toml:81`, `pyproject.toml:83`).
- The template was not a git repo before this project was created.
- This workspace was copied from template and initialized as a new git repository.

## Next Agent Update Requirements

The next agent should update this manifest after installing/importing runtime dependencies or after moving to WSL2/Linux CUDA.

