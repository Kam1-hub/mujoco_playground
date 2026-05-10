# Scan Manifest

All source citations use paths relative to repo root `D:\mujoco_playground\LIFT-humanoid`.

| Field | Value |
|---|---|
| Repo URL | `https://github.com/bigai-ai/LIFT-humanoid` |
| Local remote | `origin https://github.com/bigai-ai/LIFT-humanoid` |
| Branch | `main` |
| Commit SHA | `32fab578446a3a7d93ad877cb1b33eab9745df25` |
| Scan date | `2026-05-10T22:37:21+08:00` |
| OS | `Microsoft Windows NT 10.0.26200.0`, PowerShell `5.1.26100.8115` |
| Python observed | `C:\msys64\ucrt64\bin\python.exe`, Python `3.12.11` |
| CUDA observed | `nvidia-smi` not found; `nvcc` not found; CUDA runtime UNKNOWN |
| JAX observed | Not installed in current Python env |
| Torch observed | Not installed in current Python env |
| MuJoCo observed | Not installed in current Python env |
| Gym/Gymnasium observed | Not installed in current Python env |
| Analysis mode | Static analysis plus environment/version probes. No training run, no smoke run, no import-valid environment run. |
| Validation status | `minimal_static_check.sh` and `minimal_smoke_run.sh` are generated but NOT VALIDATED in this audit. |

## Scanned Directories

- Top-level scripts/configs: `README.md`, `setup.py`, `requirements.txt`, `lift_configs.py`, `train_in_mujoco_playground.py`, `train_in_mujoco_playground_optuna.py`, `train_wm_from_file.py`, `finetune.py`.
- Learner and utility code: `policy_pretrain/`, `world_model/`, `lift_utils/`.
- Local MuJoCo Playground fork: `mujoco_playground/`.
- Local Brax-style package/fork: `brax_env/`.
- Deployment-related code was inspected where relevant, especially `mujoco_env/` by filename and `mujoco_playground/mujoco_playground/_src/wrapper_torch.py`.

## Ignored Or Lightly Scanned

- `.git/`, `__pycache__/`, generated logs/checkpoints/output folders.
- Binary assets, meshes, images, videos, and notebooks were not semantically audited.
- Tests were not executed.
- External upstream repositories were not fetched; upstream commit identity is UNKNOWN unless stated by local metadata.

## Commands Used

Representative commands:

```powershell
git remote -v
git branch --show-current
git rev-parse HEAD
git status --short
rg --files -g '!**/__pycache__/**' -g '!**/.git/**'
rg -n "TD3|td3|REDQ|redq|FastTD3|PPO|SAC|GAIL|AIRL|FAIRL|DIAYN|MAPPO"
python --version
python -c "import importlib.util ..."
nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
nvcc --version
```

`rg --files` returned 722 files after excluding `.git` and `__pycache__`.

## Static Evidence Anchors

- Top-level package declares `lift` version `0.1.0` and Apache 2.0 (`setup.py:10`, `setup.py:11`, `setup.py:15`).
- The repo depends on JAX/JAXlib at package level (`setup.py:21`, `setup.py:22`) and pins CUDA JAX/Torch packages in `requirements.txt` (`requirements.txt:16`, `requirements.txt:19`, `requirements.txt:20`).
- The local MuJoCo Playground package is named `playground`, version `0.0.4`, Python `>=3.10`, with `mujoco-mjx>=3.2.7` and `mujoco>=3.2.7` dependencies (`mujoco_playground/pyproject.toml:6`, `mujoco_playground/pyproject.toml:7`, `mujoco_playground/pyproject.toml:12`, `mujoco_playground/pyproject.toml:32`, `mujoco_playground/pyproject.toml:33`).
- The local Brax-style package is named `brax_env`, version `0.1.0`, and includes packages found by `find_packages()` (`brax_env/setup.py:25`, `brax_env/setup.py:26`, `brax_env/setup.py:27`, `brax_env/setup.py:31`).

