# Next Actions

Status: updated on 2026-05-11 after GPU migration prep package.

## Immediate State

- Route B `train-g1-sac` is locally runnable on CPU for tiny smoke.
- Env API readiness is validated for flat and rough G1 tasks.
- CPU tiny smoke passed with checkpoint `./logs/sac_lift_cpu_tiny\sac_lift_step_256.pkl`.
- GPU migration prep scripts are added.
- CPU-side migration prep checks pass: compileall, checkpoint inspection, and `gpu_preflight.py --impl jax`.
- GPU smoke is still `NOT VALIDATED`.
- CPU dev machine should not continue to larger training.

## Recommended Next Command On GPU Machine

Run this first on WSL2/Linux CUDA or another machine where CUDA JAX should be available:

```bash
python scripts/gpu_preflight.py --impl jax --require_gpu
```

If preflight passes, run the Route B 10k smoke wrapper:

```bash
bash scripts/gpu_smoke_route_b.sh
```

## Local CPU Dev Commands

These are CPU-side checks only and do not expand training:

```powershell
.\.venv\Scripts\python.exe scripts\check_sac_checkpoint.py --checkpoint .\logs\sac_lift_cpu_tiny\sac_lift_step_256.pkl
.\.venv\Scripts\python.exe scripts\gpu_preflight.py --impl jax
```

Do not pass `--require_gpu` on this CPU dev host.

## Migration Reminders

- Rebuild Python env on the GPU machine; do not copy `.venv`.
- Ensure menagerie exists at `g1_env/external_deps/mujoco_menagerie`.
- Ensure menagerie commit is `1b86ece576591213e2b666ebf59508454200ca97`.
- Keep menagerie, logs, checkpoints, and `.venv` untracked.
- Record JAX backend/devices, MuJoCo version, Brax version, SPS, losses, alpha, checkpoint path, and any full traceback.

## Recommended Follow-Ups After GPU 10k Smoke

1. Add deterministic eval rollout metrics after 10k smoke passes.
2. Run a longer sanity run only after GPU smoke is stable.
3. Compare against PPO baseline only after SAC 10k smoke has a clean report.
4. Add Route A CPU/GPU `--impl` parity only if the Brax SAC fallback route becomes necessary.

## Do Not Start Yet

- 1M training
- GPU smoke on this Windows CPU host
- domain randomization
- world model
- fine-tuning
- vision SAC
- real deployment
