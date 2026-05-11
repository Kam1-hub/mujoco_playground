# Next Actions

Status: updated on 2026-05-11 after CPU tiny SAC smoke passed.

## Immediate State

- Route B `train-g1-sac` is locally runnable on CPU for tiny smoke.
- Env API readiness is validated for flat and rough G1 tasks.
- Actor obs defaults to `obs["state"]`.
- Critic obs defaults to `obs["privileged_state"]`.
- Actions are tanh-normalized and passed directly to the env; G1 applies `action_scale` internally, so no external double scaling is used.
- Missing truncation info is handled as zeros and reported through `truncation_fraction`.
- Checkpoints are saved under ignored `logs/`.

## Next Concrete Commands

Run this only in a WSL2/Linux CUDA or otherwise verified CUDA JAX runtime:

```powershell
.\.venv\Scripts\python.exe -m learning.train_jax_sac_lift --env_name G1JoystickFlatTerrain --num_timesteps 10000 --num_envs 128 --num_eval_envs 32 --batch_size 256 --min_replay_size 1024 --max_replay_size 8192 --grad_updates_per_step 2 --render False --use_wandb False --logdir ./logs/sac_lift_smoke
```

Before that command, verify CUDA visibility:

```powershell
.\.venv\Scripts\python.exe -c "import jax; print(jax.default_backend()); print(jax.devices())"
nvidia-smi
```

## Recommended Follow-Ups

1. Add the same explicit `--impl` route to `learning/train_jax_sac_brax.py` if Route A must run on CPU.
2. Run Route B 10k GPU smoke in WSL2/Linux CUDA and record loss, alpha, SPS, checkpoint, and eval metrics.
3. Add deterministic eval rollout metrics after 10k smoke passes.
4. Run a longer sanity run only after GPU smoke is stable.
5. Compare against PPO baseline only after SAC 10k smoke has a clean report.

## Do Not Start Yet

- 1M training
- GPU smoke on this Windows host without CUDA JAX
- domain randomization
- world model
- fine-tuning
- vision SAC
- real deployment
