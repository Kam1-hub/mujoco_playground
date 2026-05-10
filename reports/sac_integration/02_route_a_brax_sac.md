# Route A: Minimal Brax SAC

Status: TODO for next agent.

## Goal

Find the fastest minimal SAC path using the installed/upstream Brax stack. This route is allowed to be symmetric first:

- actor obs = `state`
- critic obs = `state`

## Required Files

- `learning/sac_wrappers.py`
- `learning/train_jax_sac_brax.py`
- optional `g1_env/config/sac_params.py`

## Validation

```bash
python -m learning.train_jax_sac_brax --help
python -m learning.train_jax_sac_brax --env_name G1JoystickFlatTerrain --num_timesteps 10000 --num_envs 128 --num_eval_envs 32 --batch_size 256 --min_replay_size 1024 --max_replay_size 8192 --grad_updates_per_step 2 --use_wandb False --render False
```

Runtime status: NOT VALIDATED.

