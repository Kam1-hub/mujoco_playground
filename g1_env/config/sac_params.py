"""SAC config helpers for G1 baselines."""

from __future__ import annotations

from typing import Optional

from ml_collections import config_dict

from g1_env._src import locomotion


def brax_sac_config(
    env_name: str, unused_impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns conservative upstream Brax SAC parameters for G1."""
  del unused_impl
  env_config = locomotion.get_default_config(env_name)
  if env_name not in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
    raise ValueError(
        f"Unsupported env: {env_name}. Only G1 environments are supported."
    )

  return config_dict.create(
      num_timesteps=10_000,
      num_evals=2,
      episode_length=env_config.episode_length,
      action_repeat=env_config.action_repeat,
      learning_rate=3e-4,
      discounting=0.99,
      reward_scaling=1.0,
      tau=0.005,
      normalize_observations=True,
      num_envs=128,
      num_eval_envs=32,
      batch_size=256,
      min_replay_size=1024,
      max_replay_size=8192,
      grad_updates_per_step=2,
      deterministic_eval=True,
      policy_obs_key="state",
      hidden_layer_sizes=(256, 256),
  )
