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


def lift_sac_config(
    env_name: str, impl: Optional[str] = None
) -> config_dict.ConfigDict:
  """Returns the main asymmetric SAC config for G1."""
  env_config = locomotion.get_default_config(env_name)
  if env_name not in ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain"):
    raise ValueError(
        f"Unsupported env: {env_name}. Only G1 environments are supported."
    )

  return config_dict.create(
      env_name=env_name,
      impl=impl or "jax",
      seed=1,
      num_timesteps=10_000,
      num_evals=2,
      episode_length=env_config.episode_length,
      action_repeat=env_config.action_repeat,
      actor_learning_rate=1e-4,
      critic_learning_rate=1e-4,
      alpha_learning_rate=3e-4,
      discounting=0.99,
      reward_scaling=1.0,
      tau=0.005,
      target_entropy_coef=0.5,
      init_log_alpha=-3.0,
      fixed_alpha=0.0,
      alpha_floor=0.0,
      alpha_loss_type="exp_alpha",
      env_feet_slip_mode=None,
      env_feet_slip_scale=None,
      env_push_enable=None,
      env_zero_command_phase_freeze=None,
      normalize_observations=True,
      deterministic_eval=True,
      num_envs=128,
      num_eval_envs=32,
      batch_size=256,
      min_replay_size=1024,
      max_replay_size=8192,
      grad_updates_per_step=2,
      policy_hidden_layer_sizes=(512, 256, 128),
      q_hidden_layer_sizes=(1024, 512, 256),
      activation="swish",
      log_std_min=-5.0,
      log_std_max=2.0,
      policy_obs_key="state",
      value_obs_key="privileged_state",
      policy_obs_size=103,
      value_obs_size=216,
      action_size=29,
      logdir="./logs/sac_lift",
      dry_run=False,
      render=False,
      use_wandb=False,
      allow_menagerie_download=False,
  )
