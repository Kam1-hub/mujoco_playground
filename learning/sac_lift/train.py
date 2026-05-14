"""Training orchestration for the local G1 SAC baseline."""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import optax

from learning.sac_lift import checkpoint
from learning.sac_lift import config as sac_config
from learning.sac_lift import evaluator
from learning.sac_lift import losses
from learning.sac_lift import networks
from learning.sac_lift import normalizer
from learning.sac_lift import replay_buffer
from learning.sac_lift.types import SACTrainingState
from learning.sac_lift.types import Transition


DIAGNOSTIC_METRIC_NAMES = (
    "alpha",
    "log_alpha",
    "alpha_raw",
    "log_alpha_raw",
    "alpha_effective",
    "log_alpha_effective",
    "alpha_floor",
    "fixed_alpha",
    "alpha_floor_active",
    "alpha_loss_type_id",
    "alpha_log_prob",
    "target_entropy",
    "alpha_error_log_prob_plus_target",
    "alpha_error_neg_log_prob_minus_target",
    "alpha_grad_proxy_exp",
    "alpha_grad_proxy_log",
    "reward_mean",
    "done_fraction",
    "discount_mean",
    "truncation_fraction",
    "q",
    "target_q",
    "actor_policy_mean_abs_mean",
    "actor_policy_mean_abs_max",
    "actor_log_std_mean",
    "actor_log_std_min",
    "actor_log_std_max",
    "actor_policy_std_mean",
    "sampled_action_abs_mean",
    "sampled_action_saturation_fraction_095",
    "deterministic_action_abs_mean",
    "deterministic_action_saturation_fraction_095",
    "deterministic_action_l2",
    "actor_mean_l2",
    "actor_regularization_loss",
    "deterministic_action_l2_coef",
    "actor_mean_l2_coef",
)


def dry_run(config: Any) -> dict[str, Any]:
  """Initializes SAC objects without loading the MuJoCo env."""
  policy_obs_size = int(config.get("policy_obs_size", sac_config.DEFAULT_POLICY_OBS_SIZE))
  value_obs_size = int(config.get("value_obs_size", sac_config.DEFAULT_VALUE_OBS_SIZE))
  action_size = int(config.get("action_size", sac_config.DEFAULT_ACTION_SIZE))
  sac_networks = networks.make_networks(
      action_size=action_size,
      policy_hidden_layer_sizes=tuple(config.policy_hidden_layer_sizes),
      q_hidden_layer_sizes=tuple(config.q_hidden_layer_sizes),
      activation=config.activation,
      log_std_min=config.log_std_min,
      log_std_max=config.log_std_max,
  )
  policy_params, q_params = networks.init_params(
      sac_networks,
      jax.random.PRNGKey(config.seed),
      policy_obs_size,
      value_obs_size,
      action_size,
  )
  state = _init_training_state(
      config, sac_networks, policy_params, q_params, policy_obs_size, value_obs_size
  )
  replay_state = replay_buffer.init(
      config.max_replay_size, policy_obs_size, value_obs_size, action_size
  )
  dummy_batch = Transition(
      policy_obs=jnp.zeros((config.batch_size, policy_obs_size), dtype=jnp.float32),
      value_obs=jnp.zeros((config.batch_size, value_obs_size), dtype=jnp.float32),
      action=jnp.zeros((config.batch_size, action_size), dtype=jnp.float32),
      reward=jnp.zeros((config.batch_size,), dtype=jnp.float32),
      discount=jnp.ones((config.batch_size,), dtype=jnp.float32),
      next_policy_obs=jnp.zeros(
          (config.batch_size, policy_obs_size), dtype=jnp.float32
      ),
      next_value_obs=jnp.zeros(
          (config.batch_size, value_obs_size), dtype=jnp.float32
      ),
      done=jnp.zeros((config.batch_size,), dtype=jnp.float32),
      truncation=jnp.zeros((config.batch_size,), dtype=jnp.float32),
  )
  state, update_metrics = _update(
      state,
      dummy_batch,
      jax.random.PRNGKey(config.seed + 1),
      sac_networks,
      config,
  )
  metric_sums, metric_count = _accumulate_metrics(
      _new_metric_sums(), 0, update_metrics
  )
  metrics = {
      "status": "DRY_RUN_OK",
      "env_steps": 0,
      "policy_obs_shape": [policy_obs_size],
      "value_obs_shape": [value_obs_size],
      "action_shape": [action_size],
      "replay_capacity": int(replay_state.action.shape[0]),
      "gradient_steps": int(state.gradient_steps),
      **_float_metrics(update_metrics),
      **_interval_metrics(metric_sums, metric_count),
  }
  os.makedirs(config.logdir, exist_ok=True)
  ckpt_path = checkpoint.save(
      config.logdir,
      0,
      {
          "config": sac_config.to_plain_dict(config),
          "policy_params": state.policy_params,
          "q_params": state.q_params,
          "target_q_params": state.target_q_params,
          "log_alpha": state.log_alpha,
          "policy_normalizer": state.policy_normalizer,
          "value_normalizer": state.value_normalizer,
          "metrics": metrics,
      },
  )
  metrics["checkpoint"] = ckpt_path
  return metrics


def train(config: Any) -> dict[str, Any]:
  """Runs a small non-pmap SAC loop."""
  if config.dry_run:
    return dry_run(config)

  from g1_env import registry
  from g1_env import wrapper
  from g1_env._src import mjx_env

  if not config.allow_menagerie_download and not mjx_env.MENAGERIE_PATH.exists():
    raise RuntimeError(
        "mujoco_menagerie is missing at "
        f"{mjx_env.MENAGERIE_PATH}; rerun with --allow_menagerie_download "
        "only when network/file writes are authorized."
    )

  env_cfg = registry.get_default_config(config.env_name)
  env_overrides = {}
  if config.get("impl", None):
    env_overrides["impl"] = config.impl
  env_feet_slip_mode = config.get("env_feet_slip_mode", None)
  if env_feet_slip_mode is not None:
    if env_feet_slip_mode not in sac_config.FEET_SLIP_MODES:
      raise ValueError(
          "env_feet_slip_mode must be one of "
          f"{sac_config.FEET_SLIP_MODES}, got {env_feet_slip_mode!r}."
      )
    env_overrides["feet_slip_mode"] = env_feet_slip_mode
  env_feet_slip_scale = config.get("env_feet_slip_scale", None)
  if env_feet_slip_scale is not None:
    env_overrides["reward_config.scales.feet_slip"] = float(env_feet_slip_scale)
  env_push_enable = config.get("env_push_enable", None)
  if env_push_enable is not None:
    env_overrides["push_config.enable"] = bool(env_push_enable)
  env_zero_command_phase_freeze = config.get("env_zero_command_phase_freeze", None)
  if env_zero_command_phase_freeze is not None:
    env_overrides["zero_command_phase_freeze"] = bool(
        env_zero_command_phase_freeze
    )
  env_feet_air_time_command_mask = config.get(
      "env_feet_air_time_command_mask", None
  )
  if env_feet_air_time_command_mask is not None:
    env_overrides["feet_air_time_command_mask"] = bool(
        env_feet_air_time_command_mask
    )
  env_reset_joint_noise_scale = config.get("env_reset_joint_noise_scale", None)
  if env_reset_joint_noise_scale is not None:
    env_overrides["reset_joint_noise_scale"] = float(
        env_reset_joint_noise_scale
    )
  env_reset_root_qvel_scale = config.get("env_reset_root_qvel_scale", None)
  if env_reset_root_qvel_scale is not None:
    env_overrides["reset_root_qvel_scale"] = float(env_reset_root_qvel_scale)
  env_reward_action_rate_scale = config.get(
      "env_reward_action_rate_scale", None
  )
  if env_reward_action_rate_scale is not None:
    env_overrides["reward_config.scales.action_rate"] = float(
        env_reward_action_rate_scale
    )
  env_reward_ang_vel_xy_scale = config.get(
      "env_reward_ang_vel_xy_scale", None
  )
  if env_reward_ang_vel_xy_scale is not None:
    env_overrides["reward_config.scales.ang_vel_xy"] = float(
        env_reward_ang_vel_xy_scale
    )
  env = registry.load(
      config.env_name,
      config=env_cfg,
      config_overrides=env_overrides or None,
  )
  env = wrapper.wrap_for_brax_training(
      env,
      episode_length=config.episode_length,
      action_repeat=config.action_repeat,
      randomization_fn=None,
  )
  env_reset = jax.jit(env.reset)

  def env_step_once(env_state: Any, training_state: SACTrainingState, key: jax.Array):
    return _env_step(
        env.step,
        env_state,
        training_state,
        sac_networks,
        key,
        config.policy_obs_key,
        config.value_obs_key,
        config.normalize_observations,
    )

  env_step_once = jax.jit(env_step_once)

  def update_once(
      training_state: SACTrainingState,
      batch: Transition,
      key: jax.Array,
  ):
    return _update(training_state, batch, key, sac_networks, config)

  update_once = jax.jit(update_once)

  obs_size = env.observation_size
  policy_obs_size = _obs_size(obs_size, config.policy_obs_key)
  value_obs_size = _obs_size(obs_size, config.value_obs_key, fallback_key=config.policy_obs_key)
  action_size = env.action_size
  sac_networks = networks.make_networks(
      action_size=action_size,
      policy_hidden_layer_sizes=tuple(config.policy_hidden_layer_sizes),
      q_hidden_layer_sizes=tuple(config.q_hidden_layer_sizes),
      activation=config.activation,
      log_std_min=config.log_std_min,
      log_std_max=config.log_std_max,
  )
  key = jax.random.PRNGKey(config.seed)
  key, init_key, env_key = jax.random.split(key, 3)
  policy_params, q_params = networks.init_params(
      sac_networks, init_key, policy_obs_size, value_obs_size, action_size
  )
  training_state = _init_training_state(
      config, sac_networks, policy_params, q_params, policy_obs_size, value_obs_size
  )
  rb_state = replay_buffer.init(
      config.max_replay_size, policy_obs_size, value_obs_size, action_size
  )
  env_state = env_reset(jax.random.split(env_key, config.num_envs))

  start = time.monotonic()
  last_metrics: dict[str, Any] = {}
  metric_sums = _new_metric_sums()
  metric_count = 0
  actor_steps = max(1, config.num_timesteps // max(config.num_envs, 1))
  for _ in range(actor_steps):
    key, action_key, sample_key = jax.random.split(key, 3)
    transition, env_state = env_step_once(env_state, training_state, action_key)
    training_state = training_state.replace(
        policy_normalizer=normalizer.update(
            training_state.policy_normalizer, transition.policy_obs
        ),
        value_normalizer=normalizer.update(
            training_state.value_normalizer, transition.value_obs
        ),
        env_steps=training_state.env_steps + config.num_envs,
    )
    rb_state = replay_buffer.insert(rb_state, transition)
    if int(rb_state.size) >= config.min_replay_size:
      for _ in range(config.grad_updates_per_step):
        key, sample_key, update_key = jax.random.split(key, 3)
        batch = replay_buffer.sample(rb_state, sample_key, config.batch_size)
        training_state, last_metrics = update_once(training_state, batch, update_key)
        metric_sums, metric_count = _accumulate_metrics(
            metric_sums, metric_count, last_metrics
        )

  wall_time = time.monotonic() - start
  metrics = {
      "status": "TRAIN_OK",
      "env_steps": int(training_state.env_steps),
      "gradient_steps": int(training_state.gradient_steps),
      "wall_time": wall_time,
      "sps": float(training_state.env_steps) / max(wall_time, 1e-6),
      **_float_metrics(last_metrics),
      **_interval_metrics(metric_sums, metric_count),
  }
  ckpt_path = checkpoint.save(
      config.logdir,
      int(training_state.env_steps),
      {
          "config": sac_config.to_plain_dict(config),
          "policy_params": training_state.policy_params,
          "q_params": training_state.q_params,
          "target_q_params": training_state.target_q_params,
          "log_alpha": training_state.log_alpha,
          "policy_normalizer": training_state.policy_normalizer,
          "value_normalizer": training_state.value_normalizer,
          "metrics": metrics,
      },
  )
  metrics["checkpoint"] = ckpt_path
  return metrics


def main(config: Any) -> int:
  try:
    result = train(config)
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    return 2
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


def _float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
  host_metrics = jax.device_get(metrics)
  return {str(k): float(v) for k, v in host_metrics.items()}


def _new_metric_sums() -> dict[str, float]:
  return {name: 0.0 for name in DIAGNOSTIC_METRIC_NAMES}


def _accumulate_metrics(
    metric_sums: dict[str, float],
    metric_count: int,
    metrics: dict[str, Any],
) -> tuple[dict[str, float], int]:
  host_metrics = jax.device_get(metrics)
  for name in DIAGNOSTIC_METRIC_NAMES:
    if name in host_metrics:
      metric_sums[name] += float(host_metrics[name])
  return metric_sums, metric_count + 1


def _interval_metrics(metric_sums: dict[str, float], metric_count: int) -> dict[str, float]:
  if metric_count <= 0:
    return {}
  return {
      f"interval/{name}": value / metric_count
      for name, value in metric_sums.items()
  }


def _init_training_state(
    config: Any,
    sac_networks: networks.SACNetworks,
    policy_params: Any,
    q_params: Any,
    policy_obs_size: int,
    value_obs_size: int,
) -> SACTrainingState:
  del sac_networks
  policy_tx = optax.adam(config.actor_learning_rate)
  q_tx = optax.adam(config.critic_learning_rate)
  alpha_tx = optax.adam(config.alpha_learning_rate)
  log_alpha = jnp.asarray(config.init_log_alpha, dtype=jnp.float32)
  return SACTrainingState(
      policy_params=policy_params,
      q_params=q_params,
      target_q_params=q_params,
      log_alpha=log_alpha,
      policy_opt_state=policy_tx.init(policy_params),
      q_opt_state=q_tx.init(q_params),
      alpha_opt_state=alpha_tx.init(log_alpha),
      policy_normalizer=normalizer.init(policy_obs_size),
      value_normalizer=normalizer.init(value_obs_size),
      env_steps=jnp.asarray(0, dtype=jnp.int32),
      gradient_steps=jnp.asarray(0, dtype=jnp.int32),
  )


def _env_step(
    env_step: Any,
    env_state: Any,
    training_state: SACTrainingState,
    sac_networks: networks.SACNetworks,
    key: jax.Array,
    policy_obs_key: str,
    value_obs_key: str,
    normalize_observations: bool,
) -> tuple[Transition, Any]:
  policy_obs = _select_obs(env_state.obs, policy_obs_key)
  value_obs = _select_obs(env_state.obs, value_obs_key, fallback_key=policy_obs_key)
  norm_policy_obs = normalizer.normalize(
      training_state.policy_normalizer, policy_obs, normalize_observations
  )
  action, _ = networks.sample_action(
      sac_networks, training_state.policy_params, norm_policy_obs, key
  )
  next_state = env_step(env_state, action)
  next_policy_obs = _select_obs(next_state.obs, policy_obs_key)
  next_value_obs = _select_obs(next_state.obs, value_obs_key, fallback_key=policy_obs_key)
  truncation = _truncation(next_state.info, next_state.done)
  transition = Transition(
      policy_obs=policy_obs,
      value_obs=value_obs,
      action=action,
      reward=next_state.reward,
      discount=1.0 - next_state.done,
      next_policy_obs=next_policy_obs,
      next_value_obs=next_value_obs,
      done=next_state.done,
      truncation=truncation,
  )
  return transition, next_state


def _update(
    training_state: SACTrainingState,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    config: Any,
) -> tuple[SACTrainingState, dict[str, Any]]:
  policy_tx = optax.adam(config.actor_learning_rate)
  q_tx = optax.adam(config.critic_learning_rate)
  alpha_tx = optax.adam(config.alpha_learning_rate)
  alpha_key, critic_key, actor_key = jax.random.split(key, 3)
  target_entropy = -float(config.target_entropy_coef) * float(batch.action.shape[-1])
  fixed_alpha = float(config.get("fixed_alpha", sac_config.DEFAULT_FIXED_ALPHA))
  alpha_floor = float(config.get("alpha_floor", sac_config.DEFAULT_ALPHA_FLOOR))
  alpha_loss_type = str(
      config.get("alpha_loss_type", sac_config.DEFAULT_ALPHA_LOSS_TYPE)
  )
  alpha_loss_type_id = sac_config.ALPHA_LOSS_TYPE_IDS[alpha_loss_type]
  alpha_effective = losses.alpha_values(
      training_state.log_alpha, fixed_alpha, alpha_floor
  )["alpha_effective"]

  (q_loss, q_metrics), q_grads = jax.value_and_grad(losses.critic_loss, has_aux=True)(
      training_state.q_params,
      training_state.target_q_params,
      training_state.policy_params,
      alpha_effective,
      training_state.policy_normalizer,
      training_state.value_normalizer,
      batch,
      critic_key,
      sac_networks,
      config.reward_scaling,
      config.discounting,
      config.normalize_observations,
  )
  q_updates, q_opt_state = q_tx.update(
      q_grads, training_state.q_opt_state, training_state.q_params
  )
  q_params = optax.apply_updates(training_state.q_params, q_updates)

  (actor_loss, actor_metrics), actor_grads = jax.value_and_grad(
      losses.actor_loss, has_aux=True
  )(
      training_state.policy_params,
      q_params,
      alpha_effective,
      training_state.policy_normalizer,
      training_state.value_normalizer,
      batch,
      actor_key,
      sac_networks,
      config.normalize_observations,
      float(
          config.get(
              "deterministic_action_l2_coef",
              sac_config.DEFAULT_DETERMINISTIC_ACTION_L2_COEF,
          )
      ),
      float(config.get("actor_mean_l2_coef", sac_config.DEFAULT_ACTOR_MEAN_L2_COEF)),
  )
  policy_updates, policy_opt_state = policy_tx.update(
      actor_grads, training_state.policy_opt_state, training_state.policy_params
  )
  policy_params = optax.apply_updates(training_state.policy_params, policy_updates)

  alpha_loss_args = (
      training_state.log_alpha,
      policy_params,
      training_state.policy_normalizer,
      batch,
      alpha_key,
      sac_networks,
      target_entropy,
      config.normalize_observations,
      fixed_alpha,
      alpha_floor,
      alpha_loss_type_id,
  )
  if fixed_alpha > 0.0:
    alpha_loss, alpha_metrics = losses.alpha_loss(*alpha_loss_args)
    alpha_opt_state = training_state.alpha_opt_state
    log_alpha = training_state.log_alpha
  else:
    (alpha_loss, alpha_metrics), alpha_grads = jax.value_and_grad(
        losses.alpha_loss, has_aux=True
    )(*alpha_loss_args)
    alpha_updates, alpha_opt_state = alpha_tx.update(
        alpha_grads, training_state.alpha_opt_state, training_state.log_alpha
    )
    log_alpha = optax.apply_updates(training_state.log_alpha, alpha_updates)
  target_q_params = jax.tree.map(
      lambda target, source: target * (1.0 - config.tau) + source * config.tau,
      training_state.target_q_params,
      q_params,
  )
  new_state = training_state.replace(
      policy_params=policy_params,
      q_params=q_params,
      target_q_params=target_q_params,
      log_alpha=log_alpha,
      policy_opt_state=policy_opt_state,
      q_opt_state=q_opt_state,
      alpha_opt_state=alpha_opt_state,
      gradient_steps=training_state.gradient_steps + 1,
  )
  metrics = {
      **q_metrics,
      **actor_metrics,
      **alpha_metrics,
      "critic_loss": q_loss,
      "actor_loss": actor_loss,
      "alpha_loss": alpha_loss,
  }
  return new_state, metrics


def _select_obs(obs: Any, key: str, fallback_key: str | None = None) -> jnp.ndarray:
  if not isinstance(obs, dict):
    return obs
  if key in obs:
    return obs[key]
  if fallback_key is not None and fallback_key in obs:
    warnings.warn(
        f"Observation key {key!r} missing; falling back to {fallback_key!r}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return obs[fallback_key]
  raise KeyError(f"Observation key {key!r} missing from {sorted(obs.keys())}.")


def _obs_size(obs_size: Any, key: str, fallback_key: str | None = None) -> int:
  if isinstance(obs_size, dict):
    if key in obs_size:
      value = obs_size[key]
    elif fallback_key is not None and fallback_key in obs_size:
      value = obs_size[fallback_key]
    else:
      raise KeyError(f"Observation size key {key!r} missing from {sorted(obs_size.keys())}.")
  else:
    value = obs_size
  if isinstance(value, tuple):
    if len(value) != 1:
      raise ValueError(f"Expected 1D observation, got {value}.")
    return int(value[0])
  return int(value)


def _truncation(info: dict[str, Any], done: jnp.ndarray) -> jnp.ndarray:
  if "truncation" in info:
    return info["truncation"].astype(jnp.float32)
  warnings.warn(
      "state.info['truncation'] missing; synthesizing zeros.",
      RuntimeWarning,
      stacklevel=2,
  )
  return jnp.zeros_like(done, dtype=jnp.float32)
