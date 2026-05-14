"""SAC losses for dict-observation asymmetric actor critic."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from learning.sac_lift import distributions
from learning.sac_lift import networks
from learning.sac_lift import normalizer
from learning.sac_lift.types import Transition


def critic_loss(
    q_params: Any,
    target_q_params: Any,
    policy_params: Any,
    alpha_effective: jnp.ndarray,
    policy_normalizer: normalizer.RunningStats,
    value_normalizer: normalizer.RunningStats,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    reward_scaling: float,
    discounting: float,
    normalize_observations: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  policy_next_obs = normalizer.normalize(
      policy_normalizer, batch.next_policy_obs, normalize_observations
  )
  value_obs = normalizer.normalize(
      value_normalizer, batch.value_obs, normalize_observations
  )
  next_value_obs = normalizer.normalize(
      value_normalizer, batch.next_value_obs, normalize_observations
  )
  next_action, next_log_prob = networks.sample_action(
      sac_networks, policy_params, policy_next_obs, key
  )
  next_q = networks.q_values(sac_networks, target_q_params, next_value_obs, next_action)
  next_v = jnp.min(next_q, axis=-1) - alpha_effective * next_log_prob
  target_q = jax.lax.stop_gradient(
      batch.reward * reward_scaling + batch.discount * discounting * next_v
  )
  q = networks.q_values(sac_networks, q_params, value_obs, batch.action)
  q_error = q - target_q[..., None]
  truncation_mask = 1.0 - batch.truncation[..., None]
  loss = 0.5 * jnp.mean(jnp.square(q_error) * truncation_mask)
  metrics = {
      "critic_loss": loss,
      "target_q": jnp.mean(target_q),
      "q": jnp.mean(q),
      "reward_mean": jnp.mean(batch.reward),
      "discount_mean": jnp.mean(batch.discount),
      "done_fraction": jnp.mean(batch.done),
      "truncation_fraction": jnp.mean(batch.truncation),
  }
  return loss, metrics


def actor_loss(
    policy_params: Any,
    q_params: Any,
    alpha_effective: jnp.ndarray,
    policy_normalizer: normalizer.RunningStats,
    value_normalizer: normalizer.RunningStats,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    normalize_observations: bool,
    deterministic_action_l2_coef: float = 0.0,
    actor_mean_l2_coef: float = 0.0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  policy_obs = normalizer.normalize(
      policy_normalizer, batch.policy_obs, normalize_observations
  )
  value_obs = normalizer.normalize(
      value_normalizer, batch.value_obs, normalize_observations
  )
  mean, log_std = sac_networks.actor.apply(policy_params, policy_obs)
  action, log_prob = distributions.sample_tanh_normal(mean, log_std, key)
  deterministic_action = jnp.tanh(mean)
  std = jnp.exp(log_std)
  q = networks.q_values(sac_networks, q_params, value_obs, action)
  min_q = jnp.min(q, axis=-1)
  base_loss = jnp.mean(alpha_effective * log_prob - min_q)
  deterministic_action_l2 = jnp.mean(jnp.square(deterministic_action))
  actor_mean_l2 = jnp.mean(jnp.square(mean))
  deterministic_action_l2_coef_value = jnp.asarray(
      deterministic_action_l2_coef, dtype=base_loss.dtype
  )
  actor_mean_l2_coef_value = jnp.asarray(actor_mean_l2_coef, dtype=base_loss.dtype)
  actor_regularization_loss = (
      deterministic_action_l2_coef_value * deterministic_action_l2
      + actor_mean_l2_coef_value * actor_mean_l2
  )
  loss = base_loss + actor_regularization_loss
  metrics = {
      "actor_loss": loss,
      "policy_log_prob": jnp.mean(log_prob),
      "policy_q": jnp.mean(min_q),
      "deterministic_action_l2": deterministic_action_l2,
      "actor_mean_l2": actor_mean_l2,
      "actor_regularization_loss": actor_regularization_loss,
      "deterministic_action_l2_coef": deterministic_action_l2_coef_value,
      "actor_mean_l2_coef": actor_mean_l2_coef_value,
      "actor_policy_mean_abs_mean": jnp.mean(jnp.abs(mean)),
      "actor_policy_mean_abs_max": jnp.max(jnp.abs(mean)),
      "actor_log_std_mean": jnp.mean(log_std),
      "actor_log_std_min": jnp.min(log_std),
      "actor_log_std_max": jnp.max(log_std),
      "actor_policy_std_mean": jnp.mean(std),
      "sampled_action_abs_mean": jnp.mean(jnp.abs(action)),
      "sampled_action_saturation_fraction_095": jnp.mean(
          (jnp.abs(action) >= 0.95).astype(action.dtype)
      ),
      "deterministic_action_abs_mean": jnp.mean(jnp.abs(deterministic_action)),
      "deterministic_action_saturation_fraction_095": jnp.mean(
          (jnp.abs(deterministic_action) >= 0.95).astype(deterministic_action.dtype)
      ),
  }
  return loss, metrics


def alpha_values(
    log_alpha: jnp.ndarray,
    fixed_alpha: float = 0.0,
    alpha_floor: float = 0.0,
) -> dict[str, jnp.ndarray]:
  """Returns raw and effective temperature values for diagnostics."""
  raw_alpha = jnp.exp(log_alpha)
  fixed_alpha_value = jnp.asarray(fixed_alpha, dtype=raw_alpha.dtype)
  alpha_floor_value = jnp.asarray(alpha_floor, dtype=raw_alpha.dtype)
  use_fixed_alpha = fixed_alpha_value > 0.0
  use_alpha_floor = alpha_floor_value > 0.0
  floored_alpha = jnp.maximum(raw_alpha, alpha_floor_value)
  effective_alpha = jnp.where(
      use_fixed_alpha,
      fixed_alpha_value,
      jnp.where(use_alpha_floor, floored_alpha, raw_alpha),
  )
  alpha_floor_active = jnp.where(
      jnp.logical_and(use_alpha_floor, raw_alpha < alpha_floor_value),
      1.0,
      0.0,
  )
  alpha_floor_active = jnp.where(use_fixed_alpha, 0.0, alpha_floor_active)
  return {
      "alpha_raw": raw_alpha,
      "log_alpha_raw": log_alpha,
      "alpha_effective": effective_alpha,
      "log_alpha_effective": jnp.log(jnp.maximum(effective_alpha, 1e-12)),
      "fixed_alpha": fixed_alpha_value,
      "alpha_floor": alpha_floor_value,
      "alpha_floor_active": alpha_floor_active.astype(raw_alpha.dtype),
  }


def alpha_loss(
    log_alpha: jnp.ndarray,
    policy_params: Any,
    policy_normalizer: normalizer.RunningStats,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    target_entropy: float,
    normalize_observations: bool,
    fixed_alpha: float = 0.0,
    alpha_floor: float = 0.0,
    alpha_loss_type_id: int = 0,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  policy_obs = normalizer.normalize(
      policy_normalizer, batch.policy_obs, normalize_observations
  )
  _, log_prob = networks.sample_action(sac_networks, policy_params, policy_obs, key)
  alpha_metrics = alpha_values(log_alpha, fixed_alpha, alpha_floor)
  alpha = alpha_metrics["alpha_raw"]
  target_entropy_value = jnp.asarray(target_entropy, dtype=log_prob.dtype)
  mean_log_prob = jnp.mean(log_prob)
  alpha_error = jnp.mean(log_prob + target_entropy_value)
  brax_error = jnp.mean(-log_prob - target_entropy_value)
  error = jax.lax.stop_gradient(-log_prob - target_entropy_value)
  exp_alpha_loss = jnp.mean(alpha * error)
  log_alpha_loss = jnp.mean(log_alpha * error)
  alpha_loss_type_value = jnp.asarray(alpha_loss_type_id, dtype=jnp.int32)
  loss = jnp.where(alpha_loss_type_value == 1, log_alpha_loss, exp_alpha_loss)
  return loss, {
      "alpha_loss": loss,
      "alpha": alpha,
      "log_alpha": log_alpha,
      **alpha_metrics,
      "target_entropy": target_entropy_value,
      "alpha_log_prob": mean_log_prob,
      "alpha_error_log_prob_plus_target": alpha_error,
      "alpha_error_neg_log_prob_minus_target": brax_error,
      "alpha_grad_proxy_exp": alpha * brax_error,
      "alpha_grad_proxy_log": brax_error,
      "alpha_loss_type_id": alpha_loss_type_value,
  }
