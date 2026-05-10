"""SAC losses for dict-observation asymmetric actor critic."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from learning.sac_lift import networks
from learning.sac_lift import normalizer
from learning.sac_lift.types import Transition


def critic_loss(
    q_params: Any,
    target_q_params: Any,
    policy_params: Any,
    log_alpha: jnp.ndarray,
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
  next_v = jnp.min(next_q, axis=-1) - jnp.exp(log_alpha) * next_log_prob
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
      "truncation_fraction": jnp.mean(batch.truncation),
  }
  return loss, metrics


def actor_loss(
    policy_params: Any,
    q_params: Any,
    log_alpha: jnp.ndarray,
    policy_normalizer: normalizer.RunningStats,
    value_normalizer: normalizer.RunningStats,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    normalize_observations: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  policy_obs = normalizer.normalize(
      policy_normalizer, batch.policy_obs, normalize_observations
  )
  value_obs = normalizer.normalize(
      value_normalizer, batch.value_obs, normalize_observations
  )
  action, log_prob = networks.sample_action(
      sac_networks, policy_params, policy_obs, key
  )
  q = networks.q_values(sac_networks, q_params, value_obs, action)
  min_q = jnp.min(q, axis=-1)
  loss = jnp.mean(jnp.exp(log_alpha) * log_prob - min_q)
  metrics = {
      "actor_loss": loss,
      "policy_log_prob": jnp.mean(log_prob),
      "policy_q": jnp.mean(min_q),
  }
  return loss, metrics


def alpha_loss(
    log_alpha: jnp.ndarray,
    policy_params: Any,
    policy_normalizer: normalizer.RunningStats,
    batch: Transition,
    key: jax.Array,
    sac_networks: networks.SACNetworks,
    target_entropy: float,
    normalize_observations: bool,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
  policy_obs = normalizer.normalize(
      policy_normalizer, batch.policy_obs, normalize_observations
  )
  _, log_prob = networks.sample_action(sac_networks, policy_params, policy_obs, key)
  loss = jnp.mean(
      jnp.exp(log_alpha) * jax.lax.stop_gradient(-log_prob - target_entropy)
  )
  return loss, {"alpha_loss": loss, "alpha": jnp.exp(log_alpha)}
