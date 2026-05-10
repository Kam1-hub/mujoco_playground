"""Tanh-squashed Gaussian distribution helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

LOG_2PI = jnp.log(2.0 * jnp.pi)


def sample_tanh_normal(
    mean: jnp.ndarray,
    log_std: jnp.ndarray,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Samples an action and returns the corrected log probability."""
  std = jnp.exp(log_std)
  noise = jax.random.normal(key, mean.shape)
  pre_tanh = mean + std * noise
  action = jnp.tanh(pre_tanh)
  log_prob = normal_log_prob(pre_tanh, mean, log_std)
  log_prob -= jnp.log(jnp.clip(1.0 - jnp.square(action), min=1e-6))
  return action, jnp.sum(log_prob, axis=-1)


def mode_tanh_normal(mean: jnp.ndarray) -> jnp.ndarray:
  return jnp.tanh(mean)


def normal_log_prob(
    value: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
  inv_std = jnp.exp(-log_std)
  scaled = (value - mean) * inv_std
  return -0.5 * jnp.square(scaled) - log_std - 0.5 * LOG_2PI
