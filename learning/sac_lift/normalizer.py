"""Running mean/std normalization utilities."""

from __future__ import annotations

from flax import struct
import jax.numpy as jnp


@struct.dataclass
class RunningStats:
  mean: jnp.ndarray
  var: jnp.ndarray
  count: jnp.ndarray


def init(size: int, epsilon: float = 1e-4) -> RunningStats:
  return RunningStats(
      mean=jnp.zeros((size,), dtype=jnp.float32),
      var=jnp.ones((size,), dtype=jnp.float32),
      count=jnp.asarray(epsilon, dtype=jnp.float32),
  )


def update(stats: RunningStats, values: jnp.ndarray) -> RunningStats:
  values = jnp.asarray(values, dtype=jnp.float32)
  values = values.reshape((-1, values.shape[-1]))
  batch_count = jnp.asarray(values.shape[0], dtype=jnp.float32)
  batch_mean = jnp.mean(values, axis=0)
  batch_var = jnp.var(values, axis=0)

  delta = batch_mean - stats.mean
  total_count = stats.count + batch_count
  new_mean = stats.mean + delta * batch_count / total_count
  m_a = stats.var * stats.count
  m_b = batch_var * batch_count
  m2 = m_a + m_b + jnp.square(delta) * stats.count * batch_count / total_count
  new_var = m2 / total_count
  return RunningStats(new_mean, jnp.maximum(new_var, 1e-6), total_count)


def normalize(
    stats: RunningStats,
    values: jnp.ndarray,
    enabled: bool = True,
    clip: float = 5.0,
) -> jnp.ndarray:
  if not enabled:
    return values
  values = (values - stats.mean) / jnp.sqrt(stats.var + 1e-6)
  return jnp.clip(values, -clip, clip)
