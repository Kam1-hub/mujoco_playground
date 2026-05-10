"""Dataclasses shared by the custom SAC implementation."""

from __future__ import annotations

from typing import Any

from flax import struct
import jax.numpy as jnp


@struct.dataclass
class Transition:
  policy_obs: jnp.ndarray
  value_obs: jnp.ndarray
  action: jnp.ndarray
  reward: jnp.ndarray
  discount: jnp.ndarray
  next_policy_obs: jnp.ndarray
  next_value_obs: jnp.ndarray
  done: jnp.ndarray
  truncation: jnp.ndarray


@struct.dataclass
class SACTrainingState:
  policy_params: Any
  q_params: Any
  target_q_params: Any
  log_alpha: jnp.ndarray
  policy_opt_state: Any
  q_opt_state: Any
  alpha_opt_state: Any
  policy_normalizer: Any
  value_normalizer: Any
  env_steps: jnp.ndarray
  gradient_steps: jnp.ndarray
