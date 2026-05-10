"""Uniform replay buffer for SAC."""

from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp

from learning.sac_lift.types import Transition


@struct.dataclass
class ReplayBufferState:
  policy_obs: jnp.ndarray
  value_obs: jnp.ndarray
  action: jnp.ndarray
  reward: jnp.ndarray
  discount: jnp.ndarray
  next_policy_obs: jnp.ndarray
  next_value_obs: jnp.ndarray
  done: jnp.ndarray
  truncation: jnp.ndarray
  insert_position: jnp.ndarray
  size: jnp.ndarray


def init(
    capacity: int,
    policy_obs_size: int,
    value_obs_size: int,
    action_size: int,
) -> ReplayBufferState:
  return ReplayBufferState(
      policy_obs=jnp.zeros((capacity, policy_obs_size), dtype=jnp.float32),
      value_obs=jnp.zeros((capacity, value_obs_size), dtype=jnp.float32),
      action=jnp.zeros((capacity, action_size), dtype=jnp.float32),
      reward=jnp.zeros((capacity,), dtype=jnp.float32),
      discount=jnp.zeros((capacity,), dtype=jnp.float32),
      next_policy_obs=jnp.zeros((capacity, policy_obs_size), dtype=jnp.float32),
      next_value_obs=jnp.zeros((capacity, value_obs_size), dtype=jnp.float32),
      done=jnp.zeros((capacity,), dtype=jnp.float32),
      truncation=jnp.zeros((capacity,), dtype=jnp.float32),
      insert_position=jnp.asarray(0, dtype=jnp.int32),
      size=jnp.asarray(0, dtype=jnp.int32),
  )


def insert(state: ReplayBufferState, transition: Transition) -> ReplayBufferState:
  transition = jax.tree.map(_flatten_batch, transition)
  batch_size = transition.action.shape[0]
  capacity = state.action.shape[0]
  indices = (jnp.arange(batch_size) + state.insert_position) % capacity
  new_insert = (state.insert_position + batch_size) % capacity
  new_size = jnp.minimum(state.size + batch_size, capacity)
  return state.replace(
      policy_obs=state.policy_obs.at[indices].set(transition.policy_obs),
      value_obs=state.value_obs.at[indices].set(transition.value_obs),
      action=state.action.at[indices].set(transition.action),
      reward=state.reward.at[indices].set(transition.reward),
      discount=state.discount.at[indices].set(transition.discount),
      next_policy_obs=state.next_policy_obs.at[indices].set(
          transition.next_policy_obs
      ),
      next_value_obs=state.next_value_obs.at[indices].set(
          transition.next_value_obs
      ),
      done=state.done.at[indices].set(transition.done),
      truncation=state.truncation.at[indices].set(transition.truncation),
      insert_position=new_insert,
      size=new_size,
  )


def sample(
    state: ReplayBufferState, key: jax.Array, batch_size: int
) -> Transition:
  max_index = jnp.maximum(state.size, 1)
  indices = jax.random.randint(key, (batch_size,), 0, max_index)
  return Transition(
      policy_obs=state.policy_obs[indices],
      value_obs=state.value_obs[indices],
      action=state.action[indices],
      reward=state.reward[indices],
      discount=state.discount[indices],
      next_policy_obs=state.next_policy_obs[indices],
      next_value_obs=state.next_value_obs[indices],
      done=state.done[indices],
      truncation=state.truncation[indices],
  )


def _flatten_batch(value: jnp.ndarray) -> jnp.ndarray:
  value = jnp.asarray(value)
  if value.ndim == 0:
    return value.reshape((1,))
  if value.ndim == 1:
    return value.reshape((-1,))
  return value.reshape((-1, value.shape[-1]))
