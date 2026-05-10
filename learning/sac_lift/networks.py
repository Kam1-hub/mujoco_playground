"""Flax networks for asymmetric SAC."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple

from flax import linen as nn
import jax
import jax.numpy as jnp

from learning.sac_lift import distributions


def _activation(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
  if name == "relu":
    return nn.relu
  if name == "tanh":
    return nn.tanh
  if name in ("swish", "silu"):
    return nn.swish
  if name == "elu":
    return nn.elu
  raise ValueError(f"Unsupported activation {name!r}.")


class MLP(nn.Module):
  hidden_layer_sizes: Sequence[int]
  activation: str = "swish"
  activate_final: bool = False

  @nn.compact
  def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
    act = _activation(self.activation)
    for i, size in enumerate(self.hidden_layer_sizes):
      x = nn.Dense(size)(x)
      if i != len(self.hidden_layer_sizes) - 1 or self.activate_final:
        x = act(x)
    return x


class TanhGaussianActor(nn.Module):
  action_size: int
  hidden_layer_sizes: Sequence[int]
  activation: str = "swish"
  log_std_min: float = -5.0
  log_std_max: float = 2.0

  @nn.compact
  def __call__(self, obs: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    x = MLP(self.hidden_layer_sizes, activation=self.activation)(obs)
    mean = nn.Dense(self.action_size)(x)
    log_std = nn.Dense(self.action_size)(x)
    log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
    return mean, log_std


class TwinQ(nn.Module):
  hidden_layer_sizes: Sequence[int]
  activation: str = "swish"

  @nn.compact
  def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
    x = jnp.concatenate([obs, action], axis=-1)
    q1 = MLP((*self.hidden_layer_sizes, 1), activation=self.activation)(x)
    q2 = MLP((*self.hidden_layer_sizes, 1), activation=self.activation)(x)
    return jnp.concatenate([q1, q2], axis=-1)


class SACNetworks(NamedTuple):
  actor: TanhGaussianActor
  critic: TwinQ


def make_networks(
    action_size: int,
    policy_hidden_layer_sizes: Sequence[int],
    q_hidden_layer_sizes: Sequence[int],
    activation: str = "swish",
    log_std_min: float = -5.0,
    log_std_max: float = 2.0,
) -> SACNetworks:
  return SACNetworks(
      actor=TanhGaussianActor(
          action_size=action_size,
          hidden_layer_sizes=tuple(policy_hidden_layer_sizes),
          activation=activation,
          log_std_min=log_std_min,
          log_std_max=log_std_max,
      ),
      critic=TwinQ(
          hidden_layer_sizes=tuple(q_hidden_layer_sizes),
          activation=activation,
      ),
  )


def init_params(
    networks: SACNetworks,
    key: jax.Array,
    policy_obs_size: int,
    value_obs_size: int,
    action_size: int,
) -> tuple[object, object]:
  policy_key, q_key = jax.random.split(key)
  policy_params = networks.actor.init(
      policy_key, jnp.zeros((1, policy_obs_size), dtype=jnp.float32)
  )
  q_params = networks.critic.init(
      q_key,
      jnp.zeros((1, value_obs_size), dtype=jnp.float32),
      jnp.zeros((1, action_size), dtype=jnp.float32),
  )
  return policy_params, q_params


def sample_action(
    networks: SACNetworks,
    policy_params: object,
    obs: jnp.ndarray,
    key: jax.Array,
    deterministic: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  mean, log_std = networks.actor.apply(policy_params, obs)
  if deterministic:
    action = distributions.mode_tanh_normal(mean)
    log_prob = jnp.zeros(obs.shape[:-1], dtype=obs.dtype)
    return action, log_prob
  return distributions.sample_tanh_normal(mean, log_std, key)


def q_values(
    networks: SACNetworks,
    q_params: object,
    value_obs: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
  return networks.critic.apply(q_params, value_obs, action)
