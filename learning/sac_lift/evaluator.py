"""Deterministic evaluation helpers for the local SAC baseline."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from learning.sac_lift import networks
from learning.sac_lift import normalizer


def evaluate(
    env: Any,
    params: Any,
    policy_normalizer: normalizer.RunningStats,
    sac_networks: networks.SACNetworks,
    rng: jax.Array,
    policy_obs_key: str,
    episode_length: int,
    normalize_observations: bool,
) -> dict[str, float]:
  state = env.reset(rng)
  total_reward = jnp.zeros_like(state.reward)
  done_any = jnp.zeros_like(state.done)
  for _ in range(episode_length):
    policy_obs = select_obs(state.obs, policy_obs_key)
    policy_obs = normalizer.normalize(
        policy_normalizer, policy_obs, normalize_observations
    )
    action, _ = networks.sample_action(
        sac_networks, params, policy_obs, rng, deterministic=True
    )
    state = env.step(state, action)
    total_reward = total_reward + state.reward * (1.0 - done_any)
    done_any = jnp.maximum(done_any, state.done)
  return {
      "eval/episode_reward": float(jnp.mean(total_reward)),
      "eval/done_fraction": float(jnp.mean(done_any)),
  }


def select_obs(obs: Any, key: str) -> jnp.ndarray:
  if isinstance(obs, dict):
    if key not in obs:
      raise KeyError(f"Observation key {key!r} missing from {sorted(obs.keys())}.")
    return obs[key]
  return obs
