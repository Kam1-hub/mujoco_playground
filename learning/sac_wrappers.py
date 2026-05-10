"""Small wrappers used by SAC entry points."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from g1_env._src import mjx_env
from g1_env import wrapper


class SelectObsWrapper(wrapper.Wrapper):
  """Selects one key from dict observations.

  Upstream Brax SAC only accepts array observations. G1 exposes a dict with
  `state` and `privileged_state`, so Route A uses this wrapper to produce a
  symmetric SAC baseline on `state`.
  """

  def __init__(self, env: mjx_env.MjxEnv, obs_key: str = "state"):
    super().__init__(env)
    self._obs_key = obs_key

  @property
  def observation_size(self) -> mjx_env.ObservationSize:
    obs_size = self.env.observation_size
    if isinstance(obs_size, Mapping):
      if self._obs_key not in obs_size:
        raise KeyError(
            f"Observation key {self._obs_key!r} not found in observation_size "
            f"keys {sorted(obs_size.keys())}."
        )
      return obs_size[self._obs_key]
    return obs_size

  def reset(self, rng: Any) -> mjx_env.State:
    return self._select(self.env.reset(rng))

  def step(self, state: mjx_env.State, action: Any) -> mjx_env.State:
    return self._select(self.env.step(state, action))

  def _select(self, state: mjx_env.State) -> mjx_env.State:
    obs = state.obs
    if not isinstance(obs, Mapping):
      return state
    if self._obs_key not in obs:
      raise KeyError(
          f"Observation key {self._obs_key!r} not found in obs keys "
          f"{sorted(obs.keys())}."
      )
    return state.replace(obs=obs[self._obs_key])
