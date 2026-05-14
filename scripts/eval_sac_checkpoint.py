"""Run bounded evaluation for a Route B SAC checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

from g1_env import registry
from g1_env import wrapper
from g1_env.config import sac_params
from learning.sac_lift import checkpoint as sac_checkpoint
from learning.sac_lift import distributions
from learning.sac_lift import evaluator
from learning.sac_lift import networks
from learning.sac_lift import normalizer


def _str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ("true", "1", "yes", "y", "on"):
    return True
  if value in ("false", "0", "no", "n", "off"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint",
      required=True,
      help="Path to a Route B SAC pickle checkpoint.",
  )
  parser.add_argument("--env_name", default=None)
  parser.add_argument("--impl", default=None)
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num_eval_envs", type=int, default=16)
  parser.add_argument("--episode_length", type=int, default=None)
  parser.add_argument("--render", type=_str_to_bool, default=False)
  parser.add_argument(
      "--policy_mode",
      choices=("deterministic", "stochastic", "both"),
      default="deterministic",
      help=(
          "Evaluate tanh(mean), sampled stochastic actions, or both. The "
          "default deterministic mode preserves prior behavior."
      ),
  )
  parser.add_argument(
      "--action_diagnostics",
      action="store_true",
      help="Include actor mean/log_std and per-dimension action summaries.",
  )
  parser.add_argument(
      "--reward_components",
      action="store_true",
      help="Include per-component episode sums from state.metrics reward/* keys.",
  )
  parser.add_argument(
      "--top_k_actions",
      type=int,
      default=8,
      help="Number of highest-saturation action dimensions to report.",
  )
  parser.add_argument(
      "--fixed_command",
      type=_str_to_bool,
      default=False,
      help=(
          "If true, force a fixed joystick command throughout eval instead of "
          "using the command sampled by env reset/resample."
      ),
  )
  parser.add_argument(
      "--command_x",
      type=float,
      default=0.0,
      help="Fixed joystick x velocity command used with --fixed_command.",
  )
  parser.add_argument(
      "--command_y",
      type=float,
      default=0.0,
      help="Fixed joystick y velocity command used with --fixed_command.",
  )
  parser.add_argument(
      "--command_yaw",
      type=float,
      default=0.0,
      help="Fixed joystick yaw velocity command used with --fixed_command.",
  )
  parser.add_argument("--output_json", default=None)
  return parser.parse_args()


def _get(mapping: Any, key: str, default: Any = None) -> Any:
  if isinstance(mapping, Mapping):
    return mapping.get(key, default)
  return getattr(mapping, key, default)


def _require_checkpoint_ready(payload: Mapping[str, Any]) -> None:
  config = payload.get("config", {})
  normalize_observations = bool(_get(config, "normalize_observations", True))
  if "policy_params" not in payload:
    raise ValueError("checkpoint missing policy_params")
  if normalize_observations and "policy_normalizer" not in payload:
    raise ValueError(
        "checkpoint has normalize_observations=True but no policy_normalizer"
    )


def _merged_config(payload_config: Any, env_name: str, impl: str) -> dict[str, Any]:
  config = sac_params.lift_sac_config(env_name, impl=impl).to_dict()
  if isinstance(payload_config, Mapping):
    config.update(payload_config)
  config["env_name"] = env_name
  config["impl"] = impl
  return config


def _eval_env_overrides(config: Mapping[str, Any], impl: str) -> dict[str, Any]:
  overrides: dict[str, Any] = {"impl": impl}
  env_feet_slip_mode = config.get("env_feet_slip_mode")
  if env_feet_slip_mode is not None:
    overrides["feet_slip_mode"] = str(env_feet_slip_mode)
  env_feet_slip_scale = config.get("env_feet_slip_scale")
  if env_feet_slip_scale is not None:
    overrides["reward_config.scales.feet_slip"] = float(env_feet_slip_scale)
  return overrides


def _obs_size(obs_size: Any, key: str) -> int:
  if isinstance(obs_size, Mapping):
    if key not in obs_size:
      raise KeyError(f"Observation size key {key!r} missing from {sorted(obs_size)}.")
    value = obs_size[key]
  else:
    value = obs_size
  if isinstance(value, tuple):
    if len(value) != 1:
      raise ValueError(f"Expected 1D observation size, got {value}.")
    return int(value[0])
  return int(value)


def _as_float(value: Any) -> float:
  return float(jax.device_get(value))


def _as_bool(value: Any) -> bool:
  return bool(jax.device_get(value))


def _as_float_list(value: Any) -> list[float]:
  return [float(v) for v in jax.device_get(value).reshape(-1).tolist()]


def _metric_to_env_vector(value: Any, reference: jax.Array) -> jax.Array:
  value = jnp.asarray(value, dtype=jnp.float32)
  if value.shape == ():
    return jnp.broadcast_to(value, reference.shape)
  if value.shape == reference.shape:
    return value
  if value.shape[: reference.ndim] == reference.shape:
    reduce_axes = tuple(range(reference.ndim, value.ndim))
    return jnp.mean(value, axis=reduce_axes)
  return jnp.broadcast_to(jnp.mean(value), reference.shape)


def _broadcast_command(command: jax.Array, target: jax.Array) -> jax.Array:
  return jnp.broadcast_to(command, target.shape)


def _replace_obs_command(obs: Any, command: jax.Array) -> Any:
  def replace_in_array(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    command_value = jnp.broadcast_to(command, value.shape[:-1] + (3,))
    return value.at[..., 9:12].set(command_value)

  if isinstance(obs, Mapping):
    replaced = dict(obs)
    for key in ("state", "privileged_state"):
      if key in replaced:
        replaced[key] = replace_in_array(replaced[key])
    return replaced
  return replace_in_array(obs)


def _set_state_command(state: Any, command: jax.Array) -> Any:
  command = jnp.asarray(command, dtype=jnp.float32)
  if isinstance(state.info, Mapping) and "command" in state.info:
    state.info["command"] = _broadcast_command(command, state.info["command"])
  else:
    state.info["command"] = command
  return state.replace(obs=_replace_obs_command(state.obs, command))


def _reward_component_keys(env: Any, num_eval_envs: int, seed: int) -> tuple[str, ...]:
  reset_keys = jax.random.split(jax.random.PRNGKey(seed), num_eval_envs)
  state = env.reset(reset_keys)
  if not isinstance(state.metrics, Mapping):
    return ()
  return tuple(sorted(str(key) for key in state.metrics if str(key).startswith("reward/")))


def _empty_action_diag(action_size: int) -> dict[str, jax.Array]:
  zeros_dim = jnp.zeros((action_size,), dtype=jnp.float32)
  return {
      "value_count": jnp.asarray(0.0, dtype=jnp.float32),
      "per_dim_count": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_mean_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_mean_sq_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_mean_abs_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_mean_abs_max": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_log_std_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "policy_log_std_min": jnp.asarray(jnp.inf, dtype=jnp.float32),
      "policy_log_std_max": jnp.asarray(-jnp.inf, dtype=jnp.float32),
      "policy_std_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "deterministic_action_abs_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "stochastic_action_abs_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "deterministic_action_saturation_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "stochastic_action_saturation_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "deterministic_minus_stochastic_abs_sum": jnp.asarray(0.0, dtype=jnp.float32),
      "deterministic_per_dim_sum": zeros_dim,
      "deterministic_per_dim_sq_sum": zeros_dim,
      "deterministic_per_dim_abs_sum": zeros_dim,
      "deterministic_per_dim_min": jnp.full((action_size,), jnp.inf, dtype=jnp.float32),
      "deterministic_per_dim_max": jnp.full((action_size,), -jnp.inf, dtype=jnp.float32),
      "deterministic_per_dim_saturation_sum": zeros_dim,
      "stochastic_per_dim_sum": zeros_dim,
      "stochastic_per_dim_sq_sum": zeros_dim,
      "stochastic_per_dim_abs_sum": zeros_dim,
      "stochastic_per_dim_min": jnp.full((action_size,), jnp.inf, dtype=jnp.float32),
      "stochastic_per_dim_max": jnp.full((action_size,), -jnp.inf, dtype=jnp.float32),
      "stochastic_per_dim_saturation_sum": zeros_dim,
  }


def _update_action_diag(
    action_diag: dict[str, jax.Array],
    mean: jax.Array,
    log_std: jax.Array,
    deterministic_action: jax.Array,
    stochastic_action: jax.Array,
) -> dict[str, jax.Array]:
  std = jnp.exp(log_std)
  mean_abs = jnp.abs(mean)
  det_abs = jnp.abs(deterministic_action)
  stoch_abs = jnp.abs(stochastic_action)
  value_count = jnp.asarray(mean.size, dtype=jnp.float32)
  per_dim_count = jnp.asarray(mean.shape[0], dtype=jnp.float32)

  return {
      **action_diag,
      "value_count": action_diag["value_count"] + value_count,
      "per_dim_count": action_diag["per_dim_count"] + per_dim_count,
      "policy_mean_sum": action_diag["policy_mean_sum"] + jnp.sum(mean),
      "policy_mean_sq_sum": action_diag["policy_mean_sq_sum"]
      + jnp.sum(jnp.square(mean)),
      "policy_mean_abs_sum": action_diag["policy_mean_abs_sum"] + jnp.sum(mean_abs),
      "policy_mean_abs_max": jnp.maximum(
          action_diag["policy_mean_abs_max"], jnp.max(mean_abs)
      ),
      "policy_log_std_sum": action_diag["policy_log_std_sum"] + jnp.sum(log_std),
      "policy_log_std_min": jnp.minimum(
          action_diag["policy_log_std_min"], jnp.min(log_std)
      ),
      "policy_log_std_max": jnp.maximum(
          action_diag["policy_log_std_max"], jnp.max(log_std)
      ),
      "policy_std_sum": action_diag["policy_std_sum"] + jnp.sum(std),
      "deterministic_action_abs_sum": action_diag["deterministic_action_abs_sum"]
      + jnp.sum(det_abs),
      "stochastic_action_abs_sum": action_diag["stochastic_action_abs_sum"]
      + jnp.sum(stoch_abs),
      "deterministic_action_saturation_sum": action_diag[
          "deterministic_action_saturation_sum"
      ]
      + jnp.sum(det_abs > 0.95),
      "stochastic_action_saturation_sum": action_diag[
          "stochastic_action_saturation_sum"
      ]
      + jnp.sum(stoch_abs > 0.95),
      "deterministic_minus_stochastic_abs_sum": action_diag[
          "deterministic_minus_stochastic_abs_sum"
      ]
      + jnp.sum(jnp.abs(deterministic_action - stochastic_action)),
      "deterministic_per_dim_sum": action_diag["deterministic_per_dim_sum"]
      + jnp.sum(deterministic_action, axis=0),
      "deterministic_per_dim_sq_sum": action_diag["deterministic_per_dim_sq_sum"]
      + jnp.sum(jnp.square(deterministic_action), axis=0),
      "deterministic_per_dim_abs_sum": action_diag["deterministic_per_dim_abs_sum"]
      + jnp.sum(det_abs, axis=0),
      "deterministic_per_dim_min": jnp.minimum(
          action_diag["deterministic_per_dim_min"],
          jnp.min(deterministic_action, axis=0),
      ),
      "deterministic_per_dim_max": jnp.maximum(
          action_diag["deterministic_per_dim_max"],
          jnp.max(deterministic_action, axis=0),
      ),
      "deterministic_per_dim_saturation_sum": action_diag[
          "deterministic_per_dim_saturation_sum"
      ]
      + jnp.sum(det_abs > 0.95, axis=0),
      "stochastic_per_dim_sum": action_diag["stochastic_per_dim_sum"]
      + jnp.sum(stochastic_action, axis=0),
      "stochastic_per_dim_sq_sum": action_diag["stochastic_per_dim_sq_sum"]
      + jnp.sum(jnp.square(stochastic_action), axis=0),
      "stochastic_per_dim_abs_sum": action_diag["stochastic_per_dim_abs_sum"]
      + jnp.sum(stoch_abs, axis=0),
      "stochastic_per_dim_min": jnp.minimum(
          action_diag["stochastic_per_dim_min"], jnp.min(stochastic_action, axis=0)
      ),
      "stochastic_per_dim_max": jnp.maximum(
          action_diag["stochastic_per_dim_max"], jnp.max(stochastic_action, axis=0)
      ),
      "stochastic_per_dim_saturation_sum": action_diag[
          "stochastic_per_dim_saturation_sum"
      ]
      + jnp.sum(stoch_abs > 0.95, axis=0),
  }


def _finalize_action_diag(action_diag: dict[str, jax.Array]) -> dict[str, jax.Array]:
  count = jnp.maximum(action_diag["value_count"], 1.0)
  per_dim_count = jnp.maximum(action_diag["per_dim_count"], 1.0)
  policy_mean_mean = action_diag["policy_mean_sum"] / count
  policy_mean_sq_mean = action_diag["policy_mean_sq_sum"] / count

  def per_dim_stats(prefix: str) -> dict[str, jax.Array]:
    mean = action_diag[f"{prefix}_per_dim_sum"] / per_dim_count
    sq_mean = action_diag[f"{prefix}_per_dim_sq_sum"] / per_dim_count
    return {
        "mean": mean,
        "std": jnp.sqrt(jnp.maximum(sq_mean - jnp.square(mean), 0.0)),
        "min": action_diag[f"{prefix}_per_dim_min"],
        "max": action_diag[f"{prefix}_per_dim_max"],
    }

  return {
      "policy_mean_mean": policy_mean_mean,
      "policy_mean_std": jnp.sqrt(
          jnp.maximum(policy_mean_sq_mean - jnp.square(policy_mean_mean), 0.0)
      ),
      "policy_mean_abs_mean": action_diag["policy_mean_abs_sum"] / count,
      "policy_mean_abs_max": action_diag["policy_mean_abs_max"],
      "policy_log_std_mean": action_diag["policy_log_std_sum"] / count,
      "policy_log_std_min": action_diag["policy_log_std_min"],
      "policy_log_std_max": action_diag["policy_log_std_max"],
      "policy_std_mean": action_diag["policy_std_sum"] / count,
      "deterministic_action_abs_mean": action_diag[
          "deterministic_action_abs_sum"
      ]
      / count,
      "stochastic_action_abs_mean": action_diag["stochastic_action_abs_sum"] / count,
      "deterministic_action_saturation_fraction_095": action_diag[
          "deterministic_action_saturation_sum"
      ]
      / count,
      "stochastic_action_saturation_fraction_095": action_diag[
          "stochastic_action_saturation_sum"
      ]
      / count,
      "deterministic_minus_stochastic_action_abs_mean": action_diag[
          "deterministic_minus_stochastic_abs_sum"
      ]
      / count,
      "deterministic_action_per_dim": per_dim_stats("deterministic"),
      "stochastic_action_per_dim": per_dim_stats("stochastic"),
      "deterministic_action_abs_mean_per_dim": action_diag[
          "deterministic_per_dim_abs_sum"
      ]
      / per_dim_count,
      "stochastic_action_abs_mean_per_dim": action_diag[
          "stochastic_per_dim_abs_sum"
      ]
      / per_dim_count,
      "deterministic_action_saturation_fraction_095_per_dim": action_diag[
          "deterministic_per_dim_saturation_sum"
      ]
      / per_dim_count,
      "stochastic_action_saturation_fraction_095_per_dim": action_diag[
          "stochastic_per_dim_saturation_sum"
      ]
      / per_dim_count,
  }


def _format_action_diagnostics(
    diagnostics: dict[str, Any], top_k_actions: int
) -> dict[str, Any]:
  deterministic_sat = _as_float_list(
      diagnostics["deterministic_action_saturation_fraction_095_per_dim"]
  )
  stochastic_sat = _as_float_list(
      diagnostics["stochastic_action_saturation_fraction_095_per_dim"]
  )
  deterministic_abs = _as_float_list(
      diagnostics["deterministic_action_abs_mean_per_dim"]
  )
  stochastic_abs = _as_float_list(diagnostics["stochastic_action_abs_mean_per_dim"])
  top_dims = sorted(
      range(len(deterministic_sat)),
      key=lambda dim: (
          max(deterministic_sat[dim], stochastic_sat[dim]),
          max(deterministic_abs[dim], stochastic_abs[dim]),
      ),
      reverse=True,
  )[: max(top_k_actions, 0)]

  def per_dim(name: str) -> dict[str, list[float]]:
    values = diagnostics[name]
    return {
        "mean": _as_float_list(values["mean"]),
        "std": _as_float_list(values["std"]),
        "min": _as_float_list(values["min"]),
        "max": _as_float_list(values["max"]),
    }

  return {
      "policy_mean_mean": _as_float(diagnostics["policy_mean_mean"]),
      "policy_mean_std": _as_float(diagnostics["policy_mean_std"]),
      "policy_mean_abs_mean": _as_float(diagnostics["policy_mean_abs_mean"]),
      "policy_mean_abs_max": _as_float(diagnostics["policy_mean_abs_max"]),
      "policy_log_std_mean": _as_float(diagnostics["policy_log_std_mean"]),
      "policy_log_std_min": _as_float(diagnostics["policy_log_std_min"]),
      "policy_log_std_max": _as_float(diagnostics["policy_log_std_max"]),
      "policy_std_mean": _as_float(diagnostics["policy_std_mean"]),
      "deterministic_action_abs_mean": _as_float(
          diagnostics["deterministic_action_abs_mean"]
      ),
      "stochastic_action_abs_mean": _as_float(
          diagnostics["stochastic_action_abs_mean"]
      ),
      "deterministic_action_saturation_fraction_095": _as_float(
          diagnostics["deterministic_action_saturation_fraction_095"]
      ),
      "stochastic_action_saturation_fraction_095": _as_float(
          diagnostics["stochastic_action_saturation_fraction_095"]
      ),
      "deterministic_minus_stochastic_action_abs_mean": _as_float(
          diagnostics["deterministic_minus_stochastic_action_abs_mean"]
      ),
      "deterministic_action_per_dim": per_dim("deterministic_action_per_dim"),
      "stochastic_action_per_dim": per_dim("stochastic_action_per_dim"),
      "top_saturated_action_dims": [
          {
              "dim": int(dim),
              "saturation_fraction_095": max(
                  deterministic_sat[dim], stochastic_sat[dim]
              ),
              "abs_mean": max(deterministic_abs[dim], stochastic_abs[dim]),
              "deterministic_saturation_fraction_095": deterministic_sat[dim],
              "stochastic_saturation_fraction_095": stochastic_sat[dim],
              "deterministic_abs_mean": deterministic_abs[dim],
              "stochastic_abs_mean": stochastic_abs[dim],
          }
          for dim in top_dims
      ],
  }


def _format_reward_components(
    stats: dict[str, Any], component_keys: tuple[str, ...]
) -> dict[str, Any]:
  if not component_keys:
    return {"available": False, "keys": [], "components": {}}

  means = _as_float_list(stats["episode_sum_mean"])
  stds = _as_float_list(stats["episode_sum_std"])
  mins = _as_float_list(stats["episode_sum_min"])
  maxs = _as_float_list(stats["episode_sum_max"])
  return {
      "available": True,
      "keys": list(component_keys),
      "components": {
          key: {
              "episode_sum_mean": means[index],
              "episode_sum_std": stds[index],
              "episode_sum_min": mins[index],
              "episode_sum_max": maxs[index],
          }
          for index, key in enumerate(component_keys)
      },
  }


def _run_eval(
    env: Any,
    sac_networks: networks.SACNetworks,
    policy_params: Any,
    policy_normalizer: normalizer.RunningStats,
    rng: jax.Array,
    policy_obs_key: str,
    num_eval_envs: int,
    episode_length: int,
    normalize_observations: bool,
    deterministic: bool,
    action_diagnostics: bool,
    component_keys: tuple[str, ...],
    fixed_command: jax.Array | None,
) -> dict[str, Any]:
  reset_key, rollout_key = jax.random.split(rng)
  reset_keys = jax.random.split(reset_key, num_eval_envs)
  state = env.reset(reset_keys)
  if fixed_command is not None:
    state = _set_state_command(state, fixed_command)
  truncation_present = isinstance(state.info, Mapping) and "truncation" in state.info
  action_size = int(env.action_size)

  def step_fn(carry: tuple[Any, ...], _: Any) -> tuple[tuple[Any, ...], Any]:
    (
        current_state,
        total_reward,
        done_any,
        action_nan,
        reward_nan,
        obs_nan,
        truncation_sum,
        action_log_prob_sum,
        action_log_prob_count,
        action_abs_sum,
        action_count,
        action_saturation_sum,
        action_diag,
        component_sums,
        current_key,
    ) = carry
    policy_obs = evaluator.select_obs(current_state.obs, policy_obs_key)
    obs_nan = jnp.logical_or(obs_nan, jnp.any(jnp.isnan(policy_obs)))
    policy_obs = normalizer.normalize(
        policy_normalizer, policy_obs, normalize_observations
    )
    current_key, action_key = jax.random.split(current_key)
    mean, log_std = sac_networks.actor.apply(policy_params, policy_obs)
    deterministic_action = distributions.mode_tanh_normal(mean)
    stochastic_action, stochastic_log_prob = distributions.sample_tanh_normal(
        mean, log_std, action_key
    )
    if deterministic:
      action = deterministic_action
      log_prob = jnp.zeros(policy_obs.shape[:-1], dtype=policy_obs.dtype)
    else:
      action = stochastic_action
      log_prob = stochastic_log_prob
    action_diag = _update_action_diag(
        action_diag, mean, log_std, deterministic_action, stochastic_action
    )
    action_abs = jnp.abs(action)
    action_log_prob_sum = action_log_prob_sum + jnp.sum(log_prob)
    action_log_prob_count = action_log_prob_count + jnp.asarray(
        log_prob.size, dtype=jnp.float32
    )
    action_abs_sum = action_abs_sum + jnp.sum(action_abs)
    action_count = action_count + jnp.asarray(action.size, dtype=jnp.float32)
    action_saturation_sum = action_saturation_sum + jnp.sum(action_abs > 0.95)
    action_nan = jnp.logical_or(action_nan, jnp.any(jnp.isnan(action)))
    next_state = env.step(current_state, action)
    if fixed_command is not None:
      next_state = _set_state_command(next_state, fixed_command)
    reward_nan = jnp.logical_or(reward_nan, jnp.any(jnp.isnan(next_state.reward)))
    active = 1.0 - done_any
    total_reward = total_reward + next_state.reward * active
    if component_keys:
      component_values = jnp.stack(
          [
              _metric_to_env_vector(next_state.metrics[key], next_state.reward)
              for key in component_keys
          ],
          axis=0,
      )
      component_sums = component_sums + component_values * active[None, :]
    done_any = jnp.maximum(done_any, next_state.done)
    if truncation_present:
      truncation_sum = truncation_sum + jnp.sum(
          next_state.info["truncation"].astype(jnp.float32)
      )
    return (
        next_state,
        total_reward,
        done_any,
        action_nan,
        reward_nan,
        obs_nan,
        truncation_sum,
        action_log_prob_sum,
        action_log_prob_count,
        action_abs_sum,
        action_count,
        action_saturation_sum,
        action_diag,
        component_sums,
        current_key,
    ), None

  rewards = jnp.zeros_like(state.reward)
  done_any = jnp.zeros_like(state.done)
  action_diag = _empty_action_diag(action_size)
  component_sums = jnp.zeros(
      (len(component_keys),) + rewards.shape, dtype=jnp.float32
  )
  initial = (
      state,
      rewards,
      done_any,
      jnp.asarray(False),
      jnp.asarray(False),
      jnp.asarray(False),
      jnp.asarray(0.0, dtype=jnp.float32),
      jnp.asarray(0.0, dtype=jnp.float32),
      jnp.asarray(0.0, dtype=jnp.float32),
      jnp.asarray(0.0, dtype=jnp.float32),
      jnp.asarray(0.0, dtype=jnp.float32),
      jnp.asarray(0.0, dtype=jnp.float32),
      action_diag,
      component_sums,
      rollout_key,
  )
  (
      _,
      total_reward,
      done_any,
      action_nan,
      reward_nan,
      obs_nan,
      truncation_sum,
      action_log_prob_sum,
      action_log_prob_count,
      action_abs_sum,
      action_count,
      action_saturation_sum,
      action_diag,
      component_sums,
      _,
  ), _ = jax.lax.scan(step_fn, initial, None, length=episode_length)

  eval_env_steps = num_eval_envs * episode_length
  action_log_prob_count = jnp.maximum(action_log_prob_count, 1.0)
  action_count = jnp.maximum(action_count, 1.0)
  result = {
      "episode_reward_mean": jnp.mean(total_reward),
      "episode_reward_std": jnp.std(total_reward),
      "episode_reward_min": jnp.min(total_reward),
      "episode_reward_max": jnp.max(total_reward),
      "done_fraction": jnp.mean(done_any),
      "action_nan": action_nan,
      "reward_nan": reward_nan,
      "obs_nan": obs_nan,
      "truncation_fraction": jnp.asarray(
          truncation_sum / max(eval_env_steps, 1), dtype=jnp.float32
      ),
      "truncation_present": jnp.asarray(truncation_present),
      "action_log_prob_mean": action_log_prob_sum / action_log_prob_count,
      "action_abs_mean": action_abs_sum / action_count,
      "action_saturation_fraction_095": action_saturation_sum / action_count,
  }
  if action_diagnostics:
    result["action_diagnostics"] = _finalize_action_diag(action_diag)
  if component_keys:
    result["reward_components"] = {
        "episode_sum_mean": jnp.mean(component_sums, axis=1),
        "episode_sum_std": jnp.std(component_sums, axis=1),
        "episode_sum_min": jnp.min(component_sums, axis=1),
        "episode_sum_max": jnp.max(component_sums, axis=1),
    }
  return result


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
  if args.render:
    raise NotImplementedError("Route B SAC deterministic eval render is unsupported.")
  if args.num_eval_envs < 1:
    raise ValueError("--num_eval_envs must be >= 1")
  command = [
      float(args.command_x),
      float(args.command_y),
      float(args.command_yaw),
  ]

  payload = sac_checkpoint.load(args.checkpoint)
  if not isinstance(payload, Mapping):
    raise ValueError(f"checkpoint payload is {type(payload).__name__}, not mapping")
  _require_checkpoint_ready(payload)

  payload_config = payload.get("config", {})
  env_name = args.env_name or _get(payload_config, "env_name")
  if not env_name:
    raise ValueError("--env_name is required when checkpoint config lacks env_name")
  impl = args.impl or _get(payload_config, "impl", "jax")
  config = _merged_config(payload_config, env_name, impl)
  episode_length = int(
      args.episode_length
      if args.episode_length is not None
      else config.get("episode_length", 1000)
  )
  if episode_length < 1:
    raise ValueError("--episode_length must be >= 1")

  env_cfg = registry.get_default_config(env_name)
  env_overrides = _eval_env_overrides(config, impl)
  env = registry.load(
      env_name,
      config=env_cfg,
      config_overrides=env_overrides,
  )
  env = wrapper.wrap_for_brax_training(
      env,
      episode_length=episode_length,
      action_repeat=int(config.get("action_repeat", 1)),
      randomization_fn=None,
  )

  policy_obs_key = str(config.get("policy_obs_key", "state"))
  policy_obs_size = _obs_size(env.observation_size, policy_obs_key)
  action_size = int(config.get("action_size") or env.action_size)
  if action_size != int(env.action_size):
    raise ValueError(
        f"Checkpoint action_size={action_size} does not match env action_size="
        f"{env.action_size}"
    )
  policy_normalizer = payload["policy_normalizer"]
  if int(policy_normalizer.mean.shape[-1]) != policy_obs_size:
    raise ValueError(
        "policy_normalizer size does not match env policy obs size: "
        f"{policy_normalizer.mean.shape[-1]} vs {policy_obs_size}"
    )

  sac_networks = networks.make_networks(
      action_size=action_size,
      policy_hidden_layer_sizes=tuple(config["policy_hidden_layer_sizes"]),
      q_hidden_layer_sizes=tuple(config["q_hidden_layer_sizes"]),
      activation=str(config.get("activation", "swish")),
      log_std_min=float(config.get("log_std_min", -5.0)),
      log_std_max=float(config.get("log_std_max", 2.0)),
  )
  reward_component_keys = (
      _reward_component_keys(env, args.num_eval_envs, args.seed)
      if args.reward_components
      else ()
  )

  def run_policy_mode(policy_mode: str, deterministic: bool) -> dict[str, Any]:
    rollout = jax.jit(
        lambda rng: _run_eval(
            env,
            sac_networks,
            payload["policy_params"],
            policy_normalizer,
            rng,
            policy_obs_key,
            args.num_eval_envs,
            episode_length,
            bool(config.get("normalize_observations", True)),
            deterministic,
            bool(args.action_diagnostics),
            reward_component_keys,
            jnp.asarray(command, dtype=jnp.float32) if args.fixed_command else None,
        )
    )

    start = time.monotonic()
    metrics = rollout(jax.random.PRNGKey(args.seed))
    metrics = jax.tree.map(lambda x: x.block_until_ready(), metrics)
    wall_time = time.monotonic() - start
    return _format_eval_result(
        metrics=metrics,
        args=args,
        checkpoint_metrics=payload.get("metrics", {}),
        env_name=env_name,
        impl=impl,
        episode_length=episode_length,
        wall_time=wall_time,
        policy_mode=policy_mode,
        deterministic=deterministic,
        reward_component_keys=reward_component_keys,
        fixed_command=bool(args.fixed_command),
        command=command,
        env_overrides=env_overrides,
    )

  if args.policy_mode == "both":
    return {
        "status": "EVAL_OK",
        "checkpoint": str(Path(args.checkpoint)),
        "env_name": env_name,
        "impl": impl,
        "seed": int(args.seed),
        "num_eval_envs": int(args.num_eval_envs),
        "episode_length": int(episode_length),
        "policy_mode": "both",
        "fixed_command": bool(args.fixed_command),
        "command": command,
        "env_overrides": env_overrides,
        "results": {
            "deterministic": run_policy_mode("deterministic", True),
            "stochastic": run_policy_mode("stochastic", False),
        },
    }

  return run_policy_mode(args.policy_mode, args.policy_mode == "deterministic")


def _format_eval_result(
    *,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    checkpoint_metrics: Any,
    env_name: str,
    impl: str,
    episode_length: int,
    wall_time: float,
    policy_mode: str,
    deterministic: bool,
    reward_component_keys: tuple[str, ...],
    fixed_command: bool,
    command: list[float],
    env_overrides: dict[str, Any],
) -> dict[str, Any]:
  eval_env_steps = int(args.num_eval_envs * episode_length)
  result = {
      "status": "EVAL_OK",
      "checkpoint": str(Path(args.checkpoint)),
      "env_name": env_name,
      "impl": impl,
      "seed": int(args.seed),
      "num_eval_envs": int(args.num_eval_envs),
      "episode_length": int(episode_length),
      "policy_mode": policy_mode,
      "deterministic": bool(deterministic),
      "fixed_command": bool(fixed_command),
      "command": command,
      "env_overrides": env_overrides,
      "eval_env_steps": eval_env_steps,
      "episode_reward_mean": _as_float(metrics["episode_reward_mean"]),
      "episode_reward_std": _as_float(metrics["episode_reward_std"]),
      "episode_reward_min": _as_float(metrics["episode_reward_min"]),
      "episode_reward_max": _as_float(metrics["episode_reward_max"]),
      "done_fraction": _as_float(metrics["done_fraction"]),
      "wall_time": wall_time,
      "sps": float(eval_env_steps) / max(wall_time, 1e-6),
      "action_nan": _as_bool(metrics["action_nan"]),
      "reward_nan": _as_bool(metrics["reward_nan"]),
      "obs_nan": _as_bool(metrics["obs_nan"]),
      "truncation_fraction": _as_float(metrics["truncation_fraction"]),
      "truncation_present": _as_bool(metrics["truncation_present"]),
      "action_log_prob_mean": _as_float(metrics["action_log_prob_mean"]),
      "action_abs_mean": _as_float(metrics["action_abs_mean"]),
      "action_saturation_fraction_095": _as_float(
          metrics["action_saturation_fraction_095"]
      ),
      "checkpoint_env_steps": _get(checkpoint_metrics, "env_steps"),
      "checkpoint_gradient_steps": _get(checkpoint_metrics, "gradient_steps"),
  }
  if args.action_diagnostics:
    result["action_diagnostics"] = _format_action_diagnostics(
        metrics["action_diagnostics"], args.top_k_actions
    )
  if args.reward_components:
    result["reward_components"] = _format_reward_components(
        metrics.get("reward_components", {}), reward_component_keys
    )
  return result


def main() -> int:
  args = _parse_args()
  try:
    result = evaluate_checkpoint(args)
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    return 2
  if args.output_json:
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.exit(main())
