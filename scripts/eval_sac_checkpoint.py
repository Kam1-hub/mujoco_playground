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
) -> dict[str, Any]:
  reset_key, rollout_key = jax.random.split(rng)
  reset_keys = jax.random.split(reset_key, num_eval_envs)
  state = env.reset(reset_keys)
  truncation_present = isinstance(state.info, Mapping) and "truncation" in state.info

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
        current_key,
    ) = carry
    policy_obs = evaluator.select_obs(current_state.obs, policy_obs_key)
    obs_nan = jnp.logical_or(obs_nan, jnp.any(jnp.isnan(policy_obs)))
    policy_obs = normalizer.normalize(
        policy_normalizer, policy_obs, normalize_observations
    )
    current_key, action_key = jax.random.split(current_key)
    action, log_prob = networks.sample_action(
        sac_networks,
        policy_params,
        policy_obs,
        action_key,
        deterministic=deterministic,
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
    reward_nan = jnp.logical_or(reward_nan, jnp.any(jnp.isnan(next_state.reward)))
    total_reward = total_reward + next_state.reward * (1.0 - done_any)
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
        current_key,
    ), None

  rewards = jnp.zeros_like(state.reward)
  done_any = jnp.zeros_like(state.done)
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
      _,
  ), _ = jax.lax.scan(step_fn, initial, None, length=episode_length)

  eval_env_steps = num_eval_envs * episode_length
  action_log_prob_count = jnp.maximum(action_log_prob_count, 1.0)
  action_count = jnp.maximum(action_count, 1.0)
  return {
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


def evaluate_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
  if args.render:
    raise NotImplementedError("Route B SAC deterministic eval render is unsupported.")
  if args.num_eval_envs < 1:
    raise ValueError("--num_eval_envs must be >= 1")

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
  env = registry.load(
      env_name,
      config=env_cfg,
      config_overrides={"impl": impl},
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
