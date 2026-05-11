"""Inspect the G1 env API surface needed by SAC.

The script is intentionally defensive: on dependency/import/runtime failures it
prints a structured record with the full traceback, then exits non-zero.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping
from typing import Any


def _shape(value: Any) -> Any:
  shape = getattr(value, "shape", None)
  if shape is None:
    return None
  return list(shape)


def _dtype(value: Any) -> str | None:
  dtype = getattr(value, "dtype", None)
  if dtype is None:
    return None
  return str(dtype)


def _summarize_array(value: Any) -> dict[str, Any]:
  return {
      "type": type(value).__name__,
      "shape": _shape(value),
      "dtype": _dtype(value),
  }


def _summarize_obs(obs: Any) -> dict[str, Any]:
  if isinstance(obs, Mapping):
    return {
        "type": type(obs).__name__,
        "keys": sorted(str(k) for k in obs.keys()),
        "values": {str(k): _summarize_array(v) for k, v in obs.items()},
    }
  return _summarize_array(obs)


def _summarize_value(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _summarize_value(v) for k, v in value.items()}
  if hasattr(value, "to_dict"):
    return value.to_dict()
  if hasattr(value, "shape"):
    return _summarize_array(value)
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  return repr(value)


def _run_stage(result: dict[str, Any], name: str, fn):
  try:
    value = fn()
  except Exception as exc:  # pylint: disable=broad-except
    result["stages"].append({
        "name": name,
        "status": "FAIL",
        "exception": repr(exc),
        "traceback": traceback.format_exc(),
    })
    result["status"] = "FAIL"
    return None
  result["stages"].append({"name": name, "status": "PASS"})
  return value


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description="Inspect G1 env registry/load/reset/step API for SAC."
  )
  parser.add_argument(
      "--env_name",
      default="G1JoystickFlatTerrain",
      help="Environment name to inspect.",
  )
  parser.add_argument(
      "--allow_menagerie_download",
      action="store_true",
      help=(
          "Allow registry.load to download mujoco_menagerie if it is missing. "
          "Default is false to avoid hidden network/file writes."
      ),
  )
  parser.add_argument(
      "--impl",
      choices=("jax", "warp"),
      default="jax",
      help=(
          "MJX implementation override passed to registry.load. Defaults to "
          "jax so CPU-only validation does not require CUDA-backed Warp."
      ),
  )
  return parser.parse_args()


def main() -> int:
  args = _parse_args()
  result: dict[str, Any] = {
      "env_name": args.env_name,
      "status": "PASS",
      "stages": [],
      "schema": {},
      "notes": [],
  }

  imports: dict[str, Any] | None = _run_stage(
      result,
      "imports",
      lambda: _import_runtime(),
  )
  if imports is None:
    result["status"] = "BLOCKED_BY_DEPENDENCY"
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  jax = imports["jax"]
  jp = imports["jp"]
  registry = imports["registry"]
  mjx_env = imports["mjx_env"]

  if not args.allow_menagerie_download and not mjx_env.MENAGERIE_PATH.exists():
    original_ensure = mjx_env.ensure_menagerie_exists

    def _raise_missing_menagerie() -> None:
      raise RuntimeError(
          "mujoco_menagerie is missing at "
          f"{mjx_env.MENAGERIE_PATH}; rerun with "
          "--allow_menagerie_download only when network/file writes are "
          "authorized."
      )

    mjx_env.ensure_menagerie_exists = _raise_missing_menagerie
    result["notes"].append(
        "mujoco_menagerie download blocked by default; registry.load will "
        "report BLOCKED_BY_ASSET_DOWNLOAD if assets are absent."
    )
  else:
    original_ensure = None

  try:
    all_envs = _run_stage(
        result, "registry.ALL_ENVS", lambda: tuple(registry.ALL_ENVS)
    )
    if all_envs is not None:
      result["schema"]["registry_all_envs"] = list(all_envs)

    config = _run_stage(
        result,
        "registry.get_default_config",
        lambda: registry.get_default_config(args.env_name),
    )
    if config is not None:
      result["schema"]["config"] = {
          "ctrl_dt": getattr(config, "ctrl_dt", None),
          "sim_dt": getattr(config, "sim_dt", None),
          "episode_length": getattr(config, "episode_length", None),
          "action_repeat": getattr(config, "action_repeat", None),
          "action_scale": getattr(config, "action_scale", None),
          "impl": getattr(config, "impl", None),
          "requested_impl": args.impl,
      }

    randomizer = _run_stage(
        result,
        "registry.get_domain_randomizer",
        lambda: registry.get_domain_randomizer(args.env_name),
    )
    result["schema"]["domain_randomizer"] = {
        "exists": randomizer is not None,
        "repr": repr(randomizer),
    }

    env = _run_stage(
        result,
        "registry.load",
        lambda: registry.load(
            args.env_name,
            config=config,
            config_overrides={"impl": args.impl},
        ),
    )
    if env is None:
      if result["status"] == "FAIL":
        result["status"] = "BLOCKED_BY_DEPENDENCY"
      print(json.dumps(result, indent=2, sort_keys=True))
      return 2

    result["schema"]["env"] = {
        "type": type(env).__name__,
        "observation_size": _summarize_value(env.observation_size),
        "action_size": env.action_size,
        "dt": env.dt,
        "sim_dt": env.sim_dt,
        "n_substeps": env.n_substeps,
        "xml_path": getattr(env, "xml_path", None),
      }

    reset_state = _run_stage(
        result,
        "env.reset",
        lambda: env.reset(jax.random.PRNGKey(0)),
    )
    if reset_state is None:
      print(json.dumps(result, indent=2, sort_keys=True))
      return 2

    result["schema"]["reset"] = {
        "obs": _summarize_obs(reset_state.obs),
        "reward": _summarize_array(reset_state.reward),
        "done": _summarize_array(reset_state.done),
        "info_keys": sorted(str(k) for k in reset_state.info.keys()),
        "has_truncation": "truncation" in reset_state.info,
        "metrics_keys": sorted(str(k) for k in reset_state.metrics.keys()),
    }

    zero_action = jp.zeros((env.action_size,))
    step_state = _run_stage(
        result,
        "env.step_zero_action",
        lambda: env.step(reset_state, zero_action),
    )
    if step_state is None:
      print(json.dumps(result, indent=2, sort_keys=True))
      return 2

    result["schema"]["step"] = {
        "obs": _summarize_obs(step_state.obs),
        "reward": _summarize_array(step_state.reward),
        "done": _summarize_array(step_state.done),
        "info_keys": sorted(str(k) for k in step_state.info.keys()),
        "has_truncation": "truncation" in step_state.info,
        "metrics_keys": sorted(str(k) for k in step_state.metrics.keys()),
    }
  finally:
    if original_ensure is not None:
      mjx_env.ensure_menagerie_exists = original_ensure

  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["status"] == "PASS" else 2


def _import_runtime() -> dict[str, Any]:
  import jax
  import jax.numpy as jp

  from g1_env import registry
  from g1_env._src import mjx_env

  return {
      "jax": jax,
      "jp": jp,
      "registry": registry,
      "mjx_env": mjx_env,
  }


if __name__ == "__main__":
  sys.exit(main())
