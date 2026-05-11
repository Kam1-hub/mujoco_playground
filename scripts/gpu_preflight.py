"""GPU migration preflight checks for Route B G1 SAC.

This script does not start training. It validates runtime imports, assets,
registry availability, and a reset/step smoke for both G1 envs.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_MENAGERIE_COMMIT = "1b86ece576591213e2b666ebf59508454200ca97"
G1_ENVS = ("G1JoystickFlatTerrain", "G1JoystickRoughTerrain")


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--impl",
      choices=("jax", "warp"),
      default="jax",
      help="MJX implementation override for env load/reset/step checks.",
  )
  parser.add_argument(
      "--require_gpu",
      action="store_true",
      help="Fail if JAX cannot see a GPU/CUDA/ROCm backend.",
  )
  return parser.parse_args()


def _shape(value: Any) -> Any:
  shape = getattr(value, "shape", None)
  if shape is None:
    return None
  return list(shape)


def _summarize_array(value: Any) -> dict[str, Any]:
  return {
      "type": type(value).__name__,
      "shape": _shape(value),
      "dtype": str(getattr(value, "dtype", None)),
  }


def _summarize_obs(obs: Any) -> Any:
  if isinstance(obs, Mapping):
    return {
        "type": type(obs).__name__,
        "keys": sorted(str(k) for k in obs.keys()),
        "values": {str(k): _summarize_array(v) for k, v in obs.items()},
    }
  return _summarize_array(obs)


def _summarize_obs_size(obs_size: Any) -> Any:
  if isinstance(obs_size, Mapping):
    return {str(k): str(v) for k, v in obs_size.items()}
  return str(obs_size)


def _read_git_head(repo_path: Path) -> str | None:
  git_dir = repo_path / ".git"
  head_path = git_dir / "HEAD"
  if not head_path.exists():
    return None
  head = head_path.read_text(encoding="utf-8").strip()
  if not head.startswith("ref:"):
    return head
  ref = head.split(":", 1)[1].strip()
  ref_path = git_dir / ref
  if ref_path.exists():
    return ref_path.read_text(encoding="utf-8").strip()
  packed_refs = git_dir / "packed-refs"
  if packed_refs.exists():
    for line in packed_refs.read_text(encoding="utf-8").splitlines():
      if line.startswith("#") or not line.strip():
        continue
      commit, _, packed_ref = line.partition(" ")
      if packed_ref == ref:
        return commit
  return None


def _run_stage(result: dict[str, Any], name: str, fn) -> Any:
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


def _check_env(registry: Any, jax: Any, jp: Any, env_name: str, impl: str) -> dict[str, Any]:
  config = registry.get_default_config(env_name)
  env = registry.load(env_name, config=config, config_overrides={"impl": impl})
  reset_state = env.reset(jax.random.PRNGKey(0))
  step_state = env.step(reset_state, jp.zeros((env.action_size,)))
  return {
      "type": type(env).__name__,
      "impl": impl,
      "observation_size": _summarize_obs_size(env.observation_size),
      "action_size": env.action_size,
      "dt": env.dt,
      "sim_dt": env.sim_dt,
      "reset_obs": _summarize_obs(reset_state.obs),
      "step_obs": _summarize_obs(step_state.obs),
      "reset_done": _summarize_array(reset_state.done),
      "step_done": _summarize_array(step_state.done),
      "reset_reward": _summarize_array(reset_state.reward),
      "step_reward": _summarize_array(step_state.reward),
      "reset_has_truncation": "truncation" in reset_state.info,
      "step_has_truncation": "truncation" in step_state.info,
      "metrics_keys": sorted(str(k) for k in step_state.metrics.keys()),
  }


def main() -> int:
  args = _parse_args()
  result: dict[str, Any] = {
      "status": "PASS",
      "impl": args.impl,
      "require_gpu": args.require_gpu,
      "expected_menagerie_commit": EXPECTED_MENAGERIE_COMMIT,
      "stages": [],
      "errors": [],
      "summary": {},
  }

  imports = _run_stage(result, "imports", _import_runtime)
  if imports is None:
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  jax = imports["jax"]
  jp = imports["jp"]
  mujoco = imports["mujoco"]
  brax = imports["brax"]
  registry = imports["registry"]
  mjx_env = imports["mjx_env"]
  train_jax_sac_lift = imports["train_jax_sac_lift"]

  backend = jax.default_backend()
  devices = jax.devices()
  device_platforms = [str(getattr(device, "platform", "")) for device in devices]
  has_gpu = backend.lower() in ("gpu", "cuda", "rocm") or any(
      platform.lower() in ("gpu", "cuda", "rocm") for platform in device_platforms
  )
  result["summary"]["runtime"] = {
      "jax_version": jax.__version__,
      "jax_backend": backend,
      "jax_devices": [str(device) for device in devices],
      "jax_device_platforms": device_platforms,
      "mujoco_version": mujoco.__version__,
      "brax_version": getattr(brax, "__version__", "unknown"),
      "has_gpu": has_gpu,
  }
  if args.require_gpu and not has_gpu:
    result["status"] = "FAIL"
    result["errors"].append(
        f"--require_gpu set but JAX backend/devices are {backend} / {devices}"
    )

  menagerie_path = Path(str(mjx_env.MENAGERIE_PATH))
  menagerie_commit = _read_git_head(menagerie_path) if menagerie_path.exists() else None
  result["summary"]["menagerie"] = {
      "path": str(menagerie_path),
      "exists": menagerie_path.exists(),
      "commit": menagerie_commit,
      "matches_expected_commit": menagerie_commit == EXPECTED_MENAGERIE_COMMIT,
  }
  if not menagerie_path.exists():
    result["status"] = "FAIL"
    result["errors"].append(f"menagerie missing at {menagerie_path}")
  elif menagerie_commit != EXPECTED_MENAGERIE_COMMIT:
    result["status"] = "FAIL"
    result["errors"].append(
        "menagerie commit mismatch: "
        f"expected {EXPECTED_MENAGERIE_COMMIT}, got {menagerie_commit}"
    )

  result["summary"]["registry_all_envs"] = list(registry.ALL_ENVS)
  result["summary"]["train_jax_sac_lift_import"] = {
      "module": train_jax_sac_lift.__name__,
      "has_main": hasattr(train_jax_sac_lift, "main"),
      "has_run": hasattr(train_jax_sac_lift, "run"),
  }
  if not result["summary"]["train_jax_sac_lift_import"]["has_main"]:
    result["status"] = "FAIL"
    result["errors"].append("learning.train_jax_sac_lift has no main()")

  env_summaries: dict[str, Any] = {}
  for env_name in G1_ENVS:
    env_summary = _run_stage(
        result,
        f"env_check:{env_name}",
        lambda env_name=env_name: _check_env(registry, jax, jp, env_name, args.impl),
    )
    if env_summary is not None:
      env_summaries[env_name] = env_summary
  result["summary"]["envs"] = env_summaries

  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["status"] == "PASS" else 2


def _import_runtime() -> dict[str, Any]:
  import brax
  import jax
  import jax.numpy as jp
  import mujoco

  from g1_env import registry
  from g1_env._src import mjx_env
  from learning import train_jax_sac_lift

  return {
      "brax": brax,
      "jax": jax,
      "jp": jp,
      "mujoco": mujoco,
      "registry": registry,
      "mjx_env": mjx_env,
      "train_jax_sac_lift": train_jax_sac_lift,
  }


if __name__ == "__main__":
  sys.exit(main())
