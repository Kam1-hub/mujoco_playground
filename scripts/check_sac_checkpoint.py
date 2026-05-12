"""Inspect a Route B SAC pickle checkpoint."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import traceback
from collections.abc import Mapping
from pathlib import Path
from typing import Any


REQUIRED_MODEL_KEYS = (
    "config",
    "policy_params",
    "q_params",
    "target_q_params",
    "log_alpha",
)

METRIC_KEYS = (
    "status",
    "env_steps",
    "gradient_steps",
    "actor_loss",
    "critic_loss",
    "alpha",
    "sps",
)


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--checkpoint",
      required=True,
      help="Path to a Route B SAC pickle checkpoint.",
  )
  parser.add_argument(
      "--require_eval_ready",
      action="store_true",
      help=(
          "Fail when the checkpoint is not ready for deterministic actor "
          "evaluation."
      ),
  )
  return parser.parse_args()


def _jsonable(value: Any) -> Any:
  if isinstance(value, Mapping):
    return {str(k): _jsonable(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_jsonable(v) for v in value]
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if hasattr(value, "item"):
    try:
      return value.item()
    except Exception:  # pylint: disable=broad-except
      pass
  if hasattr(value, "shape"):
    return {
        "type": type(value).__name__,
        "shape": list(value.shape),
        "dtype": str(getattr(value, "dtype", None)),
    }
  return repr(value)


def _get(mapping: Any, key: str) -> Any:
  if isinstance(mapping, Mapping):
    return mapping.get(key)
  return getattr(mapping, key, None)


def main() -> int:
  args = _parse_args()
  path = Path(args.checkpoint)
  result: dict[str, Any] = {
      "checkpoint": str(path),
      "status": "PASS",
      "errors": [],
      "warnings": [],
  }

  if not path.exists():
    result["status"] = "FAIL"
    result["errors"].append(f"checkpoint does not exist: {path}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  try:
    with path.open("rb") as f:
      payload = pickle.load(f)
  except Exception:  # pylint: disable=broad-except
    result["status"] = "FAIL"
    result["errors"].append("failed to load pickle checkpoint")
    result["traceback"] = traceback.format_exc()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  if not isinstance(payload, Mapping):
    result["status"] = "FAIL"
    result["errors"].append(f"checkpoint payload is {type(payload).__name__}, not mapping")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2

  keys = sorted(str(k) for k in payload.keys())
  result["top_level_keys"] = keys
  missing = [key for key in REQUIRED_MODEL_KEYS if key not in payload]
  if missing:
    result["status"] = "FAIL"
    result["errors"].append(f"missing top-level keys: {missing}")

  config = payload.get("config", {})
  metrics = payload.get("metrics", {})
  if "metrics" not in payload:
    result["warnings"].append("missing optional metrics field")
  result["config"] = {
      "env_name": _jsonable(_get(config, "env_name")),
      "impl": _jsonable(_get(config, "impl")),
      "policy_obs_key": _jsonable(_get(config, "policy_obs_key")),
      "value_obs_key": _jsonable(_get(config, "value_obs_key")),
      "normalize_observations": _jsonable(_get(config, "normalize_observations")),
  }
  result["metrics"] = {key: _jsonable(_get(metrics, key)) for key in METRIC_KEYS}
  has_policy_params = "policy_params" in payload
  has_policy_normalizer = "policy_normalizer" in payload
  has_value_normalizer = "value_normalizer" in payload
  normalize_observations = _get(config, "normalize_observations")
  deterministic_eval_ready = bool(
      has_policy_params
      and (
          normalize_observations is False
          or (normalize_observations is True and has_policy_normalizer)
      )
  )
  result["present"] = {
      "policy_params": has_policy_params,
      "q_params": "q_params" in payload,
      "target_q_params": "target_q_params" in payload,
      "log_alpha": "log_alpha" in payload,
      "policy_normalizer": has_policy_normalizer,
      "value_normalizer": has_value_normalizer,
  }
  result["has_policy_normalizer"] = has_policy_normalizer
  result["has_value_normalizer"] = has_value_normalizer
  result["normalize_observations"] = _jsonable(normalize_observations)
  result["deterministic_eval_ready"] = deterministic_eval_ready

  for key, value in result["config"].items():
    if value is None:
      result["status"] = "FAIL"
      result["errors"].append(f"missing config field: {key}")
  for key, value in result["metrics"].items():
    if value is None:
      result["warnings"].append(f"missing metrics field: {key}")
  if args.require_eval_ready and not deterministic_eval_ready:
    result["status"] = "FAIL"
    result["errors"].append("checkpoint is not deterministic-eval ready")

  print(json.dumps(result, indent=2, sort_keys=True))
  return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
  sys.exit(main())
