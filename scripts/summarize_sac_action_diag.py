"""Summarize SAC action diagnostic JSON and join action dims to joint names."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--input_dir",
      default="./logs/sac_eval_action_diag_full",
      help="Directory containing eval_*_both_actiondiag.json files.",
  )
  parser.add_argument("--mapping_json", required=True)
  parser.add_argument("--output_json", default=None)
  return parser.parse_args()


def _fnum(value: Any) -> float:
  try:
    return float(value)
  except (TypeError, ValueError):
    return math.nan


def _avg(values: list[float]) -> float | None:
  finite = [v for v in values if math.isfinite(v)]
  return mean(finite) if finite else None


def _sd(values: list[float]) -> float:
  finite = [v for v in values if math.isfinite(v)]
  return stdev(finite) if len(finite) > 1 else 0.0


def _scale_from_path(path: Path) -> str:
  name = path.name
  if name.startswith("eval_"):
    return name.split("_")[1]
  return "unknown"


def _load_mapping(path: Path) -> dict[int, dict[str, Any]]:
  payload = json.loads(path.read_text(encoding="utf-8"))
  return {int(row["dim"]): row for row in payload.get("rows", [])}


def _mode_summary(files: list[Path], mode: str, mapping: dict[int, dict[str, Any]]):
  rows = []
  saturated_dims: dict[int, list[float]] = defaultdict(list)
  reward_components: dict[str, list[float]] = defaultdict(list)
  for path in files:
    data = json.loads(path.read_text(encoding="utf-8"))
    result = data.get("results", {}).get(mode, {})
    diag = result.get("action_diagnostics", {})
    reward = result.get("reward_components", {}).get("components", {})
    rows.append(
        {
            "status": result.get("status"),
            "reward_mean": _fnum(result.get("episode_reward_mean")),
            "reward_std": _fnum(result.get("episode_reward_std")),
            "reward_min": _fnum(result.get("episode_reward_min")),
            "reward_max": _fnum(result.get("episode_reward_max")),
            "done_fraction": _fnum(result.get("done_fraction")),
            "action_abs_mean": _fnum(result.get("action_abs_mean")),
            "action_saturation_fraction_095": _fnum(
                result.get("action_saturation_fraction_095")
            ),
            "policy_mean_abs_mean": _fnum(diag.get("policy_mean_abs_mean")),
            "policy_log_std_mean": _fnum(diag.get("policy_log_std_mean")),
            "policy_std_mean": _fnum(diag.get("policy_std_mean")),
            "deterministic_action_abs_mean": _fnum(
                diag.get("deterministic_action_abs_mean")
            ),
            "stochastic_action_abs_mean": _fnum(
                diag.get("stochastic_action_abs_mean")
            ),
            "action_nan": bool(result.get("action_nan")),
            "reward_nan": bool(result.get("reward_nan")),
            "obs_nan": bool(result.get("obs_nan")),
        }
    )
    for item in diag.get("top_saturated_action_dims", []):
      dim = int(item.get("dim"))
      saturated_dims[dim].append(_fnum(item.get("saturation_fraction_095")))
    for key, component in reward.items():
      reward_components[key].append(_fnum(component.get("episode_sum_mean")))

  top_dims = []
  for dim, values in saturated_dims.items():
    row = mapping.get(dim, {})
    top_dims.append(
        {
            "dim": dim,
            "saturation_fraction_095_avg": _avg(values),
            "joint_name": row.get("joint_name"),
            "actuator_name": row.get("actuator_name"),
            "joint_group": row.get("joint_group"),
            "action_scale": row.get("action_scale"),
        }
    )
  top_dims.sort(
      key=lambda row: (
          row["saturation_fraction_095_avg"] is None,
          -(row["saturation_fraction_095_avg"] or 0.0),
      )
  )

  components = [
      {"key": key, "episode_sum_mean_avg": _avg(values)}
      for key, values in reward_components.items()
  ]
  components.sort(key=lambda row: abs(row["episode_sum_mean_avg"] or 0.0), reverse=True)

  return {
      "n": len(rows),
      "all_eval_ok": all(row["status"] == "EVAL_OK" for row in rows),
      "any_action_nan": any(row["action_nan"] for row in rows),
      "any_reward_nan": any(row["reward_nan"] for row in rows),
      "any_obs_nan": any(row["obs_nan"] for row in rows),
      "reward_mean_avg": _avg([row["reward_mean"] for row in rows]),
      "reward_mean_stdev": _sd([row["reward_mean"] for row in rows]),
      "reward_min_avg": _avg([row["reward_min"] for row in rows]),
      "reward_max_avg": _avg([row["reward_max"] for row in rows]),
      "done_fraction_avg": _avg([row["done_fraction"] for row in rows]),
      "action_abs_mean_avg": _avg([row["action_abs_mean"] for row in rows]),
      "action_saturation_fraction_095_avg": _avg(
          [row["action_saturation_fraction_095"] for row in rows]
      ),
      "policy_mean_abs_mean_avg": _avg(
          [row["policy_mean_abs_mean"] for row in rows]
      ),
      "policy_log_std_mean_avg": _avg(
          [row["policy_log_std_mean"] for row in rows]
      ),
      "policy_std_mean_avg": _avg([row["policy_std_mean"] for row in rows]),
      "deterministic_action_abs_mean_avg": _avg(
          [row["deterministic_action_abs_mean"] for row in rows]
      ),
      "stochastic_action_abs_mean_avg": _avg(
          [row["stochastic_action_abs_mean"] for row in rows]
      ),
      "top_saturated_action_dims": top_dims[:8],
      "top_reward_components_by_abs_episode_sum": components[:12],
  }


def _summarize(input_dir: Path, mapping_json: Path) -> dict[str, Any]:
  mapping = _load_mapping(mapping_json)
  files_by_scale: dict[str, list[Path]] = defaultdict(list)
  for path in sorted(input_dir.glob("eval_*_both_actiondiag.json")):
    files_by_scale[_scale_from_path(path)].append(path)

  summary: dict[str, Any] = {
      "input_dir": str(input_dir),
      "mapping_json": str(mapping_json),
      "scales": {},
  }
  for scale, files in sorted(files_by_scale.items()):
    summary["scales"][scale] = {
        "file_count": len(files),
        "deterministic": _mode_summary(files, "deterministic", mapping),
        "stochastic": _mode_summary(files, "stochastic", mapping),
    }
  return summary


def main() -> int:
  args = _parse_args()
  summary = _summarize(Path(args.input_dir), Path(args.mapping_json))
  text = json.dumps(summary, indent=2, sort_keys=True)
  print(text)
  if args.output_json:
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
