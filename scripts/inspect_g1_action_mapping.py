"""Inspect G1 action dimension to actuator/joint mapping.

This helper is intentionally eval/training-free.  It loads the MuJoCo model
through the local G1 registry and prints JSON metadata that makes SAC action
diagnostics easier to interpret.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--env_name", default="G1JoystickFlatTerrain")
  parser.add_argument("--impl", default="jax")
  parser.add_argument("--output_json", default=None)
  return parser.parse_args()


def _name(model: Any, obj_type: Any, obj_id: int) -> str | None:
  import mujoco

  if obj_id < 0:
    return None
  value = mujoco.mj_id2name(model, obj_type, int(obj_id))
  return str(value) if value is not None else None


def _as_float(value: Any) -> float:
  import numpy as np

  return float(np.asarray(value))


def _as_float_list(value: Any) -> list[float]:
  import numpy as np

  return [float(v) for v in np.asarray(value).reshape(-1).tolist()]


def _per_dim_values(value: Any, action_size: int) -> list[float]:
  import numpy as np

  array = np.asarray(value, dtype=float)
  if array.shape == ():
    return [float(array)] * action_size
  flat = array.reshape(-1)
  if flat.size != action_size:
    raise ValueError(f"Expected {action_size} values, got {flat.size}.")
  return [float(v) for v in flat.tolist()]


def _joint_group(joint_name: str | None) -> str | None:
  if joint_name is None:
    return None
  if joint_name.startswith("left_"):
    side = "left"
    rest = joint_name[len("left_") :]
  elif joint_name.startswith("right_"):
    side = "right"
    rest = joint_name[len("right_") :]
  else:
    side = "center"
    rest = joint_name

  if rest.startswith(("hip_", "knee_", "ankle_")):
    region = "leg"
  elif rest.startswith("waist_"):
    region = "waist"
  elif any(token in rest for token in ("shoulder", "elbow", "wrist")):
    region = "arm"
  else:
    region = "other"
  return f"{side}_{region}"


def _mapping(env_name: str, impl: str) -> dict[str, Any]:
  import jax
  import mujoco

  from g1_env import registry

  config = registry.get_default_config(env_name)
  env = registry.load(env_name, config=config, config_overrides={"impl": impl})
  model = env.mj_model
  action_size = int(env.action_size)
  default_pose = _per_dim_values(jax.device_get(env._default_pose), action_size)  # pylint: disable=protected-access
  action_scale = _per_dim_values(env._config.action_scale, action_size)  # pylint: disable=protected-access

  rows = []
  for dim in range(action_size):
    actuator_name = _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, dim)
    joint_id = int(model.actuator_trnid[dim, 0])
    joint_name = _name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    qposadr = int(model.jnt_qposadr[joint_id]) if joint_id >= 0 else None
    dofadr = int(model.jnt_dofadr[joint_id]) if joint_id >= 0 else None
    body_id = int(model.jnt_bodyid[joint_id]) if joint_id >= 0 else -1
    body_name = _name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    row = {
        "dim": dim,
        "actuator_name": actuator_name,
        "joint_id": joint_id,
        "joint_name": joint_name,
        "joint_group": _joint_group(joint_name),
        "body_name": body_name,
        "qposadr": qposadr,
        "dofadr": dofadr,
        "joint_range": _as_float_list(model.jnt_range[joint_id])
        if joint_id >= 0
        else None,
        "actuator_ctrlrange": _as_float_list(model.actuator_ctrlrange[dim]),
        "default_pose": default_pose[dim],
        "action_scale": action_scale[dim],
        "default_minus_scale": default_pose[dim] - action_scale[dim],
        "default_plus_scale": default_pose[dim] + action_scale[dim],
        "gear": _as_float_list(model.actuator_gear[dim]),
        "forcerange": _as_float_list(model.actuator_forcerange[dim]),
        "joint_limited": bool(model.jnt_limited[joint_id]) if joint_id >= 0 else None,
        "actuator_ctrllimited": bool(model.actuator_ctrllimited[dim]),
        "actuator_forcelimited": bool(model.actuator_forcelimited[dim]),
    }
    rows.append(row)

  return {
      "env_name": env_name,
      "impl": impl,
      "xml_path": env.xml_path,
      "action_size": action_size,
      "rows": rows,
  }


def main() -> int:
  args = _parse_args()
  payload = _mapping(args.env_name, args.impl)
  text = json.dumps(payload, indent=2, sort_keys=True)
  print(text)
  if args.output_json:
    path = Path(args.output_json)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
