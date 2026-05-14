"""Render a Route B SAC checkpoint rollout to video or image frames."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mediapy as media
import mujoco

from g1_env import registry
from g1_env.config import sac_params
from learning.sac_lift import checkpoint as sac_checkpoint
from learning.sac_lift import evaluator
from learning.sac_lift import networks
from learning.sac_lift import normalizer


TERMINATION_REASON_METRICS: tuple[tuple[str, str], ...] = (
    ("fall", "termination/fall_torso_up_z_lt_0"),
    ("contact/right_foot_left_foot", "termination/contact/right_foot_left_foot"),
    ("contact/left_foot_right_shin", "termination/contact/left_foot_right_shin"),
    ("contact/right_foot_left_shin", "termination/contact/right_foot_left_shin"),
    ("contact_any", "termination/contact_any"),
    ("qpos_nan", "termination/qpos_nan"),
    ("qvel_nan", "termination/qvel_nan"),
)

TERMINATION_CONDITIONS: tuple[str, ...] = (
    "fall_torso_up_z_lt_0",
    "contact/right_foot_left_foot",
    "contact/left_foot_right_shin",
    "contact/right_foot_left_shin",
    "qpos_nan",
    "qvel_nan",
)

TERMINATION_SCALAR_STATE_METRICS: tuple[tuple[str, str], ...] = (
    ("torso_up_z", "termination/torso_up_z"),
    ("root_height", "termination/root_height"),
    ("orientation_cost", "termination/orientation_cost"),
    ("torso_up_xy_norm", "termination/torso_up_xy_norm"),
    ("torso_ang_vel_xy_norm", "termination/torso_ang_vel_xy_norm"),
    ("tracking_lin_vel_error", "termination/tracking_lin_vel_error"),
    ("tracking_yaw_error", "termination/tracking_yaw_error"),
)

TERMINATION_VECTOR_STATE_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "command",
        (
            "termination/command_x",
            "termination/command_y",
            "termination/command_yaw",
        ),
    ),
    (
        "pelvis_local_linvel",
        (
            "termination/pelvis_local_linvel_x",
            "termination/pelvis_local_linvel_y",
            "termination/pelvis_local_linvel_z",
        ),
    ),
    (
        "feet_floor_contact",
        (
            "termination/feet_floor_contact_left",
            "termination/feet_floor_contact_right",
        ),
    ),
)


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
  parser.add_argument("--episode_length", type=int, default=None)
  parser.add_argument(
      "--policy_mode",
      choices=("deterministic", "stochastic"),
      default="deterministic",
      help="Render tanh(mean) actions or sampled stochastic actions.",
  )
  parser.add_argument("--width", type=int, default=640)
  parser.add_argument("--height", type=int, default=480)
  parser.add_argument("--fps", type=float, default=30.0)
  parser.add_argument(
      "--render_every",
      type=int,
      default=1,
      help="Render every Nth rollout state.",
  )
  parser.add_argument(
      "--camera",
      default=None,
      help="MuJoCo camera name or id. Defaults to MuJoCo free camera.",
  )
  parser.add_argument(
      "--stop_on_done",
      type=_str_to_bool,
      default=False,
      help="If true, truncate rendered frames after the first done signal.",
  )
  parser.add_argument(
      "--fixed_command",
      type=_str_to_bool,
      default=False,
      help=(
          "If true, force a fixed joystick command throughout the render "
          "rollout instead of using the command sampled by env reset/resample."
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
  parser.add_argument(
      "--termination_diagnostics",
      action="store_true",
      help="Include first-done termination reason and terminal-state summaries.",
  )
  parser.add_argument(
      "--output",
      required=True,
      help=(
          "Output .mp4/.gif path, or a directory for PNG frames. If video "
          "encoding fails, PNG frames are written next to the requested output."
      ),
  )
  parser.add_argument(
      "--summary_json",
      default=None,
      help="Optional path for a JSON render summary.",
  )
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


def _render_env_overrides(config: Mapping[str, Any], impl: str) -> dict[str, Any]:
  overrides: dict[str, Any] = {"impl": impl}
  env_feet_slip_mode = config.get("env_feet_slip_mode")
  if env_feet_slip_mode is not None:
    overrides["feet_slip_mode"] = str(env_feet_slip_mode)
  env_feet_slip_scale = config.get("env_feet_slip_scale")
  if env_feet_slip_scale is not None:
    overrides["reward_config.scales.feet_slip"] = float(env_feet_slip_scale)
  env_push_enable = config.get("env_push_enable")
  if env_push_enable is not None:
    overrides["push_config.enable"] = _str_to_bool(env_push_enable)
  env_zero_command_phase_freeze = config.get("env_zero_command_phase_freeze")
  if env_zero_command_phase_freeze is not None:
    overrides["zero_command_phase_freeze"] = _str_to_bool(
        env_zero_command_phase_freeze
    )
  env_feet_air_time_command_mask = config.get("env_feet_air_time_command_mask")
  if env_feet_air_time_command_mask is not None:
    overrides["feet_air_time_command_mask"] = _str_to_bool(
        env_feet_air_time_command_mask
    )
  env_reset_joint_noise_scale = config.get("env_reset_joint_noise_scale")
  if env_reset_joint_noise_scale is not None:
    overrides["reset_joint_noise_scale"] = float(env_reset_joint_noise_scale)
  env_reset_root_qvel_scale = config.get("env_reset_root_qvel_scale")
  if env_reset_root_qvel_scale is not None:
    overrides["reset_root_qvel_scale"] = float(env_reset_root_qvel_scale)
  env_reward_action_rate_scale = config.get("env_reward_action_rate_scale")
  if env_reward_action_rate_scale is not None:
    overrides["reward_config.scales.action_rate"] = float(
        env_reward_action_rate_scale
    )
  env_reward_ang_vel_xy_scale = config.get("env_reward_ang_vel_xy_scale")
  if env_reward_ang_vel_xy_scale is not None:
    overrides["reward_config.scales.ang_vel_xy"] = float(
        env_reward_ang_vel_xy_scale
    )
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


def _build_policy(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
    env: Any,
) -> tuple[networks.SACNetworks, Any, normalizer.RunningStats, str, bool]:
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
  return (
      sac_networks,
      payload["policy_params"],
      policy_normalizer,
      policy_obs_key,
      bool(config.get("normalize_observations", True)),
  )


def _set_state_command(state: Any, command: jax.Array) -> Any:
  state.info["command"] = command
  state = state.replace(obs=_replace_obs_command(state.obs, command))
  return state


def _replace_obs_command(obs: Any, command: jax.Array) -> Any:
  def replace_in_array(value: jax.Array) -> jax.Array:
    return value.at[9:12].set(command)

  if isinstance(obs, Mapping):
    replaced = dict(obs)
    for key in ("state", "privileged_state"):
      if key in replaced:
        replaced[key] = replace_in_array(replaced[key])
    return replaced
  return replace_in_array(obs)


def _as_float(value: Any) -> float:
  return float(jax.device_get(value))


def _as_bool(value: Any) -> bool:
  return bool(jax.device_get(value))


def _as_float_list(value: Any) -> list[float]:
  return [float(v) for v in jax.device_get(value).reshape(-1).tolist()]


def _metric_scalar(metrics: Mapping[str, Any], key: str) -> jax.Array:
  return jnp.mean(jnp.asarray(metrics[key], dtype=jnp.float32))


def _empty_termination_diag(episode_length: int) -> dict[str, Any]:
  return {
      "available": jnp.asarray(True),
      "done_seen": jnp.asarray(False),
      "first_done_step": jnp.asarray(episode_length, dtype=jnp.int32),
      "reason_flags": {
          name: jnp.asarray(0.0, dtype=jnp.float32)
          for name, _ in TERMINATION_REASON_METRICS
      },
      "terminal_count": jnp.asarray(0.0, dtype=jnp.float32),
      "scalar_state": {
          name: jnp.asarray(0.0, dtype=jnp.float32)
          for name, _ in TERMINATION_SCALAR_STATE_METRICS
      },
      "vector_state": {
          name: jnp.zeros((len(metric_keys),), dtype=jnp.float32)
          for name, metric_keys in TERMINATION_VECTOR_STATE_METRICS
      },
  }


def _update_termination_diag(
    termination_diag: dict[str, Any],
    metrics: Mapping[str, Any],
    new_done: jax.Array,
    step_index: jax.Array,
) -> dict[str, Any]:
  new_done = jnp.asarray(new_done, dtype=bool)
  reason_flags = {}
  for name, metric_key in TERMINATION_REASON_METRICS:
    value = _metric_scalar(metrics, metric_key)
    reason_flags[name] = jnp.where(
        new_done, value, termination_diag["reason_flags"][name]
    )

  scalar_state = {}
  for name, metric_key in TERMINATION_SCALAR_STATE_METRICS:
    value = _metric_scalar(metrics, metric_key)
    scalar_state[name] = jnp.where(
        new_done, value, termination_diag["scalar_state"][name]
    )

  vector_state = {}
  for name, metric_keys in TERMINATION_VECTOR_STATE_METRICS:
    values = jnp.stack(
        [_metric_scalar(metrics, metric_key) for metric_key in metric_keys],
        axis=0,
    )
    vector_state[name] = jnp.where(
        new_done, values, termination_diag["vector_state"][name]
    )

  return {
      **termination_diag,
      "done_seen": jnp.logical_or(termination_diag["done_seen"], new_done),
      "first_done_step": jnp.where(
          new_done, step_index.astype(jnp.int32), termination_diag["first_done_step"]
      ),
      "reason_flags": reason_flags,
      "terminal_count": jnp.where(new_done, 1.0, termination_diag["terminal_count"]),
      "scalar_state": scalar_state,
      "vector_state": vector_state,
  }


def _format_termination_diagnostics(
    stats: dict[str, Any],
    episode_length: int,
    render_indices: Sequence[int],
) -> dict[str, Any]:
  if not stats:
    return {"available": False}

  done_seen = _as_bool(stats["done_seen"])
  first_done_step = int(jax.device_get(stats["first_done_step"]))
  first_done_frame = first_done_step - 1 if done_seen else None
  rendered_frame_index = None
  if first_done_frame is not None:
    try:
      rendered_frame_index = list(render_indices).index(first_done_frame)
    except ValueError:
      rendered_frame_index = None

  reason_flags = {
      name: _as_bool(stats["reason_flags"][name] > 0.0)
      for name, _ in TERMINATION_REASON_METRICS
  }
  reason_counts = {
      name: int(reason_flags[name]) for name, _ in TERMINATION_REASON_METRICS
  }

  terminal_state: dict[str, Any] = {"count": int(done_seen)}
  if done_seen:
    for name, _ in TERMINATION_SCALAR_STATE_METRICS:
      terminal_state[name] = _as_float(stats["scalar_state"][name])
    for name, _ in TERMINATION_VECTOR_STATE_METRICS:
      terminal_state[name] = _as_float_list(stats["vector_state"][name])

  return {
      "available": _as_bool(stats["available"]),
      "termination_conditions": list(TERMINATION_CONDITIONS),
      "episode_length": int(episode_length),
      "done_seen": done_seen,
      "first_done_step": first_done_step if done_seen else None,
      "first_done_frame": first_done_frame,
      "rendered_frame_index": rendered_frame_index,
      "reason_flags": reason_flags,
      "reason_counts": reason_counts,
      "terminal_state": terminal_state,
  }


def _rollout(
    env: Any,
    sac_networks: networks.SACNetworks,
    policy_params: Any,
    policy_normalizer: normalizer.RunningStats,
    policy_obs_key: str,
    normalize_observations: bool,
    seed: int,
    episode_length: int,
    deterministic: bool,
    fixed_command: jax.Array | None = None,
    termination_diagnostics: bool = False,
) -> dict[str, Any]:
  def rollout_fn(rng: jax.Array) -> dict[str, Any]:
    reset_key, action_key = jax.random.split(rng)
    state = env.reset(reset_key)
    if fixed_command is not None:
      state = _set_state_command(state, fixed_command)

    def step_fn(
        carry: tuple[Any, jax.Array, jax.Array, jax.Array, Any],
        step_index: jax.Array,
    ):
      current_state, current_key, total_reward, done_any, termination_diag = carry
      policy_obs = evaluator.select_obs(current_state.obs, policy_obs_key)
      policy_obs = normalizer.normalize(
          policy_normalizer, policy_obs, normalize_observations
      )
      current_key, sample_key = jax.random.split(current_key)
      action, _ = networks.sample_action(
          sac_networks,
          policy_params,
          policy_obs,
          sample_key,
          deterministic=deterministic,
      )
      next_state = env.step(current_state, action)
      if fixed_command is not None:
        next_state = _set_state_command(next_state, fixed_command)
      active = 1.0 - done_any
      new_done = jnp.logical_and(next_state.done > 0.0, done_any <= 0.0)
      total_reward = total_reward + next_state.reward * active
      if termination_diagnostics:
        termination_diag = _update_termination_diag(
            termination_diag, next_state.metrics, new_done, step_index + 1
        )
      done_any = jnp.maximum(done_any, next_state.done)
      return (
          next_state,
          current_key,
          total_reward,
          done_any,
          termination_diag,
      ), next_state

    initial = (
        state,
        action_key,
        jnp.zeros_like(state.reward),
        jnp.zeros_like(state.done),
        (
            _empty_termination_diag(episode_length)
            if termination_diagnostics
            else {}
        ),
    )
    (
        final_state,
        _,
        total_reward,
        done_any,
        termination_diag,
    ), trajectory = jax.lax.scan(
        step_fn, initial, jnp.arange(episode_length, dtype=jnp.int32)
    )
    del final_state
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "done_any": done_any,
        "termination_diagnostics": termination_diag,
    }

  compiled = jax.jit(rollout_fn)
  result = compiled(jax.random.PRNGKey(seed))
  return jax.device_get(result)


def _trajectory_to_list(trajectory: Any, frame_count: int) -> list[Any]:
  return [
      jax.tree.map(lambda value, i=i: value[i], trajectory)
      for i in range(frame_count)
  ]


def _first_done_frame(done_values: Any, default_length: int) -> int:
  done_flat = jnp.asarray(done_values).reshape(-1)
  done_list = [bool(value) for value in jax.device_get(done_flat).tolist()]
  for index, value in enumerate(done_list):
    if value:
      return index + 1
  return default_length


def _write_frames(frames: Sequence[Any], output_dir: Path) -> dict[str, Any]:
  output_dir.mkdir(parents=True, exist_ok=True)
  for index, frame in enumerate(frames):
    media.write_image(output_dir / f"frame_{index:04d}.png", frame)
  return {
      "output_type": "frames",
      "output_path": str(output_dir),
      "frame_count": len(frames),
      "size_bytes": sum(path.stat().st_size for path in output_dir.glob("*.png")),
  }


def _write_render_output(
    frames: Sequence[Any],
    output: Path,
    fps: float,
) -> dict[str, Any]:
  suffix = output.suffix.lower()
  if suffix in (".mp4", ".gif"):
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
      media.write_video(output, frames, fps=fps)
      return {
          "output_type": suffix.lstrip("."),
          "output_path": str(output),
          "frame_count": len(frames),
          "size_bytes": output.stat().st_size,
          "fallback_used": False,
      }
    except Exception as exc:  # pylint: disable=broad-except
      fallback_dir = output.with_suffix("")
      fallback_dir = fallback_dir.parent / f"{fallback_dir.name}_frames"
      frame_result = _write_frames(frames, fallback_dir)
      frame_result["fallback_used"] = True
      frame_result["video_error"] = repr(exc)
      return frame_result
  return {**_write_frames(frames, output), "fallback_used": False}


def render_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
  if args.episode_length is not None and args.episode_length < 1:
    raise ValueError("--episode_length must be >= 1")
  if args.render_every < 1:
    raise ValueError("--render_every must be >= 1")
  if args.width < 1 or args.height < 1:
    raise ValueError("--width and --height must be >= 1")
  if args.fps <= 0:
    raise ValueError("--fps must be > 0")
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

  env_cfg = registry.get_default_config(env_name)
  env_overrides = _render_env_overrides(config, impl)
  if args.termination_diagnostics:
    env_overrides["termination_diagnostics"] = True
  env = registry.load(
      env_name,
      config=env_cfg,
      config_overrides=env_overrides,
  )
  (
      sac_networks,
      policy_params,
      policy_normalizer,
      policy_obs_key,
      normalize_observations,
  ) = _build_policy(payload, config, env)

  start = time.monotonic()
  rollout = _rollout(
      env=env,
      sac_networks=sac_networks,
      policy_params=policy_params,
      policy_normalizer=policy_normalizer,
      policy_obs_key=policy_obs_key,
      normalize_observations=normalize_observations,
      seed=int(args.seed),
      episode_length=episode_length,
      deterministic=args.policy_mode == "deterministic",
      fixed_command=(
          jnp.asarray(command, dtype=jnp.float32) if args.fixed_command else None
      ),
      termination_diagnostics=bool(args.termination_diagnostics),
  )
  rollout_wall_time = time.monotonic() - start

  render_length = episode_length
  if args.stop_on_done:
    render_length = _first_done_frame(rollout["trajectory"].done, episode_length)
  render_indices = list(range(0, render_length, args.render_every))
  trajectory = _trajectory_to_list(rollout["trajectory"], episode_length)
  trajectory = [trajectory[index] for index in render_indices]

  scene_option = mujoco.MjvOption()
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = False
  scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

  render_start = time.monotonic()
  frames = env.render(
      trajectory,
      height=int(args.height),
      width=int(args.width),
      camera=args.camera,
      scene_option=scene_option,
  )
  render_wall_time = time.monotonic() - render_start
  output_info = _write_render_output(frames, Path(args.output), float(args.fps))

  result = {
      "status": "RENDER_OK",
      "checkpoint": str(Path(args.checkpoint)),
      "env_name": env_name,
      "impl": impl,
      "seed": int(args.seed),
      "episode_length": int(episode_length),
      "rendered_steps": int(render_length),
      "render_every": int(args.render_every),
      "policy_mode": args.policy_mode,
      "deterministic": args.policy_mode == "deterministic",
      "fixed_command": bool(args.fixed_command),
      "command": command,
      "width": int(args.width),
      "height": int(args.height),
      "fps": float(args.fps),
      "camera": args.camera,
      "total_reward": float(rollout["total_reward"]),
      "done": bool(rollout["done_any"]),
      "rollout_wall_time": rollout_wall_time,
      "render_wall_time": render_wall_time,
      **output_info,
  }
  if args.termination_diagnostics:
    result["env_overrides"] = env_overrides
    result["termination_diagnostics"] = _format_termination_diagnostics(
        rollout.get("termination_diagnostics", {}), episode_length, render_indices
    )
  return result


def main() -> int:
  args = _parse_args()
  try:
    result = render_checkpoint(args)
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    return 2
  if args.summary_json:
    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
  print(json.dumps(result, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  sys.exit(main())
