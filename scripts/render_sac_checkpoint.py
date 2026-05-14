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
) -> dict[str, Any]:
  def rollout_fn(rng: jax.Array) -> dict[str, Any]:
    reset_key, action_key = jax.random.split(rng)
    state = env.reset(reset_key)
    if fixed_command is not None:
      state = _set_state_command(state, fixed_command)

    def step_fn(carry: tuple[Any, jax.Array, jax.Array, jax.Array], _: Any):
      current_state, current_key, total_reward, done_any = carry
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
      total_reward = total_reward + next_state.reward * active
      done_any = jnp.maximum(done_any, next_state.done)
      return (next_state, current_key, total_reward, done_any), next_state

    initial = (
        state,
        action_key,
        jnp.zeros_like(state.reward),
        jnp.zeros_like(state.done),
    )
    (final_state, _, total_reward, done_any), trajectory = jax.lax.scan(
        step_fn, initial, None, length=episode_length
    )
    del final_state
    return {
        "trajectory": trajectory,
        "total_reward": total_reward,
        "done_any": done_any,
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
  env = registry.load(
      env_name,
      config=env_cfg,
      config_overrides={"impl": impl},
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
  )
  rollout_wall_time = time.monotonic() - start

  render_length = episode_length
  if args.stop_on_done:
    render_length = _first_done_frame(rollout["done_any"], episode_length)
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

  return {
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
