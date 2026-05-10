"""Train a minimal upstream Brax SAC baseline for G1."""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time
import traceback
from typing import Any, Sequence


def _str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ("true", "1", "yes", "y", "on"):
    return True
  if value in ("false", "0", "no", "n", "off"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _int_tuple(values: Sequence[str] | None) -> tuple[int, ...]:
  if not values:
    return (256, 256)
  if len(values) == 1 and "," in values[0]:
    values = [v for v in values[0].split(",") if v]
  return tuple(int(v) for v in values)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Minimal Route A SAC runner using upstream Brax SAC. This is a "
          "symmetric fallback that selects obs['state'] for both actor and Q."
      )
  )
  parser.add_argument("--env_name", default="G1JoystickFlatTerrain")
  parser.add_argument("--seed", type=int, default=1)
  parser.add_argument("--num_timesteps", type=int, default=None)
  parser.add_argument("--num_evals", type=int, default=None)
  parser.add_argument("--num_envs", type=int, default=None)
  parser.add_argument("--num_eval_envs", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=None)
  parser.add_argument("--min_replay_size", type=int, default=None)
  parser.add_argument("--max_replay_size", type=int, default=None)
  parser.add_argument("--grad_updates_per_step", type=int, default=None)
  parser.add_argument("--learning_rate", type=float, default=None)
  parser.add_argument("--reward_scaling", type=float, default=None)
  parser.add_argument("--discounting", type=float, default=None)
  parser.add_argument("--tau", type=float, default=None)
  parser.add_argument("--normalize_observations", type=_str_to_bool, default=None)
  parser.add_argument("--deterministic_eval", type=_str_to_bool, default=None)
  parser.add_argument("--policy_obs_key", default=None)
  parser.add_argument(
      "--hidden_layer_sizes",
      nargs="*",
      default=None,
      help="Policy/Q hidden layer sizes, e.g. --hidden_layer_sizes 256 256.",
  )
  parser.add_argument("--logdir", default="./logs/sac_brax")
  parser.add_argument("--use_wandb", type=_str_to_bool, default=False)
  parser.add_argument("--render", type=_str_to_bool, default=False)
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Import dependencies, build config, and exit before env load/train.",
  )
  parser.add_argument(
      "--allow_menagerie_download",
      action="store_true",
      help="Allow registry.load to clone mujoco_menagerie if missing.",
  )
  parser.add_argument(
      "--max_devices_per_host",
      type=int,
      default=1,
      help="Limit local JAX devices for the fallback runner.",
  )
  return parser.parse_args(argv)


def _apply_overrides(config: Any, args: argparse.Namespace) -> Any:
  for name in (
      "num_timesteps",
      "num_evals",
      "num_envs",
      "num_eval_envs",
      "batch_size",
      "min_replay_size",
      "max_replay_size",
      "grad_updates_per_step",
      "learning_rate",
      "reward_scaling",
      "discounting",
      "tau",
      "normalize_observations",
      "deterministic_eval",
      "policy_obs_key",
  ):
    value = getattr(args, name)
    if value is not None:
      config[name] = value
  if args.hidden_layer_sizes is not None:
    config.hidden_layer_sizes = _int_tuple(args.hidden_layer_sizes)
  return config


def _guard_menagerie_download(allow_download: bool) -> None:
  if allow_download:
    return
  from g1_env._src import mjx_env

  if mjx_env.MENAGERIE_PATH.exists():
    return

  def _raise_missing_menagerie() -> None:
    raise RuntimeError(
        "mujoco_menagerie is missing at "
        f"{mjx_env.MENAGERIE_PATH}; rerun with --allow_menagerie_download "
        "only when network/file writes are authorized."
    )

  mjx_env.ensure_menagerie_exists = _raise_missing_menagerie


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  if args.use_wandb:
    raise NotImplementedError("Route A fallback does not implement wandb logging.")
  if args.render:
    raise NotImplementedError("Route A fallback does not render rollouts.")

  try:
    import jax
    from brax.training.agents.sac import networks as sac_networks
    from brax.training.agents.sac import train as sac

    from g1_env import registry
    from g1_env import wrapper
    from g1_env.config import sac_params
    from learning.sac_wrappers import SelectObsWrapper
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    return 2

  config = _apply_overrides(sac_params.brax_sac_config(args.env_name), args)
  print("Route A Brax SAC config:")
  print(config)
  print("JAX backend:", jax.default_backend())
  print("JAX devices:", [str(device) for device in jax.devices()])

  if args.dry_run:
    return 0

  _guard_menagerie_download(args.allow_menagerie_download)

  try:
    env_cfg = registry.get_default_config(args.env_name)
    train_env = SelectObsWrapper(
        registry.load(args.env_name, config=env_cfg),
        obs_key=config.policy_obs_key,
    )
    eval_env = SelectObsWrapper(
        registry.load(args.env_name, config=registry.get_default_config(args.env_name)),
        obs_key=config.policy_obs_key,
    )

    os.makedirs(args.logdir, exist_ok=True)

    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        hidden_layer_sizes=tuple(config.hidden_layer_sizes),
    )

    times = [time.monotonic()]

    def progress(num_steps: int, metrics: dict[str, Any]) -> None:
      times.append(time.monotonic())
      flat_metrics = {
          key: float(value) if hasattr(value, "__float__") else value
          for key, value in metrics.items()
      }
      print(json.dumps({"num_steps": num_steps, "metrics": flat_metrics}, sort_keys=True))

    sac.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=config.num_timesteps,
        episode_length=config.episode_length,
        action_repeat=config.action_repeat,
        num_envs=config.num_envs,
        num_eval_envs=config.num_eval_envs,
        learning_rate=config.learning_rate,
        discounting=config.discounting,
        seed=args.seed,
        batch_size=config.batch_size,
        num_evals=config.num_evals,
        normalize_observations=config.normalize_observations,
        max_devices_per_host=args.max_devices_per_host,
        reward_scaling=config.reward_scaling,
        tau=config.tau,
        min_replay_size=config.min_replay_size,
        max_replay_size=config.max_replay_size,
        grad_updates_per_step=config.grad_updates_per_step,
        deterministic_eval=config.deterministic_eval,
        network_factory=network_factory,
        progress_fn=progress,
        checkpoint_logdir=args.logdir,
        wrap_env_fn=wrapper.wrap_for_brax_training,
    )
    if len(times) > 1:
      print(f"Route A completed in {times[-1] - times[0]:.3f}s.")
    return 0
  except Exception:  # pylint: disable=broad-except
    traceback.print_exc()
    return 2


def run() -> None:
  sys.exit(main())


if __name__ == "__main__":
  run()
