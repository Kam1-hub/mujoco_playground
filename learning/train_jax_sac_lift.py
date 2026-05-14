"""Command line entry point for the main G1 SAC baseline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from learning.sac_lift import config as sac_config


def _str_to_bool(value: str | bool) -> bool:
  if isinstance(value, bool):
    return value
  value = value.lower()
  if value in ("true", "1", "yes", "y", "on"):
    return True
  if value in ("false", "0", "no", "n", "off"):
    return False
  raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(
      description=(
          "Main Route B SAC runner for G1. Actor defaults to obs['state']; "
          "critic defaults to obs['privileged_state']."
      )
  )
  parser.add_argument("--env_name", default="G1JoystickFlatTerrain")
  parser.add_argument(
      "--impl",
      choices=("jax", "warp"),
      default=None,
      help=(
          "MJX implementation override for the G1 env. Defaults to the SAC "
          "config value, currently jax for CPU smoke compatibility."
      ),
  )
  parser.add_argument("--seed", type=int, default=None)
  parser.add_argument("--num_timesteps", type=int, default=None)
  parser.add_argument("--num_evals", type=int, default=None)
  parser.add_argument("--num_envs", type=int, default=None)
  parser.add_argument("--num_eval_envs", type=int, default=None)
  parser.add_argument("--batch_size", type=int, default=None)
  parser.add_argument("--actor_learning_rate", type=float, default=None)
  parser.add_argument("--critic_learning_rate", type=float, default=None)
  parser.add_argument("--alpha_learning_rate", type=float, default=None)
  parser.add_argument("--policy_hidden_layer_sizes", nargs="*", default=None)
  parser.add_argument("--q_hidden_layer_sizes", nargs="*", default=None)
  parser.add_argument("--policy_obs_key", default=None)
  parser.add_argument("--value_obs_key", default=None)
  parser.add_argument("--min_replay_size", type=int, default=None)
  parser.add_argument("--max_replay_size", type=int, default=None)
  parser.add_argument("--grad_updates_per_step", type=int, default=None)
  parser.add_argument("--reward_scaling", type=float, default=None)
  parser.add_argument("--discounting", type=float, default=None)
  parser.add_argument("--tau", type=float, default=None)
  parser.add_argument("--target_entropy_coef", type=float, default=None)
  parser.add_argument("--deterministic_action_l2_coef", type=float, default=None)
  parser.add_argument("--actor_mean_l2_coef", type=float, default=None)
  parser.add_argument("--fixed_alpha", type=float, default=None)
  parser.add_argument("--alpha_floor", type=float, default=None)
  parser.add_argument(
      "--alpha_loss_type",
      choices=tuple(sac_config.ALPHA_LOSS_TYPE_IDS),
      default=None,
  )
  parser.add_argument(
      "--env_feet_slip_mode",
      choices=sac_config.FEET_SLIP_MODES,
      default=None,
      help=(
          "Optional G1 env feet_slip implementation override. Defaults to the "
          "env behavior when unset."
      ),
  )
  parser.add_argument(
      "--env_feet_slip_scale",
      type=float,
      default=None,
      help=(
          "Optional override for reward_config.scales.feet_slip. Leave unset "
          "to preserve the env default."
      ),
  )
  parser.add_argument(
      "--env_push_enable",
      type=_str_to_bool,
      default=None,
      help=(
          "Optional override for push_config.enable. Leave unset to preserve "
          "the env default."
      ),
  )
  parser.add_argument(
      "--env_zero_command_phase_freeze",
      type=_str_to_bool,
      default=None,
      help=(
          "Optional override for zero_command_phase_freeze. Leave unset to "
          "preserve the env default."
      ),
  )
  parser.add_argument(
      "--env_feet_air_time_command_mask",
      type=_str_to_bool,
      default=None,
      help=(
          "Optional override for feet_air_time_command_mask. Leave unset to "
          "preserve the env default."
      ),
  )
  parser.add_argument("--normalize_observations", type=_str_to_bool, default=None)
  parser.add_argument("--deterministic_eval", type=_str_to_bool, default=None)
  parser.add_argument("--logdir", default=None)
  parser.add_argument("--dry_run", action="store_true")
  parser.add_argument("--render", type=_str_to_bool, default=None)
  parser.add_argument("--use_wandb", type=_str_to_bool, default=None)
  parser.add_argument(
      "--allow_menagerie_download",
      action="store_true",
      help="Allow registry.load to clone mujoco_menagerie if missing.",
  )
  parser.add_argument("--policy_obs_size", type=int, default=None)
  parser.add_argument("--value_obs_size", type=int, default=None)
  parser.add_argument("--action_size", type=int, default=None)
  return parser.parse_args(argv)


def _build_config(args: argparse.Namespace):
  from g1_env.config import sac_params

  config = sac_params.lift_sac_config(args.env_name)
  if "deterministic_action_l2_coef" not in config:
    config.deterministic_action_l2_coef = (
        sac_config.DEFAULT_DETERMINISTIC_ACTION_L2_COEF
    )
  if "actor_mean_l2_coef" not in config:
    config.actor_mean_l2_coef = sac_config.DEFAULT_ACTOR_MEAN_L2_COEF
  if "fixed_alpha" not in config:
    config.fixed_alpha = sac_config.DEFAULT_FIXED_ALPHA
  if "alpha_floor" not in config:
    config.alpha_floor = sac_config.DEFAULT_ALPHA_FLOOR
  if "alpha_loss_type" not in config:
    config.alpha_loss_type = sac_config.DEFAULT_ALPHA_LOSS_TYPE
  if "env_feet_slip_mode" not in config:
    config.env_feet_slip_mode = sac_config.DEFAULT_ENV_FEET_SLIP_MODE
  if "env_feet_slip_scale" not in config:
    config.env_feet_slip_scale = sac_config.DEFAULT_ENV_FEET_SLIP_SCALE
  if "env_push_enable" not in config:
    config.env_push_enable = sac_config.DEFAULT_ENV_PUSH_ENABLE
  if "env_zero_command_phase_freeze" not in config:
    config.env_zero_command_phase_freeze = (
        sac_config.DEFAULT_ENV_ZERO_COMMAND_PHASE_FREEZE
    )
  if "env_feet_air_time_command_mask" not in config:
    config.env_feet_air_time_command_mask = (
        sac_config.DEFAULT_ENV_FEET_AIR_TIME_COMMAND_MASK
    )
  for name in (
      "seed",
      "impl",
      "num_timesteps",
      "num_evals",
      "num_envs",
      "num_eval_envs",
      "batch_size",
      "actor_learning_rate",
      "critic_learning_rate",
      "alpha_learning_rate",
      "policy_obs_key",
      "value_obs_key",
      "min_replay_size",
      "max_replay_size",
      "grad_updates_per_step",
      "reward_scaling",
      "discounting",
      "tau",
      "target_entropy_coef",
      "deterministic_action_l2_coef",
      "actor_mean_l2_coef",
      "fixed_alpha",
      "alpha_floor",
      "alpha_loss_type",
      "env_feet_slip_mode",
      "env_feet_slip_scale",
      "env_push_enable",
      "env_zero_command_phase_freeze",
      "env_feet_air_time_command_mask",
      "normalize_observations",
      "deterministic_eval",
      "logdir",
      "render",
      "use_wandb",
      "policy_obs_size",
      "value_obs_size",
      "action_size",
  ):
    value = getattr(args, name)
    if value is not None:
      config[name] = value
  if args.policy_hidden_layer_sizes is not None:
    config.policy_hidden_layer_sizes = sac_config.parse_hidden_sizes(
        args.policy_hidden_layer_sizes, config.policy_hidden_layer_sizes
    )
  if args.q_hidden_layer_sizes is not None:
    config.q_hidden_layer_sizes = sac_config.parse_hidden_sizes(
        args.q_hidden_layer_sizes, config.q_hidden_layer_sizes
    )
  if args.dry_run:
    config.dry_run = True
  if args.allow_menagerie_download:
    config.allow_menagerie_download = True
  return config


def main(argv: Sequence[str] | None = None) -> int:
  args = _parse_args(argv)
  if args.render:
    raise NotImplementedError("Route B SAC render is not implemented yet.")
  if args.use_wandb:
    raise NotImplementedError("Route B SAC wandb logging is not implemented yet.")
  config = _build_config(args)
  from learning.sac_lift import train

  return train.main(config)


def run() -> None:
  sys.exit(main())


if __name__ == "__main__":
  run()
