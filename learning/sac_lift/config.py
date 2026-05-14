"""Local constants and config helpers for the G1 SAC baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

DEFAULT_POLICY_OBS_SIZE = 103
DEFAULT_VALUE_OBS_SIZE = 216
DEFAULT_ACTION_SIZE = 29
DEFAULT_DETERMINISTIC_ACTION_L2_COEF = 0.0
DEFAULT_ACTOR_MEAN_L2_COEF = 0.0
DEFAULT_FIXED_ALPHA = 0.0
DEFAULT_ALPHA_FLOOR = 0.0
DEFAULT_ALPHA_LOSS_TYPE = "exp_alpha"
ALPHA_LOSS_TYPE_IDS = {
    "exp_alpha": 0,
    "log_alpha": 1,
}


def parse_hidden_sizes(value: Any, default: Sequence[int]) -> tuple[int, ...]:
  """Parses CLI hidden-size values."""
  if value is None:
    return tuple(default)
  if isinstance(value, str):
    value = [value]
  if len(value) == 1 and isinstance(value[0], str) and "," in value[0]:
    value = [v for v in value[0].split(",") if v]
  return tuple(int(v) for v in value)


def to_plain_dict(config: Any) -> dict[str, Any]:
  if hasattr(config, "to_dict"):
    return config.to_dict()
  return dict(config)
