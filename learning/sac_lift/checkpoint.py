"""Simple pickle checkpoint helpers for SAC smoke runs."""

from __future__ import annotations

import os
import pickle
from typing import Any


def save(logdir: str, step: int, payload: dict[str, Any]) -> str:
  os.makedirs(logdir, exist_ok=True)
  path = os.path.join(logdir, f"sac_lift_step_{step}.pkl")
  tmp_path = f"{path}.tmp"
  with open(tmp_path, "wb") as f:
    pickle.dump(payload, f)
  os.replace(tmp_path, path)
  return path


def load(path: str) -> dict[str, Any]:
  with open(path, "rb") as f:
    return pickle.load(f)
