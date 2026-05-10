"""G1 Humanoid Robot Training Environment.

This package provides a complete training pipeline for the Unitree G1 humanoid robot,
extracted from MuJoCo Playground. It includes:
- G1 environment definitions (flat terrain, rough terrain)
- MJX-based simulation
- Domain randomization
- Brax PPO and RSL-RL training support
"""

from g1_env._src import locomotion
from g1_env._src import registry
from g1_env._src import wrapper
from g1_env._src import wrapper_torch
from g1_env._src.mjx_env import MjxEnv
from g1_env._src.mjx_env import render_array
from g1_env._src.mjx_env import State
from g1_env._src.mjx_env import step

__version__ = "1.0.0"

__all__ = [
    "locomotion",
    "MjxEnv",
    "registry",
    "render_array",
    "State",
    "step",
    "wrapper",
    "wrapper_torch",
]
