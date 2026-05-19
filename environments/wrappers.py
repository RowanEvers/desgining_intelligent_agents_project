"""Env construction + physics perturbation helpers.

The wrapper stack used everywhere in this repo is, inner-to-outer:

    gym.make(env_id)
      -> NormalizeObservation
      -> RecordEpisodeStatistics      # sees raw rewards (before NormalizeReward)
      -> NormalizeReward

Crucially, RecordEpisodeStatistics sits BEFORE NormalizeReward. That means
``info["episode"]["r"]`` returned on terminal steps carries the RAW episodic
return, not the scaled one — which is the signal you actually want to plot.
The reward returned from ``env.step`` is still the normalized one (used by
SAC for stable gradients).

For Experiment 2 (adaptation), the same ``make_env`` can be called with
``gravity_scale`` / ``friction_scale`` to perturb the underlying physics
before any wrappers attach. The wrappers' running stats then learn the
perturbed distribution.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from environments.utils import set_seed


def _apply_physics_perturbation(env, gravity_scale: float = 1.0,
                                friction_scale: float = 1.0) -> None:
    """Patch the unwrapped env's physics in place.

    Handles two backends:
      * MuJoCo envs (Hopper-v5, Walker2d-v5, HalfCheetah-v5, …) expose
        ``unwrapped.model.opt.gravity`` (3-vec) and
        ``unwrapped.model.geom_friction`` ((n_geom, 3) array).
      * Pendulum-v1 has no MuJoCo model — gravity is a scalar ``unwrapped.g``
        (default 10.0). It has no friction analogue (silently ignored).

    A scale of exactly 1.0 is a no-op for that axis, so passing the default
    keyword args reproduces the un-perturbed env used in Experiment 1.
    """
    unwrapped = env.unwrapped

    if gravity_scale != 1.0:
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "opt"):
            # MuJoCo env: scale the gravity 3-vector in place.
            unwrapped.model.opt.gravity[:] = (
                np.asarray(unwrapped.model.opt.gravity) * gravity_scale
            )
        elif hasattr(unwrapped, "g"):
            # Classic-control env (Pendulum) — gravity is a scalar attribute.
            unwrapped.g = float(unwrapped.g) * gravity_scale
        else:
            raise ValueError(
                f"Don't know how to scale gravity for env "
                f"{type(unwrapped).__name__}; expected MuJoCo .model or "
                f"classic-control .g"
            )

    if friction_scale != 1.0:
        if hasattr(unwrapped, "model") and hasattr(unwrapped.model, "geom_friction"):
            # geom_friction is (n_geom, 3): sliding, torsional, rolling.
            # Uniform scalar scaling is the standard sim-to-real perturbation.
            unwrapped.model.geom_friction[:] = (
                np.asarray(unwrapped.model.geom_friction) * friction_scale
            )
        # Pendulum / non-MuJoCo: no-op. We don't raise so the same call site
        # works for every env in the grid.


def make_env(env_id, seed=42, render_mode=None,
             gravity_scale: float = 1.0, friction_scale: float = 1.0):
    """Build a wrapped env.

    Args:
        env_id: gym id (e.g. ``"Hopper-v5"`` or ``"Pendulum-v1"``).
        seed: RNG seed forwarded to ``set_seed`` (which seeds Python, numpy,
            torch, and the env).
        render_mode: forwarded to ``gym.make``. Leave ``None`` for training.
        gravity_scale, friction_scale: physics perturbations. Defaults of
            ``1.0`` reproduce the Experiment 1 stack exactly.
    """
    env = gym.make(env_id, render_mode=render_mode)
    _apply_physics_perturbation(env, gravity_scale, friction_scale)
    env = gym.wrappers.NormalizeObservation(env)
    # IMPORTANT: RecordEpisodeStatistics must sit BEFORE NormalizeReward so
    # info["episode"]["r"] reports the raw episodic return. Don't reorder
    # these two lines without updating the loggers downstream.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.NormalizeReward(env)
    set_seed(seed, env)
    return env


# envs selected for testing: half-cheetah-v5, humanoid-v5, swimmer-v5, humanoidStandup-v5.

#TESTINGTESTINGTESTING
test_env = False
if test_env:
    env = make_env("HalfCheetah-v5", seed=42, render_mode="human")
    obs, info = env.reset()
    for i in range(1000):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()
    env.close()
