from __future__ import annotations

import gymnasium as gym
import numpy as np

from environments.utils import set_seed


def _apply_physics_perturbation(env, gravity_scale = 1.0,
                                friction_scale = 1.0):
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
             gravity_scale = 1.0, friction_scale = 1.0,
             normalize_reward = True):
    env = gym.make(env_id, render_mode=render_mode)
    _apply_physics_perturbation(env, gravity_scale, friction_scale)
    env = gym.wrappers.NormalizeObservation(env)
    #  RecordEpisodeStatistics must sit BEFORE NormalizeReward so
    # info["episode"]["r"] reports the raw episodic return. Don't reorder
    # these two lines without updating the loggers downstream.
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if normalize_reward:
        env = gym.wrappers.NormalizeReward(env)
    set_seed(seed, env)
    return env


# envs selected for testing: half-cheetah-v5
