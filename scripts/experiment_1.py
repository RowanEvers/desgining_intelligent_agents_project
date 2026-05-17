'''
args for testing the three agents: 

3 agents, 3 seeds per agent, 3 envs with their default gradient steps, for a total of 27 runs.

'''
import os
import sys
import argparse

from certifi.__main__ import args

# Make the package importable when running as `python scripts/smoke_test.py`
# from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import torch
import gymnasium as gym

from environments.utils import Logger, set_seed
from environments.wrappers import make_env
from agents.deliberative.latent_sac_agent import LatentSACAgent
from agents.deliberative.world_model.world_model import GaussianWorldModel
from agents.deliberative.world_model.world_model_categorical import CategoricalWorldModel
from agents.reactive.sac import SACagent
from scripts.smoke_test import banner, assert_finite

def build_agent(env, seed, logger, device, latent_kind, log_dir):
    """Construct a LatentSACAgent wired with the requested world model.

    Common SAC hyperparameters are kept identical between the two variants
    so the comparison is honest — only the latent representation changes.
    """
    common_hparams = dict(
        hidden_dim=128,
        horizon=5,
        batch_size=64,
        env_steps_per_iter=200,
        wm_updates_per_iter=20,
        sac_updates_per_iter=20,
        warmup_steps=500,
        env_buffer_size=10_000,
        imag_buffer_size=10_000,
    )

    reactive_hparams = dict(
        learning_rate=3e-4,
    )

    if latent_kind == "gaussian":
        return LatentSACAgent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            world_model_cls=GaussianWorldModel,
            world_model_kwargs={"latent_dim": 16},     # small for fast smoke test
            **common_hparams,
        )

    if latent_kind == "categorical":
        # Smaller categorical structure than Dreamer V2's 32x32 to keep
        # the smoke test fast on CPU. 8 categoricals x 8 classes = 64-d flat
        # latent — comparable capacity to the Gaussian's 16-d latent.
        return LatentSACAgent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            world_model_cls=CategoricalWorldModel,
            world_model_kwargs={"num_cat": 8, "num_classes": 8},
            **common_hparams,
        )
    if latent_kind == "sac":
        # Vanilla SAC without a world model — just to sanity check the training loop itself.
        return SACagent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            **reactive_hparams,
        )

    raise ValueError(f"Unknown latent kind: {latent_kind}")

def main():
    ENVS = ["Pendulum-v1", "Hopper-v5", "Walker2d-v5"]
    AGENTS = ["sac", "gaussian", "categorical"]
    GRADIENT_STEPS = [5e4,5e5,1e6]
    GRADIENT_STEPS_SMOKE = [1e4, 1e5, 1e5]
    seeds = [0, 1, 2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    runs = 0

    for i,env_id in enumerate(ENVS):
        for agent_kind in AGENTS:
            for seed in seeds:
                runs += 1 
                run_tag = f"{agent_kind}_{env_id}_steps{int(GRADIENT_STEPS_SMOKE[i])}_seed{seed}"
                log_dir = f"logs/experiment_1/"
                print(f"Running {run_tag} on {device}...")

                env = make_env(env_id, seed=seed) 
                env = set_seed(seed, env)
                logger = Logger(log_dir=log_dir, agent_name=run_tag, env_id=env_id, seed=seed)
                agent = build_agent(env, seed, logger, device, agent_kind, log_dir=log_dir)
                # Train the agent
                agent.train(total_timesteps=int(GRADIENT_STEPS_SMOKE[i]))
                # Save the final model
                agent.save(f"{logger.log_dir}/final.zip")
                logger.close()
                    
    banner(f"Total runs: {runs} completed, results saved to logs/experiment_1/ directory")
    
if __name__ == "__main__":
    main()