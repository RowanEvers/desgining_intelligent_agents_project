'''
args for testing the three agents: 

3 agents, 3 seeds per agent, 3 envs with their default gradient steps, for a total of 27 runs.

'''
import os
import sys
import argparse

# Make the package importable when running as `python scripts/smoke_test.py`
# from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import torch
import gymnasium as gym

from environments.utils import set_seed, Logger
from agents.deliberative.latent_sac_agent import LatentSACAgent
from agents.deliberative.world_model.world_model import GaussianWorldModel
from agents.deliberative.world_model.world_model_categorical import CategoricalWorldModel
from scripts.smoke_test import build_agent, assert_finite





def main():
    ENVS = ["Pendulum-v5", "Hopper-v5", "Walker2d-v5"]
    AGENTS = ["sac", "gaussian", "discrete"]
    GRADIENT_STEPS = [5e4,5e5,1e6]
    seeds = [0, 1, 2]
    device = "gpu" if torch.cuda.is_available() else "cpu"

    for env_id in ENVS:
        for agent_kind in AGENTS:
            for gradient_steps in GRADIENT_STEPS:
                for seed in seeds:
                    print(f"Testing {agent_kind} on {env_id} with {gradient_steps} steps (seed {seed})")
                    env = gym.make(env_id)
                    set_seed(seed, env)
                    logger = Logger(agent_name=agent_kind, env_id=env_id, seed=seed, log_dir="logs")
                    agent = build_agent(env, seed, logger, device=device, latent_kind=agent_kind)
                    agent.train(total_timesteps=gradient_steps)
                    assert_finite(agent)
                    logger.close()



if __name__ == "__main__":
    main()