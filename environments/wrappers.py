

import gymnasium as gym
from environments.utils import set_seed

def make_env(env_id, seed=42, render_mode= None):
    env = gym.make(env_id, render_mode=render_mode)
    env = gym.wrappers.NormalizeObservation(env)
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




