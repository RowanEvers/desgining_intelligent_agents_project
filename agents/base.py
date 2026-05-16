
from abc import ABC, abstractmethod
import numpy as np





class BaseAgent(ABC):  # parent class for all agents
    def __init__(self, env, seed, log_dir, logger, device, **hparams):
        self.env = env
        self.seed = seed
        self.log_dir = log_dir
        self.logger = logger
        self.device = device
        self.hparams = hparams
    
    @abstractmethod
    def train(self,total_timesteps):
        pass 

    @abstractmethod
    def act(self, obs, deterministic=False):
        pass 

    @abstractmethod
    def save(self, path):
        pass

    @classmethod
    @abstractmethod
    def load(cls, path,env):
        pass

    def evaluate(self, env, num_episodes=10, deterministic=True):

        episode_returns = []
        for episode in range(num_episodes):
            obs, info = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = self.act(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated
            episode_returns.append(episode_reward)
        avg_reward = sum(episode_returns) / num_episodes
        return {'mean': avg_reward, 'std': np.std(episode_returns), 'returns': episode_returns}
