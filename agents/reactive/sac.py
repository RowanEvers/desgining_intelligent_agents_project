import sys
import os

# Add the 'dia_repo' root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch 
import time
#RL 
from agents.base import BaseAgent
from environments.utils import set_seed, Logger
from stable_baselines3.common.callbacks import BaseCallback
from agents.base import BaseAgent

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


class LoggerCallback(BaseCallback):
    def __init__(self, logger,total_timesteps=10000):
        super().__init__()
        self.user_logger = logger 
        self.total_timesteps = total_timesteps
        self.episode_count = 0
        self.ep_start_time = time.time()
        self.start_time = time.time()
    def _on_step(self):
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self.episode_count += 1
                self.user_logger.log(self.num_timesteps, {
                    'episode_reward': info['episode']['r'],
                    'episode_length': info['episode']['l'],
                    'world_model_loss': 0.0,
                })
                print(
                    f"[{self.num_timesteps:>7} / {self.total_timesteps}]"
                    f"  ep {self.episode_count:>3}"
                    f"  reward: {info['episode']['r']:.1f}"
                    f"  len: {info['episode']['l']}"
                    f"  time: {time.time() - self.ep_start_time:.2f}s"
                )
                self.ep_start_time = time.time()
        return True

class SACagent(BaseAgent): 
    def __init__(self, env, seed, log_dir, logger, device, **hparams):
        super().__init__(env, seed, log_dir, logger, device, **hparams)
        monitored_env = Monitor(env)
        self.model = SAC(policy='MlpPolicy', env=monitored_env, seed=seed, tensorboard_log=log_dir, verbose=0, device=device, **hparams)
        self.hparams = hparams
    
    def train(self, total_timesteps):
        cb = LoggerCallback(self.logger, total_timesteps=total_timesteps)
        self.model.learn(total_timesteps=total_timesteps, callback=cb)
    
    def act(self, obs, deterministic=False):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path):
        self.model.save(path)
        np.savez(path + "_obs_norm.npz",
                    mean = self.env.env.obs_rms.mean,
                    var = self.env.env.obs_rms.var,
                    count = self.env.env.obs_rms.count
                )
    
    @classmethod
    def load(cls, path, env, seed, log_dir, logger, device, **hparams):

        instance = cls(env, seed, log_dir, logger, device, **hparams)
        instance.model = SAC.load(path.replace(".zip", ""), env=Monitor(env), device=device)
        norm_path = path + "_obs_norm.npz"
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            env.env.obs_rms.mean  = data["mean"]
            env.env.obs_rms.var   = data["var"]
            env.env.obs_rms.count = data["count"]
        return instance


