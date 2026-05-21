import sys
import os

# Add the 'dia_repo' root directory to the python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import torch
import time
#RL
from agents.base import BaseAgent
from environments.utils import set_seed, Logger, _find_wrapper_attr
from stable_baselines3.common.callbacks import BaseCallback
from agents.base import BaseAgent

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor


def _peek_raw_episode_return(training_env):
    # SB3 wraps in a VecEnv with .envs[0] = our Monitor(make_env(...)).
    candidates = []
    if hasattr(training_env, "envs"):
        candidates.extend(training_env.envs)
    else:
        candidates.append(training_env)

    for env in candidates:
        cur = env
        while cur is not None:
            if hasattr(cur, "return_queue") and len(cur.return_queue) > 0:
                # return_queue is a deque of recent raw episode returns.
                return float(cur.return_queue[-1])
            cur = getattr(cur, "env", None)
    return None


class LoggerCallback(BaseCallback):
    def __init__(self, logger, total_timesteps=10000):
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
                log_dict = {
                    'episode_reward': info['episode']['r'],   # normalized (Monitor sees post-NormalizeReward)
                    'episode_length': info['episode']['l'],
                    'world_model_loss': 0.0,
                }
                raw_r = _peek_raw_episode_return(self.training_env)
                if raw_r is not None:
                    log_dict['episode_reward_raw'] = raw_r
                self.user_logger.log(self.num_timesteps, log_dict)
                print(
                    f"[{self.num_timesteps:>7} / {self.total_timesteps}]"
                    f"  ep {self.episode_count:>3}"
                    f"  reward(norm): {info['episode']['r']:.3f}"
                    f"  reward(raw):  {raw_r if raw_r is not None else float('nan'):.1f}"
                    f"  len: {info['episode']['l']}"
                    f"  time: {time.time() - self.ep_start_time:.2f}s"
                )
                self.ep_start_time = time.time()
        return True


class SACagent(BaseAgent):
    def __init__(self, env, seed, log_dir, logger, device, **hparams):
        super().__init__(env, seed, log_dir, logger, device, **hparams)
        monitored_env = Monitor(env)
        self.model = SAC(policy='MlpPolicy', env=monitored_env, seed=seed,
                         tensorboard_log=log_dir, verbose=0, device=device, **hparams)
        self.hparams = hparams

    def prepare_for_adaptation(self) -> None:
        """Skip SB3's learning-starts warmup so the trained policy acts
        immediately from step 0 of adaptation.  Call this after load() and
        before train() whenever the agent is being fine-tuned rather than
        trained from scratch."""
        self.model.learning_starts = 0

    def train(self, total_timesteps):
        cb = LoggerCallback(self.logger, total_timesteps=total_timesteps)
        # reset_num_timesteps=True so the adaptation curve x-axis starts at 0
        # rather than continuing from the 1 M source-training steps.
        self.model.learn(total_timesteps=total_timesteps, callback=cb,
                         reset_num_timesteps=True)

    def act(self, obs, deterministic=False):
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action

    def save(self, path):
        self.model.save(path)
        # Walk the wrapper chain rather than assuming env.env.obs_rms — the
        # wrapper stack now includes RecordEpisodeStatistics in the middle,
        # so a hard-coded depth would break silently.
        _, obs_rms = _find_wrapper_attr(self.env, "obs_rms")
        if obs_rms is not None:
            np.savez(
                path + "_obs_norm.npz",
                mean=np.asarray(obs_rms.mean),
                var=np.asarray(obs_rms.var),
                count=float(obs_rms.count),
            )

    @classmethod
    def load(cls, path, env, seed, log_dir, logger, device, **hparams):
        instance = cls(env, seed, log_dir, logger, device, **hparams)
        # SB3 .load handles its own state restoration; we overwrite the
        # fresh model that __init__ built. The supplied env is what training
        # will continue on — for Exp 2, this is the *perturbed* env.
        instance.model = SAC.load(path.replace(".zip", ""),
                                  env=Monitor(env), device=device)
        norm_path = path + "_obs_norm.npz"
        if os.path.exists(norm_path):
            data = np.load(norm_path)
            _, obs_rms = _find_wrapper_attr(env, "obs_rms")
            if obs_rms is not None:
                obs_rms.mean = np.asarray(data["mean"])
                obs_rms.var = np.asarray(data["var"])
                obs_rms.count = float(data["count"])
        return instance
