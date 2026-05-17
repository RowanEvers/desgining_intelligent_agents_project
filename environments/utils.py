#seeding
import random
import numpy as np
import torch

#logging
import csv
import os
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if env is not None:
        env.reset(seed=seed)
    
    return env

class Logger:
    '''Simple logger that writes metrics to both TensorBoard and a CSV file.'''
    def __init__(self, log_dir: str, agent_name: str, env_id: str, seed: int):
        self.run_name = f"{agent_name}_{env_id}_seed{seed}"
        self.log_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # TensorBoard for live training curves
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # CSV for easy post-run analysis
        self.csv_path = os.path.join(self.log_dir, "metrics.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["step", "episode_reward", "episode_length", "world_model_loss"])

    def log(self, step: int, metrics: dict):
        # Write to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(key, value, step)
        
        # Write to CSV
        self.csv_writer.writerow([
            step,
            metrics.get("episode_reward", ""),
            metrics.get("episode_length", ""),
            metrics.get("world_model_loss", "")
        ])
        self.csv_file.flush()

    def close(self):
        self.writer.close()
        self.csv_file.close()

