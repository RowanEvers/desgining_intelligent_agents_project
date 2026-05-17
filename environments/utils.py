#seeding
import random
import numpy as np
import torch

#logging
import json
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


def _to_python_scalar(value):
    """Best-effort conversion of tensor/np scalars to plain Python numbers
    so they're JSON-serialisable. Returns the value unchanged if it's
    already a primitive."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item() if value.numel() == 1 else value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


class Logger:
    '''Logger that writes metrics to TensorBoard *and* to a JSONL file.

    The JSONL file is schemaless: every call to ``log(step, metrics)`` appends
    one line containing ``step`` plus whatever keys are in ``metrics``. This
    avoids the old CSV behaviour where only a hard-coded subset of columns
    was captured and everything else was silently dropped.

    Load with::

        import pandas as pd
        df = pd.read_json("metrics.jsonl", lines=True)
    '''
    def __init__(self, log_dir: str, agent_name: str, env_id: str = None, seed: int = None):
        # env_id and seed are kept on the signature for backward compatibility
        # with existing call sites; the run_name no longer doubles them up.
        self.run_name = f"{agent_name}"
        self.log_dir = os.path.join(log_dir, self.run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        # TensorBoard for live training curves
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # JSONL for post-run analysis — variable schema, one row per log() call
        self.jsonl_path = os.path.join(self.log_dir, "metrics.jsonl")
        self.jsonl_file = open(self.jsonl_path, "w")

    def log(self, step: int, metrics: dict):
        # Write to TensorBoard
        for key, value in metrics.items():
            scalar = _to_python_scalar(value)
            if isinstance(scalar, (int, float)):
                self.writer.add_scalar(key, scalar, step)

        # Write to JSONL — every key is preserved, no schema enforced
        row = {"step": int(step)}
        for k, v in metrics.items():
            row[k] = _to_python_scalar(v)
        self.jsonl_file.write(json.dumps(row) + "\n")
        self.jsonl_file.flush()

    def close(self):
        self.writer.close()
        self.jsonl_file.close()
