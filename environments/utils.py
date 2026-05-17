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


def _find_wrapper_attr(env, attr):
    """Walk a (possibly nested) gymnasium wrapper chain looking for an
    attribute. Returns the (wrapper, value) pair, or (None, None) if not
    found. Used to locate obs_rms / return_rms across whatever wrapper
    order ``make_env`` produces."""
    cur = env
    while cur is not None:
        if hasattr(cur, attr):
            return cur, getattr(cur, attr)
        cur = getattr(cur, "env", None)
    return None, None


def extract_env_norm_stats(env):
    """Pull the running mean/var/count out of a NormalizeObservation +
    NormalizeReward wrapper stack into a plain dict that's safe to
    pickle/torch.save. Missing wrappers are silently skipped so the same
    helper works for envs that aren't normalised.

    Returns a dict with keys ``obs_rms`` and/or ``return_rms``, each
    mapping to ``{"mean": np.ndarray, "var": np.ndarray, "count": float}``.
    """
    stats = {}
    _, obs_rms = _find_wrapper_attr(env, "obs_rms")
    if obs_rms is not None:
        stats["obs_rms"] = {
            "mean": np.asarray(obs_rms.mean),
            "var": np.asarray(obs_rms.var),
            "count": float(obs_rms.count),
        }
    # gymnasium >=0.29 uses `return_rms`; older versions used `returns_rms`.
    for attr in ("return_rms", "returns_rms"):
        _, ret_rms = _find_wrapper_attr(env, attr)
        if ret_rms is not None:
            stats["return_rms"] = {
                "mean": np.asarray(ret_rms.mean),
                "var": np.asarray(ret_rms.var),
                "count": float(ret_rms.count),
            }
            break
    return stats


def restore_env_norm_stats(env, stats, freeze=False):
    """Inverse of ``extract_env_norm_stats``: write the saved running stats
    back into the matching wrappers on ``env``. If ``freeze=True``, the
    wrappers are flipped into eval mode so further training does not
    drift the stats — this is what you want for adaptation (experiment 3).

    Silently no-ops for wrappers that aren't present in the chain.
    """
    if not stats:
        return env

    if "obs_rms" in stats:
        wrapper, obs_rms = _find_wrapper_attr(env, "obs_rms")
        if obs_rms is not None:
            obs_rms.mean = np.asarray(stats["obs_rms"]["mean"])
            obs_rms.var = np.asarray(stats["obs_rms"]["var"])
            obs_rms.count = float(stats["obs_rms"]["count"])
            if freeze and wrapper is not None:
                # NormalizeObservation has no public "freeze" flag, so we
                # monkey-patch its update method to a no-op. Crude but works.
                if hasattr(wrapper, "_update_running_mean"):
                    wrapper._update_running_mean = lambda *a, **kw: None

    if "return_rms" in stats:
        for attr in ("return_rms", "returns_rms"):
            wrapper, ret_rms = _find_wrapper_attr(env, attr)
            if ret_rms is not None:
                ret_rms.mean = np.asarray(stats["return_rms"]["mean"])
                ret_rms.var = np.asarray(stats["return_rms"]["var"])
                ret_rms.count = float(stats["return_rms"]["count"])
                if freeze and wrapper is not None:
                    if hasattr(wrapper, "_update_running_mean"):
                        wrapper._update_running_mean = lambda *a, **kw: None
                break

    return env


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
