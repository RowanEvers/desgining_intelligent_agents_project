"""A/B smoke test: does NormalizeReward hurt SAC on HalfCheetah-v5?

Trains the SB3 SAC baseline on HalfCheetah-v5 twice for 50k steps each:
  - Run A: WITH ``NormalizeReward`` (the Exp 1 wrapper stack)
  - Run B: WITHOUT ``NormalizeReward`` (standard SAC/MuJoCo practice)

After both finish, prints a side-by-side summary of:
  - mean raw episode return over the last 10% of episodes
  - max raw episode return seen
  - episode count

Decision rule:
  * If Run B's last-10% mean is ≥ 2× Run A's: NormalizeReward is hurting SAC.
    Switch experiment_3.py / experiment_4.py to use normalize_reward=False.
  * If they're comparable: keep the current wrapper stack, the slowdown
    is from somewhere else.

Expected runtime: ~15-25 min on one GPU (two 50k-step SAC runs sequentially).
Same seed used for both runs so the comparison is apples-to-apples.

Run from the repo root::

    python scripts/ab_normalize_reward.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.utils import Logger
from environments.wrappers import make_env
from agents.reactive.sac import SACagent


ENV_ID = "HalfCheetah-v5"
SEED = 0
STEPS = 50_000
LOG_ROOT = Path("logs/ab_normalize_reward")


def run_one(tag: str, normalize_reward: bool) -> Path:
    """Train one SAC run with/without NormalizeReward. Returns log dir."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_log_dir = LOG_ROOT / tag
    if (run_log_dir / "final.zip").exists():
        print(f"[skip-already-done] {tag}")
        return run_log_dir
    run_log_dir.mkdir(parents=True, exist_ok=True)

    env = make_env(ENV_ID, seed=SEED, normalize_reward=normalize_reward)
    logger = Logger(log_dir=str(LOG_ROOT), agent_name=tag,
                    env_id=ENV_ID, seed=SEED)

    print(f"\n=== Training {tag} (normalize_reward={normalize_reward}) "
          f"for {STEPS} steps on {device} ===")
    agent = SACagent(env=env, seed=SEED, log_dir=str(LOG_ROOT),
                     logger=logger, device=device, learning_rate=3e-4)
    agent.train(total_timesteps=STEPS)
    agent.save(str(run_log_dir / "final.zip"))
    logger.close()
    return run_log_dir


def summarize(run_log_dir: Path) -> dict:
    """Read the JSONL, return summary stats over raw episode returns."""
    jsonl = run_log_dir / "metrics.jsonl"
    rows = []
    with open(jsonl) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Prefer raw column if present; fall back to normalized.
            r = d.get("episode_reward_raw", d.get("episode_reward"))
            if r is not None:
                rows.append(float(r))
    if not rows:
        return {"n_episodes": 0}

    arr = np.asarray(rows)
    n_tail = max(1, len(arr) // 10)
    return {
        "n_episodes": len(arr),
        "first_5_mean": float(arr[:5].mean()) if len(arr) >= 5 else float(arr[0]),
        "last_10pct_mean": float(arr[-n_tail:].mean()),
        "last_10pct_std": float(arr[-n_tail:].std()),
        "max": float(arr.max()),
        "min": float(arr.min()),
    }


def main():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    # Sequential — no risk of GPU contention.
    log_a = run_one("with_normalizeReward",  normalize_reward=True)
    log_b = run_one("without_normalizeReward", normalize_reward=False)

    stats_a = summarize(log_a)
    stats_b = summarize(log_b)

    print("\n" + "=" * 78)
    print("  A/B summary: SAC × HalfCheetah-v5 × 50k steps × seed 0")
    print("=" * 78)
    header = f"  {'metric':<22} {'WITH NormalizeR':>22} {'WITHOUT NormalizeR':>22}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    def row(key, fmt="{:8.1f}"):
        a = stats_a.get(key, "—")
        b = stats_b.get(key, "—")
        a_s = fmt.format(a) if isinstance(a, (int, float)) else str(a)
        b_s = fmt.format(b) if isinstance(b, (int, float)) else str(b)
        print(f"  {key:<22} {a_s:>22} {b_s:>22}")

    row("n_episodes", fmt="{:8d}")
    row("first_5_mean")
    row("last_10pct_mean")
    row("last_10pct_std")
    row("max")
    row("min")

    print("=" * 78)
    a_last = stats_a.get("last_10pct_mean", 0)
    b_last = stats_b.get("last_10pct_mean", 0)
    if a_last > 0 and b_last > 0 and b_last / a_last >= 2.0:
        print(f"\n  VERDICT: WITHOUT NormalizeReward is {b_last/a_last:.1f}× better.")
        print( "    -> Switch experiment_3.py / experiment_4.py to call")
        print( "       make_env(..., normalize_reward=False).")
    elif b_last > a_last * 1.3:
        print(f"\n  VERDICT: WITHOUT NormalizeReward is {b_last/a_last:.2f}× better,")
        print( "    a noticeable but not dramatic improvement. Probably worth")
        print( "    switching.")
    elif abs(b_last - a_last) / max(abs(a_last), abs(b_last), 1) < 0.2:
        print( "\n  VERDICT: Roughly equivalent. NormalizeReward isn't the bottleneck.")
        print( "    Look elsewhere for the underperformance (hparams, buffer size, etc.)")
    else:
        print(f"\n  VERDICT: WITH NormalizeReward is better. Unusual but possible.")
        print( "    Keep current wrapper stack.")


if __name__ == "__main__":
    main()
