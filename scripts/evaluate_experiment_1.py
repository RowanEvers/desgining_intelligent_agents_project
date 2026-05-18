"""Evaluate all 27 experiment-1 checkpoints in an UN-NORMALIZED-REWARD env
to recover raw episodic returns. Lets us decide whether the flat training
curves we saw were 'agent didn't learn' or 'NormalizeReward hid the learning'.

Eval env stack: gym.make() + NormalizeObservation (with frozen training stats)
+ RecordEpisodeStatistics. NormalizeReward is OMITTED so episode returns
recorded are raw.

Run from repo root::

    python scripts/evaluate_experiment_1.py --episodes 10

Writes ``logs/experiment_1/eval/eval_results.csv`` and a summary printout.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import gymnasium as gym

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from stable_baselines3 import SAC as SB3_SAC
from agents.deliberative.latent_sac_agent import LatentSACAgent
from environments.utils import set_seed


LOG_ROOT = Path("logs/experiment_1")
OUT_DIR = LOG_ROOT / "eval"
RUN_RE = re.compile(r"^(?P<agent>sac|gaussian|categorical)_(?P<env>[\w\-]+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$")


def make_eval_env(env_id: str, seed: int) -> gym.Env:
    """Eval env: NormalizeObservation (will be patched with training stats)
    + RecordEpisodeStatistics, but NO NormalizeReward — so info['episode']['r']
    is the RAW episode return."""
    env = gym.make(env_id)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    set_seed(seed, env)
    return env


def patch_obs_rms(env: gym.Env, mean: np.ndarray, var: np.ndarray, count: float) -> None:
    """Walk wrapper chain to find NormalizeObservation and overwrite its
    running stats with the saved training-time stats. Also freeze updates."""
    cur = env
    while cur is not None:
        if hasattr(cur, "obs_rms"):
            cur.obs_rms.mean = np.asarray(mean)
            cur.obs_rms.var = np.asarray(var)
            cur.obs_rms.count = float(count)
            # Freeze running-mean updates so eval doesn't drift the stats
            if hasattr(cur, "_update_running_mean"):
                cur._update_running_mean = lambda *a, **kw: None
            return
        cur = getattr(cur, "env", None)
    raise RuntimeError("No NormalizeObservation wrapper found in env chain")


def load_sac_obs_stats(ckpt_path: Path) -> tuple[np.ndarray, np.ndarray, float] | None:
    """SACagent saves obs-norm stats to <ckpt>_obs_norm.npz."""
    npz = ckpt_path.parent / f"{ckpt_path.name}_obs_norm.npz"
    if not npz.exists():
        return None
    data = np.load(npz)
    return data["mean"], data["var"], float(data["count"])


def evaluate_checkpoint(run_dir: Path, agent_kind: str, env_id: str, seed: int,
                        n_episodes: int, device: str) -> dict:
    ckpt = run_dir / "final.zip"
    eval_env = make_eval_env(env_id, seed=seed + 10_000)  # offset seed to avoid overlap with training

    if agent_kind == "sac":
        # SB3's SAC.load handles its own state restoration. We supply the env so
        # predict() can read its action space. We then patch obs_rms ourselves.
        model = SB3_SAC.load(str(ckpt).replace(".zip", ""), env=None, device=device)
        stats = load_sac_obs_stats(ckpt)
        if stats is not None:
            patch_obs_rms(eval_env, *stats)

        def policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

    else:  # gaussian or categorical
        agent = LatentSACAgent.load(
            str(ckpt), env=eval_env, seed=seed, device=device,
            restore_env_stats=True,   # restores obs_rms (and return_rms if present)
            freeze_env_stats=True,
        )

        def policy(obs):
            return agent.act(obs, deterministic=True)

    # Roll out n_episodes
    returns = []
    lengths = []
    for ep in range(n_episodes):
        obs, _ = eval_env.reset(seed=seed + 10_000 + ep)
        ep_return = 0.0
        ep_length = 0
        done = False
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_return += float(reward)
            ep_length += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(ep_length)

    eval_env.close()

    return {
        "agent": agent_kind, "env": env_id, "seed": seed,
        "n_episodes": n_episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "min_return": float(np.min(returns)),
        "max_return": float(np.max(returns)),
        "mean_length": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10,
                        help="Deterministic eval episodes per checkpoint.")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu or cuda; eval is fast on cpu so default to that.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(LOG_ROOT.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        if not (d / "final.zip").exists():
            print(f"  [skip] no final.zip in {d.name}")
            continue
        runs.append((d, m["agent"], m["env"], int(m["seed"])))

    print(f"Evaluating {len(runs)} checkpoints, {args.episodes} eps each, on {args.device}\n")

    results = []
    for i, (d, agent_kind, env_id, seed) in enumerate(runs, 1):
        print(f"[{i:>2}/{len(runs)}] {d.name}")
        try:
            row = evaluate_checkpoint(d, agent_kind, env_id, seed,
                                      args.episodes, args.device)
            print(f"           return = {row['mean_return']:8.2f} ± {row['std_return']:.2f}"
                  f"   length = {row['mean_length']:6.1f}")
            results.append(row)
        except Exception as e:
            print(f"           [error] {type(e).__name__}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "eval_results.csv", index=False)
    print(f"\nSaved per-seed results to {OUT_DIR / 'eval_results.csv'}")

    # Aggregate across seeds for a paper-ready summary table
    if not df.empty:
        summary = (df.groupby(["env", "agent"])
                     .agg(mean_return=("mean_return", "mean"),
                          std_return=("mean_return", "std"),
                          mean_length=("mean_length", "mean"))
                     .reset_index())
        summary.to_csv(OUT_DIR / "eval_summary.csv", index=False)
        print(f"Saved (env, agent) summary to {OUT_DIR / 'eval_summary.csv'}\n")
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
