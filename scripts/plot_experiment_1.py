"""Publication-quality figures for experiment 1.

Run from the repo root::

    python scripts/plot_experiment_1.py

Outputs land in ``logs/experiment_1/figures/`` as PNG + PDF.

Produces two figures:
  1. ``learning_curves.{png,pdf}`` — 3 panels (one per env), 3 curves per panel
     (SAC / Gaussian / Categorical), mean across 3 seeds with a ±1 std band.
  2. ``final_performance.{png,pdf}`` — grouped bar chart of mean episode return
     over the last 10% of training, error bars = std across seeds.

The aggregation strategy: each run logs at episode boundaries, so different
runs have different numbers of points and different step values. To average
across seeds, we resample every run onto a common 200-point step grid via
linear interpolation, then take mean/std across seeds at each grid point.
"""
from __future__ import annotations

import json
import os
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Make the package importable when running as `python scripts/plot_experiment_1.py`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LOG_ROOT = Path("logs/experiment_1")
OUT_DIR = LOG_ROOT / "figures"
ENVS = ["Pendulum-v1", "Hopper-v5", "Walker2d-v5"]
AGENTS = ["sac", "gaussian", "categorical"]
AGENT_LABELS = {
    "sac": "SAC (baseline)",
    "gaussian": "Latent-SAC (Gaussian WM)",
    "categorical": "Latent-SAC (Categorical WM)",
}
AGENT_COLORS = {
    "sac": "#444444",          # dark grey — baseline
    "gaussian": "#1f77b4",     # blue
    "categorical": "#d62728",  # red
}
# Run-tag pattern: e.g. "categorical_Walker2d-v5_steps1000000_seed2"
RUN_RE = re.compile(r"^(?P<agent>sac|gaussian|categorical)_(?P<env>[\w\-]+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$")

N_GRID = 200       # number of points to resample each curve to
SMOOTH_FRAC = 0.04 # rolling-window size as a fraction of total episodes (~4%).
                   # Long runs (10k+ eps) get a wide smoother; short runs (200 eps)
                   # get a small one. Min window of 11 to avoid degenerate smoothing.
SMOOTH_MIN = 11
FINAL_FRAC = 0.10  # fraction of training (by episode index) to average for final perf


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_run(jsonl_path: Path) -> pd.DataFrame | None:
    """Load a single run's metrics.jsonl into a dataframe with columns
    ['step', 'episode_reward']. Returns None for empty / malformed files."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError:
        return None
    if df.empty or "episode_reward" not in df.columns:
        return None
    # Some rows may carry NaN reward (e.g. WM-loss-only rows in future versions).
    df = df.dropna(subset=["episode_reward"]).reset_index(drop=True)
    return df[["step", "episode_reward"]]


def discover_runs(root: Path) -> pd.DataFrame:
    """Walk ``root`` and return a dataframe of (agent, env, seed, path) tuples."""
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        m = RUN_RE.match(d.name)
        if not m:
            continue
        jsonl = d / "metrics.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            print(f"  [skip] no metrics: {d.name}")
            continue
        rows.append({
            "agent": m["agent"],
            "env": m["env"],
            "seed": int(m["seed"]),
            "path": jsonl,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def smoothed_curve(df: pd.DataFrame) -> pd.DataFrame:
    """Adaptive rolling mean: window proportional to run length so a 10k-episode
    Hopper run gets a much wider smoother than a 250-episode Pendulum run."""
    out = df.copy().sort_values("step").reset_index(drop=True)
    win = max(SMOOTH_MIN, int(len(out) * SMOOTH_FRAC))
    out["episode_reward"] = (
        out["episode_reward"].rolling(window=win, min_periods=1, center=True).mean()
    )
    return out


def resample_to_grid(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    """Linear-interpolate the reward curve onto a fixed step grid.
    Outside the run's step range, values are NaN (so the seed doesn't fake
    coverage it doesn't have)."""
    steps = df["step"].to_numpy()
    rewards = df["episode_reward"].to_numpy()
    out = np.interp(grid, steps, rewards, left=np.nan, right=np.nan)
    return out


def aggregate_seeds(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-seed curves and return (mean, std), ignoring NaNs."""
    arr = np.vstack(curves)  # shape (n_seeds, n_grid)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(arr, axis=0)
        std = np.nanstd(arr, axis=0)
    return mean, std


# ---------------------------------------------------------------------------
# Plot 1: learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(runs: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)

    for ax, env in zip(axes, ENVS):
        env_runs = runs[runs["env"] == env]
        if env_runs.empty:
            ax.set_title(f"{env}\n(no data)")
            continue

        # Build a common step grid spanning the env's max training horizon
        max_step = 0
        for _, r in env_runs.iterrows():
            df = load_run(r["path"])
            if df is not None:
                max_step = max(max_step, df["step"].max())
        if max_step == 0:
            ax.set_title(f"{env}\n(no data)")
            continue
        grid = np.linspace(0, max_step, N_GRID)

        for agent in AGENTS:
            agent_runs = env_runs[env_runs["agent"] == agent]
            if agent_runs.empty:
                continue

            seed_curves = []
            for _, r in agent_runs.iterrows():
                df = load_run(r["path"])
                if df is None or len(df) < 2:
                    continue
                df = smoothed_curve(df)
                seed_curves.append(resample_to_grid(df, grid))

            if not seed_curves:
                continue
            mean, std = aggregate_seeds(seed_curves)
            color = AGENT_COLORS[agent]
            ax.plot(grid, mean, label=AGENT_LABELS[agent], color=color, linewidth=2)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

        ax.set_title(env, fontsize=12)
        ax.set_xlabel("Environment steps")
        ax.grid(True, linestyle="--", alpha=0.4)
        # human-friendly x-axis (e.g. 200k instead of 200000)
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    axes[0].set_ylabel("Episode return (normalized)")
    # Shared legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Pendulum may not have all agents; fall back to whichever panel has the most
    for ax in axes[1:]:
        h, l = ax.get_legend_handles_labels()
        if len(h) > len(handles):
            handles, labels = h, l
    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, -0.06), frameon=False)
    fig.suptitle("Experiment 1: learning curves (mean ± 1 std over 3 seeds)",
                 fontsize=13, y=1.02)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "learning_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'learning_curves.png'}")
    print(f"  wrote {out_dir / 'learning_curves.pdf'}")


# ---------------------------------------------------------------------------
# Plot 2: final-performance bar chart
# ---------------------------------------------------------------------------
def plot_final_performance(runs: pd.DataFrame, out_dir: Path):
    # For each (env, agent, seed): mean reward over the last FINAL_FRAC of episodes
    records = []
    for _, r in runs.iterrows():
        df = load_run(r["path"])
        if df is None or len(df) < 2:
            continue
        n_tail = max(1, int(len(df) * FINAL_FRAC))
        final_reward = df["episode_reward"].iloc[-n_tail:].mean()
        records.append({
            "env": r["env"], "agent": r["agent"], "seed": r["seed"],
            "final_reward": final_reward,
        })
    finals = pd.DataFrame(records)
    if finals.empty:
        print("  [skip] no data for final-performance plot")
        return

    # Aggregate across seeds
    agg = (finals.groupby(["env", "agent"])["final_reward"]
                  .agg(["mean", "std"])
                  .reset_index())

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(ENVS))
    bar_w = 0.25

    for i, agent in enumerate(AGENTS):
        sub = agg[agg["agent"] == agent].set_index("env").reindex(ENVS)
        means = sub["mean"].to_numpy()
        stds = sub["std"].fillna(0).to_numpy()
        ax.bar(x + (i - 1) * bar_w, means, width=bar_w,
               yerr=stds, capsize=4,
               label=AGENT_LABELS[agent], color=AGENT_COLORS[agent],
               edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ENVS)
    ax.set_ylabel(f"Mean return over final {int(FINAL_FRAC * 100)}% of episodes")
    ax.set_title("Experiment 1: final performance (mean ± std over 3 seeds)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    fig.savefig(out_dir / "final_performance.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "final_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'final_performance.png'}")
    print(f"  wrote {out_dir / 'final_performance.pdf'}")

    # Also dump the aggregated numbers to CSV so they're easy to paste into a paper table
    agg.to_csv(out_dir / "final_performance.csv", index=False)
    print(f"  wrote {out_dir / 'final_performance.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not LOG_ROOT.exists():
        raise SystemExit(f"Log root not found: {LOG_ROOT.resolve()}")

    print(f"Discovering runs under {LOG_ROOT} ...")
    runs = discover_runs(LOG_ROOT)
    if runs.empty:
        raise SystemExit("No runs found.")
    print(f"  found {len(runs)} runs")
    print(runs.groupby(["env", "agent"]).size().unstack(fill_value=0))

    print("\nBuilding learning curves ...")
    plot_learning_curves(runs, OUT_DIR)

    print("\nBuilding final-performance bar chart ...")
    plot_final_performance(runs, OUT_DIR)

    print("\nDone. Figures saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
