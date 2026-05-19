"""Publication-quality figures for experiment 1 — RAW REWARD edition.

Run from the repo root::

    python scripts/plot_experiment_1.py

Outputs land in ``logs/experiment_1/figures/`` as PNG + PDF.

Produces two figures:
  1. ``learning_curves.{png,pdf}`` — 3 panels (one per env), 3 curves per panel
     (SAC / Gaussian / Categorical), mean across 3 seeds with a ±1 std band.
  2. ``final_performance.{png,pdf}`` — grouped bar chart of mean episode return
     (raw, from deterministic eval), error bars = std across seeds.

Reward handling
---------------
Exp 1 trained with ``gym.wrappers.NormalizeReward`` and only the normalized
reward was logged to ``metrics.jsonl``. That's what made the curves look
like flat noise — the normalization divisor (running std of the discounted
return) grows alongside the policy, so the ratio stays roughly constant.

To recover an interpretable RAW learning curve, we use the per-seed
deterministic eval values from ``logs/experiment_1/eval/eval_results.csv``
as a ground-truth anchor at the END of training. For each run we compute:

    scale = eval_mean_return / median(last 10% of normalized rewards)
    raw_curve ≈ scale × normalized_curve

This is exact at the endpoint and approximate during training (since the
true normalization divisor grew over time, but we apply a single constant
scale). It's an honest "best-effort" reconstruction — the curve's SHAPE
should be read with that caveat in mind. The y-axis label says so.

Going forward (Exp 2+), ``metrics.jsonl`` carries ``episode_reward_raw``
directly thanks to the ``RecordEpisodeStatistics`` wrapper added in
``environments/wrappers.py``. If that column is present, this script uses
it as-is and skips the eval-based rescaling entirely.
"""
from __future__ import annotations

import os
import re
import sys
import warnings
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
EVAL_DIR = LOG_ROOT / "eval"
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
RUN_RE = re.compile(r"^(?P<agent>sac|gaussian|categorical)_(?P<env>[\w\-]+)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$")

N_GRID = 200
SMOOTH_FRAC = 0.04
SMOOTH_MIN = 11
TAIL_FRAC = 0.10


# ---------------------------------------------------------------------------
# Eval-anchor loader
# ---------------------------------------------------------------------------
def load_eval_anchors(eval_dir: Path = EVAL_DIR) -> dict:
    """Return ``{(env, agent, seed): raw_mean_return}`` from eval_results.csv."""
    path = eval_dir / "eval_results.csv"
    if not path.exists():
        print(f"  [warn] No eval results at {path} — curves will fall back "
              f"to normalized-only and look like noise. Run "
              f"scripts/evaluate_experiment_1.py first.")
        return {}
    df = pd.read_csv(path)
    anchors = {}
    for _, row in df.iterrows():
        anchors[(row["env"], row["agent"], int(row["seed"]))] = float(row["mean_return"])
    return anchors


# ---------------------------------------------------------------------------
# Per-run loader (raw-reward-aware)
# ---------------------------------------------------------------------------
def load_run(jsonl_path: Path, anchor_raw: float | None) -> pd.DataFrame | None:
    """Load a single run's metrics.jsonl into a dataframe with columns
    ``['step', 'reward']`` where ``reward`` is RAW (best-effort).

    Resolution order:
      1. If the JSONL has an ``episode_reward_raw`` column (Exp 2 onwards),
         use it directly — no rescaling needed.
      2. Else, rescale ``episode_reward`` so that the median of its last
         ``TAIL_FRAC`` matches ``anchor_raw`` (from eval).
      3. Else, return the normalized column as-is and let the caller
         decide what to do.
    """
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError:
        return None
    if df.empty:
        return None

    if "episode_reward_raw" in df.columns:
        out = df.dropna(subset=["episode_reward_raw"])[["step", "episode_reward_raw"]]
        return out.rename(columns={"episode_reward_raw": "reward"}).reset_index(drop=True)

    if "episode_reward" not in df.columns:
        return None
    out = df.dropna(subset=["episode_reward"])[["step", "episode_reward"]]
    out = out.rename(columns={"episode_reward": "reward"}).reset_index(drop=True)
    if len(out) < 2:
        return None

    if anchor_raw is None:
        return out

    n_tail = max(5, int(len(out) * TAIL_FRAC))
    anchor_norm = float(np.median(out["reward"].iloc[-n_tail:]))
    if abs(anchor_norm) < 1e-8:
        anchor_norm = float(out["reward"].iloc[-n_tail:].mean())
        if abs(anchor_norm) < 1e-8:
            print(f"  [warn] anchor near zero in {jsonl_path.parent.name} — leaving normalized")
            return out
    scale = anchor_raw / anchor_norm
    out["reward"] = out["reward"] * scale
    return out


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
    out = df.copy().sort_values("step").reset_index(drop=True)
    win = max(SMOOTH_MIN, int(len(out) * SMOOTH_FRAC))
    out["reward"] = out["reward"].rolling(window=win, min_periods=1, center=True).mean()
    return out


def resample_to_grid(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, df["step"].to_numpy(), df["reward"].to_numpy(),
                     left=np.nan, right=np.nan)


def aggregate_seeds(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-seed curves and return (mean, std), ignoring NaNs.
    Suppresses 'Mean of empty slice' warnings from grid points past the
    shortest seed's last episode."""
    arr = np.vstack(curves)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom")
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ---------------------------------------------------------------------------
# Plot 1: learning curves
# ---------------------------------------------------------------------------
def plot_learning_curves(runs: pd.DataFrame, anchors: dict, out_dir: Path,
                         using_eval_anchors: bool):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)

    for ax, env in zip(axes, ENVS):
        env_runs = runs[runs["env"] == env]
        if env_runs.empty:
            ax.set_title(f"{env}\n(no data)")
            continue

        max_step = 0
        for _, r in env_runs.iterrows():
            anchor = anchors.get((r["env"], r["agent"], r["seed"]))
            df = load_run(r["path"], anchor)
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
                anchor = anchors.get((r["env"], r["agent"], r["seed"]))
                df = load_run(r["path"], anchor)
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
        ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    ylabel = ("Episode return (raw, calibrated to eval)"
              if using_eval_anchors else "Episode return (raw)")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
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
# Plot 2: final-performance bar chart (uses raw eval data directly)
# ---------------------------------------------------------------------------
def plot_final_performance(out_dir: Path):
    """Bar chart of mean deterministic-eval return per (env, agent)."""
    eval_path = EVAL_DIR / "eval_results.csv"
    if not eval_path.exists():
        print(f"  [skip] {eval_path} not found — run scripts/evaluate_experiment_1.py first")
        return
    df = pd.read_csv(eval_path)
    agg = (df.groupby(["env", "agent"])["mean_return"]
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
    ax.set_ylabel("Mean episode return (raw, deterministic eval)")
    ax.set_title("Experiment 1: final performance (mean ± std over 3 seeds)")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    fig.savefig(out_dir / "final_performance.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "final_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'final_performance.png'}")
    print(f"  wrote {out_dir / 'final_performance.pdf'}")

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

    print(f"\nLoading eval anchors from {EVAL_DIR} ...")
    anchors = load_eval_anchors(EVAL_DIR)
    using_eval_anchors = bool(anchors)
    if using_eval_anchors:
        print(f"  loaded {len(anchors)} per-seed raw anchors")
    else:
        print("  no eval anchors — learning curves will be normalized.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding learning curves (raw, calibrated to eval) ...")
    plot_learning_curves(runs, anchors, OUT_DIR, using_eval_anchors)

    print("\nBuilding final-performance bar chart (raw, from eval) ...")
    plot_final_performance(OUT_DIR)

    print("\nDone. Figures saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
