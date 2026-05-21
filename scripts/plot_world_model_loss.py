"""World-model training-loss plots for Experiment 3 (source) and
Experiment 4 (adaptation).

Run from the repo root::

    python scripts/plot_world_model_loss.py

Outputs:
  logs/experiment_3/figures/
    world_model_loss.{png,pdf}             — total WM loss vs env steps,
                                             Gaussian vs Categorical, mean ± std
    world_model_loss_components.{png,pdf}  — recon / KL / reward / continue
  logs/experiment_4_no_warmup/figures/
    world_model_loss.{png,pdf}             — total WM loss during adaptation
    world_model_loss_components.{png,pdf}  — component breakdown

Only the deliberative agents have a world model, so the model-free ``sac``
baseline is skipped. In Exp 4 only the **WM-unfrozen** runs keep training the
world model — the **frozen** runs log ``world_model_loss = 0`` with no
components, so they are excluded automatically.

Both metrics columns come straight from ``metrics.jsonl``:
  ``world_model_loss`` (total) and ``wm/{recon,kl,reward,continue}_loss``.
This script reuses the discovery, smoothing, resampling, aggregation, colours
and report styling (no titles, larger fonts) from ``plot_experiment_3.py`` so
the figures match the rest of the report.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# Make the sibling script importable whether we're launched from the repo root
# (python scripts/...) or elsewhere, then reuse its shared machinery + styling.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))            # scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # repo root

from plot_experiment_3 import (  # noqa: E402  (import after sys.path tweak)
    AGENT_COLORS, AGENT_LABELS,
    EXP3_ROOT, EXP4_ROOT, EXP3_OUT, EXP4_OUT,
    discover_exp3, discover_exp4,
    smoothed, resample, agg_seeds,
    N_GRID,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# Only the deliberative agents own a world model.
WM_AGENTS = ["gaussian", "categorical"]

# Component columns in metrics.jsonl and the title we give each panel.
WM_COMPONENTS = [
    ("wm/recon_loss", "Reconstruction loss"),
    ("wm/kl_loss", "KL loss"),
    ("wm/reward_loss", "Reward loss"),
    ("wm/continue_loss", "Continue loss"),
]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_run_metric(jsonl_path: Path, col: str):
    """Load (step, <col>) from a metrics.jsonl, returned as a DataFrame whose
    value column is renamed ``reward`` so the shared smoothed()/resample()
    helpers (which expect that name) work unchanged.

    Returns None when the column is absent (e.g. SAC, or a frozen WM has no
    ``wm/*`` keys) or when every value is exactly 0 (a frozen WM logs
    ``world_model_loss = 0`` — not a real curve)."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError:
        return None
    if df.empty or col not in df.columns:
        return None
    out = (df.dropna(subset=[col])
             [["step", col]]
             .rename(columns={col: "reward"})
             .reset_index(drop=True))
    if len(out) < 2:
        return None
    if (out["reward"] == 0).all():
        return None
    return out


# ---------------------------------------------------------------------------
# Generic curve builder
# ---------------------------------------------------------------------------
def _aggregate_agent(runs, agent: str, col: str):
    """Return (list_of_loaded_dfs, max_step, min_step) for one agent."""
    dfs, lo, hi = [], np.inf, 0
    for _, r in runs[runs["agent"] == agent].iterrows():
        df = load_run_metric(r["path"], col)
        if df is None:
            continue
        dfs.append(df)
        lo = min(lo, df["step"].min())
        hi = max(hi, df["step"].max())
    if not dfs:
        lo = 0
    return dfs, hi, lo


def plot_total_loss(runs, out_dir: Path, xlabel: str, x_div: float,
                    label_suffix: str = ""):
    """Total world-model loss vs steps, one curve per WM agent (mean ± std)."""
    if runs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    per_agent = {a: _aggregate_agent(runs, a, "world_model_loss") for a in WM_AGENTS}
    hi = max((h for _, h, _ in per_agent.values()), default=0)
    lo = min((l for dfs, _, l in per_agent.values() if dfs), default=0)
    if hi == 0:
        print(f"  [skip] no usable WM-loss curves for {out_dir}")
        plt.close(fig)
        return
    grid = np.linspace(lo, hi, N_GRID)

    for agent in WM_AGENTS:
        dfs, _, _ = per_agent[agent]
        seed_curves = [resample(smoothed(df), grid) for df in dfs]
        if not seed_curves:
            continue
        mean, std = agg_seeds(seed_curves)
        color = AGENT_COLORS[agent]
        ax.plot(grid, mean, color=color, linewidth=2,
                label=f"{AGENT_LABELS[agent]}{label_suffix}  (n={len(seed_curves)})")
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("World-model loss (total)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / x_div:.2f}"))
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "world_model_loss.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "world_model_loss.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'world_model_loss.png'}")


def plot_loss_components(runs, out_dir: Path, xlabel: str, x_div: float,
                         label_suffix: str = ""):
    """2x2 grid: each WM loss component vs steps, per agent (mean ± std)."""
    if runs.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    drew_any = False

    for ax, (col, title) in zip(axes.flat, WM_COMPONENTS):
        per_agent = {a: _aggregate_agent(runs, a, col) for a in WM_AGENTS}
        hi = max((h for _, h, _ in per_agent.values()), default=0)
        lo = min((l for dfs, _, l in per_agent.values() if dfs), default=0)
        if hi == 0:
            ax.set_visible(False)
            continue
        grid = np.linspace(lo, hi, N_GRID)
        for agent in WM_AGENTS:
            dfs, _, _ = per_agent[agent]
            seed_curves = [resample(smoothed(df), grid) for df in dfs]
            if not seed_curves:
                continue
            mean, std = agg_seeds(seed_curves)
            color = AGENT_COLORS[agent]
            ax.plot(grid, mean, color=color, linewidth=2,
                    label=f"{AGENT_LABELS[agent]}{label_suffix}")
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)
            drew_any = True
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / x_div:.2f}"))

    if not drew_any:
        print(f"  [skip] no usable WM-loss components for {out_dir}")
        plt.close(fig)
        return

    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel(xlabel)
    # Single shared legend (top-left panel has the handles we need).
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center",
                   ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "world_model_loss_components.png", dpi=200,
                bbox_inches="tight")
    fig.savefig(out_dir / "world_model_loss_components.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'world_model_loss_components.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Discovering Exp 3 runs in {EXP3_ROOT} ...")
    e3 = discover_exp3()
    e3 = e3[e3["agent"].isin(WM_AGENTS)] if not e3.empty else e3
    print(f"  found {len(e3)} world-model runs")

    print(f"\nDiscovering Exp 4 runs in {EXP4_ROOT} ...")
    e4 = discover_exp4()
    if not e4.empty:
        # Only WM agents, and only the unfrozen variant logs a real loss.
        e4 = e4[(e4["agent"].isin(WM_AGENTS)) & (e4["wm"] == "unfrozen")]
    print(f"  found {len(e4)} unfrozen world-model runs")

    if not e3.empty:
        print("\nBuilding Exp 3 world-model loss figures ...")
        plot_total_loss(e3, EXP3_OUT, r"Environment steps  ($\times 10^6$)", 1e6)
        plot_loss_components(e3, EXP3_OUT, r"Environment steps  ($\times 10^6$)", 1e6)
    else:
        print("\nNo Exp 3 world-model data — skipping.")

    if not e4.empty:
        print("\nBuilding Exp 4 world-model loss figures ...")
        plot_total_loss(e4, EXP4_OUT, r"Adaptation steps  ($\times 10^5$)", 1e5,
                        label_suffix=" — WM unfrozen")
        plot_loss_components(e4, EXP4_OUT, r"Adaptation steps  ($\times 10^5$)", 1e5,
                             label_suffix=" — WM unfrozen")
    else:
        print("\nNo Exp 4 unfrozen world-model data — skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
