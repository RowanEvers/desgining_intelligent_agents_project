"""Plots for Experiment 3 (main) + Experiment 4 (adaptation).

Run from the repo root::

    python scripts/plot_experiment_3.py

Outputs:
  logs/experiment_3/figures/
    learning_curves.{png,pdf}      — 3 curves (SAC vs Gaussian vs Categorical),
                                     mean ± std over 5 seeds
    final_performance.{png,pdf}    — bar chart of mean of last 10% raw return
  logs/experiment_4/figures/
    adaptation_curves.{png,pdf}    — adaptation under gravity×1.5
    adaptation_vs_baseline.{png,pdf} — final adaptation reward vs Exp 3 baseline

Reads ``episode_reward_raw`` directly from metrics.jsonl. No eval-anchoring
is needed for Exp 3 / Exp 4 because the new wrapper stack logs raw rewards.
If a run for some reason has no raw column, that curve is skipped with a
warning rather than silently plotting noise.
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

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# ---------------------------------------------------------------------------
# Config (mirrors experiment_3.py / experiment_4.py)
# ---------------------------------------------------------------------------
ENV_ID = "HalfCheetah-v5"
AGENTS = ["sac", "gaussian", "categorical"]
AGENT_LABELS = {
    "sac": "SAC (model-free baseline)",
    "gaussian": "Latent-SAC (Gaussian WM)",
    "categorical": "Latent-SAC (Categorical WM)",
}
AGENT_COLORS = {
    "sac": "#444444",
    "gaussian": "#1f77b4",
    "categorical": "#d62728",
}

EXP3_ROOT = Path("logs/experiment_3")
EXP4_ROOT = Path("logs/experiment_4")
EXP3_OUT = EXP3_ROOT / "figures"
EXP4_OUT = EXP4_ROOT / "figures"

EXP3_RUN_RE = re.compile(
    r"^(?P<agent>sac|gaussian|categorical)_HalfCheetah-v5_steps\d+_seed(?P<seed>\d+)$"
)
EXP4_RUN_RE = re.compile(
    r"^(?P<agent>sac|gaussian|categorical)_HalfCheetah-v5_seed(?P<seed>\d+)_"
    r"(?P<kind>gravity|friction)x(?P<scale>[\d.]+)(?:_wm(?P<wm>frozen|unfrozen))?$"
)

N_GRID = 200
SMOOTH_FRAC = 0.04
SMOOTH_MIN = 11
TAIL_FRAC = 0.10
# How much of the Exp 3 source tail to show before the perturbation in the
# source->adaptation continuity plot (matches the adaptation length so the
# pre/post scales are comparable).
SRC_TAIL = 100_000


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_run_raw(jsonl_path: Path) -> pd.DataFrame | None:
    """Load (step, raw_reward) from a metrics.jsonl. Skips runs that only
    have normalized reward, with a printed warning."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError:
        return None
    if df.empty:
        return None
    if "episode_reward_raw" not in df.columns:
        print(f"  [warn] no episode_reward_raw in {jsonl_path.parent.name} — skipped")
        return None
    out = (df.dropna(subset=["episode_reward_raw"])
              [["step", "episode_reward_raw"]]
              .rename(columns={"episode_reward_raw": "reward"})
              .reset_index(drop=True))
    return out if len(out) >= 2 else None


def discover_exp3(root: Path = EXP3_ROOT) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "figures":
            continue
        m = EXP3_RUN_RE.match(d.name)
        if not m:
            continue
        jsonl = d / "metrics.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            continue
        rows.append({"agent": m["agent"], "seed": int(m["seed"]), "path": jsonl})
    return pd.DataFrame(rows)


def discover_exp4(root: Path = EXP4_ROOT) -> pd.DataFrame:
    if not root.exists():
        return pd.DataFrame()
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "figures":
            continue
        m = EXP4_RUN_RE.match(d.name)
        if not m:
            continue
        jsonl = d / "metrics.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            continue
        rows.append({
            "agent": m["agent"], "seed": int(m["seed"]),
            "kind": m["kind"], "scale": float(m["scale"]),
            "wm": m["wm"] or "na",
            "path": jsonl,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def smoothed(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values("step").reset_index(drop=True)
    win = max(SMOOTH_MIN, int(len(out) * SMOOTH_FRAC))
    out["reward"] = out["reward"].rolling(window=win, min_periods=1, center=True).mean()
    return out


def resample(df: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    return np.interp(grid, df["step"].to_numpy(), df["reward"].to_numpy(),
                     left=np.nan, right=np.nan)


def agg_seeds(curves: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.vstack(curves)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom")
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


# ---------------------------------------------------------------------------
# Exp 3 plots
# ---------------------------------------------------------------------------
def plot_exp3_learning_curves(runs: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Build common grid from the longest run.
    max_step = 0
    for _, r in runs.iterrows():
        df = load_run_raw(r["path"])
        if df is not None:
            max_step = max(max_step, df["step"].max())
    if max_step == 0:
        print("  [skip] no usable curves")
        return
    grid = np.linspace(0, max_step, N_GRID)

    for agent in AGENTS:
        agent_runs = runs[runs["agent"] == agent]
        seed_curves = []
        for _, r in agent_runs.iterrows():
            df = load_run_raw(r["path"])
            if df is None:
                continue
            df = smoothed(df)
            seed_curves.append(resample(df, grid))
        if not seed_curves:
            continue
        mean, std = agg_seeds(seed_curves)
        color = AGENT_COLORS[agent]
        ax.plot(grid, mean, label=f"{AGENT_LABELS[agent]}  (n={len(seed_curves)})",
                color=color, linewidth=2)
        ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Episode return (raw)")
    ax.set_title(f"Experiment 3: learning curves on {ENV_ID} (mean ± 1 std over seeds)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "learning_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "learning_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'learning_curves.png'}")


def plot_exp3_final_performance(runs: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    """Per-seed mean of last TAIL_FRAC episodes; aggregated across seeds."""
    records = []
    for _, r in runs.iterrows():
        df = load_run_raw(r["path"])
        if df is None:
            continue
        n_tail = max(1, int(len(df) * TAIL_FRAC))
        records.append({
            "agent": r["agent"], "seed": r["seed"],
            "final_reward": float(df["reward"].iloc[-n_tail:].mean()),
        })
    finals = pd.DataFrame(records)
    if finals.empty:
        return finals

    agg = (finals.groupby("agent")["final_reward"]
                  .agg(["mean", "std", "count"])
                  .reset_index())

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(AGENTS))
    sub = agg.set_index("agent").reindex(AGENTS)
    means = sub["mean"].to_numpy()
    stds = sub["std"].fillna(0).to_numpy()
    colors = [AGENT_COLORS[a] for a in AGENTS]
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="black", linewidth=0.5)

    # Add per-seed scatter so the actual distribution is visible.
    for i, agent in enumerate(AGENTS):
        seed_vals = finals[finals["agent"] == agent]["final_reward"].to_numpy()
        ax.scatter([i] * len(seed_vals), seed_vals,
                   color="black", zorder=3, s=20, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS], rotation=15, ha="right")
    ax.set_ylabel("Mean raw return over final 10% of episodes")
    ax.set_title(f"Experiment 3: final performance on {ENV_ID}")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()

    fig.savefig(out_dir / "final_performance.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "final_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'final_performance.png'}")
    agg.to_csv(out_dir / "final_performance.csv", index=False)
    print(f"  wrote {out_dir / 'final_performance.csv'}")
    return finals


# ---------------------------------------------------------------------------
# Exp 4 plots
# ---------------------------------------------------------------------------
def plot_exp4_adaptation_curves(runs: pd.DataFrame, out_dir: Path):
    if runs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    max_step = 0
    for _, r in runs.iterrows():
        df = load_run_raw(r["path"])
        if df is not None:
            max_step = max(max_step, df["step"].max())
    if max_step == 0:
        print("  [skip] no usable curves")
        return
    grid = np.linspace(0, max_step, N_GRID)

    # Each deliberative agent ran TWO WM modes (frozen / unfrozen); SAC ran
    # one ("na"). Group by (agent, wm) so the modes are separate curves and
    # the label reflects the actual data instead of a hardcoded string.
    for agent in AGENTS:
        if agent == "sac":
            variants = [("na", AGENT_LABELS[agent], "-")]
        else:
            variants = [
                ("frozen", f"{AGENT_LABELS[agent]} (WM frozen)", "-"),
                ("unfrozen", f"{AGENT_LABELS[agent]} (WM unfrozen)", "--"),
            ]
        for wm_mode, label, ls in variants:
            agent_runs = runs[(runs["agent"] == agent) & (runs["wm"] == wm_mode)]
            seed_curves = []
            for _, r in agent_runs.iterrows():
                df = load_run_raw(r["path"])
                if df is None:
                    continue
                df = smoothed(df)
                seed_curves.append(resample(df, grid))
            if not seed_curves:
                continue
            mean, std = agg_seeds(seed_curves)
            color = AGENT_COLORS[agent]
            ax.plot(grid, mean, label=label, color=color, linewidth=2, linestyle=ls)
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.18)

    ax.set_xlabel("Adaptation steps")
    ax.set_ylabel("Episode return (raw)")
    ax.set_title(f"Experiment 4: adaptation to gravity×1.5 on {ENV_ID}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "adaptation_curves.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "adaptation_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'adaptation_curves.png'}")


def plot_exp4_source_to_adapt(exp3_runs: pd.DataFrame, exp4_runs: pd.DataFrame,
                              out_dir: Path):
    """Continuity plot: the last SRC_TAIL steps of Exp 3 source training joined
    to the Exp 4 adaptation, with the perturbation at x=0. Gives context to the
    post-perturbation drop by showing each agent's pre-perturbation level.

    Source steps are shifted so each run ends at x=0 (→ [-SRC_TAIL, 0]);
    adaptation keeps its own [0, adapt_len] axis. An agent's frozen/unfrozen
    branches share the same source history, so the source tail is drawn once
    per agent (thin, no legend entry) and the adaptation branches carry the
    labels. The dotted line at x=0 marks where gravity×1.5 is applied.
    """
    if exp4_runs.empty or exp3_runs.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))

    adapt_max = 0
    for _, r in exp4_runs.iterrows():
        df = load_run_raw(r["path"])
        if df is not None:
            adapt_max = max(adapt_max, df["step"].max())
    if adapt_max == 0:
        print("  [skip] no usable adaptation curves")
        plt.close(fig)
        return

    src_grid = np.linspace(-SRC_TAIL, 0, N_GRID)
    adapt_grid = np.linspace(0, adapt_max, N_GRID)

    # ---- source tail, once per agent (shared by both WM modes) ----
    for agent in AGENTS:
        seed_curves = []
        for _, r in exp3_runs[exp3_runs["agent"] == agent].iterrows():
            df = load_run_raw(r["path"])
            if df is None:
                continue
            src_end = df["step"].max()
            df = df[df["step"] >= src_end - SRC_TAIL].copy()
            if len(df) < 2:
                continue
            df["step"] = df["step"] - src_end           # → [-SRC_TAIL, 0]
            seed_curves.append(resample(smoothed(df), src_grid))
        if not seed_curves:
            continue
        mean, std = agg_seeds(seed_curves)
        color = AGENT_COLORS[agent]
        ax.plot(src_grid, mean, color=color, linewidth=1.5, alpha=0.7)
        ax.fill_between(src_grid, mean - std, mean + std, color=color, alpha=0.10)

    # ---- adaptation, per (agent, wm) ----
    for agent in AGENTS:
        if agent == "sac":
            variants = [("na", AGENT_LABELS[agent], "-")]
        else:
            variants = [
                ("frozen", f"{AGENT_LABELS[agent]} (WM frozen)", "-"),
                ("unfrozen", f"{AGENT_LABELS[agent]} (WM unfrozen)", "--"),
            ]
        for wm_mode, label, ls in variants:
            sel = (exp4_runs["agent"] == agent) & (exp4_runs["wm"] == wm_mode)
            seed_curves = []
            for _, r in exp4_runs[sel].iterrows():
                df = load_run_raw(r["path"])
                if df is None:
                    continue
                seed_curves.append(resample(smoothed(df), adapt_grid))
            if not seed_curves:
                continue
            mean, std = agg_seeds(seed_curves)
            color = AGENT_COLORS[agent]
            ax.plot(adapt_grid, mean, label=label, color=color,
                    linewidth=2, linestyle=ls)
            ax.fill_between(adapt_grid, mean - std, mean + std,
                            color=color, alpha=0.18)

    ax.axvline(0.0, color="0.3", linestyle=":", linewidth=1.5)
    # Vertical label just left of the line so it doesn't collide with the title.
    ax.text(0.0, 0.97, "gravity×1.5 applied  ",
            transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="right", fontsize=8, color="0.3")
    ax.set_xlabel("Steps relative to perturbation  (source ← 0 → adaptation)")
    ax.set_ylabel("Episode return (raw)")
    ax.set_title(f"Experiment 4: source → adaptation around gravity×1.5 on {ENV_ID}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "source_to_adaptation.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "source_to_adaptation.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'source_to_adaptation.png'}")


def plot_exp4_vs_baseline(runs: pd.DataFrame, exp3_finals: pd.DataFrame,
                          out_dir: Path):
    """Side-by-side bars: Exp 3 final (unperturbed) vs Exp 4 final (perturbed)."""
    if runs.empty or exp3_finals.empty:
        return

    # Per-seed final on Exp 4
    e4_records = []
    for _, r in runs.iterrows():
        df = load_run_raw(r["path"])
        if df is None:
            continue
        n_tail = max(1, int(len(df) * TAIL_FRAC))
        e4_records.append({
            "agent": r["agent"], "seed": r["seed"],
            "post_adapt": float(df["reward"].iloc[-n_tail:].mean()),
        })
    e4 = pd.DataFrame(e4_records)
    if e4.empty:
        return

    # Merge with Exp 3 per-seed finals (same (agent, seed) keys).
    merged = e4.merge(
        exp3_finals.rename(columns={"final_reward": "baseline"}),
        on=["agent", "seed"], how="left",
    )
    agg = (merged.groupby("agent")[["baseline", "post_adapt"]]
                  .agg(["mean", "std"])
                  .reset_index())

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(AGENTS))
    bar_w = 0.35

    for i, agent in enumerate(AGENTS):
        row = agg[agg["agent"] == agent]
        if row.empty:
            continue
        b_mean = row[("baseline", "mean")].values[0]
        b_std = row[("baseline", "std")].fillna(0).values[0]
        p_mean = row[("post_adapt", "mean")].values[0]
        p_std = row[("post_adapt", "std")].fillna(0).values[0]

        ax.bar(i - bar_w / 2, b_mean, yerr=b_std, width=bar_w, capsize=4,
               color=AGENT_COLORS[agent], edgecolor="black", linewidth=0.5,
               label="Baseline (Exp 3, unperturbed)" if i == 0 else None,
               alpha=0.9)
        ax.bar(i + bar_w / 2, p_mean, yerr=p_std, width=bar_w, capsize=4,
               color=AGENT_COLORS[agent], edgecolor="black", linewidth=0.5,
               hatch="///",
               label="Post-adapt (Exp 4, gravity×1.5)" if i == 0 else None,
               alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([AGENT_LABELS[a] for a in AGENTS], rotation=15, ha="right")
    ax.set_ylabel("Mean raw return over final 10% of episodes")
    ax.set_title(f"Experiment 4: adaptation drop vs Exp 3 baseline")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()

    fig.savefig(out_dir / "adaptation_vs_baseline.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "adaptation_vs_baseline.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'adaptation_vs_baseline.png'}")
    agg.to_csv(out_dir / "adaptation_vs_baseline.csv")
    print(f"  wrote {out_dir / 'adaptation_vs_baseline.csv'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Discovering Exp 3 runs in {EXP3_ROOT} ...")
    e3 = discover_exp3()
    print(f"  found {len(e3)} runs")
    if not e3.empty:
        print(e3.groupby(["agent"]).size())

    print(f"\nDiscovering Exp 4 runs in {EXP4_ROOT} ...")
    e4 = discover_exp4()
    print(f"  found {len(e4)} runs")
    if not e4.empty:
        print(e4.groupby(["agent"]).size())

    if not e3.empty:
        EXP3_OUT.mkdir(parents=True, exist_ok=True)
        print("\nBuilding Exp 3 learning curves ...")
        plot_exp3_learning_curves(e3, EXP3_OUT)

        print("\nBuilding Exp 3 final-performance bar chart ...")
        exp3_finals = plot_exp3_final_performance(e3, EXP3_OUT)
    else:
        print("\nNo Exp 3 data yet — skipping Exp 3 figures.")
        exp3_finals = pd.DataFrame()

    if not e4.empty:
        EXP4_OUT.mkdir(parents=True, exist_ok=True)
        print("\nBuilding Exp 4 adaptation curves ...")
        plot_exp4_adaptation_curves(e4, EXP4_OUT)

        if not e3.empty:
            print("\nBuilding Exp 4 source → adaptation continuity plot ...")
            plot_exp4_source_to_adapt(e3, e4, EXP4_OUT)

        if not exp3_finals.empty:
            print("\nBuilding Exp 4 vs baseline comparison ...")
            plot_exp4_vs_baseline(e4, exp3_finals, EXP4_OUT)
    else:
        print("\nNo Exp 4 data yet — skipping Exp 4 figures.")

    print("\nDone.")


if __name__ == "__main__":
    main()
