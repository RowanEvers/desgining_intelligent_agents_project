"""Publication-quality figures for experiment 2 (adaptation).

Run from the repo root::

    python scripts/plot_experiment_2.py

Outputs land in ``logs/experiment_2/figures/`` as PNG + PDF.

Produces three figures:
  1. ``adaptation_curves_{env}.{png,pdf}`` — one per env. Subplots for each
     perturbation, each subplot overlays the 5 agent variants
     (sac, gaussian-frozen, gaussian-unfrozen, categorical-frozen,
     categorical-unfrozen). Mean ± 1 std across 3 seeds.
  2. ``final_performance.{png,pdf}`` — grouped bar chart per env, bars
     colored by (agent, wm_mode), x-axis = perturbation.
  3. ``adaptation_gap.{png,pdf}`` — gap to Exp 1 baseline. Shows how much
     performance was retained vs the un-perturbed final eval, per
     (env, agent, wm_mode, perturbation). Pulls the baseline from
     ``logs/experiment_1/eval/eval_results.csv``.

Exp 2 logs ``episode_reward_raw`` natively (RecordEpisodeStatistics is in
the wrapper stack), so this script reads raw rewards directly — no
eval-anchored rescaling like ``plot_experiment_1.py`` needed.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ---------------------------------------------------------------------------
# Config (must match scripts/experiment_2.py)
# ---------------------------------------------------------------------------
LOG_ROOT = Path("logs/experiment_2")
OUT_DIR = LOG_ROOT / "figures"
EXP1_EVAL = Path("logs/experiment_1/eval/eval_results.csv")

ENVS = ["Pendulum-v1", "Hopper-v5", "Walker2d-v5"]
PERTURBATIONS = [
    ("gravity",  0.5),
    ("gravity",  1.5),
    ("gravity",  2.0),
    ("friction", 0.5),
    ("friction", 2.0),
]

# Variant = (agent, wm_mode). 5 variants per (env, perturbation, seed).
VARIANTS = [
    ("sac",         "na"),
    ("gaussian",    "frozen"),
    ("gaussian",    "unfrozen"),
    ("categorical", "frozen"),
    ("categorical", "unfrozen"),
]
VARIANT_LABELS = {
    ("sac",         "na"):       "SAC (baseline)",
    ("gaussian",    "frozen"):   "Gaussian WM (frozen)",
    ("gaussian",    "unfrozen"): "Gaussian WM (unfrozen)",
    ("categorical", "frozen"):   "Categorical WM (frozen)",
    ("categorical", "unfrozen"): "Categorical WM (unfrozen)",
}
VARIANT_COLORS = {
    ("sac",         "na"):       "#444444",
    ("gaussian",    "frozen"):   "#1f77b4",
    ("gaussian",    "unfrozen"): "#7fb3d5",
    ("categorical", "frozen"):   "#d62728",
    ("categorical", "unfrozen"): "#f1948a",
}
# Frozen = solid line, unfrozen = dashed, sac baseline = dotted.
VARIANT_STYLE = {
    ("sac",         "na"):       (0, (1, 1)),
    ("gaussian",    "frozen"):   "-",
    ("gaussian",    "unfrozen"): "--",
    ("categorical", "frozen"):   "-",
    ("categorical", "unfrozen"): "--",
}

# Run-tag regexes — match scripts/experiment_2.py:run_tag().
SAC_RE = re.compile(
    r"^sac_(?P<env>[\w\-]+)_seed(?P<seed>\d+)_"
    r"(?P<kind>gravity|friction)x(?P<scale>[\d.]+)$"
)
WM_RE = re.compile(
    r"^(?P<agent>gaussian|categorical)_(?P<env>[\w\-]+)_seed(?P<seed>\d+)_"
    r"(?P<kind>gravity|friction)x(?P<scale>[\d.]+)_wm(?P<wm>frozen|unfrozen)$"
)

N_GRID = 200
SMOOTH_FRAC = 0.04
SMOOTH_MIN = 11
TAIL_FRAC = 0.10   # used for final-performance aggregation


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def parse_run_dir(name: str) -> dict | None:
    """Return ``{agent, env, seed, kind, scale, wm}`` or None."""
    m = WM_RE.match(name)
    if m:
        return {
            "agent": m["agent"], "env": m["env"], "seed": int(m["seed"]),
            "kind": m["kind"], "scale": float(m["scale"]), "wm": m["wm"],
        }
    m = SAC_RE.match(name)
    if m:
        return {
            "agent": "sac", "env": m["env"], "seed": int(m["seed"]),
            "kind": m["kind"], "scale": float(m["scale"]), "wm": "na",
        }
    return None


def discover_runs(root: Path) -> pd.DataFrame:
    rows = []
    for d in sorted(root.iterdir()):
        if not d.is_dir() or d.name == "figures":
            continue
        info = parse_run_dir(d.name)
        if info is None:
            print(f"  [skip] unrecognised dir: {d.name}")
            continue
        jsonl = d / "metrics.jsonl"
        if not jsonl.exists() or jsonl.stat().st_size == 0:
            print(f"  [skip] no metrics: {d.name}")
            continue
        info["path"] = jsonl
        rows.append(info)
    return pd.DataFrame(rows)


def load_run(jsonl_path: Path) -> pd.DataFrame | None:
    """Load a run as ['step', 'reward']. Prefer raw, fall back to normalized
    with a printed warning (shouldn't happen for Exp 2 if the wrapper stack
    is correct)."""
    try:
        df = pd.read_json(jsonl_path, lines=True)
    except ValueError:
        return None
    if df.empty:
        return None
    if "episode_reward_raw" in df.columns:
        out = (df.dropna(subset=["episode_reward_raw"])
                  [["step", "episode_reward_raw"]]
                  .rename(columns={"episode_reward_raw": "reward"})
                  .reset_index(drop=True))
        return out if len(out) >= 2 else None
    if "episode_reward" in df.columns:
        print(f"  [warn] {jsonl_path.parent.name} has no episode_reward_raw — "
              f"falling back to normalized (this curve will look like noise)")
        out = (df.dropna(subset=["episode_reward"])
                  [["step", "episode_reward"]]
                  .rename(columns={"episode_reward": "reward"})
                  .reset_index(drop=True))
        return out if len(out) >= 2 else None
    return None


# ---------------------------------------------------------------------------
# Aggregation helpers (mirror plot_experiment_1.py)
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
    Suppresses noisy 'Mean of empty slice' warnings from grid points past
    the shortest seed's max step."""
    import warnings
    arr = np.vstack(curves)
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        warnings.filterwarnings("ignore", message="Degrees of freedom")
        return np.nanmean(arr, axis=0), np.nanstd(arr, axis=0)


def pert_tag(kind: str, scale: float) -> str:
    return f"{kind}×{scale}"   # e.g. "gravity×0.5" (× = unicode multiply)


# ---------------------------------------------------------------------------
# Plot 1: per-env adaptation curve grids
# ---------------------------------------------------------------------------
def plot_adaptation_curves(runs: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for env in ENVS:
        env_runs = runs[runs["env"] == env]
        if env_runs.empty:
            continue

        # Which perturbations actually exist for this env? Pendulum drops friction.
        env_perts = [(k, s) for k, s in PERTURBATIONS
                     if not (env == "Pendulum-v1" and k == "friction")]
        n_cols = len(env_perts)
        fig, axes = plt.subplots(1, n_cols, figsize=(3.2 * n_cols, 4.2), sharey=True)
        if n_cols == 1:
            axes = [axes]

        for ax, (kind, scale) in zip(axes, env_perts):
            pert_runs = env_runs[(env_runs["kind"] == kind) & (env_runs["scale"] == scale)]

            # Determine the common step grid from the longest run in this cell.
            max_step = 0
            for _, r in pert_runs.iterrows():
                df = load_run(r["path"])
                if df is not None:
                    max_step = max(max_step, df["step"].max())
            if max_step == 0:
                ax.set_title(f"{pert_tag(kind, scale)}\n(no data)")
                continue
            grid = np.linspace(0, max_step, N_GRID)

            for variant in VARIANTS:
                agent, wm = variant
                var_runs = pert_runs[(pert_runs["agent"] == agent) & (pert_runs["wm"] == wm)]
                seed_curves = []
                for _, r in var_runs.iterrows():
                    df = load_run(r["path"])
                    if df is None or len(df) < 2:
                        continue
                    df = smoothed_curve(df)
                    seed_curves.append(resample_to_grid(df, grid))
                if not seed_curves:
                    continue
                mean, std = aggregate_seeds(seed_curves)
                color = VARIANT_COLORS[variant]
                style = VARIANT_STYLE[variant]
                ax.plot(grid, mean, label=VARIANT_LABELS[variant],
                        color=color, linewidth=1.8, linestyle=style)
                ax.fill_between(grid, mean - std, mean + std,
                                color=color, alpha=0.12)

            ax.set_title(pert_tag(kind, scale), fontsize=11)
            ax.set_xlabel("Environment steps")
            ax.grid(True, linestyle="--", alpha=0.4)
            ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

        axes[0].set_ylabel("Episode return (raw)")
        handles, labels = axes[0].get_legend_handles_labels()
        for ax in axes[1:]:
            h, l = ax.get_legend_handles_labels()
            if len(h) > len(handles):
                handles, labels = h, l
        fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
                   bbox_to_anchor=(0.5, -0.08), frameon=False)
        fig.suptitle(f"Experiment 2: adaptation on {env} (mean ± 1 std over 3 seeds)",
                     fontsize=12, y=1.02)
        fig.tight_layout()

        out_png = out_dir / f"adaptation_curves_{env}.png"
        out_pdf = out_dir / f"adaptation_curves_{env}.pdf"
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")
        print(f"  wrote {out_pdf}")


# ---------------------------------------------------------------------------
# Plot 2: final performance bar chart (post-adaptation)
# ---------------------------------------------------------------------------
def compute_final_performance(runs: pd.DataFrame) -> pd.DataFrame:
    """Per (env, agent, wm, perturbation, seed): mean raw reward over the
    last TAIL_FRAC of training episodes. Aggregated downstream."""
    records = []
    for _, r in runs.iterrows():
        df = load_run(r["path"])
        if df is None or len(df) < 2:
            continue
        n_tail = max(1, int(len(df) * TAIL_FRAC))
        records.append({
            "env": r["env"], "agent": r["agent"], "wm": r["wm"],
            "kind": r["kind"], "scale": r["scale"], "seed": r["seed"],
            "final_reward": float(df["reward"].iloc[-n_tail:].mean()),
        })
    return pd.DataFrame(records)


def plot_final_performance(runs: pd.DataFrame, out_dir: Path):
    finals = compute_final_performance(runs)
    if finals.empty:
        print("  [skip] no data for final-performance plot")
        return

    agg = (finals.groupby(["env", "agent", "wm", "kind", "scale"])["final_reward"]
                  .agg(["mean", "std"])
                  .reset_index())
    agg["pert"] = agg.apply(lambda r: pert_tag(r["kind"], r["scale"]), axis=1)

    fig, axes = plt.subplots(1, len(ENVS), figsize=(5 * len(ENVS), 4.5),
                             sharey=False)
    for ax, env in zip(axes, ENVS):
        sub = agg[agg["env"] == env]
        if sub.empty:
            ax.set_title(f"{env}\n(no data)")
            continue

        perts = sorted(sub["pert"].unique())
        x = np.arange(len(perts))
        n_var = len(VARIANTS)
        bar_w = 0.8 / n_var

        for i, variant in enumerate(VARIANTS):
            agent, wm = variant
            row = sub[(sub["agent"] == agent) & (sub["wm"] == wm)] \
                    .set_index("pert").reindex(perts)
            means = row["mean"].to_numpy()
            stds = row["std"].fillna(0).to_numpy()
            offset = (i - (n_var - 1) / 2) * bar_w
            ax.bar(x + offset, means, width=bar_w, yerr=stds, capsize=3,
                   label=VARIANT_LABELS[variant],
                   color=VARIANT_COLORS[variant],
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(perts, rotation=20, ha="right")
        ax.set_title(env)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Mean raw return over final 10% of adaptation episodes")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Experiment 2: final adaptation performance "
                 "(mean ± std across seeds)", fontsize=12, y=1.02)
    fig.tight_layout()

    fig.savefig(out_dir / "final_performance.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "final_performance.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'final_performance.png'}")
    print(f"  wrote {out_dir / 'final_performance.pdf'}")

    agg.to_csv(out_dir / "final_performance.csv", index=False)
    print(f"  wrote {out_dir / 'final_performance.csv'}")
    return agg


# ---------------------------------------------------------------------------
# Plot 3: adaptation gap vs Exp 1 baseline
# ---------------------------------------------------------------------------
def plot_adaptation_gap(agg: pd.DataFrame, out_dir: Path):
    """Performance retained relative to un-perturbed final eval (Exp 1).

    For each (env, agent, perturbation), compute:
        gap = post_adapt_mean - exp1_eval_mean
    Negative gap = the agent ended worse than its Exp 1 final eval despite
    matched compute. Positive gap = adaptation actually helped (unlikely
    for hard perturbations).
    """
    if agg is None or agg.empty:
        return
    if not EXP1_EVAL.exists():
        print(f"  [skip] adaptation-gap: {EXP1_EVAL} not found")
        return

    exp1 = pd.read_csv(EXP1_EVAL)
    # Mean across seeds of the per-seed mean_return → one number per (env, agent).
    baseline = (exp1.groupby(["env", "agent"])["mean_return"]
                     .mean().reset_index()
                     .rename(columns={"mean_return": "exp1_mean"}))
    merged = agg.merge(baseline, on=["env", "agent"], how="left")
    merged["gap"] = merged["mean"] - merged["exp1_mean"]

    fig, axes = plt.subplots(1, len(ENVS), figsize=(5 * len(ENVS), 4.5),
                             sharey=False)
    for ax, env in zip(axes, ENVS):
        sub = merged[merged["env"] == env]
        if sub.empty:
            ax.set_title(f"{env}\n(no data)")
            continue

        perts = sorted(sub["pert"].unique())
        x = np.arange(len(perts))
        n_var = len(VARIANTS)
        bar_w = 0.8 / n_var

        for i, variant in enumerate(VARIANTS):
            agent, wm = variant
            row = sub[(sub["agent"] == agent) & (sub["wm"] == wm)] \
                    .set_index("pert").reindex(perts)
            gaps = row["gap"].to_numpy()
            offset = (i - (n_var - 1) / 2) * bar_w
            ax.bar(x + offset, gaps, width=bar_w,
                   label=VARIANT_LABELS[variant],
                   color=VARIANT_COLORS[variant],
                   edgecolor="black", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(perts, rotation=20, ha="right")
        ax.set_title(env)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    axes[0].set_ylabel("Post-adapt mean return − Exp 1 baseline return\n(positive = adapted past baseline)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(len(handles), 5),
               bbox_to_anchor=(0.5, -0.05), frameon=False)
    fig.suptitle("Experiment 2: adaptation gap vs un-perturbed baseline",
                 fontsize=12, y=1.02)
    fig.tight_layout()

    fig.savefig(out_dir / "adaptation_gap.png", dpi=200, bbox_inches="tight")
    fig.savefig(out_dir / "adaptation_gap.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_dir / 'adaptation_gap.png'}")
    print(f"  wrote {out_dir / 'adaptation_gap.pdf'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not LOG_ROOT.exists():
        raise SystemExit(f"Log root not found: {LOG_ROOT.resolve()} — "
                         f"run scripts/experiment_2.py first.")

    print(f"Discovering runs under {LOG_ROOT} ...")
    runs = discover_runs(LOG_ROOT)
    if runs.empty:
        raise SystemExit("No runs found.")
    print(f"  found {len(runs)} runs")
    print(runs.groupby(["env", "agent", "wm"]).size())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nBuilding per-env adapting curves ...")
    plot_adaptation_curves(runs, OUT_DIR)

    print("\nBuilding final-performance bar chart ...")
    agg = plot_final_performance(runs, OUT_DIR)

    print("\nBuilding adaptation-gap chart vs Exp 1 baseline ...")
    plot_adaptation_gap(agg, OUT_DIR)

    print("\nDone. Figures saved to:", OUT_DIR.resolve())


if __name__ == "__main__":
    main()
