"""Experiment 2: continue training each Exp 1 agent on a PERTURBED env.

For every Exp 1 checkpoint, load the trained agent and resume training on
the *same env id* with a single physics perturbation applied (gravity or
friction scaled). Training budget per env matches Exp 1, so each adaptation
curve is directly comparable to the source training curve.

Grid (sequential index → partition):
  envs           = 3        Pendulum-v1, Hopper-v5, Walker2d-v5
  agents         = 3        sac, gaussian, categorical
  seeds          = 3        0, 1, 2
  perturbations  = 5        gravity x{0.5, 1.5, 2.0}, friction x{0.5, 2.0}
                            (friction skipped for Pendulum — no analogue)
  wm modes       = 2 for gaussian/categorical, 1 for sac

Total runs (full grid):
  Per (env, seed, perturbation) we run 5 jobs:
    sac (1) + gaussian × {frozen, unfrozen} (2) + categorical × {f, u} (2)
  Pendulum:  3 seeds × 3 grav perts × 5 jobs = 45
  Hopper:    3 seeds × 5 perts     × 5 jobs = 75
  Walker2d:  3 seeds × 5 perts     × 5 jobs = 75
  -------------------------------------------- = 195 runs

Outputs go to ``logs/experiment_2/<run_tag>/`` with a JSONL of metrics and
a ``final.zip`` checkpoint. ``episode_reward_raw`` is logged on every
episode end (via the RecordEpisodeStatistics wrapper added in
environments/wrappers.py), so the adaptation curves are plottable directly
without going back to a separate eval pass.

Usage (run from repo root):

    # Pipeline smoke check — short runs, all combos, ~1-2h CPU
    python scripts/experiment_2.py --smoke

    # Full grid, single worker (slow)
    python scripts/experiment_2.py

    # Full grid, 4 parallel workers
    python scripts/experiment_2.py --partition 0 --num-partitions 4
    python scripts/experiment_2.py --partition 1 --num-partitions 4
    python scripts/experiment_2.py --partition 2 --num-partitions 4
    python scripts/experiment_2.py --partition 3 --num-partitions 4
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.utils import Logger
from environments.wrappers import make_env
from agents.deliberative.latent_sac_agent import LatentSACAgent
from agents.reactive.sac import SACagent
from scripts.smoke_test import banner


# ===========================================================================
#   Grid definition
# ===========================================================================
ENVS = ["Pendulum-v1", "Hopper-v5", "Walker2d-v5"]
AGENTS = ["sac", "gaussian", "categorical"]
SEEDS = [0, 1, 2]

# (kind, scale) tuples. Keep this list deterministic — run_tag and partition
# assignment both depend on its order.
PERTURBATIONS = [
    ("gravity",  0.5),
    ("gravity",  1.5),
    ("gravity",  2.0),
    ("friction", 0.5),
    ("friction", 2.0),
]

# For deliberative agents we run BOTH frozen and unfrozen WM. For SAC there
# is no world model, so we only run a single mode (tagged "na").
WM_MODES_DELIBERATIVE = ["frozen", "unfrozen"]
WM_MODES_REACTIVE = ["na"]

# Budgets matching Exp 1's per-env schedule. Smoke shrinks everything to 10k.
SOURCE_STEPS = {"Pendulum-v1":   50_000,
                "Hopper-v5":    500_000,
                "Walker2d-v5": 1_000_000}
SMOKE_STEPS  = {"Pendulum-v1": 10_000,
                "Hopper-v5":   10_000,
                "Walker2d-v5": 10_000}

LOG_ROOT = Path("logs/experiment_2")
SRC_ROOT = Path("logs/experiment_1")


# ===========================================================================
#   Helpers
# ===========================================================================
def perturbation_kwargs(kind: str, scale: float) -> dict:
    """Map (kind, scale) -> kwargs accepted by make_env."""
    if kind == "gravity":
        return {"gravity_scale": scale}
    if kind == "friction":
        return {"friction_scale": scale}
    raise ValueError(f"Unknown perturbation kind: {kind}")


def perturbation_tag(kind: str, scale: float) -> str:
    """Stable string tag used in run names and log paths."""
    # e.g. ('gravity', 1.5) -> 'gravityx1.5'
    return f"{kind}x{scale}"


def src_ckpt_path(agent_kind: str, env_id: str, seed: int) -> Path:
    """Path to the Exp 1 checkpoint we'll resume from."""
    src_steps = SOURCE_STEPS[env_id]
    return SRC_ROOT / f"{agent_kind}_{env_id}_steps{src_steps}_seed{seed}" / "final.zip"


def run_tag(agent_kind: str, env_id: str, seed: int,
            kind: str, scale: float, wm_mode: str) -> str:
    """Tag used as the log subdir for a single run."""
    ptag = perturbation_tag(kind, scale)
    if agent_kind == "sac":
        return f"{agent_kind}_{env_id}_seed{seed}_{ptag}"
    return f"{agent_kind}_{env_id}_seed{seed}_{ptag}_wm{wm_mode}"


def expand_run_grid():
    """Yield (env_id, agent_kind, seed, kind, scale, wm_mode) tuples in a
    deterministic order. Skips combinations that don't make sense
    (Pendulum + friction, deliberative wm-modes for sac)."""
    for env_id in ENVS:
        for agent_kind in AGENTS:
            for seed in SEEDS:
                for kind, scale in PERTURBATIONS:
                    if env_id == "Pendulum-v1" and kind == "friction":
                        continue
                    wm_modes = (WM_MODES_DELIBERATIVE
                                if agent_kind != "sac"
                                else WM_MODES_REACTIVE)
                    for wm_mode in wm_modes:
                        yield (env_id, agent_kind, seed, kind, scale, wm_mode)


# ===========================================================================
#   Per-run driver
# ===========================================================================
def load_and_adapt(env_id: str, agent_kind: str, seed: int,
                   kind: str, scale: float, wm_mode: str,
                   target_steps: int, device: str,
                   run_log_dir: Path, parent_log_dir: Path,
                   logger: Logger):
    """Construct perturbed env, load source ckpt, prep for adaptation,
    and run training. Saves final.zip into run_log_dir."""
    # Perturbed env. Offset the seed so adaptation isn't seeded identically
    # to the source run — otherwise the first few episodes are deterministic
    # repeats of training-time episodes.
    pkwargs = perturbation_kwargs(kind, scale)
    env = make_env(env_id, seed=seed + 50_000, **pkwargs)

    src = src_ckpt_path(agent_kind, env_id, seed)

    if agent_kind == "sac":
        # Reactive agent: load SB3 SAC + the saved obs_rms stats, then call
        # .train(steps) which routes through model.learn() on the new env.
        agent = SACagent.load(
            str(src), env=env, seed=seed,
            log_dir=str(parent_log_dir), logger=logger,
            device=device, learning_rate=3e-4,
        )
    else:
        # Deliberative agent: load returns a fully-initialised LatentSACAgent
        # with restored env-norm stats (NOT frozen — we want them to keep
        # updating as the perturbed env's stat distribution drifts).
        agent = LatentSACAgent.load(
            str(src), env=env, seed=seed,
            log_dir=str(run_log_dir), logger=logger, device=device,
            restore_env_stats=True, freeze_env_stats=False,
        )
        agent.prepare_for_adaptation(
            freeze_world_model=(wm_mode == "frozen"),
            reset_step_counter=True,    # adaptation curves start at step 0
            clear_buffers=True,         # don't mix source-env transitions in
        )

    agent.train(total_timesteps=int(target_steps))
    agent.save(str(run_log_dir / "final.zip"))


# ===========================================================================
#   Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Experiment 2: load Exp 1 checkpoints and adapt to "
                    "perturbed-physics versions of the same env."
    )
    parser.add_argument(
        "--partition", type=int, default=0,
        help="Which slice of the run grid this process handles "
             "(0-indexed, must be < --num-partitions).",
    )
    parser.add_argument(
        "--num-partitions", type=int, default=1,
        help="How many parallel workers will share the run grid. "
             "Each worker handles runs whose index mod num-partitions "
             "equals its --partition.",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Use SMOKE_STEPS (10k each) instead of full SOURCE_STEPS. "
             "Use this to validate the pipeline end-to-end before committing "
             "compute to the real sweep.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the run grid and which runs THIS partition would do, "
             "then exit. Doesn't load anything or touch the env.",
    )
    args = parser.parse_args()

    if not (0 <= args.partition < args.num_partitions):
        raise SystemExit(
            f"--partition must be in [0, {args.num_partitions}); got {args.partition}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)

    runs = list(expand_run_grid())
    runs_done = 0
    runs_skipped_partition = 0
    runs_skipped_missing = 0
    runs_skipped_already = 0
    runs_failed = 0

    print(f"Experiment 2: {len(runs)} total runs in grid. "
          f"Partition {args.partition}/{args.num_partitions} on {device}.\n")

    for run_idx, (env_id, agent_kind, seed, kind, scale, wm_mode) in enumerate(runs):
        if run_idx % args.num_partitions != args.partition:
            runs_skipped_partition += 1
            continue

        tag = run_tag(agent_kind, env_id, seed, kind, scale, wm_mode)
        run_log_dir = LOG_ROOT / tag

        # Skip if source ckpt missing (e.g. Exp 1 partition didn't finish).
        ckpt = src_ckpt_path(agent_kind, env_id, seed)
        if not ckpt.exists():
            print(f"  [skip-missing-ckpt] {tag}  (no {ckpt})")
            runs_skipped_missing += 1
            continue

        # Skip if this run already completed — makes the script re-runnable
        # after a crash without redoing finished work.
        if (run_log_dir / "final.zip").exists():
            print(f"  [skip-already-done] {tag}")
            runs_skipped_already += 1
            continue

        target_steps = SMOKE_STEPS[env_id] if args.smoke else SOURCE_STEPS[env_id]
        print(f"[{run_idx + 1:>3}/{len(runs)}] {tag}  steps={target_steps}")

        if args.dry_run:
            runs_done += 1
            continue

        run_log_dir.mkdir(parents=True, exist_ok=True)
        logger = Logger(log_dir=str(LOG_ROOT), agent_name=tag,
                        env_id=env_id, seed=seed)
        try:
            load_and_adapt(
                env_id=env_id, agent_kind=agent_kind, seed=seed,
                kind=kind, scale=scale, wm_mode=wm_mode,
                target_steps=target_steps, device=device,
                run_log_dir=run_log_dir, parent_log_dir=LOG_ROOT,
                logger=logger,
            )
            runs_done += 1
        except Exception as e:
            # Don't let a single bad run kill the whole partition — log it
            # and keep going. The skip-already-done check above lets you
            # re-run later to retry just the failures.
            print(f"           [error] {type(e).__name__}: {e}")
            runs_failed += 1
        finally:
            logger.close()

    banner(
        f"Partition {args.partition}/{args.num_partitions}: "
        f"{runs_done} done, "
        f"{runs_skipped_partition} other-partition, "
        f"{runs_skipped_missing} missing-ckpt, "
        f"{runs_skipped_already} already-done, "
        f"{runs_failed} failed. "
        f"Results in {LOG_ROOT}/."
    )


if __name__ == "__main__":
    main()
