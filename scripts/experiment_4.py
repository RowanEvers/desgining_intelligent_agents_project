"""Experiment 4: adaptation of Exp 3 agents to perturbed-gravity HalfCheetah.

Loads each Exp 3 checkpoint and continues training on HalfCheetah-v5 with
gravity scaled to ×1.5. Tests how quickly each agent's learned policy +
representation recovers from the dynamics shift.

For deliberative agents the world model is FROZEN during adaptation —
that's the canonical "is the learned representation transferable?" probe.
SAC has no world model, so it just continues training (one variant).

Grid: 1 env × 3 agents × 5 seeds × 1 perturbation × 1 wm mode = 15 runs.
Budget: 100k steps per run (10% of source training) — enough to see whether
each agent recovers, not so long that adaptation becomes "train from
scratch on perturbed env".

Usage (from repo root):

    python scripts/experiment_4.py --smoke      # 10k each, pipeline check
    python scripts/experiment_4.py              # full grid
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


ENV_ID = "HalfCheetah-v5"
AGENTS = ["sac", "gaussian", "categorical"]
SEEDS = [0, 1, 2, 3, 4]

# Source training was 1M steps; adaptation gets 10% of that to keep the
# experiment about *adaptation speed*, not "train from scratch on perturbed".
SRC_STEPS = 1_000_000
FULL_ADAPT_STEPS = 100_000
SMOKE_ADAPT_STEPS = 10_000

# Single perturbation: gravity ×1.5 (heavier-than-normal gravity is a
# canonical sim-to-real-style perturbation and HalfCheetah is gait-based,
# so changing gravity is a meaningful dynamics shift).
PERT_KIND = "gravity"
PERT_SCALE = 1.5

LOG_ROOT = Path("logs/experiment_4")
SRC_ROOT = Path("logs/experiment_3")


def src_ckpt_path(agent_kind: str, seed: int) -> Path:
    return SRC_ROOT / f"{agent_kind}_{ENV_ID}_steps{SRC_STEPS}_seed{seed}" / "final.zip"


def run_tag(agent_kind: str, seed: int) -> str:
    base = f"{agent_kind}_{ENV_ID}_seed{seed}_{PERT_KIND}x{PERT_SCALE}"
    return base + ("_wmfrozen" if agent_kind != "sac" else "")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 4: adapt Exp 3 ckpts to gravity-perturbed HalfCheetah."
    )
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.partition < args.num_partitions):
        raise SystemExit(
            f"--partition must be in [0, {args.num_partitions}); got {args.partition}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    adapt_steps = SMOKE_ADAPT_STEPS if args.smoke else FULL_ADAPT_STEPS

    runs = [(a, s) for s in SEEDS for a in AGENTS]
    runs_done = 0
    runs_skipped_partition = 0
    runs_skipped_done = 0
    runs_skipped_missing = 0
    runs_failed = 0

    print(f"Experiment 4: adapt to {PERT_KIND}×{PERT_SCALE} on {ENV_ID}, "
          f"{len(runs)} total runs, adapt_steps={adapt_steps}, "
          f"partition {args.partition}/{args.num_partitions}, device={device}\n")

    for run_idx, (agent_kind, seed) in enumerate(runs):
        if run_idx % args.num_partitions != args.partition:
            runs_skipped_partition += 1
            continue

        tag = run_tag(agent_kind, seed)
        run_log_dir = LOG_ROOT / tag
        if (run_log_dir / "final.zip").exists():
            print(f"  [skip-already-done] {tag}")
            runs_skipped_done += 1
            continue

        src = src_ckpt_path(agent_kind, seed)
        if not src.exists():
            print(f"  [skip-missing-ckpt] {tag}  (no {src})")
            runs_skipped_missing += 1
            continue

        print(f"[{run_idx + 1:>2}/{len(runs)}] {tag}")
        if args.dry_run:
            runs_done += 1
            continue

        run_log_dir.mkdir(parents=True, exist_ok=True)
        # Perturbed env. Offset seed so adaptation isn't a deterministic
        # replay of the first source episode.
        env = make_env(ENV_ID, seed=seed + 50_000, gravity_scale=PERT_SCALE)
        logger = Logger(log_dir=str(LOG_ROOT), agent_name=tag,
                        env_id=ENV_ID, seed=seed)

        try:
            if agent_kind == "sac":
                agent = SACagent.load(
                    str(src), env=env, seed=seed,
                    log_dir=str(LOG_ROOT), logger=logger,
                    device=device, learning_rate=3e-4,
                )
            else:
                agent = LatentSACAgent.load(
                    str(src), env=env, seed=seed,
                    log_dir=str(run_log_dir), logger=logger, device=device,
                    restore_env_stats=True, freeze_env_stats=False,
                )
                # Frozen WM only — tests representation transfer (the
                # canonical model-based RL adaptation question).
                agent.prepare_for_adaptation(
                    freeze_world_model=True,
                    reset_step_counter=True,
                    clear_buffers=True,
                )

            agent.train(total_timesteps=adapt_steps)
            agent.save(str(run_log_dir / "final.zip"))
            runs_done += 1
        except Exception as e:
            print(f"           [error] {type(e).__name__}: {e}")
            runs_failed += 1
        finally:
            logger.close()

    banner(
        f"Partition {args.partition}/{args.num_partitions}: "
        f"{runs_done} done, "
        f"{runs_skipped_partition} other-partition, "
        f"{runs_skipped_missing} missing-ckpt, "
        f"{runs_skipped_done} already-done, "
        f"{runs_failed} failed. "
        f"Results in {LOG_ROOT}/."
    )


if __name__ == "__main__":
    main()
