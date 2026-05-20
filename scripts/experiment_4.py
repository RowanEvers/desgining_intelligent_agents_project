"""Experiment 4: adaptation of Exp 3 agents to perturbed-gravity HalfCheetah.

Loads each Exp 3 checkpoint and continues training on HalfCheetah-v5 with
gravity scaled to ×1.5. Tests how quickly each agent's learned policy +
representation recovers from the dynamics shift.

For deliberative agents we run BOTH world-model modes during adaptation:
  - frozen   : WM held fixed, only actor+critic adapt. The canonical
               "is the learned representation transferable?" probe.
  - unfrozen : WM keeps learning, so it can re-fit the perturbed dynamics.
               More realistic continual learning, and especially relevant
               here because the policy is trained ONLY on imagined rollouts
               — a frozen WM would have it adapt inside a now-stale simulator.
The frozen-vs-unfrozen contrast is itself a result worth reporting.
SAC has no world model, so it just continues training (one variant).

Grid: 1 env × 5 seeds × {sac, gaussian×[frozen,unfrozen],
categorical×[frozen,unfrozen]} = 5 jobs/seed × 5 seeds = 25 runs.
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

# Deliberative agents run both WM modes during adaptation; SAC has no world
# model so it runs a single mode (tagged "na" and given no wm suffix).
WM_MODES_DELIBERATIVE = ["frozen", "unfrozen"]
WM_MODES_REACTIVE = ["na"]

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


def run_tag(agent_kind: str, seed: int, wm_mode: str) -> str:
    base = f"{agent_kind}_{ENV_ID}_seed{seed}_{PERT_KIND}x{PERT_SCALE}"
    if agent_kind == "sac":
        return base
    # e.g. gaussian_HalfCheetah-v5_seed0_gravityx1.5_wmfrozen / _wmunfrozen.
    # Note: frozen runs keep the same tag as the previous frozen-only version,
    # so any already-completed frozen runs are still picked up by skip-already-done.
    return f"{base}_wm{wm_mode}"


def expand_run_grid():
    """Yield (agent_kind, seed, wm_mode) tuples in a deterministic order.

    Order matters: partition assignment is run-index modulo num-partitions,
    so interleaving agents within each seed keeps partitions balanced.
    Deliberative agents expand to both frozen and unfrozen WM modes; SAC
    yields a single "na" mode.
    """
    for seed in SEEDS:
        for agent_kind in AGENTS:
            wm_modes = (WM_MODES_DELIBERATIVE if agent_kind != "sac"
                        else WM_MODES_REACTIVE)
            for wm_mode in wm_modes:
                yield (agent_kind, seed, wm_mode)


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

    runs = list(expand_run_grid())
    runs_done = 0
    runs_skipped_partition = 0
    runs_skipped_done = 0
    runs_skipped_missing = 0
    runs_failed = 0

    print(f"Experiment 4: adapt to {PERT_KIND}×{PERT_SCALE} on {ENV_ID}, "
          f"{len(runs)} total runs, adapt_steps={adapt_steps}, "
          f"partition {args.partition}/{args.num_partitions}, device={device}\n")

    for run_idx, (agent_kind, seed, wm_mode) in enumerate(runs):
        if run_idx % args.num_partitions != args.partition:
            runs_skipped_partition += 1
            continue

        tag = run_tag(agent_kind, seed, wm_mode)
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
        # normalize_reward=False to match Exp 3's setup (per the A/B test).
        # If Exp 3 source ckpts were trained with NormalizeReward, swap this
        # back to True so the env's reward scale matches what the loaded
        # agent expects.
        env = make_env(ENV_ID, seed=seed + 50_000, gravity_scale=PERT_SCALE,
                       normalize_reward=False)
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
                # wm_mode picks the adaptation probe:
                #   frozen   -> WM held fixed, only actor+critic adapt
                #               (representation-transfer question).
                #   unfrozen -> WM keeps learning, re-fitting perturbed
                #               dynamics (continual learning; matters here
                #               because the policy trains only in imagination).
                agent.prepare_for_adaptation(
                    freeze_world_model=(wm_mode == "frozen"),
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
