"""Experiment 3: focused deep comparison on HalfCheetah-v5.

Replaces the original three-env sweep (Exp 1) with a single-env, more-seeds,
larger-world-model study. The motivation is that state-based MuJoCo isn't
a regime where small world models are expected to beat SAC; HalfCheetah
specifically is the canonical MBRL benchmark (dense reward, no early
termination, smooth dynamics) so it gives the deliberative agents their
best chance.

Grid: 1 env × 3 agents × 5 seeds = 15 runs at 1M steps each.

Key hparam differences vs Exp 1 (latent agents only — SAC keeps its
SB3 defaults so the model-free baseline stays comparable to literature):

  hidden_dim:        128  ->  256
  latent_dim (gauss): 16  ->   32
  categorical:       8x8  -> 16x16  (256-d flat latent)
  horizon:             5  ->   15   (longer imagination)
  wm_updates_per_iter: 20 ->   50
  sac_updates_per_iter: 20 ->   50
  env_steps_per_iter: 200 ->  500
  warmup_steps:       500 -> 5000
  env_buffer_size:    10k ->  200k
  batch_size:          64 ->  128

Usage (from repo root):

    # Pipeline check (10k steps each)
    python scripts/experiment_3.py --smoke

    # Full grid, single worker (estimated ~12-18h on MLiS GPU)
    python scripts/experiment_3.py

    # Split across multiple GPU jobs
    python scripts/experiment_3.py --partition 0 --num-partitions 3
    python scripts/experiment_3.py --partition 1 --num-partitions 3
    python scripts/experiment_3.py --partition 2 --num-partitions 3
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
from agents.deliberative.world_model.world_model import GaussianWorldModel
from agents.deliberative.world_model.world_model_categorical import CategoricalWorldModel
from agents.reactive.sac import SACagent
from scripts.smoke_test import banner


# ===========================================================================
#   Grid
# ===========================================================================
ENV_ID = "HalfCheetah-v5"
AGENTS = ["sac", "gaussian", "categorical"]
SEEDS = [0, 1, 2, 3, 4]
FULL_STEPS = 1_000_000
SMOKE_STEPS = 10_000

LOG_ROOT = Path("logs/experiment_3")


# ===========================================================================
#   Hyperparameters — explicit so the paper can quote them verbatim
# ===========================================================================
LATENT_COMMON_HPARAMS = dict(
    hidden_dim=256,             # was 128 — bumped for HalfCheetah-scale obs
    horizon=15,                 # was  5 — longer imagined rollouts
    batch_size=128,             # was 64
    env_steps_per_iter=500,     # was 200 — fewer outer iters per run
    wm_updates_per_iter=50,     # was 20 — WM was undertrained in Exp 1
    sac_updates_per_iter=50,    # was 20
    warmup_steps=5_000,         # was 500 — more random data before policy kicks in
    env_buffer_size=200_000,    # was 10k — 1M-step run shouldn't overwrite early data
    imag_buffer_size=200_000,   # was 10k
)

# Reactive SAC keeps SB3's tuned defaults; only learning_rate is exposed.
REACTIVE_HPARAMS = dict(
    learning_rate=3e-4,
)


def build_agent(env, seed, logger, device, agent_kind, log_dir):
    if agent_kind == "gaussian":
        return LatentSACAgent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            world_model_cls=GaussianWorldModel,
            world_model_kwargs={"latent_dim": 32},  # was 16
            **LATENT_COMMON_HPARAMS,
        )
    if agent_kind == "categorical":
        # 16×16 categorical (256-d flat) — closer to Dreamer's 32×32 than
        # Exp 1's 8×8, but small enough to still run quickly.
        return LatentSACAgent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            world_model_cls=CategoricalWorldModel,
            world_model_kwargs={"num_cat": 16, "num_classes": 16},
            **LATENT_COMMON_HPARAMS,
        )
    if agent_kind == "sac":
        return SACagent(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            **REACTIVE_HPARAMS,
        )
    raise ValueError(f"Unknown agent_kind: {agent_kind}")


def expand_run_grid():
    """Yield (agent_kind, seed) tuples in a deterministic order.
    Order matters: partition assignment is run-index modulo num-partitions.
    Interleaving agents (rather than grouping by agent) means each partition
    gets a balanced mix, so a partition that crashes early doesn't take
    out an entire agent's data.
    """
    for seed in SEEDS:
        for agent_kind in AGENTS:
            yield (agent_kind, seed)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 3: HalfCheetah-v5 deep comparison "
                    "(SAC vs Gaussian-WM vs Categorical-WM, 5 seeds each)."
    )
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--num-partitions", type=int, default=1)
    parser.add_argument("--smoke", action="store_true",
                        help="Use SMOKE_STEPS (10k) instead of FULL_STEPS (1M).")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not (0 <= args.partition < args.num_partitions):
        raise SystemExit(
            f"--partition must be in [0, {args.num_partitions}); got {args.partition}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    total_steps = SMOKE_STEPS if args.smoke else FULL_STEPS

    runs = list(expand_run_grid())
    runs_done = 0
    runs_skipped_partition = 0
    runs_skipped_done = 0
    runs_failed = 0

    print(f"Experiment 3: env={ENV_ID}, "
          f"{len(runs)} total runs, steps/run={total_steps}, "
          f"partition {args.partition}/{args.num_partitions}, device={device}\n")

    for run_idx, (agent_kind, seed) in enumerate(runs):
        if run_idx % args.num_partitions != args.partition:
            runs_skipped_partition += 1
            continue

        run_tag = f"{agent_kind}_{ENV_ID}_steps{total_steps}_seed{seed}"
        run_log_dir = LOG_ROOT / run_tag
        if (run_log_dir / "final.zip").exists():
            print(f"  [skip-already-done] {run_tag}")
            runs_skipped_done += 1
            continue

        print(f"[{run_idx + 1:>2}/{len(runs)}] {run_tag}")

        if args.dry_run:
            runs_done += 1
            continue

        run_log_dir.mkdir(parents=True, exist_ok=True)
        # normalize_reward=False per the A/B test (scripts/ab_normalize_reward.py):
        # SAC + NormalizeReward shrinks Q-values into a range where automatic
        # entropy tuning collapses exploration. At 50k steps the unnormalized
        # run reached 1512 mean vs 523 with NormalizeReward — 2.9× gap that
        # widens further by 1M steps.
        env = make_env(ENV_ID, seed=seed, normalize_reward=False)
        logger = Logger(log_dir=str(LOG_ROOT), agent_name=run_tag,
                        env_id=ENV_ID, seed=seed)
        try:
            agent = build_agent(env, seed, logger, device, agent_kind,
                                log_dir=str(LOG_ROOT))
            agent.train(total_timesteps=total_steps)
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
        f"{runs_skipped_done} already-done, "
        f"{runs_failed} failed. "
        f"Results in {LOG_ROOT}/."
    )


if __name__ == "__main__":
    main()
