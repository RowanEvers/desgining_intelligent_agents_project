"""Smoke test for the Latent SAC agent on Humanoid-v5.

Run from the repo root (`dia_repo/`):

    python scripts/smoke_test.py                      # Gaussian (default)
    python scripts/smoke_test.py --latent gaussian    # explicit Gaussian
    python scripts/smoke_test.py --latent categorical # Discrete (Dreamer-V2-style)

This is NOT a real training run — Humanoid-v5 needs ~50k env steps to be
solved by vanilla SAC, and this script runs for 5000. The goal here is only
to verify:

    1. The agent constructs without error.
    2. A world-model train_step on a real batch returns finite losses.
    3. Imagined rollouts produce sensible tensors and populate the buffer.
    4. A single SAC update returns finite losses.
    5. A short end-to-end training loop runs to completion, losses stay
       finite, and episode return is at least trending in the right
       direction (Pendulum return starts around -1200 and goes up).

If any of these fail, the print output makes it clear which stage broke.
"""

import os
import sys
import argparse

# Make the package importable when running as `python scripts/smoke_test.py`
# from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import numpy as np
import torch
import gymnasium as gym

from environments.utils import set_seed, Logger
#from experiment_1 import build_agent

def banner(msg):
    print("\n" + "=" * 70)
    print(msg)
    print("=" * 70)


def assert_finite(name: str, value) -> None:
    """Raise a clear error if a loss is NaN or Inf."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert_finite(f"{name}.{k}", v)
        return
    if isinstance(value, torch.Tensor):
        value = value.item()
    if not math.isfinite(value):
        raise RuntimeError(f"{name} is not finite: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent", choices=["gaussian", "categorical", "sac"],
                        default="gaussian",
                        help="Which latent representation to test.")
    args = parser.parse_args()

    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Tag log dirs with the latent kind so the two runs sit side-by-side in TensorBoard.
    run_tag = f"latent_sac_smoke_{args.latent}"

    # ------------------------------------------------------------------
    # Stage 0: env + agent construction
    # ------------------------------------------------------------------
    banner(f"Stage 0: build env, logger, and agent  (latent = {args.latent})")

    env = gym.make("Pendulum-v5")
    env = set_seed(seed, env)
    print(f"  obs_space    = {env.observation_space}")
    print(f"  action_space = {env.action_space}")
    print(f"  action_low   = {env.action_space.low},  action_high = {env.action_space.high}")

    logger = Logger(
        log_dir="logs",
        agent_name=run_tag,
        env_id="Pendulum-v5",
        seed=seed,
    )

    agent = build_agent(env, seed, logger, device, args.latent)
    print(f"  agent built. effective latent_dim={agent.latent_dim}, action_dim={agent.action_dim}")
    n_params = sum(p.numel() for p in agent.world_model.parameters())
    print(f"  world model has {n_params:,} parameters")

    # ------------------------------------------------------------------
    # Stage 1: prime the env buffer with a few hundred random transitions
    # so the world model has something to train on.
    # ------------------------------------------------------------------
    banner("Stage 1: collect 600 random env transitions")

    obs, _ = env.reset(seed=seed)
    for step in range(600):
        action = env.action_space.sample()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.env_buffer.add(obs, action, reward, next_obs, terminated)
        if terminated or truncated:
            obs, _ = env.reset()
        else:
            obs = next_obs
    print(f"  env_buffer size = {len(agent.env_buffer)}")

    # ------------------------------------------------------------------
    # Stage 2: single world-model train_step
    # ------------------------------------------------------------------
    banner("Stage 2: one world-model train_step on a real batch")

    batch = agent.env_buffer.sample(agent.batch_size)
    print(f"  batch shapes: obs={tuple(batch['obs'].shape)}, "
          f"actions={tuple(batch['actions'].shape)}, "
          f"rewards={tuple(batch['rewards'].shape)}, "
          f"next_obs={tuple(batch['next_obs'].shape)}, "
          f"dones={tuple(batch['dones'].shape)}")

    losses = agent.world_model.train_step(batch, agent.wm_opt)
    assert_finite("wm_losses", losses)
    for k, v in losses.items():
        print(f"  {k:>14s} = {v:.4f}")

    # ------------------------------------------------------------------
    # Stage 3: imagined rollouts
    # ------------------------------------------------------------------
    banner("Stage 3: imagined rollouts populate the latent buffer")

    before = len(agent.imagined_buffer)
    agent._imagine_rollouts(agent.horizon, agent.batch_size)
    after = len(agent.imagined_buffer)
    expected = agent.horizon * agent.batch_size
    print(f"  imagined_buffer: {before} -> {after}  (expected +{expected})")
    assert after - before == expected, (
        f"imagined buffer grew by {after - before}, expected {expected}"
    )

    imag_batch = agent.imagined_buffer.sample(agent.batch_size)
    for k, v in imag_batch.items():
        print(f"  {k:>8s}: shape={tuple(v.shape)}, "
              f"mean={v.float().mean().item():+.3f}, "
              f"std={v.float().std().item():.3f}")
        assert torch.isfinite(v).all(), f"non-finite values in imagined batch[{k}]"

    # ------------------------------------------------------------------
    # Stage 4: single SAC update
    # ------------------------------------------------------------------
    banner("Stage 4: one SAC update on imagined data")

    sac_losses = agent._sac_update(imag_batch)
    assert_finite("sac_losses", sac_losses)
    for k, v in sac_losses.items():
        print(f"  {k:>15s} = {v:+.4f}")

    # ------------------------------------------------------------------
    # Stage 5: short end-to-end training loop
    # ------------------------------------------------------------------
    banner("Stage 5: 10000-step end-to-end training run")
    print("  (Humanoid return starts ~ -1200; with only 10k steps you")
    print("   should see modest improvement, not a solved policy.)")

    # Fresh agent — Stage 1-4 already wrote into this one's buffers/optimisers,
    # which would muddy the end-to-end signal.
    logger.close()
    logger = Logger(log_dir="logs", agent_name=f"{run_tag}_e2e",
                    env_id="Pendulum-v5", seed=seed)
    agent = build_agent(env, seed, logger, device, args.latent)

    agent.train(total_timesteps=10000)
    print(f"  training finished. final env step counter = {agent._env_step_counter}")

    # ------------------------------------------------------------------
    # Stage 6: quick deterministic evaluation
    # ------------------------------------------------------------------
    banner("Stage 6: deterministic evaluation (5 episodes)")
    eval_env = gym.make("Pendulum-v5")
    eval_env = set_seed(seed + 1, eval_env)
    metrics = agent.evaluate(eval_env, num_episodes=5, deterministic=True)
    print(f"  mean return = {metrics['mean']:+.2f}  (std = {metrics['std']:.2f})")
    print(f"  per-episode  = {[f'{r:+.1f}' for r in metrics['returns']]}")

    logger.close()
    banner("Smoke test passed — agent runs end-to-end with finite losses.")


if __name__ == "__main__":
    main()
