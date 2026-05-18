"""Diagnostic probe for a trained LatentSACAgent checkpoint.

Answers three questions:

  1. What's the entropy temperature (alpha) at end of training? If it's still
     large (>1.0), the policy was very stochastic — eval-deterministic mode
     will be OOD relative to anything seen during training.

  2. (Categorical only) Is the encoder's mean close to one-hot? If not, eval
     mode is feeding the actor inputs it never saw at training time.

  3. Is the world-model reward head accurate on real env transitions? Reports
     correlation between r_hat and r_real and the magnitude ratio.

Run on MLiS from repo root::

    python scripts/diagnose_latent.py \
        --ckpt logs/experiment_1/gaussian_Pendulum-v1_steps50000_seed0/final.zip \
        --env  Pendulum-v1 \
        --n-samples 200
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from environments.wrappers import make_env
from agents.deliberative.latent_sac_agent import LatentSACAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to a final.zip checkpoint.")
    p.add_argument("--env", required=True, help="Env id (matches training env).")
    p.add_argument("--n-samples", type=int, default=200,
                   help="Number of real env transitions to probe.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(args.env, seed=args.seed)
    agent = LatentSACAgent.load(
        args.ckpt, env=env, seed=args.seed, device=device,
        restore_env_stats=True, freeze_env_stats=True,
    )
    print(f"\n=== Diagnostics for {Path(args.ckpt).parent.name} ===\n")

    # --- 1. alpha ----------------------------------------------------------
    alpha = agent.log_alpha.exp().item()
    print(f"[1] alpha = {alpha:.4f}")
    if alpha > 1.0:
        print(f"    WARNING: alpha is large. Policy was very stochastic. "
              f"Eval with deterministic=True will be OOD.")
    elif alpha < 0.01:
        print(f"    NOTE: alpha is very small. Policy is near-deterministic in training.")
    else:
        print(f"    OK: alpha in healthy range.")

    # --- 2. Encoder mean vs one-hot (categorical only) ---------------------
    print()
    print("[2] Encoder output structure on real states:")
    obs, _ = env.reset(seed=args.seed)
    obs_batch = []
    for _ in range(min(64, args.n_samples)):
        action = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(action)
        obs_batch.append(obs)
        if term or trunc:
            obs, _ = env.reset()
    obs_t = torch.as_tensor(np.stack(obs_batch), dtype=torch.float32, device=device)

    with torch.no_grad():
        q_z = agent.world_model.encode(obs_t)
        z_mean = q_z.mean
        z_sample = q_z.rsample()

    # Reshape categorical to (B, num_cat, num_classes) if applicable
    wm_kwargs = getattr(agent, "_world_model_kwargs", {})
    is_categorical = "num_cat" in wm_kwargs and "num_classes" in wm_kwargs
    if is_categorical:
        num_cat = wm_kwargs["num_cat"]
        num_classes = wm_kwargs["num_classes"]
        z_mean_r = z_mean.view(-1, num_cat, num_classes)
        z_samp_r = z_sample.view(-1, num_cat, num_classes)
        # How peaked is the mean? max prob per category
        max_prob = z_mean_r.max(dim=-1).values.mean().item()
        # Is the sample one-hot?
        sample_max = z_samp_r.max(dim=-1).values.mean().item()
        print(f"    Categorical WM: num_cat={num_cat}, num_classes={num_classes}")
        print(f"    Mean encoder output: avg max-prob per category = {max_prob:.4f}")
        print(f"      (1.0 = perfectly peaked; 1/{num_classes}={1/num_classes:.3f} = uniform)")
        print(f"    Sample encoder output: avg max value per category = {sample_max:.4f}")
        print(f"      (should be ~1.0 for one-hot / straight-through samples)")
        if max_prob < 0.5:
            print(f"    WARNING: Encoder mean is far from one-hot. The actor was"
                  f" trained on near-one-hot samples; feeding it the mean at eval"
                  f" produces OOD behaviour.")
    else:
        z_norm_mean = z_mean.norm(dim=-1).mean().item()
        z_norm_samp = z_sample.norm(dim=-1).mean().item()
        print(f"    Gaussian WM: latent_dim={z_mean.shape[-1]}")
        print(f"    ||z_mean||  avg = {z_norm_mean:.4f}")
        print(f"    ||z_sample|| avg = {z_norm_samp:.4f}")
        print(f"    Ratio (mean/sample) = {z_norm_mean/max(z_norm_samp,1e-8):.3f}")

    # --- 3. WM reward head accuracy ---------------------------------------
    print()
    print("[3] WM reward-head accuracy on real env transitions:")
    obs, _ = env.reset(seed=args.seed + 1)
    z_list, a_list, r_real_list = [], [], []
    for _ in range(args.n_samples):
        action = env.action_space.sample()
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            z = agent.world_model.encode(obs_t).rsample()
        a_t = torch.as_tensor(action, dtype=torch.float32, device=device).unsqueeze(0)
        next_obs, r, term, trunc, _ = env.step(action)
        z_list.append(z)
        a_list.append(a_t)
        r_real_list.append(r)
        if term or trunc:
            obs, _ = env.reset()
        else:
            obs = next_obs

    z = torch.cat(z_list, dim=0)
    a = torch.cat(a_list, dim=0)
    r_real = torch.tensor(r_real_list, dtype=torch.float32, device=device)
    with torch.no_grad():
        r_hat = agent.world_model.reward(z, a).squeeze(-1)

    r_hat_np = r_hat.cpu().numpy()
    r_real_np = r_real.cpu().numpy()
    corr = np.corrcoef(r_hat_np, r_real_np)[0, 1]
    print(f"    Samples: {len(r_hat_np)}")
    print(f"    r_real  : mean={r_real_np.mean():+.4f}  std={r_real_np.std():.4f}"
          f"  min={r_real_np.min():+.4f}  max={r_real_np.max():+.4f}")
    print(f"    r_hat   : mean={r_hat_np.mean():+.4f}  std={r_hat_np.std():.4f}"
          f"  min={r_hat_np.min():+.4f}  max={r_hat_np.max():+.4f}")
    print(f"    correlation(r_hat, r_real) = {corr:+.4f}")
    if abs(corr) < 0.3:
        print(f"    WARNING: WM reward head is essentially uncorrelated with"
              f" real reward. SAC was trained on noise.")
    elif corr < 0.7:
        print(f"    NOTE: WM reward head has moderate correlation. Imagined"
              f" returns are biased but not entirely wrong.")
    else:
        print(f"    OK: WM reward head tracks real reward.")

    # --- 4. Open-loop dynamics drift ---------------------------------------
    # Take a real (o, a) trajectory. Imagine forward from o_0 using the
    # *real* actions. Compare imagined z_t to encoded real o_t at each step.
    # Healthy WM: drift grows slowly (linearly). Broken WM: drift explodes.
    print("[4] Open-loop dynamics drift over the imagination horizon:")
    HORIZON = max(getattr(agent, "horizon", 5), 10)
    N_TRAJ = 32
    drifts_per_step = [[] for _ in range(HORIZON + 1)]  # +1 for t=0

    for traj in range(N_TRAJ):
        # Build a real trajectory of HORIZON steps
        obs, _ = env.reset(seed=args.seed + 1000 + traj)
        real_obs = [obs]
        real_actions = []
        for _ in range(HORIZON):
            a = env.action_space.sample()
            real_actions.append(a)
            obs, _, term, trunc, _ = env.step(a)
            real_obs.append(obs)
            if term or trunc:
                break
        if len(real_actions) < HORIZON:
            continue  # episode ended early — skip

        # Encode all real obs along the trajectory
        with torch.no_grad():
            real_obs_t = torch.as_tensor(np.stack(real_obs), dtype=torch.float32, device=device)
            z_real = agent.world_model.encode(real_obs_t).mean  # (H+1, D)

            # Imagine forward from z_0 using real actions
            z_imag = [z_real[0:1]]  # (1, D)
            for t in range(HORIZON):
                a_t = torch.as_tensor(real_actions[t], dtype=torch.float32, device=device).unsqueeze(0)
                z_next = agent.world_model.imagine(z_imag[-1], a_t).rsample()
                z_imag.append(z_next)
            z_imag_cat = torch.cat(z_imag, dim=0)  # (H+1, D)

            # Per-step L2 error in latent space
            for t in range(HORIZON + 1):
                err = (z_imag_cat[t] - z_real[t]).norm().item()
                drifts_per_step[t].append(err)

    print(f"    Averaged over {N_TRAJ} trajectories, random actions, horizon {HORIZON}:")
    print(f"    {'step':>4}  {'||z_imag - z_real||':>22}  {'ratio_to_t1':>12}")
    if drifts_per_step[1]:
        baseline = np.mean(drifts_per_step[1])
        for t, errs in enumerate(drifts_per_step):
            if errs:
                m = np.mean(errs)
                print(f"    {t:>4}  {m:>22.4f}  {m/max(baseline,1e-8):>12.2f}x")
        ratio = np.mean(drifts_per_step[5]) / max(np.mean(drifts_per_step[1]), 1e-8)
        if ratio > 3.0:
            print(f"    WARNING: drift grows {ratio:.1f}x over 5 steps. Imagined"
                  f" trajectories diverge fast — SAC trains on hallucinations.")
        else:
            print(f"    OK: drift grows {ratio:.1f}x over 5 steps. Dynamics tracks"
                  f" real env reasonably well.")

    env.close()
    print()


if __name__ == "__main__":
    main()
