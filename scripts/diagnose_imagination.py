"""Imagination-quality diagnostics for a trained LatentSACAgent checkpoint.

This is the companion to ``scripts/diagnose_latent.py``. That script answers
"is the model individually healthy?" (alpha, encoder structure, one-step reward
correlation, latent drift). This script answers the question those leave open:

    The world model fits its own losses, SAC reports a positive imagined value
    (q_mean), yet the real return sits near zero. Is SAC maximising an imagined
    return that does NOT exist in the real environment?

That decoupling — "model exploitation" — is the classic failure of model-based
RL when a learned dynamics model is rolled out further than it is accurate and
the policy is optimised against the resulting hallucination. The exp-3 logs are
consistent with it: q_mean ~ 15-24 while raw episodic return is negative for
most of training, and horizon was raised 5 -> 15 for exp 3.

Four probes, in order of how directly they implicate exploitation:

  [A] Real eval return + action saturation. Establishes ground truth (what the
      deterministic policy actually scores) and checks for bang-bang collapse
      (|a| pinned at the action bounds), which alpha -> 0 early in training
      tends to produce.

  [B] Imagined vs real return GAP (the headline probe). Along on-policy real
      trajectories, compare the H-step return the world model PREDICTS from a
      state to the H-step return the agent ACTUALLY receives from that state.
      A large positive gap (imagined >> real) is the signature of exploitation.

  [C] Reward-head accuracy vs imagination depth. Using the *real* actions taken,
      compare the reward head evaluated on (i) the encoder latent at each step
      (in-distribution; the latents it was trained on) against (ii) the latent
      reached by rolling the dynamics forward from step 0 (what SAC actually
      sees at depth t in imagination). The gap between these two curves, as a
      function of depth, is exactly how much the H-step rollout corrupts the
      reward signal that drives the policy.

  [D] Reward-head R^2 and the mean-predictor baseline. A small reward MSE can be
      meaningless if the real reward barely varies (a non-moving cheetah). R^2
      relative to "always predict the mean" tells you whether the head carries
      real signal or just the mean.

Agent-agnostic: works for both Gaussian and Categorical world models, since it
only uses the shared public API (encode / imagine / reward / continue_prob /
actor).

Run on MLiS from the repo root, e.g.::

    python scripts/diagnose_imagination.py \
        --ckpt logs/experiment_3/gaussian_HalfCheetah-v5_steps1000000_seed0/final.zip \
        --env  HalfCheetah-v5 \
        --episodes 5

Compare a Gaussian and a Categorical checkpoint side by side; if both show the
same imagined-vs-real gap, the problem is the shared model-based loop (horizon,
rollout correction), not the latent representation.
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


# ---------------------------------------------------------------------------
#  Small helpers
# ---------------------------------------------------------------------------
def _t(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _encode_samples(agent, obs, device, n=1):
    """Draw ``n`` posterior latent samples for a single obs -> (n, D).

    Uses rsample() (not the mean) so the latents match what training feeds the
    dynamics/reward heads in _imagine_rollouts. This matters for the categorical
    world model, whose dynamics never saw the (non-one-hot) posterior mean.
    """
    with torch.no_grad():
        q = agent.world_model.encode(_t(obs, device).unsqueeze(0))
        return torch.cat([q.rsample() for _ in range(n)], dim=0)


@torch.no_grad()
def imagine_return(agent, z, horizon, gamma, deterministic_actor=True):
    """Expected H-step imagined return the world model predicts from latent ``z``.

    Mirrors LatentSACAgent._imagine_rollouts exactly (actor -> reward(z,a) ->
    imagine(z,a).rsample()), accumulating r_hat. ``z`` is (S, D); the S parallel
    rollouts average out the rsample() noise. Returns the mean discounted return
    AND the mean undiscounted sum for interpretability.
    """
    device = z.device
    disc_return = torch.zeros(z.shape[0], device=device)
    undisc_return = torch.zeros(z.shape[0], device=device)
    discount = 1.0
    for _ in range(horizon):
        action, _ = agent.actor(z, deterministic=deterministic_actor)
        r_hat = agent.world_model.reward(z, action)   # (S,)
        disc_return += discount * r_hat
        undisc_return += r_hat
        discount *= gamma
        z = agent.world_model.imagine(z, action).rsample()
    return disc_return.mean().item(), undisc_return.mean().item()


# ---------------------------------------------------------------------------
#  Probe A — real eval return + action saturation
# ---------------------------------------------------------------------------
def probe_eval(agent, env, episodes, device, sat_threshold=0.99):
    print("[A] Real-env eval (deterministic policy):")
    returns, lengths = [], []
    abs_actions, sat_frac = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=1234 + ep)
        ep_ret_raw, done, length = 0.0, False, 0
        while not done:
            a = agent.act(obs, deterministic=True)
            abs_actions.append(np.abs(a))
            sat_frac.append(np.mean(np.abs(a) >= sat_threshold))
            obs, _, term, trunc, info = env.step(a)
            length += 1
            done = term or trunc
            if done:
                ep_info = info.get("episode") if isinstance(info, dict) else None
                if ep_info is not None and "r" in ep_info:
                    ep_ret_raw = float(ep_info["r"])
        returns.append(ep_ret_raw)
        lengths.append(length)
    returns = np.array(returns)
    abs_actions = np.stack(abs_actions)
    print(f"    episodes={episodes}  raw return: mean={returns.mean():+.1f}  "
          f"std={returns.std():.1f}  min={returns.min():+.1f}  max={returns.max():+.1f}")
    print(f"    mean episode length = {np.mean(lengths):.0f}")
    print(f"    action |a|: mean={abs_actions.mean():.3f}  "
          f"per-dim mean={np.array2string(abs_actions.mean(0), precision=2)}")
    sat = float(np.mean(sat_frac))
    print(f"    fraction of action components saturated (|a|>={sat_threshold}) = {sat:.2%}")
    if sat > 0.5:
        print("    WARNING: policy is largely bang-bang (saturated). Consistent with "
              "alpha collapsing to ~0 early — exploration died and the actor only "
              "explores the corners of the action cube.")
    return returns.mean()


# ---------------------------------------------------------------------------
#  Probe B — imagined vs real return gap (the headline)
# ---------------------------------------------------------------------------
def probe_return_gap(agent, env, episodes, horizon, gamma, device, imag_samples=8):
    print("\n[B] Imagined vs real H-step return along on-policy trajectories "
          f"(H={horizon}, gamma={gamma}):")
    imagined, real = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=4321 + ep)
        # Roll the deterministic policy in the REAL env, recording obs and reward.
        obs_traj, rew_traj, done = [obs], [], False
        steps = 0
        while not done and steps < 1000:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            obs_traj.append(obs)
            rew_traj.append(float(r))
            done = term or trunc
            steps += 1
        rew_traj = np.array(rew_traj)
        # For a set of anchor states along the trajectory, compare the model's
        # predicted H-step return to the real discounted return actually obtained.
        anchors = range(0, max(len(rew_traj) - horizon, 1), max(horizon // 2, 1))
        gammas = gamma ** np.arange(horizon)
        for t in anchors:
            if t + horizon > len(rew_traj):
                break
            z0 = _encode_samples(agent, obs_traj[t], device, n=imag_samples)
            imag_disc, _ = imagine_return(agent, z0, horizon, gamma)
            real_disc = float(np.sum(gammas * rew_traj[t:t + horizon]))
            imagined.append(imag_disc)
            real.append(real_disc)
    imagined, real = np.array(imagined), np.array(real)
    gap = imagined - real
    print(f"    anchor states: {len(imagined)}")
    print(f"    imagined H-step return: mean={imagined.mean():+.3f}  std={imagined.std():.3f}")
    print(f"    real     H-step return: mean={real.mean():+.3f}  std={real.std():.3f}")
    print(f"    GAP (imagined - real) : mean={gap.mean():+.3f}  std={gap.std():.3f}")
    if real.std() > 1e-6 and imagined.std() > 1e-6:
        corr = np.corrcoef(imagined, real)[0, 1]
        print(f"    corr(imagined, real)  = {corr:+.3f}")
    else:
        corr = float("nan")
        print("    corr(imagined, real)  = n/a (a series is ~constant)")
    # Heuristic verdict.
    scale = max(abs(real.mean()), abs(imagined.mean()), 1.0)
    if gap.mean() > 0.5 * scale and (np.isnan(corr) or corr < 0.5):
        print("    WARNING: the model predicts substantially MORE return than the "
              "agent actually obtains, and the two are weakly correlated. SAC is "
              "optimising an imagined return that the real env does not deliver "
              "(model exploitation). Lower the horizon and/or correct rollouts "
              "with encoder posteriors.")
    elif abs(gap.mean()) <= 0.5 * scale and (not np.isnan(corr) and corr > 0.5):
        print("    OK: imagined and real returns are close and correlated — the "
              "imagined MDP is faithful at this horizon.")
    else:
        print("    NOTE: partial mismatch. Inspect probe [C] to see at what depth "
              "the imagined reward signal degrades.")


# ---------------------------------------------------------------------------
#  Probe C — reward-head accuracy vs imagination depth
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_reward_vs_depth(agent, env, episodes, horizon, device):
    print("\n[C] Reward-head error vs imagination depth (using REAL actions, so "
          "this isolates dynamics drift from policy choice):")
    # Accumulate squared errors per depth for two latent sources:
    #   enc[t]  : reward head on the encoder latent of the real obs at step t
    #             (in-distribution — what the head was trained on)
    #   imag[t] : reward head on the latent reached by rolling dynamics forward
    #             from step 0 with the real actions (what SAC sees at depth t)
    se_enc = [[] for _ in range(horizon)]
    se_imag = [[] for _ in range(horizon)]
    for ep in range(episodes):
        obs, _ = env.reset(seed=9000 + ep)
        obs_traj, act_traj, rew_traj, done, steps = [obs], [], [], False, 0
        while not done and steps < 1000:
            a = agent.act(obs, deterministic=True)
            obs, r, term, trunc, _ = env.step(a)
            obs_traj.append(obs)
            act_traj.append(a)
            rew_traj.append(float(r))
            done = term or trunc
            steps += 1
        if len(rew_traj) <= horizon:
            continue
        # Walk windows of length `horizon` across the trajectory.
        for t0 in range(0, len(rew_traj) - horizon, horizon):
            z_imag = _encode_samples(agent, obs_traj[t0], device, n=1)
            for d in range(horizon):
                a_d = _t(act_traj[t0 + d], device).unsqueeze(0)
                r_true = rew_traj[t0 + d]
                # in-distribution reward prediction
                z_enc = _encode_samples(agent, obs_traj[t0 + d], device, n=1)
                r_enc = agent.world_model.reward(z_enc, a_d).item()
                # imagined-latent reward prediction at depth d
                r_im = agent.world_model.reward(z_imag, a_d).item()
                se_enc[d].append((r_enc - r_true) ** 2)
                se_imag[d].append((r_im - r_true) ** 2)
                z_imag = agent.world_model.imagine(z_imag, a_d).rsample()
    print(f"    {'depth':>5}  {'RMSE(enc latent)':>16}  {'RMSE(imag latent)':>17}  "
          f"{'degradation':>11}")
    for d in range(horizon):
        if not se_enc[d]:
            continue
        rmse_enc = float(np.sqrt(np.mean(se_enc[d])))
        rmse_im = float(np.sqrt(np.mean(se_imag[d])))
        ratio = rmse_im / max(rmse_enc, 1e-8)
        print(f"    {d:>5}  {rmse_enc:>16.4f}  {rmse_im:>17.4f}  {ratio:>10.2f}x")
    print("    Read: column 1 (enc) should stay flat — that's the head on the "
          "latents it trained on. If column 2 (imag) climbs with depth, the "
          "rollout is feeding the reward head off-distribution latents, so the "
          "deep-horizon imagined reward is unreliable.")


# ---------------------------------------------------------------------------
#  Probe D — reward-head R^2 and mean-predictor baseline
# ---------------------------------------------------------------------------
@torch.no_grad()
def probe_reward_r2(agent, env, n_samples, device):
    print("\n[D] Reward-head R^2 on real transitions (is low MSE real signal or "
          "just mean-prediction?):")
    obs, _ = env.reset(seed=777)
    z_list, a_list, r_list = [], [], []
    for _ in range(n_samples):
        a = env.action_space.sample()
        z = agent.world_model.encode(_t(obs, device).unsqueeze(0)).rsample()
        nobs, r, term, trunc, _ = env.step(a)
        z_list.append(z)
        a_list.append(_t(a, device).unsqueeze(0))
        r_list.append(float(r))
        obs = nobs if not (term or trunc) else env.reset()[0]
    z = torch.cat(z_list, 0)
    a = torch.cat(a_list, 0)
    r_real = np.array(r_list)
    r_hat = agent.world_model.reward(z, a).cpu().numpy()
    var = r_real.var()
    mse = np.mean((r_hat - r_real) ** 2)
    r2 = 1.0 - mse / max(var, 1e-12)
    corr = np.corrcoef(r_hat, r_real)[0, 1] if r_real.std() > 1e-9 else float("nan")
    print(f"    samples={n_samples} (random actions)")
    print(f"    real reward: mean={r_real.mean():+.4f}  var={var:.4f}  std={r_real.std():.4f}")
    print(f"    reward MSE={mse:.4f}   ->   R^2={r2:+.3f}   corr={corr:+.3f}")
    if r2 < 0.2:
        print("    WARNING: R^2 near 0 — the head explains little reward variance. "
              "A low MSE here is mostly the head predicting the (nearly constant) "
              "mean reward of a non-moving cheetah, not learning what earns reward.")
    elif r2 < 0.6:
        print("    NOTE: moderate R^2 — the head has signal but is noisy.")
    else:
        print("    OK: the reward head explains most of the one-step reward variance.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to a final.zip checkpoint.")
    p.add_argument("--env", default="HalfCheetah-v5", help="Env id (match training).")
    p.add_argument("--episodes", type=int, default=5,
                   help="Eval/trajectory episodes for probes A-C.")
    p.add_argument("--horizon", type=int, default=None,
                   help="Imagination horizon. Default: the agent's trained horizon.")
    p.add_argument("--n-reward-samples", type=int, default=400,
                   help="Random transitions for the probe-D R^2 estimate.")
    p.add_argument("--imag-samples", type=int, default=8,
                   help="Stochastic dynamics rollouts averaged per anchor in probe B.")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # normalize_reward=False matches experiment_3 (rewards are raw); this keeps
    # the reward head's targets and our real-return comparison on the same scale.
    env = make_env(args.env, seed=args.seed, normalize_reward=False)
    agent = LatentSACAgent.load(
        args.ckpt, env=env, seed=args.seed, device=device,
        restore_env_stats=True, freeze_env_stats=True,
    )
    agent.actor.eval()
    agent.world_model.eval()

    horizon = args.horizon if args.horizon is not None else int(getattr(agent, "horizon", 15))
    gamma = float(getattr(agent, "gamma", 0.99))
    alpha = agent.log_alpha.exp().item()

    print(f"\n=== Imagination diagnostics for {Path(args.ckpt).parent.name} ===")
    print(f"    device={device}  horizon={horizon}  gamma={gamma}  "
          f"final alpha={alpha:.4f}  (target entropy={-agent.action_dim})\n")

    probe_eval(agent, env, args.episodes, device)
    probe_return_gap(agent, env, args.episodes, horizon, gamma, device, args.imag_samples)
    probe_reward_vs_depth(agent, env, args.episodes, horizon, device)
    probe_reward_r2(agent, env, args.n_reward_samples, device)

    env.close()
    print("\nDone. If [B] shows imagined >> real and [C] shows the imag-latent "
          "RMSE climbing with depth, the headline issue is long-horizon model "
          "exploitation, not the latent representation (re-run for both "
          "gaussian and categorical to confirm they share it).\n")


if __name__ == "__main__":
    main()
