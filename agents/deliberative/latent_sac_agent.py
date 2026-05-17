"""LatentSACAgent — the main agent class.

Architecture
------------
    obs --[Encoder]--> z (continuous Gaussian latent)
                       |
                       |--[Decoder]--> obs_hat        (ELBO reconstruction)
                       |--[RewardHead]--> r_hat       (used in imagined rollouts)
                       |--[ContinueHead]--> cont_hat  (terminations in imagination)
                       |
                       v
                   [Actor] --> action  --[Dynamics]--> z'

Training loop (per outer iteration)
-----------------------------------
    1. Act in the real env for K steps using the encoder + actor.
       Push real transitions into EnvReplayBuffer.
    2. Train the world model for M gradient steps on EnvReplayBuffer batches
       (ELBO: recon + KL + reward + continue).
    3. Sample a batch of real obs, encode them, and roll the dynamics + actor
       forward H steps in latent space. Push imagined transitions into
       ImaginedBuffer.
    4. Train SAC for N gradient steps on ImaginedBuffer batches
       (Q-loss + actor-loss + entropy-temperature update).
    5. Polyak-update target critics. Log everything. Repeat.

The categorical agent (later) will subclass or re-use most of this — only the
world-model class swaps to a categorical variant.
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from agents.deliberative.world_model.world_model import GaussianWorldModel
from agents.deliberative.sac.actor import SquashedGaussianActor
from agents.deliberative.sac.critic import TwinCritic, polyak_update
from agents.deliberative.sac.replay import EnvReplayBuffer, ImaginedBuffer


class LatentSACAgent(BaseAgent):
    def __init__(self, env, seed, log_dir, logger, device,
                 world_model_cls=GaussianWorldModel,
                 world_model_kwargs=None,
                 **hparams):
        """Latent-world-model + SAC agent.

        Args:
            world_model_cls: class to instantiate as the world model. Defaults
                to GaussianWorldModel for backward compatibility. Swap to
                CategoricalWorldModel for the discrete-latent variant.
            world_model_kwargs: dict of world-model-specific kwargs (e.g.
                {'latent_dim': 32} for Gaussian or
                {'num_cat': 32, 'num_classes': 32} for Categorical).
                Common kwargs (obs_dim, action_dim, hidden_dim, kl_beta,
                reward_loss_scale, continue_loss_scale) are filled in
                automatically from `hparams` if not supplied here.
        """
        super().__init__(env, seed, log_dir, logger, device, **hparams)

        # ---- read hyperparameters with sensible defaults ----
        # latent_dim is a *fallback* default for the Gaussian agent only —
        # for the categorical agent, the effective latent dim is derived
        # from num_cat * num_classes and read off the world model after
        # construction (see "resolve latent_dim" block below).
        self.latent_dim         = hparams.get("latent_dim", 32)
        self.hidden_dim         = hparams.get("hidden_dim", 256)
        self.gamma              = hparams.get("gamma", 0.99)
        self.tau                = hparams.get("tau", 0.005)
        self.horizon            = hparams.get("horizon", 5)
        self.batch_size         = hparams.get("batch_size", 256)
        self.env_steps_per_iter = hparams.get("env_steps_per_iter", 1000)
        self.wm_updates_per_iter  = hparams.get("wm_updates_per_iter", 100)
        self.sac_updates_per_iter = hparams.get("sac_updates_per_iter", 100)
        self.warmup_steps       = hparams.get("warmup_steps", 1000)
        self.env_buffer_size    = hparams.get("env_buffer_size", 1_000_000)
        self.imag_buffer_size   = hparams.get("imag_buffer_size", 100_000)
        self.lr_world           = hparams.get("lr_world", 3e-4)
        self.lr_actor           = hparams.get("lr_actor", 3e-4)
        self.lr_critic          = hparams.get("lr_critic", 3e-4)
        self.lr_alpha           = hparams.get("lr_alpha", 3e-4)
        self.kl_beta            = hparams.get("kl_beta", 1.0)
        self.reward_loss_scale  = hparams.get("reward_loss_scale", 1.0)
        self.continue_loss_scale = hparams.get("continue_loss_scale", 1.0)

        # ---- dimensions from the env ----
        obs_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(np.prod(env.action_space.shape))
        action_low  = torch.as_tensor(env.action_space.low,  dtype=torch.float32)
        action_high = torch.as_tensor(env.action_space.high, dtype=torch.float32)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # ---- world model (pluggable: Gaussian or Categorical) ----
        # Build kwargs: shared hparams first, then overrides from caller.
        wm_kwargs = {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "hidden_dim": self.hidden_dim,
            "kl_beta": self.kl_beta,
            "reward_loss_scale": self.reward_loss_scale,
            "continue_loss_scale": self.continue_loss_scale,
        }
        # For Gaussian, supply latent_dim as a default if caller didn't.
        if world_model_cls is GaussianWorldModel:
            wm_kwargs.setdefault("latent_dim", self.latent_dim)
        # Caller-provided kwargs (e.g. num_cat/num_classes) override.
        wm_kwargs.update(world_model_kwargs or {})
        self.world_model = world_model_cls(**wm_kwargs).to(device)

        # ---- resolve latent_dim seen by downstream nets (actor/critic/buffer) ----
        # CategoricalWorldModel exposes `flat_dim` = num_cat * num_classes.
        # GaussianWorldModel stores latent_dim in its encoder.
        if hasattr(self.world_model, "flat_dim"):
            self.latent_dim = self.world_model.flat_dim
        elif hasattr(self.world_model, "encoder") and hasattr(self.world_model.encoder, "latent_dim"):
            self.latent_dim = self.world_model.encoder.latent_dim
        # else: leave self.latent_dim at whatever the caller passed.

        # ---- actor ----
        self.actor = SquashedGaussianActor(
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
            action_low=action_low,
            action_high=action_high,
        ).to(device)

        # ---- twin critic + frozen target ----
        self.critic = TwinCritic(
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            hidden_dim=self.hidden_dim,
        ).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        for p in self.target_critic.parameters():
            p.requires_grad = False  # target is updated by Polyak, never by gradients

        # ---- automatic entropy tuning ----
        # log_alpha is a learnable scalar; alpha = exp(log_alpha) is always > 0.
        # target_entropy is the canonical Haarnoja choice: -|A|.
        self.log_alpha = nn.Parameter(torch.zeros(1, device=device))
        self.target_entropy = -float(action_dim)

        # ---- optimisers ----
        self.wm_opt     = optim.Adam(self.world_model.parameters(), lr=self.lr_world)
        self.actor_opt  = optim.Adam(self.actor.parameters(),       lr=self.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(),      lr=self.lr_critic)
        self.alpha_opt  = optim.Adam([self.log_alpha],              lr=self.lr_alpha)

        # ---- replay buffers ----
        self.env_buffer = EnvReplayBuffer(
            capacity=self.env_buffer_size,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device=device,
        )
        self.imagined_buffer = ImaginedBuffer(
            capacity=self.imag_buffer_size,
            latent_dim=self.latent_dim,
            action_dim=action_dim,
            device=device,
        )

        self._env_step_counter = 0  # incremented as we collect env data

    # =====================================================================
    #  BaseAgent interface
    # =====================================================================
    def train(self, total_timesteps: int) -> None:
        """Outer training loop — see module docstring for the five steps."""
        obs, _ = self.env.reset(seed=self.seed)
        ep_return = 0.0
        ep_len = 0

        while self._env_step_counter < total_timesteps:
            # ---- Step 1: collect env data ----
            for _ in range(self.env_steps_per_iter):
                if self._env_step_counter < self.warmup_steps:
                    # Random exploration before the actor has any signal.
                    action = self.env.action_space.sample()
                else:
                    action = self.act(obs, deterministic=False)

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                # Store `terminated` (true end-of-episode) as the done flag —
                # `truncated` shouldn't bootstrap to zero, so we exclude it.
                self.env_buffer.add(obs, action, reward, next_obs, terminated)

                ep_return += float(reward)
                ep_len += 1
                self._env_step_counter += 1

                if terminated or truncated:
                    self.logger.log(self._env_step_counter, {
                        "episode_reward": ep_return,
                        "episode_length": ep_len,
                    })
                    obs, _ = self.env.reset()
                    ep_return = 0.0
                    ep_len = 0
                else:
                    obs = next_obs

                if self._env_step_counter >= total_timesteps:
                    break

            # No training until the warmup is over and the env buffer is non-trivial.
            if len(self.env_buffer) < max(self.warmup_steps, self.batch_size):
                continue

            # ---- Step 2: train world model ----
            wm_last = {}
            for _ in range(self.wm_updates_per_iter):
                batch = self.env_buffer.sample(self.batch_size)
                wm_last = self.world_model.train_step(batch, self.wm_opt)

            # ---- Step 3: imagine rollouts ----
            self._imagine_rollouts(self.horizon, self.batch_size)

            # SAC needs at least batch_size imagined transitions to train.
            if len(self.imagined_buffer) < self.batch_size:
                continue

            # ---- Step 4: SAC updates (+ polyak target update inside) ----
            sac_last = {}
            for _ in range(self.sac_updates_per_iter):
                batch = self.imagined_buffer.sample(self.batch_size)
                sac_last = self._sac_update(batch)

            # ---- Step 5: log everything ----
            self.logger.log(self._env_step_counter, {
                "world_model_loss": wm_last.get("total_loss", 0.0),
                **{f"wm/{k}": v for k, v in wm_last.items()},
                **{f"sac/{k}": v for k, v in sac_last.items()},
            })

    def act(self, obs, deterministic: bool = False):
        """Encode obs -> latent z, then sample (or take mean) from the actor."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_z = self.world_model.encode(obs_t)
            z = q_z.mean if deterministic else q_z.rsample()
            action, _ = self.actor(z, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def save(self, path: str) -> None:
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.data,
            "hparams": self.hparams,
            "env_step_counter": self._env_step_counter,
        }, path)

    @classmethod
    def load(cls, path, env):
        # NOTE: caller must provide seed/log_dir/logger/device; we rebuild from saved hparams.
        ckpt = torch.load(path, map_location="cpu")
        agent = cls(
            env=env, seed=0, log_dir="", logger=None, device="cpu", **ckpt["hparams"]
        )
        agent.world_model.load_state_dict(ckpt["world_model"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.target_critic.load_state_dict(ckpt["target_critic"])
        agent.log_alpha.data.copy_(ckpt["log_alpha"])
        agent._env_step_counter = ckpt.get("env_step_counter", 0)
        return agent

    # =====================================================================
    #  Internal helpers
    # =====================================================================
    def _imagine_rollouts(self, horizon: int, batch_size: int) -> None:
        """Encode a batch of real obs, then roll forward H steps in latent space.

        Produces `horizon * batch_size` imagined transitions per call. All
        outputs are detached inside ImaginedBuffer.add_batch — we don't
        backprop through imagination here (Dreamer would; we keep it simple).
        """
        if len(self.env_buffer) < batch_size:
            return  # nothing to imagine from yet

        with torch.no_grad():
            batch = self.env_buffer.sample(batch_size)
            z = self.world_model.encode(batch["obs"]).rsample()

            for _ in range(horizon):
                action, _ = self.actor(z)
                r_hat = self.world_model.reward(z, action)
                cont = torch.sigmoid(self.world_model.continue_prob(z))  # logit -> prob
                z_next = self.world_model.imagine(z, action).rsample()

                self.imagined_buffer.add_batch(z, action, r_hat, z_next, cont)
                z = z_next

    def _sac_update(self, batch: dict) -> dict:
        """One SAC gradient step on an imagined-transition batch.

        Critic loss : MSE(Q_i(z,a), y) for i in {1,2}, with
            y = r + gamma * cont * (min Q_target(z', a') - alpha * log pi(a'|z'))
        Actor loss  : alpha * log pi(a|z) - min Q(z, a)
        Alpha loss  : -log_alpha * (log_prob.detach() + target_entropy)
        """
        z       = batch["z"]
        action  = batch["actions"]
        reward  = batch["rewards"]
        next_z  = batch["next_z"]
        cont    = batch["cont"]

        # ---- Critic update ----
        with torch.no_grad():
            next_action, next_log_prob = self.actor(next_z)
            q_target = self.target_critic.q_min(next_z, next_action)
            target_value = q_target - self.log_alpha.exp() * next_log_prob
            y = reward + self.gamma * cont * target_value

        q1, q2 = self.critic(z, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # ---- Actor update ----
        new_action, log_prob = self.actor(z)
        q_pi = self.critic.q_min(z, new_action)
        actor_loss = (self.log_alpha.exp().detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ---- Entropy temperature (alpha) update ----
        # Auto-tunes alpha so that the policy's entropy is ~ target_entropy.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ---- Polyak target-critic update ----
        polyak_update(self.critic, self.target_critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "q_mean": q1.mean().item(),
            "log_prob_mean": log_prob.mean().item(),
        }
