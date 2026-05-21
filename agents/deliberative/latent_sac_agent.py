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
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.base import BaseAgent
from agents.deliberative.world_model.world_model import GaussianWorldModel
from agents.deliberative.world_model.world_model_categorical import CategoricalWorldModel
from agents.deliberative.sac.actor import SquashedGaussianActor
from agents.deliberative.sac.critic import TwinCritic, polyak_update
from agents.deliberative.sac.replay import EnvReplayBuffer, ImaginedBuffer
from environments.utils import extract_env_norm_stats, restore_env_norm_stats


# Registry used to round-trip world-model identity through a checkpoint.
# Saving the class itself in torch.save is brittle across refactors —
# we save the name and look it up on load.
WORLD_MODEL_REGISTRY = {
    "GaussianWorldModel": GaussianWorldModel,
    "CategoricalWorldModel": CategoricalWorldModel,
}


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
        # Stash so save() can round-trip identity + kwargs through the checkpoint.
        # Storing the resolved kwargs (including the auto-filled obs_dim etc.)
        # means load() reconstructs an architecturally identical world model
        # without needing the env to expose the same shapes.
        self._world_model_cls_name = type(self.world_model).__name__
        self._world_model_kwargs = dict(wm_kwargs)

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
        episode_count = 0
        ep_start_time = time.time()

        while self._env_step_counter < total_timesteps:
            # ---- Step 1: collect env data ----
            for _ in range(self.env_steps_per_iter):
                if self._env_step_counter < self.warmup_steps:
                    # Random exploration before the actor has any signal.
                    action = self.env.action_space.sample()
                else:
                    action = self.act(obs, deterministic=False)

                next_obs, reward, terminated, truncated, info = self.env.step(action)
                # Store `terminated` (true end-of-episode) as the done flag —
                # `truncated` shouldn't bootstrap to zero, so we exclude it.
                self.env_buffer.add(obs, action, reward, next_obs, terminated)

                ep_return += float(reward)
                ep_len += 1
                self._env_step_counter += 1

                if terminated or truncated:
                    episode_count += 1
                    log_dict = {
                        "episode_reward": ep_return,         # normalized — SAC's update signal
                        "episode_length": ep_len,
                    }
                    # When RecordEpisodeStatistics is in the wrapper chain
                    # (the make_env default), info["episode"]["r"] is the
                    # RAW episodic return — log it so the training curve is
                    # interpretable without re-running eval.
                    ep_info = info.get("episode") if isinstance(info, dict) else None
                    if ep_info is not None and "r" in ep_info:
                        log_dict["episode_reward_raw"] = float(ep_info["r"])
                    self.logger.log(self._env_step_counter, log_dict)
                    # Console output — matches LoggerCallback in reactive/sac.py
                    # so latent and reactive runs look the same in the terminal.
                    raw_str = (f"  reward(raw): {log_dict['episode_reward_raw']:.1f}"
                               if "episode_reward_raw" in log_dict else "")
                    print(
                        f"[{self._env_step_counter:>7} / {total_timesteps}]"
                        f"  ep {episode_count:>3}"
                        f"  reward(norm): {ep_return:.3f}"
                        f"{raw_str}"
                        f"  len: {ep_len}"
                        f"  time: {time.time() - ep_start_time:.2f}s"
                    )
                    obs, _ = self.env.reset()
                    ep_return = 0.0
                    ep_len = 0
                    ep_start_time = time.time()
                else:
                    obs = next_obs

                if self._env_step_counter >= total_timesteps:
                    break

            # No training until the env buffer has enough transitions to sample
            # a batch.  During initial training warmup_steps > batch_size, but
            # during adaptation the counter starts at warmup_steps so we only
            # need batch_size samples before updates begin.
            if len(self.env_buffer) < self.batch_size:
                continue

            # ---- Step 2: train world model (skip when frozen) ----
            # A frozen WM has requires_grad=False on every parameter, so its
            # train_step would call backward() on a loss with no grad_fn and
            # raise. Skipping the update is also the correct semantics for the
            # "frozen" adaptation probe: imagine with the source WM held fixed.
            wm_last = {}
            wm_trainable = any(p.requires_grad for p in self.world_model.parameters())
            if wm_trainable:
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
        """Persist enough to reconstruct this exact agent later.

        Includes:
          - world-model class *name* + its constructor kwargs (Problem A)
          - env normalisation running stats (Problem B), if present
          - all network state dicts, log_alpha, hparams, env step counter
        """
        torch.save({
            "world_model": self.world_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "log_alpha": self.log_alpha.data,
            "hparams": self.hparams,
            "env_step_counter": self._env_step_counter,
            # --- Problem A: world-model identity ---
            "world_model_cls_name": self._world_model_cls_name,
            "world_model_kwargs": self._world_model_kwargs,
            # --- Problem B: env normalisation stats ---
            "env_norm_stats": extract_env_norm_stats(self.env),
        }, path)

    @classmethod
    def load(cls, path, env, seed=0, log_dir="", logger=None, device="cpu",
             restore_env_stats=True, freeze_env_stats=False):
        """Reconstruct a LatentSACAgent from a checkpoint.

        Args:
            path: checkpoint path.
            env: a fresh env instance (already wrapped the same way training
                used it — i.e. via ``make_env``). Its normalisation stats
                will be overwritten from the checkpoint if ``restore_env_stats``
                is True.
            seed, log_dir, logger, device: standard agent kwargs — needed if
                you want to continue training or run experiment 3 adaptation.
            restore_env_stats: if True (default), copy saved obs_rms /
                return_rms onto the wrappers in ``env``.
            freeze_env_stats: if True, freeze the running stats so further
                steps in ``env`` do not update them. Use this for adaptation
                (experiment 3) — keeps the agent's expected input
                distribution stable.
        """
        # weights_only=False: our checkpoint legitimately contains non-tensor
        # objects (numpy arrays in env_norm_stats, dict of hparams). PyTorch
        # >=2.6 defaults this to True and refuses to unpickle them. Safe to
        # disable because we're loading a checkpoint *we* produced — never
        # one from an untrusted source.
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # --- Problem A: round-trip world-model identity through the registry ---
        wm_cls_name = ckpt.get("world_model_cls_name", "GaussianWorldModel")
        if wm_cls_name not in WORLD_MODEL_REGISTRY:
            raise ValueError(
                f"Checkpoint references unknown world_model_cls '{wm_cls_name}'. "
                f"Known: {list(WORLD_MODEL_REGISTRY)}"
            )
        wm_cls = WORLD_MODEL_REGISTRY[wm_cls_name]
        wm_kwargs = ckpt.get("world_model_kwargs", None)

        agent = cls(
            env=env, seed=seed, log_dir=log_dir, logger=logger, device=device,
            world_model_cls=wm_cls,
            world_model_kwargs=wm_kwargs,
            **ckpt["hparams"],
        )
        agent.world_model.load_state_dict(ckpt["world_model"])
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.target_critic.load_state_dict(ckpt["target_critic"])
        agent.log_alpha.data.copy_(ckpt["log_alpha"])
        agent._env_step_counter = ckpt.get("env_step_counter", 0)

        # --- Problem B: restore env normalisation stats ---
        if restore_env_stats:
            restore_env_norm_stats(env, ckpt.get("env_norm_stats", {}),
                                   freeze=freeze_env_stats)

        return agent

    # =====================================================================
    #  Adaptation hook (experiment 3)
    # =====================================================================
    def prepare_for_adaptation(self, freeze_world_model: bool = True,
                               reset_step_counter: bool = True,
                               clear_buffers: bool = True) -> None:
        """Prep a loaded agent for experiment-3 style adaptation to a new env.

        - ``freeze_world_model``: turn off grads on the world model so only
          the actor + critic adapt. This is the standard "is the learned
          representation transferable?" probe.
        - ``reset_step_counter``: zero the env-step axis so adaptation
          curves start at 0 rather than wherever source training ended.
        - ``clear_buffers``: wipe the env and imagined replay buffers so
          source-task transitions don't pollute the target task. Almost
          always what you want — leave on unless you have a reason.
        """
        if freeze_world_model:
            for p in self.world_model.parameters():
                p.requires_grad = False
            self.world_model.eval()

        if reset_step_counter:
            # Jump straight past the warmup threshold so the trained policy
            # acts immediately — random-action warmup is only meaningful when
            # training from scratch, not when adapting a pre-trained agent.
            self._env_step_counter = self.warmup_steps

        if clear_buffers:
            # Re-create the buffers cleanly rather than mutating internals.
            self.env_buffer = type(self.env_buffer)(
                capacity=self.env_buffer_size,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )
            self.imagined_buffer = type(self.imagined_buffer)(
                capacity=self.imag_buffer_size,
                latent_dim=self.latent_dim,
                action_dim=self.action_dim,
                device=self.device,
            )

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
