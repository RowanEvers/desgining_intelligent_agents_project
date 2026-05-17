"""GaussianWorldModel — bundles encoder + dynamics + decoder + reward + continue.

Exposes:
  - encode(obs)                  -> Normal over z
  - imagine(z, action)           -> Normal over z'
  - decode(z)                    -> obs_hat
  - reward(z, a)                 -> r_hat
  - continue_prob(z)             -> P(not done)
  - train_step(batch)            -> dict of loss components

The training objective is the standard variational world-model ELBO:

    L = recon_loss + beta * KL[q(z|s) || p(z|z_prev,a_prev)]
        + reward_loss + continue_loss

For the very first step in a sequence there's no prior dynamics output, so we
fall back to a standard Normal(0, I) prior on z_0.

The categorical variant will live in a sibling file with the same public API,
swapping Normal for OneHotCategoricalStraightThrough.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import GaussianEncoder
from .dynamics import GaussianDynamics
from .heads import Decoder, RewardHead, ContinueHead    

class GaussianWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, latent_dim = 32,
                 hidden_dim = 256, kl_beta = 1.0,
                 reward_loss_scale = 1.0, continue_loss_scale = 1.0):
        super().__init__()
        # TODO: instantiate encoder, dynamics, decoder, reward_head, continue_head
        self.encoder = GaussianEncoder(obs_dim, latent_dim, hidden_dim)
        self.dynamics = GaussianDynamics(latent_dim, action_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, obs_dim, hidden_dim)
        self.reward_head = RewardHead(latent_dim, action_dim, hidden_dim)
        self.continue_head = ContinueHead(latent_dim, hidden_dim)

        # TODO: store loss-scaling hyperparameters
        self.kl_beta = kl_beta
        self.reward_loss_scale = reward_loss_scale
        self.continue_loss_scale = continue_loss_scale
        

    def encode(self, obs):
        """q(z | s)."""
        return self.encoder(obs)

    def imagine(self, z, action):
        """p(z' | z, a)."""
        return self.dynamics(z, action)

    def decode(self, z):
        return self.decoder(z)

    def reward(self, z, action):
        return self.reward_head(z, action)

    def continue_prob(self, z):
        return self.continue_head(z)

    def train_step(self, batch, optimizer):
        """One ELBO update.

        Expected batch keys: 'obs', 'actions', 'rewards', 'next_obs', 'dones'.
        All tensors are shape (batch, seq_len, ...) once we sequence-batch — for
        the first version a single-step batch (seq_len=1) is fine and simpler.

        Returns a dict of {loss_name: scalar} for logging.
        """
        # TODO: encode observation into the latent distribution, sample z from the distribution 
        q_z = self.encoder(batch['obs'])
        z = q_z.rsample()
        # TODO: encode next observation into q(z_next | obs_next), sample z_next from the distribution
        q_z_next = self.encoder(batch['next_obs'])
        # TODO: dynamics(z, a) -> prior_z_next (Normal distribution over next z given current z and action)
        prior_z_next = self.dynamics(z, batch['actions'])

        # calculating losses:
        recon_loss    = ((self.decoder(z) - batch['obs'])**2).sum(-1).mean()
        kl_loss       = torch.distributions.kl_divergence(q_z_next, prior_z_next).sum(-1).mean()
        reward_loss   = ((self.reward_head(z, batch['actions']) - batch['rewards'])**2).mean()
        continue_loss = F.binary_cross_entropy_with_logits(
                            self.continue_head(z), 1 - batch['dones'].float()).mean()
        total = recon_loss + self.kl_beta * kl_loss + self.reward_loss_scale * reward_loss + self.continue_loss_scale * continue_loss
        # TODO: optimizer.zero_grad(); total.backward(); optimizer.step()
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        # TODO: return all scalars in a dict
        return {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'total_loss': total.item(),
        }

