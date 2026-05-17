"""Auxiliary prediction heads attached to the latent.

- Decoder:        z          -> reconstructed observation       (used in ELBO recon loss)
- Reward head:    (z, a)     -> predicted reward                (lets us reward imagined rollouts)
- Continue head:  z          -> probability the episode continues (lets us mask dones in imagination)

Each is a small MLP. They are deliberately kept simple — the heavy lifting is
done by the encoder and dynamics; these heads just ground the latent.
"""

import torch
import torch.nn as nn


class Decoder(nn.Module):
    """MLP z -> reconstructed observation."""

    def __init__(self, latent_dim, input_dim, hidden_dim = 256, num_layers = 2):
        super().__init__()
        # TODO: trunk + linear out to obs_dim
        self.linear_decode = nn.Linear(latent_dim, hidden_dim)
        self.linear_likelihood = nn.Linear(hidden_dim, input_dim)
        # TODO: For continuous obs we model recon as Normal with fixed unit variance,
        #       so we just predict the mean. Loss becomes MSE up to a constant.
        

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return predicted observation mean."""
        return self.linear_likelihood(torch.relu(self.linear_decode(z)))


class RewardHead(nn.Module):
    """MLP (z, a) -> predicted scalar reward."""

    def __init__(self, latent_dim, action_dim, hidden_dim = 256, num_layers = 2):
        super().__init__()
        # TODO: MLP outputting a single scalar
        self.linear_reward = nn.Linear(latent_dim + action_dim, hidden_dim)
        self.linear_output = nn.Linear(hidden_dim, 1)
        

    def forward(self, z, action):
        """Return predicted reward."""
        return self.linear_output(torch.relu(self.linear_reward(torch.cat([z, action], dim=-1)))).squeeze(-1)
        


class ContinueHead(nn.Module):
    """MLP z -> logit for P(not done). Lets imagined rollouts respect terminations."""

    def __init__(self, latent_dim: int, hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        # TODO: MLP outputting a single logit
        self.linear_continue = nn.Linear(latent_dim, hidden_dim)
        self.linear_output = nn.Linear(hidden_dim, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Return logit; caller applies sigmoid + Bernoulli when needed."""
        return self.linear_output(torch.relu(self.linear_continue(z))).squeeze(-1)