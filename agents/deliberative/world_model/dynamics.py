"""Gaussian latent dynamics.

p(z_{t+1} | z_t, a_t) = Normal(mu(z_t, a_t), sigma(z_t, a_t))

Lets us imagine forward in latent space without ever decoding back to raw
observations during rollouts. The categorical version will share this file's
interface but parameterise a Categorical instead.
"""
import torch
import torch.nn as nn


class GaussianDynamics(nn.Module):
    """MLP that predicts a Gaussian over the next latent given (z, a)."""

    def __init__(self, latent_dim, action_dim, hidden_dim = 256,
                 num_layers = 2, min_log_std = -5.0, max_log_std = 2.0,
                 predict_delta = True):
        super().__init__()
        # TODO: trunk MLP ((latent_dim + action_dim) -> hidden_dim, num_layers deep)
        trunk_layers = []
        in_features = latent_dim + action_dim
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_log_std = nn.Linear(hidden_dim, latent_dim)

        self.clamping = (min_log_std, max_log_std)
        self.predict_delta = predict_delta


    def forward(self, z, action):
        """Return Normal over z_{t+1} given current z and action."""
        # TODO: x = cat([z, action]); features = trunk(x)
        features = self.trunk(torch.cat([z, action], dim=-1))
        # TODO: delta_mu = mu_head(features); log_std = clamp(...)
        delta_mu = self.linear_mu(features)
        log_std = self.linear_log_std(features)
        log_std = torch.clamp(log_std, self.clamping[0], self.clamping[1])
        # TODO: mu = z + delta_mu if self.predict_delta else delta_mu
        mu = delta_mu + z if self.predict_delta else delta_mu
        # TODO: return Normal(mu, log_std.exp())
        return torch.distributions.Normal(mu, log_std.exp())

