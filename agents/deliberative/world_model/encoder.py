"""Gaussian observation encoder.

q(z | s) = Normal(mu(s), sigma(s))

For the continuous (Gaussian) world-model agent, this maps a raw observation
into the parameters of a diagonal Gaussian over the latent z. The categorical
variant will live alongside this file and have the same interface but produce
logits over discrete categories instead.

The interface is intentionally minimal: forward(obs) -> Distribution-like object
with .rsample(), .mean, and a method to compute KL against a prior.
"""

import torch
import torch.nn as nn

class GaussianEncoder(nn.Module):
    """MLP that outputs (mu, log_std) of a diagonal Gaussian latent."""

    def __init__(self, input_dim, latent_dim, hidden_dim = 256,
                 num_layers = 2, min_log_std = -5.0, max_log_std = 2.0):
        
        super().__init__()
        # TODO: build trunk MLP (obs_dim -> hidden_dim, num_layers deep, ReLU)
        
        self.input_dim = input_dim # input dim depends on dataset
        self.hidden_dim = hidden_dim # hidden dim for first linear layer
        self.latent_dim = latent_dim # dimension in latent space

        trunk_layers = []
        in_features = input_dim
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim
        
        self.trunk = nn.Sequential(*trunk_layers)

        self.linear_mu = nn.Linear(self.hidden_dim, self.latent_dim)
        self.linear_log_std = nn.Linear(self.hidden_dim, self.latent_dim)

        # TODO: build heads: mu_head (hidden -> latent_dim), log_std_head (hidden -> latent_dim)
        # TODO: store min_log_std / max_log_std for clamping
        self.clamping = (min_log_std, max_log_std)


    def forward(self, obs):
        features = self.trunk(obs)                                     
        mu = self.linear_mu(features)
        log_std = torch.clamp(self.linear_log_std(features),
                            self.clamping[0], self.clamping[1])
        return torch.distributions.Normal(mu, log_std.exp())
        
