"""SAC actor: squashed-Gaussian policy over actions given latent z.

pi(a | z) = tanh( Normal(mu(z), sigma(z)) ),   action bounds applied afterwards.

The tanh squash means we need the log-prob correction:
    log pi(a|z) = log N(u|mu, sigma) - sum log(1 - tanh(u)^2 + eps)
where u is the pre-tanh sample.
"""
import torch
import torch.nn as nn

class SquashedGaussianActor(nn.Module):
    def __init__(self, latent_dim, action_dim, hidden_dim=256,
                 action_low=None, action_high=None,
                 min_log_std=-20.0, max_log_std=2.0):
        super().__init__()
        # Trunk MLP: latent_dim -> hidden_dim, 2 hidden layers, ReLU
        trunk_layers = []
        in_features = latent_dim
        for _ in range(2):
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # mu and log_std heads
        self.linear_mu = nn.Linear(hidden_dim, action_dim)
        self.linear_log_std = nn.Linear(hidden_dim, action_dim)

        # Always register action bounds — default to [-1, 1] if env doesn't specify.
        # Using buffers means they move with .to(device) and are saved with state_dict.
        if action_low is None:
            action_low = torch.full((action_dim,), -1.0)
        if action_high is None:
            action_high = torch.full((action_dim,), 1.0)
        self.register_buffer('action_low', action_low.float())
        self.register_buffer('action_high', action_high.float())

        # Precompute rescaling constants (also buffers so device-correct).
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2.0)
        self.register_buffer('action_bias',  (self.action_high + self.action_low) / 2.0)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, z: torch.Tensor, deterministic: bool = False):
        """Return (action, log_prob).

        rsample() is reparameterised so gradients flow from log_prob and the
        critic value Q(z, a) back into the actor parameters during the SAC
        actor update.
        """
        features = self.trunk(z)
        mu = self.linear_mu(features)
        log_std = torch.clamp(self.linear_log_std(features),
                              min=self.min_log_std, max=self.max_log_std)
        dist = torch.distributions.Normal(mu, log_std.exp())

        u = mu if deterministic else dist.rsample()
        a_squashed = torch.tanh(u)

        # Tanh-squash log-prob correction.
        log_prob = dist.log_prob(u) - torch.log(1 - a_squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)

        # Linear rescale from [-1, 1] to [action_low, action_high].
        action = a_squashed * self.action_scale + self.action_bias

        # Jacobian correction for the linear rescale: log|det(d action / d a_squashed)| = sum(log scale).
        # If action_scale == 1 (e.g. action range [-1, 1]) this term is 0.
        log_prob = log_prob - torch.log(self.action_scale).sum()

        return action, log_prob
