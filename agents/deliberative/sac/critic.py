"""SAC twin Q-critic operating on (z, a).

Two independent Q networks — we take the minimum for the policy target to
combat overestimation bias (Twin Delayed DDPG / SAC trick). Each has a target
network updated via Polyak averaging (tau ~ 0.005).

Why two Q networks?
    A single Q net consistently overestimates value because the max/argmax in
    the target is biased upward by network noise. Taking the elementwise min
    of two independently-trained Q nets cancels most of that bias for free.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """Single Q(z, a) -> scalar."""

    def __init__(self, latent_dim, action_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        layers = []
        in_features = latent_dim + action_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        # Final linear projects hidden -> scalar Q value (no activation; values are unbounded).
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, z, action):
        x = torch.cat([z, action], dim=-1)
        # squeeze last dim so output is (batch,) rather than (batch, 1) —
        # matches the shape of reward/cont tensors from the replay buffer.
        return self.net(x).squeeze(-1)


class TwinCritic(nn.Module):
    """Pair of Q networks. forward() returns both; q_min() returns elementwise min."""

    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = QNetwork(latent_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(latent_dim, action_dim, hidden_dim)

    def forward(self, z, action):
        """Return (q1, q2) — used in the critic loss, which trains both."""
        return self.q1(z, action), self.q2(z, action)

    def q_min(self, z, action):
        """Elementwise min — used in the actor target and the bootstrapped Bellman target."""
        return torch.min(self.q1(z, action), self.q2(z, action))


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """target = (1 - tau) * target + tau * source, in place on `target`.

    Iterates both .parameters() and .buffers() so non-trainable state (e.g.
    BatchNorm running stats, if you ever add them) also tracks. For plain
    Linear/ReLU networks, only the params line matters.
    """
    for src_p, tgt_p in zip(source.parameters(), target.parameters()):
        tgt_p.data.mul_(1.0 - tau).add_(src_p.data, alpha=tau)
    for src_b, tgt_b in zip(source.buffers(), target.buffers()):
        tgt_b.data.mul_(1.0 - tau).add_(src_b.data, alpha=tau)
