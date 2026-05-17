"""Categorical observation encoder — the discrete counterpart to encoder.py.

q(z | s) is a stack of `num_cat` independent categorical distributions, each
over `num_classes` mutually exclusive options. A sample is therefore
`num_cat` one-hot vectors of length `num_classes`, stacked along a new axis.

Why categorical instead of Gaussian?
    The Hafner et al. Dreamer V2/V3 result is that discrete latents are
    empirically better behaved than Gaussian latents in many tasks — they
    don't collapse to a noise distribution, the policy can act on a
    sparse/sharp representation, and the dynamics can model multimodal
    futures naturally (the categorical can put mass on two distinct outcomes
    where a unimodal Gaussian would smear between them).

Why `num_cat` × `num_classes` and not just one big categorical?
    A single 1024-way categorical has the same information content as
    32 × 32, but training a 1024-way softmax is unstable and concentrates
    gradient on one site. The "K small categoricals" factorisation gives
    the same representational capacity with much better-behaved gradients
    and is what Dreamer V2 actually uses.

Why straight-through?
    The `OneHotCategoricalStraightThrough` distribution samples discretely
    on the forward pass (you get an actual one-hot vector) but treats the
    sample as if it were the continuous softmax probabilities on the
    backward pass. Without this trick, the discrete sampling step would have
    no gradient and the encoder would never learn.
"""

import torch
import torch.nn as nn


class CategoricalEncoder(nn.Module):
    """MLP that outputs logits for a stack of `num_cat` categorical distributions."""

    def __init__(self, input_dim, num_cat=32, num_classes=32,
                 hidden_dim=256, num_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.num_cat = num_cat
        self.num_classes = num_classes
        # Convenience: the FLATTENED latent dimension that downstream
        # modules (decoder, reward, continue, actor, critic) will see.
        # Whoever instantiates downstream nets should use this value.
        self.flat_dim = num_cat * num_classes

        # ---- shared trunk (identical pattern to GaussianEncoder) ----
        trunk_layers = []
        in_features = input_dim
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # ---- logits head ----
        # One Linear projects to ALL num_cat * num_classes logits at once.
        # We reshape into (..., num_cat, num_classes) in forward().
        self.logits_head = nn.Linear(hidden_dim, num_cat * num_classes)

    def forward(self, obs):
        """Return a OneHotCategoricalStraightThrough over (num_cat, num_classes).

        Shapes:
            obs        : (batch, input_dim)
            features   : (batch, hidden_dim)
            logits raw : (batch, num_cat * num_classes)
            logits     : (batch, num_cat, num_classes)

        The returned distribution has:
            batch_shape = (batch, num_cat)
            event_shape = (num_classes,)
        A sample via .rsample() will have shape (batch, num_cat, num_classes)
        and consist of one-hot vectors along the last dim.
        """
        features = self.trunk(obs)
        logits = self.logits_head(features)
        # Reshape so each of the `num_cat` sites gets its own logits row.
        logits = logits.view(*logits.shape[:-1], self.num_cat, self.num_classes)

        # NOTE: Dreamer V2 also adds a small "unimix" mixture with a uniform
        # distribution here (typically eps=0.01) to prevent dead categories:
        #     probs = (1 - eps) * softmax(logits) + eps / num_classes
        # Skipped here for simplicity. If KL collapses to zero or the model
        # gets stuck early in training, add it back.
        return torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
