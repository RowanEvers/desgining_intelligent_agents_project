"""Categorical latent dynamics — the discrete counterpart to dynamics.py.

p(z_{t+1} | z_t, a_t) is a stack of `num_cat` categorical distributions,
exactly matching the structure produced by the CategoricalEncoder so that
KL(q || p) is well-defined between them.

Notes on what does NOT translate from the Gaussian version
----------------------------------------------------------
- `predict_delta` is omitted. The Gaussian dynamics learned a residual on
  the mean, which made sense because z_t and z_{t+1} live in the same real
  vector space and tend to be close. For a categorical latent, "the delta"
  doesn't have a natural meaning (one-hot vectors don't add residually). We
  predict the next-step logits outright.

- `log_std` clamping is irrelevant — categoricals don't have a scale
  parameter. The only knob that affects sharpness is the logit magnitude,
  which the network learns freely.

Notes on what DOES translate
----------------------------
- The trunk is identical: an MLP from (z_flat + action) into hidden_dim.
- The output head is also a single Linear, just projecting to
  num_cat * num_classes instead of 2 * latent_dim (mu + log_std).
- The forward() shape contract matches the encoder so KL between two
  dynamics-or-encoder distributions is straightforward.

Input convention
----------------
The dynamics receives the latent as a FLAT vector (batch, num_cat*num_classes).
This is because the rest of the system (decoder, reward head, continue head,
actor, critic) all see the latent flat — so we keep the flat representation
as the "canonical" interface and only reshape internally where needed.
"""

import torch
import torch.nn as nn


class CategoricalDynamics(nn.Module):
    """Predicts the next categorical latent given a flat current latent and an action."""

    def __init__(self, flat_latent_dim, num_cat, num_classes, action_dim,
                 hidden_dim=256, num_layers=2):
        super().__init__()
        self.num_cat = num_cat
        self.num_classes = num_classes

        # Sanity check that the flat dim and the (cat, classes) factorisation agree.
        assert flat_latent_dim == num_cat * num_classes, (
            f"flat_latent_dim ({flat_latent_dim}) must equal "
            f"num_cat * num_classes ({num_cat * num_classes})"
        )

        # ---- trunk: (flat_latent_dim + action_dim) -> hidden_dim ----
        trunk_layers = []
        in_features = flat_latent_dim + action_dim
        for _ in range(num_layers):
            trunk_layers.append(nn.Linear(in_features, hidden_dim))
            trunk_layers.append(nn.ReLU())
            in_features = hidden_dim
        self.trunk = nn.Sequential(*trunk_layers)

        # ---- logits head: hidden_dim -> num_cat * num_classes ----
        self.logits_head = nn.Linear(hidden_dim, num_cat * num_classes)

    def forward(self, z_flat, action):
        """Return a OneHotCategoricalStraightThrough over the next latent.

        Shapes:
            z_flat   : (batch, num_cat * num_classes)   ← flat one-hot stack
            action   : (batch, action_dim)
            features : (batch, hidden_dim)
            logits   : (batch, num_cat, num_classes)
        """
        x = torch.cat([z_flat, action], dim=-1)
        features = self.trunk(x)
        logits = self.logits_head(features)
        logits = logits.view(*logits.shape[:-1], self.num_cat, self.num_classes)
        return torch.distributions.OneHotCategoricalStraightThrough(logits=logits)
