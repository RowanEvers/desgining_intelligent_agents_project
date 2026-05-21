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
