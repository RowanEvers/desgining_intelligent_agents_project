import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoder_categorical import CategoricalEncoder
from .dynamics_categorical import CategoricalDynamics
from .heads import Decoder, RewardHead, ContinueHead


class _FlatLatentDist:
    """Wraps a categorical-per-site distribution so its samples come out flat.

    The agent code (`world_model.encode(obs).rsample()`) expects a flat
    tensor; this wrapper preserves that interface. The raw distribution is
    kept as `.inner` so that `train_step` can still call
    `kl_divergence(inner_q, inner_p)` directly on the proper categorical
    objects.
    """

    def __init__(self, inner: torch.distributions.Distribution):
        self.inner = inner

    def rsample(self, sample_shape=torch.Size()):
        # Sample is (..., num_cat, num_classes) one-hot; flatten last two dims.
        sample = self.inner.rsample(sample_shape)
        return sample.flatten(start_dim=-2)

    @property
    def mean(self):
        # For deterministic evaluation we use the per-site probability vector
        # (concatenated softmaxes) as the "mean" — this is the natural analog
        # of the Gaussian's mean for downstream networks that just want a
        # smooth representation. It's not a sample so it doesn't carry
        # gradient through any sampling op, which is fine at eval time.
        return self.inner.probs.flatten(start_dim=-2)


class CategoricalWorldModel(nn.Module):
    def __init__(self, obs_dim, action_dim, num_cat=32, num_classes=32,
                 hidden_dim=256, kl_beta=1.0,
                 reward_loss_scale=1.0, continue_loss_scale=1.0):
        super().__init__()
        self.num_cat = num_cat
        self.num_classes = num_classes
        # The "effective" latent dim seen by every downstream module.
        self.flat_dim = num_cat * num_classes

        # ---- world-model components ----
        # Encoder and dynamics live in (num_cat, num_classes) space.
        self.encoder = CategoricalEncoder(
            input_dim=obs_dim, num_cat=num_cat, num_classes=num_classes,
            hidden_dim=hidden_dim,
        )
        self.dynamics = CategoricalDynamics(
            flat_latent_dim=self.flat_dim,
            num_cat=num_cat, num_classes=num_classes,
            action_dim=action_dim, hidden_dim=hidden_dim,
        )

        # Decoder / reward / continue all see the FLAT latent.
        # They're literally the same classes used by the Gaussian world model.
        self.decoder       = Decoder(self.flat_dim, obs_dim, hidden_dim)
        self.reward_head   = RewardHead(self.flat_dim, action_dim, hidden_dim)
        self.continue_head = ContinueHead(self.flat_dim, hidden_dim)

        # ---- loss-scaling hyperparameters (identical names + meanings to Gaussian) ----
        self.kl_beta = kl_beta
        self.reward_loss_scale = reward_loss_scale
        self.continue_loss_scale = continue_loss_scale

    # =====================================================================
    #  Public interface — matches GaussianWorldModel exactly so LatentSACAgent
    #  can be reused as-is.
    # =====================================================================
    def encode(self, obs):
        """q(z | s), wrapped so .rsample() returns a flat latent."""
        return _FlatLatentDist(self.encoder(obs))

    def imagine(self, z_flat, action):
        """p(z' | z, a), wrapped so .rsample() returns a flat latent."""
        return _FlatLatentDist(self.dynamics(z_flat, action))

    def decode(self, z_flat):
        return self.decoder(z_flat)

    def reward(self, z_flat, action):
        return self.reward_head(z_flat, action)

    def continue_prob(self, z_flat):
        return self.continue_head(z_flat)

    # =====================================================================
    #  ELBO training step — same five losses, just different KL family.
    # =====================================================================
    def train_step(self, batch, optimizer):
        """One ELBO update.

        The five losses are identical in *intent* to the Gaussian version:
            recon       — z must reconstruct s
            kl          — encoder posterior must match dynamics prior
            reward      — z, a must predict r
            continue    — z must predict episode termination

        What's different:
            The KL is now between two categorical-per-site distributions.
            `torch.distributions.kl_divergence` has a registered handler
            for OneHotCategorical so this just works. The returned KL has
            shape (batch, num_cat) — we sum over the num_cat dim (each
            site contributes independently) and mean over batch.
        """
        # ---- encode current obs (raw distribution; we'll need inner for KL) ----
        q_z_dist = self.encoder(batch['obs'])                  # raw OneHotCategoricalStraightThrough
        z = q_z_dist.rsample().flatten(start_dim=-2)           # (batch, flat_dim)

        # ---- encode next obs and predict next-latent prior from the dynamics ----
        q_z_next_dist  = self.encoder(batch['next_obs'])       # raw
        prior_z_next   = self.dynamics(z, batch['actions'])    # raw — takes FLAT z

        # ---- losses ----
        recon_loss = ((self.decoder(z) - batch['obs']) ** 2).sum(-1).mean()

        # KL shape note:
        # Both distributions have batch_shape=(batch, num_cat), event_shape=(num_classes,).
        # kl_divergence returns shape (batch, num_cat) — one KL per categorical site.
        # We sum over sites (independent factorisation) and mean over batch.
        kl_loss = torch.distributions.kl_divergence(
            q_z_next_dist, prior_z_next
        ).sum(-1).mean()

        reward_loss = ((self.reward_head(z, batch['actions']) - batch['rewards']) ** 2).mean()
        continue_loss = F.binary_cross_entropy_with_logits(
            self.continue_head(z), 1 - batch['dones'].float()
        ).mean()

        total = (
            recon_loss
            + self.kl_beta * kl_loss
            + self.reward_loss_scale * reward_loss
            + self.continue_loss_scale * continue_loss
        )

        optimizer.zero_grad()
        total.backward()
        optimizer.step()

        return {
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item(),
            'continue_loss': continue_loss.item(),
            'total_loss': total.item(),
        }
