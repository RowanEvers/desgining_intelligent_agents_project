"""Replay buffers.

Two buffers, two access patterns:

  EnvReplayBuffer
      Stores REAL transitions (obs, a, r, obs', done) collected from the
      environment. Numpy-backed because env interaction is one-step-at-a-time
      (slow path anyway) and numpy avoids holding GPU memory hostage for cold
      data we'll only sample occasionally. Used to train the WORLD MODEL.

  ImaginedBuffer
      Stores IMAGINED transitions (z, a, r_hat, z', cont_hat) produced in
      batches by rolling the dynamics forward H steps. Torch-backed and kept
      on-device because (a) it's written in big batches, (b) it's read every
      SAC update, and (c) it stores latent vectors not raw observations so
      memory is bounded. Used to train SAC.
"""

import numpy as np
import torch


class EnvReplayBuffer:
    """Numpy ring buffer for real env transitions."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device

        # Pre-allocate — avoids per-step allocation overhead during rollouts.
        self.obs_buf      = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.action_buf   = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward_buf   = np.zeros(capacity,               dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim),    dtype=np.float32)
        self.done_buf     = np.zeros(capacity,               dtype=np.float32)

        self.ptr = 0     # write index
        self.full = False  # have we wrapped around at least once?

    def add(self, obs, action, reward, next_obs, done) -> None:
        """Add a single real transition."""
        self.obs_buf[self.ptr]      = obs
        self.action_buf[self.ptr]   = action
        self.reward_buf[self.ptr]   = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr]     = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        if self.ptr == 0:
            self.full = True

    def sample(self, batch_size: int) -> dict:
        """Sample uniformly without replacement-style — np.random.randint is fast and ok with replacement."""
        max_idx = self.capacity if self.full else self.ptr
        idx = np.random.randint(0, max_idx, size=batch_size)
        # Push to device once per batch — cheap relative to the gradient step.
        return {
            'obs':      torch.as_tensor(self.obs_buf[idx],      device=self.device),
            'actions':  torch.as_tensor(self.action_buf[idx],   device=self.device),
            'rewards':  torch.as_tensor(self.reward_buf[idx],   device=self.device),
            'next_obs': torch.as_tensor(self.next_obs_buf[idx], device=self.device),
            'dones':    torch.as_tensor(self.done_buf[idx],     device=self.device),
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr


class ImaginedBuffer:
    """Torch ring buffer for imagined latent transitions, kept on `device`.

    Stored values are detached — we don't backprop through stored data; SAC
    treats imagined transitions just like real ones from its point of view.
    """

    def __init__(self, capacity: int, latent_dim: int, action_dim: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device

        # Allocate directly on device — saves an H2D copy per sample().
        self.z_buf       = torch.zeros((capacity, latent_dim), device=device)
        self.action_buf  = torch.zeros((capacity, action_dim), device=device)
        self.reward_buf  = torch.zeros(capacity,               device=device)
        self.next_z_buf  = torch.zeros((capacity, latent_dim), device=device)
        self.cont_buf    = torch.zeros(capacity,               device=device)

        self.ptr = 0
        self.full = False

    def add_batch(self, z, action, reward, next_z, cont) -> None:
        """Add a batch of imagined transitions at once.

        All inputs are torch tensors; we detach them so the buffer doesn't
        retain the imagination computation graph (would blow up memory).
        Handles wrap-around if the batch crosses the end of the buffer.
        """
        z       = z.detach()
        action  = action.detach()
        reward  = reward.detach()
        next_z  = next_z.detach()
        cont    = cont.detach()

        batch_size = z.shape[0]
        assert batch_size <= self.capacity, "Batch larger than buffer capacity"

        end = self.ptr + batch_size
        if end <= self.capacity:
            # No wrap — single contiguous write.
            self.z_buf[self.ptr:end]      = z
            self.action_buf[self.ptr:end] = action
            self.reward_buf[self.ptr:end] = reward
            self.next_z_buf[self.ptr:end] = next_z
            self.cont_buf[self.ptr:end]   = cont
            self.ptr = end % self.capacity
            if end == self.capacity:
                self.full = True
        else:
            # Wrap — two writes.
            first = self.capacity - self.ptr
            second = batch_size - first
            self.z_buf[self.ptr:]      = z[:first]
            self.action_buf[self.ptr:] = action[:first]
            self.reward_buf[self.ptr:] = reward[:first]
            self.next_z_buf[self.ptr:] = next_z[:first]
            self.cont_buf[self.ptr:]   = cont[:first]
            self.z_buf[:second]        = z[first:]
            self.action_buf[:second]   = action[first:]
            self.reward_buf[:second]   = reward[first:]
            self.next_z_buf[:second]   = next_z[first:]
            self.cont_buf[:second]     = cont[first:]
            self.ptr = second
            self.full = True

    def sample(self, batch_size: int) -> dict:
        max_idx = self.capacity if self.full else self.ptr
        # torch.randint stays on-device; no host roundtrip.
        idx = torch.randint(0, max_idx, (batch_size,), device=self.device)
        return {
            'z':       self.z_buf[idx],
            'actions': self.action_buf[idx],
            'rewards': self.reward_buf[idx],
            'next_z':  self.next_z_buf[idx],
            'cont':    self.cont_buf[idx],
        }

    def __len__(self) -> int:
        return self.capacity if self.full else self.ptr
