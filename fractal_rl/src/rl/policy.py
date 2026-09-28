"""CNN actor-critic with tanh-squashed Gaussian actions and exact log-probability."""
import numpy as np
import torch
from torch import nn
import copy
from torch.distributions import Normal
from torch.nn import functional as F


class ActorCritic(nn.Module):
    def __init__(self, fov_size=32, state_dim=6, architecture='shared', jepa_enabled=False):
        super().__init__()
        if architecture not in {'shared', 'unshared'}:
            raise ValueError("architecture must be 'shared' or 'unshared'")
        self.architecture = architecture

        def make_cnn():
            return nn.Sequential(nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(), nn.Flatten())

        def make_encoder(channels):
            return nn.Sequential(nn.Linear(channels + state_dim, 128), nn.Tanh(),
                                 nn.Linear(128, 128), nn.Tanh())

        # Keep the original module names in the shared variant: old checkpoints
        # remain loadable, while unshared is a directly comparable ablation.
        self.cnn = make_cnn()
        with torch.no_grad():
            channels = self.cnn(torch.zeros(1, 1, fov_size, fov_size)).shape[-1]
        self.encoder = make_encoder(channels)
        if architecture == 'unshared':
            self.critic_cnn = make_cnn()
            self.critic_encoder = make_encoder(channels)
        self.actor = nn.Linear(128, 2)
        self.critic = nn.Linear(128, 1)
        self.log_std = nn.Parameter(torch.full((2,), -0.5))
        self.jepa_enabled = jepa_enabled
        if jepa_enabled:
            # Target parameters are EMA-updated and never receive gradients.
            self.target_cnn = copy.deepcopy(self.cnn)
            self.target_encoder = copy.deepcopy(self.encoder)
            for parameter in self.target_cnn.parameters():
                parameter.requires_grad_(False)
            for parameter in self.target_encoder.parameters():
                parameter.requires_grad_(False)
            self.jepa_predictor = nn.Sequential(nn.Linear(128 + 2, 128), nn.Tanh(), nn.Linear(128, 128))

    def encode(self, terrain, state):
        """Return the actor representation (shared representation when selected)."""
        return self.encoder(torch.cat([self.cnn(terrain), state], dim=-1))

    def encode_critic(self, terrain, state):
        if self.architecture == 'shared':
            return self.encode(terrain, state)
        return self.critic_encoder(torch.cat([self.critic_cnn(terrain), state], dim=-1))

    def target_encode(self, terrain, state):
        if not self.jepa_enabled:
            raise RuntimeError('JEPA target encoder is disabled')
        return self.target_encoder(torch.cat([self.target_cnn(terrain), state], dim=-1))

    def jepa_loss(self, terrain, state, action, next_terrain, next_state,
                  embedding=None):
        """Scale-free action-conditioned prediction of the next target latent."""
        if not self.jepa_enabled:
            return terrain.new_zeros(())
        embedding = self.encode(terrain, state) if embedding is None else embedding
        prediction = self.jepa_predictor(torch.cat([embedding, action.flatten(1)], dim=-1))
        with torch.no_grad():
            target = self.target_encode(next_terrain, next_state)
        return (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()

    @torch.no_grad()
    def update_jepa_target(self, tau):
        if not self.jepa_enabled:
            return
        if not 0 < tau <= 1:
            raise ValueError('jepa_target_tau must be in (0, 1]')
        for online, target in zip(self.cnn.parameters(), self.target_cnn.parameters()):
            target.lerp_(online, tau)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.lerp_(online, tau)

    def forward(self, terrain, state):
        return self.actor(self.encode(terrain, state)), self.critic(self.encode_critic(terrain, state)).squeeze(-1)

    def distribution(self, terrain, state):
        mean, value = self(terrain, state)
        return Normal(mean, self.log_std.clamp(-5, 2).exp().expand_as(mean)), value

    @staticmethod
    def log_prob(dist, raw):
        # Stable log(1-tanh(x)^2) = 2(log(2)-x-softplus(-2x)).
        correction = 2 * (np.log(2.0) - raw - torch.nn.functional.softplus(-2 * raw))
        return (dist.log_prob(raw) - correction).sum(-1)

    def act(self, terrain, state, deterministic=False):
        dist, value = self.distribution(terrain, state)
        raw = dist.mean if deterministic else dist.sample()
        return torch.tanh(raw), raw, self.log_prob(dist, raw), value


def tensor_obs(observations, device):
    return (torch.as_tensor(np.stack([o['terrain'] for o in observations]), dtype=torch.float32, device=device),
            torch.as_tensor(np.stack([o['state'] for o in observations]), dtype=torch.float32, device=device))
