"""Tanh-Gaussian SAC actor, optional JEPA auxiliary branch, and twin Q networks."""
import copy
import math
import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import functional as F
from src.rl.cdd_convnext import CDDScaleAwareConvNeXt


def make_encoder(fov_size, state_dim=6, cdd_scales=None):
    if cdd_scales:
        cnn = CDDScaleAwareConvNeXt(cdd_scales)
    else:
        cnn = nn.Sequential(nn.Conv2d(1, 16, 5, stride=2), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(), nn.Flatten())
    channels = len(cdd_scales) if cdd_scales else 1
    with torch.no_grad(): features = cnn(torch.zeros(1, channels, fov_size, fov_size)).shape[-1]
    return cnn, nn.Sequential(nn.Linear(features + state_dim, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh())


class SACActor(nn.Module):
    def __init__(self, fov_size=32, state_dim=6, jepa_enabled=False, jepa_horizon=1,
                 jepa_inverse_enabled=False, cdd_scales=None):
        super().__init__()
        self.cnn, self.encoder = make_encoder(fov_size, state_dim, cdd_scales)
        self.actor = nn.Linear(128, 4)
        self.jepa_enabled = jepa_enabled
        self.jepa_horizon = int(jepa_horizon)
        self.jepa_inverse_enabled = jepa_inverse_enabled
        if self.jepa_horizon < 1:
            raise ValueError('jepa_horizon must be positive')
        if jepa_enabled:
            # This target is separate from SAC's target critics. It is an EMA
            # copy of the actor encoder and receives no gradient updates.
            self.target_cnn = copy.deepcopy(self.cnn)
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_cnn.requires_grad_(False)
            self.target_encoder.requires_grad_(False)
            self.jepa_predictor = nn.Sequential(
                nn.Linear(128 + 2 * self.jepa_horizon, 128), nn.Tanh(), nn.Linear(128, 128))
            if jepa_inverse_enabled:
                self.jepa_inverse_predictor = nn.Sequential(
                    nn.Linear(256, 128), nn.Tanh(), nn.Linear(128, 2))

    def encode(self, terrain, state):
        return self.encoder(torch.cat([self.cnn(terrain), state], dim=-1))

    def target_encode(self, terrain, state):
        if not self.jepa_enabled:
            raise RuntimeError('JEPA target encoder is disabled')
        return self.target_encoder(torch.cat([self.target_cnn(terrain), state], dim=-1))

    def jepa_loss(self, terrain, state, action, next_terrain, next_state,
                  embedding=None, return_details=False):
        """Predict a K-step EMA target from an online latent and action sequence."""
        if not self.jepa_enabled:
            return terrain.new_zeros(())
        embedding = self.encode(terrain, state) if embedding is None else embedding
        if action.ndim == 2:
            action = action[:, None, :]
        if action.shape[1:] != (self.jepa_horizon, 2):
            raise ValueError(f'expected action sequence [batch, {self.jepa_horizon}, 2]')
        prediction = self.jepa_predictor(torch.cat([embedding, action.flatten(1)], dim=-1))
        with torch.no_grad():
            target = self.target_encode(next_terrain, next_state)
        loss = (1.0 - F.cosine_similarity(prediction, target, dim=-1)).mean()
        return (loss, prediction, target) if return_details else loss

    def inverse_loss(self, embedding, target_embedding, action_sequence):
        """Recover the first applied action from the two latent endpoints."""
        if not self.jepa_inverse_enabled:
            return embedding.new_zeros(())
        prediction = self.jepa_inverse_predictor(torch.cat([embedding, target_embedding], dim=-1))
        return F.mse_loss(prediction, action_sequence[:, 0])

    @torch.no_grad()
    def update_jepa_target(self, tau):
        """Move the JEPA target encoder ``tau`` of the way toward online weights."""
        if not self.jepa_enabled:
            return
        if not 0 < tau <= 1:
            raise ValueError('jepa_target_tau must be in (0, 1]')
        for online, target in zip(self.cnn.parameters(), self.target_cnn.parameters()):
            target.lerp_(online, tau)
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.lerp_(online, tau)

    def act_from_embedding(self, embedding, deterministic=False):
        """Sample an action after a caller has already encoded an observation."""
        mean, log_std = self.actor(embedding).chunk(2, dim=-1)
        distribution = Normal(mean, log_std.clamp(-5, 2).exp())
        raw = mean if deterministic else distribution.rsample()
        action = torch.tanh(raw)
        correction = 2 * (math.log(2) - raw - F.softplus(-2 * raw))
        return action, raw, (distribution.log_prob(raw) - correction).sum(-1), None

    def act(self, terrain, state, deterministic=False):
        return self.act_from_embedding(self.encode(terrain, state), deterministic)


class QNetwork(nn.Module):
    def __init__(self, fov_size=32, state_dim=6, shared_base=None,
                 jepa_enabled=False, jepa_horizon=1, cdd_scales=None,
                 detach_shared=True):
        super().__init__()
        # When supplied, ``shared_base`` is the online actor.  It remains a
        # registered submodule deliberately: deepcopy(TwinQ) then creates a
        # separate, Polyak-updated encoder for target critics.
        self.shared_base = shared_base
        self.detach_shared = bool(detach_shared)
        self.jepa_enabled = bool(jepa_enabled)
        self.jepa_horizon = int(jepa_horizon)
        if self.jepa_horizon < 1:
            raise ValueError('jepa_horizon must be positive')
        if shared_base is None:
            self.cnn, self.encoder = make_encoder(fov_size, state_dim, cdd_scales)
        elif self.jepa_enabled:
            raise ValueError('critic JEPA requires independent critic encoders')
        self.head = nn.Sequential(nn.Linear(130, 128), nn.ReLU(), nn.Linear(128, 1))
        if self.jepa_enabled:
            self.jepa_predictor = nn.Sequential(
                nn.Linear(128 + 2 * self.jepa_horizon, 128), nn.Tanh(), nn.Linear(128, 128))

    def encode(self, terrain, state):
        if self.shared_base is not None:
            embedding = self.shared_base.encode(terrain, state)
            return embedding.detach() if self.detach_shared else embedding
        return self.encoder(torch.cat([self.cnn(terrain), state], dim=-1))

    def forward(self, terrain, state, action):
        embedding = self.encode(terrain, state)
        return self.head(torch.cat([embedding, action], dim=-1)).squeeze(-1)

    def jepa_loss(self, terrain, state, action, next_terrain, next_state, target_network):
        """Train a critic encoder to predict its own target-critic future latent."""
        if not self.jepa_enabled:
            return terrain.new_zeros(())
        if action.ndim == 2:
            action = action[:, None, :]
        if action.shape[1:] != (self.jepa_horizon, 2):
            raise ValueError(f'expected action sequence [batch, {self.jepa_horizon}, 2]')
        prediction = self.jepa_predictor(torch.cat([self.encode(terrain, state), action.flatten(1)], dim=-1))
        with torch.no_grad():
            target_embedding = target_network.encode(next_terrain, next_state)
        return (1.0 - F.cosine_similarity(prediction, target_embedding, dim=-1)).mean()


class TwinQ(nn.Module):
    def __init__(self, fov_size=32, state_dim=6, shared_base=None,
                 jepa_enabled=False, jepa_horizon=1, cdd_scales=None,
                 detach_shared=True):
        super().__init__()
        self.q1 = QNetwork(fov_size, state_dim, shared_base, jepa_enabled, jepa_horizon, cdd_scales, detach_shared)
        self.q2 = QNetwork(fov_size, state_dim, shared_base, jepa_enabled, jepa_horizon, cdd_scales, detach_shared)

    def value_parameters(self):
        """Parameters owned by critic value heads, excluding a shared actor."""
        yield from self.q1.head.parameters()
        yield from self.q2.head.parameters()

    def set_value_requires_grad(self, requires_grad):
        """Freeze only critic heads; never freeze a shared actor backbone."""
        for parameter in self.value_parameters():
            parameter.requires_grad_(requires_grad)

    def set_shared_encoder_detach(self, detach):
        """Control whether value heads backpropagate through a shared base."""
        self.q1.detach_shared = self.q2.detach_shared = bool(detach)

    def forward(self, terrain, state, action):
        return self.q1(terrain, state, action), self.q2(terrain, state, action)
