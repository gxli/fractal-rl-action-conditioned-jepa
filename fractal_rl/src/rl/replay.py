"""Fixed-size replay storage for off-policy algorithms."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity, observation, seed=0):
        self.capacity, self.pos, self.size = int(capacity), 0, 0
        if self.capacity < 1:
            raise ValueError('capacity must be positive')
        self.rng = np.random.default_rng(seed)
        def alloc(shape): return np.empty((self.capacity, *shape), dtype=np.float32)
        self.terrain, self.state = alloc(observation['terrain'].shape), alloc(observation['state'].shape)
        self.next_terrain, self.next_state = alloc(observation['terrain'].shape), alloc(observation['state'].shape)
        self.action, self.reward, self.terminated, self.ended = alloc((2,)), alloc(()), alloc(()), alloc(())

    def add(self, observation, action, reward, next_observation, terminated, ended=False):
        i = self.pos
        self.terrain[i], self.state[i] = observation['terrain'], observation['state']
        self.action[i], self.reward[i], self.terminated[i], self.ended[i] = action, reward, float(terminated), float(ended)
        self.next_terrain[i], self.next_state[i] = next_observation['terrain'], next_observation['state']
        self.pos, self.size = (i + 1) % self.capacity, min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        if self.size < batch_size:
            raise ValueError('not enough transitions')
        i = self.rng.integers(self.size, size=batch_size)
        as_tensor = lambda array: torch.as_tensor(array[i], dtype=torch.float32, device=device)
        return tuple(as_tensor(array) for array in (self.terrain, self.state, self.action, self.reward,
                                                     self.next_terrain, self.next_state, self.terminated))

    def sample_sequence(self, batch_size, horizon, device):
        """Sample contiguous replay sequences without crossing episode resets.

        The returned action sequence starts at ``o_t`` and its target is the
        real next observation after the K-th action.  ``ended`` is separate
        from ``terminated`` because time-limit resets must break JEPA
        sequences while SAC still bootstraps through them.
        """
        horizon = int(horizon)
        if horizon < 1 or self.size < horizon:
            raise ValueError('not enough transitions for requested horizon')
        candidates = self.size - horizon + 1
        oldest = self.pos if self.size == self.capacity else 0
        offsets = np.arange(horizon)
        valid = []
        # Episodes are long relative to the default horizon, so rejection
        # sampling is cheaper than scanning a large replay buffer per update.
        while len(valid) < batch_size:
            logical = self.rng.integers(candidates, size=max(batch_size * 2, 32))
            physical = (oldest + logical[:, None] + offsets[None, :]) % self.capacity
            valid.extend(logical[~self.ended[physical].astype(bool).any(axis=1)].tolist())
            if len(valid) == 0 and candidates < batch_size:
                raise ValueError('no complete replay sequence without an episode boundary')
        logical = np.asarray(valid[:batch_size], dtype=np.int64)
        sequence = (oldest + logical[:, None] + offsets[None, :]) % self.capacity
        start = (oldest + logical) % self.capacity
        target = (oldest + logical + horizon - 1) % self.capacity
        as_tensor = lambda array: torch.as_tensor(array, dtype=torch.float32, device=device)
        return (as_tensor(self.terrain[start]), as_tensor(self.state[start]), as_tensor(self.action[sequence]),
                as_tensor(self.next_terrain[target]), as_tensor(self.next_state[target]))
