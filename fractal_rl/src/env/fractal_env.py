"""Minimal Gymnasium-style API without requiring gymnasium to be installed."""
import numpy as np
from src.physics.dynamics import Body, step as physics_step
from src.env.observation import observe
from src.env.reward import reward
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample


class FractalEnv:
    def __init__(self, cfg, terrain=None, seed=None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg['training']['seed'] if seed is None else seed)
        self.terrain = terrain if terrain is not None else generate(cfg['terrain'])
        self.body = None
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.distance = 0.0

    def _global_minimum_position(self):
        """Deterministic pixel-centre global potential minimum for this tile."""
        row, column = np.unravel_index(np.argmin(self.terrain.potential),
                                       self.terrain.potential.shape)
        dx = self.terrain.pixel_size
        # ``sample`` indexes integer world coordinates at array samples.
        return np.asarray([column * dx, row * dx], dtype=np.float64)

    def reset(self, seed=None, terrain=None, start_position=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if terrain is not None:
            self.terrain = terrain
        width = self.terrain.size * self.terrain.pixel_size
        if start_position is not None:
            position = np.asarray(start_position, dtype=np.float64)
            if position.shape != (2,) or not np.all(np.isfinite(position)):
                raise ValueError('start_position must be a finite two-element vector')
            # Positions live on a periodic tile.  Canonicalizing supplied
            # locations makes repeated-location evaluation unambiguous.
            position = position % width
        elif self.cfg['environment'].get('start_position_mode', 'random') == 'global_minimum':
            # A terrain minimum has no immediately available downhill impulse.
            # It is deterministic for this terrain seed, aligning all methods.
            position = self._global_minimum_position()
        elif self.cfg['environment'].get('start_position_mode', 'random') == 'centre':
            position = np.full(2, width / 2.0)
        elif self.cfg['environment'].get('start_position_mode', 'random') == 'random' and self.cfg['environment'].get('random_start_position', True):
            # Uniform starts cover the fundamental tile; periodic sampling then
            # makes every translated tile physically equivalent.
            position = self.rng.uniform(0, width, size=2)
        elif self.cfg['environment'].get('start_position_mode', 'random') == 'random':
            position = np.full(2, width / 2.0)
        else:
            raise ValueError("environment.start_position_mode must be 'random', 'global_minimum', or 'centre'")
        heading = float(self.rng.uniform(-np.pi, np.pi))
        pcfg = self.cfg['physics']
        v = float(self.rng.uniform(pcfg['initial_speed_min'], pcfg['initial_speed_max']))
        self.body = Body(position, v * np.array([np.cos(heading), np.sin(heading)]),
                         heading, float(pcfg['mass']))
        self.previous_action = np.zeros(2, dtype=np.float32)
        self.steps = 0
        self.distance = 0.0
        return observe(self.body, self.terrain, self.cfg['agent'], self.previous_action), {
            'start_position': position.copy(),
            'start_potential': float(sample(self.terrain.potential, *position, self.terrain.pixel_size)),
            'start_position_mode': self.cfg['environment'].get('start_position_mode', 'random'),
        }

    def step(self, action):
        if self.body is None:
            raise RuntimeError('reset() must be called before step()')
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError('action must be a finite two-element vector')
        action = np.clip(action, -1.0, 1.0)
        repeats = int(self.cfg['environment'].get('action_repeat', 1))
        if repeats < 1:
            raise ValueError('environment.action_repeat must be positive')
        dist = 0.0
        rew = 0.0
        # The policy action is held constant while the existing stable .02 s
        # integrator advances.  Episode length remains measured in physics
        # steps, so repeat changes decision rate, not simulated duration.
        previous_action = self.previous_action
        for _ in range(repeats):
            if self.steps >= int(self.cfg['environment']['episode_steps']):
                break
            step_distance, _ = physics_step(self.body, action, self.terrain, self.cfg['physics'])
            # Charge an action-change cost once per decision, rather than once
            # per physics substep while the action is held constant.
            rew += reward(step_distance, action, previous_action, self.cfg['physics']['dt'], self.cfg['reward'])
            previous_action = action
            dist += step_distance
            self.steps += 1
        # Info is consistently post-step: position, speed, and potential refer
        # to the same state visible in the returned observation.
        potential = float(sample(self.terrain.potential, *self.body.position, self.terrain.pixel_size))
        self.previous_action = action.copy()
        self.distance += dist
        truncated = self.steps >= int(self.cfg['environment']['episode_steps'])
        obs = observe(self.body, self.terrain, self.cfg['agent'], self.previous_action)
        info = {'speed': float(np.linalg.norm(self.body.velocity)), 'distance_step': dist,
                'episode_distance': self.distance, 'mean_speed': self.distance / (self.steps * self.cfg['physics']['dt']),
                'potential': potential, 'position': self.body.position.copy()}
        return obs, rew, False, truncated, info
