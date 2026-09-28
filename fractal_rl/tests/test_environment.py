import numpy as np
import torch
from src.env.fractal_env import FractalEnv
from src.rl.policy import ActorCritic, tensor_obs
from src.rl.sac import SACActor
from src.rl.rollout import compute_gae
from src.terrain.pink_noise import generate
from src.utils.config import load_config


def small_config():
    cfg = load_config()
    cfg['terrain']['resolution'] = 32
    cfg['environment']['episode_steps'] = 3
    cfg['agent']['fov_size'] = 32
    return cfg


def test_reset_step_and_time_limit():
    cfg = small_config()
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    obs, _ = env.reset(seed=8)
    assert obs['terrain'].shape == (1, 32, 32)
    assert obs['state'].shape == (6,)
    for i in range(3):
        obs, rew, term, trunc, info = env.step(np.array([1, 0]))
        assert np.isfinite(rew)
        assert not term
        assert trunc == (i == 2)
        assert info['mean_speed'] >= 0


def test_random_start_positions_cover_periodic_tile():
    cfg = small_config()
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    env.reset(seed=1)
    first = env.body.position.copy()
    env.reset()
    second = env.body.position.copy()
    width = env.terrain.size * env.terrain.pixel_size
    assert np.all(first >= 0) and np.all(first < width)
    assert np.all(second >= 0) and np.all(second < width)
    assert not np.array_equal(first, second)


def test_policy_actions_finite_and_gae_bootstrap():
    torch.manual_seed(0)
    net = ActorCritic()
    cfg = small_config()
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    observation, _ = env.reset()
    t, s = tensor_obs([observation], 'cpu')
    with torch.no_grad():
        actions, raw, logp, val = net.act(t, s)
    assert actions.shape == (1, 2)
    assert torch.isfinite(logp).all()
    assert (actions.abs() <= 1).all()
    adv, returns = compute_gae(np.array([[1.0]], dtype=np.float32),
                                np.array([[0.0]], dtype=np.float32),
                                np.array([[2.0]], dtype=np.float32),
                                np.array([[0.0]], dtype=np.float32),
                                np.array([[1.0]], dtype=np.float32), 0.9, 0.95)
    np.testing.assert_allclose(returns, [[2.8]])


def test_jepa_loss_and_ema_target_are_finite():
    net = ActorCritic(jepa_enabled=True)
    terrain = torch.zeros(2, 1, 32, 32)
    state = torch.zeros(2, 6)
    loss = net.jepa_loss(terrain, state, torch.zeros(2, 2), terrain, state)
    assert torch.isfinite(loss)
    net.update_jepa_target(0.01)


def test_sac_jepa_loss_and_ema_target_are_finite():
    net = SACActor(jepa_enabled=True)
    terrain = torch.zeros(2, 1, 32, 32)
    state = torch.zeros(2, 6)
    loss = net.jepa_loss(terrain, state, torch.zeros(2, 2), terrain, state)
    assert torch.isfinite(loss)
    net.update_jepa_target(0.01)
