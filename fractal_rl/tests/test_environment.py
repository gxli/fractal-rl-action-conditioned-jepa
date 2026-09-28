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
    cfg['environment']['start_position_mode'] = 'random'
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    env.reset(seed=1)
    first = env.body.position.copy()
    env.reset()
    second = env.body.position.copy()
    width = env.terrain.size * env.terrain.pixel_size
    assert np.all(first >= 0) and np.all(first < width)
    assert np.all(second >= 0) and np.all(second < width)
    assert not np.array_equal(first, second)


def test_global_minimum_start_is_shared_and_has_minimum_potential():
    cfg = small_config()
    cfg['environment']['start_position_mode'] = 'global_minimum'
    terrain = generate(cfg['terrain'], 17)
    env = FractalEnv(cfg, terrain=terrain)
    _, first = env.reset(seed=1)
    _, second = env.reset(seed=999)
    assert np.array_equal(first['start_position'], second['start_position'])
    assert np.isclose(first['start_potential'], float(terrain.potential.min()), atol=1e-5)


def test_explicit_start_position_is_periodic_and_seed_preserves_motion_state():
    cfg = small_config()
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    env.reset(seed=9, start_position=[-1.5, 65.5])
    first_velocity = env.body.velocity.copy()
    width = env.terrain.size * env.terrain.pixel_size
    np.testing.assert_allclose(env.body.position, np.array([-1.5, 65.5]) % width)
    env.reset(seed=9, start_position=[1.5, 2.5])
    np.testing.assert_allclose(env.body.velocity, first_velocity)


def test_world_frame_keeps_both_velocity_components():
    cfg = small_config()
    cfg['agent']['observation_frame'] = 'world'
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    env.reset(seed=1)
    env.body.velocity = np.array([3.0, 4.0])
    obs, _ = env.reset(seed=1)
    # Reset supplies a fresh body; set a non-axis-aligned velocity before
    # observing so the test checks that neither component is normalized away.
    env.body.velocity = np.array([3.0, 4.0])
    from src.env.observation import observe
    obs = observe(env.body, env.terrain, cfg['agent'], env.previous_action)
    np.testing.assert_allclose(obs['state'][:3], [4.0 / 6.0, 3.0 / 6.0, 5.0 / 6.0])


def test_velocity_frame_preserves_legacy_redundancy_for_old_configs():
    cfg = small_config()
    cfg['agent']['observation_frame'] = 'velocity'
    env = FractalEnv(cfg, terrain=generate(cfg['terrain'], 17))
    env.reset(seed=1)
    env.body.velocity = np.array([3.0, 4.0])
    from src.env.observation import observe
    obs = observe(env.body, env.terrain, cfg['agent'], env.previous_action)
    np.testing.assert_allclose(obs['state'][:3], [5.0 / 6.0, 0.0, 5.0 / 6.0], atol=1e-7)


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
