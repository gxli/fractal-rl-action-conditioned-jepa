"""Synchronous vectorized-environment PPO baseline with reproducible checkpoints."""
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import random
import numpy as np
import torch
from src.rl.policy import ActorCritic, tensor_obs
from src.rl.rollout import compute_gae
from src.env.fractal_env import FractalEnv
from src.terrain.pink_noise import generate
from src.utils.config import save_config


def _step_environment(item):
    """Top-level worker target so independent NumPy environments can step concurrently."""
    environment, action = item
    return environment.step(action)


def evaluate(policy, cfg, device, episodes_per_seed=1):
    scores = []
    for seed in cfg['environment']['evaluation_terrain_seeds']:
        terrain = generate(cfg['terrain'], seed=int(seed))
        env = FractalEnv(cfg, terrain=terrain, seed=int(seed))
        for rep in range(episodes_per_seed):
            obs, _ = env.reset(seed=int(seed) + rep * 10000)
            done = False
            while not done:
                with torch.no_grad():
                    t, s = tensor_obs([obs], device)
                    action, _, _, _ = policy.act(t, s, deterministic=True)
                obs, _, term, trunc, info = env.step(action[0].cpu().numpy())
                done = term or trunc
            scores.append(info['mean_speed'])
    return float(np.mean(scores)), float(np.std(scores))


def train(cfg, device='cpu', max_steps=None):
    tc = cfg['training']
    seed = int(tc['seed'])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    output = Path(cfg['output']['dump_dir'])
    (output / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output / 'metrics').mkdir(parents=True, exist_ok=True)
    save_config(cfg, output / 'resolved_config.yaml')
    train_seeds = {int(cfg['terrain']['seed']) + i for i in range(int(cfg['environment']['training_terrain_count']))}
    evaluation_seeds = {int(seed) for seed in cfg['environment']['evaluation_terrain_seeds']}
    if train_seeds & evaluation_seeds:
        raise ValueError(f'training and evaluation terrain seeds overlap: {sorted(train_seeds & evaluation_seeds)}')
    terrain_bank = [generate(cfg['terrain'], seed=seed) for seed in sorted(train_seeds)]
    rng = np.random.default_rng(seed)
    n = int(cfg['environment']['num_parallel_envs'])
    environments = [FractalEnv(cfg, terrain=terrain_bank[i % len(terrain_bank)], seed=seed + i)
                    for i in range(n)]
    obs = [env.reset()[0] for env in environments]
    jepa_coef = float(tc.get('jepa_coef', 0.0))
    if jepa_coef < 0:
        raise ValueError('training.jepa_coef must be non-negative')
    policy = ActorCritic(cfg['agent']['fov_size'], architecture=cfg['agent'].get('architecture', 'shared'),
                         jepa_enabled=jepa_coef > 0).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(tc['learning_rate']))
    target = int(max_steps if max_steps is not None else tc['total_timesteps'])
    count = 0
    next_eval = int(tc['evaluation_interval'])
    next_checkpoint = int(tc['checkpoint_interval'])
    metrics = output / 'metrics' / 'training.csv'
    workers = int(tc.get('parallel_env_workers', 1))
    if workers < 1:
        raise ValueError('training.parallel_env_workers must be positive')
    executor = ThreadPoolExecutor(max_workers=min(workers, n)) if workers > 1 else None
    with metrics.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['steps', 'rollout_mean_speed', 'evaluation_mean_speed',
                                                   'evaluation_std_speed', 'policy_loss', 'value_loss', 'jepa_loss'])
        writer.writeheader()
        while count < target:
            storage = {k: [] for k in ['terrain', 'state', 'raw', 'logp', 'value', 'reward', 'next_value',
                                        'next_terrain', 'next_state', 'terminated', 'ended']}
            speed_samples = []
            horizon = min(int(tc['rollout_steps']), max(1, (target - count + n - 1) // n))
            for _ in range(horizon):
                with torch.no_grad():
                    terrain_tensor, state_tensor = tensor_obs(obs, device)
                    actions, raw, logp, value = policy.act(terrain_tensor, state_tensor)
                storage['terrain'].append(terrain_tensor.cpu().numpy())
                storage['state'].append(state_tensor.cpu().numpy())
                storage['raw'].append(raw.cpu().numpy())
                storage['logp'].append(logp.cpu().numpy())
                storage['value'].append(value.cpu().numpy())
                next_obs, next_values, rewards, terminated, ended = [], [], [], [], []
                action_np = actions.cpu().numpy()
                results = (list(executor.map(_step_environment, zip(environments, action_np)))
                           if executor is not None else
                           [env.step(action_np[i]) for i, env in enumerate(environments)])
                for o, r, term, trunc, info in results:
                    rewards.append(r)
                    terminated.append(float(term))
                    ended.append(float(term or trunc))
                    speed_samples.append(info['speed'])
                    next_obs.append(o)
                # Critic on actual terminal observations BEFORE resetting time-limited episodes.
                with torch.no_grad():
                    tt, ss = tensor_obs(next_obs, device)
                    _, bootstrap = policy(tt, ss)
                    next_values = bootstrap.cpu().numpy()
                next_terrain_np, next_state_np = tt.cpu().numpy(), ss.cpu().numpy()
                for i, finished in enumerate(ended):
                    if finished:
                        new_terrain = terrain_bank[int(rng.integers(len(terrain_bank)))]
                        next_obs[i], _ = environments[i].reset(terrain=new_terrain)
                obs = next_obs
                storage['reward'].append(rewards)
                storage['next_value'].append(next_values)
                storage['next_terrain'].append(next_terrain_np)
                storage['next_state'].append(next_state_np)
                storage['terminated'].append(terminated)
                storage['ended'].append(ended)
            data = {key: np.asarray(value, dtype=np.float32) for key, value in storage.items()}
            advantages, returns = compute_gae(data['reward'], data['value'], data['next_value'],
                                               data['terminated'], data['ended'],
                                               float(tc['gamma']), float(tc['gae_lambda']))
            batch_size = horizon * n
            flatten = lambda value: torch.as_tensor(value.reshape((batch_size,) + value.shape[2:]), device=device)
            terrains, states = flatten(data['terrain']), flatten(data['state'])
            raws, old_logp, old_value = flatten(data['raw']), flatten(data['logp']), flatten(data['value'])
            next_terrains, next_states = flatten(data['next_terrain']), flatten(data['next_state'])
            adv, ret = flatten(advantages), flatten(returns)
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            policy_loss = value_loss = jepa_loss = 0.0
            for _ in range(int(tc['update_epochs'])):
                for indices in torch.randperm(batch_size, device=device).split(int(tc['minibatch_size'])):
                    dist, val = policy.distribution(terrains[indices], states[indices])
                    new_logp = policy.log_prob(dist, raws[indices])
                    ratio = (new_logp - old_logp[indices]).exp()
                    clipped = ratio.clamp(1 - float(tc['clip_range']), 1 + float(tc['clip_range']))
                    p_loss = -torch.minimum(ratio * adv[indices], clipped * adv[indices]).mean()
                    value_clip = tc.get('value_clip_range')
                    if value_clip is None:
                        v_loss = 0.5 * (val - ret[indices]).square().mean()
                    else:
                        clipped_value = old_value[indices] + (val - old_value[indices]).clamp(
                            -float(value_clip), float(value_clip))
                        v_loss = 0.5 * torch.maximum((val - ret[indices]).square(),
                                                     (clipped_value - ret[indices]).square()).mean()
                    # Gaussian pre-squash entropy is a proxy, not exact squashed entropy.
                    j_loss = policy.jepa_loss(terrains[indices], states[indices], torch.tanh(raws[indices]),
                                              next_terrains[indices], next_states[indices])
                    loss = (p_loss + float(tc['value_coef']) * v_loss
                            - float(tc['entropy_coef']) * dist.entropy().sum(-1).mean() + jepa_coef * j_loss)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), float(tc['max_grad_norm']))
                    optimizer.step()
                    policy.update_jepa_target(float(tc.get('jepa_target_tau', 0.01)))
                    policy_loss, value_loss, jepa_loss = float(p_loss.item()), float(v_loss.item()), float(j_loss.item())
            count += horizon * n
            mean_eval = std_eval = float('nan')
            if count >= next_eval or count >= target:
                mean_eval, std_eval = evaluate(policy, cfg, device)
                next_eval = count + int(tc['evaluation_interval'])
            if count >= next_checkpoint or count >= target:
                torch.save({'model': policy.state_dict(), 'optimizer': optimizer.state_dict(),
                            'steps': count, 'config': cfg, 'torch_rng': torch.get_rng_state(),
                            'numpy_rng': rng.bit_generator.state, 'python_rng': random.getstate()},
                           output / 'checkpoints' / f'ppo_{count:09d}.pt')
                next_checkpoint = count + int(tc['checkpoint_interval'])
            writer.writerow(dict(steps=count, rollout_mean_speed=float(np.mean(speed_samples)),
                                 evaluation_mean_speed=mean_eval, evaluation_std_speed=std_eval,
                                 policy_loss=policy_loss, value_loss=value_loss, jepa_loss=jepa_loss))
            file.flush()
            eval_label = f'{mean_eval:.3f}' if np.isfinite(mean_eval) else 'pending'
            print(f'steps={count} rollout_speed={np.mean(speed_samples):.3f} eval_speed={eval_label}', flush=True)
    if executor is not None:
        executor.shutdown()
    return policy


def load_policy(path, device='cpu', trusted_legacy=False):
    """Load an inference checkpoint without unpickling arbitrary Python by default."""
    checkpoint = torch.load(path, map_location=device, weights_only=not trusted_legacy)
    cfg = checkpoint['config']
    if checkpoint.get('algorithm', 'ppo') == 'sac':
        from src.rl.sac import SACActor
        policy = SACActor(cfg['agent']['fov_size'],
                          jepa_enabled=float(cfg['training'].get('jepa_coef', 0.0)) > 0,
                          jepa_horizon=int(cfg['training'].get('jepa_horizon', 1)),
                          jepa_inverse_enabled=float(cfg['training'].get('jepa_inverse_coef', 0.0)) > 0,
                          cdd_scales=cfg['terrain'].get('cdd_scales')).to(device)
    else:
        # Checkpoints saved before the architecture option are the original shared model.
        policy = ActorCritic(cfg['agent']['fov_size'], architecture=cfg['agent'].get('architecture', 'shared'),
                             jepa_enabled=float(cfg['training'].get('jepa_coef', 0.0)) > 0).to(device)
    policy.load_state_dict(checkpoint['model'])
    policy.eval()
    return policy, cfg
