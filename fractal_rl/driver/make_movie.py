import argparse
from pathlib import Path
import numpy as np
import torch
from src.rl.trainer import load_policy
from src.rl.policy import tensor_obs
from src.terrain.pink_noise import generate
from src.env.fractal_env import FractalEnv
from src.visualization.renderer import frame
from src.visualization.video import save_video


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--seed', type=int, default=100001)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--every', type=int, default=3)
    p.add_argument('--format', choices=['mp4', 'gif'], default='mp4')
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    if a.every < 1:
        p.error('--every must be positive')
    policy, cfg = load_policy(a.checkpoint, a.device)
    env = FractalEnv(cfg, generate(cfg['terrain'], a.seed), a.seed)
    obs, _ = env.reset(seed=a.seed)
    trail, images, trajectory = [], [], []
    for step in range(a.steps):
        with torch.no_grad():
            terrain, state = tensor_obs([obs], a.device)
            action, _, _, _ = policy.act(terrain, state, deterministic=True)
        act = action[0].cpu().numpy()
        obs, _, term, trunc, info = env.step(act)
        trail.append(env.body.position.copy())
        trajectory.append([step, *env.body.position, *env.body.velocity, info['speed'], info['potential'], *act])
        if step % a.every == 0:
            images.append(frame(env, trail))
        if term or trunc:
            break
    dest = Path(cfg['output']['movie_dir']) / f'trajectory_seed_{a.seed}.{a.format}'
    save_video(images, dest, cfg['output']['fps'])
    path = Path(cfg['output']['dump_dir']) / 'trajectories' / f'trajectory_seed_{a.seed}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, trajectory, delimiter=',', comments='',
               header='step,x,y,vx,vy,speed,potential,throttle,steering')
    print(dest)
    print(path)


if __name__ == '__main__':
    main()
