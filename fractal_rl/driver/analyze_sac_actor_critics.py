"""Probe SAC actor and both independent critic encoders on one held-out rollout.

The physical path is shared by Actor, Q1, and Q2: each representation panel
therefore describes what a different learner sees at exactly the same state.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.env.fractal_env import FractalEnv
from src.rl.policy import tensor_obs
from src.rl.sac import SACActor, TwinQ
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample


def pca(z):
    z = z - z.mean(0, keepdims=True)
    return np.linalg.svd(z, full_matrices=False)[0][:, :2] * np.linalg.svd(z, full_matrices=False)[1][:2]


def embed_umap(z, seed):
    from cuml.manifold import UMAP
    z = (z - z.mean(0)) / np.maximum(z.std(0), 1e-6)
    return np.asarray(UMAP(n_components=2, n_neighbors=min(30, max(2, len(z) - 1)),
                         min_dist=.1, metric='euclidean', random_state=seed,
                         output_type='numpy').fit_transform(z))


def load(path, device):
    data = torch.load(path, map_location=device, weights_only=False)
    cfg = data['config']; tc = cfg['training']
    actor_jepa = any(key.startswith('target_cnn.') for key in data['model'])
    critic_jepa = any(key.startswith('q1.jepa_predictor.') for key in data['critics'])
    actor = SACActor(cfg['agent']['fov_size'], jepa_enabled=actor_jepa,
                     jepa_horizon=int(tc.get('jepa_horizon', 1)),
                     jepa_inverse_enabled=any(key.startswith('jepa_inverse_predictor.') for key in data['model']),
                     cdd_scales=cfg['terrain'].get('cdd_scales')).to(device)
    sharing = tc.get('encoder_sharing', 'actor_stop_gradient' if tc.get('share_encoder_with_critic', False) else 'separate')
    shared = sharing != 'separate'
    critics = TwinQ(cfg['agent']['fov_size'], shared_base=actor if shared else None,
                    jepa_enabled=critic_jepa,
                    jepa_horizon=int(tc.get('jepa_horizon', 1)),
                    cdd_scales=cfg['terrain'].get('cdd_scales'),
                    detach_shared=sharing in ('separate', 'actor_stop_gradient')).to(device)
    actor.load_state_dict(data['model']); critics.load_state_dict(data['critics'])
    actor.eval(); critics.eval()
    return actor, critics, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint'); ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=100001); ap.add_argument('--steps', type=int, default=600)
    ap.add_argument('--output-dir', default=None)
    args = ap.parse_args()
    actor, critics, cfg = load(args.checkpoint, args.device)
    root = Path(args.output_dir or cfg['output']['dump_dir']); out = root / 'analysis'; out.mkdir(parents=True, exist_ok=True)
    terrain = generate(cfg['terrain'], seed=args.seed); env = FractalEnv(cfg, terrain=terrain, seed=args.seed)
    obs, _ = env.reset(seed=args.seed); positions=[]; speeds=[]; actor_z=[]; q1_z=[]; q2_z=[]
    for _ in range(args.steps):
        with torch.no_grad():
            t, s = tensor_obs([obs], args.device)
            actor_z.append(actor.encode(t, s)[0].cpu().numpy())
            q1_z.append(critics.q1.encode(t, s)[0].cpu().numpy())
            q2_z.append(critics.q2.encode(t, s)[0].cpu().numpy())
            action, *_ = actor.act(t, s, deterministic=True)
        positions.append(env.body.position.copy()); speeds.append(np.linalg.norm(env.body.velocity))
        obs, _, term, trunc, _ = env.step(action[0].cpu().numpy())
        if term or trunc: break
    zs = [np.asarray(actor_z), np.asarray(q1_z), np.asarray(q2_z)]
    pos=np.asarray(positions); speed=np.asarray(speeds); n=len(pos); stem=f'{Path(args.checkpoint).stem}_seed{args.seed}'
    np.savez_compressed(out / f'actor_q_representations_{stem}.npz', positions=pos, speed=speed,
                        actor=zs[0], q1=zs[1], q2=zs[2])
    fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    pad=8; gx=np.linspace(pos[:,0].min()-pad,pos[:,0].max()+pad,180); gy=np.linspace(pos[:,1].min()-pad,pos[:,1].max()+pad,180); xx,yy=np.meshgrid(gx,gy)
    field=sample(terrain.potential,xx,yy,terrain.pixel_size)
    ax=axes[0,0]; c=ax.contourf(xx,yy,field,levels=24,cmap='viridis_r'); ax.plot(pos[:,0],pos[:,1],color='#ef4444',lw=1.2); ax.scatter(*pos[0],c='white',edgecolor='black',s=28); ax.set(title='Held-out potential + trajectory',xlabel='x',ylabel='y',aspect='equal'); fig.colorbar(c,ax=ax,label='potential')
    names=('Actor encoder','Q1 encoder','Q2 encoder')
    for row,(name,z) in enumerate(zip(names,zs)):
        coords=[pca(z),embed_umap(z,int(cfg['training']['seed']))]
        if row:
            axes[row,0].axis('off'); axes[row,0].text(.5,.5,name,ha='center',va='center',fontsize=13,weight='bold',wrap=True)
        for col,(kind,e) in enumerate(zip(('PCA','UMAP'),coords),start=1):
            ax=axes[row,col]; sc=ax.scatter(e[:,0],e[:,1],c=np.arange(n),s=11,cmap='plasma'); ax.plot(e[:,0],e[:,1],color='#334155',lw=.4,alpha=.5); ax.set(title=f'{name}: {kind}',xlabel=f'{kind} 1',ylabel=f'{kind} 2'); fig.colorbar(sc,ax=ax,label='rollout step')
    # Keep the map at upper left and label the remaining two left cells clearly.
    axes[1,0].set(title='Same held-out observations',xticks=[],yticks=[]); axes[2,0].set(title='Same held-out observations',xticks=[],yticks=[])
    path=out / f'actor_q_representation_panel_{stem}.png'; fig.savefig(path,dpi=180); plt.close(fig)
    print(path)


if __name__ == '__main__': main()
