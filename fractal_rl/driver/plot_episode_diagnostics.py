"""Make x(t), y(t), latent-speed and map-trajectory plots for inference episodes."""
import argparse
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.env.fractal_env import FractalEnv
from src.rl.policy import tensor_obs
from src.rl.trainer import load_policy
from src.terrain.pink_noise import generate
from src.terrain.periodic_field import sample

def step(path): return int(path.stem.rsplit('_', 1)[-1])
def main():
 p=argparse.ArgumentParser(); p.add_argument('checkpoints'); p.add_argument('--device',default='cpu'); p.add_argument('--seed',type=int,default=100001); p.add_argument('--steps',type=int,default=2000); p.add_argument('--output-dir',default=None); a=p.parse_args()
 for ckpt in sorted(Path().glob(a.checkpoints),key=step):
  policy,cfg=load_policy(ckpt,a.device); policy.eval(); terrain=generate(cfg['terrain'],a.seed); env=FractalEnv(cfg,terrain,a.seed); obs,_=env.reset(seed=a.seed); xy=[]; z=[]
  for _ in range(a.steps):
   with torch.no_grad():
    ti,si=tensor_obs([obs],a.device); z.append(policy.encode(ti,si)[0].cpu().numpy()); act,_,_,_=policy.act(ti,si,deterministic=True)
   obs,_,term,trunc,_=env.step(act[0].cpu().numpy()); xy.append(env.body.position.copy())
   if term or trunc: break
  xy=np.asarray(xy); z=np.asarray(z); ls=np.r_[0.,np.linalg.norm(np.diff(z,axis=0),axis=1)]; t=np.arange(len(xy)); root=Path(a.output_dir or cfg['output']['dump_dir']); out=root/'plots'; out.mkdir(parents=True,exist_ok=True)
  fig,ax=plt.subplots(2,2,figsize=(10,7),constrained_layout=True); ax[0,0].plot(t,xy[:,0]); ax[0,0].set(title='x(t)',xlabel='step',ylabel='x'); ax[0,1].plot(t,xy[:,1]); ax[0,1].set(title='y(t)',xlabel='step',ylabel='y'); ax[1,0].plot(t,ls,color='#7c3aed'); ax[1,0].set(title='Encoder latent speed',xlabel='step',ylabel=r'$||z_t-z_{t-1}||$')
  pad=12.; gx=np.linspace(xy[:,0].min()-pad,xy[:,0].max()+pad,150); gy=np.linspace(xy[:,1].min()-pad,xy[:,1].max()+pad,150); mx,my=np.meshgrid(gx,gy); ax[1,1].contourf(mx,my,sample(terrain.potential,mx,my,terrain.pixel_size),levels=24,cmap='viridis_r'); ax[1,1].plot(xy[:,0],xy[:,1],color='#ef4444'); ax[1,1].scatter(*xy[0],color='white',edgecolor='#111827',s=24); ax[1,1].set(title='x-y trajectory on potential map',aspect='equal',xticks=[],yticks=[])
  for aa in ax.flat: aa.grid(alpha=.2) if aa is not ax[1,1] else None
  dest=out/f'episode_diagnostics_{ckpt.stem}_seed{a.seed}.png'; fig.savefig(dest,dpi=180); plt.close(fig); print(dest)
if __name__=='__main__': main()
