"""Temporal actor/critic CKA for checkpoint directories on fixed observations."""
import argparse, re
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, torch
from driver.analyze_sac_actor_critics import load
from src.env.fractal_env import FractalEnv
from src.rl.policy import tensor_obs
from src.terrain.pink_noise import generate

def cka(x,y):
 x=x-x.mean(0); y=y-y.mean(0); return float(np.linalg.norm(x.T@y,'fro')**2/max(np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro'),1e-12))
def fixed_obs(cfg, device, n=400, seed=100001):
 env=FractalEnv(cfg,generate(cfg['terrain'],seed),seed); obs,_=env.reset(seed=seed); rng=np.random.default_rng(77); out=[]
 for _ in range(n): out.append(obs); obs,*_=env.step(rng.uniform(-1,1,2).astype('float32'))
 return out
def main():
 p=argparse.ArgumentParser(); p.add_argument('runs',nargs='+'); p.add_argument('--device',default='cpu'); p.add_argument('--output',required=True); a=p.parse_args()
 fig,axes=plt.subplots(1,len(a.runs),figsize=(5.1*len(a.runs),3.7),squeeze=False,constrained_layout=True)
 for ax,run in zip(axes[0],a.runs):
  files=sorted(Path(run,'checkpoints').glob('sac_*.pt')); actor,q,cfg=load(str(files[0]),a.device); observations=fixed_obs(cfg,a.device); xs=[]; ys=[]
  for f in files:
   actor,q,_=load(str(f),a.device); bank=[[],[],[]]
   with torch.no_grad():
    for o in observations:
     t,s=tensor_obs([o],a.device)
     for b,m in zip(bank,(actor,q.q1,q.q2)): b.append(m.encode(t,s)[0].cpu().numpy())
   x=np.asarray(bank); xs.append(int(re.search(r'(\d+)',f.stem).group(1))/1000); ys.append((cka(x[0],x[1]),cka(x[0],x[2]),cka(x[1],x[2])))
  y=np.asarray(ys); ax.plot(xs,y[:,0],'-o',label='actor--Q1'); ax.plot(xs,y[:,1],'-o',label='actor--Q2'); ax.plot(xs,y[:,2],'-o',label='Q1--Q2'); ax.set(title=Path(run).name,xlabel='checkpoint (thousands)',ylabel='linear CKA',ylim=(0,1.05)); ax.grid(alpha=.25); ax.legend(frameon=False,fontsize=8)
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=200)
if __name__=='__main__': main()
