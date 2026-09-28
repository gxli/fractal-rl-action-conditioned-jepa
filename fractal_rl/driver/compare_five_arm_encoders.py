"""Five-run source CKA shown as pairwise 3x3 actor/Q blocks."""
import argparse
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
def main():
 p=argparse.ArgumentParser(); p.add_argument('checkpoints',nargs=5); p.add_argument('--device',default='cpu'); p.add_argument('--output',required=True); a=p.parse_args(); names=('SAC','JEPA','variance','SIGReg','spherical MMD')
 loaded=[load(x,a.device) for x in a.checkpoints]; cfg=loaded[0][2]; env=FractalEnv(cfg,generate(cfg['terrain'],100001),100001); o,_=env.reset(seed=100001); rng=np.random.default_rng(77); obs=[]
 for _ in range(400): obs.append(o); o,*_=env.step(rng.uniform(-1,1,2).astype('float32'))
 z=[]
 with torch.no_grad():
  for actor,q,_ in loaded:
   b=[[],[],[]]
   for o in obs:
    t,s=tensor_obs([o],a.device)
    for x,m in zip(b,(actor,q.q1,q.q2)): x.append(m.encode(t,s)[0].cpu().numpy())
   z.append(np.asarray(b))
 matrix=np.array([[cka(x,y) for y in z for y in y] for x in []]) if False else np.block([[np.array([[cka(z[i][u],z[j][v]) for v in range(3)] for u in range(3)]) for j in range(5)] for i in range(5)])
 fig,axs=plt.subplots(5,5,figsize=(12,11),constrained_layout=True)
 for i in range(5):
  for j in range(5):
   b=matrix[3*i:3*i+3,3*j:3*j+3]; ax=axs[i,j]; im=ax.imshow(b,vmin=0,vmax=1,cmap='magma'); ax.set(title=f'{names[i]} × {names[j]}',xticks=[],yticks=[])
   for u in range(3):
    for v in range(3): ax.text(v,u,f'{b[u,v]:.2f}',ha='center',va='center',fontsize=5.8,color='white' if b[u,v]<.55 else 'black')
 fig.colorbar(im,ax=axs,shrink=.6,label='linear CKA'); fig.suptitle('Five-arm source encoder similarity: each tile is actor/Q1/Q2 × actor/Q1/Q2')
 out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=200)
if __name__=='__main__': main()
