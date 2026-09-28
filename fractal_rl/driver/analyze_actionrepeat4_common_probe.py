"""Common-observation CKA and manifold diagnostics for repeat-4 sources."""
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from src.env.fractal_env import FractalEnv
from src.terrain.pink_noise import generate
from src.rl.policy import tensor_obs
from driver.analyze_sac_actor_critics import load

R=Path('dumps'); OUT=R/'ar4_analysis'; OUT.mkdir(exist_ok=True)
RUNS=[('ar4_sac_source_a3','SAC'),('ar4_jepa_k32_source_a3','JEPA K32'),('ar4_jepa_k64_source_a3','JEPA K64'),('ar4_jepa_spherical_k32_source_a3','JEPA+MMD K32')]
def cka(x,y):
 x=x-x.mean(0); y=y-y.mean(0); return np.linalg.norm(x.T@y,'fro')**2/(np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro')+1e-12)
def pr(x):
 v=np.linalg.eigvalsh(np.cov(x,rowvar=False)); return v.sum()**2/(v@v+1e-12)
def lid(x,k=10):
 d=np.sqrt(((x[:,None]-x[None,:])**2).sum(-1)); d.sort(1); r=d[:,1:k+1]; return float(np.mean((k-1)/np.maximum(np.log(r[:,-1,None]/np.maximum(r[:,:-1],1e-8)).sum(1),1e-8)))
def main():
 dev='cpu'; a0,q0,cfg=load(R/RUNS[0][0]/'checkpoints/sac_000020000.pt',dev); env=FractalEnv(cfg,terrain=generate(cfg['terrain'],seed=100001),seed=100001); obs,_=env.reset(seed=100001); probe=[]
 for _ in range(500):
  probe.append(obs); t,s=tensor_obs([obs],dev); act,*_=a0.act(t,s,deterministic=True); obs,*rest=env.step(act[0].detach().numpy())
 models=[]
 for run,label in RUNS:
  a,q,_=load(R/run/'checkpoints/sac_000020000.pt',dev); zs=[[],[],[]]
  for o in probe:
   t,s=tensor_obs([o],dev); zs[0].append(a.encode(t,s)[0].detach().numpy()); zs[1].append(q.q1.encode(t,s)[0].detach().numpy()); zs[2].append(q.q2.encode(t,s)[0].detach().numpy())
  models += [(label+' actor',np.array(zs[0])),(label+' Q1',np.array(zs[1])),(label+' Q2',np.array(zs[2]))]
 M=np.array([[cka(x,y) for _,y in models] for _,x in models]); fig,ax=plt.subplots(figsize=(9,8)); im=ax.imshow(M,vmin=0,vmax=1,cmap='magma'); ax.set(xticks=range(12),yticks=range(12),xticklabels=[n.replace(' ','\n') for n,_ in models],yticklabels=[n for n,_ in models]); plt.setp(ax.get_xticklabels(),rotation=90,fontsize=6); plt.setp(ax.get_yticklabels(),fontsize=7); fig.colorbar(im,ax=ax,label='centered linear CKA'); fig.tight_layout(); fig.savefig(OUT/'actionrepeat4_common_probe_cka.png',dpi=200)
 vals=[(n,pr(z),lid(z)) for n,z in models]; fig,ax=plt.subplots(1,2,figsize=(11,3.5)); x=np.arange(12); ax[0].bar(x,[v[1] for v in vals]);ax[1].bar(x,[v[2] for v in vals]);
 for a,title in zip(ax,['participation-ratio effective dimension','kNN local intrinsic dimension']): a.set(title=title,xticks=x,xticklabels=[n.replace(' ','\n') for n,_,_ in vals]); a.tick_params(axis='x',labelsize=6)
 fig.tight_layout();fig.savefig(OUT/'actionrepeat4_manifold_diagnostics.png',dpi=200)
if __name__=='__main__': main()
