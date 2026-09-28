"""Cross-run actor/Q encoder CKA on one common held-out SAC trajectory."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from driver.analyze_sac_actor_critics import load
from src.env.fractal_env import FractalEnv
from src.rl.policy import tensor_obs
from src.terrain.pink_noise import generate


def linear_cka(x, y):
    x=x-x.mean(0,keepdims=True); y=y-y.mean(0,keepdims=True)
    return float(np.linalg.norm(x.T@y,'fro')**2 / max(np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro'),1e-12))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('vanilla'); ap.add_argument('jepa'); ap.add_argument('jepa_reg', nargs='?'); ap.add_argument('--device',default='cpu'); ap.add_argument('--seed',type=int,default=100001); ap.add_argument('--steps',type=int,default=600); ap.add_argument('--output',default='output/pdf/clean_sac_jepa_encoder_cka.png'); args=ap.parse_args()
    vanilla,vq,cfg=load(args.vanilla,args.device); jepa,jq,_=load(args.jepa,args.device)
    reg,rq,_ = load(args.jepa_reg,args.device) if args.jepa_reg else (None,None,None)
    terrain=generate(cfg['terrain'],seed=args.seed); env=FractalEnv(cfg,terrain=terrain,seed=args.seed); obs,_=env.reset(seed=args.seed)
    groups=[('SAC',vanilla,vq), ('JEPA',jepa,jq)] + ([] if reg is None else [('JEPA+reg',reg,rq)])
    banks=[[] for _ in range(3*len(groups))]
    with torch.no_grad():
        for _ in range(args.steps):
            t,s=tensor_obs([obs],args.device)
            for block,(_name,actor,critics) in enumerate(groups):
                for index,model in enumerate((actor,critics.q1,critics.q2)): banks[3*block+index].append(model.encode(t,s)[0].cpu().numpy())
            action,*_=vanilla.act(t,s,deterministic=True)
            obs,_,term,trunc,_=env.step(action[0].cpu().numpy())
            if term or trunc: break
    arrays=[np.asarray(x,np.float32) for x in banks]; cka=np.asarray([[linear_cka(x,y) for y in arrays] for x in arrays])
    labels=[f'{name} {part}' for name,_,_ in groups for part in ('actor','Q1','Q2')]
    count=len(labels)
    fig,ax=plt.subplots(figsize=(8.8 if count==9 else 7,7.8 if count==9 else 6),constrained_layout=True); image=ax.imshow(cka,vmin=0,vmax=1,cmap='magma'); ax.set(xticks=range(count),yticks=range(count),xticklabels=labels,yticklabels=labels,title='Encoder similarity on identical held-out observations (linear CKA)'); ax.tick_params(axis='x',rotation=42)
    for i in range(count):
        for j in range(count): ax.text(j,i,f'{cka[i,j]:.2f}',ha='center',va='center',fontsize=7.2,color='white' if cka[i,j]<.55 else 'black')
    fig.colorbar(image,ax=ax,label='linear CKA (1 = same representational geometry)')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=220); np.savez_compressed(out.with_suffix('.npz'),cka=cka,labels=np.asarray(labels),latents=np.asarray(arrays)); print(out)


if __name__=='__main__': main()
