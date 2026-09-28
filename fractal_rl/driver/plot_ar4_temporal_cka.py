"""Checkpoint-to-checkpoint CKA on one fixed probe for SAC vs repeat-4 JEPA."""
from pathlib import Path
import re, numpy as np, torch, matplotlib.pyplot as plt
from driver.analyze_sac_actor_critics import load
from driver.plot_checkpoint_cka import fixed_obs, cka
from src.rl.policy import tensor_obs

runs=[('dumps/ar4_sac_source_a3','SAC'),('dumps/ar4_jepa_separate_k32_source_a3','JEPA K=32, separate encoders'),('dumps/ar4_jepa_k32_source_a3','JEPA K=32, critic-trained shared')]
parts=[('actor',lambda a,q:a),('Q1',lambda a,q:q.q1),('Q2',lambda a,q:q.q2)]
fig,axs=plt.subplots(3,3,figsize=(12,10),constrained_layout=True)
for row,(run,title) in enumerate(runs):
 files=sorted(Path(run,'checkpoints').glob('sac_*.pt')); a,q,cfg=load(str(files[0]),'cpu'); obs=fixed_obs(cfg,'cpu',n=400); labels=[str(int(re.search(r'(\d+)',f.stem).group(1))//1000) for f in files]
 banks=[]
 for f in files:
  a,q,_=load(str(f),'cpu'); b=[[] for _ in parts]
  with torch.no_grad():
   for o in obs:
    t,s=tensor_obs([o],'cpu')
    for z,(_,fn) in zip(b,parts): z.append(fn(a,q).encode(t,s)[0].numpy())
  banks.append([np.asarray(z) for z in b])
 for col,(name,_) in enumerate(parts):
  m=np.array([[cka(x[col],y[col]) for y in banks] for x in banks]); ax=axs[row,col]; im=ax.imshow(m,vmin=0,vmax=1,cmap='magma'); ax.set(title=f'{title}: {name}',xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,xlabel='checkpoint (k)',ylabel='checkpoint (k)')
  for i in range(len(labels)):
   for j in range(len(labels)): ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',fontsize=5,color='white' if m[i,j]<.55 else 'black')
fig.colorbar(im,ax=axs,label='centered linear CKA');Path('dumps/ar4_analysis').mkdir(exist_ok=True);fig.savefig('dumps/ar4_analysis/actionrepeat4_temporal_cka.png',dpi=220)
