"""Fair actor/critic alignment: independent SAC vs independent JEPA encoders."""
from pathlib import Path
import numpy as np, torch, matplotlib.pyplot as plt
from driver.analyze_sac_actor_critics import load
from driver.plot_checkpoint_cka import fixed_obs, cka
from src.rl.policy import tensor_obs
runs=[('dumps/ar4_sac_source_a3','SAC'),('dumps/ar4_jepa_separate_k32_source_a3','separate-encoder JEPA K=32')]
out=Path('dumps/ar4_analysis');out.mkdir(exist_ok=True)
fig,axs=plt.subplots(1,2,figsize=(8,3.5),constrained_layout=True)
for ax,(run,label) in zip(axs,runs):
 a,q,cfg=load(str(Path(run)/'checkpoints/sac_000020000.pt'),'cpu'); obs=fixed_obs(cfg,'cpu',n=500); z=[[],[],[]]
 with torch.no_grad():
  for o in obs:
   t,s=tensor_obs([o],'cpu')
   for dst,m in zip(z,(a,q.q1,q.q2)):dst.append(m.encode(t,s)[0].numpy())
 z=[np.asarray(x) for x in z]; m=np.array([[cka(x,y) for y in z] for x in z]); im=ax.imshow(m,vmin=0,vmax=1,cmap='magma')
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{m[i,j]:.2f}',ha='center',va='center',color='white' if m[i,j]<.55 else 'black')
 ax.set(title=label,xticks=range(3),yticks=range(3),xticklabels=['actor','Q1','Q2'],yticklabels=['actor','Q1','Q2'])
fig.colorbar(im,ax=axs,label='centered linear CKA');fig.savefig(out/'actionrepeat4_fair_encoder_alignment.png',dpi=220)
