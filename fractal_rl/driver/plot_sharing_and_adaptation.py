import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
R=Path('dumps')
def load(p):
 with open(p) as f: rows=list(csv.DictReader(f))
 return {k:np.array([float(x[k]) for x in rows]) for k in rows[0]}
def line(ax,p,label,color):
 d=load(p); x=d['steps']/1000; ax.plot(x,d['evaluation_mean_speed'],'-o',ms=3,label=label,color=color); ax.fill_between(x,d['evaluation_mean_speed']-d['evaluation_std_speed'],d['evaluation_mean_speed']+d['evaluation_std_speed'],color=color,alpha=.1)
def main():
 out=R/'minstart_v2_analysis';out.mkdir(exist_ok=True)
 fig,axs=plt.subplots(1,3,figsize=(12,3.4),constrained_layout=True)
 arms=[('minstart_v2_sac_jepa_source_a3','Separate JEPA','#dc2626'),('minstart_share_fully_source_a3','Fully shared','#7c3aed'),('minstart_share_critic_source_a3','Critic-trained','#059669')]
 for d,l,c in arms: line(axs[0],R/d/'metrics/training.csv',l,c)
 for d,l,c in [('minstart_v2_sac_jepa_frozen_a2','Separate JEPA','#dc2626'),('minstart_share_fully_frozen_a2','Fully shared','#7c3aed'),('minstart_share_critic_frozen_a2','Critic-trained','#059669')]: line(axs[1],R/d/'metrics/training.csv',l,c)
 for d,l,c in [('minstart_v2_sac_jepa_frozen_a35','Separate JEPA','#dc2626'),('minstart_share_fully_frozen_a35','Fully shared','#7c3aed'),('minstart_share_critic_frozen_a35','Critic-trained','#059669')]: line(axs[2],R/d/'metrics/training.csv',l,c)
 for ax,t in zip(axs,['Source α=3','Frozen α=2','Frozen α=3.5']): ax.set(title=t,xlabel='transitions (thousands)',ylabel='held-out speed',xlim=(0,20));ax.grid(alpha=.2)
 axs[0].legend(frameon=False,fontsize=8);fig.savefig(out/'sharing_curves.png',dpi=180)
 fig,axs=plt.subplots(1,2,figsize=(8,3.4),constrained_layout=True)
 for ax,stage,label in zip(axs,['a2','a35'],['α=2','α=3.5']):
  line(ax,R/f'minstart_v2_sac_jepa_frozen_{stage}'/'metrics/training.csv','both encoders frozen','#dc2626');line(ax,R/f'minstart_actor_adapts_jepa_frozencritic_{stage}'/'metrics/training.csv','critic frozen; actor adapts','#2563eb');ax.set(title=f'JEPA transfer {label}',xlabel='transitions (thousands)',ylabel='held-out speed',xlim=(0,20));ax.grid(alpha=.2)
 axs[0].legend(frameon=False,fontsize=8);fig.savefig(out/'freeze_ablation.png',dpi=180)
if __name__=='__main__':main()
