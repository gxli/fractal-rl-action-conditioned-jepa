import csv
from pathlib import Path
import matplotlib.pyplot as plt
R=Path('dumps'); O=R/'ar4_analysis'
def load(n):
 r=list(csv.DictReader(open(R/n/'metrics/training.csv'))); return {k:[float(x[k]) for x in r] for k in r[0]}
fig,axs=plt.subplots(1,3,figsize=(12,3.3),constrained_layout=True)
for ax,s,t in zip(axs,['source_a3','frozen_a2','frozen_a3.5'],['source alpha=3','frozen alpha=2','frozen alpha=3.5']):
 for n,l,c in [(f'ar4_jepa_k32_{s}','critic-trained shared','#dc2626'),(f'ar4_jepa_actorstop_k32_{s}','actor-owned, critic stop-grad','#2563eb')]:
  d=load(n); x=[v*4/1000 for v in d['steps']]; ax.plot(x,d['evaluation_mean_speed'],'-o',ms=3,label=l,color=c); ax.fill_between(x,[a-b for a,b in zip(d['evaluation_mean_speed'],d['evaluation_std_speed'])],[a+b for a,b in zip(d['evaluation_mean_speed'],d['evaluation_std_speed'])],color=c,alpha=.12)
 ax.set(title=t,xlabel='physics steps (thousands)',ylabel='held-out speed',xlim=(0,80));ax.grid(alpha=.2)
axs[0].legend(frameon=False,fontsize=7);O.mkdir(exist_ok=True);fig.savefig(O/'actionrepeat4_encoder_sharing_ablation.png',dpi=220)
