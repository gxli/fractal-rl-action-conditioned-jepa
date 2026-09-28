import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('dumps'); O=R/'ar4k128_analysis';O.mkdir(exist_ok=True)
arms=[
 ('sac','SAC (separate encoders)','#111827'),
 ('sac_shared','SAC (shared critic encoder)','#16a34a'),
 ('jepa_shared','SAC+JEPA K=128 (shared critic encoder)','#dc2626'),
 ('jepa_sigreg_shared','SAC+JEPA K=128 + SIGReg (shared critic encoder)','#7c3aed'),
]
fig,axs=plt.subplots(1,3,figsize=(12.5,3.5),constrained_layout=True)
fig.patch.set_facecolor('white')
fig.patch.set_alpha(1)
for ax,stage,title in zip(axs,['source_a3','frozen_a2','frozen_a3.5'],['source alpha=3','frozen alpha=2','frozen alpha=3.5']):
 for arm,label,c in arms:
  path=R/f'ar4k128_{arm}_{stage}'/'metrics/training.csv'
  if not path.exists():
   continue
  r=list(csv.DictReader(open(path)));x=[float(z['steps'])*4/1000 for z in r];y=[float(z['evaluation_mean_speed']) for z in r];s=[float(z['evaluation_std_speed']) for z in r]
  ax.plot(x,y,'-o',ms=3,label=label,color=c);ax.fill_between(x,[a-b for a,b in zip(y,s)],[a+b for a,b in zip(y,s)],color=c,alpha=.12)
  ax.set(title=title,xlabel='physics steps (thousands)',ylabel='held-out speed',xlim=(0,80));ax.grid(alpha=.2)
axs[0].legend(frameon=False,fontsize=7,loc='lower right')
fig.savefig(O/'k128_curves.png',dpi=220,facecolor='white',transparent=False)
fig.savefig(O/'k128_curves_white.png',dpi=220,facecolor='white',transparent=False)
