"""Four-panel spherical-MMD reproduction against vanilla and variance-JEPA."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with Path(path).open(newline='') as f: rows=list(csv.DictReader(f))
    return {k:np.asarray([float(r[k]) for r in rows]) for k in ('steps','rollout_mean_speed','evaluation_mean_speed','evaluation_std_speed')}


def add(ax,path,label,metric,color):
    if not Path(path).exists(): return False
    d=read(path); grid=np.arange(2000,min(20000,int(d['steps'][-1]))+1,2000)
    if not len(grid): return False
    y=np.interp(grid,d['steps'],d[metric]); s=np.interp(grid,d['steps'],d['evaluation_std_speed'])
    ax.plot(grid/1000,y,marker='o',ms=3,lw=2,label=label,color=color)
    if metric=='evaluation_mean_speed': ax.fill_between(grid/1000,y-s,y+s,color=color,alpha=.12)
    return True


root=Path('dumps')
source=[('Vanilla SAC',root/'sac_50k_control_v2/metrics/training.csv','#1f77b4'),('SAC+JEPA (variance)',root/'sac_jepa_v3_50k/metrics/training.csv','#ff7f0e'),('SAC+JEPA spherical MMD',root/'sac_spherical_mmd_repro_alpha3_source/metrics/training.csv','#7c3aed')]
transfers={2:[('Vanilla SAC',root/'sac_frozen_curve50k_alpha2/metrics/training.csv','#1f77b4'),('SAC+JEPA (variance)',root/'sac_jepa_actorcritic_frozen_curve50k_alpha2/metrics/training.csv','#ff7f0e'),('SAC+JEPA spherical MMD',root/'sac_spherical_mmd_repro_alpha2_frozen/metrics/training.csv','#7c3aed')],4:[('Vanilla SAC',root/'sac_frozen_curve50k_alpha4/metrics/training.csv','#1f77b4'),('SAC+JEPA (variance)',root/'sac_jepa_actorcritic_frozen_curve50k_alpha4/metrics/training.csv','#ff7f0e'),('SAC+JEPA spherical MMD',root/'sac_spherical_mmd_repro_alpha4_frozen/metrics/training.csv','#7c3aed')]}
fig,axes=plt.subplots(2,2,figsize=(12,7.8),constrained_layout=True,sharex=True)
for ax,metric,title in [(axes[0,0],'rollout_mean_speed','Source alpha=3: rollout speed'),(axes[0,1],'evaluation_mean_speed','Source alpha=3: held-out speed')]:
    for label,path,color in source: add(ax,path,label,metric,color)
    ax.set_title(title)
for ax,alpha in [(axes[1,0],2),(axes[1,1],4)]:
    for label,path,color in transfers[alpha]: add(ax,path,label,'evaluation_mean_speed',color)
    ax.set_title(f'Frozen transfer to alpha={alpha}: held-out speed')
for ax in axes.flat: ax.set(xlabel='transitions (thousands)',ylabel='mean speed',xlim=(1.5,20.5)); ax.grid(alpha=.25)
axes[0,1].legend(frameon=False,fontsize=8,loc='lower right')
fig.suptitle('Spherical-MMD reproduction: alpha=3 source, frozen transfer to alpha=2 and 4',weight='bold')
Path('output/pdf').mkdir(parents=True,exist_ok=True); fig.savefig('output/pdf/spherical_mmd_alpha_reproduction.png',dpi=220)
