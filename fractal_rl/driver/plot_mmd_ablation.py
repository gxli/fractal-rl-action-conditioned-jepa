"""Four-panel Gaussian-MMD/KerJEPA coefficient sweep."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    with open(path, newline='') as f: rows = list(csv.DictReader(f))
    return {k: np.asarray([float(r[k]) for r in rows]) for k in ('steps', 'rollout_mean_speed', 'evaluation_mean_speed', 'evaluation_std_speed')}


def add(ax, path, label, metric, shade=False):
    d=read(path); grid=np.arange(2000, min(20000, int(d['steps'][-1]))+1, 2000)
    x=grid/1000; y=np.interp(grid,d['steps'],d[metric]); line,=ax.plot(x,y,marker='o',ms=3,lw=1.8,label=label)
    if shade and metric == 'evaluation_mean_speed':
        s=np.interp(grid,d['steps'],d['evaluation_std_speed']); ax.fill_between(x,y-s,y+s,color=line.get_color(),alpha=.12)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='dumps'); ap.add_argument('--output',default='output/pdf/mmd_strength_ablation.png'); args=ap.parse_args(); root=Path(args.root)
    arms=[('Vanilla SAC',root/'sac_50k_control_v2/metrics/training.csv',root/'sac_frozen_curve50k_alpha2/metrics/training.csv')]
    for token,coef in [('005','.005'),('020','.02'),('050','.05')]: arms.append((f'Gaussian MMD $\\lambda={coef}$',root/f'sac_jepa_mmd_{token}_source/metrics/training.csv',root/f'sac_jepa_mmd_{token}_frozen_alpha2/metrics/training.csv'))
    fig,axes=plt.subplots(2,2,figsize=(12,7.8),constrained_layout=True,sharex=True)
    panels=[(axes[0,0],1,'rollout_mean_speed','Source alpha=3: training speed'),(axes[0,1],1,'evaluation_mean_speed','Source alpha=3: held-out speed'),(axes[1,0],2,'rollout_mean_speed','Frozen transfer alpha=2: training speed'),(axes[1,1],2,'evaluation_mean_speed','Frozen transfer alpha=2: held-out speed')]
    for ax,index,metric,title in panels:
        auxiliary_drawn=False
        for label,source,transfer in arms:
            path=(source,transfer)[index-1]
            if path.exists():
                add(ax,path,label,metric,metric.startswith('evaluation'))
                auxiliary_drawn |= label != 'Vanilla SAC'
        ax.set(title=title,xlabel='transitions (thousands)',ylabel='mean speed',xlim=(1.5,20.5)); ax.grid(alpha=.25)
        if index == 2 and not auxiliary_drawn:
            ax.text(.5,.55,'MMD transfers queued\nafter 20k source checkpoints',transform=ax.transAxes,
                    ha='center',va='center',color='#64748b',fontsize=10)
    axes[0,1].legend(frameon=False,fontsize=8,loc='lower right'); fig.suptitle('Gaussian-MMD/KerJEPA strength sweep (common 2k grid)',weight='bold')
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=220); print(out)


if __name__=='__main__': main()
