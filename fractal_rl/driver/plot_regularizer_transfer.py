"""Plot frozen-encoder JEPA regularizer transfer curves."""
import csv
from pathlib import Path
import matplotlib.pyplot as plt

METHODS = [('vanilla', 'Vanilla SAC'), ('variance', 'SAC+JEPA (variance)'),
           ('visreg', 'VISReg'), ('kerjepa', 'KerJEPA'), ('susreg', 'SUSReg'), ('spherical_mmd', 'Spherical MMD'),
           ('sigreg', 'SIGReg (lambda=0.005)'), ('mmd', 'Gaussian MMD (lambda=0.005)')]
ALPHAS = [('15', '1.5'), ('2', '2'), ('4', '4')]

def read(path):
    with open(path, newline='') as f: rows=list(csv.DictReader(f))
    return [int(r['steps'])/1000 for r in rows], [float(r['evaluation_mean_speed']) for r in rows], [float(r['evaluation_std_speed']) for r in rows]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True, constrained_layout=True)
for ax, (key, label) in zip(axes, ALPHAS):
    for method, name in METHODS:
        # The selected SIGReg screen is currently a frozen alpha=2 transfer;
        # do not manufacture alpha=1.5/4 curves that were never run.
        if method in ('sigreg', 'mmd') and key != '2':
            continue
        if method == 'vanilla': path=f'dumps/sac_frozen_curve50k_alpha{key}/metrics/training.csv'
        elif method == 'variance': path=f'dumps/sac_jepa_actorcritic_frozen_curve50k_alpha{key}/metrics/training.csv'
        elif method == 'sigreg': path='dumps/sac_jepa_sigreg_005_frozen_alpha2/metrics/training.csv'
        elif method == 'mmd': path='dumps/sac_jepa_mmd_005_frozen_alpha2/metrics/training.csv'
        else: path=f'dumps/sac_jepa_reg_{method}_frozen_alpha{key}_transfer/metrics/training.csv'
        x,y,s=read(path)
        ax.plot(x,y,marker='o',linewidth=2,label=name); ax.fill_between(x,[a-b for a,b in zip(y,s)],[a+b for a,b in zip(y,s)],alpha=.12)
    ax.set(title=f'Transfer to alpha = {label}', xlabel='transfer transitions (thousands)'); ax.grid(alpha=.25)
axes[0].set_ylabel('held-out mean speed'); axes[-1].legend(frameon=False,fontsize=8)
fig.suptitle('Frozen-encoder SAC+JEPA transfer: latent regularizer ablation', weight='bold')
Path('output/pdf').mkdir(parents=True,exist_ok=True); fig.savefig('output/pdf/jepa_regularizer_transfer_curves.png',dpi=220)
