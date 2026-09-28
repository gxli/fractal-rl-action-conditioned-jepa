"""Three visible gradient-routing diagrams for the sharing ablation."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def box(ax, xy, text, color):
    ax.add_patch(FancyBboxPatch(xy, .28, .13, boxstyle='round,pad=.015', fc=color, ec='#334155'))
    ax.text(xy[0]+.14, xy[1]+.065, text, ha='center', va='center', fontsize=9)
def arrow(ax, a, b, label='', dashed=False):
    ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', lw=1.5, linestyle='--' if dashed else '-'))
    if label: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+.03,label,ha='center',fontsize=7)
def main():
 fig,axs=plt.subplots(1,3,figsize=(11.2,3.6),constrained_layout=True)
 modes=[('Separate (current)','own encoder per branch'),('Fully shared','TD + actor + JEPA update encoder'),('Critic-trained shared','TD updates encoder; actor stop-gradient')]
 for ax,(title,note) in zip(axs,modes):
  ax.set(xlim=(0,1),ylim=(0,1),axis_off=True,title=title)
  if title.startswith('Separate'):
   box(ax,(.36,.82),'observation','#dbeafe'); box(ax,(.08,.55),'actor enc.','#fef3c7'); box(ax,(.64,.55),'critic enc.','#dcfce7'); box(ax,(.08,.28),'actor','#fce7f3'); box(ax,(.64,.28),'Q1 / Q2','#ede9fe'); arrow(ax,(.5,.82),(.22,.68)); arrow(ax,(.5,.82),(.78,.68)); arrow(ax,(.22,.55),(.22,.41)); arrow(ax,(.78,.55),(.78,.41))
  else:
   box(ax,(.36,.82),'observation','#dbeafe'); box(ax,(.36,.58),'shared encoder','#dcfce7'); box(ax,(.08,.28),'actor','#fce7f3'); box(ax,(.64,.28),'Q1 / Q2','#ede9fe'); arrow(ax,(.5,.82),(.5,.71)); arrow(ax,(.42,.58),(.22,.41),'z' if title.startswith('Fully') else 'sg[z]',title.startswith('Critic')); arrow(ax,(.58,.58),(.78,.41),'z');
   if title.startswith('Critic'): ax.text(.5,.48,'actor path detached',ha='center',fontsize=7,color='#b91c1c')
  ax.text(.5,.06,note,ha='center',fontsize=8,wrap=True)
 out=Path('dumps/minstart_v2_analysis/encoder_sharing_architectures.png'); out.parent.mkdir(parents=True,exist_ok=True); fig.savefig(out,dpi=200)
if __name__=='__main__': main()
