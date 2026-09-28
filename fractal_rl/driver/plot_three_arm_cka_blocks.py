"""Render the three-arm CKA matrix as interpretable 3x3 run-pair blocks."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def main():
    root = Path('dumps/minstart_v2_analysis')
    data = np.load(root / 'three_arm_encoder_cka.npz')
    cka = data['cka']
    runs = ('SAC', 'SAC+JEPA', 'SAC+JEPA+variance')
    parts = ('actor', 'Q1', 'Q2')
    fig, axes = plt.subplots(3, 3, figsize=(9.2, 8.5), constrained_layout=True)
    image = None
    for row, left in enumerate(runs):
        for col, right in enumerate(runs):
            block = cka[3*row:3*row+3, 3*col:3*col+3]
            ax = axes[row, col]
            image = ax.imshow(block, vmin=0, vmax=1, cmap='magma')
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, f'{block[i,j]:.2f}', ha='center', va='center', fontsize=8,
                            color='white' if block[i,j] < .55 else 'black')
            ax.set(xticks=range(3), yticks=range(3), xticklabels=parts if row == 2 else [],
                   yticklabels=parts if col == 0 else [], title=f'{left} × {right}')
    fig.colorbar(image, ax=axes, shrink=.78, label='centered linear CKA')
    fig.suptitle('Encoder similarity blocks on identical fresh minimum-start observations', fontsize=12)
    fig.savefig(root / 'three_arm_encoder_cka_blocks.png', dpi=200)


if __name__ == '__main__': main()
