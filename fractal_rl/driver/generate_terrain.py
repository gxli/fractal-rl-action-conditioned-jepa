import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from src.utils.config import load_config
from src.terrain.pink_noise import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--seed', type=int)
    a = p.parse_args()
    cfg = load_config(a.config)
    terrain = generate(cfg['terrain'], a.seed)
    dest = Path(cfg['output']['dump_dir']) / 'terrains'
    dest.mkdir(parents=True, exist_ok=True)
    seed = cfg['terrain']['seed'] if a.seed is None else a.seed
    np.savez_compressed(dest / f'terrain_{seed}.npz', intensity=terrain.intensity, potential=terrain.potential,
                        gradient_x=terrain.gradient_x, gradient_y=terrain.gradient_y,
                        gradient_reference=terrain.gradient_reference)
    normalized = terrain.potential / max(1e-8, float(terrain.potential.max()))
    Image.fromarray(np.uint8(np.clip(normalized, 0, 1) * 255)).save(dest / f'terrain_{seed}_potential.png')
    print(dest / f'terrain_{seed}.npz')


if __name__ == '__main__':
    main()
