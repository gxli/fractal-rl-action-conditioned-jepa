import argparse
import json
from pathlib import Path
from src.rl.trainer import load_policy, evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--episodes-per-seed', type=int, default=1)
    a = p.parse_args()
    policy, cfg = load_policy(a.checkpoint, a.device)
    mean, std = evaluate(policy, cfg, a.device, a.episodes_per_seed)
    result = {'mean_speed': mean, 'std_speed': std, 'episodes_per_seed': a.episodes_per_seed}
    dest = Path(cfg['output']['dump_dir']) / 'metrics' / 'evaluation.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
