import argparse
import json
from pathlib import Path
import yaml
from src.rl.trainer import load_policy, evaluate


def apply_override(cfg, entry):
    key, sep, raw = entry.partition('=')
    if not sep:
        raise ValueError(f'Override must be section.key=value: {entry}')
    target = cfg
    parts = key.split('.')
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise KeyError(key)
        target = target[part]
    if parts[-1] not in target:
        raise KeyError(key)
    target[parts[-1]] = yaml.safe_load(raw)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--device', default='cpu')
    p.add_argument('--episodes-per-seed', type=int, default=1)
    p.add_argument('--set', dest='overrides', action='append', default=[],
                   help='Evaluation-only config override, e.g. terrain.spectral_exponent=2.0')
    p.add_argument('--output', help='Write JSON here instead of the source run directory')
    a = p.parse_args()
    policy, cfg = load_policy(a.checkpoint, a.device)
    for entry in a.overrides:
        apply_override(cfg, entry)
    mean, std = evaluate(policy, cfg, a.device, a.episodes_per_seed)
    result = {'mean_speed': mean, 'std_speed': std, 'episodes_per_seed': a.episodes_per_seed}
    dest = Path(a.output) if a.output else Path(cfg['output']['dump_dir']) / 'metrics' / 'evaluation.json'
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
