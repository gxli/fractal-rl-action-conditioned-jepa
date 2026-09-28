"""Run from project root: python -m driver.train --set training.total_timesteps=10000"""
import argparse
from src.utils.config import load_config
from src.rl.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--overlay', action='append', default=[])
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train(load_config(args.config, args.overlay, args.overrides), device=args.device)


if __name__ == '__main__':
    main()
