"""Train SAC with the same FractalEnv and output conventions as PPO."""
import argparse
from src.rl.sac_trainer import train_sac
from src.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config'); parser.add_argument('--overlay', action='append', default=[]); parser.add_argument('--set', dest='overrides', action='append', default=[]); parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    train_sac(load_config(args.config, overlays=['config/sac.yaml', *args.overlay], overrides=args.overrides), args.device)


if __name__ == '__main__': main()
