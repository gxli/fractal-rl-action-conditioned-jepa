#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m driver.train_sac --device cuda --overlay config/sac_actionrepeat4.yaml --overlay config/sac_jepa_separate_k32.yaml --set training.total_timesteps=20000 --set training.evaluation_interval=2000 --set training.checkpoint_interval=2000 --set output.dump_dir=dumps/ar4_jepa_separate_k32_source_a3
