#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base=(--device cuda --overlay config/sac_actionrepeat4.yaml --overlay config/sac_sharedcritic_vanilla.yaml --set training.total_timesteps=20000 --set training.evaluation_interval=2000 --set training.checkpoint_interval=2000)
python -m driver.train_sac "${base[@]}" --set output.dump_dir=dumps/ar4_sac_sharedcritic_source_a3
for a in 2.0 3.5; do tag="${a/.0/}"; python -m driver.train_sac "${base[@]}" --set output.dump_dir="dumps/ar4_sac_sharedcritic_frozen_a${tag}" --set training.init_checkpoint=dumps/ar4_sac_sharedcritic_source_a3/checkpoints/sac_000020000.pt --set training.freeze_encoder=true --set terrain.spectral_exponent="$a"; done
