#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
common=(--device cuda --overlay config/sac_actionrepeat4.yaml --overlay config/sac_jepa_actorstop_k32.yaml --set training.total_timesteps=20000 --set training.evaluation_interval=2000 --set training.checkpoint_interval=2000)
run(){ local n="$1"; shift; [[ -e "dumps/$n/metrics/training.csv" ]] && return; python -m driver.train_sac "${common[@]}" --set "output.dump_dir=dumps/$n" "$@"; }
run ar4_jepa_actorstop_k32_source_a3
for a in 2.0 3.5; do tag="${a/.0/}"; run "ar4_jepa_actorstop_k32_frozen_a${tag}" --set training.init_checkpoint=dumps/ar4_jepa_actorstop_k32_source_a3/checkpoints/sac_000020000.pt --set training.freeze_encoder=true --set terrain.spectral_exponent="$a"; done
