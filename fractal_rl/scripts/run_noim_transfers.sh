#!/usr/bin/env bash
# Start only after the matching no-inverse/no-margin source checkpoint exists.
set -euo pipefail
cd "$(dirname "$0")/.."

for spec in "variance 010 0.01" "sigreg 005 0.005" "spherical_mmd 010 0.01"; do
  read -r method token coef <<< "$spec"
  source_path="dumps/sac_noim_${method}_${token}_source_a3/checkpoints/sac_000020000.pt"
  while [[ ! -f "$source_path" ]]; do sleep 20; done
  for target in "2 2.0" "35 3.5"; do
    read -r tag alpha <<< "$target"
    output="dumps/sac_noim_${method}_${token}_frozen_a${tag}"
    mkdir -p "$output"
    /home/gxli/bin/micromamba run -n astro_jepa python -m driver.train_sac \
      --config config/sac_jepa_v3.yaml --device cuda \
      --set terrain.spectral_exponent="$alpha" \
      --set training.total_timesteps=20000 \
      --set training.evaluation_interval=2000 \
      --set training.checkpoint_interval=20000 \
      --set training.jepa_regularizer="$method" \
      --set training.jepa_regularizer_coef="$coef" \
      --set training.jepa_variance_coef=0.0 \
      --set training.freeze_encoder=true \
      --set training.init_checkpoint="$source_path" \
      --set output.dump_dir="$output" > "$output/train.log" 2>&1 &
  done
done
wait
