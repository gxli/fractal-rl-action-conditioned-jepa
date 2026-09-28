#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=(/home/gxli/bin/micromamba run -n astro_jepa python)
for encoder in actor q1; do
  "${PYTHON[@]}" -m driver.plot_dense_latent_maps \
    "dumps/ar4k128_sac_source_a3/checkpoints/sac_*.pt" \
    --device cuda --seed 100001 --steps 500 --resolution 128 --batch-size 512 \
    --limit 5 --encoder "$encoder" --output-dir dumps/ar4k128_sac_source_a3
done
