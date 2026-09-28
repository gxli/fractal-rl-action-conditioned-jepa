#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=(/home/gxli/bin/micromamba run -n astro_jepa python)
for run in ar4k128_sac_shared_source_a3 ar4k128_jepa_shared_source_a3; do
  for step in 000002000 000006000 000010000 000014000 000020000; do
    "${PYTHON[@]}" -m driver.analyze_sac_actor_critics "dumps/${run}/checkpoints/sac_${step}.pt" --device cuda --seed 100001 --steps 500 --output-dir "dumps/${run}"
  done
  "${PYTHON[@]}" -m driver.plot_dense_latent_maps "dumps/${run}/checkpoints/sac_*.pt" \
    --device cuda --seed 100001 --steps 500 --resolution 128 --batch-size 512 --limit 5 \
    --encoder actor --output-dir "dumps/${run}"
done
