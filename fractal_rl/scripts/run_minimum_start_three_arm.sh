#!/usr/bin/env bash
# Fresh, fair source runs.  Each arm uses deterministic global-minimum starts.
# Run from the repository root, preferably inside the GPU environment.
set -euo pipefail

steps="${1:-20000}"
common=(--device cuda --set "training.total_timesteps=${steps}" --set output.dump_dir=PLACEHOLDER)

python -m driver.train_sac "${common[@]/PLACEHOLDER/dumps/minstart_sac_source_a3}"
python -m driver.train_sac --overlay config/sac_jepa_only.yaml "${common[@]/PLACEHOLDER/dumps/minstart_sac_jepa_source_a3}"
python -m driver.train_sac --overlay config/sac_jepa_reg_variance.yaml "${common[@]/PLACEHOLDER/dumps/minstart_sac_jepa_reg_source_a3}"
