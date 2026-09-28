#!/usr/bin/env bash
# Serial 3090 queue for the longer-effective-horizon SAC/JEPA experiment.
#
# Each policy transition holds the action for four 20 ms physics updates:
#   decision interval = 0.08 s; gamma = 0.99^4 = 0.96059601.
# Therefore K=32 and K=64 span 2.56 s and 5.12 s respectively.  With the
# current dynamics those intervals are approximately 7.7 px and 15.4 px at
# the reference 3 px/s speed.  Runs use 20k decision transitions; the report
# will label this as 80k physics steps to keep the physical budget explicit.
#
# Run from the repository root:
#   nohup bash scripts/run_actionrepeat4_queue.sh > dumps/actionrepeat4_queue.log 2>&1 &
set -euo pipefail

steps="${1:-20000}"
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
mkdir -p dumps

common=(
  --device cuda
  --overlay config/sac_actionrepeat4.yaml
  --set "training.total_timesteps=${steps}"
  --set training.evaluation_interval=2000
  --set training.checkpoint_interval=2000
)

run_source() {
  local name="$1"; shift
  if [[ -e "dumps/${name}/metrics/training.csv" ]]; then
    echo "[$(date -Is)] SKIP completed/existing ${name}"
    return
  fi
  echo "[$(date -Is)] START ${name}"
  python -m driver.train_sac "${common[@]}" "$@" --set "output.dump_dir=dumps/${name}"
  echo "[$(date -Is)] DONE ${name}"
}

run_transfer() {
  local name="$1" checkpoint="$2" alpha="$3"; shift 3
  if [[ -e "dumps/${name}/metrics/training.csv" ]]; then
    echo "[$(date -Is)] SKIP completed/existing ${name}"
    return
  fi
  echo "[$(date -Is)] START ${name} (frozen encoder, alpha=${alpha})"
  python -m driver.train_sac "${common[@]}" "$@" \
    --set "output.dump_dir=dumps/${name}" \
    --set "training.init_checkpoint=${checkpoint}" \
    --set training.freeze_encoder=true \
    --set training.critic_jepa_coef=0.0 \
    --set "terrain.spectral_exponent=${alpha}"
  echo "[$(date -Is)] DONE ${name}"
}

# Source terrain: alpha = 3.  SAC is the action-repeat baseline; JEPA arms
# use the critic-trained shared encoder to keep representation access fair.
run_source ar4_sac_source_a3
run_source ar4_jepa_k32_source_a3 --overlay config/sac_jepa_criticshared_k32.yaml
run_source ar4_jepa_k64_source_a3 --overlay config/sac_jepa_criticshared_k64.yaml
run_source ar4_jepa_spherical_k32_source_a3 \
  --overlay config/sac_jepa_criticshared_k32.yaml \
  --overlay config/sac_jepa_reg_spherical_mmd.yaml \
  --set training.critic_jepa_coef=0.0 --set training.jepa_horizon=32

# Transfer: freeze both actor and critic encoders; only policy/value heads
# adapt.  This isolates representation transfer across spectral exponents.
for alpha in 2.0 3.5; do
  tag="${alpha/.0/}"
  run_transfer "ar4_sac_frozen_a${tag}" \
    dumps/ar4_sac_source_a3/checkpoints/sac_000020000.pt "$alpha"
  run_transfer "ar4_jepa_k32_frozen_a${tag}" \
    dumps/ar4_jepa_k32_source_a3/checkpoints/sac_000020000.pt "$alpha" \
    --overlay config/sac_jepa_criticshared_k32.yaml
  run_transfer "ar4_jepa_k64_frozen_a${tag}" \
    dumps/ar4_jepa_k64_source_a3/checkpoints/sac_000020000.pt "$alpha" \
    --overlay config/sac_jepa_criticshared_k64.yaml
  run_transfer "ar4_jepa_spherical_k32_frozen_a${tag}" \
    dumps/ar4_jepa_spherical_k32_source_a3/checkpoints/sac_000020000.pt "$alpha" \
    --overlay config/sac_jepa_criticshared_k32.yaml \
    --overlay config/sac_jepa_reg_spherical_mmd.yaml \
    --set training.jepa_horizon=32
done

echo "[$(date -Is)] QUEUE COMPLETE"
