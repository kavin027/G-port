#!/usr/bin/env bash
set -euo pipefail

python run_external_baselines.py \
  --external-baselines original_sfcl rltune_style sailor_style straggler_whatif \
  --external-baseline-mode "${EXTERNAL_BASELINE_MODE:-network_stress}" \
  --external-baseline-seeds "${EXTERNAL_BASELINE_SEEDS:-7,11,17,23,31,37,43,53}" \
  --workers ${EXTERNAL_BASELINE_WORKERS:-8 16 24} \
  --artifact-out "${EXTERNAL_BASELINE_OUT:-results/external_baselines}" \
  --rounds "${EXTERNAL_BASELINE_ROUNDS:-8}" \
  --samples "${EXTERNAL_BASELINE_SAMPLES:-1600}" \
  --features "${EXTERNAL_BASELINE_FEATURES:-240}" \
  --shards "${EXTERNAL_BASELINE_SHARDS:-8}" \
  --network-rtt-ms "${EXTERNAL_BASELINE_RTT_MS:-3.0}" \
  --network-bandwidth-mbps "${EXTERNAL_BASELINE_BW_MBPS:-250.0}" \
  "$@"
