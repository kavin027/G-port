#!/usr/bin/env bash
set -u

ROOT=/root/coded_k8s_results_ext

for _ in $(seq 1 240); do
  namespaces=$(kubectl get ns -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}' | grep '^coded-majorrev-w' || true)
  for ns in $namespaces; do
    tail=${ns#coded-majorrev-w}
    workers=${tail%-*}
    seed=${tail##*-}
    run_dir="$ROOT/majorrev_k8s_w${workers}_seed${seed}"
    if [ -f "$run_dir/network_summary.csv" ] && [ -f "$run_dir/k8s_resource_counters.csv" ]; then
      kubectl delete namespace "$ns" --ignore-not-found=true --wait=false >/dev/null 2>&1 || true
      echo "$(date -Is) deleted $ns"
    fi
  done
  sleep 20
done
