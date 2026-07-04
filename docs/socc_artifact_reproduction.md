# SoCC Artifact Reproduction Map

This document maps the main paper claims to stable commands and output
directories.  It avoids machine-specific secrets; remote experiments should
receive credentials through environment variables.

## Anonymous Package Boundary

For review, package the source, scripts, compact diagnostics, K3s manifests,
pod-placement logs, and per-seed CSV summaries that regenerate the paper
tables.  Exclude cloud credentials, literal IP addresses, hostnames, SSH logs,
local user paths, mock-review notes, and bulky exploratory result directories.
The smoke and replay paths below run without private servers.  A fresh K3s rerun
requires a reviewer-provided three-node cluster with shared source and problem
directories on all scheduled nodes.

Anonymous review artifact:
`https://anonymous.4open.science/r/G-port-4FD5/`.

Public repository placeholder for the camera-ready version:
`<GITHUB_URL_TO_BE_FILLED_AFTER_ACCEPTANCE>`.  The dual-anonymous review paper
should cite only the anonymous artifact URL above.

## Repository Layout

The artifact follows the scheduler-paper convention of separating environment,
policy, analysis, and collected logs:

- `src/coded_learning_exp/`: worker-service runtime, sparse-code scheduling
  policies, TCP master/worker code, and online guard implementation.
- `run_*k8s*.py` and `k8s/`: K3s job orchestration, worker manifests, pod
  placement, and stress/reissue entrypoints.
- `run_*docker*.py` and `run_network_container_experiment.py`: local TCP and
  Docker smoke paths.
- `analyze_*.py`: table rebuilders for external baselines, K3s matrices,
  feature ablations, guard prediction, stress, and additive-workload checks.
- `results/`: compact per-seed CSV summaries and diagnostics used to regenerate
  the submitted tables.
- `paper/socc26/`: LaTeX source and generated paper tables.

## Reviewer Fast Path

Purpose: give reviewers a single short command that exercises the main artifact
plumbing before they decide which full experiments to reproduce.  This path
runs an independent TCP worker-service smoke test, a direct Docker-bridge
container-to-container smoke test, and the fixed guarded-policy replay.  It is
not intended to reproduce the paper-scale numbers.

```bash
python run_socc_artifact_fast_path.py --clean
```

Expected output:

- `socc_fast_path_artifact/summary_report.md`
- `socc_fast_path_artifact/tcp_smoke/network_summary.csv`
- `socc_fast_path_artifact/direct_docker_bridge_smoke/network_summary.csv`
- `socc_fast_path_artifact/guarded_policy_diagnostics/guarded_policy_report.md`

If Docker is not available, reviewers can still check the TCP and guard paths:

```bash
python run_socc_artifact_fast_path.py --clean --skip-docker
```

## Reviewer Table Rebuild

Purpose: regenerate the current paper-facing figures, CSVs, and LaTeX table
fragments from the collected logs included in the artifact.  These targets do
not contact private servers and do not require SSH credentials.  Override the
root variables if the artifact unpacks logs into different paths.

```bash
make reproduce-figure2
make reproduce-figure3
make reproduce-table1
make reproduce-table2
make reproduce-table3
make reproduce-table4
make reproduce-table5
make reproduce-k3s-main
make reproduce-external-baselines
make reproduce-best-safe
make reproduce-stress
make reproduce-tables
make reproduce-paper-assets
```

On systems without `make`, use the cross-platform dispatcher.  The dispatcher
rebuilds from raw logs when they are present; in compact public artifacts, it
uses the checked-in summary CSVs and generated figure/table fragments.

```bash
python scripts/reproduce_paper_artifacts.py figure2
python scripts/reproduce_paper_artifacts.py figure3
python scripts/reproduce_paper_artifacts.py table1
python scripts/reproduce_paper_artifacts.py table2
python scripts/reproduce_paper_artifacts.py table3
python scripts/reproduce_paper_artifacts.py table4
python scripts/reproduce_paper_artifacts.py table5
python scripts/reproduce_paper_artifacts.py all
```

The current paper mapping is:

- `reproduce-figure1`: rebuilds the PDF containing the in-text TikZ motivation
  figure.
- `reproduce-figure2`: exports the Draw.io design overview into SVG/PDF.
- `reproduce-figure3`: rebuilds the guard threshold-sensitivity CSV and paper
  PDF/PNG figure.
- `reproduce-table1`: rebuilds the external-baseline comparison across TCP,
  TCP+stress, and K3s, then writes
  `results/paper_reproduction/table1_main_external.{csv,tex}`.
- `reproduce-table2`: rebuilds the G-PORT ablation table and writes
  `results/paper_reproduction/table2_gport_ablation.{csv,tex}`.
- `reproduce-table3`: rebuilds the threshold-sensitivity data used by Figure 3
  and the paper's threshold-setting table.
- `reproduce-table4`: rebuilds the K3s closed-connection reissue stress table.
- `reproduce-table5`: rebuilds the row-score diagnostic table.

Legacy aliases remain for old scripts: `reproduce-table6` maps to the current
K3s stress table, and `reproduce-guard-prediction` maps to the current Figure 3
and Table 3 diagnostics.

Equivalent explicit variables:

```bash
make reproduce-tables \
  TCP_BASELINE_ROOT=results/external_baselines_tcp_plain_full_best_safe \
  TCP_STRESS_ROOT=results/external_baselines_network_stress_full_best_safe \
  K3S_MAIN_ROOT=results/server_k3s_20260702/coded_k3s_external_full \
  K3S_BEST_SAFE_ROOT=results/server_k3s_20260702/coded_k3s_best_safe_full \
  K3S_STRESS_ROOT=results/server_k3s_20260702/coded_k3s_stress_full \
  K3S_RECOVERY_ROOT=results/server_k3s_20260702/coded_k3s_recovery_stress \
  PAPER_REPRO_ROOT=results/paper_reproduction
```

An anonymity smoke check is also provided:

```bash
make audit-anonymity ARTIFACT_ROOT=<anonymous-package-root>
```

## Local Smoke

Purpose: verify the TCP worker-service code path quickly.  Expected runtime is
usually under a few minutes on a modern laptop or server.

```bash
python run_network_container_experiment.py --quick \
  --strategies speed_aware_uncoded sparse_flexible_static \
    rank_aware_sparse_flexible system_portfolio guarded_system_portfolio \
  --portfolio-fallback static \
  --common-jitter-across-strategies \
  --out local_two_node_codepath_smoke \
  --base-port 30100
```

Expected output:

- `local_two_node_codepath_smoke/network_metrics.csv`
- `local_two_node_codepath_smoke/network_summary.csv`

The paper reports both the static-fallback K3s matrix and a separate online
`best_safe` fallback K3s matrix.  Paired replay is used only as a diagnostic
over the static-fallback trace.

## External Baseline Matrix

Purpose: reproduce the same-runtime comparison against recent cloud and
ML-systems scheduling ideas.  These commands run adapters inspired by Original-SFCL, RLTune,
Sailor, and StragglerAnalysis in the same worker-service path; they do not
execute the external projects' original GPU cluster code.  The paper table uses
the full eight-seed three-node K3s run below.  A network-stressed TCP run is
kept as a secondary stress check, not as the main external-baseline table.

```bash
python run_majorrev_k8s_extended.py \
  --workers 8 16 24 \
  --seeds 7 11 17 23 31 37 43 53 \
  --rounds 8 \
  --samples 1600 \
  --features 240 \
  --shards 8 \
  --skip-problem-build \
  --out-root /root/coded_k3s_external_full \
  --diagnostics-out /root/coded_k3s_external_full/guard_prediction_diagnostics \
  --namespace-prefix coded-ext-full \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2> \
  --wait-timeout 1200s \
  --strategies speed_aware_uncoded speculative_replication sparse_flexible_static \
    original_sfcl_static worker_aware_sparse_flexible rank_aware_sparse_flexible \
    deadline_aware_sparse_flexible system_portfolio guarded_system_portfolio \
    online_counter_guard_rank_aware_sparse_flexible \
    online_counter_guard_deadline_aware_sparse_flexible \
    rltune_style_selector sailor_style_heterogeneity_aware \
  --continue-on-error

python analyze_external_baselines.py \
  --root /root/coded_k3s_external_full \
  --best-safe-root /root/coded_k3s_best_safe_full \
  --out /root/coded_k3s_external_full/external_analysis
```

Expected output:

- `/root/coded_k3s_external_full/external_analysis/k3s_external_summary.csv`
- `/root/coded_k3s_external_full/external_analysis/external_baseline_matrix.csv`
- `/root/coded_k3s_external_full/external_analysis/external_baseline_matrix_table.tex`
- `/root/coded_k3s_external_full/external_analysis/best_safe_k3s_summary.csv`
- `/root/coded_k3s_external_full/external_analysis/external_arm_distribution.csv`
- `/root/coded_k3s_external_full/external_analysis/per_seed_external_results.csv`
- `/root/coded_k3s_external_full/external_analysis/whatif_diagnostics.csv`

The collected paper artifact mirrors this run under
`results/server_k3s_20260702/coded_k3s_external_full/`.  On small review
machines, use `--seeds 7 11 --rounds 4` as a smoke check only; do not use that
abbreviated run for the paper table.

Feature ablation table:

```bash
python analyze_feature_ablation.py \
  --root /root/coded_k3s_external_full \
  --out /root/coded_k3s_external_full/feature_ablation \
  --label static-fallback-k3s
```

Expected output:

- `/root/coded_k3s_external_full/feature_ablation/feature_ablation_table.csv`
- `/root/coded_k3s_external_full/feature_ablation/feature_ablation_table.tex`

## Online Best-Safe K3s Matrix

Purpose: reproduce the fixed online `best_safe` fallback deployment reported in
the main K3s matrix.  This is not a replay: the master chooses between
speed-aware uncoded and static coded safe baselines before the round using the
same predictor.

```bash
python run_majorrev_k8s_extended.py \
  --workers 8 16 24 \
  --seeds 7 11 17 23 31 37 43 53 \
  --rounds 8 \
  --samples 1600 \
  --features 240 \
  --shards 8 \
  --skip-problem-build \
  --out-root /root/coded_k3s_best_safe_full \
  --diagnostics-out /root/coded_k3s_best_safe_full/guard_prediction_diagnostics \
  --namespace-prefix coded-best-safe \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2> \
  --wait-timeout 1200s \
  --portfolio-fallback best_safe \
  --strategies speed_aware_uncoded sparse_flexible_static \
    system_portfolio guarded_system_portfolio \
  --continue-on-error

python analyze_majorrev_k8s.py --root /root/coded_k3s_best_safe_full
```

Expected output:

- `/root/coded_k3s_best_safe_full/majorrev_k8s_group_summary.csv`
- `/root/coded_k3s_best_safe_full/majorrev_k8s_all_summary.csv`
- `/root/coded_k3s_best_safe_full/majorrev_k8s_report.md`

Known headline result:

- Guarded portfolio with online `best_safe`: W8/W16/W24 barrier gains
  9.6%/35.9%/22.7% relative to static sparse-flexible placement in the same
  run.

## K3s Closed-Connection Reissue Stress

Purpose: verify bounded worker-failure semantics in the TCP worker-service
path.  If a worker connection closes before first decode, the master marks it
unavailable and reissues unfinished rows to the fastest live worker.  This is a
prototype recovery hook, not production fault tolerance.

```bash
python run_majorrev_k8s_stress.py \
  --workers 24 \
  --seeds 7 31 \
  --rounds 6 \
  --samples 1600 \
  --features 240 \
  --shards 8 \
  --problem-host-root /root/coded_k8s_problem \
  --out-root /root/coded_k3s_recovery_stress \
  --namespace-prefix coded-recovery \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2> \
  --cases close_connection \
  --worker-failure-recovery reissue \
  --continue-on-error

python analyze_k8s_stress.py \
  --root /root/coded_k3s_recovery_stress \
  --out /root/coded_k3s_recovery_stress
```

Expected output:

- `/root/coded_k3s_recovery_stress/k8s_stress_summary.csv`
- `/root/coded_k3s_recovery_stress/k8s_stress_table.tex`

Known headline result:

- Closed-connection reissue restores decode success to 1.0 for Static,
  Guarded portfolio, and Guard-D on the W24 two-seed stress; Guarded portfolio
  keeps a 24.2% barrier gain.

## Network-Constrained TCP Stress

Purpose: reproduce Table `tab:wan-tcp-runtime`, where gains are measured
against speed-aware uncoded placement under 8 ms RTT and 50 Mbps bandwidth.
The full four-seed run took under one minute on a 96-core server and should be
treated as a longer optional artifact check on smaller machines.

Run one seed at a time on the server-side TCP runtime:

```bash
python run_network_container_experiment.py \
  --samples 12000 --features 1200 --density 0.008 \
  --shards 16 --workers 16 --rounds 12 \
  --scenario phase --drift-period 6 \
  --straggler-fraction 0.45 --straggler-slowdown 0.08 \
  --sleep-scale 0.03 --cost-scale 0.006 \
  --network-rtt-ms 8 --network-bandwidth-mbps 50 \
  --common-jitter-across-strategies \
  --seed 11 --base-port 21000 \
  --out network_wan_common_stream_sweep_newserver/w16_seed_11 \
  --strategies speed_aware_uncoded speculative_replication \
    sparse_flexible_static rank_aware_sparse_flexible \
    system_portfolio learned_system_portfolio
```

Repeat with seeds `23`, `31`, and `43` using non-overlapping base ports.

Analyze:

```bash
python analyze_network_container_results.py \
  network_wan_common_stream_sweep_newserver/w16_seed_11 \
  network_wan_common_stream_sweep_newserver/w16_seed_23 \
  network_wan_common_stream_sweep_newserver/w16_seed_31 \
  network_wan_common_stream_sweep_newserver/w16_seed_43 \
  --baseline-strategy speed_aware_uncoded \
  --out network_wan_common_stream_newserver_diagnostics
```

Known output from the latest run:

- `network_wan_common_stream_newserver_diagnostics/network_report.md`
- `network_wan_common_stream_newserver_diagnostics/aggregate_vs_speed_aware_uncoded.csv`
- `network_wan_common_stream_newserver_diagnostics/bootstrap_ci_vs_speed_aware_uncoded.csv`

Headline result:

- Rank-aware coded: 28.3% mean decode gain, 48.5% p95 decode gain.
- System portfolio: 31.6% mean decode gain, 48.8% p95 decode gain.

## Fresh-Server Direct TCP Service Check

Purpose: confirm that the submitted artifact package runs cleanly on a fresh
server with independent TCP worker services.  The server used for this check
did not expose arbitrary external ports and did not have Docker installed, so
this is not a direct multi-node container deployment.

Configuration:

- 16 workers, 16 shards, 12 rounds, four seeds.
- 8 ms RTT and 50 Mbps transfer model.
- Common worker/iteration jitter across strategies.
- Baseline: speed-aware uncoded placement.

Known output:

- `docs/direct_tcp_service_results_fresh_server.md`

Headline result:

- Rank-aware coded: 13.9% mean decode gain, 29.8% p95 decode gain.
- System portfolio: 18.3% mean decode gain, 31.4% p95 decode gain, and 14.6%
  mean barrier gain.

## Direct Docker-Bridge TCP Check

Purpose: validate the direct container-to-container worker-service path on a
local Docker host.  The master and workers run as containers on one Docker
network, the master reaches workers through Docker DNS, worker host-port
publishing is disabled, and SSH forwarding is disabled.  This is a deployment
boundary check rather than a production multi-node Kubernetes result.  The
scaling sweep below uses the same problem size and network model for
8/16/24-worker anti-aligned runs, and builds the Docker image once before
reusing it across runs.

```bash
python run_direct_docker_sweep.py \
  --out-root direct_docker_scale_sweep \
  --diagnostics-out direct_docker_scale_diagnostics \
  --workers 8 16 24 \
  --alignments anti \
  --seeds 17 23 31 \
  --samples 1500 --features 220 --density 0.02 \
  --rounds 4 --sleep-scale 0.010 --cost-scale 0.002 \
  --network-rtt-ms 4 --network-bandwidth-mbps 100
```

Known output:

- `direct_docker_scale_diagnostics/direct_docker_scale_report.md`
- `direct_docker_scale_diagnostics/direct_docker_scale_by_workers.csv`
- `direct_docker_scale_diagnostics/network_report.md`

Headline result:

- Zero worker errors across the 8/16/24-worker direct bridge sweep.
- The 8-worker case is a useful negative boundary: rank-aware/deadline-aware
  p95 decode gains are -4.8%/-3.4%.
- At 16 workers, rank-aware/deadline-aware p95 decode gains are 18.4%/15.1%.
- At 24 workers, rank-aware/deadline-aware p95 decode gains are 15.5%/30.6%;
  the 24-worker run is noisier, so the paper uses it as directional scaling
  evidence rather than a standalone production-cluster claim.

## Sparse Embedding Update Microbenchmark

Purpose: check the mechanism on an additive sparse-update workload beyond ridge
regression.  Each interaction touches one user embedding and one item embedding;
the shard gradients remain additive, so the existing sparse-flexible decoding
path can recover the full update.  This is not an end-to-end recommendation
system claim.

```bash
python run_embedding_microbenchmark.py \
  --out embedding_microbenchmark_diagnostics \
  --rounds 35 \
  --run-ids 17 23 31 \
  --strategies sparse_flexible_static \
    rank_aware_sparse_flexible deadline_aware_sparse_flexible \
  --shard-cost-skew 2.5 \
  --straggler-fraction 0.45 \
  --straggler-slowdown 0.08
```

Known output:

- `embedding_microbenchmark_diagnostics/embedding_microbenchmark_report.md`
- `embedding_microbenchmark_diagnostics/aggregate_embedding_summary.csv`
- `embedding_microbenchmark_diagnostics/combined_embedding_summary.csv`

Headline result:

- Rank-aware coded: 36.2% mean latency gain and 78.2% p95 latency gain.
- Deadline-aware coded: 41.9% mean latency gain and 79.9% p95 latency gain.
- The mean completed-row prefix shrinks by 2.2 rows for rank-aware and 2.9 rows
  for deadline-aware placement.

## Sparse Logistic Classification Workload

Purpose: address ridge-only concerns with an end-to-end sparse binary
classification loop on real LIBSVM data.  Successful coded decoding recovers
the exact full logistic gradient, so final loss, accuracy, and AUC should match
static placement; the comparison is wall-clock time to the same sequence of
logistic updates.

```bash
python run_logistic_workload_experiment.py \
  --out logistic_workload_diagnostics \
  --datasets a9a w8a \
  --max-samples 5000 \
  --shards 16 --workers 24 --rounds 20 \
  --run-ids 11 23 31 \
  --strategies sparse_flexible_static \
    rank_aware_sparse_flexible deadline_aware_sparse_flexible
```

Known output:

- `logistic_workload_diagnostics/logistic_workload_report.md`
- `logistic_workload_diagnostics/aggregate_logistic_summary.csv`
- `logistic_workload_diagnostics/combined_logistic_summary.csv`
- `logistic_workload_diagnostics/logistic_workload_summary.png`

Headline result:

- On `a9a`, rank-aware/deadline-aware reduce fixed-update training time by
  29.4%/28.1% and p95 iteration latency by 62.3%/62.8%.
- On `w8a`, rank-aware/deadline-aware reduce fixed-update training time by
  25.9%/24.0% and p95 iteration latency by 57.3%/58.2%.
- Final loss, accuracy, AUC, and decode success match static placement in the
  reported runs.

## Guarded-Policy Diagnostics

Purpose: reproduce the fixed counter-based guard analysis used for the boundary
case discussion.  The script aggregates logged mismatch and first-decode prefix
counters; it does not train a selector on final latency.

```bash
python analyze_guarded_policy.py
```

Known output:

- `guarded_policy_diagnostics/guarded_policy_report.md`
- `guarded_policy_diagnostics/guard_ablation_summary.csv`
- `guarded_policy_diagnostics/chronological_guard_replay.csv`
- `guarded_policy_diagnostics/mechanism_trace_prefix_latency.png`
- `guarded_policy_diagnostics/mechanism_trace_prefix_latency.pdf`

Headline result:

- Across alignment, scaling, Docker TCP, and network-constrained TCP regimes,
  the full guard raises mean p95 gain from 19.9% for always-on placement to
  24.2%, while removing all six negative-p95 regimes.
- In a stricter chronological replay over the local Docker runs, the first
  calibration segment sets the guard and the remaining iterations are used for
  evaluation; this raises mean p95 gain from 9.1% to 16.1% and removes the one
  negative run.
- The ablation table over regimes with mismatch and prefix diagnostics shows
  the full counter guard improves mean p95 gain to 22.3% over the 17
  coded-candidate regimes and removes all negative regimes, while mismatch-only
  and prefix-only guards still leave negative cases.

## Online Guard Sensitivity

Purpose: answer the systems-PC question of whether the guard depends on a
single offline replay window.  This script uses only an early warm-up segment
of each Docker TCP run to decide whether to enable a candidate scheduler, and
computes p95 gain only on later iterations.

```bash
python analyze_online_guard_sensitivity.py
```

Known output:

- `online_guard_sensitivity_diagnostics/online_guard_sensitivity_report.md`
- `online_guard_sensitivity_diagnostics/online_guard_sensitivity_summary.csv`
- `online_guard_sensitivity_diagnostics/online_guard_sensitivity_runs.csv`
- `online_guard_sensitivity_diagnostics/rank-aware-sparse-flexible_online_guard_sensitivity.png`

Headline result:

- For rank-aware scheduling, a strict 20% warm-up rule with zero allowed
  completed-prefix growth enables 3/6 Docker runs, raises post-warm-up mean p95
  gain from 9.1% to 16.1%, and removes the one negative run.
- The sweep over 20/40/60% warm-up windows and 0/1/2-row tolerances shows that
  looser prefix tolerances can re-admit harmful runs.  The paper should
  therefore present the guard as a conservative enablement rule, not as a
  broadly safe online controller.

## Optional Direct Remote TCP Path

Purpose: run the master on one host and workers on a remote host without SSH
forwarding.  This requires the remote worker ports to be directly routable from
the master.  Some rented cloud containers expose only SSH; in that case this
path will fail during the direct worker-port readiness check.

```bash
export REMOTE_PASSWORD='...'
python run_direct_remote_sweep.py \
  --samples 6000 --features 800 --density 0.008 \
  --shards 8 --workers 8 --rounds 8 \
  --scenario phase --drift-period 4 \
  --straggler-fraction 0.45 --straggler-slowdown 0.08 \
  --sleep-scale 0.03 --cost-scale 0.006 \
  --seeds 17 23 31 43 \
  --worker-host <ROUTABLE_WORKER_HOST> \
  --remote-ssh-host <REMOTE_SSH_HOST> \
  --remote-ssh-port <REMOTE_SSH_PORT> \
  --remote-user root \
  --remote-repo /root/coded_distributed_computing_socc_runtime \
  --remote-base-port 38000 \
  --output-root direct_remote_sweep \
  --diagnostics-out direct_remote_diagnostics
```

Known probe result on the current rented host:

- `docs/direct_multinode_port_probe.md`
- The host did not expose routable worker ports, so this raw remote-port path
  remains optional and environment-dependent.

## Direct Three-Node Kubernetes Evidence

Purpose: validate the same TCP worker-service entrypoint and online counter
guard on a small real multi-node container deployment without publishing worker
host ports.  The run used three cloud VMs on one private network: one k3s
control-plane node for the master Job and two worker nodes for a StatefulSet
behind a headless Service.  Worker pods were reached through Kubernetes DNS,
with placements balanced across the two worker nodes.

Known output archived from the major-revision matrix:

- `majorrev_k8s_diagnostics/majorrev_k8s_report.md`
- `majorrev_k8s_diagnostics/majorrev_k8s_group_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_all_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_paper_table.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_per_seed_core.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_mismatch_split.csv`
- `majorrev_k8s_diagnostics/*/network_summary.csv`
- `majorrev_k8s_diagnostics/*/k8s_pods_wide.txt`

Regenerate the aggregate report with:

```bash
python analyze_majorrev_k8s.py --root majorrev_k8s_diagnostics
```

Headline result:

- 100% decode success and no observed worker errors across 8/16/24 workers and
  seeds 7, 11, 17, 23, 31, 37, 43, and 53.
- 8 workers: always-on rank/deadline scheduling hurts barrier latency, while
  the online guard enables on 35.9% of rounds and gives a 10.0% barrier gain.
- 24 workers: the aggregate is intentionally reported with a static-only
  diagnostic split.  Low-tail seeds `7,11,37,43,53` have static barrier 16.4 ms
  and Guard-D changes little (16.0 ms, 2.4% gain); high-tail seeds `17,23,31`
  have static barrier 71.4 ms, portfolio/rank around 25.7/25.6 ms, and Guard-D
  at 38.4 ms (46.6% coded-only gain).

This is direct small-scale Kubernetes evidence for the worker-service path and
online guard, not a claim of production-scale cluster scheduling, failure
handling, or interference management.

### Prepared extended K3s seed and resource-counter run

The major-revision extension adds guarded system portfolio to the K3s matrix
and uses seeds `7`, `11`, `17`, `23`, `31`, `37`, `43`, and `53` for 8/16/24 workers.  It also
snapshots pod/node/resource counters after every run and then rebuilds the K3s
matrix, the Algorithm 3 guard diagnostics, and the online high-tail predictor.

The manifest uses `hostPath` for both source and problem inputs.  Before
starting the Kubernetes jobs, make sure `/root/coded_distributed_computing` and
the generated `/root/coded_k8s_problem_w*` directories are present on every
node that may host a worker pod.

```bash
python run_majorrev_k8s_extended.py \
  --prepare-problems-only \
  --source-host-path /root/coded_distributed_computing \
  --problem-host-root /root/coded_k8s_problem
```

After syncing the source and problem directories to worker nodes, run:

```bash
kubectl get nodes -o wide

python run_majorrev_k8s_extended.py \
  --skip-problem-build \
  --skip-existing \
  --continue-on-error \
  --source-host-path /root/coded_distributed_computing \
  --problem-host-root /root/coded_k8s_problem \
  --out-root /root/coded_k8s_results \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2>
```

Important outputs:

- `/root/coded_k8s_results/majorrev_k8s_w*_seed*/k8s_resource_counters.csv`
- `/root/coded_k8s_results/majorrev_k8s_group_summary.csv`
- `/root/coded_k8s_results/majorrev_k8s_per_seed_core.csv`
- `/root/coded_k8s_results/majorrev_k8s_mismatch_split.csv`
- `guard_prediction_diagnostics/k8s_live_resource_snapshots.csv`
- `guard_prediction_diagnostics/k8s_live_resource_by_workers.csv`
- `/root/coded_k8s_results/tail_predictor_diagnostics/online_tail_predictor_summary.csv`

If `metrics-server` is unavailable, `collect_k8s_resource_counters.py` leaves
`kubectl top` columns blank and preserves raw pod, event, node, and stats API
outputs under each run's `k8s_resource_raw/` directory.

### K3s guarded-portfolio and stress extension

The guarded system portfolio and worker-service stress checks use the same
K3s entrypoint and output layout.  The stress runner can inject best-effort
CPU interference and cancellation-ack delay:

```bash
python run_majorrev_k8s_stress.py \
  --workers 24 \
  --seeds 7 31 \
  --rounds 6 \
  --cases baseline cpu_hog cancel_ack_20ms close_connection \
  --source-host-path /root/coded_distributed_computing \
  --problem-host-root /root/coded_k8s_problem \
  --out-root /root/coded_k8s_stress_results \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2>
```

Expected outputs:

- `/root/coded_k8s_stress_results/k8s_stress_summary.csv`
- `/root/coded_k8s_stress_results/k8s_stress_table.tex`
- `/root/coded_k8s_stress_results/*/majorrev_k8s_w*_seed*/network_summary.csv`

When using `hostPath`, the stress runner appends the case name to
`--problem-host-root`.  Make sure the case-specific directories exist on every
worker-host node, for example
`/root/coded_k8s_problem_baseline_w24`,
`/root/coded_k8s_problem_cpu_hog_w24`,
`/root/coded_k8s_problem_cancel_ack_20ms_w24`, and
`/root/coded_k8s_problem_close_connection_w24`.

### Worker-service failure and cancellation stress

The Docker stress suite exercises the same worker entrypoint with injected
cancel ACK delay, closed worker connections, and worker process exits.  It
requires an active Docker daemon; a `docker info` failure should be read as an
environment setup failure rather than a systems result:

```bash
python run_worker_service_stress.py \
  --out-root worker_service_stress_diagnostics
```

Expected outputs:

- `worker_service_stress_diagnostics/worker_service_stress_summary.csv`
- `worker_service_stress_diagnostics/worker_service_stress_report.md`
- `worker_service_stress_diagnostics/*/stress_case.log`

The `exit_on_task` case is allowed to return nonzero on a healthy Docker host.
That indicates the prototype currently treats a hard worker-process failure as
a run-level failure instead of silently recovering.

## Score Ablation Diagnostic

Purpose: compare the implemented `rho*C` row-pair priority with simpler score
variants and a small-code oracle based on enumerated minimal decodable subsets.
This is an offline diagnostic over the predictor used by the guard, not a new
systems benchmark.

```bash
python analyze_score_ablation.py
```

Known output:

- `score_ablation_diagnostics/score_ablation_report.md`
- `score_ablation_diagnostics/score_ablation_summary.csv`
- `score_ablation_diagnostics/score_ablation_runs.csv`
- `score_ablation_diagnostics/score_ablation_oracle_correlations.csv`

Headline result:

- The minimal-subset oracle reduces predicted mean/p95 first-decode time by
  38.9%/55.1% and shortens the predicted prefix by 1.2 rows.
- Sampled-minset-128/512 approximations recover part of the oracle signal and
  correlate with the exact minimal-subset frequency at 0.51/0.62 in this tiny
  diagnostic.
- The implemented `rho*C` feature improves predicted p95 by 26.8% but does not
  improve predicted mean latency in this tiny-code diagnostic, so the paper
  treats it as a guarded runtime feature rather than a criticality proof.

## Restricted Two-Node TCP Validation

Purpose: validate the first-decodable-time mechanism on a real cross-host
socket path when the remote cloud host exposes SSH but not arbitrary worker
ports.

```bash
export REMOTE_PASSWORD='...'
python run_tunneled_remote_sweep.py \
  --samples 6000 --features 800 --density 0.008 \
  --shards 8 --workers 8 --rounds 8 \
  --scenario phase --drift-period 4 \
  --straggler-fraction 0.45 --straggler-slowdown 0.08 \
  --sleep-scale 0.03 --cost-scale 0.006 \
  --seeds 17 23 31 43 \
  --remote-host <REMOTE_HOST> \
  --remote-ssh-port <REMOTE_SSH_PORT> \
  --remote-user root \
  --remote-repo /root/coded_distributed_computing_socc_runtime \
  --remote-out-prefix tunneled_remote_seed \
  --local-base-port 29300 \
  --remote-base-port 30400 \
  --output-root tunneled_remote_sweep_port50076 \
  --diagnostics-out tunneled_remote_port50076_diagnostics
```

Known output from the latest run:

- `tunneled_remote_sweep_port50076/seed_*/network_metrics.csv`
- `tunneled_remote_sweep_port50076/seed_*/network_summary.csv`
- `tunneled_remote_port50076_diagnostics/network_report.md`
- `docs/two_node_tcp_results_port50076.md`

Headline result:

- Rank-aware coded: 24.4% mean decode gain, 22.5% p95 decode gain over
  speed-aware uncoded.
- Barrier improvement is not stable; this run is a cross-host sanity check, not
  the main barrier-time claim.

## Paper Build

```bash
cd paper/socc26
latexmk -pdf -interaction=nonstopmode main.tex
```

Known current output:

- `paper/socc26/main.pdf`
- 11 pages in the latest checked build, within the 12-page research-paper limit
- No unresolved references in the latest build
