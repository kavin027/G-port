# Reviewer Quick Reproduction Guide

This is the short entry point for artifact reviewers.  The full command map is
in `docs/socc_artifact_reproduction.md`.

## What This Artifact Checks

The paper's main claim is a runtime scheduling claim: fixed sparse-flexible
codes expose multiple recoverable row sets, and code-aware placement can make a
useful set appear earlier under heterogeneous workers.  The artifact checks the
mechanism chain:

```text
mismatch -> prefix reduction -> guard -> tail gain
```

The artifact includes a direct three-node Kubernetes validation of the TCP
worker-service path and online counter guard over 8/16/24 workers, plus a
prepared guarded-portfolio and stress extension.  It does not claim
production-scale cluster scheduling, interference control, or deployment
automation; those remain the next systems step.

## Anonymous Package Boundary

The submitted package should include source, scripts, compact diagnostics, K3s
manifests, pod-placement logs, and per-seed CSV summaries.  It should exclude
private IPs, passwords, hostnames, raw SSH logs, local user paths, and internal
mock-review notes.  The smoke and replay checks below run without private
servers; a full fresh K3s rerun needs a reviewer-provided three-node cluster
with shared source and problem directories.

The authors prepared the review ZIPs with
`python tools/prepare_socc_submission_package.py --clean`, which keeps compact
diagnostics and sanitized K3s evidence while excluding raw cluster dumps and
local logs.

## 15-Minute Smoke Path

Run from the repository root:

```bash
python run_network_container_experiment.py --quick \
  --strategies speed_aware_uncoded sparse_flexible_static \
    rank_aware_sparse_flexible system_portfolio guarded_system_portfolio \
  --portfolio-fallback static \
  --common-jitter-across-strategies \
  --out local_two_node_codepath_smoke \
  --base-port 30100
```

Expected outputs:

- `local_two_node_codepath_smoke/network_metrics.csv`
- `local_two_node_codepath_smoke/network_summary.csv`

This validates the TCP worker-service code path and output schema quickly.  It
is a smoke test, not the full paper table.
Use `--portfolio-fallback best_safe` to exercise the newly implemented
performance-mode safe-baseline fallback; the paper's collected K3s table used
the reproducible `static` fallback trace and reports best-safe as a paired
diagnostic.

## Guard Replay Check

Run:

```bash
python analyze_guarded_policy.py
```

Expected outputs:

- `guarded_policy_diagnostics/guarded_policy_report.md`
- `guarded_policy_diagnostics/guard_ablation_summary.csv`
- `guarded_policy_diagnostics/chronological_guard_replay.csv`
- `guarded_policy_diagnostics/mechanism_trace_prefix_latency.png`

Expected headline numbers from the current artifact:

- Full counter guard: 22.3% mean p95 gain over the 17 coded-candidate regimes
  with mismatch and prefix diagnostics, with zero negative-p95 regimes.
- Overall guard report: 24.2% mean p95 gain over 19 aggregate regimes after
  adding the two TCP system-portfolio regimes, with negative regimes reduced
  from six to zero.
- Chronological Docker replay: mean p95 gain improves from 9.1% to 16.1%, and
  the one negative run is removed.

The guard is a fixed counter rule over mismatch and first-decode prefix
diagnostics.  It is not a selector trained on final latency.

## Online Guard Sensitivity Check

Run:

```bash
python analyze_online_guard_sensitivity.py
```

Expected output: `online_guard_sensitivity_diagnostics/`.

This replay uses only an early warm-up segment to decide whether to enable the
candidate scheduler, then evaluates only later iterations.  The conservative
rank-aware setting (20% warm-up, zero allowed completed-prefix growth) enables
3/6 Docker runs, improves mean post-warm-up p95 gain from 9.1% to 16.1%, and
removes the one negative run.  Looser prefix tolerances can re-admit harmful
runs, so the intended policy is conservative enablement rather than universal
online scheduling.

## K3s Guard Prediction Check

Run:

```bash
python analyze_guard_prediction.py --root majorrev_k8s_diagnostics
python analyze_online_tail_predictor.py \
  --per-round guard_prediction_diagnostics/guard_prediction_per_round.csv
```

Expected outputs:

- `guard_prediction_diagnostics/guard_prediction_accuracy.csv`
- `guard_prediction_diagnostics/guard_threshold_sensitivity.csv`
- `guard_prediction_diagnostics/k8s_resource_counters.csv`
- `guard_prediction_diagnostics/guard_prediction_tables.tex`
- `tail_predictor_diagnostics/online_tail_predictor_summary.csv`

This paired-log analysis recomputes Algorithm 3's predicted first-decodable
time from the archived K3s problems and scheduler state, then compares it with
the candidate policy observed under the same worker-state and jitter stream.
It also replays the fixed thresholds for `theta_cv`, `theta_g`, `theta_K`, and
`theta_a`.  The check documents guard accuracy and sensitivity; it does not tune
thresholds on final latency.

## Prepared Major-Revision K3s Extension

The extended SoCC-review run adds guarded system portfolio to the K3s matrix,
adds seeds `7`, `11`, `37`, `43`, and `53` for 8/16/24 workers, records a
Kubernetes resource-counter snapshot after each run, and regenerates the K3s,
guard-prediction, and high-tail-prediction summaries.

Because the K3s manifest intentionally uses `hostPath`, the source tree and
problem directories must exist at the same absolute paths on all scheduled
nodes.  On the control-plane node, first prepare the problem directories:

```bash
python run_majorrev_k8s_extended.py \
  --prepare-problems-only \
  --source-host-path /root/coded_distributed_computing \
  --problem-host-root /root/coded_k8s_problem
```

Then sync `/root/coded_distributed_computing` and
`/root/coded_k8s_problem_w*` to the worker nodes.  After `kubectl get nodes -o
wide` confirms the node names, run:

```bash
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

Expected new outputs include:

- `/root/coded_k8s_results/majorrev_k8s_w*_seed*/k8s_resource_counters.csv`
- `/root/coded_k8s_results/majorrev_k8s_group_summary.csv`
- `/root/coded_k8s_results/majorrev_k8s_per_seed_core.csv`
- `guard_prediction_diagnostics/k8s_live_resource_by_workers.csv`
- `/root/coded_k8s_results/tail_predictor_diagnostics/online_tail_predictor_summary.csv`

If metrics-server is absent, the resource collector still preserves pod JSON,
events, node JSON, and API stats if available; unavailable `kubectl top`
counters are left blank with a note.

## Worker-Service Failure and Cancellation Stress

On a Docker host, run:

```bash
python run_worker_service_stress.py --out-root worker_service_stress_diagnostics
```

This suite requires an active Docker daemon.  It reuses the direct Docker
worker-service path and injects cancellation ACK delay, closed TCP connections,
and worker exits through worker environment variables.  If `docker info` fails,
the generated log is an environment failure, not an experimental result.  The
`exit_on_task` case may fail even on a healthy Docker host; that is recorded as
an explicit prototype limitation rather than hidden recovery behavior.

## K3s Interference and Cancellation Stress

When a three-node K3s cluster is available, run:

```bash
python run_majorrev_k8s_stress.py \
  --workers 24 \
  --seeds 17 23 31 53 \
  --cases baseline cpu_hog cancel_ack_20ms \
  --source-host-path /root/coded_distributed_computing \
  --out-root /root/coded_k8s_stress_results \
  --master-node <CONTROL_NODE_NAME> \
  --worker-nodes <WORKER_NODE_1> <WORKER_NODE_2>
```

Expected outputs:

- `/root/coded_k8s_stress_results/k8s_stress_summary.csv`
- `/root/coded_k8s_stress_results/k8s_stress_table.tex`
- `/root/coded_k8s_stress_results/*/majorrev_k8s_w*_seed*/network_summary.csv`

## Longer Optional TCP Stress

The network-constrained TCP stress path in
`docs/socc_artifact_reproduction.md` regenerates the table inputs for the
speed-aware uncoded comparison.  It is the preferred extended check when a
reviewer has a larger server available.

## Optional Kubernetes Evidence

The direct multi-node Kubernetes run used three cloud VMs on a private network:
one k3s control-plane/master Job node and two worker StatefulSet nodes behind a
headless Service.  The major-revision matrix is under
`majorrev_k8s_diagnostics/` and is regenerated with:

```bash
python analyze_majorrev_k8s.py --root majorrev_k8s_diagnostics
```

Expected outputs:

- `majorrev_k8s_diagnostics/majorrev_k8s_group_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_all_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_paper_table.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_per_seed_core.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_mismatch_split.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_report.md`

The headline validation has 100% decode success and no observed worker errors across
the 8/16/24-worker matrix over eight seeds.  At 8 workers, the online guard
changes an always-on negative barrier regime into a 10.0% gain; at 24 workers,
the aggregate is heterogeneous.  The static-only split in
`majorrev_k8s_mismatch_split.csv` shows low-tail seeds
`7,11,37,43,53`, where Guard-D mostly falls back and gives a 2.4% coded gain,
and high-tail seeds `17,23,31`, where portfolio/rank are strongest and Guard-D
preserves a 46.6% coded-only gain.  The split explains K3s heterogeneity and is
not an input to the online guard.  This supports the paper's claim that the guard
is a conservative coded safety policy rather than a global performance optimizer.

## Score Ablation Diagnostic

Run:

```bash
python analyze_score_ablation.py
```

Expected outputs:

- `score_ablation_diagnostics/score_ablation_report.md`
- `score_ablation_diagnostics/score_ablation_summary.csv`
- `score_ablation_diagnostics/score_ablation_oracle_correlations.csv`

This is a small 8-worker offline diagnostic.  It shows that a minimal-subset
oracle is stronger than the implemented `rho*C` feature, which is why the paper
presents the score as a runtime feature protected by the online guard rather
than as a theoretical recovery-criticality certificate.  The current diagnostic
also includes sampled-minset-128/512 approximations to show how much a more
expensive row-criticality feature could recover from the oracle.
