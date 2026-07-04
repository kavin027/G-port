# Artifact and Reproduction Appendix

This appendix maps the paper claims to the submitted artifact commands.  The
full command list and expected outputs are maintained in
`docs/socc_artifact_reproduction.md`.

Anonymous artifact URL:
`https://anonymous.4open.science/r/G-port-4FD5/`.

## Anonymous Review Package Boundary

The anonymous artifact package should contain the source, run scripts, compact
diagnostic CSV/Markdown outputs, K3s manifests, pod-placement logs, and
per-seed summaries needed to regenerate the paper tables.  It should not contain
cloud credentials, literal IP addresses, hostnames, SSH logs, local user paths,
mock-review notes, or bulky exploratory directories that are not cited by the
paper.  The short smoke and replay paths run without private servers; a fresh
end-to-end K3s rerun requires a reviewer-provided three-node cluster with shared
source/problem paths.

## Minimum Smoke Path

Run `python run_socc_artifact_fast_path.py --clean` from the repository root to
exercise the independent TCP worker-service smoke test, the direct
Docker-bridge container-to-container smoke test, and guarded-policy replay.
The command writes `socc_fast_path_artifact/summary_report.md` plus per-stage
logs and CSV outputs.  If Docker is unavailable, use `--skip-docker` to keep
the TCP and guard checks.

## Main TCP Stress Path

Run `run_network_container_experiment.py` with the network-constrained TCP
configuration and analyze it with `analyze_network_container_results.py`.  This
regenerates the table inputs for the speed-aware uncoded comparison and writes
the aggregate report, paired intervals, and summary CSVs.

## Guard Replay Path

Run `python analyze_guarded_policy.py`.  This regenerates:

- `guarded_policy_diagnostics/guarded_policy_report.md`
- `guarded_policy_diagnostics/guard_ablation_summary.csv`
- `guarded_policy_diagnostics/chronological_guard_replay.csv`
- `guarded_policy_diagnostics/mechanism_trace_prefix_latency.png`

The guard replay uses fixed counter rules over mismatch and first-decode prefix
diagnostics.  It is a diagnostic replay over logged counters, not a selector
trained on final latency.

Run `python analyze_online_guard_sensitivity.py` for the early-window guard
sensitivity check.  This writes `online_guard_sensitivity_diagnostics/*` and
evaluates later iterations only after using the warm-up segment to set the
enable/fallback decision.  The conservative rank-aware setting raises
post-warm-up mean p95 gain from 9.1% to 16.1% and removes the one negative
Docker run, while looser prefix tolerances can re-admit harmful runs.

## Direct Kubernetes Evidence

The three-node Kubernetes validation is archived in
`majorrev_k8s_diagnostics/`.  It includes generated manifests, pod placement
files, per-run metrics, and an aggregate report.  The run uses a k3s
control-plane/master Job node and two worker nodes with a StatefulSet behind a
headless Service; worker pods are reached by Kubernetes DNS with no published
worker ports.  Run `python analyze_majorrev_k8s.py --root
majorrev_k8s_diagnostics` to regenerate:

- `majorrev_k8s_diagnostics/majorrev_k8s_group_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_all_summary.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_paper_table.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_per_seed_core.csv`
- `majorrev_k8s_diagnostics/majorrev_k8s_report.md`

Run `python analyze_performance_fallback.py --root majorrev_k8s_diagnostics`
to regenerate the performance-mode fallback replay:

- `majorrev_k8s_diagnostics/performance_fallback_summary.csv`
- `majorrev_k8s_diagnostics/performance_fallback_seed_summary.csv`
- `majorrev_k8s_diagnostics/performance_fallback_details.csv`
- `majorrev_k8s_diagnostics/performance_fallback_table.tex`
- `majorrev_k8s_diagnostics/performance_fallback_per_seed_table.tex`
- `majorrev_k8s_diagnostics/performance_fallback_report.md`

This replay changes only rounds whose guarded-portfolio config records
`fallback-static`; enabled guarded-portfolio rounds keep the deployed latency.
The best-safe replay uses the cheaper observed safe baseline in the paired
round, so it is a paired diagnostic for the fallback choice, not a
result-selected controller.  The submitted K3s results also include an
independent online deployment of the fixed rule with
`--portfolio-fallback best_safe`, which chooses the safe baseline before a
round from predictor features.

Run `python analyze_guard_prediction.py --root majorrev_k8s_diagnostics` to
regenerate the Algorithm 3 prediction and threshold diagnostics:

- `guard_prediction_diagnostics/guard_prediction_accuracy.csv`
- `guard_prediction_diagnostics/guard_prediction_per_round.csv`
- `guard_prediction_diagnostics/guard_threshold_sensitivity.csv`
- `guard_prediction_diagnostics/guard_prediction_tables.tex`
- `guard_prediction_diagnostics/guard_prediction_report.md`

The matrix covers 8/16/24 workers over seeds 7, 11, 17, 23, 31, 37, 43, and
53 with 100% decode success and no observed worker errors.  The companion
stress directories contain the W24 baseline, CPU-hog, delayed-cancellation,
closed-connection, and bounded reissue checks; run
`python analyze_k8s_stress.py --root <stress_dir>` to regenerate each stress
table.  These files are evidence for the worker-service path and online guard
at small multi-node scale rather than a push-button production-cluster
artifact.

## Score Ablation Diagnostic

Run `python analyze_score_ablation.py` to regenerate
`score_ablation_diagnostics/score_ablation_report.md` and the corresponding
CSV files.  This small 8-worker diagnostic compares `rho`, `rho*C`, `rho/C`,
cost-only, random, and an enumerated minimal-subset oracle.  It supports the
paper's scoped claim that `rho*C` is a guarded scheduling feature rather than a
proof of minimal-subset criticality.  It also writes
`score_ablation_diagnostics/score_ablation_score_build_times.csv` and
`score_ablation_diagnostics/score_ablation_score_build_summary.csv`, which
record the score-construction overhead used to keep sampled-minset features as
offline diagnostics.
