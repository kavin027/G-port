# SoCC Evidence Matrix

This matrix is intended for rebuttal preparation and artifact orientation.  It
keeps the paper's claims tied to concrete evidence without expanding the main
10-page text.

| Claim | Evidence in paper | Artifact command or output | Boundary |
| --- | --- | --- | --- |
| Sparse-flexible code construction and runtime scheduling are separate. | Introduction, Related Work, Figure 1 | Paper source and prototype strategy names | The code is fixed; the contribution is placement and enablement. |
| Decode-speed mismatch creates opportunity. | Controlled alignment and real-data sensitivity | `realdata_alignment_diagnostics/*` and `guarded_policy_diagnostics/*` | Mismatch is necessary but not sufficient. |
| Prefix reduction realizes the gain. | Mechanism table and mechanism trace | `guarded_policy_diagnostics/mechanism_trace_prefix_latency.png` | A scheduler can lose when it increases the first-decode prefix. |
| Guarded enablement avoids harmful regimes. | Boundary Cases, guard ablation table | `python analyze_guarded_policy.py` | The guard is fixed-counter replay, not post-result policy selection. |
| Early-window guard replay is causal with respect to the evaluated iterations. | Guard replay paragraph and rebuttal notes | `python analyze_online_guard_sensitivity.py`; `online_guard_sensitivity_diagnostics/*` | This is still replay over separately logged policies, not a deployed online controller. |
| Tail gain must dominate overhead. | Evaluation roadmap, overhead inequality, worker scaling | `runtime_scaling_diagnostics/*` | The scheduler should be disabled when overhead or prefix growth dominates. |
| TCP worker-service path is real code, not simulator-only. | TCP runtime, local Docker, direct Docker-bridge sweep, two-node sanity check | `run_network_container_experiment.py --quick`; `run_direct_docker_sweep.py`; `direct_docker_scale_diagnostics/*` | Docker bridge is same-host container networking; the current artifact is not a production multi-node cluster deployment. |
| Artifact has a short sanity path. | Artifact appendix and reproduction map | `python run_socc_artifact_fast_path.py --clean`; `socc_fast_path_artifact/summary_report.md` | Fast path validates plumbing only; full paper-scale numbers require the longer experiment commands. |
| Speed-aware uncoded is a serious boundary baseline. | Network-constrained TCP stress and Boundary Cases | `analyze_network_container_results.py ...` | Coded placement is useful only when communication/cancellation and row-span effects matter. |
| Workload claim is mechanism-level but not ridge-only. | Experimental Setup and Limitations | Real-data LIBSVM ridge runs, sparse logistic workload, and time-to-loss figure | Sparse ridge/logistic experiments isolate exact additive gradients; larger recommendation and GNN systems are future work. |
| Sparse classification reaches the same model quality faster. | Real-data workload paragraph | `run_logistic_workload_experiment.py` and `logistic_workload_diagnostics/*` | This is a small end-to-end logistic loop, not a production ML platform. |
| Additive sparse updates beyond ridge regression keep the same scheduling mechanism. | Sparse embedding update microbenchmark | `run_embedding_microbenchmark.py` and `embedding_microbenchmark_diagnostics/*` | This is a microbenchmark, not an end-to-end recommender deployment. |

## One-Sentence Meta-Review Position

The paper is a systems mechanism paper: it shows that sparse flexible coded
learning leaves a runtime scheduling problem open, demonstrates when
first-decodable-time placement helps, and gives a fixed guard for when to fall
back.
