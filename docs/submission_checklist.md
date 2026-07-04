# SoCC Submission Checklist

Date: 2026-07-01

## Current Paper Build

- Paper PDF: `paper/socc26/main.pdf`
- Page count: 13 total pages, with body text ending on page 12 and references
  beginning on page 13.
- Build command:

```bash
cd paper/socc26
latexmk -pdf -interaction=nonstopmode main.tex
```

Latest build status:

- No unresolved references observed.
- No overfull hbox observed.
- Remaining warnings are underfull vbox/page-fill warnings from figure
  placement.
- PDF metadata has no author field; the source uses
  `\documentclass[sigconf,review,anonymous]{acmart}`.

## Required Paper Source Files

Include:

- `paper/socc26/main.tex`
- `paper/socc26/references.bib`
- `paper/socc26/sections/*.tex`
- Required figures only:
  - `paper/socc26/figures/hypothesis_improvements.png`
  - `paper/socc26/figures/tcp_time_to_loss.png`
  - `paper/socc26/figures/runtime_realdata_improvements_a9a_w8a_rcv1.png`
  - `paper/socc26/figures/controlled_alignment_p95.png`
  - `paper/socc26/figures/mechanism_trace_prefix_latency.png`

Exclude from a clean source package:

- `paper/socc26/main.aux`
- `paper/socc26/main.bbl`
- `paper/socc26/main.blg`
- `paper/socc26/main.fdb_latexmk`
- `paper/socc26/main.fls`
- `paper/socc26/main.log`
- `paper/socc26/main.out`
- Unused exploratory figures unless the artifact package needs them.

## Required Artifact Files

Include:

- `README.md`
- `README_REVIEWER.md`
- `requirements.txt`
- `.gitignore`
- `src/coded_learning_exp/*.py`
- `run_socc_artifact_fast_path.py`
- `run_network_container_experiment.py`
- `analyze_network_container_results.py`
- `analyze_guarded_policy.py`
- `analyze_online_guard_sensitivity.py`
- `analyze_guard_prediction.py`
- `analyze_performance_fallback.py`
- `analyze_online_tail_predictor.py`
- `analyze_score_ablation.py`
- `analyze_majorrev_k8s.py`
- `analyze_k8s_stress.py`
- `collect_k8s_resource_counters.py`
- `run_majorrev_k8s_extended.py`
- `run_majorrev_k8s_stress.py`
- `run_worker_service_stress.py`
- `tools/prepare_socc_submission_package.py`
- `run_tunneled_remote_network_experiment.py`
- `run_tunneled_remote_sweep.py`
- `run_direct_remote_network_experiment.py`
- `run_direct_remote_sweep.py`
- `run_multiprocess_experiment.py`
- `run_multiprocess_sweep.py`
- `run_multiprocess_worker_scaling.py`
- `run_realdata_multiprocess_experiment.py`
- `run_realdata_multiprocess_sweep.py`
- `run_realdata_alignment_sweep.py`
- `run_realdata_sensitivity_sweep.py`
- `analyze_runtime_scaling.py`
- `analyze_realdata_results.py`
- `analyze_realdata_alignment.py`
- `analyze_realdata_sensitivity.py`
- `docs/socc_artifact_reproduction.md`
- `paper/socc26/artifact_appendix.md`
- `docs/reproducibility_tcp_runtime.md`
- `docs/two_node_tcp_experiment.md`
- `docs/direct_tcp_service_results_fresh_server.md`
- `docs/direct_multinode_port_probe.md`

Optional data:

- `data/libsvm/a9a`
- `data/libsvm/w8a`

The `rcv1` dataset is downloaded by the real-data script when needed and is not
currently stored in `data/libsvm/`.

## Result Directories To Keep For Internal Traceability

These directories support the current paper numbers and should be archived
internally, but they do not all need to be included in a small artifact ZIP:

- `network_wan_common_stream_sweep_newserver/`
- `network_wan_common_stream_newserver_diagnostics/`
- `majorrev_k8s_diagnostics/`
- `majorrev_k8s_stress_diagnostics/`
- `guard_prediction_diagnostics/`
- `tail_predictor_diagnostics/`
- `score_ablation_diagnostics/`
- `online_guard_sensitivity_diagnostics/`
- `worker_service_stress_diagnostics/`
- `direct_tcp_fresh_server_diagnostics/`
- `remote_direct_tcp_port*_results/`
- `tunneled_remote_sweep_port50076/`
- `tunneled_remote_port50076_diagnostics/`
- `runtime_sweep_highhetero_server/`
- `runtime_worker_scaling_server/`
- `runtime_worker_scaling_proportional_server/`
- `runtime_realdata_a9a_sweep_bjb1/`
- `runtime_realdata_w8a_sweep_bjb1/`
- `runtime_realdata_rcv1_sweep_bjb1/`
- `runtime_realdata_alignment_bjb1/`
- `runtime_realdata_sensitivity_bjb1/`

For a small public artifact, prefer commands plus compact diagnostics over all
raw exploratory directories.

## Files To Exclude From Artifact ZIP

Exclude:

- `__pycache__/`
- `*.pyc`
- `*.log`
- LaTeX build intermediates listed above
- Local smoke outputs unless explicitly needed
- Exploratory result directories not cited by the paper
- `docs/socc_mock_reviews*.md`
- `docs/final_claim_audit.md`
- Any server-specific stdout/stderr logs
- Files containing literal cloud IPs, hostnames, passwords, raw SSH
  transcripts, or local user paths

The repository now has a `.gitignore` covering Python caches, logs, LaTeX
intermediates, virtual environments, and local secret files.

## Anonymity And Secret Check

Checks performed:

- Literal supplied server passwords were not found in source, paper, README, or
  docs.
- Remote hostnames/IPs in reproduction docs were replaced with placeholders.
- Temporary logs containing local machine paths were removed.
- Python `__pycache__` directories containing local paths were removed.
- Paper source uses ACM anonymous review mode.
- PDF metadata does not expose an author.

Remaining acceptable strings:

- `REMOTE_PASSWORD` and `--remote-password` appear as option names and
  environment-variable placeholders.
- Internal result directory names may contain server nicknames or port numbers;
  keep them out of a public artifact ZIP unless renamed/sanitized.

## Final Pre-Submission Steps

1. Rebuild `paper/socc26/main.pdf` from a clean checkout or clean copy.
2. Open the PDF and visually inspect all pages for figure/table placement;
   verify that body text ends on page 12 and page 13 contains references only.
3. Run `python tools/prepare_socc_submission_package.py --clean` to create a
   sanitized dry-run source/artifact package under
   `submission/socc26_dryrun_20260701/`.
4. Read `submission/socc26_dryrun_20260701/PACKAGE_MANIFEST.md` and confirm
   that the sensitive scan has no private-IP, cloud-node-name, or local-user-path
   hits.
5. Do not include internal mock-review docs or local logs in the submitted
   artifact.
6. If SoCC requires source submission, include only the required paper source
   files and required figures.
