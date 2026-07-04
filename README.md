# Adaptive Sparse-Flexible Coded Learning Experiment

This repository contains a lightweight prototype for testing whether online
adaptive sparse-flexible coded computing is promising for distributed learning.

For artifact review, start with `README_REVIEWER.md`.  The detailed SoCC
reproduction map is in `docs/socc_artifact_reproduction.md`, and the
claim-to-evidence matrix is in `docs/socc_evidence_matrix.md`.

The experiment uses sparse ridge regression because its shard gradients are
linearly aggregatable. Each strategy receives the same simulated worker speeds
and straggler events at every training round, then tries to recover the full
gradient from completed uncoded, replicated, one-layer coded, or two-layer coded
tasks.

## Quick Start

```powershell
python run_experiment.py --quick
```

Outputs are written to `results/`:

- `metrics.csv`: per-round metrics for every strategy
- `summary.csv`: aggregate metrics by strategy
- `latency_over_time.png`: iteration latency curve
- `loss_over_time.png`: wall-clock training loss curve
- `summary_latency.png`: mean and P95 latency comparison

## Main Strategies

- `uncoded_sync`: waits for one copy of every shard.
- `replication`: uses spare workers as simple replicas.
- `static_sparse_code`: fixed one-layer sparse coded computing.
- `sparse_flexible_static`: fixed two-layer sparse-flexible coded computing.
- `worker_aware_sparse_flexible`: same fixed two-layer code, but assigns
  higher-cost encoded rows to faster workers.
- `rank_aware_sparse_flexible`: assigns rows by decode leverage and cost, which
  is safer when high-cost rows are not the rows needed for early decoding.
- `adaptive_sparse_flexible`: chooses a two-layer sparse coding configuration
  online with an epsilon-greedy bandit.
- `ucb_sparse_flexible`: chooses configurations with UCB exploration.
- `window_sparse_flexible`: adapts from a recent reward window.
- `contextual_sparse_flexible` and `contextual_ucb_sparse_flexible`: maintain
  separate bandit estimates for coarse worker-state contexts.
- `worker_aware_adaptive_sparse_flexible` and `worker_aware_ucb_sparse_flexible`:
  combine online configuration selection with worker-aware assignment.

## Example Experiments

```powershell
python run_experiment.py --scenario drift --rounds 160 --seed 7
python run_experiment.py --scenario burst --rounds 160 --seed 8
python run_experiment.py --scenario stable --rounds 160 --seed 9
python run_experiment.py --scenario phase --straggler-fraction 0.35 --straggler-slowdown 0.12
```

Run a multi-factor feasibility sweep:

```powershell
python run_sweep.py --scenarios stable burst drift --densities 0.005 0.01 0.03 --seeds 3 7
```

Run the reorganized paper-facing experiment suites:

```powershell
python run_reorganized_experiments.py --mode smoke
```

The reorganized runner separates the evidence into four hypothesis-driven
suites:

- `assignment`: whether cost-aware or decode-aware task assignment is the main
  effect.
- `sparsity`: whether the assignment benefit survives different input sparsity.
- `adaptation`: whether online code-density selection beats fixed flexible
  codes or closes the oracle gap.
- `ablation`: whether compute-aware rewards, window adaptation, and worker-aware
  adaptive variants each add value.

Its main output is `hypothesis_report.csv`, plus a short
`research_report.md`.

The sweep writes:

- `combined_summary.csv`: all run-level strategy summaries
- `aggregate_by_strategy.csv`: averaged strategy metrics by scenario and density
- `idea_report.csv`: direct checks for adaptive scheduling, contextual scheduling,
  sliding-window adaptation, and compute-aware reward design
- `sweep_strategy_latency.png` and `idea_improvements.png`: compact comparison plots

Useful knobs:

```powershell
python run_experiment.py `
  --samples 8000 `
  --features 1000 `
  --density 0.01 `
  --shards 16 `
  --workers 24 `
  --rounds 200 `
  --scenario drift
```

The code is intended as a feasibility scaffold. If the adaptive strategy does
not improve latency or wall-clock loss in this simulator, it is a warning sign
that the research idea needs a stronger scheduler, lower decoding overhead, or a
different workload model before moving to a full Ray/PyTorch cluster prototype.

## Multi-Process Runtime Prototype

The repository also includes a real multi-process runtime experiment.  Unlike
the simulator, it starts one Python worker process per logical worker, computes
encoded sparse gradients inside those worker processes, streams row results back
to the master, and stops an iteration when the completed rows become decodable.

Smoke test:

```powershell
python run_multiprocess_experiment.py --quick
```

Server-style run:

```powershell
python run_multiprocess_experiment.py `
  --samples 20000 `
  --features 2500 `
  --density 0.004 `
  --shards 16 `
  --workers 24 `
  --rounds 30 `
  --scenario phase `
  --drift-period 8 `
  --straggler-fraction 0.45 `
  --straggler-slowdown 0.08 `
  --sleep-scale 0.06 `
  --cost-scale 0.004
```

Outputs:

- `runtime_metrics.csv`: per-round runtime measurements
- `runtime_summary.csv`: aggregate latency, overhead, and decode metrics

On the 4-seed server sweep in `runtime_sweep_highhetero_server/`, decode-aware
assignment improves mean decode latency by about 9.7% and p95 latency by about
35.8% over static sparse-flexible assignment.  Deadline-aware assignment
improves mean decode latency by about 10.4% and p95 latency by about 37.0%.

Reproduce the multi-seed sweep:

```powershell
python run_multiprocess_sweep.py --start-method fork
```

Run worker-count scaling:

```powershell
python run_multiprocess_worker_scaling.py --start-method fork
```

Two scaling modes are useful:

```powershell
# Fixed data partition count, increasing redundant worker pool.
python run_multiprocess_worker_scaling.py --workers 8 16 24 32 --shards 16 --start-method fork

# Proportional scaling, n_shards = n_workers.
python run_multiprocess_worker_scaling.py --workers 8 16 24 32 --scale-shards-with-workers --start-method fork
```

In the server runs saved under `runtime_worker_scaling_server/` and
`runtime_worker_scaling_proportional_server/`, decode-aware/deadline-aware
scheduling consistently improves p95 first-decode latency for 16--32 worker
settings, with the strongest proportional-scaling result at 24 workers
(56.4--59.1% p95 improvement).  The benefit is not strictly monotonic, which is
important for the paper: assignment interacts with the sparse code instance,
the shard-to-worker ratio, and cancellation overhead.

Analyze scaling diagnostics and overhead:

```powershell
python analyze_runtime_scaling.py
```

This writes `runtime_scaling_diagnostics/`, including selected-row fractions,
extra compute deltas, scheduler overhead, gain-vs-overhead plots, and a short
diagnostic report explaining the 8-worker negative case.

## Real Sparse Data

Run on a LIBSVM/SVMLight dataset such as `a9a` on a Linux server:

```bash
python run_realdata_multiprocess_experiment.py --dataset a9a --start-method fork
```

Run a multi-seed real-data sweep:

```bash
python run_realdata_multiprocess_sweep.py --dataset a9a --start-method fork
```

On Windows, use `--start-method spawn` for local smoke tests.

Built-in dataset names are `a9a`, `w8a`, and `rcv1`.  You can also pass
`--url` and `--n-features` for another LIBSVM-format file.

## SoCC Artifact Map

The paper-facing reproduction guide is
[`docs/socc_artifact_reproduction.md`](docs/socc_artifact_reproduction.md).
It maps the headline claims to stable commands and output directories,
including the local TCP smoke test, the network-constrained TCP stress, the
restricted two-node TCP validation, and the LaTeX build.

## Paper Figure and Table Reproduction

The current SoCC draft uses stable Make targets for every paper-facing figure
and table.  These targets rebuild CSV/LaTeX artifacts from checked-in logs; the
K3s targets do not contact private servers unless you explicitly rerun the
raw experiments.

| Paper item | Make command | Python fallback | Main outputs |
| --- | --- | --- | --- |
| Figure 1, motivation | `make compile-paper` | `python scripts/reproduce_paper_artifacts.py compile` | TikZ source in `paper/socc26/sections/background.tex` |
| Figure 2, design overview | `make reproduce-figure2` | `python scripts/reproduce_paper_artifacts.py figure2` | `paper/socc26/figures/figure1_architecture.{svg,pdf}` |
| Figure 3, guard sensitivity | `make reproduce-figure3` | `python scripts/reproduce_paper_artifacts.py figure3` | `paper/socc26/figures/guard_threshold_sensitivity.{pdf,png}` |
| Table 1, external comparison | `make reproduce-table1` | `python scripts/reproduce_paper_artifacts.py table1` | `results/paper_reproduction/table1_main_external.{csv,tex}` |
| Table 2, G-PORT ablations | `make reproduce-table2` | `python scripts/reproduce_paper_artifacts.py table2` | `results/paper_reproduction/table2_gport_ablation.{csv,tex}` |
| Table 3, threshold settings | `make reproduce-table3` | `python scripts/reproduce_paper_artifacts.py table3` | `results/server_k3s_20260702/coded_k3s_external_full/guard_prediction_diagnostics_rebuild/guard_threshold_sensitivity.csv` |
| Table 4, K3s stress | `make reproduce-table4` | `python scripts/reproduce_paper_artifacts.py table4` | `results/server_k3s_20260702/coded_k3s_recovery_stress/analysis_rebuild/k8s_stress_table.tex` |
| Table 5, score diagnostic | `make reproduce-table5` | `python scripts/reproduce_paper_artifacts.py table5` | `score_ablation_diagnostics/score_ablation_summary.csv` |

To rebuild the paper-facing artifacts together:

```powershell
make reproduce-paper-assets
```

On systems without `make`:

```powershell
python scripts/reproduce_paper_artifacts.py all
```

To rebuild the PDF:

```powershell
make compile-paper
```

For submission packaging and anonymity checks, see
[`docs/submission_checklist.md`](docs/submission_checklist.md).
