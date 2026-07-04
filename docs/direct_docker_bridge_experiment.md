# Direct Docker-Bridge TCP Validation

This note records the local direct-container TCP experiment used as a
deployment-path check for the TCP-isolated worker runtime.

## Topology

- Master: a Docker container running `src.coded_learning_exp.direct_docker_master`.
- Workers: one Docker container per logical worker, each running the same TCP
  worker entrypoint used by the host-published container experiment.
- Network path: `master container -> Docker bridge DNS -> worker containers`.
- Host port publishing: disabled.
- SSH forwarding: disabled.
- Problem data: bind-mounted into containers from the shared output directory.

The local machine did not have an available Kubernetes cluster (`kubectl` was
installed, but no cluster/kind runtime was available).  This experiment therefore
uses Docker bridge networking as the smallest runnable approximation of direct
container-to-container service traffic.  It should be interpreted as a runtime
boundary validation, not as a production multi-node cluster result.

## Reproduction

Build the worker image once and run the direct bridge scaling sweep:

```powershell
python run_direct_docker_sweep.py `
  --out-root direct_docker_scale_sweep `
  --diagnostics-out direct_docker_scale_diagnostics `
  --workers 8 16 24 `
  --alignments anti `
  --seeds 17 23 31 `
  --samples 1500 --features 220 --density 0.02 `
  --rounds 4 --sleep-scale 0.010 --cost-scale 0.002 `
  --network-rtt-ms 4 --network-bandwidth-mbps 100
```

The host orchestrator removes containers and the Docker network after each run
unless `--keep-containers` is set.  Each run writes a `direct_docker_manifest.json`
that records `host_port_publishing=false` and `ssh_forwarding=false`.

## Result Summary

The anti-alignment sweep produced zero worker errors.  Gains below are paired
against static sparse-flexible placement within the same run:

| Workers | Rank-aware p95 gain | Deadline-aware p95 gain | Interpretation |
| ---: | ---: | ---: | --- |
| 8 | -4.8% | -3.4% | Boundary case where static placement is already competitive. |
| 16 | 18.4% | 15.1% | Container isolation makes first-decode scheduling beneficial. |
| 24 | 15.5% | 30.6% | Positive but noisier directional scaling evidence. |

We therefore use the direct bridge result as evidence that the TCP
worker-service boundary and cancellation path survive container DNS and direct
container networking, while keeping multi-node Kubernetes deployment as future
systems work.

Primary artifacts:

- `run_direct_docker_sweep.py`
- `analyze_direct_docker_scale.py`
- `src/coded_learning_exp/direct_docker_master.py`
- `direct_docker_scale_diagnostics/direct_docker_scale_report.md`
- `direct_docker_scale_diagnostics/network_report.md`
