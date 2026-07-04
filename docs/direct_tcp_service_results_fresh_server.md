# Fresh-Server Direct TCP Service Results

Date: 2026-05-26

Purpose: check that the network-constrained TCP result is reproducible on a
fresh server using the submitted artifact package, independent TCP worker
services, common jitter across strategies, and no SSH worker forwarding.

## Environment

- Fresh cloud server with 96 CPU cores and roughly 500 GiB memory.
- Python was provided by Miniconda; Docker was not installed on this server.
- The worker entrypoint is Docker-ready, but this run used one independent
  Python TCP service per logical worker.
- A direct external-port probe failed, so this server cannot support a
  non-forwarded cross-host worker deployment.  Cross-host validation should
  remain described as SSH-forwarded.

## Configuration

- Synthetic sparse ridge-regression workload.
- 12,000 samples, 1,200 features, density 0.008.
- 16 shards, 16 workers, 12 rounds.
- Phase-changing heterogeneous workers.
- Straggler fraction 0.35, straggler slowdown 0.12.
- Sleep scale 0.025, cost scale 0.005.
- Network stress: 8 ms RTT and 50 Mbps bandwidth model.
- Seeds: 11, 23, 31, 43.
- Strategies: speed-aware uncoded, speculative replication, static sparse
  flexible, rank-aware sparse flexible, and system portfolio.
- Baseline for paired gains: speed-aware uncoded.

## Aggregate Results

Latencies are averaged over four seeds and reported in milliseconds.

| Strategy | Mean Decode | p95 Decode | Mean Barrier | Mean Gain | p95 Gain | Barrier Gain |
|---|---:|---:|---:|---:|---:|---:|
| Speed-aware uncoded | 78.65 | 116.62 | 86.01 | 0.00% | 0.00% | 0.00% |
| Speculative replication | 79.74 | 118.91 | 87.39 | -1.37% | -1.90% | -1.59% |
| Static sparse flexible | 93.42 | 203.36 | 103.56 | -22.74% | -100.02% | -23.75% |
| Rank-aware sparse flexible | 66.79 | 77.00 | 79.15 | 13.87% | 29.79% | 6.83% |
| System portfolio | 63.29 | 75.09 | 72.53 | 18.34% | 31.40% | 14.63% |

## Bootstrap Confidence Intervals

Paired seed bootstrap, gains relative to speed-aware uncoded.

| Strategy | Metric | Mean | 95% CI |
|---|---|---:|---:|
| Rank-aware sparse flexible | Mean decode | 13.87% | 8.09--24.16% |
| Rank-aware sparse flexible | p95 decode | 29.79% | 19.51--47.03% |
| Rank-aware sparse flexible | Mean barrier | 6.83% | -2.58--18.89% |
| Rank-aware sparse flexible | p95 barrier | 16.29% | -11.84--42.61% |
| System portfolio | Mean decode | 18.34% | 12.49--27.92% |
| System portfolio | p95 decode | 31.40% | 21.23--48.09% |
| System portfolio | Mean barrier | 14.63% | 9.16--23.80% |
| System portfolio | p95 barrier | 26.65% | 16.21--43.65% |

## Interpretation

This fresh-server run is more conservative than the strongest paper table, but
it supports the same mechanism: static sparse flexible coding can be slower
than speed-aware uncoded when communication and extra coded work are
misaligned, while rank-aware scheduling and the fixed system portfolio recover
positive first-decode gains.

The cleanest paper use is as a reproducibility/rebuttal point:

> An independent fresh-server direct-service rerun was more conservative but
> positive: rank-aware p95 first-decode gain was 29.8%, and the fixed portfolio
> improved mean, p95, and barrier latency by 18.3%, 31.4%, and 14.6% over
> speed-aware uncoded.

This result does not remove the limitation that the current artifact lacks a
direct routable multi-node container deployment.
