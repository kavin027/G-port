# Two-Node TCP Validation Results

Run date: 2026-05-26

This run validates the remote-worker TCP path on a fresh server using SSH
local-forwarded worker ports.  The master ran locally and each logical worker
ran as an independent Python TCP service on the remote host.

## Configuration

- Workers/shards: 8/8
- Seeds: 17, 23, 31, 43
- Rounds per seed: 8
- Samples/features/density: 6000/800/0.008
- Scenario: `phase`
- Drift period: 4
- Straggler fraction: 0.45
- Straggler slowdown: 0.08
- Sleep/cost scale: 0.03/0.006
- Baseline: `speed_aware_uncoded`
- Network emulation: disabled; this uses the real SSH-forwarded TCP path
- Output root: `tunneled_remote_sweep_port50076`
- Diagnostics: `tunneled_remote_port50076_diagnostics`

## Aggregate Result

Positive gains are relative to `speed_aware_uncoded` within the same seed.

| Strategy | Mean decode (ms) | p95 decode (ms) | Mean barrier (ms) | Mean decode gain | p95 decode gain | Mean barrier gain |
|---|---:|---:|---:|---:|---:|---:|
| `speed_aware_uncoded` | 394.8 | 922.3 | 521.3 | 0.0% | 0.0% | 0.0% |
| `speculative_replication` | 452.2 | 1287.0 | 559.3 | -23.6% | -42.1% | -9.3% |
| `sparse_flexible_static` | 321.6 | 946.8 | 458.6 | 20.4% | -2.0% | 15.4% |
| `rank_aware_sparse_flexible` | 302.0 | 728.4 | 497.3 | 24.4% | 22.5% | 3.3% |
| `system_portfolio` | 408.6 | 936.4 | 670.9 | -6.5% | -4.8% | -30.1% |

## Bootstrap Confidence Intervals

The key positive result is rank-aware sparse flexible coding:

| Metric | Mean gain | 95% CI |
|---|---:|---:|
| Mean decode latency | 24.4% | 8.8% to 39.1% |
| p95 decode latency | 22.5% | 5.7% to 40.3% |
| Mean barrier latency | 3.3% | -15.3% to 20.7% |
| p95 barrier latency | -7.7% | -29.3% to 18.8% |

## Interpretation

This is a useful artifact-style sanity check, not the main scaling experiment.
It shows that the first-decodable-time mechanism still reduces mean and tail
decode latency after real cross-host socket transfer, remote worker processes,
and cancellation messages are included.

The barrier result is weaker than the single-server TCP experiment because the
SSH-forwarded setup adds cancellation and forwarding overhead.  This should be
described as a restricted-cloud validation path rather than as a full cluster
deployment.

The system portfolio scheduler is not a good headline for this two-node run:
its additional switching/cancellation behavior is too expensive under the SSH
tunnel.  For the paper, the clean statement is that rank-aware sparse flexible
coding keeps a statistically positive first-decode benefit on a real two-node
TCP path, while barrier-time claims should rely on the larger server-side TCP
experiment.

## Cleanup

After the sweep, no `coded_learning_exp.network_runtime` worker processes were
left running on the remote server.
