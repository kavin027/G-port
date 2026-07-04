# System baseline findings

These experiments were added after the simulated SoCC reviews asked for uncoded and speculative straggler-mitigation baselines.

## baseline_24w16s

| strategy                       |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:-------------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| uncoded_sync                   |                0.1895 |               0.3095 |                 0.1926 |              0.0001 |          1.0000 |         16.0000 |
| replication                    |                0.2073 |               0.3532 |                 0.2283 |              0.0008 |          1.3613 |         21.7917 |
| speculative_replication        |                0.2035 |               0.3236 |                 0.2264 |              0.0010 |          1.2892 |         20.6167 |
| sparse_flexible_static         |                0.2204 |               0.3978 |                 0.2738 |              0.0001 |          3.1124 |         21.8833 |
| rank_aware_sparse_flexible     |                0.2447 |               0.3628 |                 0.3002 |              0.0045 |          3.4805 |         24.3833 |
| deadline_aware_sparse_flexible |                0.2372 |               0.3636 |                 0.2916 |              0.0055 |          3.2768 |         23.0667 |

- Best mean_decode_latency: `uncoded_sync` = 0.1895s.
- Best p95_decode_latency: `uncoded_sync` = 0.3095s.
- Best mean_barrier_latency: `uncoded_sync` = 0.1926s.
- Decode-aware vs speculative replication: mean -20.3%, p95 -12.1%.

## stress_32w16s

| strategy                       |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:-------------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| uncoded_sync                   |                0.2543 |               0.3633 |                 0.2579 |              0.0001 |          1.0000 |         16.0000 |
| speculative_replication        |                0.2442 |               0.3989 |                 0.2758 |              0.0014 |          1.4824 |         23.7167 |
| sparse_flexible_static         |                0.2854 |               0.4657 |                 0.3533 |              0.0009 |          3.7459 |         26.3583 |
| rank_aware_sparse_flexible     |                0.2530 |               0.3997 |                 0.3271 |              0.0094 |          3.0977 |         21.7417 |
| deadline_aware_sparse_flexible |                0.2720 |               0.4138 |                 0.3372 |              0.0092 |          3.1415 |         22.0333 |

- Best mean_decode_latency: `speculative_replication` = 0.2442s.
- Best p95_decode_latency: `uncoded_sync` = 0.3633s.
- Best mean_barrier_latency: `uncoded_sync` = 0.2579s.
- Decode-aware vs speculative replication: mean -3.6%, p95 -0.2%.

## degree_probe_32w16s_2seed

| strategy                         |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:---------------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| uncoded_sync                     |                0.2347 |               0.4418 |                 0.2354 |              0.0001 |          1.0000 |         16.0000 |
| speculative_replication          |                0.1772 |               0.3261 |                 0.2264 |              0.0004 |          1.3804 |         22.0833 |
| thin_sparse_flexible_static      |                0.3294 |               0.6432 |                 0.3719 |              0.0027 |          3.5985 |         42.4444 |
| thin_rank_aware_sparse_flexible  |                0.2498 |               0.4065 |                 0.3095 |              0.0120 |          2.6747 |         32.6944 |
| light_rank_aware_sparse_flexible |                0.2381 |               0.4693 |                 0.2990 |              0.0072 |          2.8615 |         22.8889 |
| sparse_flexible_static           |                0.2634 |               0.4138 |                 0.3208 |              0.0004 |          3.8028 |         26.6111 |
| rank_aware_sparse_flexible       |                0.2414 |               0.4536 |                 0.2991 |              0.0085 |          3.1933 |         22.5278 |

- Best mean_decode_latency: `speculative_replication` = 0.1772s.
- Best p95_decode_latency: `speculative_replication` = 0.3261s.
- Best mean_barrier_latency: `speculative_replication` = 0.2264s.
- Decode-aware vs speculative replication: mean -36.2%, p95 -39.1%.

## hybrid_probe_32w16s_2seed

| strategy                   |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:---------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| uncoded_sync               |                0.2443 |               0.4207 |                 0.2473 |              0.0001 |          1.0000 |         16.0000 |
| speculative_replication    |                0.2271 |               0.3719 |                 0.2626 |              0.0055 |          1.4741 |         23.5833 |
| sparse_flexible_static     |                0.2594 |               0.4731 |                 0.3098 |              0.0004 |          3.5311 |         24.9167 |
| rank_aware_sparse_flexible |                0.2383 |               0.4166 |                 0.2853 |              0.0115 |          3.1565 |         22.1667 |
| hybrid_decode_replication  |                0.2647 |               0.4556 |                 0.3199 |              0.0924 |          1.6181 |         23.7778 |

- Best mean_decode_latency: `speculative_replication` = 0.2271s.
- Best p95_decode_latency: `speculative_replication` = 0.3719s.
- Best mean_barrier_latency: `uncoded_sync` = 0.2473s.
- Decode-aware vs speculative replication: mean -4.9%, p95 -12.0%.

## Interpretation

- Speculative replication is a strong baseline in the current single-machine runtime because it avoids coded-row expansion and can place or duplicate uncoded shards on fast workers.
- Decode-aware assignment still improves over static sparse-flexible placement in many settings, but it does not yet dominate uncoded or speculative replication in the added system-baseline sweeps.
- This is a paper-scope boundary: the current contribution is best framed as a runtime scheduler for sparse flexible coded learning, not as a universal replacement for replication/speculation.
- A stronger SoCC version needs either a hybrid code/replication portfolio with low prediction overhead or a multi-node setting where network and tail effects make coded recovery more valuable than uncoded replication.

## system_portfolio_probe_32w16s_2seed

| strategy                       |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:-------------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| uncoded_sync                   |                0.2260 |               0.3722 |                 0.2293 |              0.0001 |          1.0000 |         16.0000 |
| speed_aware_uncoded            |                0.1877 |               0.3428 |                 0.1883 |              0.0002 |          1.0000 |         16.0000 |
| speculative_replication        |                0.2120 |               0.3675 |                 0.2487 |              0.0003 |          1.4790 |         23.6667 |
| sparse_flexible_static         |                0.2460 |               0.4201 |                 0.2989 |              0.0033 |          3.4640 |         24.3056 |
| rank_aware_sparse_flexible     |                0.2329 |               0.4215 |                 0.3003 |              0.0001 |          3.2219 |         22.5278 |
| fast_hybrid_decode_replication |                0.3114 |               0.5553 |                 0.3487 |              0.0144 |          1.6650 |         24.1111 |
| system_portfolio               |                0.2268 |               0.3441 |                 0.2381 |              0.0279 |          1.0729 |         16.6389 |

- Best mean_decode_latency: `speed_aware_uncoded` = 0.1877s.
- Best p95_decode_latency: `speed_aware_uncoded` = 0.3428s.
- Best mean_barrier_latency: `speed_aware_uncoded` = 0.1883s.
- System portfolio vs speed-aware uncoded: mean -20.8%, p95 -0.4%, barrier -26.5%.

Updated interpretation:

- Speed-aware uncoded scheduling is the strongest current single-machine system baseline.
- The low-overhead portfolio reduces the previous hybrid overhead substantially and nearly matches the speed-aware uncoded p95 latency in the 2-seed probe, but it does not beat the best baseline yet.
- The SoCC-facing story should therefore keep portfolio scheduling as future work or an explicit negative prototype, not as a claimed main result.


## portfolio_sweep_32w16s_4seed

| strategy                   |   mean_decode_latency |   p95_decode_latency |   mean_barrier_latency |   scheduler_seconds |   extra_compute |   selected_rows |
|:---------------------------|----------------------:|---------------------:|-----------------------:|--------------------:|----------------:|----------------:|
| speed_aware_uncoded        |                0.1957 |               0.2919 |                 0.2018 |              0.0012 |          1.0000 |         16.0000 |
| speculative_replication    |                0.2349 |               0.3892 |                 0.2720 |              0.0013 |          1.4651 |         23.4375 |
| sparse_flexible_static     |                0.2778 |               0.4386 |                 0.3528 |              0.0032 |          3.7020 |         25.9271 |
| rank_aware_sparse_flexible |                0.2260 |               0.3770 |                 0.2966 |              0.0001 |          3.0596 |         21.6042 |
| system_portfolio           |                0.2348 |               0.3882 |                 0.2404 |              0.0431 |          1.0397 |         16.4375 |
| learned_system_portfolio   |                0.2258 |               0.4146 |                 0.2595 |              0.0023 |          1.6431 |         20.6875 |

- Best mean_decode_latency: `speed_aware_uncoded` = 0.1957s.
- Best p95_decode_latency: `speed_aware_uncoded` = 0.2919s.
- Best mean_barrier_latency: `speed_aware_uncoded` = 0.2018s.
- system_portfolio vs speed-aware uncoded: mean -19.9%, p95 -33.0%, barrier -19.1%.
- learned_system_portfolio vs speed-aware uncoded: mean -15.3%, p95 -42.0%, barrier -28.6%.
- rank_aware_sparse_flexible vs speed-aware uncoded: mean -15.5%, p95 -29.1%, barrier -47.0%.

Updated 4-seed conclusion:

- Speed-aware uncoded remains the strongest single-machine baseline in this stress setting.
- The prediction-based portfolio improves over static sparse-flexible barrier latency and uses little extra compute, but its scheduler overhead and p95 latency prevent it from beating speed-aware uncoded.
- The online learned portfolio has low overhead, but the current contextual bandit does not learn a superior policy within 24 rounds.
- For a SoCC-strength claim, the next decisive evidence must come from a real multi-node deployment or a workload where coded recovery reduces communication/straggler cost that speed-aware uncoded cannot avoid.

## network_constrained_tcp_16w16s_4seed

We reran the TCP worker-service runtime with an emulated network transfer
model: 16 workers, 16 shards, phase-changing stragglers, 8 ms RTT, 50 Mbps
bandwidth, and four seeds (11/23/31/43).  The reported numbers are from a
fresh-server reproduction on May 26, 2026.  The rerun uses a common
worker/iteration jitter stream across strategies while preserving per-row
lognormal jitter.  Positive gain is relative to the speed-aware uncoded
baseline in the same seed.

| strategy                   |   mean_decode_ms |   p95_decode_ms |   mean_barrier_ms |   mean_gain_vs_speed |   p95_gain_vs_speed |   barrier_gain_vs_speed |
|:---------------------------|-----------------:|----------------:|------------------:|---------------------:|--------------------:|------------------------:|
| speed_aware_uncoded        |            102.3 |           164.3 |             110.1 |                 0.0% |                0.0% |                    0.0% |
| speculative_replication    |            103.7 |           167.4 |             111.5 |                -1.4% |               -1.8% |                   -1.2% |
| sparse_flexible_static     |            110.4 |           265.8 |             120.9 |                -7.8% |              -61.8% |                   -9.7% |
| rank_aware_sparse_flexible |             73.0 |            83.8 |              83.6 |                28.3% |               48.5% |                   23.7% |
| system_portfolio           |             69.7 |            83.4 |              78.9 |                31.6% |               48.8% |                   28.1% |
| learned_system_portfolio   |            103.8 |           164.3 |             111.7 |                -1.4% |               -0.0% |                   -1.3% |

Paired bootstrap intervals over the four seeds are positive for the strongest
signals: system_portfolio improves p95 first-decode latency by 48.8% with a
95% interval of 44.1--52.8%, while rank-aware sparse-flexible improves p95 by
48.5% with a 43.5--52.9% interval.

Updated interpretation:

- The single-machine negative result is a boundary condition, not a universal
  defeat of coded scheduling by speed-aware uncoded placement.
- When per-row network transfer and cancellation matter, first-decodable row
  count again becomes a useful systems objective.  The system portfolio is
  especially strong because it avoids unnecessary coded expansion on easy
  rounds while switching to coded placement when the decode prefix is likely
  to be shorter.
- This is still not a true multi-node deployment.  The result is best framed
  as a network-constrained TCP stress test that motivates, but does not replace,
  a future cross-host experiment.
