# Score Ablation Diagnostic

Offline 8-worker/8-shard diagnostic.  The oracle column enumerates inclusion-minimal decodable subsets and is used only as a small-code reference, not as an online scheduler.

## Summary

| Policy | Mean pred. ms | p95 pred. ms | Gain vs static | Prefix delta |
|---|---:|---:|---:|---:|
| oracle-minset | 19.28 | 52.58 | 38.9% | -1.22 |
| cost-only | 25.39 | 60.78 | 19.5% | -0.35 |
| sampled-minset-128 | 27.63 | 54.53 | 12.5% | -0.82 |
| sampled-minset-512 | 28.28 | 65.01 | 10.4% | 0.02 |
| static | 31.56 | 117.01 | 0.0% | 0.00 |
| rho/C | 32.77 | 85.62 | -3.8% | 0.00 |
| rho*C | 32.77 | 85.62 | -3.8% | 0.00 |
| rho | 32.77 | 85.62 | -3.8% | 0.00 |
| random | 38.22 | 141.23 | -21.1% | 0.45 |

## Oracle Correlation

- mean corr(rho, oracle-minset frequency): -0.00
- mean corr(rho*C, oracle-minset frequency): -0.00
- mean corr(cost, oracle-minset frequency): 0.12
- mean corr(sampled-minset-128, oracle-minset frequency): 0.51
- mean corr(sampled-minset-512, oracle-minset frequency): 0.62

## Score Build Time

| Policy | Mean build ms | p95 build ms | Max build ms |
|---|---:|---:|---:|
| rho-family | 0.10 | 0.11 | 0.11 |
| sampled-minset-128 | 68.05 | 74.11 | 75.00 |
| sampled-minset-512 | 279.02 | 312.91 | 315.59 |
| oracle-minset | 1490.29 | 1527.91 | 1529.16 |
