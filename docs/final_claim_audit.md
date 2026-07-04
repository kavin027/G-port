# Final Claim Audit

Date: 2026-05-26

## Main Claim

The paper should consistently claim:

> First-decodable-time scheduling is a runtime layer for sparse flexible coded
> learning.  It helps when decode-important rows are misaligned with fast
> workers and when communication/cancellation pressure makes the
> first-decodable row prefix valuable.

## Non-Claims

The paper should not imply:

- The method is a new sparse flexible code construction.
- Decode-aware scheduling always beats uncoded, speculative, or speed-aware
  placement.
- The surrogate analysis is a full probabilistic optimality theorem for exact
  first-decode time.
- The restricted two-node SSH-forwarded experiment is a full cluster
  deployment.
- The current ridge-regression workload proves general benefit for all ML
  training systems.

## Claim-to-Evidence Map

| Claim | Evidence | Current framing |
|---|---|---|
| Row placement changes first-decode time without changing code recovery | Lemma 1 and runtime exact decodability checks | Safe |
| Decode-speed mismatch creates scheduling opportunity | Decode-mass surrogate and controlled alignment sweep | Safe if called surrogate analysis |
| Adaptive placement can reduce mean/tail latency | Simulation, multi-process runtime, same-host TCP | Safe |
| Strong uncoded baselines are boundary cases | CPU stress negative results plus WAN/TCP reversal | Safe |
| Communication pressure can restore coded advantage over speed-aware uncoded | 8 ms / 50 Mbps TCP stress | Main systems evidence |
| Cross-host path does not erase first-decode benefit | Restricted two-node SSH-forwarded run | Sanity check only |
| Barrier-time gains are robust | Same-host TCP and network-constrained TCP only | Do not extend to two-node |

## Text Checks Applied

- Abstract now says gains are conditional and ends with the row-prefix/overhead
  condition.
- Introduction explicitly says the systems contribution is a runtime layer, not
  a new code construction.
- Theory section is titled `Surrogate Analysis` and states its limitation.
- Evaluation keeps two-node results as a cross-host sanity check.
- Limitations distinguish same-host TCP, analytic network stress, and
  SSH-forwarded two-node validation.

## Remaining Risk

The strongest remaining rejection path is still systems realism: a reviewer may
want direct multi-node worker ports, container scheduling, and a larger ML
training job.  The best response is to keep the submission scoped as a
prototype systems paper about a missing scheduling layer, with clear artifact
commands and honest boundary cases.
