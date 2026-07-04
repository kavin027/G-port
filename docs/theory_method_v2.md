# Decode-Aware Sparse Flexible Coded Learning: Theory and Method Draft

## 1. Motivation From Pilot Experiments

The pilot experiments suggest three important lessons.

1. Cost-aware assignment is not reliable. Assigning the most expensive encoded
   rows to the fastest workers can delay the rows that actually determine
   decodability.
2. Decode-aware and deadline-aware assignment are consistently useful in the
   small-scale experiments. In `organized_strengthen_smoke`, decode-aware
   assignment improves mean latency by 24.6%, and deadline-aware assignment by
   24.7% over the sparse-flexible baseline.
3. A naive decode-balanced code construction is not enough. It balances the full
   decoding vector, but it may increase computation and hurt early stopping.

The paper should therefore focus on optimizing the first decodable time, not
only sparsity or computation cost.

## 2. Problem Formulation

At iteration t, the full gradient is decomposed over m data shards:

```text
g(w_t) = sum_{j=1}^m g_j(w_t).
```

A two-layer sparse flexible code uses

```text
B = [A^(1); A^(2)] in R^{2n x m},
```

where n is the number of workers. Encoded row r computes

```text
g_tilde_r = b_r^T G_t = sum_j b_{rj} g_j(w_t).
```

For a completed row set S, decoding is possible if

```text
1_m in span(B_S^T).
```

Equivalently, there exists a decoding vector c_S such that

```text
B_S^T c_S = 1_m.
```

Let assignment pi map encoded row pairs to workers. The true objective is the
first decodable time:

```text
tau(pi, B) = inf { time u : 1_m in span(B_{S_pi(u)}^T) }.
```

The scheduling problem is

```text
min_pi E[tau(pi, B)].
```

This objective is stronger and more direct than minimizing total computation
or assigning high-cost rows to fast workers.

## 3. Target-Specific Decode Importance

For the full two-layer code B, define the minimum-norm target decoder:

```text
z* = argmin_z ||z||_2^2
     s.t. B^T z = 1_m.
```

When B has full column rank with respect to the target vector,

```text
z* = B (B^T B)^dagger 1_m.
```

The target-specific decode importance of row r is

```text
rho_r = |z*_r|.
```

For worker task pair i, which contains first-layer row i and second-layer row
i+n, define pair importance:

```text
R_i = (rho_i + rho_{i+n}) C_i,
```

where C_i is the estimated sparse computation cost of the two rows.

This score measures whether a row pair is both useful for decoding and costly
enough that assignment matters.

## 4. Decode-Aware Assignment

The decode-aware rule sorts encoded row pairs by R_i and workers by speed:

```text
R_{i_1} >= R_{i_2} >= ... >= R_{i_n}
s_{j_1} >= s_{j_2} >= ... >= s_{j_n}.
```

Then assign pair i_k to worker j_k.

This is the implemented `rank_aware_sparse_flexible` method.

### Theorem 1: Decodability Invariance

For any assignment pi, row assignment changes completion order but not the row
space of B. Therefore, the algebraic recovery capability of the sparse flexible
code is unchanged.

In particular, for any permutation matrix P,

```text
rank(B) = rank(PB),
```

and

```text
1_m in span(B^T) iff 1_m in span((PB)^T).
```

Thus decode-aware assignment preserves the recovery threshold of the underlying
sparse flexible code.

### Theorem 2: Optimality for a Decode-Mass Surrogate

Fix a deadline u. Suppose worker completion probabilities p_j(u) are monotone
in worker speed, and the scheduler maximizes expected completed decode mass:

```text
max_pi sum_i p_{pi(i)}(u) R_i.
```

Then the optimal assignment is the monotone matching that pairs the largest
R_i with the largest p_j(u).

This follows from the rearrangement inequality. It gives a theoretical reason
why decode-aware assignment improves early decoding probability, while pure
cost-aware assignment can fail.

### Conditional Benefit: Opportunity vs. Realized Gain

The server worker-scaling experiments show that the right theory should not
claim unconditional improvement. The refined claim has two parts:

1. Decode-speed mismatch creates a scheduling opportunity.
2. Reducing the first-decodable completed-row set realizes latency gain.

For a deadline u, define the normalized decode-speed mismatch of the static
assignment pi_0:

```text
M(pi_0, u) =
  [max_pi sum_i R_i p_{pi(i)}(u) - sum_i R_i p_{pi_0(i)}(u)]
  /
  [max_pi sum_i R_i p_{pi(i)}(u) - min_pi sum_i R_i p_{pi(i)}(u)].
```

If M=0, static placement is already optimal for the decode-mass surrogate, so
decode-aware scheduling has no surrogate opportunity. If M>0, monotone
decode-aware assignment improves the surrogate by M times the assignment
range.

This still does not guarantee lower latency. Let T_{pi,(k)} be the kth encoded
row arrival time under assignment pi, and let K_pi be the first k such that the
first k arrivals are decodable. Then

```text
tau_pi = T_{pi,(K_pi)} + H(pi).
```

For static placement pi_0 and adaptive placement pi, if K_pi <= K_pi0:

```text
tau_pi0 - tau_pi =
  [T_{pi0,(K_pi0)} - T_{pi0,(K_pi)}]     # row-set reduction
  + [T_{pi0,(K_pi)} - T_{pi,(K_pi)}]     # faster placement
  - H(pi).
```

If K_pi increases, adaptive scheduling must compensate for extra completed
rows through faster placement. This explains the negative cases.

- Worker heterogeneity must be large enough that p_j(u) varies across workers.
- Decode importance must be heterogeneous enough that sorting by R_i matters.
- The code and assignment should reduce the first-decodable set size, not only
  move high-cost rows earlier.
- Scheduler overhead must be small compared with the deadline-latency savings.

In the proportional server scaling experiment:

- At 8 workers / 8 shards, static sparse-flexible already decodes after about
  51.6% of encoded rows. Decode-aware increases this to about 54.8%, so p95
  latency becomes worse even though scheduler overhead is only about 1.05 ms.
- At 16 workers / 16 shards, decode-aware reduces selected rows from 59.7% to
  54.4% and improves p95 latency by 43.1%.
- At 24 workers / 24 shards, decode-aware reduces selected rows from 70.6% to
  54.1% and improves p95 latency by 56.4%; deadline-aware reaches 59.1%.
- At 32 workers / 32 shards, decode-aware reduces selected rows from 68.8% to
  60.7% and improves p95 latency by 30.4%.

Therefore the strongest theoretical statement for the current paper is:

```text
Decode-aware assignment is conditionally optimal for a decode-mass surrogate.
Decode-speed mismatch creates the opportunity to improve first-decode latency,
but the realized latency gain requires the induced first-decodable row index
and scheduler overhead to be favorable.
```

The controlled real-data alignment sweep supports this version. When static
placement is aligned with decode-priority rows (M close to 0), decode-aware and
deadline-aware scheduling lose tail latency. When placement is random or
anti-aligned (M around 0.5 or 1.0), positive tail gains appear in some dataset
and strategy combinations. The overall correlation between controlled mismatch
and adaptive p95 improvement is positive (about r=0.50), but not perfect,
confirming that mismatch is an opportunity condition rather than a sufficient
condition.

## 5. Deadline-Aware Assignment

Decode-aware sorting assumes a separable surrogate. A stronger approximation
directly models the deadline effect.

For each task pair i and worker j, estimate nominal completion time:

```text
T_hat_{ij} = delay_j + alpha C_i / s_j.
```

Let q be a target deadline, chosen as the median nominal completion time. Define
a soft completion probability:

```text
P_{ij}(q) = exp(-T_hat_{ij} / q).
```

Then solve the assignment:

```text
max_pi sum_i R_i P_{i,pi(i)}(q).
```

This is a linear assignment problem and can be solved by the Hungarian
algorithm. It is implemented as `deadline_aware_sparse_flexible`.

This method is closer to the true objective E[tau(pi, B)] because it jointly
uses decode importance, row cost, worker speed, and current delay.

## 6. Online Configuration Selection

The assignment rule can be combined with online choice of code degrees:

```text
K = {(1,2), (2,2), (2,3), (3,4), (4,5)}.
```

At iteration t, the scheduler chooses k_t in K and observes iteration time
T_t(k_t). A simple UCB reward is

```text
R_t(k_t) = -T_t(k_t).
```

The pilot experiments show that UCB/window adaptation helps, but it should be
an enhancement, not the main claim.

## 7. Negative Result: Naive Decode-Balanced Code

We tested a sampled code design:

```text
min_B max_r |z*_r| / mean_r |z*_r|
```

with additional coefficient-variance penalties. In the small experiment, the
balanced code alone is slower and balanced rank-aware assignment is weaker than
random-code rank-aware assignment.

Interpretation: balancing the full decoding vector is not sufficient. A useful
code-design objective should optimize early decodability under partial row
completion, not only full-code coefficient concentration.

This gives a natural future direction:

```text
min_B E[tau(pi(B), B)]
```

subject to sparsity and flexible decoding constraints.

## 8. Current Main Claim

The strengthened main claim should be:

```text
Sparse flexible coded learning should schedule rows by target-specific decoding
contribution, not by computation cost alone. A deadline-aware decode assignment
preserves sparse-flexible decodability and can substantially reduce mean and
tail iteration latency when decode-priority rows are misaligned with early
worker completions. Its largest gains occur when reassignment reduces the
first-decodable completed-row set; negative cases occur when it adds extra
rows or overhead without changing the first decodable time.
```

In the latest smoke suite:

- Decode-aware assignment: 24.6% mean latency improvement.
- Deadline-aware assignment: 24.7% mean latency improvement.
- Decode-aware under sparse inputs: 28.6% mean latency improvement.
- Best online policy vs best fixed: 29.3% mean latency improvement.
- Naive decode-balanced code: negative result.
