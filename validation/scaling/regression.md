# Regression — scaling profile

How does regression `fit` time grow with `n`? Run the benches below and
fill in the table.

## Estimators measured

| Estimator | Bench group | Sizes (n × d) | Expected asymptotic |
|---|---|---|---|
| Ridge | `scaling_ridge` | (1k, 10k, 100k) × 20 | **O(n · p² + p³)** — linear in n once p is fixed |
| RandomForestRegressor | `scaling_rf_regressor` | (1k, 5k, 25k) × 20 | **O(n_trees · n · log n · p)** — quasi-linear in n |

Ridge gets a 100× sweep (1k → 100k) because the closed-form solver is
linear in n; you should see exactly that. The forest sweep stops at 25k
because each tree's fit is O(n · log n) and 100 trees × 25k × 14 bits ≈
seconds even on a modest laptop.

## Running

```bash
cargo bench -p anofox-ml -- scaling_ridge scaling_rf_regressor
```

## Results template

### Ridge `fit`, α=1.0, p=20

| n | time | grow-vs-prev | linear ratio |
|--:|--:|--:|--:|
|   1,000 | <fill ms> | — | 1× |
|  10,000 | <fill ms> | <fill> | 10× |
| 100,000 | <fill ms> | <fill> | 100× |

Closed-form Ridge is dominated by the `XᵀX` matmul (O(n·p²)) and the
p×p Cholesky (O(p³)). At p=20 the matmul wins for n ≥ 1000, so growth
should be cleanly linear. A super-linear curve here means the p³ term
is unexpectedly dominant — file a perf issue.

### RandomForestRegressor `fit`, 100 trees, max_depth=10, p=20

| n | time | grow-vs-prev | n log n ratio |
|--:|--:|--:|--:|
|  1,000 | <fill ms> | — | 1× |
|  5,000 | <fill ms> | <fill> | ~5.8× |
| 25,000 | <fill ms> | <fill> | ~35× |

The "n log n" ratio approximates how a single tree's split-finding grows
with n (sort each feature column → log n; pick best splits → n). With
rayon-parallel tree construction, observed growth is usually
sub-linear in n_trees but matches the n·log n trend in n.
