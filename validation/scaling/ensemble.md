# Ensemble — scaling profile

How does ensemble `fit` time grow with `n`? Run the benches below and fill
in the table.

## Estimators measured

| Estimator | Bench group | Sizes (n × d) | Expected asymptotic |
|---|---|---|---|
| RandomForestClassifier | `scaling_random_forest` | (1k, 5k, 25k) × 20 | **O(n_trees · n · log n · p)** — embarrassingly parallel across trees |
| GradientBoostingClassifier | `scaling_gradient_boosting` | (1k, 5k, 25k) × 20 | **O(n_estimators · n · log n · p)** — sequential (residual-dependent) |

The two grow with the same per-tree complexity but RF runs trees in
parallel via rayon, GBM runs them sequentially because each tree learns
the residuals of the previous one. So at 25k samples the wall-time gap
between the two algorithms widens by roughly `num_cpus`.

## Running

```bash
cargo bench -p anofox-ml -- scaling_random_forest scaling_gradient_boosting
```

## Results template

### RandomForestClassifier `fit`, 100 trees, max_depth=10, p=20

| n | time | grow-vs-prev | n log n ratio | parallel speedup vs GBM at same n |
|--:|--:|--:|--:|--:|
|  1,000 | <fill ms> | — | 1× | <fill> |
|  5,000 | <fill ms> | <fill> | ~5.8× | <fill> |
| 25,000 | <fill ms> | <fill> | ~35× | <fill> |

The last column is the differentiator: it should approach `num_cpus` at
larger n once the per-tree cost dominates rayon's scheduling overhead.

### GradientBoostingClassifier `fit`, 100×depth-3, p=20

| n | time | grow-vs-prev | n log n ratio |
|--:|--:|--:|--:|
|  1,000 | <fill ms> | — | 1× |
|  5,000 | <fill ms> | <fill> | ~5.8× |
| 25,000 | <fill ms> | <fill> | ~35× |

Sequential — no rayon speedup. Lighter trees (max_depth=3) means each tree
is cheaper than RF's, but the lack of parallelism makes total wall-time
larger past a few k samples.

## What this section answers

The single-point comparison vs sklearn (in the README) doesn't show how a
choice between RF and GBM scales with the user's actual data size. Pick
the algorithm that wins at *your* n, not the one that wins at our 5k
benchmark.
