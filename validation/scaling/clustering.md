# Clustering — scaling profile

How does clustering wall-time grow as `n` grows? Run the bench groups below
and fill in the table; the asymptotic column tells you what "good" looks
like before you click run.

## Estimators measured

| Estimator | Bench group | Sizes (n × d) | Expected asymptotic |
|---|---|---|---|
| KMeans | `scaling_kmeans` | (1k, 5k, 25k) × 20 | **O(n · k · d · iter)** — linear in n |
| AgglomerativeClustering (Ward) | `scaling_agglo_ward` | (200, 800, 2500) × 20 | **O(n²)** (nn-chain); O(n³) on the legacy naive sweep |

Why the size split: KMeans is linear in `n` so 25× more samples should cost
~25× more wall-time. Ward's nn-chain is quadratic, so 12.5× more samples
(200 → 2500) should cost ~155×; the naive sweep would be ~2000× — that's
where the recent algorithm change earns its keep.

## Running

```bash
cargo bench -p anofox-ml -- scaling_
```

criterion writes per-sample timings to `target/criterion/scaling_<group>/`.
Look at `report/index.html` (browser) or the `estimates.json` files (jq).

## Results template

Fill in once you've run the bench. The growth-factor columns are derived
(t(n) / t(n_prev)).

### KMeans `fit`, k=10, d=20, 100 iter cap

| n | time | grow-vs-prev | linear ratio |
|--:|--:|--:|--:|
|  1,000 | <fill ms> | — | 1× |
|  5,000 | <fill ms> | <fill> | 5× |
| 25,000 | <fill ms> | <fill> | 25× |

Read the "grow-vs-prev" column: if it tracks the "linear ratio" column
within ~1.5×, the implementation matches the theoretical O(n) per
iteration. Higher = either iter-count grew (k-means++ initialization is
n-dependent) or the per-step assignment got slower (cache pressure).

### AgglomerativeClustering Ward `fit`, d=20

| n | time | grow-vs-prev | n² ratio |
|--:|--:|--:|--:|
|    200 | <fill ms> | — | 1× |
|    800 | <fill ms> | <fill> | 16× |
|  2,500 | <fill ms> | <fill> | ~156× |

The nn-chain path is the default for Ward; set
`RUSTML_AGGLO_NAIVE=1` to force the legacy O(n³) sweep for comparison.
Expect the naive path to be ~10× slower than nn-chain at n=2500 and
unusable past n≈5000.
