# Benchmarks: rustml vs scikit-learn

Benchmark comparison between rustml (Rust) and scikit-learn (Python) on
standardised datasets with identical shapes and parameters.

## Prerequisites

Python (with numpy and scikit-learn):

```bash
pip install numpy scikit-learn
```

Rust toolchain with cargo installed.

## Running the benchmarks

### 1. Run sklearn benchmarks

```bash
python benchmarks/sklearn_benchmark.py > benchmarks/sklearn_results.json
```

### 2. Run Rust benchmarks

```bash
cargo bench -p rustml
```

### 3. Compare results

```bash
python benchmarks/compare.py
```

The comparison script will:
- Load cached sklearn results from `benchmarks/sklearn_results.json` (or run
  `sklearn_benchmark.py` if the file does not exist).
- Parse criterion results from `target/criterion/*/new/estimates.json`.
- Print a side-by-side table with speedup ratios.

## Benchmark scenarios

| Algorithm       | Dataset        | Parameters                   |
|-----------------|----------------|------------------------------|
| StandardScaler  | 1000x50        | fit + transform              |
| StandardScaler  | 10000x100      | fit + transform              |
| KNN             | 1000x50        | k=5, Euclidean, 200 test     |
| Decision Tree   | 5000x20        | max_depth=10, 5 classes      |
| Random Forest   | 5000x20        | 100 trees, max_depth=10      |
| KMeans          | 5000x20        | k=10, max_iter=300           |
| Gaussian NB     | 5000x20        | 5 classes                    |

All datasets use seed 42 for reproducibility.
