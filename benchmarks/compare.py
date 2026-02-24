"""Compare sklearn and rustml benchmark results side-by-side.

1. Runs sklearn_benchmark.py and captures its JSON output.
2. Parses criterion benchmark results from target/criterion/*/new/estimates.json.
3. Prints a formatted comparison table with speedup ratios.
"""

import json
import os
import subprocess
import sys

# Root of the project (parent of benchmarks/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRITERION_DIR = os.path.join(PROJECT_ROOT, "target", "criterion")
SKLEARN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sklearn_benchmark.py")
SKLEARN_RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sklearn_results.json")

# Mapping from criterion benchmark group/id to the sklearn benchmark name.
# Each entry maps (criterion_group, criterion_id) -> sklearn name.
BENCHMARK_MAP = [
    # (criterion_group, criterion_function, sklearn_name)
    ("scaler_1000x50", "fit_transform", "StandardScaler fit+transform 1000x50"),
    ("scaler_10000x100", "fit_transform", "StandardScaler fit+transform 10000x100"),
    ("knn_1000x50", "fit", "KNN fit 1000x50"),
    ("knn_1000x50", "predict", "KNN predict 1000x50"),
    ("tree_5000x20", "fit", "DecisionTree fit 5000x20"),
    ("tree_5000x20", "predict", "DecisionTree predict 5000x20"),
    ("rf_5000x20", "fit", "RandomForest fit 5000x20"),
    ("rf_5000x20", "predict", "RandomForest predict 5000x20"),
    ("kmeans_5000x20", "fit", "KMeans fit 5000x20"),
    ("gnb_5000x20", "fit", "GaussianNB fit 5000x20"),
    ("gnb_5000x20", "predict", "GaussianNB predict 5000x20"),
]


def load_sklearn_results():
    """Load sklearn results, running the benchmark if no cached results exist."""
    # Try cached results file first
    if os.path.exists(SKLEARN_RESULTS_FILE):
        print(f"Loading cached sklearn results from {SKLEARN_RESULTS_FILE}", file=sys.stderr)
        with open(SKLEARN_RESULTS_FILE) as f:
            return json.load(f)

    # Run the sklearn benchmark
    print("Running sklearn benchmarks...", file=sys.stderr)
    result = subprocess.run(
        [sys.executable, SKLEARN_SCRIPT],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"sklearn benchmark failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return json.loads(result.stdout)


def load_criterion_results():
    """Parse criterion estimate files and return a dict of (group, id) -> mean_ms."""
    results = {}

    if not os.path.isdir(CRITERION_DIR):
        print(
            f"Criterion directory not found: {CRITERION_DIR}\n"
            "Run 'cargo bench -p rustml' first.",
            file=sys.stderr,
        )
        return results

    for group_name in os.listdir(CRITERION_DIR):
        group_path = os.path.join(CRITERION_DIR, group_name)
        if not os.path.isdir(group_path):
            continue

        # Criterion stores benchmarks as group_name/bench_id/new/estimates.json
        # or group_name/new/estimates.json for non-parameterised benchmarks.
        for entry in os.listdir(group_path):
            estimates_path = os.path.join(group_path, entry, "new", "estimates.json")
            if os.path.isfile(estimates_path):
                with open(estimates_path) as f:
                    data = json.load(f)
                # Point estimate is in nanoseconds
                mean_ns = data.get("mean", {}).get("point_estimate", None)
                if mean_ns is not None:
                    mean_ms = mean_ns / 1_000_000.0
                    results[(group_name, entry)] = mean_ms

        # Also check for direct new/estimates.json (single function in group)
        direct_path = os.path.join(group_path, "new", "estimates.json")
        if os.path.isfile(direct_path):
            with open(direct_path) as f:
                data = json.load(f)
            mean_ns = data.get("mean", {}).get("point_estimate", None)
            if mean_ns is not None:
                mean_ms = mean_ns / 1_000_000.0
                results[(group_name, "")] = mean_ms

    return results


def print_comparison(sklearn_data, criterion_data):
    """Print a formatted comparison table."""
    sklearn_by_name = {b["name"]: b for b in sklearn_data.get("benchmarks", [])}

    col_algo = "Algorithm"
    col_sklearn = "sklearn (ms)"
    col_rustml = "rustml (ms)"
    col_speedup = "Speedup"

    rows = []

    for group, bench_id, sklearn_name in BENCHMARK_MAP:
        sklearn_ms = None
        rustml_ms = None

        if sklearn_name in sklearn_by_name:
            sklearn_ms = sklearn_by_name[sklearn_name]["mean_ms"]

        if (group, bench_id) in criterion_data:
            rustml_ms = criterion_data[(group, bench_id)]

        speedup = ""
        if sklearn_ms is not None and rustml_ms is not None and rustml_ms > 0:
            ratio = sklearn_ms / rustml_ms
            speedup = f"{ratio:.1f}x"

        rows.append((
            sklearn_name,
            f"{sklearn_ms:>12.4f}" if sklearn_ms is not None else "         N/A",
            f"{rustml_ms:>11.4f}" if rustml_ms is not None else "        N/A",
            f"{speedup:>7s}" if speedup else "    N/A",
        ))

    # Print header
    w_algo = max(len(col_algo), max(len(r[0]) for r in rows))
    w_sk = max(len(col_sklearn), 12)
    w_rs = max(len(col_rustml), 11)
    w_sp = max(len(col_speedup), 7)

    header = f" {col_algo:<{w_algo}} | {col_sklearn:>{w_sk}} | {col_rustml:>{w_rs}} | {col_speedup:>{w_sp}}"
    sep = f" {'=' * w_algo} | {'=' * w_sk} | {'=' * w_rs} | {'=' * w_sp}"

    print()
    print(header)
    print(sep)

    for name, sk, rs, sp in rows:
        print(f" {name:<{w_algo}} | {sk:>{w_sk}} | {rs:>{w_rs}} | {sp:>{w_sp}}")

    print()


def main():
    sklearn_data = load_sklearn_results()
    criterion_data = load_criterion_results()

    if not criterion_data:
        print(
            "\nWarning: No criterion results found. Showing sklearn-only results.\n"
            "Run 'cargo bench -p rustml' to generate Rust benchmarks.\n",
            file=sys.stderr,
        )

    print_comparison(sklearn_data, criterion_data)


if __name__ == "__main__":
    main()
