#!/usr/bin/env python3
"""Master runner: generates all golden data JSON fixtures for integration tests."""

import json
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from generators import gen_metrics, gen_preprocessing, gen_knn, gen_decision_tree

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "golden_data")


def write_json(filename, data):
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Written: {path} ({len(data)} cases)")


def main():
    print("Generating golden data fixtures...")
    print()

    write_json("metrics.json", gen_metrics.generate())
    write_json("preprocessing.json", gen_preprocessing.generate())
    write_json("knn.json", gen_knn.generate())
    write_json("decision_tree.json", gen_decision_tree.generate())

    print()
    print("Done! All fixtures written to tests/golden_data/")


if __name__ == "__main__":
    main()
