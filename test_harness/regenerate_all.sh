#!/usr/bin/env bash
# Regenerate every golden fixture from the pinned sklearn version.
# Run this any time the sklearn pin in requirements.txt is bumped.
#
# Usage: ./regenerate_all.sh
set -euo pipefail

cd "$(dirname "$0")"

uv run --with-requirements requirements.txt python3 check_sklearn_version.py

for gen in generators/gen_*.py; do
    echo "::: regenerating $(basename "$gen")"
    uv run --with-requirements requirements.txt python3 "$gen"
done

echo "All fixtures regenerated. Now run:"
echo "    cd .. && cargo test --workspace"
