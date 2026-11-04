#!/usr/bin/env bash
# Regenerate every golden fixture from the pinned sklearn version.
# Run this any time the sklearn pin in requirements.txt is bumped.
#
# Usage: ./regenerate_all.sh
set -euo pipefail

cd "$(dirname "$0")"

uv run --with-requirements requirements.txt python3 check_sklearn_version.py

# Generators without a __main__ block are run through generate_golden_data.py.
echo "::: regenerating generate_golden_data.py generators"
uv run --with-requirements requirements.txt python3 generate_golden_data.py
for gen in generators/gen_*.py; do
    grep -q "__main__" "$gen" || continue
    echo "::: regenerating $(basename "$gen")"
    uv run --with-requirements requirements.txt python3 "$gen"
done

echo "::: comparing with the committed fixtures"
python3 check_fixture_drift.py ../crates/anofox-ml/tests/golden_data \
    ../crates/anofox-ml-conformal/tests/golden_data || true

echo "All fixtures regenerated. Now run:"
echo "    cd .. && cargo test --workspace"
