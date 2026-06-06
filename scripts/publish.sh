#!/usr/bin/env bash
#
# Publishes every rustml-* crate to crates.io in topological order.
#
# Why an explicit order: cargo refuses to upload a crate whose internal
# dependencies aren't yet on crates.io, so leaf-first is mandatory. Between
# crates we sleep briefly to let the registry index catch up — otherwise
# the next `cargo publish` may not see the version we just pushed.
#
# Usage:
#     scripts/publish.sh           # dry-run (default — won't actually upload)
#     scripts/publish.sh --execute # real publish; requires `cargo login`
#
# Prerequisites:
#  - `cargo login <token>` against crates.io
#  - workspace clean (`git status` empty) and tagged for the release
#  - workspace.package.version bumped in the root Cargo.toml
#
# Failure modes:
#  - Network flakes: re-run; cargo will skip crates that already match the
#    published version.
#  - "crate version already exists": you bumped the workspace version but
#    a previous run already pushed some crates. Either bump again or push
#    only the unpublished tail manually.

set -euo pipefail

DRY_RUN_FLAG="--dry-run"
SLEEP_SECS=0
if [[ "${1:-}" == "--execute" ]]; then
    DRY_RUN_FLAG=""
    SLEEP_SECS=20
fi

# Topological order: rustml-core has no internal deps, then everything else
# layered upward to the umbrella.
CRATES=(
    rustml-core
    rustml-text
    rustml-io
    rustml-svm
    rustml-linear
    rustml-neural-networks
    rustml-naive-bayes
    rustml-neighbors
    rustml-trees
    rustml-discriminant
    rustml-metrics
    rustml-cluster
    rustml-gaussian-process
    rustml-manifold
    rustml-preprocessing
    rustml-ensemble
    rustml-regression
    rustml
)

for crate in "${CRATES[@]}"; do
    echo "=========================================================="
    echo "Publishing ${crate}${DRY_RUN_FLAG:+ (dry-run)}"
    echo "=========================================================="
    cargo publish -p "${crate}" ${DRY_RUN_FLAG}
    if [[ -z "${DRY_RUN_FLAG}" ]]; then
        echo "Sleeping ${SLEEP_SECS}s for crates.io index to update..."
        sleep "${SLEEP_SECS}"
    fi
done

echo ""
echo "All crates pushed."
