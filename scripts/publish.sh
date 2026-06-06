#!/usr/bin/env bash
#
# Publishes every anofox-ml-* crate to crates.io in topological order.
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

# Topological order: anofox-ml-core has no internal deps, then everything else
# layered upward to the umbrella.
CRATES=(
    anofox-ml-core
    anofox-ml-text
    anofox-ml-io
    anofox-ml-svm
    anofox-ml-linear
    anofox-ml-neural-networks
    anofox-ml-naive-bayes
    anofox-ml-neighbors
    anofox-ml-trees
    anofox-ml-discriminant
    anofox-ml-metrics
    anofox-ml-cluster
    anofox-ml-gaussian-process
    anofox-ml-manifold
    anofox-ml-preprocessing
    anofox-ml-ensemble
    anofox-ml-regression
    anofox-ml
)

# Detect the workspace version once; we use it to skip crates that have
# already been published at this exact version (idempotent re-runs).
WS_VERSION=$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')
echo "Workspace version: ${WS_VERSION}"

is_already_published() {
    local name="$1"
    local code
    code=$(curl -fsS -o /dev/null -w "%{http_code}" \
        -H "User-Agent: anofox-ml-publish (sm@data-zoo.de)" \
        "https://crates.io/api/v1/crates/${name}/${WS_VERSION}" 2>/dev/null || echo "0")
    [[ "${code}" == "200" ]]
}

for crate in "${CRATES[@]}"; do
    echo "=========================================================="
    if [[ -z "${DRY_RUN_FLAG}" ]] && is_already_published "${crate}"; then
        echo "Skipping ${crate} v${WS_VERSION} — already on crates.io"
        echo "=========================================================="
        continue
    fi
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
