#!/usr/bin/env python3
"""Sanity-check: golden fixtures were generated against sklearn 1.8.0.

Any change to the pinned version requires re-running every generator under
`generators/` and re-running the golden-test suite to confirm tolerances
still hold. This script fails fast if the installed sklearn differs from
the pinned version.

Used as a pre-flight in CI before `cargo test`.
"""

import sys

EXPECTED = "1.8.0"


def main() -> int:
    try:
        import sklearn
    except ImportError:
        print("ERROR: scikit-learn is not installed.", file=sys.stderr)
        return 1
    if sklearn.__version__ != EXPECTED:
        print(
            f"ERROR: golden fixtures were generated against sklearn {EXPECTED}, "
            f"but the installed version is {sklearn.__version__}.",
            file=sys.stderr,
        )
        print(
            "Re-run all generators under test_harness/generators/ and the "
            "golden-test suite before bumping requirements.txt.",
            file=sys.stderr,
        )
        return 2
    print(f"sklearn {sklearn.__version__} matches pin.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
