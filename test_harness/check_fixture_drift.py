#!/usr/bin/env python3
"""Compare regenerated golden fixtures with the committed ones.

Run after the generators (they overwrite the fixtures in the working tree).
Numbers may differ by floating-point noise between machines (BLAS / CPU
paths), so they are compared with a tolerance; any structural change (keys,
lengths, strings, booleans) or a number beyond the tolerance is drift.

Fields listed in drift_ignore.txt ("file:key", any depth) are skipped:
they are known to differ between machines (e.g. sklearn's unstable tie
breaking).

Usage: check_fixture_drift.py [--rtol 1e-9] [--atol 1e-12] DIR...
Exit status 1 if any fixture drifted.
"""

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path


def committed(path):
    out = subprocess.run(["git", "show", f"HEAD:{path}"], capture_output=True, text=True)
    return json.loads(out.stdout) if out.returncode == 0 else None


IGNORE_FILE = Path(__file__).with_name("drift_ignore.txt")


def load_ignored():
    """{file name: {keys}} from drift_ignore.txt."""
    ignored = {}
    if IGNORE_FILE.exists():
        for line in IGNORE_FILE.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                name, key = line.split(":", 1)
                ignored.setdefault(name.strip(), set()).add(key.strip())
    return ignored


def compare(a, b, path, rtol, atol, issues, noise, skip=frozenset()):
    if isinstance(a, bool) or isinstance(b, bool) or not (
        isinstance(a, (int, float)) and isinstance(b, (int, float))
    ):
        if type(a) is not type(b):
            issues.append(f"{path}: type {type(a).__name__} -> {type(b).__name__}")
        elif isinstance(a, dict):
            for k in sorted(a.keys() | b.keys()):
                if k in skip:
                    continue
                if k not in a or k not in b:
                    issues.append(f"{path}/{k}: {'added' if k not in a else 'removed'}")
                else:
                    compare(a[k], b[k], f"{path}/{k}", rtol, atol, issues, noise, skip)
        elif isinstance(a, list):
            if len(a) != len(b):
                issues.append(f"{path}: length {len(a)} -> {len(b)}")
            else:
                for i, (x, y) in enumerate(zip(a, b)):
                    compare(x, y, f"{path}[{i}]", rtol, atol, issues, noise, skip)
        elif a != b:
            issues.append(f"{path}: {a!r} -> {b!r}")
        return
    if a == b:
        return
    if math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
        noise.append(abs(a - b))
    else:
        issues.append(f"{path}: {a!r} -> {b!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-12)
    ap.add_argument("dirs", nargs="+")
    args = ap.parse_args()
    ignored = load_ignored()
    drifted = 0
    for d in args.dirs:
        for f in sorted(Path(d).glob("*.json")):
            old = committed(str(f))
            if old is None:
                print(f"NEW      {f} (not committed)")
                drifted += 1
                continue
            issues, noise = [], []
            skip = frozenset(ignored.get(f.name, ()))
            compare(old, json.loads(f.read_text()), "", args.rtol, args.atol, issues, noise, skip)
            if issues:
                drifted += 1
                print(f"DRIFT    {f}: {len(issues)} difference(s)")
                for line in issues[:10]:
                    print(f"           {line}")
            elif noise:
                print(f"noise    {f}: {len(noise)} number(s), max |diff| {max(noise):.3g}")
    print(f"{drifted} fixture(s) drifted" if drifted else "no drift")
    return 1 if drifted else 0


if __name__ == "__main__":
    sys.exit(main())
