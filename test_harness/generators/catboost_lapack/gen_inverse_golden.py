"""Golden bits of catboost's InverseMatrix (sgetrf + sgetri) on symmetric
positive definite and general matrices; see README.md."""
import os, struct, subprocess, sys

import numpy as np

HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
rng = np.random.default_rng(11)


def hx(a):
    return " ".join("%08x" % struct.unpack("<I", struct.pack("<f", v))[0]
                    for v in np.asarray(a, np.float32).ravel())


lines = []
for n in [1, 2, 3, 4, 5, 6, 9, 16, 24]:
    for kind in ["spd", "general"]:
        X = rng.normal(size=(3 * n + 2, n)).astype(np.float32)
        A = X.T @ X / len(X) + 5e-5 * np.eye(n) if kind == "spd" else rng.normal(size=(n, n))
        A = np.asarray(A, np.float32)
        out = subprocess.run([HARNESS, "inv"], input=f"{n}\n{hx(A.T)}\n", capture_output=True,
                             text=True, check=True).stdout.strip()
        lines += [str(n), hx(A.T), out]
with open(sys.argv[1], "w") as f:
    f.write("\n".join(lines) + "\n")
print(len(lines) // 3)
