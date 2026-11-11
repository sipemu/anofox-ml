import os, struct, subprocess, sys

import numpy as np
rng = np.random.default_rng(7)
def hx(a): return " ".join("%08x" % struct.unpack("<I", struct.pack("<f", v))[0] for v in np.asarray(a, np.float32).ravel())
def run(n, A, B):
    inp = f"{n}\n{hx(A.T)}\n{hx(B.T)}\n"
    out = subprocess.run([os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")], input=inp, capture_output=True, text=True, check=True).stdout.split("\n")
    return out[0].strip(), out[1].strip(), out[2].strip()
def sym(M): M = np.atleast_2d(np.asarray(M, np.float32)); return np.triu(M) + np.triu(M, 1).T
cases = []
for n in [1, 2, 3, 4, 5, 6, 9, 16, 24]:
    for kind in ["lda", "lda3", "spd"]:
        k = 2 if kind == "lda" else 4
        C = rng.normal(size=(k, n)).astype(np.float32)
        w = rng.dirichlet(np.ones(k)).astype(np.float32)
        m = (w[:, None] * C).sum(0)
        A = sym((C.T * w) @ C - np.outer(m, m)) if kind != "spd" else sym(np.cov(rng.normal(size=(n, 3 * n + 2))))
        X = rng.normal(size=(5 * n + 3, n)).astype(np.float32)
        B = sym(X.T @ X / len(X) + 5e-5 * np.eye(n))
        cases.append((n, A, B))
cases.append((5, np.zeros((5, 5), np.float32), sym(np.eye(5) * 2)))
cases.append((4, sym(np.diag([3, 1, 2, 1])), sym(np.eye(4))))
cases.append((3, sym(np.ones((3, 3)) * 1e-30), sym(np.eye(3))))
cases.append((3, sym(np.ones((3, 3)) * 1e30), sym(np.eye(3))))
with open(sys.argv[1], "w") as f:
    for n, A, B in cases:
        a, wv, info = run(n, A, B)
        f.write(f"{n}\n{hx(A.T)}\n{hx(B.T)}\n{a}\n{wv}\n{info}\n")
print(len(cases))
