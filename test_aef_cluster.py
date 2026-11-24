# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu==1.16.0", "numpy==2.5.5", "scikit-learn==1.9.1", "optuna==5.1.0"]
# ///
"""Tests for aef_cluster.py that need no data. Run: uv run --python 3.12 test_aef_cluster.py"""
import pathlib
import tempfile

import numpy as np

import aef_cluster as A


def blobs(sizes, spread=0.02, seed=0):
    """Unit vectors around random directions; returns the vectors and the true blob of each."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((len(sizes), A.D))
    y = np.repeat(np.arange(len(sizes)), sizes)
    x = centers[y] / np.linalg.norm(centers[y], axis=1, keepdims=True) + spread * rng.standard_normal((len(y), A.D))
    order = rng.permutation(len(y))
    x = (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)
    return np.ascontiguousarray(x[order]), y[order]


# min_pts grows with n, and never goes below the limit
assert A.scale_min_pts(10, 16_777_216, 1_000_000, 1) == 168
assert A.scale_min_pts(66, 320_000, 20_000, 2) == 1056
assert A.scale_min_pts(2, 10, 1000, 2) == 2

# The combined method finds the blobs; the small blob (1%) is noise, at the tuned n and at 10 times n.
for factor in (1, 10):
    x, y = blobs([900 * factor, 900 * factor, 20 * factor])
    mp = A.scale_min_pts(50, len(x), 1820, 1)
    lab, _ = A.combo(x, 16, 0.2, mp, A.train_cpu)
    big = y < 2
    assert A.ari(y[big], lab[big]) == 1 and (lab[big] >= 0).all(), factor
    assert (lab[~big] == -1).all(), factor
    # without the scaling the small blob becomes a cluster at 10 times n: this is the bug that happened
    if factor == 10:
        lab, _ = A.combo(x, 16, 0.2, 50, A.train_cpu)
        assert (lab[~big] >= 0).all()

# The neighbour count is exact: it has no cap, and the chunks do not change it
x, _ = blobs([5000], spread=0.01)
assert A.mean_neighbors(x[:50], x, 0.5, chunk=700) == 5000
brute = ((1 - x[:50] @ x.T) <= 0.004).sum(1).mean()
assert abs(A.mean_neighbors(x[:50], x, 0.004) - brute) < 0.5

# A result folder is never reused
with tempfile.TemporaryDirectory() as d:
    A.HERE = pathlib.Path(d)
    A.out_dir("_t")
    try:
        A.out_dir("_t")
        raise AssertionError("out_dir reused a folder")
    except SystemExit:
        pass
print("ok")
