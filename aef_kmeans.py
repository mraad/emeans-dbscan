# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu==1.16.0", "numpy==2.5.5", "scikit-learn==1.9.1", "optuna==5.1.0"]
# ///
"""Plain spherical e-means baseline on the Cairo windows (no DBSCAN).

k = 3 on the test window, three seeds. k = 3 was first seen in a test-window result, so the script also
selects k on the tune window only (k = 2..16, mean land ARI over three seeds) and scores that k on the test window.
Writes out/<host><tag>/results.json and, for k = 3, out/<host><tag>/test_map_k3.npy.
"""
import argparse
import json
import platform
import time

import faiss
import numpy as np
from sklearn.metrics.cluster import contingency_matrix

import aef_cluster as A

SEEDS = (1234, 4321, 7)

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["cpu", "gpu"], required=True)
p.add_argument("--tag", required=True)
a = p.parse_args()

out = A.out_dir(a.tag)
gpu = A.GPU() if a.backend == "gpu" else None
train = gpu.train if gpu else A.train_cpu
R = {"host": A.HOST, "machine": platform.machine(), "backend": a.backend, "threads": faiss.omp_get_max_threads(),
     "versions": A.versions(),
     "seeds": SEEDS, "nepochs": 5}


def run(x, y, k):
    rows = []
    for seed in SEEDS:
        (_, lab), t = A.timed(lambda: train(x, k, seed))
        rows.append({"k": k, "seed": seed, "emeans_s": t, **A.scores(y, lab)})
        if seed == SEEDS[0]:
            first = lab
    return rows, first


def save():                                                # after every k, so that a stopped run keeps its results
    (out / "results.json").write_text(json.dumps(R, indent=1))


def mean(rows, m):
    return float(np.mean([r[m] for r in rows]))


xt, yt, _ = A.load("tune")
s = np.random.default_rng(0).choice(len(xt), 1_000_000, replace=False)   # same subsample as aef_cluster.py
xs, ys = np.ascontiguousarray(xt[s]), yt[s]
del xt, yt
if gpu:
    train(xs[:200_000], 256, 0, 1)                         # warm-up: compile kernels
R["tune_sweep"] = []
for k in range(2, 17):
    rows, _ = run(xs, ys, k)
    R["tune_sweep"].append({"k": k, "mean_land_ari": mean(rows, "land_ari"), "runs": rows})
    save()
    print(time.strftime("%H:%M:%S"), f"tune k={k} land ARI {mean(rows, 'land_ari'):.4f}", flush=True)
best = max(R["tune_sweep"], key=lambda r: r["mean_land_ari"])["k"]
R["tune_selected_k"] = best
save()

x, y, valid = A.load("test")
if gpu:
    train(x[:200_000], 256, 0, 1)
    gpu.put(x)
R["test"] = {}
for k in sorted({2, 3, 8, best}):
    rows, first = run(x, y, k)
    R["test"][str(k)] = {"mean_land_ari": mean(rows, "land_ari"), "mean_emeans_s": mean(rows, "emeans_s"), "runs": rows}
    if k == 3:                                             # what the three clusters are (seed 1234)
        R["test"]["3"]["classes"] = [int(c) for c in np.unique(y)]
        R["test"]["3"]["contingency"] = contingency_matrix(y, first).tolist()
        np.save(out / "test_map_k3.npy", A.to_map(first, valid, np.int8))
    save()
    print(time.strftime("%H:%M:%S"), f"test k={k} land ARI {mean(rows, 'land_ari'):.4f} {mean(rows, 'emeans_s'):.2f}s", flush=True)
print("done", flush=True)
