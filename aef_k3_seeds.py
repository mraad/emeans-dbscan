# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu==1.16.0", "numpy==2.5.5", "scikit-learn==1.9.1", "optuna==5.1.0"]
# ///
"""Plain e-means k = 3, ten seeds, on the Cairo test window: score and exact objective for every local optimum.

Objective: mean cosine similarity of each vector to its nearest final centroid (higher is better).
Writes out/<host><tag>/results.json.
"""
import argparse
import json

import numpy as np

import aef_cluster as A

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["cpu", "gpu"], required=True)
p.add_argument("--fp32", action="store_true", help="GPU only: FP32 input instead of BF16")
p.add_argument("--tag", required=True)
a = p.parse_args()

out = A.out_dir(a.tag)
x, y, _ = A.load("test")
if a.backend == "gpu":
    gpu = A.GPU("float32" if a.fp32 else "bfloat16")
    train = gpu.train
    train(x[:200_000], 256, 0, 1)
else:
    train = A.train_cpu


def objective(C):
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    return float(sum((x[i:i + 2_000_000] @ C.T).max(1).sum(dtype="float64") for i in range(0, len(x), 2_000_000)) / len(x))


rows = []
R = {"backend": a.backend, "fp32": a.fp32, "versions": A.versions(), "runs": rows}
for seed in range(1, 11):
    (C, lab), t = A.timed(lambda: train(x, 3, seed))
    rows.append({"seed": seed, "emeans_s": t, "mean_cosine": objective(C), **A.scores(y, lab)})
    (out / "results.json").write_text(json.dumps(R, indent=1))   # after every seed
    print(seed, round(t, 2), round(rows[-1]["mean_cosine"], 6), round(rows[-1]["land_ari"], 4), flush=True)
print("done", flush=True)
