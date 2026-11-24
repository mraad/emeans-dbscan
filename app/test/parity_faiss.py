# /// script
# requires-python = ">=3.10"
# dependencies = ["faiss-cpu==1.16.0", "numpy", "scikit-learn"]
# ///
"""Reference results from faiss for parity.test.mjs. Run after prepare_data.py. Writes test/reference.json."""
import json
import pathlib

import faiss
import numpy as np
from faiss.contrib.e_means import NumpyEmeans
from sklearn.metrics import adjusted_rand_score

HERE = pathlib.Path(__file__).parent
DATA = HERE.parent / "data"
N = 100_000
out = {"n": N, "faiss": faiss.__version__, "runs": []}
emb = np.fromfile(DATA / "test_emb.bin", np.int8).reshape(-1, 64)
wc = np.fromfile(DATA / "test_wc.bin", np.uint8)
ok = (emb[:, 0] != -128) & (wc > 0)
x = emb[ok].astype(np.float32) / 127.5
x *= np.abs(x)
x /= np.linalg.norm(x, axis=1, keepdims=True)
x, y = np.ascontiguousarray(x[:N]), wc[ok][:N]
land = y != 80
for k, seed, T in [(3, 1234, 5), (3, 1, 2), (8, 7, 5), (64, 1234, 5), (256, 3, 3)]:
    em = NumpyEmeans(64, k, nepochs=T, seed=seed, spherical=True)
    em.train(x)
    sim, lab = em.assign(x)
    out["runs"].append({"k": k, "seed": seed, "nepochs": T, "batch": em.recipe["batch_size"], "alpha0": em.recipe["alpha0"],
                        "n_splits": em.n_splits, "mean_cosine": float(sim.mean(dtype="float64")), "labels": lab.ravel().tolist(),
                        "land_ari": float(adjusted_rand_score(y[land], lab.ravel()[land]))})
(HERE / "reference.json").write_text(json.dumps(out))
print([(r["k"], r["seed"], r["batch"], r["n_splits"], round(r["mean_cosine"], 5)) for r in out["runs"]])
