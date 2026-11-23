# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu==1.16.0", "scikit-learn==1.9.1", "optuna==5.1.0", "matplotlib"]
# ///
"""e-means micro-clusters + weighted DBSCAN on two moons, tuned with Optuna.

Tunes on one two-moons sample (seed 0) and scores the best settings on a
second sample (seed 1). Plain DBSCAN gets the same Optuna budget as a baseline.
Writes results.json and img/moons.jpg next to this file.
"""
import json
import pathlib
import time

import numpy as np
import optuna
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
from sklearn.metrics import adjusted_rand_score as ari

N, NOISE, TRIALS = 20_000, 0.08, 150
HERE = pathlib.Path(__file__).parent


def data(seed):
    x, y = make_moons(N, noise=NOISE, random_state=seed)
    return x.astype("float32"), y


def emeans_dbscan(x, k, eps, min_pts, seed=1234):
    """Stage 1: k e-means micro-clusters. Stage 2: DBSCAN on centroids weighted by size."""
    em = NumpyEmeans(x.shape[1], k, seed=seed)
    C, _ = em.train(x)
    _, micro = em.assign(x)                       # exact nearest final centroid
    micro = micro.ravel()
    w = np.bincount(micro, minlength=k)
    keep = w > 0                                  # empty cells carry no points
    lab_c = np.full(k, -1)
    lab_c[keep] = DBSCAN(eps=eps, min_samples=min_pts).fit(C[keep], sample_weight=w[keep]).labels_
    return lab_c[micro], C


def tune(objective):
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=TRIALS)
    vals = np.array([t.value for t in study.trials])
    return study, time.perf_counter() - t0, float(np.mean(vals >= 0.99))


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


xtr, ytr = data(0)
xte, yte = data(1)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def obj_combo(t):
    labels, _ = emeans_dbscan(xtr, t.suggest_int("k", 8, 1024, log=True),
                              t.suggest_float("eps", 0.01, 0.5, log=True),
                              t.suggest_int("min_pts", 1, 2000, log=True))
    return ari(ytr, labels)


def obj_dbscan(t):
    labels = DBSCAN(eps=t.suggest_float("eps", 0.01, 0.5, log=True),
                    min_samples=t.suggest_int("min_pts", 1, 2000, log=True)).fit_predict(xtr)
    return ari(ytr, labels)


res = {"n": N, "noise": NOISE, "trials": TRIALS}
for name, obj in [("emeans_dbscan", obj_combo), ("dbscan", obj_dbscan)]:
    study, secs, frac = tune(obj)
    res[name] = {"best_params": study.best_params, "train_ari": study.best_value,
                 "tuning_seconds": secs, "share_trials_ari_ge_0.99": frac}

bp = res["emeans_dbscan"]["best_params"]
(lab, C), t = timed(lambda: emeans_dbscan(xte, bp["k"], bp["eps"], bp["min_pts"]))
res["emeans_dbscan"].update(test_ari=ari(yte, lab), test_seconds=t, clusters=len(set(lab) - {-1}),
                            noise_share=float(np.mean(lab == -1)))

bd = res["dbscan"]["best_params"]
db, t = timed(lambda: DBSCAN(eps=bd["eps"], min_samples=bd["min_pts"]).fit_predict(xte))
res["dbscan"].update(test_ari=ari(yte, db), test_seconds=t, clusters=len(set(db) - {-1}),
                     noise_share=float(np.mean(db == -1)))

(_, km), t = timed(lambda: NumpyEmeans(2, 2).train(xte))
res["emeans_k2"] = {"test_ari": ari(yte, km.ravel()), "test_seconds": t}

(HERE / "results.json").write_text(json.dumps(res, indent=2) + "\n")
print(json.dumps(res, indent=2))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = np.array(["#3b5bdb", "#e03131", "#2f9e44", "#f08c00", "#7048e8"])


def colors(labels):
    """Fixed colours per cluster id; noise (-1) in grey, so the panels compare at a glance."""
    labels = np.asarray(labels)
    out = PALETTE[labels % len(PALETTE)].astype(object)
    out[labels == -1] = "#adb5bd"
    return out


fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
ax[0].scatter(*xte.T, c=colors(km.ravel()), s=0.5); ax[0].set_title("e-means, k = 2")
ax[1].scatter(*xte.T, c=colors(lab), s=0.5); ax[1].scatter(*C.T, c="k", s=5)
ax[1].set_title(f"e-means k = {bp['k']} + DBSCAN (centroids in black)")
ax[2].scatter(*xte.T, c=colors(db), s=0.5); ax[2].set_title("plain DBSCAN, tuned (noise in grey)")
for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
(HERE / "img").mkdir(exist_ok=True)
fig.savefig(HERE / "img" / "moons.jpg", dpi=90, bbox_inches="tight", pil_kwargs={"quality": 70, "optimize": True})
