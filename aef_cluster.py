# /// script
# requires-python = ">=3.12"
# dependencies = ["faiss-cpu==1.16.0", "numpy==2.5.5", "scikit-learn==1.9.1", "optuna==5.1.0"]
# ///
"""e-means + weighted DBSCAN + Optuna on AlphaEarth embeddings around Cairo, scored against WorldCover.

Needs data/cairo/ from aef_fetch.py. Writes out/<host>/results.json and out/<host>/test_maps.npz.
Results are written after every stage, so a stopped run keeps what it finished.
"""
import argparse
import json
import os
import pathlib
import platform
import socket
import time

import faiss
import numpy as np
import optuna
import sklearn
from faiss.contrib.e_means import NumpyEmeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics.cluster import contingency_matrix

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "cairo"
D = 64


# ---------- data ----------
def load(name, crop=None):
    """Unit-length float32 vectors of valid pixels, their WorldCover labels, and the pixel mask."""
    emb = np.load(DATA / f"{name}_emb.npy", mmap_mode="r")[:crop, :crop]
    wc = np.load(DATA / f"{name}_wc.npy")[:crop, :crop]
    valid = (emb[..., 0] != -128) & (wc > 0)
    idx = np.flatnonzero(valid.ravel())
    flat = emb.reshape(-1, D)                              # a view of the memory-mapped file, or a copy when cropped
    x = np.empty((len(idx), D), np.float32)
    for i in range(0, len(idx), 1 << 21):                  # chunked: the int8 file stays memory-mapped
        v = flat[idx[i:i + (1 << 21)]].astype(np.float32) / 127.5
        v *= np.abs(v)                                     # AEF dequantization: (v/127.5)^2 * sign(v)
        v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        x[i:i + len(v)] = v
    return x, wc.ravel()[idx], valid


def purity(y, lab):
    c = contingency_matrix(y, lab, sparse=True)
    return float(c.max(axis=0).sum() / len(y))


WATER = 80  # WorldCover "permanent water bodies"; excluded from the land scores


def scores(y, lab):
    """Scores on all pixels, and on land pixels only (the sea is an easy, dominant class)."""
    land = y != WATER
    if not land.any():                                     # e.g. a smoke-test crop of open sea
        land = np.ones_like(land)
    return {"ari": float(ari(y, lab)), "nmi": float(nmi(y, lab)), "purity": purity(y, lab),
            "land_ari": float(ari(y[land], lab[land])), "land_nmi": float(nmi(y[land], lab[land])),
            "land_purity": purity(y[land], lab[land]),
            "clusters": int(len(set(np.unique(lab)) - {-1})), "noise_share": float(np.mean(lab == -1))}


# ---------- e-means backends ----------
def train_cpu(x, k, seed, nepochs=5):
    C, lab = NumpyEmeans(D, k, nepochs=nepochs, seed=seed, spherical=True).train(x)
    return np.asarray(C), np.asarray(lab).ravel()


class GPU:
    def __init__(self, dtype="bfloat16"):
        import torch
        self.torch = torch
        self.dtype = getattr(torch, dtype)
        self.cache = {}

    def put(self, x):                                      # data on the GPU before any clock starts
        key = (x.__array_interface__["data"][0], x.shape)
        if key not in self.cache:
            self.cache.clear()
            self.cache[key] = self.torch.from_numpy(x).cuda().to(self.dtype)
        return self.cache[key]

    def train(self, x, k, seed, nepochs=5):
        from faiss.contrib.e_means import Emeans
        xg = self.put(x)
        self.torch.cuda.synchronize()
        em = Emeans(D, k, nepochs=nepochs, seed=seed, spherical=True)
        C, lab = em.train(xg)
        self.torch.cuda.synchronize()
        self.backend = em.recipe.get("assignment_backend")
        return C.float().cpu().numpy(), lab.cpu().numpy().ravel().astype(np.int64)


def out_dir(tag):
    """A new result folder. Fails when it exists, so that a rerun cannot overwrite recorded results."""
    d = HERE / "out" / (socket.gethostname().split(".")[0] + tag)
    try:
        d.mkdir(parents=True)
    except FileExistsError:
        raise SystemExit(f"{d} exists. Use another --tag, or move the old results first.") from None
    return d


def versions():
    return {"python": platform.python_version(), "faiss": faiss.__version__, "numpy": np.__version__,
            "scikit-learn": sklearn.__version__, "optuna": optuna.__version__}


def scale_min_pts(min_pts, n, n_tuned, least):
    """min_pts counts vectors: a value tuned on n_tuned vectors must grow with n."""
    return max(least, round(min_pts * n / n_tuned))


def mean_neighbors(q, x, eps, chunk=20_000):
    """Mean number of rows of x within cosine distance eps of the rows of q (unit vectors). Exact, no cap."""
    return sum(int((q @ x[i:i + chunk].T >= 1 - eps).sum()) for i in range(0, len(x), chunk)) / len(q)


def to_map(lab, valid, dtype=np.int32):
    """Labels on the pixel grid, every 4th row and column; -2 is no data."""
    m = np.full(valid.size, -2, dtype)
    m[np.flatnonzero(valid.ravel())] = lab
    return m.reshape(valid.shape)[::4, ::4]


def timed(f):
    t0 = time.perf_counter()
    out = f()
    return out, time.perf_counter() - t0


# ---------- the combined method ----------
def combo(x, k, eps, min_pts, train, seed=1234):
    (C, lab), t_em = timed(lambda: train(x, k, seed))
    w = np.bincount(lab[lab >= 0], minlength=k)
    keep = w > 0
    Cn = C[keep] / np.maximum(np.linalg.norm(C[keep], axis=1, keepdims=True), 1e-12)
    db, t_db = timed(lambda: DBSCAN(eps=eps, min_samples=min_pts, metric="cosine", algorithm="brute")
                     .fit(Cn, sample_weight=w[keep]).labels_)
    lab_c = np.full(k, -1)
    lab_c[keep] = db
    out = np.where(lab >= 0, lab_c[np.maximum(lab, 0)], -1)
    return out, {"emeans_s": t_em, "dbscan_s": t_db}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=60)
    p.add_argument("--tune-n", type=int, default=1_000_000)
    p.add_argument("--dbscan-n", type=int, default=20_000)
    p.add_argument("--tune-backend", choices=["cpu", "gpu"], default="cpu")
    p.add_argument("--final-backends", nargs="+", default=["cpu"])
    p.add_argument("--crop", type=int, default=None, help="smoke test: use only the top-left crop x crop pixels")
    p.add_argument("--tag", default="", help="suffix for the output folder")
    a = p.parse_args()

    out = out_dir(a.tag)
    gpu = GPU() if ("gpu" in a.final_backends or a.tune_backend == "gpu") else None
    trainers = {"cpu": train_cpu, **({"gpu": gpu.train} if gpu else {})}
    R = {"host": socket.gethostname().split(".")[0], "machine": platform.machine(), "processor": platform.processor(),
         "threads": faiss.omp_get_max_threads(), "versions": versions(), "args": vars(a)}
    if gpu:
        R["gpu"] = gpu.torch.cuda.get_device_name(0)
        R["torch"] = gpu.torch.__version__

    def save():
        (out / "results.json").write_text(json.dumps(R, indent=1))

    def log(msg):
        print(time.strftime("%H:%M:%S"), msg, flush=True)

    (xt, yt, _), t = timed(lambda: load("tune", a.crop))
    log(f"tune window: {len(xt):,} valid vectors, load {t:.0f}s")
    rng = np.random.default_rng(0)
    s = rng.choice(len(xt), min(a.tune_n, len(xt)), replace=False)
    xs, ys = np.ascontiguousarray(xt[s]), yt[s]
    R["tune"] = {"valid": int(len(xt)), "subsample": int(len(s)),
                 "classes": {int(c): int(n) for c, n in zip(*np.unique(yt, return_counts=True))}}

    # 1. Optuna: combined method on the tune subsample
    train = trainers[a.tune_backend]
    if a.tune_backend == "gpu":
        train(xs[:200_000], 256, 0, 1)                    # warm-up: compile kernels
    trials = []

    def objective(trial):
        k = trial.suggest_int("k", 64, 8192, log=True)
        eps = trial.suggest_float("eps", 0.002, 0.3, log=True)
        mp = trial.suggest_int("min_pts", 10, 100_000, log=True)
        (lab, tm), t = timed(lambda: combo(xs, k, eps, mp, train))
        sc = scores(ys, lab)
        trials.append({"params": trial.params, **sc, **tm, "seconds": t})
        return sc["land_ari"]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    _, t = timed(lambda: st.optimize(objective, n_trials=a.trials))
    best = st.best_params
    R["optuna_combo"] = {"backend": a.tune_backend, "seconds": t, "best_params": best,
                         "best_land_ari": st.best_value, "trials": trials}
    save()
    log(f"Optuna combo: {t:.0f}s, best {best}, land ARI {st.best_value:.4f}")

    # 2. Plain DBSCAN, same Optuna budget, on a small subsample (it does not scale to 1M)
    sd = s[: a.dbscan_n]
    xd, yd = np.ascontiguousarray(xt[sd]), yt[sd]

    def obj_db(trial):
        eps = trial.suggest_float("eps", 0.002, 0.3, log=True)
        mp = trial.suggest_int("min_pts", 2, 500, log=True)
        lab = DBSCAN(eps=eps, min_samples=mp, metric="cosine", algorithm="brute").fit_predict(xd)
        return scores(yd, lab)["land_ari"]

    sdb = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    _, t = timed(lambda: sdb.optimize(obj_db, n_trials=a.trials))
    R["optuna_dbscan"] = {"seconds": t, "subsample": int(len(sd)), "best_params": sdb.best_params,
                          "best_land_ari": sdb.best_value}
    save()
    log(f"Optuna plain DBSCAN: {t:.0f}s, best {sdb.best_params}, land ARI {sdb.best_value:.4f}")
    del xt, yt, xs, ys, xd, yd

    # 3. Test window, all valid pixels
    (x, y, valid), t = timed(lambda: load("test", a.crop))
    log(f"test window: {len(x):,} valid vectors, load {t:.0f}s")
    classes = np.unique(y)
    R["test"] = {"valid": int(len(x)), "load_s": t,
                 "classes": {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}, "final": {}}
    maps = {"valid": valid[::4, ::4], "wc": np.load(DATA / "test_wc.npy")[:a.crop, :a.crop][::4, ::4]}

    full_labels = {}
    mp_full = scale_min_pts(best["min_pts"], len(x), len(s), 1)
    for b in a.final_backends:
        tr = trainers[b]
        if b == "gpu":
            assert gpu is not None
            tr(x[:200_000], 256, 0, 1)                    # warm-up
            gpu.put(x)
        (lab, tm), t = timed(lambda: combo(x, best["k"], best["eps"], mp_full, tr))
        R["test"]["final"][b] = {**scores(y, lab), **tm, "total_s": t, "min_pts_scaled": mp_full,
                                 **({"assignment_backend": gpu.backend} if b == "gpu" else {})}
        maps[f"combo_{b}"] = to_map(lab, valid)
        full_labels[b] = lab
        save()
        log(f"final {b}: {R['test']['final'][b]}")

        (C, lab_k), t = timed(lambda: tr(x, len(classes), 1234))
        R["test"]["final"][f"emeans_k{len(classes)}_{b}"] = {**scores(y, lab_k), "emeans_s": t}
        maps[f"emeans_k_{b}"] = to_map(lab_k, valid)
        # same number of clusters as the combined method: purity is only comparable at equal cluster count
        k_same = max(2, R["test"]["final"][b]["clusters"])
        (C, lab_s), t = timed(lambda: tr(x, k_same, 1234))
        R["test"]["final"][f"emeans_same_k{k_same}_{b}"] = {**scores(y, lab_s), "emeans_s": t}
        save()

    # 4. Label-free check: same settings, another seed
    b = a.final_backends[-1]
    lab2, _ = combo(x, best["k"], best["eps"], mp_full, trainers[b], seed=4321)
    R["test"]["seed_stability_ari"] = {"backend": b, "ari": float(ari(full_labels[b], lab2))}
    save()

    # 5. Plain DBSCAN on test subsamples: score, and time while memory allows
    bp = sdb.best_params
    sub = np.random.default_rng(1).permutation(len(x))
    R["test"]["plain_dbscan"] = []
    mem_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    for n in [10_000, 20_000, 40_000, 80_000, 160_000, 320_000]:
        if n > len(x):
            break
        ii = sub[:n]
        if n > 10_000:
            last = R["test"]["plain_dbscan"][-1]
            est_gb = last["mean_neighbors"] * (n / last["n"]) * n * 12 / 1e9
            if est_gb > mem_gb / 2 or last["seconds"] * (n / last["n"]) ** 2 > 900:
                R["test"]["plain_dbscan"].append({"n": n, "skipped": f"estimated {est_gb:.1f} GB (limit {mem_gb / 2:.1f}), or more than 900 s"})
                break
        ms = scale_min_pts(bp["min_pts"], n, len(sd), 2)
        db = DBSCAN(eps=bp["eps"], min_samples=ms, metric="cosine", algorithm="brute")
        lab, t = timed(lambda: db.fit_predict(x[ii]))
        mean_nb = mean_neighbors(x[ii[:2000]], x[ii], bp["eps"])
        R["test"]["plain_dbscan"].append({"n": n, "min_samples": ms, "seconds": t, "mean_neighbors": mean_nb, **scores(y[ii], lab)})
        save()
        log(f"plain DBSCAN n={n}: {t:.1f}s ARI {R['test']['plain_dbscan'][-1]['ari']:.4f}")

    # the combined method, scored on the same 20k pixels that plain DBSCAN saw
    ii = sub[:min(20_000, len(x))]
    R["test"]["combo_on_20k"] = {b: scores(y[ii], lab[ii]) for b, lab in full_labels.items()}
    np.savez_compressed(out / "test_maps.npz", **maps)
    save()
    log("done")


if __name__ == "__main__":
    main()
