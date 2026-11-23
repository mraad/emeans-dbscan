"""GPU vs CPU timing of e-means and Lloyd k-means on a synthetic Gaussian mixture.

Data: d=128, 4,096 Gaussian clusters with uneven sizes and scales. Every method
fits k=4,096 centroids to the same rows. Score: mean squared distance from each
held-out row to its nearest centroid (exact faiss.knn on the CPU).

Timing covers training only. GPU data is on the GPU before the clock starts.
Each GPU configuration gets one untimed warm-up run (kernel compile).
Results go to a JSON file after every run, so a stopped job keeps its data.
"""
import argparse
import json
import platform
import statistics
import time

import faiss
import numpy as np
import torch
from faiss.contrib.e_means import Emeans, NumpyEmeans

D, K, N_TEST, SEED = 128, 4096, 100_000, 1234


def mixture(n_max):
    rng = np.random.default_rng(0)
    centers = rng.standard_normal((K, D)).astype("float32")
    scale = rng.lognormal(-1.0, 0.4, K).astype("float32")         # cluster spread
    weight = rng.dirichlet(np.full(K, 2.0))                        # uneven cluster sizes

    def draw(n, r):
        lab = r.choice(K, n, p=weight)
        return centers[lab] + scale[lab, None] * r.standard_normal((n, D), dtype="float32")

    return draw(n_max, rng), draw(N_TEST, np.random.default_rng(1))


def mse(C, xte):
    dist, _ = faiss.knn(xte, np.ascontiguousarray(C, dtype="float32"), 1)
    return float(dist.mean())


def cpu_lloyd(x):
    km = faiss.Kmeans(D, K, niter=25, seed=SEED, max_points_per_centroid=1 << 30)
    km.train(x)
    return km.centroids


def cpu_emeans(x, T):
    return NumpyEmeans(D, K, nepochs=T, seed=SEED).train(x)[0], None


def gpu_emeans(xg, T):
    em = Emeans(D, K, nepochs=T, seed=SEED)
    C, _ = em.train(xg)
    return C.float().cpu().numpy(), em.recipe.get("assignment_backend")


def gpu_lloyd(xg, iters=25, chunk=65_536):
    """Plain PyTorch Lloyd in FP32: exact argmin by chunked matmul, then means."""
    g = torch.Generator(device="cpu").manual_seed(SEED)
    C = xg[torch.randperm(len(xg), generator=g)[:K].to(xg.device)].clone()
    for _ in range(iters):
        sums = torch.zeros_like(C)
        cnt = torch.zeros(K, device=xg.device)
        cn = (C * C).sum(1)
        for i in range(0, len(xg), chunk):
            xb = xg[i:i + chunk]
            lab = (cn - 2 * xb @ C.T).argmin(1)
            sums.index_add_(0, lab, xb)
            cnt.index_add_(0, lab, torch.ones_like(lab, dtype=cnt.dtype))
        keep = cnt > 0
        C[keep] = sums[keep] / cnt[keep, None]
    return C.cpu().numpy()


def timed(f, gpu):
    if gpu:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = f()
    if gpu:
        torch.cuda.synchronize()
    return out, time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, nargs="+", default=[250_000, 1_000_000, 4_000_000])
    p.add_argument("--epochs", type=float, nargs="+", default=[1, 2, 3, 5])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--lloyd-repeats-big", type=int, default=1, help="CPU Lloyd repeats when n > 1M")
    p.add_argument("--out", default="bench_results.json")
    a = p.parse_args()

    xall, xte = mixture(max(a.n))
    res = {"config": {"d": D, "k": K, "n_test": N_TEST, "seed": SEED, "epochs": a.epochs,
                      "repeats": a.repeats, "cpu": platform.processor() or platform.machine(),
                      "threads": faiss.omp_get_max_threads(), "gpu": torch.cuda.get_device_name(0),
                      "torch": torch.__version__, "faiss": faiss.__version__}, "runs": []}

    def record(**kw):
        res["runs"].append(kw)
        print(json.dumps(kw), flush=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)

    for n in a.n:
        x = np.ascontiguousarray(xall[:n])
        xg32 = torch.from_numpy(x).cuda()
        xg16 = xg32.to(torch.bfloat16)
        reps = lambda big: a.lloyd_repeats_big if (big and n > 1_000_000) else a.repeats

        def run(method, f, gpu, T=None, r=a.repeats):
            ts, out = [], None
            for _ in range(r):
                out, t = timed(f, gpu)
                ts.append(t)
            C, backend = out if isinstance(out, tuple) else (out, None)
            record(n=n, method=method, epochs=T, seconds=statistics.median(ts), times=ts,
                   test_mse=mse(C, xte), backend=backend)

        run("cpu_lloyd", lambda: cpu_lloyd(x), False, 25, reps(True))
        gpu_lloyd(xg32, iters=1)                                   # warm-up
        run("gpu_lloyd_torch_fp32", lambda: gpu_lloyd(xg32), True, 25)
        for T in a.epochs:
            run("cpu_emeans", lambda: cpu_emeans(x, T), False, T)
        for name, xg in [("gpu_emeans_fp32_input", xg32), ("gpu_emeans_bf16", xg16)]:
            gpu_emeans(xg, a.epochs[0])                            # warm-up: compile kernels
            for T in a.epochs:
                run(name, lambda: gpu_emeans(xg, T), True, T)
        del xg32, xg16
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
