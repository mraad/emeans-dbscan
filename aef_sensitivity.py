"""Best settings of the three Cairo jobs, three seeds each, with GPU e-means on the test window.

Needs data/cairo/ and a CUDA GPU with torch. Writes out/<host><tag>/results.json.
The recorded run of this script is out/sensitivity_gpu.json.
"""
import argparse
import json
import time

import aef_cluster as A

# Best Optuna settings from out/*/results.json: k, eps, min_pts (tuned on 1M vectors)
SETTINGS = {"mac_cpu": (199, 0.034556626617712874, 89), "xeon_gpu": (823, 0.023896584382321364, 10),
            "xeon_cpu": (7393, 0.054923323551137035, 29600)}

p = argparse.ArgumentParser()
p.add_argument("--tag", required=True)
out = A.out_dir(p.parse_args().tag)
x, y, _ = A.load("test")
gpu = A.GPU()
gpu.train(x[:200_000], 256, 0, 1)                          # warm-up: compile kernels
gpu.put(x)
rows = []
for name, (k, eps, mp) in SETTINGS.items():
    for seed in (1234, 4321, 7):
        t = time.perf_counter()
        lab, tm = A.combo(x, k, eps, A.scale_min_pts(mp, len(x), 1_000_000, 1), gpu.train, seed=seed)
        sc = A.scores(y, lab)
        rows.append({"settings": name, "k": k, "eps": eps, "min_pts": mp, "seed": seed, "seconds": time.perf_counter() - t, **tm, **sc})
        (out / "results.json").write_text(json.dumps({"versions": A.versions(), "runs": rows}, indent=1))
        print(name, seed, {m: round(sc[m], 3) for m in ("land_ari", "land_purity", "clusters", "noise_share")}, round(tm["emeans_s"], 2), flush=True)
