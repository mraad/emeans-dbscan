# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Figures for the GPU vs CPU benchmark, from bench_results.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).parent
R = json.loads((HERE / "bench_results.json").read_text())
runs = R["runs"]
STYLE = {"cpu_lloyd": ("CPU Lloyd (faiss.Kmeans)", "#868e96", "s"),
         "gpu_lloyd_torch_fp32": ("GPU Lloyd (PyTorch FP32)", "#212529", "D"),
         "cpu_emeans": ("CPU e-means (NumpyEmeans)", "#1c7ed6", "o"),
         "gpu_emeans_bf16": ("GPU e-means (BF16, Triton)", "#e8590c", "o")}

# Figure 1: error against training time at n = 1M.
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for m, (label, color, marker) in STYLE.items():
    pts = sorted((r["seconds"], r["test_mse"], r["epochs"]) for r in runs if r["n"] == 1_000_000 and r["method"] == m)
    if not pts:
        continue
    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color, label=label)
    for t, e, T in pts:
        ax.annotate(f"{T:g}", (t, e), textcoords="offset points", xytext=(4, 4), fontsize=7.5, color=color)
ax.set_xscale("log"); ax.set_xlabel("training time, seconds (log scale)")
ax.set_ylabel("held-out error (MSE)"); ax.set_title("n = 1,000,000, d = 128, k = 4,096 (labels: epochs or iterations)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(HERE / "img" / "bench_error_time.png", dpi=85)

# Figure 2: training time against n, per method (Lloyd 25 iterations, e-means 5 epochs).
fig, ax = plt.subplots(figsize=(7.5, 4.2))
for m, (label, color, marker) in STYLE.items():
    want = 25 if "lloyd" in m else 5
    pts = sorted((r["n"], r["seconds"]) for r in runs if r["method"] == m and r["epochs"] == want)
    if pts:
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=marker, color=color,
                label=f"{label}, {want} {'iterations' if 'lloyd' in m else 'epochs'}")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("training vectors n (log scale)"); ax.set_ylabel("training time, seconds (log scale)")
ax.set_title("Training time against data size (d = 128, k = 4,096)")
ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8.5)
fig.tight_layout(); fig.savefig(HERE / "img" / "bench_time_n.png", dpi=85)
