# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib"]
# ///
"""Draw the two-stage pipeline diagram: e-means micro-clusters, then weighted DBSCAN."""
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

steps = [("Input vectors", "n rows, d columns"),
         ("e-means", "k micro-clusters\n(k much larger than\nthe real cluster count)"),
         ("Count vectors\nper centroid", "weight w_j"),
         ("Weighted DBSCAN", "on k centroids only\n(eps, min_samples)"),
         ("Label each vector", "label of its centroid\n(-1 = noise)")]
fig, ax = plt.subplots(figsize=(11, 2.4))
ax.set_xlim(0, len(steps) * 2.2); ax.set_ylim(0, 2.4); ax.axis("off")
for i, (title, sub) in enumerate(steps):
    x = i * 2.2 + 0.1
    ax.add_patch(FancyBboxPatch((x, 0.3), 1.8, 1.8, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc="#e7f5ff" if i in (1, 3) else "#f1f3f5", ec="#495057", lw=1.2))
    ax.text(x + 0.9, 1.6, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x + 0.9, 0.85, sub, ha="center", va="center", fontsize=8.5, color="#343a40")
    if i < len(steps) - 1:
        ax.annotate("", xy=(x + 2.18, 1.2), xytext=(x + 1.92, 1.2),
                    arrowprops=dict(arrowstyle="-|>", color="#495057", lw=1.4))
fig.tight_layout()
out = pathlib.Path(__file__).parent / "img" / "pipeline.png"
fig.savefig(out, dpi=80, bbox_inches="tight")
