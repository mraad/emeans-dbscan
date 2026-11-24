# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib", "numpy"]
# ///
"""Cairo map figure: WorldCover, two combined-method results, plain e-means k = 3. Writes img/cairo_map.jpg."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

HERE = pathlib.Path(__file__).parent
OUT = HERE / "out"
MAC, XEON, K3 = OUT / "mac_cpu", OUT / "xeon_cpu", OUT / "mac_k3_cpu"   # result folders: <host><tag>
NAMES = {10: "Tree", 20: "Shrub", 30: "Grass", 40: "Crop", 50: "Built-up", 60: "Bare", 80: "Water", 90: "Wetland"}
COL = {10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff", 50: "#fa0000", 60: "#b4b4b4", 80: "#0064c8", 90: "#0096a0"}
mac = np.load(MAC / "test_maps.npz")
xeon = np.load(XEON / "test_maps.npz")
k3 = np.load(K3 / "test_map_k3.npy")
wc, valid = mac["wc"], mac["valid"]


def n_clusters(run):
    return json.loads((run / "results.json").read_text())["test"]["final"]["cpu"]["clusters"]


def majority(lab):
    """Give every cluster the WorldCover class most of its pixels have; noise (-1) and invalid become 0 (white)."""
    out = np.zeros_like(wc)
    ok = valid & (lab >= 0)
    for c in np.unique(lab[ok]):
        m = ok & (lab == c)
        v, n = np.unique(wc[m], return_counts=True)
        out[m] = v[np.argmax(n)]
    return out


codes = [0] + [c for c in COL if (wc == c).any()]
cmap = ListedColormap(["#ffffff"] + [COL[c] for c in codes[1:]])
lut = {c: i for i, c in enumerate(codes)}
panels = [("WorldCover 2021 (reference)", wc), (f"e-means + DBSCAN, Mac settings ({n_clusters(MAC)} clusters)", majority(mac["combo_cpu"])),
          (f"e-means + DBSCAN, Xeon settings ({n_clusters(XEON)} clusters, white = noise)", majority(xeon["combo_cpu"])),
          ("Plain e-means, k = 3", majority(k3))]
fig, ax = plt.subplots(2, 2, figsize=(10, 10.6))
for a, (title, arr) in zip(ax.flat, panels):
    idx = np.vectorize(lambda v: lut.get(int(v), 0))(arr)
    a.imshow(idx, cmap=cmap, norm=BoundaryNorm(np.arange(len(codes) + 1) - 0.5, len(codes)), interpolation="nearest")
    a.set_title(title, fontsize=10)
    a.set_xticks([])
    a.set_yticks([])
fig.legend(handles=[Patch(facecolor=COL[c], label=NAMES[c]) for c in codes[1:]], loc="lower center", ncol=8, fontsize=9, frameon=False)
fig.tight_layout(rect=(0, 0.035, 1, 1))
fig.savefig(HERE / "img" / "cairo_map.jpg", dpi=85, pil_kwargs={"quality": 75})
print("saved img/cairo_map.jpg")
