# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pyproj"]
# ///
"""Export the Cairo windows for the web app: nearest-pixel samples on a Web Mercator grid.

Needs data/cairo/ from aef_fetch.py. Writes app/data/meta.json, <window>_emb.bin (int8, h x w x 64, -128 = no data)
and <window>_wc.bin (uint8 WorldCover class, 0 = no data or no embedding). The vectors are copied, never averaged.
"""
import json
import pathlib

import numpy as np
from pyproj import Transformer

APP = pathlib.Path(__file__).parent
SRC = APP.parent / "data" / "cairo"
SIZE, PX, SRC_N, D = 1024, 10.0, 4096, 64
# Same windows as aef_fetch.py (not imported: that needs duckdb and rasterio).
WINDOWS = {"test": (296250.0, 3299830.0), "tune": (296250.0, 3258870.0)}   # UTM 36N lower-left corners
to_wm = Transformer.from_crs(32636, 3857, always_xy=True)
to_utm = Transformer.from_crs(3857, 32636, always_xy=True)

out = APP / "data"
out.mkdir(exist_ok=True)
meta = {"d": D, "size": SIZE, "windows": {}}
for name, (x0, y0) in WINDOWS.items():
    side = SRC_N * PX
    t = np.linspace(0, side, 200)                          # the edges curve a little in Web Mercator
    ex = np.concatenate([x0 + t, x0 + t, np.full(200, x0), np.full(200, x0 + side)])
    ey = np.concatenate([np.full(200, y0), np.full(200, y0 + side), y0 + t, y0 + t])
    mx, my = to_wm.transform(ex, ey)
    xmin, xmax, ymin, ymax = mx.min(), mx.max(), my.min(), my.max()
    gx = xmin + (np.arange(SIZE) + 0.5) * (xmax - xmin) / SIZE
    gy = ymax - (np.arange(SIZE) + 0.5) * (ymax - ymin) / SIZE
    ux, uy = to_utm.transform(*np.meshgrid(gx, gy))
    col = np.floor((ux - x0) / PX).astype(np.int64)
    row = np.floor((y0 + side - uy) / PX).astype(np.int64)  # row 0 is north
    ok = (col >= 0) & (col < SRC_N) & (row >= 0) & (row < SRC_N)
    emb_src = np.load(SRC / f"{name}_emb.npy", mmap_mode="r")
    wc_src = np.load(SRC / f"{name}_wc.npy", mmap_mode="r")
    assert emb_src.shape == (SRC_N, SRC_N, D) and emb_src.dtype == np.int8 and wc_src.shape == (SRC_N, SRC_N)
    emb = np.full((SIZE, SIZE, D), -128, np.int8)
    wc = np.zeros((SIZE, SIZE), np.uint8)
    emb[ok] = emb_src[row[ok], col[ok]]
    wc[ok] = wc_src[row[ok], col[ok]]
    wc[emb[..., 0] == -128] = 0                            # wc > 0 is the only validity rule downstream
    emb.tofile(out / f"{name}_emb.bin")
    wc.tofile(out / f"{name}_wc.bin")
    meta["windows"][name] = {"extent": [xmin, ymin, xmax, ymax]}
    print(name, f"{(wc > 0).sum():,} valid pixels")
(out / "meta.json").write_text(json.dumps(meta, indent=1))
