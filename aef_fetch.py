# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "rasterio", "duckdb"]
# ///
"""Fetch AlphaEarth (AEF) 2021 embeddings and ESA WorldCover 2021 for two windows around Cairo and the Nile.

Writes data/cairo/{name}_emb.npy (H, W, 64) int8 (nodata -128) and data/cairo/{name}_wc.npy (H, W) uint8
on the same 10 m UTM 36N grid. Both sources are public (CC-BY 4.0) and need no account.
"""
import math
import os
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds

HERE = pathlib.Path(__file__).parent
DATA = HERE / "data" / "cairo"
CRS = "EPSG:32636"
PX = 10
SIZE = 4096                                    # 40.96 km
WINDOWS = {                                    # (x0, y0) lower-left, UTM 36N metres
    "test": (296250, 3299830),                 # Cairo, the Nile, the pyramids plateau (30.89-31.31E, 29.81-30.19N)
    "tune": (296250, 3258870),                 # directly south, no overlap (30.90-31.32E, 29.44-29.82N)
}
INDEX = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"
WORLDCOVER = "/vsis3/esa-worldcover/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif"
GDAL = dict(AWS_NO_SIGN_REQUEST="YES", GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
            GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", GDAL_HTTP_MULTIPLEX="YES", VSI_CACHE="TRUE",
            GDAL_HTTP_CONNECTTIMEOUT="20", GDAL_HTTP_LOW_SPEED_TIME="60", GDAL_HTTP_LOW_SPEED_LIMIT="20000",
            GDAL_HTTP_MAX_RETRY="8", GDAL_HTTP_RETRY_DELAY="3")   # a stalled transfer fails and retries instead of hanging


def aef_tiles(x0, y0, x1, y1):
    con = duckdb.connect()
    con.sql("INSTALL httpfs; LOAD httpfs;")
    q = f"""SELECT path FROM read_parquet('{INDEX}')
            WHERE year = 2021 AND crs = '{CRS}' AND utm_west < {x1} AND utm_east > {x0}
              AND utm_south < {y1} AND utm_north > {y0}"""
    return [p.replace("s3://", "/vsis3/") for (p,) in con.sql(q).fetchall()]


BLOCK = 1024                                   # COG block size; reads aligned to it avoid re-reading blocks
_local = threading.local()


def _open(path):
    cache = _local.__dict__.setdefault("ds", {})
    if path not in cache:
        cache[path] = rasterio.open(path)
    return cache[path]


def read_aef(paths, x0, y0, x1, y1, workers=16):
    """Read the window (north-up, H x W x 64 int8) from the bottom-up AEF .tiff files, block by block.

    rasterio.merge splits a 64-band window into 64 MB chunks that do not match the COG blocks, so it
    downloads most blocks several times. Here every block is read once, by a thread pool.
    """
    out = np.full((SIZE, SIZE, 64), -128, np.int8)
    jobs = []
    with rasterio.Env(AWS_REGION="us-west-2", **GDAL):
        for p in paths:
            with rasterio.open(p) as ds:
                T, (H, W) = ds.transform, ds.shape
            assert T.a == PX and T.e == PX, f"expected a bottom-up 10 m grid: {T}"
            c0, c1 = max(0, round((x0 - T.c) / PX)), min(W, round((x1 - T.c) / PX))
            r0, r1 = max(0, round((y0 - T.f) / PX)), min(H, round((y1 - T.f) / PX))
            for br in range(r0 // BLOCK, -(-r1 // BLOCK)):
                for bc in range(c0 // BLOCK, -(-c1 // BLOCK)):
                    a, b = max(r0, br * BLOCK), min(r1, (br + 1) * BLOCK)
                    c, d = max(c0, bc * BLOCK), min(c1, (bc + 1) * BLOCK)
                    if a < b and c < d:
                        jobs.append((p, T, a, b, c, d))

    def run(job):
        p, T, a, b, c, d = job
        for attempt in range(5):
            try:
                with rasterio.Env(AWS_REGION="us-west-2", **GDAL):
                    arr = _open(p).read(window=Window(c, a, d - c, b - a))
                break
            except Exception as err:                      # a failed block is retried, then raised
                _local.__dict__.pop("ds", None)
                if attempt == 4:
                    raise
                print(f"retry {p[-20:]} {err}", flush=True)
        top = round((y1 - T.f) / PX) - b                  # tile rows run south to north; the output runs north to south
        left = round((T.c + c * PX - x0) / PX)
        out[top:top + (b - a), left:left + (d - c)] = arr[:, ::-1].transpose(1, 2, 0)

    with ThreadPoolExecutor(workers) as ex:
        for i, _ in enumerate(ex.map(run, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  block {i}/{len(jobs)}", flush=True)
    return out


def check_against_vrt(paths, out, x0, y0, y1):
    """Compare random 32 x 32 patches with the provided north-up .vrt files."""
    rng = np.random.default_rng(0)
    checked = 0
    with rasterio.Env(AWS_REGION="us-west-2", **GDAL):
        for p in paths:
            with rasterio.open(p.replace(".tiff", ".vrt")) as ds:
                for _ in range(2):
                    X, Y = x0 + PX * int(rng.integers(0, SIZE - 32)), y0 + PX * int(rng.integers(0, SIZE - 32))
                    if not (ds.bounds.left <= X and X + 32 * PX <= ds.bounds.right and ds.bounds.bottom <= Y and Y + 32 * PX <= ds.bounds.top):
                        continue
                    ref = ds.read(window=from_bounds(X, Y, X + 32 * PX, Y + 32 * PX, ds.transform)).transpose(1, 2, 0)
                    top = round((y1 - (Y + 32 * PX)) / PX)
                    left = round((X - x0) / PX)
                    assert np.array_equal(ref, out[top:top + 32, left:left + 32]), f"mismatch with {p}"
                    checked += 1
    assert checked, "no patch was checked against the .vrt files"
    print(f"  checked {checked} patches against the .vrt files", flush=True)


def wc_tiles(x0, y0, x1, y1):
    """Names of the 3x3 degree WorldCover tiles under the window."""
    w, s, e, n = transform_bounds(CRS, "EPSG:4326", x0, y0, x1, y1)
    lat0, lon0 = math.floor(s / 3) * 3, math.floor(w / 3) * 3
    return [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            for la in range(lat0, math.ceil(n / 3) * 3, 3) for lo in range(lon0, math.ceil(e / 3) * 3, 3)]


def save(path, array):
    """Write under a temporary name, so that a stopped run leaves no short file under the final name."""
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, array)
    os.replace(tmp, path)


def fetch(name, x0, y0):
    x1, y1 = x0 + SIZE * PX, y0 + SIZE * PX
    emb_path, wc_path = DATA / f"{name}_emb.npy", DATA / f"{name}_wc.npy"
    t0 = time.perf_counter()
    if emb_path.exists():
        print(f"{name}: AEF cached")
    else:
        paths = aef_tiles(x0, y0, x1, y1)
        print(f"{name}: {len(paths)} AEF tiles")
        out = read_aef(paths, x0, y0, x1, y1)
        check_against_vrt(paths, out, x0, y0, y1)
        save(emb_path, out)
        del out
        print(f"{name}: AEF {time.perf_counter() - t0:.0f}s")

    if wc_path.exists():
        print(f"{name}: WorldCover cached")
        return
    wc = np.zeros((SIZE, SIZE), np.uint8)
    for tile in wc_tiles(x0, y0, x1, y1):
        with rasterio.Env(AWS_REGION="eu-central-1", **GDAL), rasterio.open(WORLDCOVER.format(tile)) as src:
            part = np.zeros_like(wc)
            reproject(rasterio.band(src, 1), part, dst_transform=from_origin(x0, y1, PX, PX), dst_crs=CRS,
                      resampling=Resampling.nearest, dst_nodata=0)
            wc = np.where(wc == 0, part, wc)
    save(wc_path, wc)
    vals, cnt = np.unique(wc, return_counts=True)
    print(f"{name}: WorldCover classes {dict(zip(vals.tolist(), cnt.tolist()))}, total {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    os.environ.update(GDAL)
    DATA.mkdir(parents=True, exist_ok=True)
    for name, (x0, y0) in WINDOWS.items():
        fetch(name, x0, y0)
